# core/proxy_tunnel.py

## 一、这个文件是干什么的

这个文件实现了一个**只用 Python 标准库**的「本地回环 HTTP CONNECT 隧道」工具，专门用来让**没有原生代理支持的 Neo4j Python 驱动**能够穿过本机的正向代理（例如 Clash 监听在 `127.0.0.1:7890`）去访问云端 Bolt 端点（Neo4j Aura）。

它的核心思路是：驱动仍然使用**真实主机名**（这样 SNI 和证书校验都正确），但把 TCP 连接落到本机上为「该主机名单独开的一个监听端口」上；这个本地监听端口把收到的字节用 `CONNECT host:port` 转发给代理，代理再去连真正的服务器。TLS 依然在真实服务器上终结，所以中间没有任何解密行为。

文件里主要包含三块东西：一个模块级判断函数 `_should_proxy_host`（决定哪些主机名需要走隧道）、一个隧道类 `ConnectTunnel`（本地监听 + 双向转发）、一个代理协调类 `ProxyBroker`（按主机名复用/管理隧道，并提供 `resolve` 与 `remap_getaddrinfo` 两种重映射入口），以及两个模块级小工具函数 `_numeric_port` 和 `_pipe`。

它被用到的典型场景是：项目在初始化 Neo4j 驱动前，构造一个 `ProxyBroker`，把它的 `resolve` 或 `remap_getaddrinfo` 交给驱动/连接层，让所有指向 `*.neo4j.io` 的连接自动走代理；不用时调用 `close()` 一次性关掉全部隧道。文件里反复强调的一点是：**必须为每个主机名单独开一条隧道**，因为 Aura 的路由表会下发 `p-mt-….neo4j.io` 这类成员名，如果所有主机名都钉到同一条入口隧道，连接会被送到错误的后端。

## 二、函数与类逐条详解

### `_should_proxy_host(host: str | None) -> bool` （第 28 行）
- **作用**：这是一个纯判断函数，用来回答「这个主机名是否应该被劫持进本地 CONNECT 隧道」。它是整套机制的第一道闸门：只有返回 `True` 的主机名，`ProxyBroker` 才会为它创建隧道并改写解析结果。它同时承担了「排除本机地址」的职责——如果传入的是 `127.0.0.1`、`localhost`、`::1`，直接返回 `False`，否则一旦本地回环地址也被劫持，就会出现「隧道连隧道」的自环，把代理自己（或本机其他服务）也塞进代理里。对 Neo4j 云端的判断使用后缀匹配，覆盖 `.neo4j.io` 与 `.databases.neo4j.io` 两个域名族，从而兼容 Aura 下发的各种成员主机名。判断前统一 `casefold()`，因此大小写混写的主机名也能正确命中。
- **参数**：
  - `host`：`str | None`，待判断的主机名（域名或 IP 字符串）。允许为 `None`，也允许为空字符串，此时视为「无需代理」。没有默认值，是必传参数。
- **返回**：`bool`。`host` 为假值（`None` 或 `""`）时返回 `False`；归一化后属于 `{"127.0.0.1", "localhost", "::1"}` 时返回 `False`；否则当且仅当主机名以 `.neo4j.io` 或 `.databases.neo4j.io` 结尾时返回 `True`，其余情况一律返回 `False`。
- **内部流程**：第一步用 `if not host: return False` 挡掉空值；第二步 `name = host.casefold()` 做大小写归一化；第三步用 `in` 判断是否为本机回环名单；第四步用两个 `endswith` 做或运算，把结果直接返回。全程无循环、无副作用、不涉及任何 I/O。
- **异常/边界**：不主动抛异常。传入 `None`、空串安全返回 `False`。注意它要求 `host` 具备 `casefold` 方法，若调用方传入非字符串对象（例如 int）会抛 `AttributeError`——调用点通过 `str(host) if host else None` 做了保护。后缀匹配是「宽松」的：像 `evil-neo4j.io`（以 `-neo4j.io` 结尾但不以 `.neo4j.io` 结尾）不会命中，但 `anything.neo4j.io` 会命中。
- **同文件关系**：被 `ProxyBroker.resolve`（第 219 行）和 `ProxyBroker.remap_getaddrinfo` 内嵌的 `remapped` 函数（第 227 行）调用。它自身不调用本文件里的任何函数。

### `class ConnectTunnel` （第 37 行）
- **作用**：这是本文件的「单条隧道」实现。一个 `ConnectTunnel` 实例代表「本机上某个随机端口 ↔ 某个固定目标 `host:port`」的一条通路：它在本机 `127.0.0.1` 上随机挑一个端口监听，每当有连接进来，就用 HTTP `CONNECT` 方法请求本地代理帮忙连到目标主机，然后把两侧的字节互相搬运。之所以需要它，是因为 Neo4j 驱动不认代理设置，只能通过「假装目标就在本地」的方式骗过驱动。类文档明确说明：每条被转发的连接由一个线程同时负责两个方向，且所有线程都是 daemon 线程，这样即使调用方忘记 `close()`，解释器退出时也不会被这些线程卡住。
- **参数**：类本身不接收参数，构造参数见下面的 `__init__`。
- **返回**：类是类型本身，实例化后得到 `ConnectTunnel` 对象。
- **内部流程**：作为容器，它维护了几个状态：`proxy_host` / `proxy_port`（代理地址）、`target_host` / `target_port`（目标地址）、`connect_timeout`（连代理的超时）、`_server`（监听 socket，`None` 表示未启动或已关闭）、`_threads`（本对象派生的线程集合）、`_lock`（保护 `_threads` 的互斥锁）、`local_port`（实际监听到的本地端口，启动前为 `None`）。
- **异常/边界**：构造时可能因代理 URL 非法而抛 `ValueError`；其余方法各自处理自己的异常（详见各方法条目）。
- **同文件关系**：被 `ProxyBroker.ensure` 创建并缓存，被 `ProxyBroker.close` 统一关闭；内部调用模块级的 `_pipe` 与自身的 `_read_connect_response`。

### `ConnectTunnel.__init__(self, proxy_url: str, target_host: str, target_port: int, *, connect_timeout: float = 10.0) -> None` （第 44 行）
- **作用**：解析并保存「通过哪个代理」和「要连到哪里」这两组信息，同时初始化运行时状态。它会把代理 URL 做一次宽松解析——如果传进来的字符串里没有 `//`（例如只写 `127.0.0.1:7890`），会自动补上前导 `//` 再交给 `urlsplit`，这样用户可以省略 scheme。解析后还会校验 scheme 只能是空串、`http` 或 `https`，并且必须能取出主机名，否则立刻报错，避免后面在 `_relay` 里才出现难以定位的失败。注意它在构造阶段**不做任何网络操作**，真正的监听在 `start()` 里。
- **参数**：
  - `proxy_url`：`str`，必传。本地正向代理地址，形如 `http://127.0.0.1:7890`、`https://proxy:8443` 或简写的 `127.0.0.1:7890`。scheme 必须是空、`http`、`https` 之一，且必须含主机名。
  - `target_host`：`str`，必传。隧道要连到的真实目标主机名（例如 `xxx.databases.neo4j.io`）。这个值会原样出现在 `CONNECT` 请求行里，因此保留域名而不预解析成 IP，代理的域名规则才能生效。
  - `target_port`：`int`，必传。目标端口，Neo4j Bolt 通常是 `7687`。
  - `connect_timeout`：`float`，关键字参数，默认 `10.0`。连接本地代理时 `socket.create_connection` 使用的超时秒数。
- **返回**：无返回值（`None`）。
- **内部流程**：先用 `"//" in proxy_url` 判断是否需要补 `//`，再 `urlsplit`；然后检查 `parsed.scheme not in ("", "http", "https") or not parsed.hostname`，命中则 `raise ValueError`。校验通过后把 `parsed.hostname` 存入 `self.proxy_host`，端口用 `parsed.port or 7890`（未写端口时回退到 Clash 默认的 7890），接着保存 `target_host`、`target_port`、`connect_timeout`；最后把 `_server` 置 `None`、`_threads` 置空集合、`_lock` 建为 `threading.Lock()`、`local_port` 置 `None`。
- **异常/边界**：`proxy_url` scheme 非法或无法解析出主机名时抛 `ValueError`，异常信息里回显原始 URL。端口缺失时静默回退 `7890`；若 URL 里的端口不是合法数字，`parsed.port` 的取值/异常由 `urllib.parse` 决定（在 Python 中会抛 `ValueError`），本函数不做额外兜底。`target_host` 为空串不会被拦截，会一路带到 `CONNECT` 请求里。
- **同文件关系**：被 `ProxyBroker.ensure` 调用（唯一的实例化点）。自身不调用本文件里的其他函数。

### `ConnectTunnel.start(self) -> ConnectTunnel` （第 65 行）
- **作用**：真正把本地监听器架起来：创建 TCP socket、绑定到 `127.0.0.1` 的随机空闲端口、开始监听，并启动一个后台接受线程 `_accept_loop`。它做成幂等的——如果已经启动过（`_server` 不为 `None`）就直接返回自身，重复调用不会泄漏第二个监听端口。绑定 `127.0.0.1` 而非 `0.0.0.0` 是安全考虑：隧道只对本机可见，不会被外部网络利用来蹭代理。返回 `self` 使得 `ConnectTunnel(...).start()` 可以链式写，也可以配合 `__enter__` 用 `with` 语法。
- **参数**：无（`self` 除外）。
- **返回**：`ConnectTunnel`，返回自身（`self`），无论本次是真的启动了还是因为已启动而直接跳过。
- **内部流程**：①若 `self._server is not None` 直接 `return self`；②`socket.socket(AF_INET, SOCK_STREAM)` 建流式 TCP socket；③`setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)` 允许地址快速复用，避免刚关掉的端口处于 TIME_WAIT 时绑定失败；④`bind(("127.0.0.1", 0))` 由内核分配随机端口；⑤`listen(16)` 设置等待队列长度；⑥`settimeout(0.5)` 给监听 socket 设半秒超时——这是为了让 `_accept_loop` 能周期性醒来检查是否该退出，而不是永久阻塞在 `accept` 上；⑦保存 `_server`，用 `getsockname()[1]` 取回真实端口写入 `local_port`；⑧创建名为 `connect-tunnel-accept` 的 daemon 线程指向 `_accept_loop` 并 `start()`，再把线程登记进 `_threads`。
- **异常/边界**：绑定失败（例如极端情况下无可用端口）会抛 `OSError`，本函数不捕获，由调用方（`ProxyBroker.ensure`）承担；`SO_REUSEADDR` 或 `settimeout` 在正常平台上不会失败。注意 `_threads` 的写入此处**没有加锁**，因为启动阶段还没有并发访问。
- **同文件关系**：调用 `self._accept_loop`（作为线程目标）。被 `ConnectTunnel.__enter__` 和 `ProxyBroker.ensure` 调用。

### `ConnectTunnel.close(self) -> None` （第 80 行）
- **作用**：关闭这条隧道：停止接受新连接、关闭监听 socket，并给所有已派生的线程最多 1 秒的收尾时间。它同样是幂等的——未启动或已经关闭过（`_server` 为 `None`）时直接返回。做法上先把 `_server` 取出来并**立刻置 `None`**，这样 `_accept_loop` 的 `while self._server is not None` 条件会在下一次循环判断时变为假，线程自然退出；同时 `ProxyBroker.ensure` 也靠检查 `_server is None` 来判断某个缓存隧道是否已经失效。关闭时对 `shutdown` 和 `close` 分别吞掉 `OSError`，因为对端可能已经断开、socket 可能已关闭，这些都不是需要上报的错误。
- **参数**：无（`self` 除外）。
- **返回**：无返回值（`None`）。
- **内部流程**：①`_server is None` 则 `return`；②用元组赋值 `server, self._server = self._server, None` 原子地取走引用并清空；③`server.shutdown(SHUT_RDWR)` 尝试立刻打断阻塞中的 `accept`，失败则忽略 `OSError`；④`server.close()` 释放描述符，失败同样忽略；⑤加 `self._lock` 并复制一份 `_threads` 快照（先复制再遍历是为了避免持锁 join 造成死锁）；⑥对快照里每个线程 `join(timeout=1.0)`，超时即放弃等待。注意 `_threads` 集合本身没有被清空，且已结束的线程仍留在集合里。
- **异常/边界**：主动吞掉 `shutdown` / `close` 的 `OSError`，所以不会因为重复关闭或对端已断而报错。若某个 relay 线程在 1 秒内没结束，`join` 会超时返回而不抛异常，线程作为 daemon 会在解释器退出时被回收。因为监听 socket 已关闭，`_accept_loop` 里的 `accept()` 会抛 `OSError` 并 `break` 退出循环。若在从未 `start()` 的对象上调用，直接静默返回。
- **同文件关系**：被 `ConnectTunnel.__exit__`、`ProxyBroker.ensure`（清理失效隧道时）、`ProxyBroker.close` 调用。自身不调用本文件里的其他函数。

### `ConnectTunnel.__enter__(self) -> Self` （第 97 行）
- **作用**：上下文管理器协议的进入方法，让调用方可以写 `with ConnectTunnel(...) as t:`，从而保证离开代码块时隧道一定被关闭。它的实现就是「进入即启动」，把 `start()` 的返回值（也就是 `self`）交出去，这样 `as` 绑定的对象立刻就有可用的 `local_port`。这样设计的好处是把「启动」和「清理」配对交给语言机制，避免忘记 `close()` 导致端口和线程泄漏。
- **参数**：无（`self` 除外）。
- **返回**：`Self`（即 `ConnectTunnel` 实例本身），由 `start()` 返回。
- **内部流程**：单行 `return self.start()`，其余行为全部复用 `start()`。
- **异常/边界**：`start()` 里可能出现的 `OSError`（绑定/监听失败）会原样向外抛出，此时 `with` 语句不会进入代码块，也就不会调用 `__exit__`。
- **同文件关系**：调用 `ConnectTunnel.start`；与 `ConnectTunnel.__exit__` 成对使用。

### `ConnectTunnel.__exit__(self, *exc_info: object) -> None` （第 100 行）
- **作用**：上下文管理器协议的退出方法，无论 `with` 代码块是正常结束还是抛异常，都会被执行，用来确保隧道被关闭。它刻意忽略 `exc_info`（异常类型、异常值、traceback 三元组），也就是**不吞掉异常**——返回 `None` 在 Python 中表示「不抑制异常」，异常会继续向上传播。参数用 `*exc_info` 收集是为了兼容协议传入的 3 个位置参数，同时明确表达「这些信息本方法不关心」。
- **参数**：
  - `*exc_info`：`object` 可变位置参数，Python 会依次传入 `exc_type`、`exc_value`、`traceback`；正常退出时三者均为 `None`。本方法不使用这些值。
- **返回**：无返回值（`None`），因此异常不被抑制。
- **内部流程**：单行 `self.close()`。
- **异常/边界**：`close()` 内部已经吞掉了所有 `OSError`，所以本方法基本不会抛异常；返回 `None` 保证原异常继续传播。
- **同文件关系**：调用 `ConnectTunnel.close`；与 `ConnectTunnel.__enter__` 成对使用。

### `ConnectTunnel._accept_loop(self) -> None` （第 103 行）
- **作用**：这是监听线程的主体，在一个循环里不断 `accept()` 新进来的本地连接，并为每一个连接启动一个 relay 线程去处理。它必须运行在独立线程里，因为 `accept` 是阻塞操作，放在主线程会把整个程序卡住。循环退出条件绑定在 `self._server` 上：只要 `close()` 把 `_server` 置为 `None`，循环就会在下一轮判断时结束。它给每个连接都单开线程（而不是串行处理），因为 Neo4j 驱动会同时保持多条连接（路由连接 + 各成员的数据连接），串行处理会让后续连接一直排队甚至超时。
- **参数**：无（`self` 除外）。
- **返回**：无返回值（`None`）。线程结束时静默退出。
- **内部流程**：①开头 `assert self._server is not None` 做一个内部不变量检查（同时让类型检查器知道后面 `self._server.accept()` 不是 `None`）；②`while self._server is not None:` 循环；③`client, _ = self._server.accept()` 取客户端 socket，忽略对端地址；④若抛 `socket.timeout`（即那 0.5 秒的监听超时到点）则 `continue` 回到循环顶部，重新检查 `_server` 是否已被关闭——这是「半秒轮询退出标志」的实现方式；⑤若抛 `OSError`（例如 socket 已被 `close()` 关闭，`accept` 直接报错）则 `break` 跳出循环；⑥正常拿到 `client` 后创建名为 `connect-tunnel-relay` 的 daemon 线程指向 `self._relay`，参数为 `client`，`start()` 启动；⑦在 `self._lock` 保护下把该线程登记进 `_threads`。
- **异常/边界**：`socket.timeout` 被当作正常节奏处理（`continue`）；`OSError` 被当作「该退出了」处理（`break`）；`assert` 在 `python -O` 优化模式下会被剥离，但那时 `self._server` 为 `None` 也不会进入循环，所以不构成实际风险。若 `threading.Thread.start()` 失败（极少见的资源耗尽），异常会传播出线程函数，仅终止本监听线程。
- **同文件关系**：由 `ConnectTunnel.start` 作为线程目标启动；内部调用 `self._relay`（在新线程中）。

### `ConnectTunnel._relay(self, client: socket.socket) -> None` （第 119 行）
- **作用**：处理**一条**被接受的本地连接：连上本地代理、发 `CONNECT` 请求建立隧道、校验代理的响应码，然后开两个方向线程把字节对拷，最后等两个方向都结束后清理 socket。这是整个转发流程的中枢。之所以要发完整的 `CONNECT host:port HTTP/1.1` 请求（带 `Host` 和 `Proxy-Connection: keep-alive` 头）而不是先解析 IP，是因为要保留域名给代理做规则匹配。响应码校验只接受 `HTTP/1.1 200` 或 `HTTP/1.0 200` 开头，其他一律视为失败并抛 `ConnectionError`。
- **参数**：
  - `client`：`socket.socket`，必传。由 `_accept_loop` 从本地监听端口 `accept` 出来的、连接驱动那一侧的 socket。
- **返回**：无返回值（`None`）。无论成功与否都在 `finally` 里关掉两侧 socket。
- **内部流程**：①`upstream` 初始化为 `None`，用于 `finally` 里安全关闭；②`socket.create_connection((self.proxy_host, self.proxy_port), timeout=self.connect_timeout)` 连本地代理，超时由构造参数决定；③拼出请求字符串：请求行 `CONNECT {target_host}:{target_port} HTTP/1.1`、头 `Host:` 同目标、`Proxy-Connection: keep-alive`，以空行结束（`\r\n\r\n`）；④`upstream.sendall(request.encode("ascii"))` 发送；⑤`response = self._read_connect_response(upstream)` 读取代理响应头；⑥若响应既不 `startswith(b"HTTP/1.1 200")` 也不 `startswith(b"HTTP/1.0 200")`，抛 `ConnectionError`，异常信息里把状态行按 `\r` 切出第一行并截断到 200 字符，便于排查代理返回的 407/403 等原因；⑦创建两个 daemon 线程：`tunnel-a` 执行 `_pipe(client, upstream, client)`（驱动→代理方向），`tunnel-b` 执行 `_pipe(upstream, client, upstream)`（代理→驱动方向）；⑧`start()` 两个线程后 `join()` 等待两者结束（没有超时，靠 `_pipe` 在任一侧读到 EOF 或出错时返回）；⑨`except OSError: pass` 吞掉连接/发送/接收过程中的网络异常；⑩`finally` 中依次对 `upstream` 和 `client` 调 `close()`，每次都用 `try/except OSError` 包裹。
- **异常/边界**：`socket.create_connection` 超时抛 `socket.timeout`（`OSError` 的子类）被吞掉；代理返回非 200 时抛 `ConnectionError`——注意 `ConnectionError` 是 `OSError` 的子类，所以这个「明确失败」也会被同一个 `except OSError` 捕获并静默处理，不会打到调用方；`request.encode("ascii")` 在目标主机名含非 ASCII 字符时会抛 `UnicodeEncodeError`（不是 `OSError`），会穿透 `except` 但被 `finally` 保证清理。`finally` 里对 `None` 做了判断，因此即使在第②步就失败也不会 `AttributeError`。两个 `join()` 没有超时，理论上若某个 `_pipe` 卡在 `recv` 上会长时间占用该 relay 线程（daemon 属性保证不会阻塞退出）。
- **同文件关系**：由 `_accept_loop` 在新线程中调用；调用 `ConnectTunnel._read_connect_response`（静态方法）和模块级 `_pipe`。

### `ConnectTunnel._read_connect_response(sock: socket.socket) -> bytes` （第 156 行）
- **作用**：从代理连接上读取 `CONNECT` 的响应头，一直读到出现 `\r\n\r\n`（头部结束标志）或读满 16384 字节为止。它有一个非常关键的副作用，代码里的中文注释专门解释了原因：**只给 CONNECT 响应阶段设置 10 秒读取超时，读完必须恢复阻塞模式**。因为 `_pipe` 是直接对同一个 socket 做 `recv` 的，如果 socket 一直带着 10 秒超时，空闲超过 10 秒时 `_pipe` 会收到 `socket.timeout`，被它当成断链而关闭，Neo4j 的长期连接就会断，驱动随后报 `defunct connection / No data`。所以这个 `finally: sock.settimeout(None)` 不是可选的清理，而是保证长连接可用的必要步骤。
- **参数**：
  - `sock`：`socket.socket`，必传。已经建立、且已发送过 `CONNECT` 请求的代理侧 socket。
- **返回**：`bytes`。返回累计读到的原始响应数据；正常情况下包含完整的响应头（以 `\r\n\r\n` 结束）；如果对端在读完头部前就关闭连接（`recv` 返回空字节），则返回已经读到的部分（可能为空 `b""`）。
- **内部流程**：①`sock.settimeout(10.0)` 设置读取超时；②`data = b""` 初始化缓冲区；③`while b"\r\n\r\n" not in data and len(data) < 16384:` 循环——两个条件分别是「还没读到头部结束」和「还没超过 16KB 上限」；④循环体 `chunk = sock.recv(_READ_CHUNK)`（每次最多 4096 字节，即模块常量 `_READ_CHUNK`）；⑤若 `chunk` 为空（对端关闭）则 `break`；⑥否则 `data += chunk` 累加；⑦`finally` 中无条件 `sock.settimeout(None)` 恢复阻塞模式，即使循环中抛异常也会执行。
- **异常/边界**：读取超时会抛 `socket.timeout`，被上层 `_relay` 的 `except OSError` 捕获；对端关闭返回空 `b""` 而不抛异常；超过 16KB 仍未见到头部结束符时循环退出并返回已读内容，上层会因不以 `200` 开头而判定失败；`finally` 保证任何情况下都恢复阻塞超时，避免污染后续 `_pipe` 的行为。另外它**不会**把响应头之后可能已读到的正文数据单独留存——超出头部的内容仍在 `data` 里并被丢弃，这对 CONNECT 场景是安全的，因为 200 响应之后代理不会发多余数据。
- **同文件关系**：被 `ConnectTunnel._relay` 调用；使用模块常量 `_READ_CHUNK`。是 `@staticmethod`，因此不依赖实例状态，可独立测试。

### `class ProxyBroker` （第 174 行）
- **作用**：这是对外的「门面」类，负责按 `(host, port)` 维护一张隧道表，并把「要连 `xxx.neo4j.io:7687`」这样的解析请求改写成「要连 `127.0.0.1:<本地隧道端口>`」。它的存在是为了解决一个具体问题：Neo4j 驱动的集群路由会下发多个成员主机名（例如 `p-mt-….neo4j.io`），每个主机名必须对应**自己那条**隧道，否则连接会被送到错误的后端；因此它不能只建一条隧道，而要按主机名逐一创建并按需复用。类文档还特别记录了设计取舍：它**不再**去 patch `socket.getaddrinfo`（早期做法让驱动以为自己连的是 `127.0.0.1`，随后集群路由报 `Unable to retrieve routing information`），而是提供 `resolve` 给驱动层在保留原始主机名的前提下做地址改写。
- **参数**：类本身不接收参数，构造参数见 `__init__`。
- **返回**：实例化得到 `ProxyBroker` 对象。
- **内部流程**：内部只维护三样状态：`proxy_url`（代理地址原样保存）、`connect_timeout`（传给每条隧道的连接超时）、`_tunnels`（`dict[tuple[str, int], ConnectTunnel]` 缓存）、`_lock`（保护缓存的互斥锁）。
- **异常/边界**：构造时不做校验、不建连接，因此 `proxy_url` 非法要到 `ensure` 真正创建 `ConnectTunnel` 时才抛 `ValueError`。
- **同文件关系**：内部创建并管理 `ConnectTunnel`，调用模块级 `_should_proxy_host` 和 `_numeric_port`。

### `ProxyBroker.__init__(self, proxy_url: str, *, connect_timeout: float = 10.0) -> None` （第 181 行）
- **作用**：保存代理地址与连接超时，并初始化隧道缓存和互斥锁。它非常轻量：不解析 URL、不建立任何 socket，因此可以安全地在模块导入期或应用启动早期构造，真正的工作延迟到第一次 `resolve` / `remap_getaddrinfo` 命中需要代理的主机时才发生（惰性建隧道）。`connect_timeout` 用关键字参数限定（`*` 之后的参数只能按名传递），避免调用方把超时和 URL 位置搞混。
- **参数**：
  - `proxy_url`：`str`，必传。本地正向代理地址，格式要求与 `ConnectTunnel.__init__` 相同，会在创建隧道时被逐个传递下去。
  - `connect_timeout`：`float`，关键字参数，默认 `10.0`。每条隧道连接代理时的超时秒数，会原样传给 `ConnectTunnel`。
- **返回**：无返回值（`None`）。
- **内部流程**：把两个参数分别存入 `self.proxy_url` 和 `self.connect_timeout`；`self._tunnels = {}` 建立空字典；`self._lock = threading.Lock()` 建立互斥锁。没有其他逻辑。
- **异常/边界**：无特殊处理——不校验 `proxy_url`、不校验超时正负，这些约束推迟到 `ConnectTunnel` 构造时检查。
- **同文件关系**：被 `ProxyBroker.ensure` 通过 `self.proxy_url` / `self.connect_timeout` 间接使用。自身不调用本文件里的函数。

### `ProxyBroker.ensure(self, host: str, port: int) -> ConnectTunnel` （第 187 行）
- **作用**：获取（或创建）指定 `(host, port)` 对应的隧道，是缓存逻辑的核心。它承担三件事：查缓存命中则直接复用；命中但发现该隧道已经失效（`_server` 被 `close()` 置成 `None`）则先关掉再从缓存里剔除并重建；完全未命中则新建一条隧道并 `start()` 后存入缓存。之所以要检查 `_server`，是因为隧道可能因为外部调用 `close()` 而「死了但还挂在字典里」，如果不检查就会把流量送进一个不再监听的端口，导致连接被拒绝。整个「查—判—建—存」都在 `self._lock` 保护下完成，避免多线程（Neo4j 驱动会在多个线程里发起连接）同时为同一主机名建出两条隧道。
- **参数**：
  - `host`：`str`，必传。目标主机名，作为缓存键的第一部分，同时会作为 `ConnectTunnel` 的 `target_host`（也就是 `CONNECT` 请求里的目标）。
  - `port`：`int`，必传。目标端口，作为缓存键的第二部分，同时作为 `ConnectTunnel` 的 `target_port`。
- **返回**：`ConnectTunnel`。总是返回一个已 `start()`、`local_port` 可用的隧道实例（除非 `start()` 抛异常）。
- **内部流程**：①`key = (host, port)` 组装缓存键；②`with self._lock:` 进入临界区；③`tunnel = self._tunnels.get(key)` 查缓存；④若 `tunnel is not None and getattr(tunnel, "_server", None) is None`，说明隧道已关闭：在 `try/except OSError` 里调用 `tunnel.close()` 做幂等清理，然后把 `tunnel` 置 `None` 以便重建；⑤若 `tunnel is None`，用 `ConnectTunnel(self.proxy_url, host, port, connect_timeout=self.connect_timeout).start()` 创建并启动，再写回 `self._tunnels[key]`；⑥`return tunnel`（在锁内返回）。注意 `getattr(..., "_server", None)` 用反射访问了 `ConnectTunnel` 的私有属性，是为了容错——万一传入的对象没有该属性也不会崩。
- **异常/边界**：`ConnectTunnel(...)` 构造时的 `ValueError`（代理 URL 非法）和 `start()` 时的 `OSError`（绑定/监听失败）都会在持锁状态下直接向外抛出，此时缓存里不会留下半成品条目。若缓存中的隧道处于失效状态，`close()` 的 `OSError` 被吞掉，保证清理失败不会阻止重建。`host` 为空串或 `port` 非法不会被校验，会原样成为缓存键。
- **同文件关系**：被 `ProxyBroker.resolve` 和 `remap_getaddrinfo` 内嵌的 `remapped` 调用；内部构造并启动 `ConnectTunnel`、调用 `ConnectTunnel.close`。

### `ProxyBroker.resolve(self, address)` （第 204 行）
- **作用**：这是给驱动层用的「地址改写」入口：输入一个地址（可以是对象，也可以是元组），如果主机名属于需要代理的 Neo4j 域名，就返回「指向本地隧道的地址列表」；否则原样返回。它的设计目的是**在不 patch DNS 的前提下**完成重映射——文档字符串解释了为什么放弃 patch `socket.getaddrinfo`：那样会让驱动认为自己连到了 `127.0.0.1`，进而导致集群路由失败并报 `Unable to retrieve routing information`。保留原始主机名给驱动的路由与 SNI 使用，只在真正建 TCP 连接时改成本地端口，是更安全的做法。返回值统一包装成**列表**（元素是 `(host, port)` 元组），以贴近 `getaddrinfo` 风格的返回形态，方便调用方按列表处理。
- **参数**：
  - `address`：无类型标注，必传。既支持带 `.host` / `.port` 属性的对象（例如某些驱动内部地址对象），也支持可下标访问的序列（如 `("host", 7687)` 元组）。允许为 `None`（此时 `host` 会退化为空串、`port` 走默认值），但通常不会这样调用。
- **返回**：列表。命中代理时返回 `[("127.0.0.1", tunnel.local_port)]`（单元素列表，端口是内核分配的本地隧道端口）；未命中时返回 `[address]`，即把原始输入原封不动包在一个列表里。
- **内部流程**：①`host = getattr(address, "host", None)` 尝试按对象属性取主机；②`port = getattr(address, "port", None)` 尝试按对象属性取端口；③若 `host is None`，则 `host = address[0] if address else ""`，退化为按下标取第一个元素（同时兼容 `address` 为 `None` 的情况）；④若 `port is None and address is not None and len(address) > 1`，则 `port = address[1]` 按下标取第二个元素——这里用 `len()` 说明它假设 `address` 支持长度查询；⑤`if _should_proxy_host(str(host) if host else None):` 判断是否需要代理，注意这里做了 `str()` 转换和空值保护，避免把非字符串对象传给 `_should_proxy_host` 导致 `AttributeError`；⑥命中则 `tunnel = self.ensure(str(host), _numeric_port(port))` 拿到（或创建）隧道；⑦`assert tunnel.local_port is not None` 断言端口已就绪；⑧返回 `[("127.0.0.1", tunnel.local_port)]`；⑨未命中直接 `return [address]`。
- **异常/边界**：`address` 为 `None` 时 `host` 变成 `""`，`_should_proxy_host` 返回 `False`，最终返回 `[None]`（不抛异常）。若 `address` 是既无 `.host` 属性、又不支持下标且没有 `len()` 的对象，第④步的 `len(address)` 会抛 `TypeError`；这是本函数对输入形态的隐含约束。端口缺失或非法时由 `_numeric_port` 兜底为 `7687`，不会因为端口是 `None` 而崩。`assert` 在 `-O` 模式下被剥离，但 `start()` 成功后 `local_port` 必然有值，所以实际不会出现 `None` 参与连接的情况。
- **同文件关系**：调用 `_should_proxy_host`、`ProxyBroker.ensure`、`_numeric_port`。被项目里的 Neo4j 连接初始化代码调用（本文件之外）。

### `ProxyBroker.remap_getaddrinfo(self, original)` （第 225 行）
- **作用**：这是一个**工厂方法**，接收原始的 `socket.getaddrinfo` 函数，返回一个包装版函数；把包装版装回 `socket.getaddrinfo` 后，所有对 `*.neo4j.io` 的名字解析都会被改写成对本地隧道端口的解析，其他主机名则完全透传。它提供的是与 `resolve` 不同的接入方式：`resolve` 需要驱动层显式调用，而 `remap_getaddrinfo` 是「全局劫持」，适合那些直接走标准 socket 解析、无法注入自定义解析器的调用链。它只在**主机名需要代理**时才改 `host`/`port`，然后把 `family`、`type`、`proto`、`flags` 原样转交给原始函数，因此不改变解析语义。
- **参数**：
  - `original`：无类型标注，必传。原始的解析函数（通常是未打补丁的 `socket.getaddrinfo`）。它必须能被以 `(host, port, family, type, proto, flags)` 六个位置参数调用。
- **返回**：一个函数对象 `remapped`，签名等价于 `(host, port, family=0, type=0, proto=0, flags=0)`，返回值与 `original` 相同（`getaddrinfo` 风格的地址信息列表）。
- **内部流程**：①在 `remap_getaddrinfo` 内部定义闭包 `remapped`，闭包捕获 `original` 与 `self`；②`remapped` 被调用时，先用 `_should_proxy_host(host)` 判断；③命中则 `target_port = _numeric_port(port)` 归一化端口，`tunnel = self.ensure(str(host), target_port)` 获取隧道，`assert tunnel.local_port is not None`，然后本地变量改写为 `host, port = "127.0.0.1", tunnel.local_port`；④未命中则不改动 `host` / `port`；⑤两种情况最后都 `return original(host, port, family, type, proto, flags)`；⑥`remap_getaddrinfo` 本身把 `remapped` 作为返回值交出去。
- **异常/边界**：`remapped` 内部的 `ensure` 可能抛 `ValueError`（代理 URL 非法）或 `OSError`（本地端口绑定失败），会从解析调用处向外传播——这意味着一旦代理配置错误，DNS 解析阶段就会失败，属于「快速失败」。`host` 为 `None` 时 `_should_proxy_host` 返回 `False`，原样透传。`port` 为 `None` 时命中代理分支会用 `_numeric_port` 兜底成 `7687`。注意它是**每次调用 `remap_getaddrinfo` 就产生一个新闭包**，若反复调用并层层包装同一个函数，会造成重复包装（本文件不做去重）。
- **同文件关系**：内部定义并返回闭包 `remapped`；`remapped` 调用 `_should_proxy_host`、`_numeric_port`、`ProxyBroker.ensure`。与 `ProxyBroker.resolve` 是同一目的的两条不同接入路径。

### `ProxyBroker.remapped(host, port, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0)` （第 226 行，`remap_getaddrinfo` 的嵌套函数）
- **作用**：这是真正被当作 `socket.getaddrinfo` 使用的包装函数（注意 `type` 参数名遮蔽了内置的 `type`，这是为了与标准库签名保持一致）。它的职责是：判断本次解析的目标主机是否属于需要走代理的 Neo4j 域名，若是，就把解析目标从真实域名换成 `127.0.0.1` 加上该域名专属隧道的本地端口，然后调用原始解析函数；若不是，就完全按原样转发，保证本机、内网、其他公网域名的解析行为一丝不变。这样 Neo4j 驱动在「解析域名」这一步就已经被导向本地隧道，而它自己完全感知不到代理的存在。
- **参数**：
  - `host`：无类型标注，必传。要解析的主机名，会被 `_should_proxy_host` 判断；命中时被改写为字符串 `"127.0.0.1"`。
  - `port`：无类型标注，必传。要解析的端口，可能是 `int` 或数字字符串；命中时经 `_numeric_port` 归一化，并作为隧道缓存键的一部分。
  - `family`：`int`，默认 `0`。地址族（如 `AF_INET`、`AF_INET6`、`AF_UNSPEC`），原样透传给 `original`。
  - `type`：`int`，默认 `0`。socket 类型（如 `SOCK_STREAM`），原样透传。
  - `proto`：`int`，默认 `0`。协议号，原样透传。
  - `flags`：`int`，默认 `0`。解析标志位（如 `AI_PASSIVE`），原样透传。
- **返回**：`original(...)` 的返回值，即 `getaddrinfo` 风格的地址信息列表；命中代理时列表里的地址是 `127.0.0.1` 与隧道本地端口。
- **内部流程**：①`if _should_proxy_host(host):` 判断；②命中后 `target_port = _numeric_port(port)`；③`tunnel = self.ensure(str(host), target_port)` 取隧道；④`assert tunnel.local_port is not None`；⑤`host, port = "127.0.0.1", tunnel.local_port` 就地改写局部变量；⑥跳出 `if` 后统一 `return original(host, port, family, type, proto, flags)`。
- **异常/边界**：`ensure` 的 `ValueError` / `OSError` 会向外传播（代理配置错误或端口绑定失败时解析直接失败）；`host` 为 `None` 时走透传分支；`port` 为 `None` 或非数字字符串时被 `_numeric_port` 兜底成 `7687`；不对 `family`/`type`/`proto`/`flags` 做任何校验或修正。
- **同文件关系**：由 `ProxyBroker.remap_getaddrinfo` 定义并返回；调用 `_should_proxy_host`、`_numeric_port`、`ProxyBroker.ensure`。它是本文件中唯一的嵌套函数。

### `ProxyBroker.close(self) -> None` （第 236 行）
- **作用**：一次性关闭这个 broker 管理的所有隧道，并清空缓存，用于应用退出或断开 Neo4j 时做整体清理。它的做法是先持锁把 `_tunnels` 的值复制成列表并**立刻 `clear()`**，然后**在锁外**逐个 `tunnel.close()`——这个顺序很关键：既避免在持锁期间执行可能耗时的 `join` 造成其他线程阻塞，也保证即使某个 `tunnel.close()` 抛异常，缓存也已经清空、不会残留失效引用。清空后 broker 仍可继续使用，下一次 `ensure` 会按需重新建隧道。
- **参数**：无（`self` 除外）。
- **返回**：无返回值（`None`）。
- **内部流程**：①`with self._lock:` 进入临界区；②`tunnels = list(self._tunnels.values())` 快照；③`self._tunnels.clear()` 清空字典；④退出锁后 `for tunnel in tunnels: tunnel.close()` 逐个关闭。
- **异常/边界**：`ConnectTunnel.close` 内部已吞掉 `OSError`，所以正常路径不会抛异常；本函数自身没有额外的 `try`，如果某个 `close` 意外抛出非 `OSError` 异常，会中断循环导致后面的隧道没被关闭（但缓存已经清空）。对从未使用过的 broker（空字典）调用是安全的空操作。
- **同文件关系**：调用 `ConnectTunnel.close`。通常由项目在关闭 Neo4j 驱动或应用退出时调用（本文件之外）。

### `_numeric_port(port) -> int` （第 244 行）
- **作用**：把各种形态的端口输入归一化成合法的 `int` 端口号，并在拿不到合法值时回退到 Neo4j Bolt 的默认端口 `7687`。它存在的意义是让上层的 `resolve` 与 `remapped` 不必到处写类型判断：驱动或调用方传进来的端口可能是 `int`、可能是字符串（例如从 URL 或环境变量解析出来的 `"7687"`）、也可能是 `None`。它刻意排除了 `bool`——因为在 Python 里 `True` 是 `int` 的子类且等于 `1`，如果不排除，`True` 会被当成「端口 1」这种明显错误的值接受下来。注意 `"0"` 这种字符串会被 `isdigit()` 判真并转成 `0`，本函数不做 `> 0` 校验。
- **参数**：
  - `port`：无类型标注，必传。候选端口值，可以是 `int`（含 `bool`）、数字字符串，或其他任意类型。
- **返回**：`int`。`port` 是 `int`、不是 `bool`、且 `> 0` 时原样返回；`port` 是全部由数字组成的字符串时返回 `int(port)`；其余一切情况（`None`、`0`、负数、`True`/`False`、非数字字符串、其他对象）返回默认值 `7687`。
- **内部流程**：①`if isinstance(port, int) and not isinstance(port, bool) and port > 0: return port`；②`if isinstance(port, str) and port.isdigit(): return int(port)`；③`return 7687`。三步都是提前返回，没有循环和副作用。
- **异常/边界**：不会抛异常。注意字符串 `"0"` 会返回 `0`（`isdigit()` 为真，且这里没有正数校验），字符串 `"-1"` 因 `isdigit()` 为假而回退 `7687`，带空格的 `" 7687"` 同样回退；`float` 类型的端口（如 `7687.0`）不匹配任何分支，回退 `7687`。`bool` 被显式排除，`True` 会回退到 `7687` 而不是 `1`。
- **同文件关系**：被 `ProxyBroker.resolve`（第 220 行）和嵌套函数 `remapped`（第 228 行）调用。自身不调用本文件里的其他函数。

### `_pipe(source: socket.socket, sink: socket.socket, half: socket.socket) -> None` （第 252 行）
- **作用**：这是隧道的「单向搬运工」：从 `source` 循环读取数据并原样写到 `sink`，直到 `source` 关闭（`recv` 返回空字节）或出现网络错误；结束时对 `half` 做**半关闭**（`SHUT_WR`），告诉对端「我这个方向不再发数据了」，但保留反方向的读取能力。`_relay` 会为每个连接启动两个 `_pipe` 线程（`client→upstream` 和 `upstream→client`），两者共享同一对 socket，因此这里的半关闭而不是全关闭是必须的：如果任一方直接把 socket 关掉，反方向正在传输的数据就会丢失，Neo4j 的长连接会立刻断。文档字符串明确写出了这个契约：「Copy bytes one way; when the source closes, half-close the sink side.」
- **参数**：
  - `source`：`socket.socket`，必传。数据来源端，本函数只对它调用 `recv`。
  - `sink`：`socket.socket`，必传。数据目的端，本函数只对它调用 `sendall`。
  - `half`：`socket.socket`，必传。需要做半关闭的那个 socket。在 `_relay` 的调用里，`client→upstream` 方向传的是 `client`，`upstream→client` 方向传的是 `upstream`——也就是「谁关闭了读取，就对谁关写」，从而把 EOF 正确传递给对端。
- **返回**：无返回值（`None`）。线程结束时静默退出。
- **内部流程**：①`try:` 进入主循环；②`while True:` 无限循环；③`chunk = source.recv(_READ_CHUNK)` 每次最多读 4096 字节（模块常量 `_READ_CHUNK`），这一步在 socket 为阻塞模式时会挂起等待数据；④`if not chunk: break`——收到空字节意味着对端已经正常关闭；⑤`sink.sendall(chunk)` 确保整块数据写完（`sendall` 内部处理了部分写入的情况）；⑥`except OSError: pass` 捕获读取或写入过程中的任何 socket 错误（包括 `socket.timeout`、连接重置 `ECONNRESET`、管道破裂 `EPIPE`）并静默结束循环；⑦`finally:` 中 `half.shutdown(socket.SHUT_WR)` 做半关闭，同样用 `try/except OSError` 吞掉「已经关闭」之类的错误。
- **异常/边界**：所有 `OSError` 都被吞掉，函数本身不会向外抛异常（这一点很重要：它跑在线程里，异常本也无法被主线程感知，静默处理避免打印无意义的堆栈）。`chunk` 为空是正常终止路径而非异常。`half.shutdown` 在 socket 已被关闭时抛 `OSError`，被忽略。若上游 `_read_connect_response` 已正确把 socket 恢复为阻塞模式，本函数的 `recv` 不会因 10 秒空闲超时而误判断链；若阻塞模式没恢复，这里就会周期性收到 `socket.timeout` 并关闭隧道——这正是 `_read_connect_response` 的 `finally` 要防的问题。
- **同文件关系**：被 `ConnectTunnel._relay` 作为两个方向线程的目标调用；使用模块常量 `_READ_CHUNK`。自身不调用本文件里的其他函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_should_proxy_host` | 判断主机名是否属于需要走代理的 Neo4j 域名（并排除本机回环地址）。 |
| `ConnectTunnel` | 单条隧道：本地回环监听 + 用 HTTP CONNECT 经代理转发到某个固定目标 `host:port`。 |
| `ConnectTunnel.__init__` | 解析并校验代理 URL，保存目标地址、超时与运行时状态，不做任何网络操作。 |
| `ConnectTunnel.start` | 在 `127.0.0.1` 随机端口建立监听并启动接受线程，幂等且返回自身。 |
| `ConnectTunnel.close` | 幂等关闭监听 socket 并给所有派生线程最多 1 秒收尾时间。 |
| `ConnectTunnel.__enter__` | 上下文管理器入口，进入即调用 `start()` 并返回实例本身。 |
| `ConnectTunnel.__exit__` | 上下文管理器出口，调用 `close()` 且不抑制异常。 |
| `ConnectTunnel._accept_loop` | 监听线程主体，循环 `accept` 新连接并为每个连接启动一个 relay 线程。 |
| `ConnectTunnel._relay` | 单连接处理：连代理、发 CONNECT、校验 200、开双向线程搬运字节并清理。 |
| `ConnectTunnel._read_connect_response` | 读到 CONNECT 响应头结束为止，并确保读完把 socket 恢复成阻塞模式。 |
| `ProxyBroker` | 按 `(host, port)` 管理多条隧道，并把 Neo4j 域名解析改写到对应的本地隧道端口。 |
| `ProxyBroker.__init__` | 保存代理 URL 与连接超时，初始化隧道缓存和互斥锁，惰性建隧道。 |
| `ProxyBroker.ensure` | 线程安全地获取、复用或重建指定 `(host, port)` 的隧道。 |
| `ProxyBroker.resolve` | 把驱动传入的地址改写为本地隧道地址列表（命中代理时）或原样返回。 |
| `ProxyBroker.remap_getaddrinfo` | 工厂方法，返回一个可替换 `socket.getaddrinfo` 的包装函数。 |
| `ProxyBroker.remapped` | 嵌套的包装函数，命中 Neo4j 域名时把解析目标换成对应隧道的本地端口。 |
| `ProxyBroker.close` | 清空隧道缓存并在锁外逐个关闭所有隧道。 |
| `_numeric_port` | 把 `int`/数字字符串端口归一化，其余情况回退到 Neo4j 默认端口 `7687`。 |
| `_pipe` | 单向搬运字节，源端关闭时对指定 socket 做半关闭（`SHUT_WR`）。 |
