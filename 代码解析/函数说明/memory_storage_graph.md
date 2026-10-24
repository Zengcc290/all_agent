# memory/storage/graph.py

## 一、这个文件是干什么的

这个文件是四层记忆系统中「图记忆层」的存储实现，对外只暴露一个 `Neo4jGraphStore` 类，用 Neo4j 保存实体（`MemoryEntity`）、关系边（`RELATED`）和 n 元事实观测节点（`MemoryObservation`），并在没有驱动时自动退化为一套纯 Python 的内存字典实现，让本地开发不需要真的起一个图数据库。

它同时承担三类职责：一是**写入与幂等合并**（`add_relation` / `add_observation` / `_merge_entity` / `update_entity`），二是**回忆强化**（`_bump_relation` / `_apply_bump`，按权重增长因子累加边权重并封顶），三是**读取与投影**（`entity` / `entity_aliases` / `get_relations` / `path_query` / `graph_snapshot`），其中 `graph_snapshot` 是知识星云 Web 可视化真正消费的图投影，支持用 `at` 参数做时间点切片。

文件顶部还有一段进程级的 `socket.getaddrinfo` 补丁机制：当通过本地代理访问 Neo4j Aura 时，驱动在连接与路由阶段才会解析成员主机名，所以必须在驱动整个生命周期内把 `*.neo4j.io` 重定向到本地隧道；该补丁用「锁 + 引用计数 + 链式帧」实现，保证同进程多个 `Neo4jGraphStore` 实例互相不覆盖、谁安装谁卸载。

文件还包含时间语义相关的静态/类方法（`_temporal_bounds` / `_observations_at` / `_relations_at`），以及内存回退下的 BFS 简单路径搜索（`_neighbours` / `_local_paths`）。整个文件通过 `__all__ = ["Neo4jGraphStore"]` 只导出这一个类。

## 二、函数与类逐条详解

### `_chained_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0)` （第 26 行）
- **作用**：这是替换掉 `socket.getaddrinfo` 的那个全局代理函数。它的存在意义是：Neo4j 驱动在连接与路由阶段会调用 `socket.getaddrinfo` 去解析成员主机名，而驱动本身不支持代理，所以这里在真正的解析动作之前插入一串「帧」，每个帧判断自己是否要接管该主机名（Aura 主机就映射到本地隧道端口），没人接管时才落回原始解析函数。它让「多个 store 实例各带一个代理」这件事能在一个进程里安全共存，而不是互相覆盖全局函数。
- **参数**：
  - `host`：要解析的主机名（或 IP 字符串），由 socket 层传入，通常是 Neo4j 集群成员主机名。
  - `port`：端口，可能是 int，也可能是字符串形式的服务名（如 `"bolt"`），本函数原样透传。
  - `family`：地址族，默认 `0`（不限定），透传。
  - `type`：socket 类型，默认 `0`，透传。
  - `proto`：协议，默认 `0`，透传。
  - `flags`：解析标志位，默认 `0`，透传。
- **返回**：返回 `socket.getaddrinfo` 标准格式的结果，即 `[(family, type, proto, canonname, sockaddr), ...]` 列表。当某个帧返回了非 `None` 的结果时直接返回该帧的结果；所有帧都返回 `None` 时返回最初捕获的原始 `getaddrinfo` 的返回值。
- **内部流程**：第一步读取模块级全局 `_socket_patch_original`，若为 `None` 说明补丁状态被破坏，直接抛 `RuntimeError`。第二步用 `tuple(reversed(_socket_patch_frames))` 取帧的逆序快照（后安装的帧优先尝试），之所以要 `tuple(...)` 包一层，是为了在遍历时不受其他线程并发增删列表影响。第三步逐个调用 `frame(host, port, family, type, proto, flags)`，第一个返回非 `None` 的帧直接 `return`。第四步全部落空时调用 `original(host, port, family, type, proto, flags)`。
- **异常/边界**：`_socket_patch_original` 为 `None` 时抛 `RuntimeError("socket.getaddrinfo patch installed without a captured original")`。帧函数内部自己抛出的异常（例如隧道建立失败）不会在这里被捕获，会向上冒泡到驱动。空帧列表时行为等价于原始 `getaddrinfo`。
- **同文件关系**：由 `_install_socket_patch` 安装进 `socket.getaddrinfo`；调用 `_socket_patch_frames` 里的帧（帧由 `_make_proxy_frame` 生成）；最终回退调用 `_socket_patch_original`。它本身不调用本文件其他函数。

### `_make_proxy_frame(broker) -> Callable[..., Any]` （第 39 行）
- **作用**：为某一个具体的代理 broker 生成一个「链式帧」闭包。之所以不直接在 `_install_socket_patch` 里写死逻辑，是因为每个 `Neo4jGraphStore` 实例可能有自己的 `proxy_url` 和 `ProxyBroker`，帧需要把对应 broker 绑定进闭包，才能做到「这个实例安装的帧只服务这个实例的代理隧道」。该帧的判定规则很窄：只对需要走代理的 Aura 主机做重映射，其他主机一律返回 `None` 让位给后面的帧或原始解析。
- **参数**：
  - `broker`：`core.proxy_tunnel.ProxyBroker` 实例（这里用 `Any` 声明以避免顶层强依赖），负责按 `(host, port)` 惰性建立并复用本地 CONNECT 隧道。
- **返回**：返回一个签名与 `socket.getaddrinfo` 一致的可调用对象 `frame`，该对象要么返回解析结果列表，要么返回 `None` 表示「不接管」。
- **内部流程**：第一步在函数体内**惰性导入** `core.proxy_tunnel` 的 `_numeric_port` 和 `_should_proxy_host`，避免模块导入期的循环依赖与无用开销。第二步读取全局 `_socket_patch_original` 并 `assert original is not None`，因为帧最终必须用原始解析函数去解析 `127.0.0.1`。第三步定义内部函数 `frame`：先调用 `_should_proxy_host(host)`，返回假值就 `return None`；否则调用 `broker.ensure(str(host), _numeric_port(port))` 拿到（或建立）隧道对象，`assert tunnel.local_port is not None`，最后用原始解析函数解析 `"127.0.0.1"` 和 `tunnel.local_port`，并把 `family/type/proto/flags` 原样带上。第四步返回 `frame`。
- **异常/边界**：若补丁未安装（`_socket_patch_original` 为 `None`），`assert` 会抛 `AssertionError`；若 broker 建隧道失败，异常从 `broker.ensure` 冒出；若隧道对象没有 `local_port`，同样由 `assert` 触发 `AssertionError`。`host` 为 `None` 之类非法值时交由 `_should_proxy_host` 判断处理。
- **同文件关系**：调用 `_make_proxy_frame` 的是 `_install_socket_patch`；它生成的 `frame` 被 `_chained_getaddrinfo` 逐个调用；它读取模块级全局 `_socket_patch_original`。

### `frame(host, port, family=0, type=0, proto=0, flags=0)` （第 47 行，嵌套在 `_make_proxy_frame` 内部）
- **作用**：这是真正执行「Aura 主机 → 本地回环隧道」映射的那个闭包。它被注册进 `_socket_patch_frames` 列表后，会在每一次 `socket.getaddrinfo` 调用中被链式调用一次，用极小的判定成本决定本次解析是否由自己接管。之所以要做成闭包而不是普通函数，是因为它必须携带生成它的那个 `broker` 以及安装时捕获的原始解析函数。
- **参数**：
  - `host`：待解析主机名；只有 `_should_proxy_host(host)` 为真时才接管。
  - `port`：端口，先用 `_numeric_port(port)` 归一成数字，再交给 broker 建隧道。
  - `family`、`type`、`proto`、`flags`：默认均为 `0`，原样透传给原始 `getaddrinfo`，保证调用方对地址族/类型的要求不被破坏。
- **返回**：需要代理时返回 `original("127.0.0.1", tunnel.local_port, family, type, proto, flags)` 的解析结果列表；不需要代理时返回 `None`。
- **内部流程**：判断 `_should_proxy_host(host)`；不匹配直接 `return None`；匹配则 `broker.ensure(str(host), _numeric_port(port))` 取隧道（首次调用会真正建立连接并缓存），断言 `tunnel.local_port is not None`；最后用模块级 `original`（即最初的 `socket.getaddrinfo`）解析回环地址与本隧道端口并返回。
- **异常/边界**：`assert tunnel.local_port is not None` 失败抛 `AssertionError`；隧道建立过程中的网络异常（超时、代理不可达）会原样抛出，导致驱动的解析失败。非字符串 `host` 会先被 `str()` 归一，`port` 由 `_numeric_port` 处理非法值。
- **同文件关系**：由 `_make_proxy_frame` 定义并返回；被 `_chained_getaddrinfo` 调用；读取模块级全局 `_socket_patch_original`。

### `_install_socket_patch(broker) -> Callable[..., Any]` （第 59 行）
- **作用**：把一个新帧安装进进程级补丁，并返回这个帧的引用，方便调用方在 `close()` 时精确卸载自己的那一份。它的核心价值是「引用计数 + 只在第一次安装时替换全局函数」：如果同进程已经有别的实例装过补丁，就只追加帧，绝不重复替换 `socket.getaddrinfo`，从而避免出现两份互相打架的全局补丁。
- **参数**：
  - `broker`：本实例的 `ProxyBroker`，会传给 `_make_proxy_frame` 绑定进新帧。
- **返回**：返回新生成的帧可调用对象（`Callable[..., Any]`），调用方需要保存它并交给 `_uninstall_socket_patch`。
- **内部流程**：声明 `global _socket_patch_original`；用 `_socket_patch_lock` 加锁进入临界区；判断 `_socket_patch_frames` 是否为空——为空说明是首个安装者，于是把当前 `socket.getaddrinfo` 捕获进 `_socket_patch_original`，并把 `socket.getaddrinfo` 替换成 `_chained_getaddrinfo`；随后调用 `_make_proxy_frame(broker)` 生成帧，`append` 进帧列表；最后返回该帧。整个「检查 + 替换 + 追加」都在同一把锁内完成，因此是原子的。
- **异常/边界**：`_make_proxy_frame` 内部的 `assert` 失败时会抛 `AssertionError`，此时全局函数已经被替换但帧未入列表，属于极端异常路径（正常情况下 `_socket_patch_original` 刚被赋值，断言必成立）。多次调用不会重复替换全局函数。
- **同文件关系**：调用 `_make_proxy_frame`；操作模块级 `_socket_patch_lock` / `_socket_patch_original` / `_socket_patch_frames`；被 `Neo4jGraphStore._open_driver` 调用；其返回值最终交给 `_uninstall_socket_patch`。

### `_uninstall_socket_patch(frame) -> None` （第 70 行）
- **作用**：按引用精确卸载一个帧，并在「这是最后一个帧」时把 `socket.getaddrinfo` 恢复成最初捕获的原函数、同时把 `_socket_patch_original` 置回 `None`。它保证进程级全局补丁不会泄漏：只要还有别的实例在用，补丁就继续存在；最后一个使用者退出时才彻底撤销。这正是「谁安装谁卸载，互不覆盖」这条设计约束的落地处。
- **参数**：
  - `frame`：要卸载的帧对象，必须是 `_install_socket_patch` 当时返回的那个引用（用身份比较从列表里 `remove`）。
- **返回**：无返回值（`None`）。
- **内部流程**：用 `_socket_patch_lock` 加锁；`try: _socket_patch_frames.remove(frame)`，若抛 `ValueError`（说明这个帧不在列表里，可能已被卸载过）则直接 `return`，做到幂等；若移除成功且列表已空，则把 `socket.getaddrinfo` 恢复为 `_socket_patch_original`，并把 `_socket_patch_original` 置为 `None`。
- **异常/边界**：重复卸载同一个帧不会报错，走 `except ValueError: return`。若帧列表空了但 `_socket_patch_original` 为 `None`（状态不一致），会把 `socket.getaddrinfo` 赋成 `None`，这是理论上的边界情形，正常调用序列不会出现。
- **同文件关系**：被 `Neo4jGraphStore._discard_socket_patch` 和 `Neo4jGraphStore._reopen_driver` 调用；操作与 `_install_socket_patch` 相同的模块级全局。

### `class Neo4jGraphStore` （第 82 行）
- **作用**：这是本文件唯一的公开类，封装「图关系存储」的全部能力：连接管理（惰性建驱动、断线重连、关闭）、写入（关系边与 n 元观测事实的幂等合并）、实体属性维护、回忆强化（权重与召回计数）、查询（单实体属性、批量别名、邻接关系、路径、全图快照）、时间切片过滤、删除与清空。它在有 Neo4j 驱动时走 Cypher，在 `driver is None` 时自动落到四份内存字典上，两种模式对外行为保持一致，因此上层记忆逻辑不需要关心后端到底是哪种。类上还带一个别名属性 `related = get_relations`，方便调用方用更短的名字取邻接关系。
- **参数**：类本身没有构造参数，参数都在 `__init__`。
- **返回**：类，实例化得到存储对象。
- **内部流程**：实例内部维护 `database`（目标数据库名）、`driver`（Bolt 驱动，可为 `None` 表示内存模式）、`proxy_url`、私有连接参数 `_uri/_username/_password`、代理相关 `_broker/_socket_patch_frame`，以及内存回退用的 `_local`（源实体 → 出边列表）、`_reverse`（目标实体 → (源实体, 边) 列表，用于反向查询）、`_entities`（实体属性镜像）、`_observations`（n 元观测事实）。
- **异常/边界**：见各方法。类层面不吞异常，参数校验失败一律抛 `ValueError`。
- **同文件关系**：调用 `_install_socket_patch`、`_uninstall_socket_patch`；内部方法之间互相调用关系在下面逐条说明。

### `__init__(self, uri=None, username=None, password=None, *, driver=None, database=None, proxy_url=None) -> None` （第 85 行）
- **作用**：构造存储对象并决定它进入哪种模式。显式传入 `driver` 时直接用外部驱动（测试或复用连接的场景）；只给 `uri` 时惰性调用 `_open_driver()` 自己建驱动；两者都不给时保持 `driver is None`，全部操作落到内存字典上，这就是「本地开发不需要 Neo4j」的实现方式。它还会初始化内存回退所需的四份索引，保证后续任何写入路径都有容器可用。
- **参数**：
  - `uri`：`str | None`，默认 `None`。Neo4j 连接串（如 `neo4j+s://xxx.databases.neo4j.io`）；为真值且未传 `driver` 时触发建驱动。
  - `username`：`str | None`，默认 `None`。Bolt 认证用户名，建驱动时必填。
  - `password`：`str | None`，默认 `None`。Bolt 认证密码，建驱动时必填（注意判空用的是 `is None`，空字符串也算「已提供」）。
  - `driver`：关键字专用，默认 `None`。外部传入的驱动对象；非 `None` 时完全跳过 `_open_driver`。
  - `database`：关键字专用，默认 `None`。后续所有 `driver.session(database=...)` 的目标库名，`None` 表示用驱动默认库。
  - `proxy_url`：关键字专用，默认 `None`。本地代理地址；非空时建驱动会安装 socket 补丁走 CONNECT 隧道。
- **返回**：无（构造器返回实例）。
- **内部流程**：依次给 `self.database`、`self.driver`、`self.proxy_url`、`self._uri`、`self._username`、`self._password` 赋值；把 `self._broker` 和 `self._socket_patch_frame` 初始化为 `None`；建立 `self._local = {}`、`self._reverse = {}`、`self._entities = {}`、`self._observations = {}` 四个空字典；最后判断 `if self.driver is None and uri:`，成立就调用 `self._open_driver()`。
- **异常/边界**：`uri` 为空串或 `None` 时不会建驱动，静默进入内存模式（不会报错）；`driver` 已传入时即使 `uri` 有值也不会建驱动；真正的连接参数校验与导入错误由 `_open_driver` 抛出。
- **同文件关系**：调用 `_open_driver`；初始化 `_discard_socket_patch`、`_reopen_driver` 等后续方法使用的全部实例属性。

### `_open_driver(self) -> None` （第 113 行）
- **作用**：真正创建 Bolt 驱动。它负责导入 `neo4j`、校验认证参数、组装带超时与连接生命周期的驱动参数，并在配置了 `proxy_url` 时先建 `ProxyBroker`、安装进程级 socket 补丁，再让驱动在「解析即被重映射」的环境下启动。之所以把补丁安装放在 `GraphDatabase.driver(...)` 之前，是因为驱动的构造与后续路由都会解析主机名；之所以失败时要立刻卸载补丁，是为了不把进程级全局补丁泄漏出去。
- **参数**：无显式参数，全部取自实例属性 `_uri`、`_username`、`_password`、`proxy_url`。
- **返回**：无返回值（`None`），副作用是把 `self.driver`（以及可能还有 `self._broker`、`self._socket_patch_frame`）设好。
- **内部流程**：`try: from neo4j import GraphDatabase`，失败则抛 `RuntimeError("Neo4jGraphStore requires neo4j") from exc`；校验 `self._username` 非空且 `self._password is not None`，否则抛 `ValueError`；构造 `kwargs = {"auth": (username, password), "connection_timeout": 15.0, "max_connection_lifetime": 60.0, "liveness_check_timeout": 10.0}`；若 `self.proxy_url` 为真，则惰性导入 `core.proxy_tunnel.ProxyBroker`，创建 `self._broker`，调用 `_install_socket_patch(self._broker)` 并把返回的帧存进 `self._socket_patch_frame`；最后 `try: self.driver = GraphDatabase.driver(self._uri, **kwargs)`，捕获任何 `Exception` 时先调用 `self._discard_socket_patch()` 再 `raise` 原样抛出。
- **异常/边界**：缺少 `neo4j` 包 → `RuntimeError`；用户名缺失或密码为 `None` → `ValueError`；驱动构造失败 → 先卸载补丁与关闭 broker，再把原异常抛出（异常类型不被包装）。
- **同文件关系**：调用 `_install_socket_patch`、`_discard_socket_patch`；被 `__init__`（当 `driver is None` 且 `uri` 为真）和 `_reopen_driver` 调用。

### `_discard_socket_patch(self) -> None` （第 144 行）
- **作用**：把本实例安装过的 socket 补丁与代理 broker 清理干净，是一个幂等的收尾动作。它被用在两条路径上：驱动构造失败的错误路径（避免泄漏全局补丁）和 `close()` 的正常关闭路径。它只清理自己持有的那一个帧和 broker，不会误伤同进程其他实例。
- **参数**：无。
- **返回**：无返回值（`None`）。
- **内部流程**：若 `self._socket_patch_frame` 不为 `None`，调用 `_uninstall_socket_patch(frame)` 并把属性置回 `None`；若 `self._broker` 不为 `None`，在 `try/except Exception: pass` 中调用 `self._broker.close()`（尽力而为，关闭失败不影响流程），随后把 `self._broker` 置为 `None`。
- **异常/边界**：`broker.close()` 的异常被显式吞掉（注释标明 `best effort during error path`）；`_uninstall_socket_patch` 自身对重复卸载是幂等的；两个属性都为 `None` 时本方法什么都不做。
- **同文件关系**：调用 `_uninstall_socket_patch`；被 `_open_driver`（异常路径）、`close` 调用。

### `_reopen_driver(self) -> None` （第 155 行）
- **作用**：把当前驱动整体作废并重新建立一套，用于 Aura 路由抖动导致连接失效后的自动恢复。它比「只调 `close()` 再建」多做了一步：连同 socket 补丁和 broker 一起拆掉重建，确保重连后的驱动解析主机名时仍然走隧道（因为补丁是绑定在具体 broker 上的）。整个过程中所有关闭动作都是尽力而为，绝不让「旧驱动关不掉」阻塞重连。
- **参数**：无。
- **返回**：无返回值（`None`）。
- **内部流程**：若 `self.driver` 不为 `None` 且它的 `close` 属性可调用，则在 `try/except Exception: pass` 中关闭它，然后把 `self.driver = None`；若 `self._socket_patch_frame` 不为 `None`，卸载并置 `None`；若 `self._broker` 不为 `None`，尽力关闭并置 `None`；最后调用 `self._open_driver()` 重建驱动（可能重新安装补丁与 broker）。
- **异常/边界**：旧驱动关闭、broker 关闭的异常均被吞掉；`_open_driver` 的异常（缺包、缺认证、建连失败）会向上抛出，此时 `self.driver` 为 `None`。
- **同文件关系**：调用 `_uninstall_socket_patch`、`_open_driver`；被 `_with_session` 在识别到路由类异常时调用。

### `_with_session(self, runner)` （第 173 行）
- **作用**：所有需要执行 Cypher 的方法共用的一层「会话执行 + 单次重连重试」包装。Neo4j Aura 在集群成员切换时会抛 `ServiceUnavailable`、`SessionExpired` 或带 `routing information` 字样的异常，这些是瞬时的，重建驱动再试一次通常就能成功；而其他异常（语法错误、约束冲突等）必须原样抛出，不能被重试掩盖。它还把「driver 为 `None`」统一映射为返回 `None`，让调用方可以安全地在内存模式下走别的分支。
- **参数**：
  - `runner`：一个可调用对象，签名约定为 `runner(session)`，内部用 `session.run(...)` 执行语句并返回结果；本文件里所有 `_run` 嵌套函数都是这种 runner。
- **返回**：返回 `runner(session)` 的返回值；当 `self.driver is None` 时返回 `None`（调用方通常用 `bool(...)` 包裹，或在内存模式下提前 return 而根本不会走到这里）。
- **内部流程**：先判断 `self.driver is None`，是则直接 `return None`；否则 `try: with self.driver.session(database=self.database) as session: return runner(session)`；捕获 `Exception as exc` 后取 `name = type(exc).__name__` 与 `message = str(exc)`；若 `name` 不在 `{"ServiceUnavailable", "SessionExpired"}` 且 `message` 中不含 `"routing information"`，则 `raise` 原异常；否则调用 `self._reopen_driver()`，再开一个新 session 执行 `runner(session)` 并返回结果（重试只做一次，第二次失败会直接抛出）。
- **异常/边界**：非路由类异常直接抛出，不做任何包装；路由类异常重试一次，重试仍失败则抛出第二次的异常；`driver` 为 `None` 时静默返回 `None`（注意：`session(database=self.database)` 的参数是实例属性，构造后不会被本方法修改）。
- **同文件关系**：调用 `_reopen_driver`；被 `add_relation`、`add_observation`、`update_entity`、`graph_snapshot` 调用（`_bump_relation`、`entity`、`entity_aliases`、`relation_memory_ids`、`get_relations`、`path_query`、`delete_memory_relation`、`clear` 则直接自己开 session，不走这一层）。

### `add_relation(self, source, relation, target, *, properties=None, source_domain="", target_domain="", source_aliases=None, target_aliases=None, source_importance=0.5, target_importance=0.5, bump=False) -> None` （第 190 行）
- **作用**：写入一条「源实体 —关系— 目标实体」的三元边，同时顺带维护两端实体节点。它被抽取管道用来把从文本里抽到的关系落库，也被上层用来补写人工关系。它有两个关键设计：一是**幂等合并**，同一对 `(source, relation, target)` 重复写入只更新属性、不产生重复边；二是**计数与权重不被常规写入覆盖**，`weight`、`recall_count`、`last_accessed_at` 只在创建时给默认值，因为 Cypher 的 `SET r += $properties` 是覆盖语义，若允许调用方传这些字段，幂等重写会把累计的回忆计数清零。另外 `bump=True` 时它不写任何东西，转交给回忆强化逻辑，这是 F1「回忆强化」的入口开关。
- **参数**：
  - `source`：`str`，关系起点实体名，必须是非空（去空白后非空）字符串。
  - `relation`：`str`，关系类型（写入边的 `kind`），必须非空字符串。
  - `target`：`str`，关系终点实体名，必须非空字符串。
  - `properties`：关键字专用，`Mapping[str, Any] | None`，默认 `None`。边的附加属性，会被 `dict()` 复制成 `props`；`None` 视为空字典。
  - `source_domain`：关键字专用，`str`，默认 `""`。源实体领域，仅在实体首次创建（`ON CREATE`）时写入。
  - `target_domain`：关键字专用，`str`，默认 `""`。目标实体领域，同样只在 `ON CREATE` 写入。
  - `source_aliases`：关键字专用，`list[str] | None`，默认 `None`。源实体别名；Neo4j 侧在 `ON MATCH` 时用 `size(...) = 0` 判断——空列表不覆盖已有别名。
  - `target_aliases`：关键字专用，`list[str] | None`，默认 `None`。目标实体别名，语义同上。
  - `source_importance`：关键字专用，`float`，默认 `0.5`。源实体重要度，仅在创建时写入。
  - `target_importance`：关键字专用，`float`，默认 `0.5`。目标实体重要度，仅在创建时写入。
  - `bump`：关键字专用，`bool`，默认 `False`。为 `True` 时只做回忆强化，不写实体、不造边。
- **返回**：无返回值（`None`）。注意它不告诉调用方边是新建还是命中已有。
- **内部流程**：先做参数校验——`all(isinstance(v, str) and v.strip() for v in (source, relation, target))`，不满足抛 `ValueError`。若 `bump` 为真，调用 `self._bump_relation(source, relation, target)` 后直接 `return`。接着 `props = dict(properties or {})`。**内存分支**（`self.driver is None`）：用 `self._local.setdefault(source, [])` 取出边列表，用 `next((...), None)` 找到 `relation` 与 `target` 都相同的已有边；没有则新建字典（含 `source/relation/target/properties`，属性里用 `**props` 打底并补齐 `weight=1.0`、`recall_count=0`、`last_accessed_at=""`），`append` 进列表，并在 `self._reverse[target]` 里追加 `(source, edge)` 建立反向索引；已有则只 `existing["properties"].update(props)`；最后调用 `self._merge_entity` 分别合并源与目标实体（含 domain / aliases / importance）。**Neo4j 分支**：拼一条 Cypher，用两个 `MERGE (a:MemoryEntity {name: $source})` / `MERGE (b:MemoryEntity {name: $target})` 配合 `ON CREATE` 写 domain/aliases/importance、`ON MATCH` 只做「别名为空则保留原值」的条件更新，再 `MERGE (a)-[r:RELATED {kind: $relation}]->(b)`，`ON CREATE` 时 `r += $properties` 并补 `weight/recall_count/last_accessed_at` 默认值，`ON MATCH` 只 `r += $properties`；随后定义嵌套 `_run(session)` 执行 `session.run(query, ...).consume()`；最后 `self._with_session(_run)` 提交。
- **异常/边界**：三个必填字段非字符串或空白 → `ValueError`；`properties` 传 `None` → 视为空字典；`source_aliases` / `target_aliases` 传 `None` → 用 `list(... or [])` 转成 `[]`（在 Neo4j 侧 `size([]) = 0`，因此不会覆盖已有别名）；`bump=True` 且边不存在 → 什么都不发生（由 `_bump_relation` 返回 `False`），不会报错；内存模式下 `properties` 里若自带 `weight` 等字段会被固定默认值覆盖（因为固定键写在 `**props` 之后）。
- **同文件关系**：调用 `_bump_relation`、`_merge_entity`、`_with_session`；被 `add_relation` 内部定义的 `_run` 调用；本身被上层记忆写入逻辑调用（文件内没有其他函数调用它）。

### `_run(session)` （第 261 行，嵌套在 `add_relation` 内部）
- **作用**：`add_relation` 交给 `_with_session` 执行的那个闭包，唯一职责是把已经拼好的 Cypher 与参数送进当前会话并等它执行完。它之所以存在，是因为 `_with_session` 需要「可重复执行」的单元才能在路由抖动后重放一次；把它写成闭包就能自然捕获 `query`、`props` 以及各个参数。
- **参数**：
  - `session`：由 `_with_session` 打开的 Neo4j 会话对象，提供 `run(...)` 方法。
- **返回**：无返回值（隐式 `None`）；`_with_session` 会把它作为执行结果返回，调用方忽略。
- **内部流程**：调用 `session.run(query, source=..., target=..., relation=..., properties=props, source_domain=..., target_domain=..., source_aliases=list(source_aliases or []), target_aliases=list(target_aliases or []), source_importance=..., target_importance=...)`，紧接 `.consume()` 把结果游标消费掉（丢弃返回记录，只确保语句真正执行完）。
- **异常/边界**：不做任何异常处理，Cypher 报错或连接异常直接冒泡给 `_with_session`，由后者决定是否重连重试。
- **同文件关系**：由 `add_relation` 定义并传给 `_with_session`；调用方是 `_with_session`；捕获 `add_relation` 的局部变量。

### `add_observation(self, observation_id, predicate, participants, *, properties=None) -> None` （第 278 行）
- **作用**：把一条 **n 元事实**作为 `MemoryObservation` 节点持久化，参与实体通过 `HAS_PARTICIPANT` 边挂在观测节点上。它解决的是三元组表达力不足的问题：同一条 `(主语, 谓词, 宾语)` 在不同时间发生多次时，如果只用边就会被折叠成一条，而独立的观测标识可以保留每次事件；同时一条事实可以携带任意多个带角色的参与者（subject / object / 时间 / 地点等）。写入是幂等的：以 `observation_id` 为键 `MERGE`，并先把该观测的旧参与者边全部删掉再重建，因此重复写入同 id 会得到「替换」语义而不是累加。
- **参数**：
  - `observation_id`：`str`，观测事实的唯一标识，必须是非空（去空白后非空）字符串；同时也是幂等合并的键。
  - `predicate`：`str`，谓词（事实类型），必须是非空字符串。
  - `participants`：`list[Mapping[str, Any]]`，参与者列表。每个元素期望包含 `name`、`role`（这两项必填），可选 `ordinal`、`domain`、`aliases`、`importance`、`entity_type`。
  - `properties`：关键字专用，`Mapping[str, Any] | None`，默认 `None`。观测节点的附加属性，会被 `dict()` 复制，随后强制写入 `predicate` 与 `observation_id`。
- **返回**：无返回值（`None`）。
- **内部流程**：先校验 `observation_id` 与 `predicate` 均为非空字符串，否则 `ValueError`；然后 `enumerate(participants)` 逐个归一化：`name` 与 `role` 取 `str(... or "").strip()`，任一为空就抛 `ValueError("observation participants require name and role")`，否则往 `normalized` 追加字典，其中 `ordinal` 取 `int(participant.get("ordinal", ordinal))`（缺省用枚举下标）、`domain` 取 `str(... or "")`、`aliases` 把每个元素 `str()` 化、`importance` 用 `float(participant.get("importance", 0.5))`、`entity_type` 缺省为 `"概念"`。接着校验 `len(normalized) >= 2` 且至少有一个 `role == "subject"`，否则抛 `ValueError("an observation requires a subject and at least one other participant")`。然后 `props = dict(properties or {})` 并 `props.update({"predicate": predicate, "observation_id": observation_id})`。**内存分支**：把 `{"id", "predicate", "properties", "participants"}` 存进 `self._observations[observation_id]`（同 id 直接覆盖），再对每个参与者调用 `self._merge_entity` 合并实体，并把 `self._entities[name]["entity_type"]` 更新为该参与者的 `entity_type`。**Neo4j 分支**：拼一条 Cypher，`MERGE (o:MemoryObservation {id: $observation_id})` 并 `SET o += $properties, o.predicate = ..., o.name = ...`；`WITH o OPTIONAL MATCH (o)-[old:HAS_PARTICIPANT]->() DELETE old` 清掉旧参与者边；`WITH DISTINCT o UNWIND $participants AS participant` 展开新参与者，`MERGE (e:MemoryEntity {name: participant.name})`，`ON CREATE` 写 domain/aliases/importance/entity_type，`ON MATCH` 用 `size(...) = 0` 保护已有别名并用 `coalesce(e.entity_type, participant.entity_type)` 保留已有类型，再 `MERGE (o)-[r:HAS_PARTICIPANT {role, ordinal}]->(e)` 并 `SET r.entity_type / r.kind = participant.role / r.weight = coalesce(r.weight, 1.0)`；最后定义嵌套 `_run` 执行并 `self._with_session(_run)`。
- **异常/边界**：`observation_id` 非字符串或空白 → `ValueError`；`predicate` 同上；参与者缺 `name` 或 `role` → `ValueError`；参与者少于 2 个或没有 `subject` 角色 → `ValueError`；`participants` 传入空列表时必然触发上面那条「至少一个其他参与者」的校验；`ordinal` 或 `importance` 传了无法转换的值时 `int()` / `float()` 会抛 `ValueError`/`TypeError`，本函数不额外包裹；`properties` 里的 `predicate` / `observation_id` 会被函数强制覆盖。
- **同文件关系**：调用 `_merge_entity`、`_with_session`；`self._entities` 在本方法里被直接写 `entity_type`；被内部嵌套的 `_run` 间接依赖；由上层记忆写入逻辑调用。

### `_run(session)` （第 351 行，嵌套在 `add_observation` 内部）
- **作用**：`add_observation` 的会话执行闭包，把观测写入的 Cypher 与 `observation_id`、`predicate`、`props`、归一化后的 `participants` 一起送进会话执行。它同样是「可重放单元」，让 `_with_session` 在路由类异常后能整体重试一次。
- **参数**：
  - `session`：`_with_session` 提供的 Neo4j 会话。
- **返回**：无返回值（隐式 `None`）。
- **内部流程**：调用 `session.run(query, observation_id=observation_id, predicate=predicate, properties=props, participants=normalized)` 并立即 `.consume()`。
- **异常/边界**：无特殊处理，异常交由 `_with_session` 判定是否重连重试。
- **同文件关系**：由 `add_observation` 定义并传给 `_with_session`；调用方是 `_with_session`。

### `_merge_entity(self, name, domain, aliases, importance) -> None` （第 362 行）
- **作用**：内存回退模式下「实体节点 upsert」的实现，是 Cypher 里 `ON CREATE SET ... ON MATCH SET ...` 那几段子句的 Python 孪生版本。它保证内存模式与 Neo4j 模式语义一致：实体不存在就按给定属性创建；已存在则**只在传入别名非空时**覆盖别名，`domain` 和 `importance` 保持首次创建时的值不变——这与 Cypher 侧「后续边写入不该把 domain/importance 覆盖成空值」的注释意图完全对应。
- **参数**：
  - `name`：`str`，实体名，作为 `self._entities` 的键。本函数不校验其非空性（调用方已校验）。
  - `domain`：`str`，实体领域，仅在首次创建时写入。
  - `aliases`：`list[str] | None`，别名列表；`None` 与空列表等价，都不会触发覆盖。
  - `importance`：`float`，实体重要度，仅在首次创建时写入。
- **返回**：无返回值（`None`），直接修改 `self._entities`。
- **内部流程**：`known = list(aliases or [])` 先做一次浅拷贝，避免外部列表后续被修改而串改内部状态；`entity = self._entities.get(name)`；若 `entity is None`，写入 `{"domain": domain, "aliases": known, "importance": importance}`；否则（实体已存在）判断 `elif known:`，为真时执行 `entity["aliases"] = known`，为假时什么都不做。
- **异常/边界**：不做类型校验；`name` 为 `None` 时会被当成合法字典键存进去（调用方 `add_relation` / `add_observation` 已保证非空）；`aliases` 传入非列表的可迭代对象时 `list()` 会尝试转换，失败则抛异常。
- **同文件关系**：被 `add_relation`（内存分支，源与目标各一次）和 `add_observation`（内存分支，每个参与者一次）调用；它操作 `self._entities`，与 `entity`、`entity_aliases`、`update_entity`、`graph_snapshot` 读的是同一份数据。

### `update_entity(self, name, *, domain=None, aliases=None, importance=None) -> bool` （第 378 行）
- **作用**：显式「编辑一个已存在的图实体节点」的入口。它与 `add_relation` / `add_observation` 的关键区别是只用 `MATCH`、绝不用 `MERGE`，因此**不会因为一次编辑而顺手创建新节点**——节点创建权始终留在写入关系的路径上。它用 `None` 表达「这个字段不动」，并且对空别名列表也做忽略处理，防止调用方漏传参数时把别名清空。
- **参数**：
  - `name`：`str`，要更新的实体名，必须非空（去空白后非空），否则抛 `ValueError`。
  - `domain`：关键字专用，`str | None`，默认 `None`。`None` 或空串都表示保留原值（内存分支用 `if domain:` 判断，Neo4j 分支用 `$domain = ''` 判断）。
  - `aliases`：关键字专用，`list[str] | None`，默认 `None`。`None` 或空列表都表示保留原值。
  - `importance`：关键字专用，`float | None`，默认 `None`。`None` 表示保留原值；传 `bool` 或非数字类型会被拒绝。
- **返回**：返回 `bool`。内存模式下找到实体并更新返回 `True`，实体不存在返回 `False`；Neo4j 模式下由 `RETURN 1 AS updated` 是否有记录决定（`record is not None`），最终用 `bool(self._with_session(_run))` 返回，`False` 表示 `MATCH` 没命中任何节点。
- **内部流程**：校验 `name` 是非空字符串，否则 `ValueError`；校验 `importance` 若不为 `None` 则必须是 `int`/`float` 且**不能是 `bool`**，否则 `ValueError("importance must be a number or None")`。**内存分支**：取 `self._entities.get(name)`，为 `None` 直接 `return False`；否则 `if domain:` 写 `entity["domain"]`，`if aliases:` 写 `entity["aliases"] = list(aliases)`，`if importance is not None:` 写 `entity["importance"] = float(importance)`，最后 `return True`。**Neo4j 分支**：拼 `MATCH (e:MemoryEntity {name: $name}) SET e.domain = CASE WHEN $domain = '' THEN e.domain ELSE $domain END, e.aliases = CASE WHEN size($aliases) = 0 THEN e.aliases ELSE $aliases END, e.importance = CASE WHEN $importance IS NULL THEN e.importance ELSE $importance END RETURN 1 AS updated`；定义嵌套 `_run` 执行 `session.run(...).single()`，把 `record is not None` 作为结果返回；最后 `return bool(self._with_session(_run))`。
- **异常/边界**：`name` 非法 → `ValueError`；`importance` 传 `True`/`False`（bool 是 int 子类，被显式排除）或字符串 → `ValueError`；`domain=""`、`aliases=[]`、`importance=None` 都会被当作「不修改」；实体不存在返回 `False` 而不是抛异常；内存模式下即使三个字段都没传，只要实体存在也返回 `True`（表示「找到了」）。
- **同文件关系**：Neo4j 分支调用 `_with_session` 与其内部嵌套的 `_run`；与 `_merge_entity` 操作同一份 `self._entities`；被上层实体编辑逻辑调用。

### `_run(session)` （第 421 行，嵌套在 `update_entity` 内部）
- **作用**：`update_entity` 的会话执行闭包，负责把实体更新语句执行掉并把「是否命中」这一事实带回给调用方。它是本文件里少数需要返回值的 runner 之一（其他多数只 `consume()` 丢弃结果）。
- **参数**：
  - `session`：`_with_session` 提供的 Neo4j 会话。
- **返回**：返回 `bool`——`session.run(query, name=name, domain=str(domain or ""), aliases=list(aliases or []), importance=importance).single()` 得到的记录不为 `None` 时为 `True`，否则 `False`。
- **内部流程**：先调用 `session.run(...)`，参数把 `domain` 归一为字符串（`None` → `""`）、`aliases` 归一为列表（`None` → `[]`）、`importance` 原样传（可能为 `None`，由 Cypher 的 `IS NULL` 分支处理）；然后 `.single()` 取第一条记录；最后返回 `record is not None`。
- **异常/边界**：无特殊处理；Cypher 异常或连接异常冒泡给 `_with_session`。
- **同文件关系**：由 `update_entity` 定义并传给 `_with_session`；调用方是 `_with_session`。

### `_bump_relation(self, source, relation, target) -> bool` （第 433 行）
- **作用**：F1「回忆强化」的落地点：给一条**已存在**的边把 `recall_count` 加 1、把 `last_accessed_at` 刷成当前时间、把 `weight` 乘上增长因子并封顶到最大值。它被 `add_relation(bump=True)` 以及上层「再次回忆起某个事实」的路径调用。之所以坚持「边不存在就什么都不做」，是因为强化语义上只应奖励已经建立的记忆，凭空造边会污染图结构。
- **参数**：
  - `source`：`str`，边的起点实体名。
  - `relation`：`str`，边的关系类型（匹配 `r.kind`）。
  - `target`：`str`，边的终点实体名。
- **返回**：返回 `bool`。内存模式下遍历到匹配边并调用 `_apply_bump` 后返回 `True`，遍历完没有匹配返回 `False`；Neo4j 模式下 `RETURN 1 AS bumped` 有记录则 `True`，否则 `False`。
- **内部流程**：惰性导入 `from ..base import utc_now`，取 `accessed_at = utc_now().isoformat()` 作为统一时间戳（内存与 Neo4j 两条路径共用同一个值）。**内存分支**：遍历 `self._local.get(source, [])`，命中 `relation` 与 `target` 都相同的边时调用 `self._apply_bump(edge["properties"], accessed_at)` 并 `return True`，循环结束返回 `False`。**Neo4j 分支**：拼 `MATCH (a:MemoryEntity {name: $source})-[r:RELATED {kind: $relation}]->(b:MemoryEntity {name: $target}) SET r.recall_count = coalesce(r.recall_count, 0) + 1, r.last_accessed_at = $accessed_at, r.weight = CASE WHEN coalesce(r.weight, 1.0) * $grow_factor > $weight_max THEN $weight_max ELSE coalesce(r.weight, 1.0) * $grow_factor END RETURN 1 AS bumped`，用 `with self.driver.session(database=self.database) as session:` 直接开会话执行（**不走 `_with_session`**），把 `grow_factor=MEMORY_EDGE_WEIGHT_GROWTH`、`weight_max=MEMORY_EDGE_WEIGHT_MAX` 作为参数传入，`.single()` 取记录，最后 `return record is not None`。
- **异常/边界**：边不存在时返回 `False`，不抛异常；`recall_count`/`weight` 缺失时用 `coalesce(..., 0)` / `coalesce(..., 1.0)` 兜底；权重超上限时被夹到 `MEMORY_EDGE_WEIGHT_MAX`；本方法不做参数非空校验（由 `add_relation` 在调用前完成）；Neo4j 分支不复用 `_with_session`，因此这里遇到 Aura 路由抖动不会被自动重连重试，异常直接抛出。
- **同文件关系**：调用 `_apply_bump`；被 `add_relation`（`bump=True` 时）调用；依赖常量 `MEMORY_EDGE_WEIGHT_GROWTH`、`MEMORY_EDGE_WEIGHT_MAX` 与 `..base.utc_now`。

### `_apply_bump(properties, accessed_at) -> None` （第 469 行，`@staticmethod`）
- **作用**：内存回退模式下回忆强化的纯计算部分，是 Cypher 里那三段 `SET` 的 Python 孪生版本。它被抽成静态方法，是因为它不需要访问实例状态，只需要一个属性字典和已算好的时间戳，这样 `_bump_relation` 的内存分支可以直接把它当工具函数用，也让「内存与 Neo4j 行为一致」这件事更容易对照检查。
- **参数**：
  - `properties`：`dict[str, Any]`，边对象的属性字典，会被**原地修改**。
  - `accessed_at`：`str`，ISO 格式的访问时间字符串，由 `_bump_relation` 用 `utc_now().isoformat()` 统一生成。
- **返回**：无返回值（`None`），效果体现在 `properties` 的原地变更上。
- **内部流程**：`properties["recall_count"] = int(properties.get("recall_count", 0) or 0) + 1`，先取旧值、用 `or 0` 把 `None`/`0`/空串统一成 `0` 再加一；`properties["last_accessed_at"] = accessed_at`；`grown = float(properties.get("weight", 1.0) or 1.0) * MEMORY_EDGE_WEIGHT_GROWTH`；最后 `properties["weight"] = min(grown, MEMORY_EDGE_WEIGHT_MAX)` 完成封顶。
- **异常/边界**：`recall_count` 若是无法 `int()` 的字符串会抛 `ValueError`；`weight` 无法 `float()` 时抛 `ValueError`/`TypeError`；`recall_count` 为 `None` 或 `0` 时被 `or 0` 归零；`weight` 为 `None` 或 `0` 时被 `or 1.0` 归一到 `1.0`；本方法不做任何 try/except。
- **同文件关系**：被 `_bump_relation` 的内存分支调用；依赖模块级常量 `MEMORY_EDGE_WEIGHT_GROWTH` 与 `MEMORY_EDGE_WEIGHT_MAX`。

### `entity(self, name) -> dict[str, Any]` （第 478 行）
- **作用**：读取单个实体的属性（别名 / 领域 / 重要度）。它有一个重要的实现约定：**启用 Neo4j 时以图库为准**，因为 `self._entities` 只是本进程写入过的内存镜像，新起的进程里它是空的，直接读它会静默返回「没有别名」这种错误答案。因此本方法在有驱动时一定走 Cypher。
- **参数**：
  - `name`：`str`，实体名，作为查询条件 `MemoryEntity {name: $name}`；本方法不校验非空。
- **返回**：返回 `dict[str, Any]`。内存模式下返回 `dict(self._entities.get(name, {}))`——实体不存在时是空字典，存在时是含 `domain`/`aliases`/`importance` 的浅拷贝。Neo4j 模式下 `record` 为 `None` 时返回 `{}`；否则返回 `{"domain": record["domain"] or "", "aliases": list(record["aliases"] or []), "importance": float(importance) if importance is not None else 0.5}`。
- **内部流程**：判断 `self.driver is None`，是则直接返回内存镜像的浅拷贝（`dict(...)` 保证调用方改动不会污染内部状态，但 `aliases` 列表本身是共享引用）。否则拼 `MATCH (e:MemoryEntity {name: $name}) RETURN e.domain AS domain, e.aliases AS aliases, e.importance AS importance`，用 `with self.driver.session(database=self.database) as session:` 执行 `session.run(query, name=name).single()`；`record is None` 返回 `{}`；否则取 `importance = record["importance"]` 并按上述规则归一后返回。
- **异常/边界**：实体不存在返回空字典（不抛异常）；`domain` 为 `None` 归一为空串，`aliases` 为 `None` 归一为空列表，`importance` 为 `None` 归一为默认 `0.5`；不走 `_with_session`，因此路由类异常不会被自动重连重试。
- **同文件关系**：读取 `self._entities`（由 `_merge_entity` / `add_observation` / `update_entity` 写入）；与 `entity_aliases` 是同类查询（单个 vs 批量）；被上层实体查询逻辑调用。

### `entity_aliases(self) -> dict[str, list[str]]` （第 502 行）
- **作用**：一次性把所有实体的别名表取回来，返回 `{实体名: 别名列表}`。它存在的直接原因是知识星云图要给每个实体节点带上别名，如果逐个实体调用 `entity()` 会产生 N+1 次查询；这里在 Neo4j 模式下用一条 Cypher 把全部实体查完，内存模式下直接遍历镜像字典，保证可视化渲染时的查询次数是常数级。
- **参数**：无。
- **返回**：返回 `dict[str, list[str]]`。内存模式下遍历 `self._entities.items()`，对每项取 `list(attributes.get("aliases") or [])`；Neo4j 模式下返回 `{str(record["name"]): [str(alias) for alias in (record["aliases"] or [])] for record in session.run(query)}`。实体没有任何别名时对应值是空列表。
- **内部流程**：判断 `self.driver is None`，是则用字典推导式从 `self._entities` 构造结果；否则拼 `MATCH (e:MemoryEntity) RETURN e.name AS name, e.aliases AS aliases`，在 `with self.driver.session(database=self.database) as session:` 中直接迭代 `session.run(query)` 的游标并用字典推导式构造返回值。
- **异常/边界**：图库为空时返回空字典；`aliases` 为 `None` 时用 `or []` 兜底；`name` 与每个 `alias` 都强制 `str()`；本方法不校验参数（没有参数），不走 `_with_session`，连接异常直接抛出。
- **同文件关系**：读取 `self._entities`（与 `_merge_entity`、`update_entity`、`entity` 同一份数据）；被上层（星云图数据组装）调用。

### `graph_snapshot(self, *, at=None) -> dict[str, Any]` （第 520 行）
- **作用**：返回 Web 可视化真正消费的那份「图投影」，一次性给出实体、观测事实和关系三类数据。它是整个存储类里唯一的全量读取入口：内存模式直接遍历四份字典拼装，Neo4j 模式用三条 Cypher 分别取实体、观测（连同参与者聚合）与关系。`at` 参数让它可以做时间点切片——只保留该时刻「最新」的时序观测，从而支持「回看某一天的知识状态」；不传 `at` 时返回全部数据，让前端自己渲染完整时间线。
- **参数**：
  - `at`：关键字专用，`str | None`，默认 `None`。ISO-8601 时间字符串；传了非 `None` 值时先用 `ensure_datetime` 校验，非法值抛 `ValueError`。
- **返回**：返回 `dict[str, Any]`，固定包含四个键：`mode`（`"neo4j"` 或 `"inmemory"`，由 `self.driver is not None` 决定）、`entities`（`[{"name", "properties"}, ...]`）、`observations`（经 `_observations_at` 过滤后的观测列表）、`relations`（经 `_relations_at` 过滤后的关系列表）。
- **内部流程**：若 `at` 为真值，惰性导入 `ensure_datetime` 并校验，`None` 则抛 `ValueError("at must be an ISO-8601 datetime")`。**内存分支**：用列表推导式分别构造 `entities`（`self._entities` 的 `name` + `properties` 浅拷贝）、`observations`（`self._observations` 的 `id`/`predicate`/`properties`/`participants`，参与者逐项 `dict()` 拷贝）、`relations`（双层推导展开 `self._local` 的每条边为 `source`/`relation`/`target`/`properties`）。**Neo4j 分支**：定义三条查询——`entity_query` 取 `e.name` 与 `properties(e)`；`observation_query` 用 `OPTIONAL MATCH (o)-[r:HAS_PARTICIPANT]->(e:MemoryEntity)` 并 `collect({name, role, ordinal, entity_type, domain, aliases, importance})` 聚合参与者；`relation_query` 取 `a.name`/`r.kind`/`b.name`/`properties(r)`；随后定义嵌套 `_run(session)`，在里面分别执行三条语句，把实体归一为 `{"name": str(...), "properties": dict(... or {})}`、观测归一为 `{"id": str(...), "predicate": str(... or "关联"), "properties": dict(...), "participants": [dict(item) for item in (...) if item and item.get("name")]}`（过滤掉参与者为空或没有 `name` 的脏数据）、关系用 `[dict(record) for record in session.run(relation_query)]`；最后 `entities, observations, relations = self._with_session(_run)` 解包三元组。收尾统一返回上述四键字典，其中 `observations` 与 `relations` 分别交给 `_observations_at`、`_relations_at` 做时间过滤。
- **异常/边界**：`at` 不是合法 ISO 时间 → `ValueError`；`at` 为 `None` 或空串时不做过滤，返回全量；Neo4j 模式下观测的 `predicate` 缺失时归一为 `"关联"`，参与者的 `name` 为空会被丢弃；内存模式下所有返回的字典都是浅拷贝，调用方修改不会串改内部状态（但嵌套的 `properties` 里若有可变对象仍是共享引用）；`_with_session` 在驱动为 `None` 时返回 `None`，但本方法的内存分支已提前返回，因此不会解包 `None`。
- **同文件关系**：调用 `_observations_at`、`_relations_at`、`_with_session` 及其内部嵌套的 `_run`；间接调用 `_temporal_bounds`（由两个时间过滤方法调用）；读取 `self._entities`、`self._observations`、`self._local`；被 Web 可视化层调用。

### `_run(session)` （第 575 行，嵌套在 `graph_snapshot` 内部）
- **作用**：`graph_snapshot` 的会话执行闭包，负责在一个会话里依次跑完实体、观测、关系三条查询，并把三份结果打包成元组交回去。把它做成单个 runner 的好处是：三条查询共享同一个会话，路由抖动时 `_with_session` 重连后整套查询会被整体重放，不会出现「实体查到了、关系没查到」的半截状态。
- **参数**：
  - `session`：`_with_session` 提供的 Neo4j 会话。
- **返回**：返回三元组 `(fetched_entities, fetched_observations, fetched_relations)`，分别是实体字典列表、观测字典列表（含参与者列表）、关系字典列表。
- **内部流程**：用列表推导式迭代 `session.run(entity_query)` 构造 `fetched_entities`（`name` 强制 `str()`，`properties` 用 `dict(record["properties"] or {})`）；迭代 `session.run(observation_query)` 构造 `fetched_observations`，其中 `predicate` 用 `str(record["predicate"] or "关联")`，参与者列表用 `[dict(item) for item in (record["participants"] or []) if item and item.get("name")]` 过滤并拷贝；用 `[dict(record) for record in session.run(relation_query)]` 构造 `fetched_relations`；最后返回三元组。
- **异常/边界**：`properties`、`participants`、`predicate` 为 `None` 时分别用 `or {}`、`or []`、`or "关联"` 兜底；参与者中 `name` 为空（或整个条目为空）的会被过滤掉；异常不在此处理，交由 `_with_session` 判断是否重连重试。
- **同文件关系**：由 `graph_snapshot` 定义并传给 `_with_session`；捕获 `graph_snapshot` 里的 `entity_query`、`observation_query`、`relation_query` 三个局部变量。

### `_temporal_bounds(properties, at) -> tuple[bool, Any, Any]` （第 607 行，`@staticmethod`）
- **作用**：时间切片过滤的公共判定函数。它根据一条边或一个观测节点上的 `valid_from` / `valid_to` / `event_at` 三个属性，判断这条数据在给定时刻 `at` 是否「有效可见」，并把解析好的 `event_at` 与 `moment` 一并返回，供上层做「同一主语+谓词只保留最新一条」的挑选。把它抽成静态方法是为了让观测过滤（`_observations_at`）和关系过滤（`_relations_at`）共用同一套时间语义，避免两处判断逻辑漂移。
- **参数**：
  - `properties`：`Mapping[str, Any]`，边或观测节点的属性字典，期望包含可选的 `valid_from`、`valid_to`、`event_at` 字符串。
  - `at`：`str`，查询时刻的 ISO-8601 字符串，必须能被 `ensure_datetime` 解析。
- **返回**：返回三元组 `(visible, event_at, moment)`：
  - `visible`（`bool`）：是否落在有效窗口内且事件时间不晚于查询时刻。判定式为 `in_window and (event_at is None or event_at <= moment)`，其中 `in_window = (valid_from is None or valid_from <= moment) and (valid_to is None or moment <= valid_to)`。
  - `event_at`：解析后的事件时间（`datetime` 或 `None`，无法解析或缺失时为 `None`）。
  - `moment`：查询时刻的 `datetime` 对象。
- **内部流程**：惰性导入 `ensure_datetime`；调用 `ensure_datetime(at)` 得到 `moment`，若为 `None` 抛 `ValueError("at must be an ISO-8601 datetime")`；在 `try/except ValueError` 中把 `str(properties.get("valid_from") or "")` 解析成 `valid_from`，失败置 `None`；同样方式解析 `valid_to`；计算 `in_window`；同样方式解析 `event_at`（失败置 `None`）；返回组合结果。
- **异常/边界**：`at` 非法 → `ValueError`；三个时间属性缺失、为空串或格式非法时都被归为 `None`，其中 `valid_from`/`valid_to` 为 `None` 表示该侧不设限（窗口开放），`event_at` 为 `None` 表示该数据不参与「取最新」的排序（调用方会把它当作非时序数据处理）；注意 `properties.get(...)` 取到的值若本身不是字符串，会先被 `str()` 转换。
- **同文件关系**：被 `_observations_at` 与 `_relations_at` 调用；依赖 `..base.ensure_datetime`；本身不调用本文件其他函数。

### `_observations_at(observations, at) -> list[dict[str, Any]]` （第 633 行，`@classmethod`）
- **作用**：对观测事实列表做时间切片。规则有两条：一是先按有效窗口与 `status` 过滤掉不可见或已过期（`status == "expired"`）的观测；二是对标记为 `cardinality == "temporal"` 且带 `event_at` 的观测，按「主语 + 谓词」分组，每组只保留事件时间最新的一条——这正是「某个属性在某时刻的取值」这种时序事实该有的表现。不满足时序条件（没有 `cardinality` 标记或没有 `event_at`）的观测被视为「无时间性」，一律保留。
- **参数**：
  - `observations`：`list[dict[str, Any]]`，观测列表，每项期望含 `properties` 和 `participants`（参与者里 `role == "subject"` 的那条用来做分组键）。
  - `at`：`str | None`，查询时刻；为假值（`None` 或空串）时**直接原样返回**整个列表，不做任何过滤。
- **返回**：返回 `list[dict[str, Any]]`。`at` 为空时是输入列表本身；否则是 `[*timeless, *(value[1] for value in latest.values())]`，即「所有无时间性观测」加上「每个 (主语, 谓词) 键下最新的那条时序观测」，顺序上无时间性的在前、时序精选的在后。
- **内部流程**：`if not at: return observations`；初始化 `timeless` 列表与 `latest` 字典（键为 `(subject, predicate)`，值为 `(event_at, observation)` 元组）；遍历每条观测，取 `properties = observation.get("properties") or {}`；调用 `cls._temporal_bounds(properties, at)` 拿到 `visible`、`event_at`；若 `not visible` 或 `str(properties.get("status") or "fact") == "expired"` 则 `continue` 跳过；若 `properties.get("cardinality") != "temporal"` 或 `event_at is None`，把整条观测加入 `timeless` 并 `continue`；否则用 `next((...), "")` 从参与者里找第一个 `role == "subject"` 的 `name` 作为 `subject`，键为 `(subject, str(observation.get("predicate") or ""))`，用 `current is None or event_at > current[0]` 判断是否更新 `latest[key]`；最后拼装返回。
- **异常/边界**：`at` 为空时不触发任何解析，也不会调用 `_temporal_bounds`；`properties` 为 `None` 时用 `or {}` 兜底；找不到 subject 参与者时 `subject` 为空串，这类观测会共享 `("", predicate)` 这个键并互相覆盖，属于调用方数据不完整的边界情形；`_temporal_bounds` 可能抛出的 `ValueError` 不会被这里捕获（`at` 已在 `graph_snapshot` 里预先校验过）。
- **同文件关系**：调用 `_temporal_bounds`；被 `graph_snapshot` 调用（对 `observations` 做过滤）。

### `_relations_at(relations, at) -> list[dict[str, Any]]` （第 663 行，`@classmethod`）
- **作用**：关系边的版本时间切片，语义与 `_observations_at` 平行：先过滤不可见与 `status == "expired"` 的边；对 `cardinality == "temporal"` 且带 `event_at` 的边，按「源实体 + 关系类型」分组只留最新一条；其余边一律保留。它让「A 的职位是 X（2023 年）」这类带时间戳的边在某个时间点查询时只显示当时生效的那条。
- **参数**：
  - `relations`：`list[dict[str, Any]]`，关系列表，每项期望含 `source`、`relation`、`target`、`properties`。
  - `at`：`str | None`，查询时刻；为假值时原样返回输入列表。
- **返回**：返回 `list[dict[str, Any]]`。`at` 为空时是输入列表本身；否则是 `[*visible, *(value[1] for value in latest.values())]`，即「所有非时序边」加上「每个 (source, relation) 键下最新的时序边」。
- **内部流程**：`if not at: return relations`；初始化 `visible` 列表与 `latest` 字典（键为 `(source, relation)`）；遍历每条关系，取 `properties = relation.get("properties") or {}`；调用 `cls._temporal_bounds(properties, at)` 得到 `matches` 与 `event_at`；若 `not matches` 或 `status` 为 `"expired"` 则跳过；若 `cardinality == "temporal"` 且 `event_at is not None`，用 `(str(relation.get("source") or ""), str(relation.get("relation") or ""))` 作为键，在 `current is None or event_at > current[0]` 时更新；否则把关系加入 `visible`；最后拼装返回。
- **异常/边界**：`at` 为空时完全不做解析；`properties` 为 `None` 时兜底为空字典；`source` 或 `relation` 缺失时键里对应位置是空串；`_temporal_bounds` 的 `ValueError` 不由本方法捕获。
- **同文件关系**：调用 `_temporal_bounds`；被 `graph_snapshot`、`get_relations` 调用（后者在返回前对结果做时间过滤）。

### `relation_memory_ids(self) -> list[str]` （第 685 行）
- **作用**：把所有关系边上携带的 `memory_id` 收集成一个列表。它服务于「对账（reconcile）」场景：上层拿这份清单跟语义记忆里现存的事实做比对，从而发现哪些图上的边已经没有对应的记忆条目（该删）、哪些记忆条目还没有边（该补）。之所以做成一次性全量返回而不是逐条查询，是为了避免 N+1 查询。
- **参数**：无。
- **返回**：返回 `list[str]`。内存模式下是双层推导式展开 `self._local` 得到的所有边属性里的 `memory_id`（用 `str(edge["properties"].get("memory_id") or "")` 归一，没有该属性的边会产生空字符串元素）；Neo4j 模式下是 `[str(record["memory_id"]) for record in session.run(query)]`，注意这里没有 `or ""` 兜底，`memory_id` 为 `None` 时会变成字符串 `"None"`。
- **内部流程**：判断 `self.driver is None`，是则用 `[str(edge["properties"].get("memory_id") or "") for edges in self._local.values() for edge in edges]` 返回；否则拼 `MATCH ()-[r:RELATED]->() RETURN r.memory_id AS memory_id`，在 `with self.driver.session(database=self.database) as session:` 中迭代游标并逐个 `str()` 后返回列表。
- **异常/边界**：内存模式下没有 `memory_id` 的边返回空串元素（不会跳过）；Neo4j 模式下 `memory_id` 为 `None` 的边会返回字符串 `"None"`（与内存模式行为不一致，是一个需要注意的差异）；图上没有任何关系时返回空列表；不走 `_with_session`，连接异常直接抛出。
- **同文件关系**：读取 `self._local`（由 `add_relation` 内存分支写入）；与 `delete_memory_relation` 配合构成「按 memory_id 对账与清理」的一组能力。

### `get_relations(self, entity, *, relation=None, direction="both", at=None) -> list[dict[str, Any]]` （第 698 行）
- **作用**：查询某个实体的邻接边，支持按关系类型过滤、按方向（出边/入边/双向）过滤，以及按时间点过滤。它是图检索最常用的读取入口，类末尾还给它挂了别名 `related`，让调用方可以用更短的名字调用。内存模式下它把 `_local`（出边）与 `_reverse`（入边）两边的索引拼起来，Neo4j 模式下则按方向动态拼 `WHERE` 条件，两条路径最终都返回同样形状的字典列表。
- **参数**：
  - `entity`：`str`，中心实体名。
  - `relation`：关键字专用，`str | None`，默认 `None`。为 `None` 时不过滤关系类型；否则只保留 `relation` 字段等于该值的边（Neo4j 侧对应 `r.kind = $relation`）。
  - `direction`：关键字专用，`str`，默认 `"both"`。取值必须是 `"in"`、`"out"`、`"both"` 之一，否则抛 `ValueError`。
  - `at`：关键字专用，`str | None`，默认 `None`。时间点过滤参数，交给 `_relations_at` 处理；`None` 表示返回全部。
- **返回**：返回 `list[dict[str, Any]]`，每项形如 `{"source": ..., "relation": ..., "target": ..., "properties": {...}}`（Neo4j 分支的字典由 `dict(record)` 直接从 `source`/`relation`/`target`/`properties` 四个别名构造，形状一致）。结果会先经过 `_relations_at` 做时间切片。
- **内部流程**：先校验 `direction` 合法性。**内存分支**：若 `direction` 是 `"out"` 或 `"both"`，用 `list(self._local.get(entity, []))` 取出出边（注意这里对列表做了拷贝）；若 `direction` 是 `"in"` 或 `"both"`，再把 `self._reverse.get(entity, [])` 里的 `(source, edge)` 重组成以 `entity` 为 `target` 的边字典追加进去；若 `relation is not None`，用列表推导式按 `edge["relation"] == relation` 过滤；最后 `return self._relations_at(values, at)`。**Neo4j 分支**：根据方向设置 `match` 与 `condition`——`"out"` 用 `"a.name = $entity"`，`"in"` 用 `"b.name = $entity"`，`"both"` 用 `"a.name = $entity OR b.name = $entity"`，匹配模式统一是 `(a)-[r:RELATED]->(b)`；把这些条件收进 `clauses` 列表，若 `relation is not None` 再追加 `"r.kind = $relation"`；用 f-string 拼出 `MATCH ... WHERE ... RETURN a.name AS source, r.kind AS relation, b.name AS target, properties(r) AS properties`（`clauses` 用 `" AND ".join` 连接，条件是代码里写死的常量，参数仍走 `$entity` / `$relation`）；在 `with self.driver.session(database=self.database) as session:` 中执行并 `dict(record)` 化；最后同样交给 `self._relations_at(values, at)` 返回。
- **异常/边界**：`direction` 不在三个合法值内 → `ValueError`；实体不存在时内存模式返回空列表，Neo4j 模式也是空列表；`relation` 为 `None` 不做类型过滤；`at` 非法时由 `_relations_at` → `_temporal_bounds` 抛 `ValueError`；不走 `_with_session`，连接异常直接抛出。
- **同文件关系**：调用 `_relations_at`（进而调用 `_temporal_bounds`）；内存分支读取 `self._local` 与 `self._reverse`（均由 `add_relation` 写入）；类属性 `related = get_relations`（第 747 行）是它的别名；被上层图检索逻辑调用。

### `related = get_relations` （第 747 行，类属性别名）
- **作用**：这不是一个独立函数，而是把 `get_relations` 这个函数对象直接绑定到类属性 `related` 上，等价于给同一个方法起了一个更短的名字。存在的意义是调用方（例如上层记忆检索代码）可以用 `store.related(entity)` 这种更简洁的写法获取邻接关系，同时不需要维护两份实现，两者行为、签名、异常完全一致。因为它只是别名，所以没有独立的参数、返回与流程，全部沿用 `get_relations` 的定义。
- **参数**：与 `get_relations` 完全相同（`entity`，以及关键字参数 `relation`、`direction`、`at`）。
- **返回**：与 `get_relations` 完全相同。
- **内部流程**：无独立实现，属性查找直接命中 `get_relations` 的函数对象。
- **异常/边界**：与 `get_relations` 完全相同。
- **同文件关系**：别名指向 `get_relations`；在类命名空间内定义，属于 `Neo4jGraphStore` 的一部分。

### `path_query(self, start, target, *, max_depth=3, limit=50) -> list[dict[str, Any]]` （第 749 行）
- **作用**：查询两个实体之间的**简单路径**（不重复经过同一实体），最多 `max_depth` 跳。它用于「这两个概念是怎么联系起来的」这类解释性检索。结果按路径总权重降序、再按跳数升序排列，因此被反复回忆强化过的路径会排在前面（F1）；内存模式用 BFS 复刻同样的语义。返回结果里包含路径上的实体名序列与每一条边（含 `weight`），方便上层直接展示或再打分。
- **参数**：
  - `start`：`str`，起点实体名，必须非空（去空白后非空），否则抛 `ValueError`。
  - `target`：`str`，终点实体名，同样必须非空，否则抛 `ValueError`。
  - `max_depth`：关键字专用，`int`，默认 `3`。必须是正整数，且**不能是 `bool`**（`True`/`False` 会被显式拒绝），否则抛 `ValueError`。它同时被拼进 Cypher 的变长区间 `[*1..{int(max_depth)}]`。
  - `limit`：关键字专用，`int`，默认 `50`。必须是正整数且不能是 `bool`，否则抛 `ValueError`；用于限制返回的路径条数。
- **返回**：返回 `list[dict[str, Any]]`，每项是 `{"entities": [实体名...], "relations": [{"source", "relation", "target", "weight"}, ...]}`。内存模式下由 `_local_paths` 产生，Neo4j 模式下从记录里取 `entities` 与 `relations` 两个字段（查询里额外算出的 `path_weight` **不会**出现在返回结果中，它只用于排序）。没有路径时返回空列表。
- **内部流程**：先做三组校验（`start`/`target` 非空字符串、`max_depth` 正整数且非 bool、`limit` 正整数且非 bool）；`self.driver is None` 时直接 `return self._local_paths(start, target, max_depth=max_depth, limit=limit)`。否则拼 Cypher：`MATCH p=(a:MemoryEntity {name: $start})-[*1..{int(max_depth)}]-(b:MemoryEntity {name: $target})`（无方向，任意跳数区间；注释说明变长区间上界不能参数化，只能拼进语句，已用 `int()` 收敛为整数以防注入），`RETURN` 里用 `[n IN nodes(p) | n.name] AS entities` 取实体名列表、用 `[r IN relationships(p) | {source: startNode(r).name, relation: r.kind, target: endNode(r).name, weight: coalesce(r.weight, 1.0)}] AS relations` 取边列表、用 `reduce(w = 0.0, r IN relationships(p) | w + coalesce(r.weight, 1.0)) AS path_weight` 累加权重；`ORDER BY path_weight DESC, length(p) ASC LIMIT $limit`；在 `with self.driver.session(database=self.database) as session:` 中执行 `session.run(query, start=start, target=target, limit=limit)`，把每条记录转成 `{"entities": list(record["entities"]), "relations": [dict(relation) for relation in record["relations"]]}` 后返回。
- **异常/边界**：`start`/`target` 非字符串或空白 → `ValueError`；`max_depth`/`limit` 为 `bool`、非整数或小于 1 → `ValueError`；起点或终点不存在、或两跳以内无连通路径 → 返回空列表；`max_depth` 为 1 时只找直接相邻；`weight` 缺失时用 `coalesce(..., 1.0)` 兜底；本方法不走 `_with_session`，连接异常直接抛出。
- **同文件关系**：内存分支调用 `_local_paths`（后者又调用 `_neighbours`）；Neo4j 分支自行开会话，不经过 `_with_session`；被上层路径检索逻辑调用。

### `_neighbours(self, entity) -> list[tuple[str, str, bool, float]]` （第 788 行）
- **作用**：内存回退模式下，把一个实体的所有相邻边整理成 BFS 需要的统一格式。它同时取出边（`_local`）与入边（`_reverse`），并把「出/入」方向、关系类型和权重一并带出，让 `_local_paths` 不需要再关心底层是哪个索引。列表按权重降序排序，是为了让强边先入队，这样在 `limit` 截断时留下来的路径更可能是「更被强化过」的路径（F1）。
- **参数**：
  - `entity`：`str`，中心实体名。
- **返回**：返回 `list[tuple[str, str, bool, float]]`，每个元组是 `(other_end, relation, is_outgoing, weight)`：`other_end` 是邻居实体名，`relation` 是关系类型，`is_outgoing` 为 `True` 表示该边从 `entity` 出发、`False` 表示指向 `entity`，`weight` 是从边属性里取出的浮点权重（缺失或为 `0` 时用 `1.0`）。
- **内部流程**：先用列表推导式从 `self._local.get(entity, [])` 构造出边元组（`edge["target"]`、`edge["relation"]`、`True`、`float(edge["properties"].get("weight", 1.0) or 1.0)`）；再用 `+=` 追加来自 `self._reverse.get(entity, [])` 的入边元组（`source`、`edge["relation"]`、`False`、同样的权重换算）；最后 `values.sort(key=itemgetter(3), reverse=True)` 按元组第 4 位（权重）降序排序；返回 `values`。`itemgetter` 是文件顶部从 `operator` 导入的。
- **异常/边界**：实体没有边时返回空列表；`weight` 为 `None`、`0` 或缺失时统一归一为 `1.0`；`weight` 无法转成浮点时 `float()` 抛异常；本方法只在内存模式下被调用，不做参数校验。
- **同文件关系**：被 `_local_paths` 调用；读取 `self._local` 与 `self._reverse`（由 `add_relation` 内存分支写入）；依赖 `operator.itemgetter`。

### `_local_paths(self, start, target, *, max_depth, limit) -> list[dict[str, Any]]` （第 803 行）
- **作用**：内存回退模式下的路径搜索实现，用队列做广度优先遍历（BFS）枚举简单路径。因为用的是 FIFO 队列且按跳数逐层扩展，跳数少的路径会先被找到；又因为 `_neighbours` 已按权重降序返回邻居，同层内强边优先入队，所以结果整体上呈现「短路径优先、同长度内强边优先」的次序，与 Cypher 侧 `ORDER BY path_weight DESC, length(p) ASC` 的意图一致。它只负责枚举，不负责校验参数（校验在 `path_query` 里完成）。
- **参数**：
  - `start`：`str`，起点实体名（调用方已保证非空）。
  - `target`：`str`，终点实体名（调用方已保证非空）。
  - `max_depth`：关键字专用，`int`，最大跳数（调用方已保证是正整数）。
  - `limit`：关键字专用，`int`，最多返回的路径条数（调用方已保证是正整数）。
- **返回**：返回 `list[dict[str, Any]]`，每项形如 `{"entities": [起点, ..., 终点], "relations": [{source, relation, target, weight}, ...]}`；`relations` 的长度等于 `entities` 长度减一。找不到路径时返回空列表；提前凑满 `limit` 条时也立即停止搜索并返回已找到的部分。
- **内部流程**：初始化 `paths = []` 与队列 `queue = [(start, [start], [])]`（元素是「当前节点、已走过的实体序列、已走过的边序列」三元组）。`while queue and len(paths) < limit:` 循环：`queue.pop(0)` 取出队首三元组；若 `len(relations) >= max_depth` 则 `continue`（达到深度上限不再扩展）；否则遍历 `self._neighbours(node)` 的每个 `(other, relation, outgoing, weight)`：若 `other in entities` 则 `continue`（简单路径约束：不重复经过同一实体，因此天然无环）；构造 `step` 字典，其中 `source` 在出边时是 `node`、入边时是 `other`，`target` 相应互换，并带上 `relation` 与 `weight`；若 `other == target`，把 `{"entities": [*entities, other], "relations": [*relations, step]}` 追加进 `paths`，若已达 `limit` 则 `break` 跳出邻居循环，否则 `continue` 继续试下一个邻居；若不是终点，则把 `(other, [*entities, other], [*relations, step])` 追加进队列。循环结束后返回 `paths`。
- **异常/边界**：`start == target` 时初始队列的 `entities` 已含起点，邻居循环里 `other in entities` 与 `other == target` 的判定不会把起点自身算作路径，因此返回空列表（不会返回零长度路径）；`max_depth` 为 1 时只检查直接邻居；图中有环时由 `other in entities` 保证不会死循环；`limit` 很小（如 1）时找到第一条即停；不处理权重为负等异常数据（权重仅用于排序与输出）。
- **同文件关系**：被 `path_query` 的内存分支调用；调用 `_neighbours`（后者读取 `self._local` 与 `self._reverse`）。

### `delete_memory_relation(self, memory_id) -> bool` （第 833 行）
- **作用**：按 `memory_id` 删除某一条语义记忆在图上留下的痕迹——既删带该 `memory_id` 的 `RELATED` 边，也删 id 相同的 `MemoryObservation` 观测节点（含其全部参与者边）。它被上层在「语义记忆条目被删除/合并」时调用，用来让图记忆与语义记忆保持一致，避免出现「记忆已经没了、图上还挂着边」的悬空引用。返回值告诉调用方到底有没有删到东西。
- **参数**：
  - `memory_id`：`str`，语义记忆条目的 id，必须是非空字符串（注意校验条件是 `not isinstance(memory_id, str) or not memory_id`，因此空白字符串 `" "` 会被当成合法值），否则抛 `ValueError`。
- **返回**：返回 `bool`。内存模式下返回 `removed`，只要观测节点被弹出、或任意一条边被过滤掉，就为 `True`。Neo4j 模式下返回 `bool(record and record["removed"]) or bool(getattr(summary.counters, "nodes_deleted", 0))`，即「删掉的边数非零」或「删掉的节点数非零」任一成立就为 `True`。
- **内部流程**：先校验 `memory_id`。**内存分支**：`removed = self._observations.pop(memory_id, None) is not None` 先尝试弹出观测；然后遍历 `list(self._local.items())` 的快照（`list()` 是为了在循环中安全地删除键），对每个源实体的边列表用列表推导式构造 `kept`（保留 `edge.get("properties", {}).get("memory_id") != memory_id` 的边）；再遍历原 `edges`，对被剔除的边（`if edge in kept: continue`）从 `self._reverse[edge["target"]]` 里用 `indexed_edge is not edge` 做**身份比较**过滤掉对应反向索引项；用 `removed = removed or len(kept) != len(edges)` 累计是否删过边；`kept` 非空则写回 `self._local[source] = kept`，为空则 `self._local.pop(source, None)` 把该键整个删掉；循环结束 `return removed`。**Neo4j 分支**：拼 `MATCH ()-[r:RELATED {memory_id: $memory_id}]->() DELETE r WITH count(r) AS removed OPTIONAL MATCH (o:MemoryObservation {id: $memory_id}) DETACH DELETE o RETURN removed`；在 `with self.driver.session(database=self.database) as session:` 中执行，先 `record = result.single()`、再 `summary = result.consume()`（必须先取记录再消费游标），最后按上述布尔表达式返回。
- **异常/边界**：`memory_id` 非字符串或空字符串 → `ValueError`（但 `" "` 这种纯空白不会被拒绝）；内存模式下没有 `memory_id` 属性的边（`get` 返回 `None`）不会被误删（`None != memory_id`）；`_reverse` 里找不到对应目标键时 `.get(..., [])` 兜底为空列表；Neo4j 模式下 `record` 为 `None` 时用 `getattr(summary.counters, "nodes_deleted", 0)` 兜底（不同驱动版本计数器属性可能缺失）；什么都删不到时返回 `False`；不走 `_with_session`，连接异常直接抛出。
- **同文件关系**：与 `relation_memory_ids` 构成「按 memory_id 对账 / 清理」的配套能力；内存分支操作 `self._observations`、`self._local`、`self._reverse`（均由 `add_relation` / `add_observation` 写入）；被上层记忆删除逻辑调用。

### `clear(self) -> None` （第 875 行）
- **作用**：清空整张语义图投影，把实体、观测、关系全部抹掉。它用于测试重置、环境切换或「重建全图」的场景。内存模式下就是把四份字典清空；Neo4j 模式下只删带 `MemoryEntity` 或 `MemoryObservation` 标签的节点（用 `DETACH DELETE` 连带删除它们的关系），因此不会误删图库里其他无关标签的数据。
- **参数**：无。
- **返回**：无返回值（`None`）。
- **内部流程**：若 `self.driver is None`，依次调用 `self._local.clear()`、`self._reverse.clear()`、`self._entities.clear()`、`self._observations.clear()` 后 `return`。否则拼 `MATCH (n) WHERE n:MemoryEntity OR n:MemoryObservation DETACH DELETE n`，在 `with self.driver.session(database=self.database) as session:` 中执行 `session.run(query)` 并调用 `result.consume()` 确保语句执行完毕。
- **异常/边界**：图库为空时执行不报错；`DETACH DELETE` 会一并删除这些节点上挂的所有关系；不走 `_with_session`，连接异常直接抛出；本方法不清除 `self.driver` 本身，也不重置 `_socket_patch_frame` 等连接状态。
- **同文件关系**：操作 `self._local`、`self._reverse`、`self._entities`、`self._observations`（与 `add_relation`、`add_observation`、`_merge_entity`、`delete_memory_relation` 写的是同一批容器）；被上层（测试或重置逻辑）调用。

### `close(self) -> None` （第 892 行）
- **作用**：释放存储对象占用的外部资源：关闭 Bolt 驱动，并卸载本实例安装的进程级 socket 补丁与代理 broker。它是对象生命周期的收尾动作，通常在应用关闭或 store 被替换时调用。由于补丁是进程级全局状态，这一步尤其重要——不调用它就会把全局 `socket.getaddrinfo` 一直留在被替换的状态。
- **参数**：无。
- **返回**：无返回值（`None`）。
- **内部流程**：若 `self.driver` 不为 `None` 且 `getattr(self.driver, "close", None)` 可调用，则调用 `self.driver.close()`；随后调用 `self._discard_socket_patch()` 完成帧卸载与 broker 关闭。注意它**没有**把 `self.driver` 置为 `None`，因此关闭后该实例的 `driver` 属性仍指向已关闭的驱动对象。
- **异常/边界**：`self.driver.close()` 的异常不被捕获（会向上抛出）；broker 关闭的异常在 `_discard_socket_patch` 内部被吞掉；`driver` 为 `None`（内存模式）时跳过驱动关闭，只做补丁清理（此时通常没有补丁，等于空操作）；重复调用 `close()` 会再次对已关闭的驱动调用 `close()`。
- **同文件关系**：调用 `_discard_socket_patch`（后者调用 `_uninstall_socket_patch`）；被应用关闭流程调用。

### `__all__ = ["Neo4jGraphStore"]` （第 898 行，模块级导出声明）
- **作用**：显式声明本模块对外导出的公开名字只有 `Neo4jGraphStore` 一个。这样 `from memory.storage.graph import *` 不会把顶部的四个补丁辅助函数（`_chained_getaddrinfo`、`_make_proxy_frame`、`_install_socket_patch`、`_uninstall_socket_patch`）以及模块级私有状态泄露出去，也让静态检查工具和 IDE 能正确识别本模块的公共 API 面。它本身不是函数或类，没有参数与返回值，只是模块命名空间里的一个列表常量。
- **参数**：无。
- **返回**：无（它是模块级变量，值为 `["Neo4jGraphStore"]`）。
- **内部流程**：模块导入时直接赋值，无执行逻辑。
- **异常/边界**：无特殊处理。
- **同文件关系**：指向本文件定义的类 `Neo4jGraphStore`。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_chained_getaddrinfo` | 替换全局 `socket.getaddrinfo` 的链式代理函数，让帧优先接管 Aura 主机解析、否则回退原始解析。 |
| `_make_proxy_frame` | 为一个 `ProxyBroker` 生成「Aura 主机 → 本地隧道」的链式帧闭包。 |
| `frame`（嵌套于 `_make_proxy_frame`） | 判定并执行单次主机名到本地隧道端口的重映射，不匹配则返回 `None` 让位。 |
| `_install_socket_patch` | 加锁、引用计数地把帧装入进程级补丁，仅在首个安装者时替换全局函数。 |
| `_uninstall_socket_patch` | 按引用卸载帧，最后一个卸载者恢复原始 `socket.getaddrinfo` 并清空全局状态。 |
| `Neo4jGraphStore` | 图关系存储类：Neo4j 与内存双模式下的实体、关系、观测事实的读写与时间切片。 |
| `__init__` | 初始化连接参数与四份内存索引，仅在给出 `uri` 且无外部驱动时建驱动。 |
| `_open_driver` | 导入 neo4j、校验认证、组装超时参数，必要时安装代理补丁并创建 Bolt 驱动。 |
| `_discard_socket_patch` | 幂等地卸载本实例的 socket 补丁并尽力关闭其代理 broker。 |
| `_reopen_driver` | 尽力关闭旧驱动与旧补丁后重建整套驱动，用于路由抖动恢复。 |
| `_with_session` | 统一的「开会话执行 runner + 路由类异常重连重试一次」包装。 |
| `add_relation` | 幂等写入一条实体间关系边并顺带维护两端实体属性，`bump=True` 时转为回忆强化。 |
| `_run`（嵌套于 `add_relation`） | 把关系写入的 Cypher 与参数送进会话并 `consume()` 执行。 |
| `add_observation` | 以 `observation_id` 为键幂等写入 n 元事实观测节点并重建其参与者边。 |
| `_run`（嵌套于 `add_observation`） | 把观测写入的 Cypher 与归一化参与者送进会话并 `consume()` 执行。 |
| `_merge_entity` | 内存模式下实体 upsert 的 Python 版本，只在别名非空时覆盖、domain/importance 仅首次写入。 |
| `update_entity` | 只用 `MATCH` 编辑已存在实体的 domain/aliases/importance，不创建新节点。 |
| `_run`（嵌套于 `update_entity`） | 执行实体更新语句并返回是否命中节点。 |
| `_bump_relation` | 给已存在的边累加召回计数、刷新访问时间、按增长因子放大权重并封顶。 |
| `_apply_bump` | 内存模式下回忆强化的纯计算：计数加一、时间刷新、权重乘因子后夹到上限。 |
| `entity` | 读取单个实体的 domain/aliases/importance，有驱动时以图库为准。 |
| `entity_aliases` | 一条查询批量取回全部实体的别名表，避免可视化时的 N+1 查询。 |
| `graph_snapshot` | 返回 Web 可视化用的全图投影（实体 / 观测 / 关系），支持 `at` 时间点切片。 |
| `_run`（嵌套于 `graph_snapshot`） | 在同一会话里跑完实体、观测、关系三条查询并打包成三元组。 |
| `_temporal_bounds` | 按 `valid_from`/`valid_to`/`event_at` 判定数据在给定时刻是否可见并回传时间值。 |
| `_observations_at` | 过滤过期/窗口外观测，并对时序观测按「主语+谓词」只保留最新一条。 |
| `_relations_at` | 过滤过期/窗口外边，并对时序边按「源+关系类型」只保留最新一条。 |
| `relation_memory_ids` | 收集所有关系边上的 `memory_id`，供上层与语义记忆做对账。 |
| `get_relations` | 按实体、关系类型、方向与时间点查询邻接边列表。 |
| `related` | `get_relations` 的类属性别名，提供更短的调用名。 |
| `path_query` | 查询两实体间最多 `max_depth` 跳的简单路径，按路径总权重降序、跳数升序返回。 |
| `_neighbours` | 内存模式下把某实体的出边与入边整理成 `(邻居, 关系, 是否出边, 权重)` 并按权重降序排序。 |
| `_local_paths` | 内存模式下用 BFS 枚举简单路径，短路径优先、同层强边优先。 |
| `delete_memory_relation` | 按 `memory_id` 删除对应关系边与观测节点，返回是否真的删到东西。 |
| `clear` | 清空整张语义图（内存四份字典或图库中带两个记忆标签的节点）。 |
| `close` | 关闭驱动并卸载本实例的 socket 补丁与代理 broker。 |
| `__all__` | 模块导出声明，只公开 `Neo4jGraphStore`。 |
