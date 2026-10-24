# core/repository.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时的「工具元数据仓库」层，只负责把可被发现的工具说明（ToolSpec）持久化进 SQLite，并对外提供保存、按名读取、按意图检索、列出启用工具名的能力。它明确声明自己不保存任何可执行代码，只保存描述性的元数据（名称、版本、描述、输入输出 schema、副作用、权限、超时、幂等性、并发安全性、并发上限、标签、推荐前置工具、启用开关，以及一个可选的实现引用字符串 implementation_ref）。整个文件只包含一个模块级类 `ToolSpecRepository`，所有功能都以它的实例方法、私有辅助方法和两个静态工具方法的形式提供；文件顶层只导入了 json、re、sqlite3、threading、Iterator、contextmanager、Path、Any，以及项目内的常量 `DEFAULT_TOOLS_DB_FILENAME` 和数据模型 `ToolSpec`。运行时它被当作一个「注册表 / 目录」使用：上层把工具规范写进去，之后在需要给模型挑选工具时调用 `search` 按自然语言意图打分召回，或者在需要枚举全部可用工具时调用 `active_tool_names`。它内部用一把可重入锁 `threading.RLock` 串行化所有数据库访问，因此可以被多线程（例如 FastAPI 的线程池工作线程）共享使用；同时它用「每次调用新建短连接」的方式适配磁盘数据库，用「共享单连接」的方式适配 `:memory:` 内存数据库，兼顾了文件库的并发安全与内存库的数据存活。它还在建表时内建了轻量级迁移逻辑（检查列是否存在并 ALTER TABLE 补列），以便老版本数据库文件可以继续被新代码打开。版本比较不是简单字符串比较，而是通过 `_version_key` 做自然排序并让预发布版本排在正式版本之前，从而保证 `get` 与 `search` 总能挑到「最新」的那一条记录。

## 二、函数与类逐条详解

### `ToolSpecRepository` （第 17 行）

- **作用**：这是一个封装 SQLite 的工具元数据仓库类，是本文件唯一的对外入口。它把「工具规范如何落库、如何反序列化、如何选版本、如何检索」这些细节全部收在类内部，对外只暴露 `save`、`get`、`search`、`active_tool_names`、`close` 这几个方法。它被设计成可被多线程共享的实例：内部持有一把 `threading.RLock`，所有数据库操作都要经过它。它还区分两种运行模式：路径为 `:memory:` 时使用一个跨线程共享的内存连接（因为内存库一旦连接关闭数据就消失），否则每次操作打开一个短生命周期的文件连接。类注释明确强调「persistence for discoverable tool metadata, never executable code」，即它只存元数据、不存可执行代码，这是它的安全边界。使用方式通常是：上层在某处创建一个实例（默认指向 `DEFAULT_TOOLS_DB_FILENAME`），启动时把内置工具 `save` 进去，运行时用 `get`/`search`/`active_tool_names` 查询，进程退出时调用 `close`。
- **参数**：无（类本身不接收参数；构造行为由 `__init__` 定义）。
- **返回**：无（类定义本身不返回值；实例化后得到 `ToolSpecRepository` 对象）。
- **内部流程**：类体内依次定义了 `__init__`、`_connect`、`_connection`、`close`、`_initialize`、`save`、`get`、`search`、`active_tool_names`、`_version_key`、`_decode`。类属性层面没有类变量，所有状态都挂在实例上：`self.path`（规范化后的数据库路径或 `:memory:`）、`self._lock`（可重入锁）、`self._shared_connection`（内存库专用共享连接，文件库模式恒为 None）。`_version_key` 与 `_decode` 被声明为 `@staticmethod`，因为它们不依赖实例状态。
- **异常/边界**：类本身不抛异常；异常都发生在构造与方法调用中（例如路径类型非法抛 `TypeError`，内存库被关闭后访问抛 `RuntimeError`，主键冲突时由 sqlite3 抛 `IntegrityError`）。
- **同文件关系**：它包含并组织本文件里的全部方法；被 `__init__`、`_connect`、`_connection`、`close`、`_initialize`、`save`、`get`、`search`、`active_tool_names`、`_version_key`、`_decode` 共同构成。它依赖本文件顶部导入的 `ToolSpec` 与 `DEFAULT_TOOLS_DB_FILENAME`。

### `__init__(self, path: str | Path = DEFAULT_TOOLS_DB_FILENAME) -> None` （第 20 行）

- **作用**：构造仓库实例，决定数据落在哪里，并把数据库结构与线程安全设施准备好。它做的第一件事是类型校验，第二件事是把传入路径规范化（展开 `~` 为用户主目录），第三件事是初始化锁与共享连接字段，第四件事是按需创建父目录或创建内存共享连接，最后调用 `_initialize` 建表并做列迁移。之所以要专门展开 `~`，代码注释里说明了原因：如果把字面量 `~` 原样交给 sqlite，sqlite 会在当前目录下创建一个真的名叫 `~` 的目录和数据库文件，而不是写进用户主目录，因此这里一次性规范化，后续建目录和连库都用同一个值。这个构造函数会在实例创建时立刻触发一次数据库写入（建表），所以路径不可写、磁盘不可用等错误会在构造阶段就暴露出来，而不是等到第一次 `save` 才失败。
- **参数**：
  - `path`：`str` 或 `pathlib.Path`，默认值为常量 `DEFAULT_TOOLS_DB_FILENAME`（工具库文件名，来自 `constants` 模块）。取值可以是普通文件路径、带 `~` 的路径、相对路径，或者特殊字符串 `":memory:"` 表示使用 SQLite 内存数据库。若传入其他类型（如 `None`、`int`）则直接抛错。
- **返回**：`None`。构造函数不返回业务值，副作用是设置实例属性并完成建表。
- **内部流程**：
  1. 用 `isinstance(path, (str, Path))` 校验类型，不通过则 `raise TypeError("path must be a string or pathlib.Path")`。
  2. 计算 `self.path`：如果 `str(path) == ":memory:"` 就原样保留 `":memory:"`；否则用 `str(Path(path).expanduser())` 规范化。注意判断用的是 `str(path)`，所以 `Path(":memory:")` 也会被识别为内存模式。
  3. 创建 `self._lock = threading.RLock()`，并把 `self._shared_connection` 初始化为 `None`。
  4. 如果 `self.path == ":memory:"`：调用 `sqlite3.connect(self.path, check_same_thread=False)` 建立共享连接，并把 `row_factory` 设为 `sqlite3.Row`（`check_same_thread=False` 是为了让多线程都能用这一条连接，真正的并发安全由 `self._lock` 保证）。
  5. 否则（文件模式）：如果 `Path(self.path).parent != Path(".")`，说明路径带有目录部分，于是用 `mkdir(parents=True, exist_ok=True)` 递归创建父目录；如果父目录就是当前目录 `.`，则不创建。
  6. 最后调用 `self._initialize()` 完成建表与迁移。
- **异常/边界**：`path` 类型非法抛 `TypeError`；父目录创建失败或文件不可写会由 `Path.mkdir` / `sqlite3.connect` / `_initialize` 抛出 `OSError`、`sqlite3.OperationalError` 等；路径为空字符串时 `Path("").parent` 为 `Path(".")`，不会创建目录，随后由 sqlite 决定行为。对 `None` 没有兜底，直接抛 `TypeError`。
- **同文件关系**：调用本文件的 `_initialize()`；`_initialize()` 又间接依赖 `_connection()` 与 `_connect()`。它设置的 `self.path`、`self._lock`、`self._shared_connection` 被本文件其余所有方法读取或使用。

### `_connect(self) -> sqlite3.Connection` （第 39 行）

- **作用**：这是一个底层「取得一条可用连接」的工厂方法，被 `_connection` 独占调用。它的存在是为了让「内存库」和「文件库」两种模式在上层看起来一致：内存模式直接复用构造时建立的共享连接（保证数据不丢），文件模式则每次新建一条短连接（保证线程之间不共享 sqlite 连接对象）。它还承担了一个状态检查职责：当内存库已经被 `close()` 关闭后，`self._shared_connection` 变成 `None`，此时再访问就抛 `RuntimeError("repository is closed")`，而不是让 sqlite 去连一个字面名为 `:memory:` 的文件。文件模式下每次新建的连接都会设置 `row_factory = sqlite3.Row`，这样上层就能按列名取值（`row["tool_name"]`）而不是按下标。
- **参数**：无（仅 `self`）。
- **返回**：`sqlite3.Connection`。内存模式下返回同一个共享连接对象（同一个引用会被反复返回）；文件模式下返回一条全新的、调用方负责关闭的连接。
- **内部流程**：
  1. 若 `self._shared_connection is not None`，直接 `return` 它，文件模式不会走到这里（文件模式下该字段恒为 None）。
  2. 若 `self.path == ":memory:"`（说明是内存模式但共享连接已被关闭），抛 `RuntimeError("repository is closed")`。
  3. 否则执行 `sqlite3.connect(self.path)`，设置 `connection.row_factory = sqlite3.Row`，返回该连接。
- **异常/边界**：内存库已关闭时抛 `RuntimeError`；文件路径不可访问时由 `sqlite3.connect` 抛 `sqlite3.OperationalError`（例如目录不存在、无权限）。注意文件模式下每次调用都会泄漏出「需要调用方关闭」的连接，关闭动作统一由 `_connection` 的 `finally` 负责，`_connect` 自身不关闭任何东西。
- **同文件关系**：只被 `_connection()` 调用；它读取 `self._shared_connection` 与 `self.path`（这两个字段由 `__init__` 与 `close` 维护）。

### `_connection(self) -> Iterator[sqlite3.Connection]` （第 48 行）

- **作用**：这是一个被 `@contextmanager` 装饰的上下文管理器，是整个类访问数据库的唯一通道，所有 `execute` 都写在 `with self._connection() as connection:` 里面。它一次性解决三件事：串行化（进入时获取 `self._lock`，退出时释放）、事务化（用 `with connection:` 包裹 `yield`，正常退出自动 commit、抛异常自动 rollback）、连接生命周期（如果这条连接不是共享连接，则在 `finally` 中关闭）。由于 `RLock` 是可重入的，同一个线程在持有锁的情况下再次进入（例如嵌套调用）不会死锁；但由于锁在整个 `with` 块期间都被持有，不同线程之间的所有数据库操作都会被严格串行化，这是本类线程安全的核心机制。文档字符串「Yield a serialized shared connection or a short-lived file connection」正是这个含义。
- **参数**：无（仅 `self`）。
- **返回**：作为生成器函数，它本身返回一个上下文管理器；`with` 语句进入后 `as` 到的是 `sqlite3.Connection` 对象。
- **内部流程**：
  1. `with self._lock:` 获取可重入锁。
  2. 调用 `self._connect()` 拿到连接。
  3. `with connection:` 开启 sqlite 事务上下文，然后把连接 `yield` 给调用方。
  4. 调用方代码块正常结束后，`with connection:` 触发 commit；若调用方抛异常，则触发 rollback 并把异常继续向上抛。
  5. `finally:` 中判断 `connection is not self._shared_connection`，条件成立（文件模式的新连接）则 `connection.close()`；共享连接不会被关闭。
- **异常/边界**：进入阶段可能抛出 `_connect` 的 `RuntimeError` 或 `sqlite3.OperationalError`；调用方块内的异常会被原样传播，同时先做回滚；关闭连接时若 sqlite 报错也会抛出。对 `:memory:` 共享连接，即使调用方代码块抛异常，连接对象本身仍然保持打开，数据保留。
- **同文件关系**：调用 `_connect()`；被本文件的 `_initialize()`、`save()`、`get()`、`search()`、`active_tool_names()` 全部调用。它依赖 `close()` 与 `__init__` 维护的 `self._shared_connection`。

### `close(self) -> None` （第 60 行）

- **作用**：释放仓库占用的数据库资源，语义上只针对内存模式。因为文件模式下每次操作都是短连接、用完即关，实例本身不长期持有连接，所以 `close` 对文件模式实际上什么也不做；而对 `:memory:` 模式，它会把那条共享连接真正关掉并把字段置回 `None`，使后续任何访问都通过 `_connect` 的检查抛 `RuntimeError("repository is closed")`。这样做可以显式表达「这个仓库不再可用」，同时把内存库占用的资源还给解释器。整个关闭动作同样在 `self._lock` 保护下进行，避免与正在执行的查询并发冲突。它是幂等的：重复调用第二次时因为字段已是 `None` 而不会重复关闭。
- **参数**：无（仅 `self`）。
- **返回**：`None`。
- **内部流程**：
  1. `with self._lock:` 获取锁。
  2. 判断 `self._shared_connection is not None`；成立则调用其 `close()`，然后把 `self._shared_connection = None`。
  3. 不成立（文件模式，或已经关闭过）则直接结束。
- **异常/边界**：若底层 `sqlite3.Connection.close()` 抛错则向上传播；重复调用安全（幂等）；对文件模式是无操作，不会删除磁盘上的数据库文件。
- **同文件关系**：不调用本文件其它函数；它修改的 `self._shared_connection` 被 `_connect()` 和 `_connection()` 读取。

### `_initialize(self) -> None` （第 67 行）

- **作用**：在构造阶段被调用一次，负责创建 `tool_specs` 表并在必要时补齐缺失的列，相当于一个极简的「自带迁移」逻辑。它保证无论传入的是一个全新路径还是一个由旧版本代码创建过的数据库文件，实例创建完成后表结构都是当前代码期望的形态。建表语句里 `PRIMARY KEY (tool_name, version)` 说明同一个工具名可以有多个版本共存，这也是后续 `get`/`search` 需要做版本挑选的前提。除了主键，表上还带了一些默认值：`recommended_before_tools` 默认 `'[]'`、`enabled` 默认 `1`，这让老数据在被读出时仍能得到合理的布尔与列表语义。迁移部分只处理两个后来新增的列，且用 `PRAGMA table_info` 探测而不是盲目 ALTER，避免重复加列报错。
- **参数**：无（仅 `self`）。
- **返回**：`None`。
- **内部流程**：
  1. `with self._connection() as connection:` 取得带事务的连接。
  2. 执行 `CREATE TABLE IF NOT EXISTS tool_specs (...)`，列依次为：`tool_name TEXT NOT NULL`、`version TEXT NOT NULL`、`description TEXT NOT NULL`、`schema_hash TEXT NOT NULL`、`input_schema TEXT NOT NULL`、`output_schema TEXT NOT NULL`、`side_effect TEXT NOT NULL`、`permissions TEXT NOT NULL`、`timeout_seconds REAL NOT NULL`、`idempotent INTEGER NOT NULL`、`parallel_safe INTEGER NOT NULL`、`max_concurrency INTEGER`（可为空）、`tags TEXT NOT NULL`、`recommended_before_tools TEXT NOT NULL DEFAULT '[]'`、`enabled INTEGER NOT NULL DEFAULT 1`、`implementation_ref TEXT`（可为空），并定义 `PRIMARY KEY (tool_name, version)`。
  3. 执行 `PRAGMA table_info(tool_specs)`，用集合推导取出所有列名（`row[1]` 是列名），得到 `columns`。
  4. 如果 `"max_concurrency" not in columns`，执行 `ALTER TABLE tool_specs ADD COLUMN max_concurrency INTEGER`。
  5. 如果 `"recommended_before_tools" not in columns`，执行 `ALTER TABLE tool_specs ADD COLUMN recommended_before_tools TEXT NOT NULL DEFAULT '[]'`。
  6. 退出 `with` 块时提交。
- **异常/边界**：数据库只读或磁盘满会由 sqlite 抛 `sqlite3.OperationalError`；对已存在但列名不同的表不会做结构性纠正，只会尝试补这两列；`PRAGMA` 的取值依赖列顺序约定（索引 1 为列名），这是 sqlite 的稳定行为。无特殊空值处理需求。
- **同文件关系**：调用 `_connection()`（进而调用 `_connect()`）；只被 `__init__()` 调用。它为 `save`、`get`、`search`、`active_tool_names`、`_decode` 提供了表结构前提。

### `save(self, spec: ToolSpec, *, implementation_ref: str | None = None, replace: bool = False) -> None` （第 105 行）

- **作用**：把一个 `ToolSpec` 对象写入 `tool_specs` 表，是仓库的写入入口。它负责把对象里的 Python 结构转换成 SQLite 能存的标量：字典类型的 `input_schema`、`output_schema` 用 `json.dumps(..., sort_keys=True)` 序列化（排序键是为了让同样的 schema 产生稳定的字符串表示，便于哈希比对与去重），列表类型的 `permissions`、`tags`、`recommended_before_tools` 用 `json.dumps(list(...))` 序列化，布尔类型的 `idempotent` 与 `parallel_safe` 用 `int()` 转成 0/1。它支持两种写入语义：默认的 `INSERT`（已存在同主键则报错，用于防止意外覆盖）和 `replace=True` 时的 `INSERT OR REPLACE`（用于升级/重装时覆盖同名同版本记录）。`implementation_ref` 是一个可选的外部实现引用字符串（例如某个模块路径或标识符），它只是被原样存入 `implementation_ref` 列，仓库不会去解析或执行它。`enabled` 列在 VALUES 中被硬编码为 `1`，也就是说通过 `save` 写入的记录一律是启用状态。
- **参数**：
  - `spec`：必填，必须是 `ToolSpec` 实例，否则抛 `TypeError`。它需要提供 `name`、`version`、`description`、`schema_hash`、`input_schema`、`output_schema`、`side_effect`、`permissions`、`timeout_seconds`、`idempotent`、`parallel_safe`、`max_concurrency`、`tags`、`recommended_before_tools` 这些属性。
  - `implementation_ref`：关键字参数，`str` 或 `None`，默认 `None`。传入非 `str` 且非 `None` 的值抛 `TypeError`；对应数据库中的可空列 `implementation_ref`。
  - `replace`：关键字参数，`bool`，默认 `False`。必须严格是布尔值（`isinstance(replace, bool)` 校验，因此 `0`/`1` 这类整数会被拒绝）。为 `True` 时用 `INSERT OR REPLACE`，为 `False` 时用 `INSERT`。
- **返回**：`None`。成功时数据库多出一行（或覆盖一行）；失败时抛异常。
- **内部流程**：
  1. 依次校验 `spec` 是 `ToolSpec`、`replace` 是 `bool`、`implementation_ref` 是 `str` 或 `None`，任一不满足立即 `raise TypeError`。
  2. 根据 `replace` 选择 `statement = "INSERT OR REPLACE"` 或 `"INSERT"`，然后用 f-string 把它拼进 SQL（SQL 骨架本身是常量字符串，用户数据一律走 `?` 占位符参数绑定）。
  3. `with self._connection() as connection:` 打开事务。
  4. `connection.execute(...)` 执行插入，参数元组按列顺序给出：`spec.name`、`spec.version`、`spec.description`、`spec.schema_hash`、`json.dumps(spec.input_schema, sort_keys=True)`、`json.dumps(spec.output_schema, sort_keys=True)`、`spec.side_effect`、`json.dumps(list(spec.permissions))`、`spec.timeout_seconds`、`int(spec.idempotent)`、`int(spec.parallel_safe)`、`spec.max_concurrency`、`json.dumps(list(spec.tags))`、`json.dumps(list(spec.recommended_before_tools))`、`implementation_ref`；`enabled` 由 SQL 里的字面量 `1` 填充，不来自参数。
  5. 退出 `with` 块时提交事务。
- **异常/边界**：类型错误抛 `TypeError`（含中文注释中未提及的三种校验）；`replace=False` 且 `(tool_name, version)` 已存在时 sqlite 抛 `sqlite3.IntegrityError`；`spec.input_schema` 等字段若不是可 JSON 序列化的对象，`json.dumps` 抛 `TypeError`；`spec.permissions` 等若不是可迭代对象，`list()` 抛 `TypeError`；`timeout_seconds` 若为 `None` 会违反 `NOT NULL` 约束抛 `IntegrityError`。`max_concurrency` 允许为 `None`，因为该列可为空。
- **同文件关系**：调用 `_connection()`（进而调用 `_connect()`）；不被本文件其它函数调用，是纯写入入口。它写入的数据随后被 `get()`、`search()`、`active_tool_names()` 读取，并由 `_decode()` 反序列化。

### `get(self, tool_name: str, version: str | None = None) -> dict[str, Any] | None` （第 148 行）

- **作用**：按工具名（可选指定版本）精确读取一条工具元数据，返回已经反序列化好的字典。它是运行时最常用的读接口：上层想知道某个工具存在不存在、它的 schema 和权限是什么，就调它。当 `version` 明确给出时，它做的是「精确匹配某一行」；当 `version` 为 `None` 时，它会把该工具名下所有启用版本都取出来，然后用 `_version_key` 比较，返回版本号最大的那一条，也就是「最新版本」。查询条件里固定带 `enabled = 1`，因此被禁用（enabled 为 0）的记录对 `get` 完全不可见，表现得像不存在一样返回 `None`。方法开头做了一组防御性校验，把工具名和版本号的类型、非空、长度都限制住，避免异常长的字符串进入数据库查询或造成无意义的内存开销。
- **参数**：
  - `tool_name`：必填，`str`。必须是非空字符串（用 `strip()` 判断，全空白也算非法）且长度不超过 200 个字符，否则抛 `ValueError`。
  - `version`：可选，`str | None`，默认 `None`。为 `None` 时表示「取最新版本」；传入字符串时必须非空且长度不超过 32 个字符；传入非字符串且非 `None`（例如整数）也抛 `ValueError`。
- **返回**：`dict[str, Any] | None`。指定 `version` 且命中时返回 `_decode` 后的单行字典，未命中返回 `None`；未指定 `version` 时，若有任意启用版本则返回版本号最大的一行字典，一个版本都没有则返回 `None`。
- **内部流程**：
  1. 校验 `tool_name`：`not isinstance(tool_name, str) or not tool_name.strip()` → `ValueError("tool_name must be a non-empty string")`；`len(tool_name) > 200` → `ValueError("tool_name must be at most 200 characters")`。
  2. 校验 `version`：非 `None` 时若类型不是 `str` 或 `strip()` 后为空 → `ValueError("version must be a non-empty string or None")`；若是字符串且 `len > 32` → `ValueError("version must be at most 32 characters")`。
  3. 构造基础 SQL `SELECT * FROM tool_specs WHERE tool_name = ? AND enabled = 1`，参数列表初始为 `[tool_name]`。
  4. 若 `version is not None`，在 SQL 后追加 `" AND version = ?"` 并把 `version` 追加进参数。
  5. `with self._connection() as connection:` 执行 `fetchall()` 取回 `rows`。
  6. 若指定了 `version`：`rows` 非空就返回 `self._decode(rows[0])`，否则返回 `None`（因为主键含 version，最多一行）。
  7. 若未指定 `version`：对每一行调用 `_decode` 得到 `decoded` 列表；若列表非空，用 `max(decoded, key=lambda item: self._version_key(item["version"]))` 取版本最大者返回，否则返回 `None`。
- **异常/边界**：参数非法抛 `ValueError`；数据库错误由 `_connection` 传播；`enabled = 0` 的记录被过滤，等价于不存在；未指定版本时若所有候选版本号都是空串，`_version_key` 仍能处理（返回可比较的元组），不会抛错；被 `close()` 关闭的内存仓库会抛 `RuntimeError`。注意返回值里的 JSON 字段是直接 `json.loads` 出来的，若历史数据里存了非法 JSON，`_decode` 会抛 `json.JSONDecodeError`。
- **同文件关系**：调用 `_connection()` 与 `_decode()`，并通过 `key` 函数调用 `_version_key()`；不被本文件其它函数调用，供外部查询使用。

### `search(self, intent: str, limit: int = 5) -> list[dict[str, Any]]` （第 175 行）

- **作用**：按一段自然语言「意图」文本从工具目录里召回最相关的若干工具，是给模型做工具选择时用的检索接口。它先在内存里把整张启用表读出来并做「每个工具只留最新版本」的去重，然后根据 `intent` 分词（按空白切分并统一转小写）对每个候选打分：候选的「可搜文本」由工具名、描述、标签拼接而成，分数等于命中的不同查询词个数；最后按「分数降序、工具名升序」排序，只保留分数大于 0 的项，并截断到 `limit` 条。如果 `intent` 去掉空白后没有任何词，它不做打分，而是直接把按工具名排序后的前 `limit` 条返回，相当于「列出目录」。这种实现是朴素的词命中打分（子串包含判断），不涉及词干还原、同义词或向量检索，但胜在无依赖、可预测、对小规模工具目录足够快。
- **参数**：
  - `intent`：必填，`str`。类型不是字符串抛 `TypeError`；长度超过 500 个字符抛 `ValueError`。允许为空串或全空白（此时走「无词」分支）。
  - `limit`：可选，`int`，默认 `5`。必须严格是整数且满足 `1 <= limit <= 20`；布尔值会被显式排除（`isinstance(limit, bool)` 直接判非法），因此 `True`/`False` 也会抛 `ValueError`。
- **返回**：`list[dict[str, Any]]`，元素是 `_decode` 后的工具字典，列表长度最多为 `limit`，可能为空列表。无词时返回按 `tool_name` 排序的前 `limit` 条（全部是启用工具的最新版本）；有词时只返回至少命中一个词的条目。
- **内部流程**：
  1. 校验 `intent` 类型与长度。
  2. 校验 `limit`：`isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20` → `ValueError("limit must be an integer between 1 and 20")`。
  3. 分词：`terms = [term for term in intent.casefold().split() if term]`，即大小写折叠后按空白切分并丢掉空串。
  4. `with self._connection() as connection:` 执行 `SELECT * FROM tool_specs WHERE enabled = 1` 并 `fetchall()`，把整张表读进内存。
  5. 对每行调用 `_decode`，然后遍历 `decoded`，用字典 `latest` 按 `tool_name` 保留 `_version_key` 更大的那一条（即每个工具的最新版本）。
  6. `decoded = sorted(latest.values(), key=lambda item: item["tool_name"])`，得到一个按名字稳定排序的候选列表。
  7. 若 `terms` 为空，直接 `return decoded[:limit]`。
  8. 否则遍历候选，把 `f"{item['tool_name']} {item['description']} {' '.join(item['tags'])}"` 折叠大小写后作为 `haystack`，计算 `sum(term in haystack for term in terms)`（布尔求和，重复词只算一次命中，因为每个 term 只判断一次）并放进 `ranked` 列表，元素是 `(score, item)` 二元组。
  9. `ranked.sort(key=lambda pair: (-pair[0], pair[1]["tool_name"]))`，实现分数降序、同分按名字升序的确定性排序。
  10. 返回 `[item for score, item in ranked if score > 0][:limit]`，丢弃零分候选后再截断。
- **异常/边界**：`intent` 非字符串抛 `TypeError`；`intent` 过长、`limit` 越界或为布尔抛 `ValueError`；空/全空白意图不报错而是退化为「按名字列目录」；命中数为 0 时返回空列表而不是报错；若某行 `tags` 字段存的不是列表，`' '.join(item['tags'])` 会抛 `TypeError`，若 `description` 为 `None` 则在 f-string 中变成字符串 `"None"` 而不报错（但该列有 `NOT NULL` 约束，正常写入不会为 `None`）；数据库错误与 `RuntimeError`（仓库已关闭）照常传播。该实现把整表读进内存，工具数量极大时内存与耗时随行数线性增长，且不做分页。
- **同文件关系**：调用 `_connection()`、`_decode()`，并通过排序/比较间接调用 `_version_key()`；不被本文件其它函数调用，是外部检索入口。

### `active_tool_names(self) -> list[str]` （第 209 行）

- **作用**：返回当前所有启用状态的工具名列表，按名字升序排列，且不带 `search` 的 1~20 条数量上限。它存在的意义是给「需要完整工具清单」的场景用，例如构建工具索引、做一致性校验、把全部工具名注入系统提示词，或者判断某个工具是否被注册过。它用 SQL 的 `SELECT DISTINCT tool_name` 直接在数据库层去重，所以同一个工具的多个版本只会出现一次；`WHERE enabled = 1` 保证被禁用的工具不会出现在清单里。文档字符串「Return every enabled tool name without the catalog search limit」明确点出了它与 `search` 的区别：`search` 受 `limit` 约束且只返回字典、需要打分，而它只返回名字字符串、数量不受限。
- **参数**：无（仅 `self`）。
- **返回**：`list[str]`。元素是工具名字符串，按 `tool_name` 升序；没有任何启用工具时返回空列表 `[]`。
- **内部流程**：
  1. `with self._connection() as connection:` 打开事务连接。
  2. 执行 `"SELECT DISTINCT tool_name FROM tool_specs WHERE enabled = 1 ORDER BY tool_name"` 并 `fetchall()` 得到 `rows`。
  3. 用列表推导 `[str(row["tool_name"]) for row in rows]` 把每行转成字符串返回（这里额外套了一层 `str()`，即使列值不是文本类型也能得到字符串）。
- **异常/边界**：无参数校验；数据库错误、仓库已关闭（`RuntimeError`）会向上传播；无数据返回空列表而非 `None`；不做版本挑选，因为去重后名字本身就与版本无关。
- **同文件关系**：调用 `_connection()`（进而调用 `_connect()`）；不被本文件其它函数调用。

### `_version_key(version: str) -> tuple[object, object]` （第 219 行）

- **作用**：这是一个静态工具方法，把版本号字符串转换成一个可直接比较的排序键，用来替代朴素的字符串比较。朴素的字符串比较会得出 `"10" < "2"` 这种错误结论，所以这里把版本按「数字段」和「非数字段」切开：数字段转成 `int` 并打上标记 `0`，非数字段折叠大小写并打上标记 `1`，从而让 `2` 排在 `10` 前面（自然排序），同时保证数字与文本混排时不会因为类型不同而抛 `TypeError`。它还把版本拆成「核心部分」和「预发布后缀」两部分（以第一个 `-` 为界），并给没有后缀的正式版本 `release_key = 1`、有后缀的预发布版本 `release_key = 0`，于是 `1.0` 会排在 `1.0-alpha` 之后（即被认为更新）。文档字符串还说明：不属于语义化版本组成部分的标签，仍然按可比较的字符串处理。`get` 用它挑最新版本，`search` 用它做每个工具的最新版本去重。
- **参数**：
  - `version`：`str`，要转换的版本号文本。方法本身不做类型校验，但会先 `version.strip()`，因此传入带首尾空白的字符串是安全的；传入 `None` 会在 `strip()` 处抛 `AttributeError`。
- **返回**：`tuple[object, object]`，形如 `(core_key, (release_key, suffix_key))`。其中 `core_key` 是由 `(0, int)` 或 `(1, str)` 组成的元组，`release_key` 是 `0` 或 `1`，`suffix_key` 结构与 `core_key` 相同。因为同一位置的元素类型标记一致，所以两个键之间可以直接用 `<`、`>` 比较；也正因为键是元组，`max()` 和 `sort()` 都能直接使用。
- **内部流程**：
  1. `normalized = version.strip()` 去掉首尾空白。
  2. `core, separator, suffix = normalized.partition("-")` 以第一个连字符切成核心版本、分隔符和预发布后缀三部分；没有连字符时 `separator` 为空串、`suffix` 为空串。
  3. `parts = re.findall(r"\d+|\D+", core)`，用正则把核心部分拆成连续数字块和连续非数字块。
  4. 构造 `core_key`：每个 part 若是 `part.isdigit()` 则编码为 `(0, int(part))`，否则编码为 `(1, part.casefold())`。
  5. `release_key = 1 if not separator else 0`：没有连字符（正式版）得 1，有连字符（预发布）得 0。
  6. `suffix_key`：对 `suffix` 同样用 `re.findall(r"\d+|\D+", suffix)` 拆分并做相同的 `(0, int)` / `(1, str)` 编码。
  7. 返回 `core_key, (release_key, suffix_key)`。
- **异常/边界**：`version` 为 `None` 时 `strip()` 抛 `AttributeError`；空字符串是允许的，会得到 `core_key = ()`、`release_key = 1`、`suffix_key = ()`，仍可与其他键比较；非字符串类型同样在 `strip()` 阶段失败。方法不修改任何状态，纯函数。
- **同文件关系**：被 `get()`（作为 `max` 的 key 函数）和 `search()`（用于每个工具的最新版本去重）调用；它自己不调用本文件任何函数，只使用 `re` 模块。

### `_decode(row: sqlite3.Row) -> dict[str, Any]` （第 240 行）

- **作用**：这是一个静态工具方法，把 sqlite 返回的一行原始记录转换成对上层友好的 Python 字典：先 `dict(row)` 把 `sqlite3.Row` 变成普通字典，然后把几个以 JSON 文本形式存储的列解析回 Python 对象，最后把三个以 0/1 存储的布尔语义列转成真正的 `bool`。它是「存储表示」与「领域表示」之间的唯一转换点，`get` 和 `search` 的所有返回值都要经过它，所以只要改这里的解析规则，整个仓库的读出结果就会统一变化。需要注意 `implementation_ref` 列不在 JSON 解析名单里，因为它本来就是可空字符串，直接原样透传即可；`enabled` 虽然也被转换，但 `get`/`search` 的 SQL 已经过滤了 `enabled = 1`，所以调用方通常看到的是 `True`。
- **参数**：
  - `row`：`sqlite3.Row`，一行查询结果。方法依赖连接上设置了 `row_factory = sqlite3.Row`（由 `__init__` 或 `_connect` 保证），否则 `dict(row)` 的行为会不同；方法本身不做类型校验。
- **返回**：`dict[str, Any]`。键是列名，值为解析后的对象：`input_schema`、`output_schema` 是反序列化后的 Python 对象（通常是 dict），`permissions`、`tags`、`recommended_before_tools` 是列表，`enabled`、`idempotent`、`parallel_safe` 是 `bool`，其余列（如 `tool_name`、`version`、`description`、`schema_hash`、`side_effect`、`timeout_seconds`、`max_concurrency`、`implementation_ref`）保持数据库原值。
- **内部流程**：
  1. `item = dict(row)`，把行对象浅拷贝成普通字典，后续就地修改。
  2. 遍历固定的五个键 `("input_schema", "output_schema", "permissions", "tags", "recommended_before_tools")`，对每个键执行 `item[key] = json.loads(item[key])`，把 JSON 文本还原成 Python 对象。
  3. 依次执行 `item["enabled"] = bool(item["enabled"])`、`item["idempotent"] = bool(item["idempotent"])`、`item["parallel_safe"] = bool(item["parallel_safe"])`，把 0/1 变成 `False`/`True`。
  4. 返回 `item`。
- **异常/边界**：若某个 JSON 列的文本非法，`json.loads` 抛 `json.JSONDecodeError`；若某列值为 `None`，`json.loads(None)` 抛 `TypeError`（正常写入路径不会产生这种数据，因为这几列都是 `NOT NULL`）；若字典里缺少上述任一键（例如有人写了只 SELECT 部分列的查询），`item[key]` 抛 `KeyError`；布尔转换对 `None` 会得到 `False` 而不报错，但由于列有 `NOT NULL` 约束，实际不会遇到。方法无副作用、不改数据库。
- **同文件关系**：不调用本文件其它函数；被 `get()` 和 `search()` 调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `ToolSpecRepository` | 封装 SQLite 的工具元数据仓库类，提供保存、读取、检索、列举工具规范的能力并保证线程安全。 |
| `__init__` | 校验并规范化数据库路径，准备锁与共享连接，创建父目录或内存连接，然后建表迁移。 |
| `_connect` | 按模式返回共享内存连接或新建文件连接，并在内存库已关闭时抛 `RuntimeError`。 |
| `_connection` | 用 `@contextmanager` 把加锁、事务提交/回滚、连接关闭三件事打包成唯一的数据库访问通道。 |
| `close` | 关闭内存模式的共享连接并置空字段，对文件模式是无操作且可重复调用。 |
| `_initialize` | 创建 `tool_specs` 表，并通过 `PRAGMA table_info` 探测补齐缺失的列以实现轻量迁移。 |
| `save` | 把 `ToolSpec` 序列化写入数据库，可按 `replace` 选择插入或覆盖，`enabled` 固定为 1。 |
| `get` | 按工具名精确取一条记录，未指定版本时用 `_version_key` 挑出最新版本，只返回启用项。 |
| `search` | 把意图分词后在工具名、描述、标签上做词命中打分，返回每个工具最新版本中得分最高的前 `limit` 条。 |
| `active_tool_names` | 去重列出所有启用工具名并升序排列，不受 `search` 的条数上限约束。 |
| `_version_key` | 把版本号转成支持自然排序且让预发布版本更旧的元组比较键。 |
| `_decode` | 把一行 sqlite 记录转成字典，并把 JSON 列反序列化、把 0/1 列转成布尔值。 |
