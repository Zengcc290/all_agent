# core/update_log.py

## 一、这个文件是干什么的

这个文件是项目里「更新日志（update log）」的持久化层，负责把每一次项目变更以**只追加（append-only）**的方式写进 SQLite，并提供按 ID 精确读取的能力。它的核心设计理念写在文件开头的模块 docstring 里：写入方只需要拿到刚插入那一行的新 ID 和时间戳，历史条目继续留在 SQLite 里，**不会被整体加载进 AI 的上下文窗口**，从而避免上下文被大量历史日志撑爆。

文件里只定义了一个公开类 `UpdateLogRepository`（线程安全的 SQLite 仓储），以及若干私有辅助方法（`_connect`、`_connection`、`_initialize`、`_decode`）和一个公开的 `close` 方法。表结构在 `_initialize` 里用 `CREATE TABLE IF NOT EXISTS update_logs` 建出来，共 14 个字段：自增主键 `update_id`、`timestamp`、`system_name`、`executor`、`update_type`、`title`、`task_background`、`update_details`、`added_features`、`files_json`（文件变更列表的 JSON 文本）、`behavior_impact`、`validation`、`risks`、`follow_up`。

在项目运行中，它被用作「写日志」和「查日志」的唯一入口：Agent 完成一次改动后调用 `append()` 落库并拿到确认信息；需要审计、测试或回看某几条记录时调用 `get()` / `get_range()` / `latest_id()`。数据库路径默认取环境变量 `UPDATE_LOG_DB_PATH`，否则回退到常量 `DEFAULT_UPDATE_LOG_FILENAME`；路径为 `":memory:"` 时走共享内存连接（测试常用）。模块末尾通过 `__all__` 只导出 `DEFAULT_UPDATE_LOG_FILENAME` 和 `UpdateLogRepository` 两个名字。

## 二、函数与类逐条详解

### `class UpdateLogRepository` （第 24 行）
- **作用**：这是本文件唯一的类，代表「一次项目变更 = 一行不可变记录」的线程安全 SQLite 存储。它把连接管理、建表、路径归一化、并发加锁、写入校验、读取解码全部封装在一个对象里，调用方不需要自己碰 `sqlite3`。之所以需要它，是因为项目里可能有多处代码（不同线程）同时写更新日志，而 SQLite 在多线程/多连接下容易因为写锁冲突报 `database is locked`，所以它用 `threading.RLock` 串行化所有数据库操作，并给连接设置 `busy_timeout`。它在运行时的典型用法是：构造一个实例（触发建表），之后反复调用 `append()` 记录变更、调用 `get()`/`get_range()` 做审计回看。类本身不继承任何基类，也不实现 `__enter__`/`__exit__`，所以不能直接用于 `with` 语句；内存模式下用完需要显式调用 `close()` 释放共享连接。
- **参数**：类构造层面没有参数（参数在 `__init__` 里），但语义上有一个关键的「存储位置」概念：由 `path` 决定是文件数据库还是内存数据库。
- **返回**：类不是函数，实例化后返回一个绑定了 `path`、`_lock`、`_shared_connection` 三个实例属性、并且已经完成建表的仓储对象。
- **内部流程**：定义类属性级别的 docstring 后，依次定义 `__init__`（构造与建表）、`_connect`（拿连接）、`_connection`（上下文管理器，加锁 + PRAGMA + 事务 + 关连接）、`close`（释放内存连接）、`_initialize`（建父目录 + 建表）、`append`（校验 + 插入 + 返回确认）、`get`（按主键取一行）、`latest_id`（取最大 ID）、`get_range`（按区间批量取）、`_decode`（把 `files_json` 反序列化成 `files`）。
- **异常/边界**：类本身不抛异常；实例化时如果 `path` 类型非法会由 `__init__` 抛 `TypeError`，建表失败（如目录无写权限、磁盘只读）会由 sqlite3 抛 `sqlite3.OperationalError`。
- **同文件关系**：它内部调用了本文件的 `_initialize`、`_connect`、`_connection`、`_decode`；`append`、`get`、`latest_id`、`get_range`、`close` 都是它的方法；本文件没有其它函数调用它，但它是模块唯一对外暴露的功能载体。

### `__init__(self, path: str | Path | None = None) -> None` （第 27 行）
- **作用**：构造函数，决定日志数据库落在哪里，并立刻把仓储准备好（内存模式下建共享连接，所有模式下都执行建表）。它存在的意义是让调用方既能显式指定路径（测试里用 `":memory:"` 或临时目录），也能完全不传参走「环境变量 → 默认文件名」的自动回退，从而在不同部署环境里零配置可用。它还承担了类型校验和路径归一化的职责：把 `~/...` 展开成绝对家目录路径，保证后续「建父目录」和「真正连库」用的是同一个位置，不会出现建目录建到了字面量 `~` 目录下的经典 bug。构造末尾无条件调用 `_initialize()`，因此**「实例创建成功」就等价于「表已存在」**，后续方法不必再关心建表。实例属性 `_shared_connection` 只在内存模式下非空，这是区分两种模式的关键开关。
- **参数**：
  - `path`（`str | pathlib.Path | None`，默认 `None`）：SQLite 数据库位置。为 `None` 时先读环境变量 `UPDATE_LOG_DB_PATH`，该变量为空字符串或未设置时回退到常量 `DEFAULT_UPDATE_LOG_FILENAME`；传字符串 `":memory:"` 时表示使用进程内内存数据库（不走磁盘，且所有操作复用同一个连接）；其它字符串或 `Path` 会被 `Path(path).expanduser()` 归一化。不允许传其它类型（如 `int`、`bytes`）。
- **返回**：无返回值（`None`），但会把实例置为可用状态：`self.path`（字符串形式最终路径）、`self._lock`（`threading.RLock`）、`self._shared_connection`（内存模式下的连接，否则 `None`）三个属性就绪，且数据库表已建好。
- **内部流程**：第一步，若 `path is None`，用 `os.getenv("UPDATE_LOG_DB_PATH") or DEFAULT_UPDATE_LOG_FILENAME` 求默认值（注意 `or` 意味着空字符串环境变量也会被当成「未设置」而回退）。第二步，用 `isinstance(path, (str, Path))` 校验类型，否则抛 `TypeError`。第三步，把 `self.path` 设成 `":memory:"`（原样保留）或 `str(Path(path).expanduser())`（展开 `~` 并转字符串）。第四步，创建 `self._lock = threading.RLock()`（可重入锁，因为 `_initialize` 会在已持锁路径上再进 `_connection`，可重入锁避免自锁死）。第五步，`self._shared_connection = None` 初始化。第六步，如果路径是 `":memory:"`，用 `check_same_thread=False` 建一个跨线程可用的共享连接，并把 `row_factory` 设为 `sqlite3.Row`（让查询结果支持按列名下标访问，`_decode` 依赖这一点）。第七步，调用 `self._initialize()` 完成父目录创建与建表。
- **异常/边界**：`path` 不是 `str`/`Path`/`None` 时抛 `TypeError("path must be a string, pathlib.Path, or None")`；`Path(path).expanduser()` 对含非法字符的路径在 Windows 上可能抛 `OSError`；建表阶段可能抛 `sqlite3.OperationalError`（目录不可写、文件被占用等）。环境变量为空串按未设置处理，不会得到空路径。
- **同文件关系**：调用了 `_initialize()`；间接经由 `_initialize` 调用 `_connection()` 与 `_connect()`。它被谁调用：本文件内没有其它函数调用它（构造函数的调用方是外部代码）。

### `_connect(self) -> sqlite3.Connection` （第 44 行）
- **作用**：统一的「取连接」入口，屏蔽内存模式与文件模式的差异。内存模式下 SQLite 的 `:memory:` 数据库是**连接私有**的：如果每次操作都新开连接，就会得到一个个互相看不见的空库，所以这里必须复用构造时创建的 `_shared_connection`。文件模式则相反，每次调用都新建一个短连接，用完由 `_connection` 关闭，这样既避免长期持有句柄，又天然规避了跨线程共享同一个 sqlite 连接的限制。它同时负责给新连接装上 `row_factory = sqlite3.Row`，这是 `_decode` 能用 `dict(row)` 和按列名访问的前提。
- **参数**：无（只用 `self`）。行为完全由 `self._shared_connection` 和 `self.path` 决定。
- **返回**：`sqlite3.Connection`。内存模式下返回构造时那个长期存活的共享连接；文件模式下返回一个全新的、`row_factory` 已设置为 `sqlite3.Row` 的连接对象。
- **内部流程**：先判断 `self._shared_connection is not None`，成立就直接返回它（内存模式分支，不做任何新建）。否则调用 `sqlite3.connect(self.path, timeout=10)` 建立文件连接（`timeout=10` 表示遇到写锁时最多等待 10 秒），设置 `connection.row_factory = sqlite3.Row`，返回该连接。注意它本身不加锁、不开事务，锁与事务由调用它的 `_connection` 负责。
- **异常/边界**：文件路径所在目录不存在或不可写时，`sqlite3.connect` 会抛 `sqlite3.OperationalError`（`_initialize` 会先建父目录来降低这种概率）；路径指向目录时也会报错。`close()` 之后再取连接：内存模式下 `_shared_connection` 已置 `None`，会尝试用 `self.path == ":memory:"` 去新建文件连接——这是一个需要注意的边界（关闭后继续使用内存仓储语义已失效）。无超时以外的重试逻辑。
- **同文件关系**：被 `_connection()` 调用；`_initialize`、`append`、`get`、`latest_id`、`get_range` 都是通过 `_connection()` 间接用到它。它自己不调用本文件任何函数。

### `_connection(self) -> Iterator[sqlite3.Connection]` （第 51 行）
- **作用**：一个用 `@contextmanager` 装饰的上下文管理器，是所有数据库操作的统一外壳，负责「加锁 → 拿连接 → 设置 PRAGMA → 开启事务 → 交给你用 → 提交/回滚 → 必要时关连接」这一整套流程。之所以需要它，是因为这些步骤如果在每个方法里重复写，极易漏掉加锁或漏掉关闭连接；集中在一处后，`append`/`get`/`latest_id`/`get_range` 都只需写业务 SQL。锁保证了同一进程内多个线程不会同时操作数据库，从根源上避免写冲突；`busy_timeout` 与 `foreign_keys` 两个 PRAGMA 则保证跨进程并发时也有等待窗口、并开启外键约束。它用 `with connection:` 包住 `yield`，使得业务代码里抛异常时事务自动回滚，正常退出时自动提交，从而让 `append` 的插入是原子的。
- **参数**：无显式参数（`self` 除外）。隐式输入是 `self._lock`、`self.path`、`self._shared_connection`。
- **返回**：一个生成器上下文管理器，`with` 语句里 `as` 到的是 `sqlite3.Connection`（内存模式下就是共享连接，文件模式下是本次新建的连接）。
- **内部流程**：第一步 `with self._lock:` 获取可重入锁，整个数据库交互期间持锁。第二步调用 `self._connect()` 得到连接。第三步进入 `try`，执行 `PRAGMA busy_timeout = 10000`（遇到锁最多等 10 秒）和 `PRAGMA foreign_keys = ON`（开启外键约束）。第四步 `with connection:` 开启隐式事务并 `yield connection`，把控制权交给调用方；调用方代码块正常结束时 sqlite 提交事务，抛异常时回滚。第五步 `finally` 中判断 `connection is not self._shared_connection`，成立则 `connection.close()`——即文件模式的临时连接一定关闭，内存模式的共享连接**绝不关闭**（留给后续调用继续用）。
- **异常/边界**：`PRAGMA` 或 SQL 执行失败会抛出 `sqlite3.OperationalError`/`sqlite3.IntegrityError` 等，异常会向上传播，但事务已被 `with connection:` 回滚；`finally` 保证临时连接无论如何都会关闭。锁是可重入的，所以同一个线程嵌套使用 `_connection` 不会死锁（尽管当前代码没有嵌套调用）。无超时重试逻辑，只依赖 `busy_timeout`。
- **同文件关系**：被 `_initialize()`、`append()`、`get()`、`latest_id()`、`get_range()` 调用；它自己调用 `_connect()`。是本文件所有 SQL 的唯一出入口。

### `close(self) -> None` （第 64 行）
- **作用**：显式释放内存数据库的共享连接，主要给测试或需要主动回收资源的场景使用。因为文件模式下的仓储「每次调用都是临时连接」（docstring 原话：file-backed repositories are per-call），根本没有常驻句柄需要释放，所以这个方法对文件模式基本是空操作，只对 `":memory:"` 模式有意义。调用它之后 `_shared_connection` 被置为 `None`，防止重复关闭（重复 `close()` 第二次会因为判断为 `None` 而安全跳过）。它也是让内存数据库有机会被垃圾回收、避免测试进程里连接泄漏的手段。
- **参数**：无（只用 `self`）。
- **返回**：无返回值（`None`）。通过副作用生效：`self._shared_connection` 变成 `None`。
- **内部流程**：先 `with self._lock:` 加锁（避免与正在写库的线程竞争）。然后判断 `self._shared_connection is not None`：成立则调用它的 `.close()`，随后把属性置为 `None`；不成立（文件模式，或已经被关过一次）则什么都不做。
- **异常/边界**：如果连接上有未提交的事务，`sqlite3.Connection.close()` 会丢弃它们（本文件所有写入都在 `with connection:` 里已提交，所以正常情况下没有悬空事务）。对文件模式调用是安全的空操作。关闭后再次调用 `append`/`get` 等方法：内存模式下 `_connect` 会走文件分支用 `":memory:"` 建一个新连接，得到的是一个**全新的空内存库**，此时 `update_logs` 表不存在，SQL 会抛 `sqlite3.OperationalError: no such table`——这是需要注意的边界。无特殊异常捕获。
- **同文件关系**：不调用本文件其它函数；也没有本文件内的其它函数调用它（由外部调用方在收尾时调用）。

### `_initialize(self) -> None` （第 71 行）
- **作用**：仓储的「首次使用初始化」，做两件事：为文件模式准备好数据库文件所在的父目录，然后执行 `CREATE TABLE IF NOT EXISTS` 建出 `update_logs` 表。它的存在保证了「构造 `UpdateLogRepository` 之后就能直接写入」，调用方不需要自己建目录、建表，也不需要处理「目录不存在导致 sqlite 打不开文件」的常见错误。表结构一次性定义了更新日志的全部 14 个字段，且除自增主键外全部声明为 `NOT NULL`，把「每条记录都必须完整」这一约束下沉到数据库层。`IF NOT EXISTS` 让重复构造（同一路径多次实例化）不会报错，也不会覆盖已有数据。
- **参数**：无（只用 `self.path` 判断是否需要建目录）。
- **返回**：无返回值（`None`）。副作用是磁盘上可能出现父目录和 `.db` 文件，且 `update_logs` 表在库中存在。
- **内部流程**：第一步，若 `self.path != ":memory:"`（内存模式不需要任何文件系统操作），取 `parent = Path(self.path).parent`；若父目录字符串不是 `""` 也不是 `"."`（即确实存在一个具名父目录，避免对相对文件名如 `log.db` 去创建当前目录），调用 `parent.mkdir(parents=True, exist_ok=True)` 递归创建目录且已存在时不报错。第二步，用 `with self._connection() as connection:` 进入统一的加锁/事务外壳。第三步执行一条多行 SQL：`CREATE TABLE IF NOT EXISTS update_logs (...)`，列定义依次为 `update_id INTEGER PRIMARY KEY AUTOINCREMENT`、`timestamp TEXT NOT NULL`、`system_name TEXT NOT NULL`、`executor TEXT NOT NULL`、`update_type TEXT NOT NULL`、`title TEXT NOT NULL`、`task_background TEXT NOT NULL`、`update_details TEXT NOT NULL`、`added_features TEXT NOT NULL`、`files_json TEXT NOT NULL`、`behavior_impact TEXT NOT NULL`、`validation TEXT NOT NULL`、`risks TEXT NOT NULL`、`follow_up TEXT NOT NULL`。
- **异常/边界**：`mkdir` 可能抛 `PermissionError`/`OSError`（路径非法、无权限）；建表可能抛 `sqlite3.OperationalError`（文件被占用、只读文件系统、路径是目录）。表已存在时因 `IF NOT EXISTS` 静默跳过，不会抛异常也不会改动既有表结构（即**不会自动迁移新增列**，加字段需手工处理）。无特殊异常捕获与重试。
- **同文件关系**：被 `__init__()` 调用（构造末尾无条件调用）；它自己调用 `_connection()`，并间接经由 `_connection` 调用 `_connect()`。

### `append(self, *, executor: str, update_type: str, title: str, task_background: str, update_details: str, added_features: str, files: Sequence[Mapping[str, str]], behavior_impact: str, validation: str, risks: str, follow_up: str, system_name: str | None = None, timestamp: str | None = None) -> dict[str, Any]` （第 98 行）
- **作用**：本文件最核心的写入口，把一次项目变更作为一行**原子插入** `update_logs`，并只返回一个「紧凑确认」（新 ID、时间戳、系统名、下一个 ID），而不是把整行数据回读出来。这样设计的目的就是模块 docstring 强调的：写入方（通常是 Agent）只需要知道「写成功了、ID 是多少」，历史正文留在数据库里，不进上下文窗口。方法在插入前做了严格且完整的输入校验：10 个必填文本字段都必须是非空字符串；`files` 必须是「映射序列」且至少一项，每一项都必须带非空 `path`/`action`/`description`，校验通过后统一 `strip()` 去空白；`system_name` 缺省时用 `platform.system()`（再缺省为 `"Unknown"`），`timestamp` 缺省时用当前 UTC 时间（秒精度 ISO 格式）。文件变更列表最终以 `json.dumps(..., ensure_ascii=False, sort_keys=True)` 序列化成 `files_json` 存库，保证中文可读且键顺序稳定、便于 diff 比较。整个方法的关键字参数全部是 keyword-only（`*` 之后），调用时必须写参数名，避免长参数列表被顺序写错。
- **参数**：
  - `executor`（`str`，必填，keyword-only）：执行这次变更的主体（例如某个 Agent 或人），必须非空（`strip()` 后非空）。
  - `update_type`（`str`，必填）：变更类型标签（如新增功能、修复、重构），必须非空。
  - `title`（`str`，必填）：变更标题，必须非空。
  - `task_background`（`str`，必填）：任务背景/起因，必须非空。
  - `update_details`（`str`，必填）：具体改了什么，必须非空。
  - `added_features`（`str`，必填）：新增的能力/功能说明，必须非空（若无新增也需填占位文本，不能传空串）。
  - `files`（`Sequence[Mapping[str, str]]`，必填）：文件变更列表，必须是序列（`str`/`bytes` 被显式拒绝），每个元素是含 `path`、`action`、`description` 三个键的映射，三个值都必须是非空字符串；列表不能为空。会按原顺序存入 `files_json`。
  - `behavior_impact`（`str`，必填）：对既有行为的影响，必须非空。
  - `validation`（`str`，必填）：如何验证的（测试/手测结果），必须非空。
  - `risks`（`str`，必填）：风险与遗留问题，必须非空。
  - `follow_up`（`str`，必填）：后续待办，必须非空。
  - `system_name`（`str | None`，默认 `None`）：写入者所在系统/主机名。为 `None` 或空串时取 `platform.system()`，再为空则用 `"Unknown"`；最终结果会 `strip()`。
  - `timestamp`（`str | None`，默认 `None`）：记录时间。为 `None` 时用 `datetime.now(UTC).isoformat(timespec="seconds")`（UTC、精确到秒）。docstring 明确说这两个可选参数**只为受控的数据迁移**而留，正常调用应省略，以便仓储自己捕获真实写入时间与宿主平台。
- **返回**：`dict[str, Any]`，固定四个键：`"update_id"`（`int`，本次插入得到的自增主键，来自 `cursor.lastrowid`）、`"timestamp"`（`str`，实际写入的时间戳）、`"system_name"`（`str`，实际写入的系统名）、`"next_update_id"`（`int`，等于 `update_id + 1`，方便调用方预告下一条记录的编号）。不返回插入行的其它字段，也不回读数据库。
- **内部流程**：第一步，把 10 个必填字段装进 `values` 字典。第二步遍历 `values.items()`，任何一个不是 `str` 或 `strip()` 后为空就抛 `ValueError(f"{name} must be a non-empty string")`（消息里带字段名，便于定位）。第三步校验 `files`：不是 `Sequence` 或属于 `(str, bytes)` 就抛 `TypeError("files must be a sequence of mappings")`。第四步遍历 `files`，每个元素不是 `Mapping` 抛 `TypeError("each file change must be a mapping")`；再用 `item.get("path"/"action"/"description")` 取值，三者不全是「非空字符串」就抛 `ValueError("each file change requires non-empty path, action and description")`；合法则把三个值 `strip()` 后以 `{"path":…, "action":…, "description":…}` 追加进 `normalized_files`。第五步，若 `normalized_files` 为空（`files` 是空序列）抛 `ValueError("files must contain at least one file change")`。第六步计算 `normalized_system`：`(system_name or platform.system() or "Unknown").strip()`，如果结果为空串再兜底为 `"Unknown"`。第七步计算 `normalized_timestamp`：`timestamp or datetime.now(UTC).isoformat(timespec="seconds")`，然后校验它是非空字符串，否则抛 `ValueError("timestamp must be a non-empty string")`。第八步 `with self._connection() as connection:` 进入加锁事务外壳，执行 `INSERT INTO update_logs (...) VALUES (?, ?, …)` 共 13 个占位符，按顺序传入归一化时间戳、归一化系统名，以及 `values` 中 10 个字段的 `strip()` 结果（注意 SQL 里字段顺序是 `timestamp, system_name, executor, update_type, title, task_background, update_details, added_features, files_json, behavior_impact, validation, risks, follow_up`，与占位符元组严格对应），`files_json` 由 `json.dumps(normalized_files, ensure_ascii=False, sort_keys=True)` 生成。第九步 `update_id = int(cursor.lastrowid)` 取出新主键（显式 `int()` 转换，因为某些 sqlite 驱动返回可能是其它数值类型）。第十步退出 `with` 块（事务提交）后组装并返回那四个键的字典。
- **异常/边界**：`ValueError`（10 个必填字段任一为空、文件项三字段缺失/为空、`files` 为空序列、`timestamp` 非非空字符串）；`TypeError`（`files` 不是序列、是 `str`/`bytes`，或某个文件项不是映射）；数据库层可能抛 `sqlite3.IntegrityError`（例如字段违反 `NOT NULL`）或 `sqlite3.OperationalError`（锁等待超过 `busy_timeout`、磁盘满、表不存在）；`json.dumps` 理论上可能对不可序列化对象抛 `TypeError`，但此处 `normalized_files` 只含字符串，实际不会触发。`system_name` 为空有 `"Unknown"` 兜底，不会因空串报错；`files` 中映射含额外键会被忽略（只取三个已知键），键值前后的空白会被去除。任何异常发生时，`_connection` 的 `with connection:` 会回滚事务，因此**不会留下半条记录**。
- **同文件关系**：调用 `_connection()`（并经由它调用 `_connect()`）；不调用其它业务方法。本文件内没有其它函数调用它（由外部调用方在记录变更时调用）。

### `get(self, update_id: int) -> dict[str, Any] | None` （第 200 行）
- **作用**：按主键精确读取**一条**更新日志，供审计与测试使用，docstring 明确其目的是「不做批量加载」。这是「历史留在库里、需要时才取单条」这一设计的最小读取单元：Agent 拿到 `append` 返回的 `update_id` 后，若确实需要回看这一条内容，就可以用本方法只取这一条。它先做参数校验（拒绝 `bool`、非 `int`、小于 1 的值），再用参数化 SQL 查询，最后交给 `_decode` 把 `files_json` 文本还原成 Python 列表，使调用方拿到的是可直接使用的结构而不是原始 JSON 字符串。查不到时返回 `None` 而不是抛异常，调用方可以用 `is None` 判断「这条不存在」。
- **参数**：
  - `update_id`（`int`，必填）：要读取的记录主键，必须是正整数（`>= 1`）。显式排除 `bool`（因为 Python 里 `True` 是 `int` 的子类，`True` 会被误当成 1）。
- **返回**：`dict[str, Any] | None`。找到时返回该行的字典，键为全部 14 个字段名，但其中 `files_json` 已被移除、替换为 `files`（`list[dict[str, str]]`，由 `json.loads` 还原）；找不到时返回 `None`。
- **内部流程**：第一步参数校验：`isinstance(update_id, bool)`、`not isinstance(update_id, int)`、`update_id < 1` 三者任一成立就抛 `ValueError("update_id must be a positive integer")`。第二步 `with self._connection() as connection:` 进入统一外壳，执行 `SELECT * FROM update_logs WHERE update_id = ?` 并传入 `(update_id,)`，用 `.fetchone()` 取第一行（无结果时得到 `None`）。第三步退出上下文（读操作无副作用），返回 `self._decode(row) if row is not None else None`。
- **异常/边界**：非法 `update_id`（`bool`、非整数、`0`、负数）抛 `ValueError`；`update_id` 合法但记录不存在返回 `None`（不抛异常）；数据库异常（表不存在、锁超时）会抛 `sqlite3.OperationalError`；若库中 `files_json` 不是合法 JSON，`_decode` 内的 `json.loads` 会抛 `json.JSONDecodeError`（当前写入路径保证合法）。无其它特殊处理。
- **同文件关系**：调用 `_connection()` 和静态方法 `_decode()`；本文件内没有其它函数调用它（由外部调用方做审计/测试时调用）。

### `latest_id(self) -> int` （第 210 行）
- **作用**：返回当前表里最大的 `update_id`，也就是「目前写到第几条了」。它给调用方提供低成本的水位查询：想知道有没有新日志、想从某个位置继续读、或者想在写入前知道下一个编号，都可以先问它。因为它只执行 `SELECT MAX(update_id)` 而不取任何正文，所以即使表里积累了大量历史记录也不会把内容拉进内存或上下文，符合整个模块「只取需要的最小信息」的取向。方法没有参数校验，因为它不需要任何外部输入。
- **参数**：无（只用 `self`）。
- **返回**：`int`。表为空（还没有任何日志）时返回 `0`（因为 `MAX` 在空表上返回 `NULL`，代码用 `row["value"] or 0` 兜底）；否则返回最大的 `update_id`。返回前用 `int(...)` 显式转换。
- **内部流程**：第一步 `with self._connection() as connection:` 进入统一外壳。第二步执行 `SELECT MAX(update_id) AS value FROM update_logs`，`.fetchone()` 取结果行。第三步退出上下文后返回 `int(row["value"] or 0)`——`row` 是 `sqlite3.Row`，靠 `_connect` 设置的 `row_factory` 才能用列名 `"value"` 下标访问；`or 0` 同时处理了 `None`（空表）和假值情况。
- **异常/边界**：表不存在或数据库锁超时会抛 `sqlite3.OperationalError`；空表不抛异常而返回 `0`。无参数校验（无参数）。无特殊处理。
- **同文件关系**：调用 `_connection()`（并经由它调用 `_connect()`）；不调用其它业务方法。本文件内没有其它函数调用它（由外部调用方查询水位时调用）。

### `get_range(self, start_id: int, end_id: int) -> list[dict[str, Any]]` （第 215 行）
- **作用**：一次性取出一个**闭区间** `[start_id, end_id]` 内的所有更新日志，并按 `update_id` 升序返回。docstring 记录了它的存在理由：旧调用方式是「每个 ID 打开一次连接」，导致全量审计越来越慢，模型反复请求同一区间时甚至看起来像卡住了；本方法用**一次连接、一条 SQL** 完成整段读取，把 N 次往返压成 1 次，显著提升批量审计的性能。返回的每条记录都经过 `_decode` 处理，`files_json` 已还原成 `files` 列表。区间为空（该范围内没有记录）时返回空列表而不是 `None`，调用方可以统一用迭代处理。
- **参数**：
  - `start_id`（`int`，必填）：区间起始主键（含），必须是正整数（`>= 1`）。
  - `end_id`（`int`，必填）：区间结束主键（含），必须是整数且 `>= start_id`。两个参数都不接受 `bool`。
- **返回**：`list[dict[str, Any]]`。按 `update_id` 升序排列的解码后记录列表；若区间内没有任何记录则返回 `[]`（空列表）。每条字典的结构与 `get()` 返回的一致（含 `files` 键、不含 `files_json`）。
- **内部流程**：第一步做联合参数校验：`isinstance(start_id, bool)` 或 `not isinstance(start_id, int)` 或 `start_id < 1`，或 `isinstance(end_id, bool)` 或 `not isinstance(end_id, int)` 或 `end_id < start_id`，任一成立就抛 `ValueError("start_id and end_id must be positive integers")`（注意该消息未单独区分 `end_id < start_id` 这种情况）。第二步 `with self._connection() as connection:` 进入统一外壳。第三步执行 `SELECT * FROM update_logs WHERE update_id BETWEEN ? AND ? ORDER BY update_id ASC`，参数 `(start_id, end_id)`，用 `.fetchall()` 一次取回全部匹配行（`BETWEEN` 两端都是闭区间，与参数语义一致；`ORDER BY update_id ASC` 保证输出顺序稳定）。第四步退出上下文后，用列表推导 `[self._decode(row) for row in rows]` 逐行解码并返回。
- **异常/边界**：`start_id`/`end_id` 为 `bool`、非整数、`start_id < 1` 或 `end_id < start_id` 时抛 `ValueError`；区间内无记录返回空列表；区间很大时会一次性把所有匹配行载入内存（这是该方法的固有边界，也是它相比逐条读取的代价）；数据库层异常（表不存在、锁超时）抛 `sqlite3.OperationalError`；若某行 `files_json` 不是合法 JSON，`_decode` 会抛 `json.JSONDecodeError`。无特殊处理。
- **同文件关系**：调用 `_connection()` 和静态方法 `_decode()`；本文件内没有其它函数调用它（由外部调用方做区间审计时调用）。

### `_decode(row: sqlite3.Row) -> dict[str, Any]` （第 239 行，`@staticmethod`）
- **作用**：把数据库原始行对象转换成调用方友好的字典，是读路径上的统一收尾步骤。它做两件事：把 `sqlite3.Row` 转成普通 `dict`（因为 `Row` 不支持 `.pop()` 等字典操作），以及把存储用的 `files_json` 字符串反序列化回 Python 对象并改名为 `files`，让调用方直接拿到「文件变更列表」这一结构，而不必自己 `json.loads`。把它写成 `@staticmethod` 是因为它完全不依赖实例状态（不读 `self.path`、不用锁），既方便 `get`/`get_range` 直接以 `self._decode(row)` 调用，也便于单独测试。
- **参数**：
  - `row`（`sqlite3.Row`，必填）：由 `SELECT * FROM update_logs ...` 返回的一行，必须包含 `files_json` 列（当前表结构一定有），并且该列内容是 `json.dumps` 生成的合法 JSON 文本（当前写入路径保证如此）。
- **返回**：`dict[str, Any]`。键为 `update_id`、`timestamp`、`system_name`、`executor`、`update_type`、`title`、`task_background`、`update_details`、`added_features`、`behavior_impact`、`validation`、`risks`、`follow_up`，以及 `files`；其中 `files_json` 键已被 `pop` 移除，不再出现。`files` 的实际类型取决于库中 JSON 内容，正常情况下是 `list[dict[str, str]]`。
- **内部流程**：第一步 `item = dict(row)`，把 `sqlite3.Row` 转成普通字典（依赖连接设置了 `row_factory = sqlite3.Row`，否则这里会失败或得到元组语义）。第二步 `item["files"] = json.loads(item.pop("files_json"))`——`pop` 在取走 `files_json` 字符串的同时把它从字典里删掉，`json.loads` 解析后赋给新键 `files`。第三步返回 `item`。
- **异常/边界**：`row` 缺少 `files_json` 键时 `item.pop("files_json")` 会抛 `KeyError`；该列内容不是合法 JSON 时 `json.loads` 会抛 `json.JSONDecodeError`；若 `row` 不是 `sqlite3.Row` 而是普通元组，`dict(row)` 会抛 `TypeError`/`ValueError`。无任何 try/except 兜底，完全依赖上游查询与写入的一致性。不处理空值兜底（`files_json` 列是 `NOT NULL`，不会为 `NULL`）。
- **同文件关系**：被 `get()` 与 `get_range()` 调用（均以 `self._decode(row)` 形式调用，静态方法经实例访问也合法）；它自己不调用本文件任何函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `UpdateLogRepository` | 线程安全的 SQLite 更新日志仓储，负责建表、连接管理、写入与按 ID 读取。 |
| `__init__` | 解析数据库路径（环境变量/默认值/`:memory:`）、建锁与共享连接，并立即初始化表结构。 |
| `_connect` | 内存模式复用共享连接、文件模式每次新建带 `Row` 工厂的连接。 |
| `_connection` | 上下文管理器：加锁、设 PRAGMA、开事务、按需关闭临时连接，是所有 SQL 的统一外壳。 |
| `close` | 关闭内存模式的共享连接（文件模式为空操作），并置空引用防止重复关闭。 |
| `_initialize` | 为文件模式创建父目录并 `CREATE TABLE IF NOT EXISTS update_logs` 建出 14 列日志表。 |
| `append` | 校验全部必填字段与文件变更项后原子插入一行，只返回新 ID、时间戳、系统名和下一个 ID。 |
| `get` | 按正整数主键精确读取一条日志并解码 `files`，不存在时返回 `None`。 |
| `latest_id` | 用 `SELECT MAX(update_id)` 返回当前最大日志编号，空表返回 `0`。 |
| `get_range` | 用一次查询取出闭区间 `[start_id, end_id]` 内的全部日志并按 ID 升序返回，避免逐条开连接。 |
| `_decode` | 静态工具方法：把 `sqlite3.Row` 转成字典并把 `files_json` 反序列化为 `files`。 |
