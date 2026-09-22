# memory/storage/document.py

## 一、这个文件是干什么的

这个文件是四层记忆系统的「文档持久化层」实现，职责只有一件事：把内存里的记忆对象 `MemoryItem` 安全、完整地落到 SQLite 数据库里，并能按 id 或按类型把它们原样读回来。它定义了两种东西：一个是抽象基类 `BaseDocumentStore`，用 `ABC` + `@abstractmethod` 规定了任何文档存储后端都必须提供的五个动作（写入/更新 `upsert`、读取 `get`、删除 `delete`、枚举 `list`、关闭 `close`）；另一个是具体实现 `SQLiteDocumentStore`，用一张 `memories` 表承载全部字段。

文件顶部导入了 `json`、`sqlite3`、`threading`、`abc`、`contextlib`、`pathlib` 和 `typing`，并从同包上一级的 `..base` 取来了 `MemoryItem`、`MemoryType`、`_json_restore` 三个符号。也就是说，本文件不定义记忆对象的数据结构，只负责它的「序列化 ↔ 反序列化」和 SQL 读写；`MemoryItem` 的字段语义、过期判断（`is_expired`）、`to_dict()` 的键名约定都由 `..base` 决定。

设计上有三个明显特征。第一，`memories` 表把结构化字段（id、content、memory_type、importance、各种时间戳）拆成独立列，而把变长/嵌套的字段（metadata、embedding、payload、relations）用 `json.dumps(..., ensure_ascii=False)` 存成 TEXT，读取时再 `json.loads` 还原，这样既能用 SQL 按类型和时间排序筛选，又能保留任意结构。第二，线程安全靠一把 `threading.RLock` 加「每次操作开一个连接作用域」的 `_connection_scope()` 上下文管理器实现，并且对 `:memory:` 数据库做特殊处理——内存库是「每连接一份」，连接一关数据就没了，所以必须把连接钉住（pinned）全程复用，而文件库则每次操作临时连、用完就关。第三，写入统一走 `_upsert_on()` 里的 `INSERT ... ON CONFLICT(id) DO UPDATE SET`，即「有则全字段覆盖、无则插入」，并额外提供 `upsert_many()` 把一次投影重建的多行放进同一个事务提交。

在项目运行中，它一般被记忆系统的上层（比如记忆仓库/投影重建流程）实例化：传一个数据库文件路径得到一个持久化的 `SQLiteDocumentStore`，或者不传路径（默认 `":memory:"`）得到一个纯内存的临时库用于测试与临时会话；之后所有记忆的增删查改都经过这里。文件末尾的 `__all__ = ["BaseDocumentStore", "SQLiteDocumentStore"]` 明确对外只暴露这两个名字。

## 二、函数与类逐条详解

### `class BaseDocumentStore(ABC)` （第 17 行）
- **作用**：这是文档存储的抽象基类，本身不存任何数据，只用来「定契约」。它把记忆持久化后端必须具备的能力固化成五个方法：写入或更新一条记忆、按 id 取一条记忆、按 id 删一条记忆、按条件列出记忆、关闭后端。任何新的存储后端（例如将来换成别的数据库或远端服务）只要继承它并实现那几个抽象方法，就能被上层代码用同一套接口调用，实现存储实现的替换而不影响业务逻辑。它继承自 `abc.ABC`，配合 `@abstractmethod` 让 Python 在实例化时自动检查子类是否补齐了所有抽象方法。之所以需要它，是因为记忆系统可能同时存在多种后端（内存、SQLite、将来的其他实现），上层只应该依赖接口而不是具体类。本文件中的 `SQLiteDocumentStore` 就是它目前唯一的实现。
- **参数**：类本身不接收参数；它的抽象方法签名里带参数，含义见各方法条目。继承 `ABC` 后，若子类没有实现全部抽象方法，`BaseDocumentStore()` 或子类实例化时会抛 `TypeError`。
- **返回**：类定义，无返回值；实例化得到具体后端对象。
- **内部流程**：类体只包含四个 `@abstractmethod` 声明和一个带默认实现的 `close()`。抽象方法体是省略号 `...`，即「没有实现」，纯粹用于声明签名；`close()` 则给了 `pass` 作为默认实现，意味着子类可以什么都不做。
- **异常/边界**：直接实例化 `BaseDocumentStore()` 会因存在抽象方法而抛 `TypeError`；若某个子类漏实现了抽象方法，实例化同样抛 `TypeError`。直接以 `BaseDocumentStore.upsert(obj, item)` 这种形式调用未实现的抽象方法会返回 `None` 而不报错，但这不是正常用法。
- **同文件关系**：被 `SQLiteDocumentStore` 继承；它声明的 `upsert`、`get`、`delete`、`list`、`close` 五个签名被 `SQLiteDocumentStore` 的对应同名方法实现（覆盖）。它不调用本文件任何其他函数。

### `upsert(self, item: MemoryItem) -> None` （第 19 行）
- **作用**：抽象声明「把一条记忆写入后端」这一动作。语义上是幂等的 upsert：如果该 id 的记忆已存在就更新，不存在就插入，调用方不需要先查再决定插入还是更新。上层在做记忆写入、更新、投影重建时都会用到这个动作。它被 `@abstractmethod` 装饰，因此只是契约，不含任何实现逻辑。具体实现由子类提供，例如 `SQLiteDocumentStore.upsert`。
- **参数**：`self` 为后端实例；`item` 类型为 `MemoryItem`，是要写入的那条记忆对象，必须携带 `id`、`content`、`memory_type` 等字段，`id` 是判定「插入还是更新」的主键。
- **返回**：声明为 `None`，即不返回任何值，只通过副作用（写库）产生效果。
- **内部流程**：无内部流程，方法体是 `...`。实际流程见实现类 `SQLiteDocumentStore.upsert` 与 `SQLiteDocumentStore._upsert_on`。
- **异常/边界**：抽象方法本身不做校验；实现类约定的约束是「传入非 `MemoryItem` 会抛 `TypeError`」。
- **同文件关系**：被 `SQLiteDocumentStore.upsert` 实现；无内部调用。

### `get(self, item_id: str) -> MemoryItem | None` （第 22 行）
- **作用**：抽象声明「按唯一 id 精确取回一条记忆」。上层在需要根据 id 回溯某条具体记忆（比如按引用关系找被引用的记忆、或去重时确认是否已存在）时会调用它。返回类型标注为 `MemoryItem | None`，明确了「找不到时返回 None 而不是抛异常」的契约，调用方必须处理 None。它同样只是契约声明，具体读取与反序列化由子类完成。
- **参数**：`self` 为后端实例；`item_id` 类型为 `str`，是记忆的主键字符串，必须与写入时 `MemoryItem.id` 一致，否则查不到。
- **返回**：命中时返回一条 `MemoryItem`；未命中返回 `None`。
- **内部流程**：无内部流程，方法体是 `...`。
- **异常/边界**：无特殊处理，抽象层不校验 `item_id` 是否为空字符串。
- **同文件关系**：被 `SQLiteDocumentStore.get` 实现；无内部调用。

### `delete(self, item_id: str) -> bool` （第 25 行）
- **作用**：抽象声明「按 id 删除一条记忆」，并用 `bool` 返回值区分「确实删掉了」和「本来就不存在」。这个区分对上层很重要：调用方可以据此判断是执行了有效删除，还是重复删除/删了不存在的 id，从而决定是否记录日志或调整统计。它只是契约，不含实现。
- **参数**：`self` 为后端实例；`item_id` 类型为 `str`，为目标记忆的主键。
- **返回**：`True` 表示存在该 id 并被删除；`False` 表示不存在该 id，什么都没删。
- **内部流程**：无内部流程，方法体是 `...`。
- **异常/边界**：无特殊处理。
- **同文件关系**：被 `SQLiteDocumentStore.delete` 实现；无内部调用。

### `list(self, *, memory_type: MemoryType | str | None = None, include_expired: bool = False) -> list[MemoryItem]` （第 28 行）
- **作用**：抽象声明「按条件枚举记忆」。两个参数都是关键字参数（签名里的 `*` 强制关键字传参），分别控制「只看某一类记忆」和「是否把已过期的记忆也返回」。这是上层做记忆巡检、按类型聚合、导出/统计时的主要入口。返回 `list[MemoryItem]`，即一次性把结果全部取出成列表，而不是迭代器，调用方拿到的是稳定快照。它只是契约声明，筛选与排序规则由实现类决定。
- **参数**：`self` 为后端实例；`memory_type` 类型为 `MemoryType | str | None`，默认 `None`，`None` 表示不按类型过滤、返回所有类型，传 `MemoryType` 枚举或它的字符串值则只返回该类型；`include_expired` 类型为 `bool`，默认 `False`，为 `False` 时会剔除 `is_expired` 为真的记忆，为 `True` 时全部返回。
- **返回**：`list[MemoryItem]`，可能为空列表。
- **内部流程**：无内部流程，方法体是 `...`。
- **异常/边界**：抽象层不处理；实现类约定「非法类型字符串会抛 `ValueError`」。
- **同文件关系**：被 `SQLiteDocumentStore.list` 实现；无内部调用。

### `close(self) -> None` （第 30 行）
- **作用**：这是基类里唯一带默认实现的方法，用来释放后端持有的资源（数据库连接、文件句柄等）。把它放在基类并给一个 `pass` 的空实现，是为了让「不需要释放资源的后端」不必被迫写一个空方法，同时让上层可以无脑调用 `store.close()` 而不必判断后端类型。子类如有真实资源需要释放，应覆盖它。
- **参数**：`self` 为后端实例；无其他参数。
- **返回**：`None`。
- **内部流程**：方法体只有 `pass`，什么也不做，直接返回。
- **异常/边界**：无特殊处理，不抛异常，重复调用也安全。
- **同文件关系**：被 `SQLiteDocumentStore.close` 覆盖（`SQLiteDocumentStore` 实现了真实关闭逻辑）；无内部调用。

### `class SQLiteDocumentStore(BaseDocumentStore)` （第 34 行）
- **作用**：这是本文件的核心实现，把记忆持久化到一张名为 `memories` 的 SQLite 表。它同时支持两种运行形态：传文件路径时是磁盘持久化（进程重启后数据仍在），传 `":memory:"`（默认值）时是进程内内存库（连接一关数据即消失，适合测试和临时会话）。它对结构化字段用独立列存，对嵌套字段用 JSON 文本存，从而兼顾 SQL 查询能力与结构灵活性。线程安全方面它内部持有一把 `threading.RLock`，所有读写都在锁内通过 `_connection_scope()` 完成。类文档字符串「SQLite document persistence with safe JSON serialization」概括了它的两个卖点：SQLite 落盘 + 安全的 JSON 序列化（统一 `ensure_ascii=False`、写入前 `json.dumps`、读取时 `json.loads` 再交给 `_json_restore` 还原特殊结构）。
- **参数**：继承基类，实例化参数见 `__init__`（`path`，默认 `":memory:"`）。
- **返回**：实例化得到一个可直接使用的文档存储对象。
- **内部流程**：`__init__` 先规范化路径、必要时创建父目录、准备锁与连接占位，再调用 `_initialize()` 建表建索引；此后每个公开方法都通过 `_connection_scope()` 取得连接并执行 SQL。
- **异常/边界**：路径所在目录不可创建、文件不可写、SQLite 文件损坏等情况下，`mkdir` 或 `sqlite3.connect` 会抛 `OSError`/`sqlite3.Error`；内存模式下连接被 `close()` 置空后再调用方法会因 `self.path == ":memory:"` 而重新 `sqlite3.connect(":memory:")` 得到一个空的临时库（数据已丢失），这是一个需要注意的边界。
- **同文件关系**：继承 `BaseDocumentStore` 并实现其五个抽象/可覆盖方法；内部大量互相调用，关系见各方法条目。

### `__init__(self, path: str | Path = ":memory:") -> None` （第 37 行）
- **作用**：构造存储实例，完成「路径规范化 → 目录准备 → 并发原语准备 → 连接准备 → 建表」这一整套初始化。它最关键的判断是「是不是内存库」：因为 SQLite 的 `:memory:` 数据库是按连接隔离的，一旦连接被关掉数据就永久丢失，所以对内存库必须立刻建立连接并把它长期钉住（赋给 `self._connection`）；而文件库则故意不预先建连接，留到每次操作时临时连接、用完即关，避免长连接泄漏与跨线程问题。另外它会把用户路径里的 `~` 展开成真实家目录，并在文件不存在时提前创建父目录，避免 `sqlite3.connect` 因目录缺失而报错。
- **参数**：`self` 为实例本身；`path` 类型为 `str | Path`，默认 `":memory:"`，既可以是内存库标记字符串，也可以是任意文件路径（相对路径按进程当前工作目录解析，`~` 会被展开）。
- **返回**：`None`（构造函数返回实例本身由 Python 处理）。
- **内部流程**：第一步，`self.path` 赋值——若 `str(path) == ":memory:"` 则原样保留，否则用 `Path(path).expanduser()` 展开后再转 `str`；第二步，若 `self.path != ":memory:"` 则 `Path(self.path).parent.mkdir(parents=True, exist_ok=True)` 递归创建父目录（已存在不报错）；第三步，创建 `self._lock = threading.RLock()`（可重入锁，允许同一线程嵌套进入作用域）；第四步，`self._connection = None` 作为占位；第五步，若 `self.path == ":memory:"`，执行 `sqlite3.connect(self.path, check_same_thread=False)` 并把结果同时赋给 `self._connection`，设置 `row_factory = sqlite3.Row`（`check_same_thread=False` 是为了让钉住的连接能被多个线程使用，真正的互斥由 `_lock` 保证）；第六步，调用 `self._initialize()` 建表建索引。
- **异常/边界**：`Path(path).expanduser()` 对非法路径类型会抛 `TypeError`；`mkdir` 失败抛 `OSError`/`PermissionError`；`sqlite3.connect` 失败抛 `sqlite3.Error`。传入空字符串 `""` 时 `str(path) != ":memory:"`，会走文件分支，`Path("").parent` 为当前目录，最终连接一个名为空字符串的库（等价于临时文件库），这是需要注意的边界。
- **同文件关系**：调用本文件的 `_initialize()`（间接经 `_connection_scope()` 与 `_connect()`）；被同文件的 `connection`、`_connect`、`close` 等方法依赖其设置的 `self.path`、`self._lock`、`self._connection`。

### `connection` (property) -> `sqlite3.Connection | None` （第 49 行）
- **作用**：这是一个只读属性（`@property`），把内部字段 `self._connection` 暴露给外部查看，语义是「被钉住的那个连接」。对 `:memory:` 存储来说它返回真实连接（因为内存库必须靠这条连接存活），对文件存储来说它返回 `None`（因为文件库不保留长连接，每次操作都是临时的）。外部代码（例如需要复用同一连接执行自定义 SQL 的调试或迁移代码）可以通过它判断当前是内存模式还是文件模式，并在内存模式下拿到可用的连接对象。
- **参数**：`self` 为实例；属性形式调用，无显式参数。
- **返回**：`sqlite3.Connection | None`——内存模式下返回构造时创建的连接；文件模式下（以及内存模式已被 `close()` 关闭后）返回 `None`。
- **内部流程**：只做一次属性读取并返回 `self._connection`，没有额外逻辑，文档字符串直接说明了「pinned `:memory:` connection, or `None` for file-backed stores」。
- **异常/边界**：无特殊处理，不会抛异常；返回的连接可能已被 `close()` 关闭（关闭后 `_connection` 被置为 `None`，所以实际上不会返回已关闭对象，除非外部自行关闭了它）。
- **同文件关系**：读取 `__init__` 设置的 `self._connection`；被 `_connect()`（作为「是否需要新建连接」的判断依据）与 `_connection_scope()`（判断作用域结束时是否要关闭连接）间接使用。

### `_connect(self) -> sqlite3.Connection` （第 54 行）
- **作用**：私有辅助方法，负责「拿到一个可用连接」。它把「内存库复用钉住连接、文件库新建临时连接」这个分支逻辑集中在一处，避免每个读写方法各写一遍。之所以需要它，是因为内存库和文件库的连接生命周期完全不同：内存库每次新建连接都会得到一张空表，绝不能新建；文件库若长期持有一个连接，则会带来跨线程与文件锁的麻烦，所以每次操作新建、用完关闭更稳。
- **参数**：`self` 为实例；无其他参数。
- **返回**：`sqlite3.Connection`——若 `self._connection` 不为 `None`（即内存模式且未关闭）直接返回它；否则用 `sqlite3.connect(self.path)` 新建连接、设置 `row_factory = sqlite3.Row` 后返回（调用方负责关闭）。
- **内部流程**：第一步判断 `self._connection is not None`，成立就立即 `return self._connection`（不新建、不设置 row_factory，因为构造时已设置）；否则执行 `sqlite3.connect(self.path)` 新建连接，设置 `connection.row_factory = sqlite3.Row` 让查询结果支持按列名取值（`_decode` 依赖 `row["id"]` 这种下标方式），然后返回该连接。
- **异常/边界**：`sqlite3.connect` 在路径不可写、目录不存在、文件损坏等情况下抛 `sqlite3.Error`；对文件库新建的连接未设置 `check_same_thread=False`，若被跨线程使用可能抛 `sqlite3.ProgrammingError`，但实际使用都被 `_lock` 串行化且限定在同一线程的作用域内。注意它不负责关闭连接，关闭由 `_connection_scope()` 的 `finally` 负责。
- **同文件关系**：被 `_connection_scope()` 调用；它读取 `__init__` 设置的 `self._connection` 与 `self.path`，与 `connection` 属性共享同一字段。

### `_connection_scope(self) -> Iterator[sqlite3.Connection]` （第 62 行）
- **作用**：这是整个类的「并发与事务骨架」，用 `@contextmanager` 把「加锁 → 取连接 → 开事务 → 交出连接 → 自动提交/回滚 → 必要时关连接 → 解锁」封装成一个 `with` 块。所有公开读写方法都通过它拿连接，因此它们天然是串行的（同一时刻只有一个线程在库上操作），并且写操作在块正常结束时自动提交、抛异常时自动回滚。之所以需要它，是因为 SQLite 的连接不能随便共享，而每次操作都手写 try/finally 关闭连接既啰嗦又容易漏。
- **参数**：`self` 为实例；无其他参数。
- **返回**：作为生成器上下文管理器，`yield` 出 `sqlite3.Connection`；`with ... as connection:` 拿到的就是这个连接。函数本身标注返回 `Iterator[sqlite3.Connection]`。
- **内部流程**：第一步 `with self._lock:` 获取可重入锁，保证临界区互斥；第二步 `connection = self._connect()` 取得连接；第三步进入 `try`，执行 `with connection:`——SQLite 连接的上下文协议会在块正常退出时 `commit()`、异常退出时 `rollback()`；在块内 `yield connection` 把控制权交给调用方的 `with` 体；第四步 `finally` 中判断 `if connection is not self._connection:`，即「这条连接不是被钉住的内存连接」时调用 `connection.close()`，避免文件库连接泄漏；钉住的内存连接则保持打开。
- **异常/边界**：调用方 `with` 体内抛出的任何异常都会穿过 `yield` 传播回这里，触发 `with connection:` 的回滚，然后由 `finally` 关闭临时连接，最后 `with self._lock:` 释放锁，异常继续向上抛。`yield` 之后若生成器被外部提前关闭（例如生成器未耗尽就丢弃），`contextmanager` 机制会抛 `RuntimeError`。锁是可重入的，因此同一线程内嵌套使用不会死锁。
- **同文件关系**：调用 `_connect()`；被 `_initialize()`、`upsert()`、`upsert_many()`、`get()`、`delete()`、`list()` 共六个方法使用；依赖 `__init__` 创建的 `self._lock` 与 `self._connection`。

### `_initialize(self) -> None` （第 72 行）
- **作用**：负责建表与建索引的幂等初始化，在构造函数的最后一步被调用一次。它用 `CREATE TABLE IF NOT EXISTS` 和 `CREATE INDEX IF NOT EXISTS`，因此对已存在的数据库文件重复执行也不会报错、不会丢数据，这让「同一个库文件被多个进程/多次启动打开」变得安全。表结构把 `MemoryItem` 的所有字段一一映射成列，并对常用查询维度（`memory_type` 用于按类型过滤、`timestamp` 用于排序）建索引以加速。
- **参数**：`self` 为实例；无其他参数。
- **返回**：`None`。
- **内部流程**：通过 `with self._connection_scope() as connection:` 取得连接并开事务，然后依次执行三条语句：第一条 `CREATE TABLE IF NOT EXISTS memories (...)` 建表，列定义依次为 `id TEXT PRIMARY KEY`（主键）、`content TEXT NOT NULL`、`memory_type TEXT NOT NULL`、`metadata TEXT NOT NULL`、`importance REAL NOT NULL`、`created_at TEXT NOT NULL`、`updated_at TEXT NOT NULL`、`expires_at TEXT`（可空）、`timestamp TEXT`（可空）、`embedding TEXT`（可空）、`payload TEXT`（可空）、`modality TEXT`（可空）、`relations TEXT NOT NULL`；第二条 `CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)`；第三条 `CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)`。作用域正常退出时提交。
- **异常/边界**：若磁盘上已有一个列结构不同的同名 `memories` 表，`IF NOT EXISTS` 不会改表，后续 `_upsert_on` 的 INSERT 可能因缺列而抛 `sqlite3.OperationalError`；数据库文件只读或损坏时抛 `sqlite3.Error`，异常会向上传到构造函数。
- **同文件关系**：调用 `_connection_scope()`；只被 `__init__()` 调用一次；它建出的表结构被 `_upsert_on()`、`get()`、`delete()`、`list()`、`_decode()` 共同依赖。

### `_upsert_on(connection: sqlite3.Connection, item: MemoryItem) -> None` （第 97 行）
- **作用**：这是真正执行「写入或更新一行」的地方，被 `upsert()` 和 `upsert_many()` 复用。它写成 `@staticmethod`，因为它只依赖传入的连接和 item，不需要 `self` 的状态，这样既能被批量写入方法高效调用，也避免了在批量循环里反复绑定实例。核心是 `INSERT ... ON CONFLICT(id) DO UPDATE SET ...`：当 id 冲突时把除主键外的所有列覆盖为新的值（注意 `id` 本身不在 SET 列表里，因为主键不会变），从而一条语句完成「存在即更新、不存在即插入」。它还负责把 Python 对象转成可入库的值：枚举取 `.value`、嵌套结构 `json.dumps(..., ensure_ascii=False)` 保留中文可读性。
- **参数**：`connection` 类型为 `sqlite3.Connection`，是调用方从 `_connection_scope()` 取到的已开启事务的连接，本方法不负责提交或关闭它；`item` 类型为 `MemoryItem`，是要写入的记忆对象，必须具有 `id`、`content`、`memory_type`（含 `.value`）、`importance`、`embedding`、`modality` 属性，并能通过 `to_dict()` 提供 `metadata`、`created_at`、`updated_at`、`expires_at`、`timestamp`、`payload`、`relations` 等键。
- **返回**：`None`，通过执行 SQL 产生副作用。
- **内部流程**：第一步类型校验，`if not isinstance(item, MemoryItem): raise TypeError("item must be a MemoryItem")`；第二步 `data = item.to_dict()` 拿到用于落库的字典；第三步 `connection.execute(...)` 执行带 13 个占位符的 INSERT/UPSERT 语句，参数元组按列顺序构造为：`item.id`、`item.content`、`item.memory_type.value`、`json.dumps(data["metadata"], ensure_ascii=False)`、`item.importance`、`data["created_at"]`、`data["updated_at"]`、`data["expires_at"]`、`data["timestamp"]`、`json.dumps(item.embedding)`、`json.dumps(data.get("payload"), ensure_ascii=False)`、`item.modality`、`json.dumps(data["relations"], ensure_ascii=False)`。其中 `embedding` 用普通 `json.dumps`（向量是数字列表，无需 `ensure_ascii`），`payload` 用 `data.get("payload")` 容错取键，`metadata` 与 `relations` 直接用下标取键。
- **异常/边界**：`item` 非 `MemoryItem` 时抛 `TypeError`；`data` 缺少 `metadata`/`created_at`/`updated_at`/`expires_at`/`timestamp`/`relations` 任一键时抛 `KeyError`；`item.embedding` 或 `data["metadata"]` 等含不可 JSON 序列化的对象时抛 `TypeError`（来自 `json.dumps`）；SQL 执行失败（列不匹配、类型约束、磁盘满）抛 `sqlite3.Error`。特别注意：当 `embedding` 或 `payload` 为 `None` 时，`json.dumps(None)` 得到字符串 `"null"` 而不是 SQL NULL，因此这两列实际存的是文本 `"null"`；`_decode` 中 `embedding` 用真值判断、`payload` 用 `is not None` 判断，读取时会分别把 `"null"` 解析回 `None`。
- **同文件关系**：被 `upsert()`（单条）与 `upsert_many()`（批量循环）调用；它写入的表由 `_initialize()` 创建，写出的列被 `_decode()` 读取还原；它不调用本文件其他函数（除依赖导入的 `MemoryItem`）。

### `upsert(self, item: MemoryItem) -> None` （第 125 行）
- **作用**：`BaseDocumentStore.upsert` 的具体实现，是对外暴露的单条写入入口。它把「开事务」这件事交给 `_connection_scope()`，把「怎么拼 SQL」交给 `_upsert_on()`，自己只做一层很薄的编排。上层保存/更新一条记忆时调用它，无需关心是内存库还是文件库、也无需关心是插入还是覆盖。因为整个过程在一个 `with` 作用域内，写操作要么整体提交、要么整体回滚。
- **参数**：`self` 为实例；`item` 类型为 `MemoryItem`，要写入的那条记忆，其 `id` 决定覆盖哪一行。
- **返回**：`None`。
- **内部流程**：`with self._connection_scope() as connection:` 进入作用域（加锁、取连接、开事务），在块内调用 `self._upsert_on(connection, item)` 执行 UPSERT，块正常退出时由 `with connection:` 自动提交，随后 `finally` 关闭临时连接（内存连接除外）并释放锁。
- **异常/边界**：`item` 类型不对时由 `_upsert_on` 抛 `TypeError`；JSON 序列化失败抛 `TypeError`；SQL 失败抛 `sqlite3.Error`。任何异常都会导致本次事务回滚且锁被释放（`finally` 保证）。
- **同文件关系**：调用 `_connection_scope()` 与 `_upsert_on()`；覆盖基类 `BaseDocumentStore.upsert`；与 `upsert_many()` 共享同一写入路径。

### `upsert_many(self, items: list[MemoryItem]) -> None` （第 129 行）
- **作用**：批量写入方法，专门服务于「投影重建」这类场景——一次性把大量记忆行重新落库。它与逐条调用 `upsert()` 的关键区别在于：整个批次共用**同一个** `_connection_scope()`，也就是同一个 SQLite 事务，因此要么全部成功提交、要么中途失败全部回滚，不会出现「重建到一半、库处于半新半旧」的不一致状态；同时只加锁一次、只开一次连接，性能也明显优于循环单条写。文档字符串「Commit a projection rebuild's memory rows in one SQLite transaction」正是这个意思。
- **参数**：`self` 为实例；`items` 类型为 `list[MemoryItem]`，必须是 `list` 实例（不接受元组、生成器等），元素应全部是 `MemoryItem`。
- **返回**：`None`。
- **内部流程**：第一步 `if not isinstance(items, list): raise TypeError("items must be a list")` 做容器类型校验；第二步 `with self._connection_scope() as connection:` 只开一个作用域；第三步 `for item in items:` 逐个调用 `self._upsert_on(connection, item)`；循环结束后作用域退出，`with connection:` 一次性提交所有语句。传入空列表时循环不执行，只提交一个空事务，属于合法的无操作。
- **异常/边界**：`items` 不是 `list` 抛 `TypeError`；列表中某个元素不是 `MemoryItem`、或缺少必需键、或 JSON 序列化失败时，异常在循环中抛出，导致整个批次回滚（此前已执行的 UPSERT 全部撤销），异常继续上抛；SQL 层面的错误同理。列表很长时所有语句都在同一事务内，内存与锁占用会随批次增长。
- **同文件关系**：调用 `_connection_scope()` 与 `_upsert_on()`（在循环中）；与 `upsert()` 共享 `_upsert_on()` 的写入逻辑。

### `get(self, item_id: str) -> MemoryItem | None` （第 138 行）
- **作用**：按主键读取单条记忆并反序列化成 `MemoryItem`，是 `BaseDocumentStore.get` 的实现。它走主键索引，开销极小，适合按 id 精确定位（例如按关系引用回溯、判断某条记忆是否已存在）。需要注意的是它**不做过期过滤**：即使目标记忆的 `expires_at` 已过，也照样返回，过期与否交由调用方用 `item.is_expired` 自行判断，这与 `list()` 默认过滤过期的行为不同。
- **参数**：`self` 为实例；`item_id` 类型为 `str`，目标记忆的主键，参数以占位符方式绑定，天然免疫 SQL 注入。
- **返回**：命中时返回反序列化后的 `MemoryItem`；未命中（`fetchone()` 得到 `None`）返回 `None`。
- **内部流程**：`with self._connection_scope() as connection:` 内执行 `connection.execute("SELECT * FROM memories WHERE id = ?", (item_id,)).fetchone()` 取第一行（主键唯一，最多一行），把行赋给 `row`；退出作用域后（连接可能已关闭，但 `sqlite3.Row` 是已取出的数据副本，仍可用）执行 `return self._decode(row) if row else None`——行存在则调用 `_decode()` 还原对象，否则返回 `None`。
- **异常/边界**：库中某行列内容损坏导致 JSON 解析失败时，`_decode()` 内的 `json.loads` 抛 `json.JSONDecodeError`；SQL 执行失败抛 `sqlite3.Error`。`item_id` 为 `None` 或空串不会报错，只是必然查不到，返回 `None`。
- **同文件关系**：调用 `_connection_scope()` 与 `_decode()`；覆盖基类 `BaseDocumentStore.get`；与 `list()` 共用 `_decode()`。

### `delete(self, item_id: str) -> bool` （第 143 行）
- **作用**：按主键删除一条记忆，是 `BaseDocumentStore.delete` 的实现，并用返回值区分「删掉了」和「不存在」。这个布尔结果让上层可以做精确的删除计数与幂等处理（例如重复删除同一条时第二次返回 `False` 而不是报错）。删除在事务作用域内执行，正常退出即提交。
- **参数**：`self` 为实例；`item_id` 类型为 `str`，要删除的记忆主键，以占位符绑定。
- **返回**：`bool`——`cursor.rowcount > 0` 为 `True`（确实删除了至少一行），否则 `False`（该 id 不存在）。注意 `return` 语句写在 `with self._connection_scope()` 块内部，返回前 `with connection:` 会先提交，`finally` 会先关闭临时连接并释放锁。
- **内部流程**：进入 `_connection_scope()` 作用域，执行 `connection.execute("DELETE FROM memories WHERE id = ?", (item_id,))` 拿到 `cursor`，然后直接 `return cursor.rowcount > 0`；`rowcount` 是 SQLite 报告的被影响行数，删除不存在的主键时为 `0`。
- **异常/边界**：SQL 执行失败（表不存在、库只读、磁盘满）抛 `sqlite3.Error`，此时事务回滚；`item_id` 为 `None` 或空串不报错，返回 `False`。本方法不校验 `item_id` 类型，非字符串会由 sqlite3 的绑定规则处理（可能抛 `sqlite3.InterfaceError`）。
- **同文件关系**：调用 `_connection_scope()`；覆盖基类 `BaseDocumentStore.delete`；与 `get()` 一样直接操作 `memories` 表，但独立实现、不共用辅助函数。

### `list(self, *, memory_type: MemoryType | str | None = None, include_expired: bool = False) -> list[MemoryItem]` （第 148 行）
- **作用**：按条件枚举记忆，是 `BaseDocumentStore.list` 的实现，也是唯一带筛选与排序的读取方法。它先在 SQL 层按类型过滤、按时间倒序排好，再把所有行反序列化，最后在 Python 层根据 `include_expired` 决定是否剔除已过期的记忆。之所以把过期过滤放在 Python 层而不是 SQL 层，是因为过期判断逻辑（`MemoryItem.is_expired`）属于记忆对象的语义，取决于对象内部对 `expires_at` 的解析，本文件不重复实现它，从而保证「过期」的定义只有一处。默认不返回过期记忆，符合「记忆检索只应看到有效记忆」的常规预期。
- **参数**：`self` 为实例；`memory_type` 为仅关键字参数，类型 `MemoryType | str | None`，默认 `None`——为 `None` 时不加 WHERE 条件返回全部类型，否则用 `MemoryType(memory_type).value` 转成枚举再取其字符串值参与过滤；`include_expired` 为仅关键字参数，类型 `bool`，默认 `False`——为 `False` 时过滤掉 `is_expired` 为真的项，为 `True` 时全部保留。
- **返回**：`list[MemoryItem]`——反序列化后的记忆列表，已按时间倒序（`COALESCE(timestamp, created_at) DESC`）；无匹配时返回空列表而不是 `None`。
- **内部流程**：第一步初始化 `query = "SELECT * FROM memories"` 与空参数列表 `params: list[Any] = []`；第二步若 `memory_type is not None`，给 `query` 追加 `" WHERE memory_type = ?"` 并把 `MemoryType(memory_type).value` 追加到 `params`；第三步无条件追加 `" ORDER BY COALESCE(timestamp, created_at) DESC"`（有 `timestamp` 就用它、为空则退回 `created_at`，保证排序键不缺失）；第四步在 `_connection_scope()` 内 `connection.execute(query, params).fetchall()` 取出全部行；第五步列表推导 `items = [self._decode(row) for row in rows]` 逐行还原；第六步 `return items if include_expired else [item for item in items if not item.is_expired]` 按需过滤后返回。
- **异常/边界**：`memory_type` 传了既不是 `MemoryType` 成员也不是合法枚举值的字符串时，`MemoryType(memory_type)` 抛 `ValueError`（在查询执行之前抛出，不会产生副作用）；某行 JSON 损坏时 `_decode()` 抛 `json.JSONDecodeError`；SQL 失败抛 `sqlite3.Error`。数据量大时 `fetchall()` 会把全部行一次性读进内存，属已知边界；`memory_type` 传入 `MemoryType` 成员本身时 `MemoryType(memory_type)` 幂等返回同一成员，行为正确。
- **同文件关系**：调用 `_connection_scope()` 与 `_decode()`（在列表推导中）；覆盖基类 `BaseDocumentStore.list`；与 `get()` 共用 `_decode()` 反序列化路径。

### `_decode(row: sqlite3.Row) -> MemoryItem` （第 161 行）
- **作用**：把一行数据库记录还原成内存中的 `MemoryItem` 对象，是读取路径上的「反序列化中枢」，被 `get()` 与 `list()` 共用。它写成 `@staticmethod`，因为它不依赖实例状态，只做纯粹的字段映射与 JSON 还原。它逐列把 TEXT 形式的 JSON 解析回 Python 结构，并对两类字段做了区别处理：`payload` 和 `relations` 在 `json.loads` 之后还要经过 `_json_restore()`，说明这两个字段在写入时可能被 `_json_restore` 的逆过程做过特殊编码（例如元组/集合等非原生 JSON 类型的标记化），需要额外还原；而 `metadata` 只做普通 `json.loads`。`embedding` 用真值判断，兼容存成 `"null"` 文本的情况。
- **参数**：`row` 类型为 `sqlite3.Row`，是 `SELECT *` 取出的一行，必须包含 `memories` 表的全部列，且依赖连接设置了 `row_factory = sqlite3.Row` 才能按列名下标访问。
- **返回**：构造好的 `MemoryItem` 实例，字段与库中记录一一对应。
- **内部流程**：第一步处理 `payload`——`payload = _json_restore(json.loads(row["payload"])) if row["payload"] is not None else None`，即列值非 `None` 时先 `json.loads` 再 `_json_restore`，否则直接 `None`（注意存成文本 `"null"` 时会走前半支：`json.loads("null")` 得 `None`，再传给 `_json_restore`）；第二步 `return MemoryItem(...)`，逐字段赋值：`id=row["id"]`、`content=row["content"]`、`memory_type=row["memory_type"]`（直接传字符串，由 `MemoryItem` 负责转成 `MemoryType`）、`metadata=json.loads(row["metadata"])`、`importance=row["importance"]`、`created_at=row["created_at"]`、`updated_at=row["updated_at"]`、`expires_at=row["expires_at"]`、`timestamp=row["timestamp"]`、`embedding=json.loads(row["embedding"]) if row["embedding"] else None`、`payload=payload`、`modality=row["modality"]`、`relations=_json_restore(json.loads(row["relations"]))`。
- **异常/边界**：任一 JSON 列内容不是合法 JSON 时抛 `json.JSONDecodeError`；`metadata` 或 `relations` 为 SQL NULL 时 `json.loads(None)` 抛 `TypeError`（表中这两列声明为 `NOT NULL`，正常写入路径不会产生 NULL）；`memory_type` 字符串不是合法枚举值时由 `MemoryItem` 的构造逻辑决定是否抛 `ValueError`；`embedding` 为空串或 `None` 时返回 `None`。
- **同文件关系**：被 `get()` 与 `list()` 调用；依赖导入的 `MemoryItem` 与 `_json_restore`（来自 `..base`）；它读取的列由 `_initialize()` 定义、由 `_upsert_on()` 写入，构成写入与读取的对称闭环。

### `close(self) -> None` （第 172 行）
- **作用**：释放存储持有的资源，覆盖基类的空实现。它只针对「被钉住的内存连接」做处理：若 `self._connection` 不为 `None`（内存模式），就在锁保护下关闭它并把字段置回 `None`，表示存储已关闭；对文件模式而言 `self._connection` 本来就是 `None`，因此调用它是完全的无操作（文件库的连接都是短生命周期的临时连接，由 `_connection_scope()` 的 `finally` 负责关闭，无需在这里管）。关闭后对象仍可被调用，只是内存模式下会重新连出一个空的内存库（数据已随旧连接消失），所以它更适合在对象生命周期结束时调用一次。
- **参数**：`self` 为实例；无其他参数。
- **返回**：`None`。
- **内部流程**：`with self._lock:` 加锁（避免与正在执行的操作竞争关闭连接），块内判断 `if self._connection is not None:`，成立则依次执行 `self._connection.close()` 与 `self._connection = None`；不成立则什么也不做。由于 `close()` 幂等，重复调用第二次会因 `_connection` 已是 `None` 而直接跳过。
- **异常/边界**：`sqlite3.Connection.close()` 在极少数情况下（如仍有未提交事务）可能抛 `sqlite3.Error`，此时 `_connection` 不会被置为 `None`，对象仍持有一个可能已失效的连接引用；关闭后再次读写内存库不会报错，而是得到一个新的空库（数据丢失），这是使用内存模式时必须注意的边界。
- **同文件关系**：覆盖基类 `BaseDocumentStore.close`；读取并修改 `__init__` 设置的 `self._connection`，使用 `__init__` 创建的 `self._lock`；与 `_connect()`/`_connection_scope()` 共享同一连接字段，后两者的行为直接受本次关闭结果影响。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `BaseDocumentStore` | 文档存储的抽象基类，用抽象方法规定 upsert/get/delete/list/close 五个契约动作。 |
| `BaseDocumentStore.upsert` | 抽象声明「写入或更新一条 MemoryItem」，无实现。 |
| `BaseDocumentStore.get` | 抽象声明「按 id 取回一条 MemoryItem，缺失返回 None」。 |
| `BaseDocumentStore.delete` | 抽象声明「按 id 删除，返回是否真的删掉了」。 |
| `BaseDocumentStore.list` | 抽象声明「按类型与是否含过期条件枚举记忆列表」。 |
| `BaseDocumentStore.close` | 基类给出的空实现，让后端可选地释放资源。 |
| `SQLiteDocumentStore` | 基于 SQLite 的具体文档存储，支持文件持久化与 `:memory:` 内存模式。 |
| `SQLiteDocumentStore.__init__` | 规范化路径、建父目录、备锁与连接、调用建表初始化。 |
| `SQLiteDocumentStore.connection` | 只读属性，暴露被钉住的内存连接，文件模式返回 None。 |
| `SQLiteDocumentStore._connect` | 复用钉住连接或为文件库新建临时连接并设置 Row 工厂。 |
| `SQLiteDocumentStore._connection_scope` | 加锁 + 开事务 + 交出连接 + 自动提交/回滚 + 必要时关连接的上下文管理器。 |
| `SQLiteDocumentStore._initialize` | 幂等创建 `memories` 表与类型、时间戳两个索引。 |
| `SQLiteDocumentStore._upsert_on` | 在给定连接上执行 INSERT ... ON CONFLICT 的 13 列 UPSERT。 |
| `SQLiteDocumentStore.upsert` | 单条写入入口，在一个事务作用域内调用 `_upsert_on`。 |
| `SQLiteDocumentStore.upsert_many` | 把投影重建的多条记忆放进同一个 SQLite 事务批量写入。 |
| `SQLiteDocumentStore.get` | 按主键查询一行并反序列化，未命中返回 None，不过滤过期。 |
| `SQLiteDocumentStore.delete` | 按主键删除一行，用 `rowcount` 判断是否删除成功。 |
| `SQLiteDocumentStore.list` | 按类型过滤、按时间倒序取出全部行，并按需剔除过期记忆。 |
| `SQLiteDocumentStore._decode` | 把一行 sqlite3.Row 还原为 MemoryItem，负责各 JSON 字段的反序列化。 |
| `SQLiteDocumentStore.close` | 在锁内关闭并清空被钉住的内存连接，文件模式下为空操作。 |
