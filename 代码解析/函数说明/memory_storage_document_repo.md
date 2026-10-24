# memory/storage/document_repo.py

## 一、这个文件是干什么的

这个文件是四层记忆系统里「语料原文」这一层的持久化实现，也是整个知识库的**本地唯一事实来源（source of truth）**。它把 `documents`（文档原文）、`chunks`（分块及其字符区间）、`ingest_jobs`（一句话后台入库队列）、`embedding_lock`（嵌入空间锁）四张表建在与 `memories` 同一个 SQLite 文件里，但刻意与「四类记忆」解耦：记忆表存的是提炼后的记忆，这里存的是语料本身。模块顶部注释明确说明，向量库与图结构都只是「投影」，可以从这里的原文与分块边界重新构建，因此本模块从不与向量库或图存储通信——把原始文本留在磁盘上，正是嵌入空间变更后还能重建索引的前提。文件里主要包含四类 `@dataclass` 记录对象（`DocumentRecord`、`ChunkRecord`、`EmbeddingLockRecord`、`IngestJobRecord`）、一组模块级状态机常量（`DOCUMENT_STATUSES`、`INGEST_JOB_STATUSES`、`CHUNK_VECTOR_STATUSES`、`PERMISSIONS`、`FTS_TOKENIZERS`）、核心仓储类 `DocumentRepository`，以及一个模块级的行解码辅助函数 `_ingest_job_from_row`。仓储类同时承担 SQLite 建表与索引、FTS5 全文索引的建立与触发器维护、文档/分块的增删改查、BM25 关键词检索、统计报表、嵌入空间锁读写、以及后台入库任务的排队与状态推进。它被 Web 应用的上传/解析/向量化/抽取流水线、检索链路（FTS 与向量混合召回）、以及一句话入库 worker 共同使用；构造时既支持文件数据库（每次调用新建连接），也支持 `:memory:`（单连接固定复用，便于测试）以及由外部注入连接与 `memories` 同库复用。

## 二、函数与类逐条详解

### `DocumentRecord` （第 37 行）
- **作用**：这是「一篇已入库文档」的内存表示，是整个仓储读写的核心数据载体。上传接口解析完文件后，会用它把标题、归一化后的全文、来源、标签、权限、状态等一次性传给 `upsert_document` 落库；检索或列表页从库里读回来的行，也会先被 `_decode_document` 还原成这个对象再交给上层。它把数据库行与业务代码隔离开，让调用方不必关心 `tags` 在库里其实是 JSON 字符串、时间戳其实是 ISO 文本。字段上它把「谁」（document_id、title、source、tags、permission）与「进展」（status、error、created_at、updated_at）放在同一个对象里，方便流水线每一步只改状态而不必重新组装整个文档。它也是「原文可重建」这一设计的具体体现：`raw_text` 直接躺在记录里，分块可以据此重新切分。
- **参数**：数据类字段即参数。`document_id: str` 主键，必填，无默认值，要求非空字符串（由 `upsert_document` 校验）；`title: str = ""` 文档标题，默认空串；`raw_text: str = ""` 归一化后的完整原文，默认空串；`source: str = ""` 来源标识（如文件名或 URI），默认空串；`tags: list[str] = field(default_factory=list)` 标签列表，用 `default_factory` 避免可变默认值共享，默认空列表；`permission: str = "private"` 权限等级，取值须属于 `PERMISSIONS`，默认最保守的 `"private"`；`status: str = "uploaded"` 生命周期状态，取值须属于 `DOCUMENT_STATUSES`，默认 `"uploaded"`；`error: str | None = None` 失败原因，允许为 `None`；`created_at: str = ""` 创建时间 ISO 字符串，默认空串（落库时会用 `utc_now()` 兜底）；`updated_at: str = ""` 更新时间 ISO 字符串，默认空串（同样会被兜底）。
- **返回**：它是数据类，构造时返回一个 `DocumentRecord` 实例；本身不产生业务返回值。
- **内部流程**：由 `@dataclass` 装饰器自动生成 `__init__`、`__repr__`、`__eq__`。实例化时按字段顺序赋默认值，`tags` 通过 `field(default_factory=list)` 在每次实例化时新建一个空列表。没有 `__post_init__`，所以构造阶段不做任何校验，校验责任全部交给写入方 `upsert_document`。
- **异常/边界**：构造本身不抛异常，传任意类型都不会当场报错（例如 `document_id=123` 也能构造成功）。真正的类型与取值校验发生在 `upsert_document`：非 `DocumentRecord` 抛 `TypeError`，非法 `permission`/`status` 抛 `ValueError`，空 `document_id` 抛 `ValueError`。`error` 字段允许 `None`，因为数据库列本身可空。
- **同文件关系**：被 `DocumentRepository.upsert_document` 消费、被 `_decode_document` 构造、被 `get_document` 与 `list_documents` 返回；`list_documents` 的 SQL 依赖它的 `tags`（JSON 文本）与 `status` 字段语义。它不调用本文件任何函数。

### `ChunkRecord` （第 53 行）
- **作用**：表示「文档被切分后的一块」，并额外记录这块文本在原文 `raw_text` 中的字符区间 `[char_start, char_end)`。字符区间是它存在的关键理由：向量库和图只存 chunk_id，命中之后要展示原文上下文时，必须靠这两个偏移回到 `documents.raw_text` 里精确定位，避免重复存一份长文本。`vector_status` 让向量化流程可以按块推进并支持断点续跑——入库先写 `pending`，索引成功后改 `indexed`，失败改 `failed`，`chunk_ids(vector_status=...)` 再据此捞出需要重试的块。它同时被 `upsert_chunks` 批量写入，支撑一次性把整篇文档的所有分块提交。
- **参数**：`chunk_id: str` 主键，必填，无默认值；`document_id: str` 所属文档，必填，无默认值；`chunk_index: int` 文档内顺序号，必填，用于 `list_chunks` 与 `list_all_chunks` 的排序；`char_start: int` 在原文中的起始字符偏移，必填；`char_end: int` 结束偏移，必填；`text: str` 分块正文，必填；`vector_status: str = "pending"` 向量化状态，默认 `"pending"`，取值须属于 `CHUNK_VECTOR_STATUSES`。
- **返回**：构造返回 `ChunkRecord` 实例；自身不产生业务返回值。
- **内部流程**：同样由 `@dataclass` 生成构造与比较方法。没有默认值的六个字段必须按位置或关键字全部提供；仅 `vector_status` 可省略。没有 `__post_init__` 校验，`char_start`/`char_end` 的大小关系、是否越界都不在这里检查。
- **异常/边界**：构造不抛异常。写入时 `upsert_chunk` / `upsert_chunks` 会校验类型（非 `ChunkRecord` 抛 `TypeError`）与 `vector_status` 合法性（非法抛 `ValueError`）。空 `chunk_id`、负数偏移、`char_end < char_start` 均不校验，直接写入数据库。
- **同文件关系**：被 `_write_chunk` 序列化写入、被 `_decode_chunk` 还原、被 `upsert_chunk`/`upsert_chunks`/`get_chunk`/`list_chunks`/`list_all_chunks` 使用。它不调用本文件任何函数。

### `EmbeddingLockRecord` （第 66 行）
- **作用**：这是「本 SQLite 文件所绑定嵌入空间」的单行锁记录，用来解决一个致命的一致性问题：如果向量是用 A 模型（比如 1024 维）算的，而某天配置换成了 B 模型（比如 768 维），那么旧向量与新查询向量将不可比，混用会得到完全错误的相似度。把 `model` 与 `dimension` 固化在库里后，系统启动或写入前可以比对当前配置与锁记录，不一致就触发全量重建而不是静默出错。它被声明为 `frozen=True`，即不可变值对象，适合当作比对基准与缓存键，避免被误改。`updated_at` 让运维能知道这个锁最后一次变更的时间。
- **参数**：`model: str` 嵌入模型标识，必填，无默认值（写入时由 `set_embedding_lock` 校验非空）；`dimension: int` 向量维度，必填，无默认值（写入时校验为正整数）；`updated_at: str = ""` 最后更新时间 ISO 字符串，默认空串。
- **返回**：构造返回一个不可变的 `EmbeddingLockRecord` 实例。
- **内部流程**：`@dataclass(frozen=True)` 生成 `__init__`、`__repr__`、`__eq__`，并禁止属性赋值与删除；由于 `frozen=True` 且所有字段都是可哈希的不可变类型，实例自身也是可哈希的，可以直接放进 `set` 或当作 `dict` 键。类体只包含三个字段声明与文档字符串，没有自定义方法。
- **异常/边界**：构造不抛异常；一旦构造完成，对其字段赋值会抛 `dataclasses.FrozenInstanceError`。取值合法性由 `set_embedding_lock` 负责（空 `model` 或非正 `dimension` 抛 `ValueError`）。
- **同文件关系**：由 `get_embedding_lock` 构造并返回、由 `set_embedding_lock` 构造并返回；它不调用本文件任何函数。

### `IngestJobRecord` （第 75 行）
- **作用**：表示「一句话后台入库」任务的一条持久化记录，让「提交即返回、后台慢慢入库」成为可能，并且能跨进程重启续跑。用户提交一句话后拿到 `job_id`，Web 层可随时查询它处于排队中（`pending`）、正在入库（`running`）、成功（`done`）还是失败（`failed`）；`attempts` 记录被尝试的次数，用于判断是否反复失败；`error` 保存失败原因；`result` 保存完成后的 JSON 摘要（块数 + 抽取报告），历史记录页直接把它展示出来，不必再回查文档表。`event_at` 则保留这句话对应的事件时间，与「入库时间」`created_at` 区分开。正因为状态与结果都落在 SQLite 里，进程崩溃后 `restart_stale_ingest_jobs` 才能把卡在 `running` 的任务重新排队。
- **参数**：`job_id: str` 主键，必填，由 `create_ingest_job` 以 `f"job_{uuid4().hex}"` 生成；`text: str` 待入库的一句话正文，必填（创建时校验非空）；`event_at: str = ""` 事件发生时间，默认空串；`kind: str = "sentence"` 任务种类，默认 `"sentence"`（该字段在库里无枚举约束，可扩展）；`status: str = "pending"` 任务状态，默认 `"pending"`，取值须属于 `INGEST_JOB_STATUSES`；`attempts: int = 0` 已尝试次数，默认 0；`error: str = ""` 失败信息，默认空串；`result: str = ""` 完成后的 JSON 摘要文本，默认空串；`created_at: str = ""` 创建时间，默认空串；`updated_at: str = ""` 更新时间，默认空串。
- **返回**：构造返回 `IngestJobRecord` 实例；自身不产生业务返回值。
- **内部流程**：`@dataclass` 生成标准方法，字段顺序与 `ingest_jobs` 表的列顺序刻意保持一致，便于阅读时对照。构造阶段不做校验，`status`/`attempts` 的合法性由 `set_ingest_job_status` 与 `list_ingest_jobs` 在写入或过滤时校验。
- **异常/边界**：构造不抛异常。真正的边界处理在仓储方法里：空 `text` 由 `create_ingest_job` 抛 `ValueError`；非法 `status` 由 `set_ingest_job_status` 与 `list_ingest_jobs` 抛 `ValueError`；`attempts` 由 SQL 层的 `attempts + :attempt` 自增，不做 Python 侧校验。
- **同文件关系**：由 `create_ingest_job` 构造并返回、由 `get_ingest_job`/`list_ingest_jobs`/`reset_failed_ingest_job` 返回、由模块级函数 `_ingest_job_from_row` 从数据库行还原；它不调用本文件任何函数。

### `DocumentRepository` （第 95 行）
- **作用**：这是本文件唯一的核心类，负责 `documents`、`chunks`、`ingest_jobs`、`embedding_lock` 四张表在共享 memory SQLite 文件上的全部读写。类文档字符串说明它刻意镜像了 `memory.storage.document.SQLiteDocumentStore` 的连接与加锁模型：文件数据库「每次调用一条新连接」，所有作用域外面套一把 `RLock`，而 `:memory:` 则固定复用同一条连接，使测试可以用一个临时库。它同时负责建表、建索引、建 FTS5 虚拟表与三个同步触发器，并在 FTS5 不可用时把 `fts_tokenizer` 置为 `None` 让检索优雅退化为纯向量，而不是直接拒绝启动。上层上传/解析/向量化/抽取流水线用它落库与推进状态，检索层用它做 BM25 关键词召回，后台 worker 用它领任务与写结果。
- **参数**：类本身没有构造参数之外的输入；其方法参数各自详见下条。
- **返回**：类是类型对象，实例化后返回 `DocumentRepository` 实例。
- **内部流程**：类体按区块组织：连接管道（`_connect`、`_connection_scope`、`_initialize`、`_initialize_fts`）、文档区（`upsert_document` 至 `delete_document`）、分块区（`upsert_chunk` 至 `set_chunk_vector_status`）、关键词检索区（`_fts_query`、`search_keywords`）、报表区（`chunk_counts` 至 `stats`、`close`）、一句话后台入库队列区（`create_ingest_job` 至 `reset_failed_ingest_job`）、以及辅助区（`_check_choice`、`_decode_document`、`_decode_chunk`）。
- **异常/边界**：类本身不抛异常；各方法在参数非法时抛 `TypeError`/`ValueError`，在 SQL 层面遇到 FTS5 缺失时内部消化为 `fts_tokenizer = None`。
- **同文件关系**：它调用本文件全部数据类与 `_ingest_job_from_row`；`__all__` 把它列为对外导出符号，外部模块直接使用它。

### `DocumentRepository.__init__(self, path: str | Path = ":memory:", *, connection: sqlite3.Connection | None = None) -> None` （第 104 行）
- **作用**：构造仓储实例并立刻完成数据库的建表初始化，让对象一诞生就可直接使用。它区分三种运行形态：显式传入 `connection` 时（F2 场景）复用外部连接，与 `memories` 共用同一个库，仅测试使用；`path` 为 `:memory:` 时自己开一条 `check_same_thread=False` 的连接，让测试可以用一次性草稿库；其余情况走文件路径，每次操作时再临时开连接。它还会自动为文件路径创建父目录，避免因为目录不存在而建库失败。`fts_tokenizer` 在这里先置为 `None`，真正取值由 `_initialize_fts` 决定。整个过程用一把 `threading.RLock` 保护，因为 Web 应用是多线程的。
- **参数**：`path: str | Path = ":memory:"` 数据库路径，默认内存库；传入 `Path` 会被 `expanduser()` 展开 `~` 并转成字符串，若字符串恰为 `":memory:"` 则保持原样不做路径处理；`connection: sqlite3.Connection | None = None` 仅关键字传入的可选注入连接，非 `None` 时优先使用它，且会强制把 `row_factory` 设为 `sqlite3.Row`。
- **返回**：无返回值（`None`），构造完成后实例可立即使用。
- **内部流程**：先把 `path` 规范化存入 `self.path`；若 `self.path != ":memory:"`，调用 `Path(self.path).parent.mkdir(parents=True, exist_ok=True)` 保证目录存在；接着创建 `self._lock = threading.RLock()`、把 `self._connection` 初始化为 `None`、把 `self.fts_tokenizer` 初始化为 `None`。然后按优先级分支：`connection is not None` 时直接把它赋给 `self._connection` 并设置 `row_factory`；否则若 `self.path == ":memory:"`，用 `sqlite3.connect(self.path, check_same_thread=False)` 建连接并设置 `row_factory`；否则保持 `self._connection` 为 `None`，留给 `_connect` 每次现开。最后无条件调用 `self._initialize()` 建表建索引建 FTS。
- **异常/边界**：目录不可创建、路径非法或文件不可写时，`mkdir` 或 `sqlite3.connect` 会抛出 `OSError`/`sqlite3.OperationalError`，构造函数直接向外抛出。传入既不是 `str` 也不是 `Path` 的对象时，`str(path)` 仍会强行转换，可能得到一个意外路径。`check_same_thread=False` 意味着内存库连接会跨线程共享，安全性由 `self._lock` 保证。FTS5 不可用不会导致构造失败，只把 `fts_tokenizer` 留为 `None`。
- **同文件关系**：调用 `_initialize`，`_initialize` 内部再调用 `_connection_scope`（进而调用 `_connect`）与 `_initialize_fts`；被外部模块直接实例化。

### `DocumentRepository._connect(self) -> sqlite3.Connection` （第 122 行）
- **作用**：这是「按需取连接」的统一入口，屏蔽了固定连接与临时连接的区别。当实例是内存库或由外部注入连接时，`self._connection` 非空，直接把它交出去复用，从而保证 `:memory:` 库的内容不会因为换连接而凭空消失（内存库的生命周期与连接绑定）。当实例是文件库时，这里每次都 `sqlite3.connect` 新建一条连接，这正是类文档所说的「每次调用一条连接」模型——好处是不必操心长连接被多线程共享、也不会因为连接空闲太久而持有文件锁。它还统一设置 `row_factory = sqlite3.Row`，让上层可以用列名（如 `row["chunk_id"]`）取值而不是靠下标，`_decode_document` 等函数都依赖这一点。
- **参数**：无参数（除 `self`）。
- **返回**：返回一个可用的 `sqlite3.Connection`，其 `row_factory` 一定是 `sqlite3.Row`。
- **内部流程**：先判断 `self._connection is not None`，成立则原样 `return`；否则执行 `sqlite3.connect(self.path)` 新建连接，把 `connection.row_factory = sqlite3.Row`，再返回这条新连接。新连接的所有权交给调用方 `_connection_scope` 负责关闭。
- **异常/边界**：路径不可写、磁盘满、数据库文件损坏等情况下，`sqlite3.connect` 会抛 `sqlite3.OperationalError`，本函数不做捕获。没有超时参数，因此遇到数据库被其他写事务长时间占用时可能抛 `database is locked`。无特殊空值处理（`self.path` 在 `__init__` 已规范化）。
- **同文件关系**：被 `_connection_scope` 调用，`_connection_scope` 又被 `__init__` 间接经由 `_initialize` 以及几乎所有公开方法调用；它不调用本文件其它函数。

### `DocumentRepository._connection_scope(self) -> Iterator[sqlite3.Connection]` （第 129 行）
- **作用**：这是整个仓储的并发与事务骨架，用 `@contextmanager` 把「加锁 → 取连接 → 开事务 → 交出连接 → 提交或回滚 → 关闭临时连接」这一整套动作封装成一个 `with` 语句。它保证任意时刻只有一个线程在操作数据库（`RLock` 可重入，因此 `reset_failed_ingest_job` 在 `with` 内再调 `get_ingest_job` 不会死锁）。`with connection:` 块会在正常退出时自动 `commit`、异常时自动 `rollback`，所以 `delete_document` 里「删 chunks 再删 document」天然具备原子性。最后在 `finally` 中判断连接是否是复用连接，只有临时连接才关闭，避免把内存库或注入连接关掉导致后续操作失败。
- **参数**：无参数（除 `self`）。
- **返回**：这是一个生成器上下文管理器，`with` 语句的 `as` 值是一条 `sqlite3.Connection`；函数本身的类型标注为 `Iterator[sqlite3.Connection]`。
- **内部流程**：先用 `with self._lock:` 取到可重入锁；锁内调用 `self._connect()` 取得连接；进入 `try`，用 `with connection:` 开启 SQLite 事务上下文并把连接 `yield` 给调用者；无论调用者正常结束还是抛异常，都会走到 `finally`，其中判断 `connection is not self._connection`，成立则 `connection.close()`，把临时连接释放；复用连接则不关。
- **异常/边界**：调用者代码块内抛出的异常会穿过 `with connection:`（先触发回滚），再经过 `finally` 关闭临时连接后继续向外传播，本函数不吞异常。锁是 `RLock`，同一线程重入安全。没有设置 SQLite 忙等待超时，高并发写时可能抛 `sqlite3.OperationalError: database is locked`。
- **同文件关系**：被本文件几乎所有方法调用（`_initialize`、`upsert_document`、`get_document`、`list_documents`、`count_documents`、`set_status`、`delete_document`、`upsert_chunk`、`upsert_chunks`、`get_chunk`、`list_chunks`、`set_chunk_vector_status`、`search_keywords`、`chunk_counts`、`chunk_ids`、`list_all_chunks`、`get_embedding_lock`、`set_embedding_lock`、`stats`、`create_ingest_job`、`get_ingest_job`、`set_ingest_job_status`、`list_ingest_jobs`、`restart_stale_ingest_jobs`、`reset_failed_ingest_job`）；它调用 `_connect`。

### `DocumentRepository._initialize(self) -> None` （第 140 行）
- **作用**：这是建库入口，负责把四张表与相关索引以幂等方式准备好，使仓储可以反复实例化而不破坏已有数据。所有语句都用 `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`，因此第二次启动时是空操作。它依次建立：`documents`（文档主表，`tags` 存 JSON 文本、`error` 可空）、`documents(status)` 索引（列表页按状态过滤用）、`chunks`（分块表，带 `document_id` 外键语义与字符区间）、`chunks(document_id)` 索引（按文档取块用）、`ingest_jobs`（一句话后台入库队列表，带默认值让 INSERT 可以省略大量列）、`ingest_jobs(status)` 索引（worker 领 `pending` 任务用）、`embedding_lock`（单行锁表，用 `CHECK (id = 1)` 强制只有一行）。最后调用 `_initialize_fts` 建全文索引。
- **参数**：无参数（除 `self`）。
- **返回**：无返回值（`None`）。
- **内部流程**：用 `with self._connection_scope() as connection:` 拿到连接与事务；随后顺序执行多条 `connection.execute(...)`：建 `documents` 表 → 建 `idx_documents_status` → 建 `chunks` 表 → 建 `idx_chunks_document` → 建 `ingest_jobs` 表 → 建 `idx_ingest_jobs_status` → 建 `embedding_lock` 表（含 `id INTEGER PRIMARY KEY CHECK (id = 1)`，`model`/`dimension`/`updated_at` 均 `NOT NULL`）；最后调用 `self._initialize_fts(connection)`，把 FTS 的建立放在同一个事务作用域内，保证 DDL 与触发器一起提交。
- **异常/边界**：若数据库文件不可写或已存在同名对象但结构不兼容，`sqlite3.OperationalError` 会向外抛出，构造随之失败。已存在的表不会被修改或迁移，因此旧库新增列不会自动补上。`documents.raw_text` 与 `chunks.text` 都是 `NOT NULL`，写入时传 `None` 会触发 `sqlite3.IntegrityError`。
- **同文件关系**：被 `__init__` 调用；它调用 `_connection_scope` 与 `_initialize_fts`。

### `DocumentRepository._initialize_fts(self, connection: sqlite3.Connection) -> None` （第 203 行）
- **作用**：负责建立和维护 `chunks.text` 上的 FTS5 全文索引，这是「嵌入隧道不可用时检索仍然活着」的兜底能力（D8），也是混合召回里的关键词一路。它用「外部内容表」模式（`content='chunks'`、`content_rowid='rowid'`）建立虚拟表 `chunks_fts`，这样索引里不重复存一份正文，只存倒排结构，节省空间；代价是外部内容表不会自己同步，必须靠三个触发器（插入/删除/更新）把 `chunks` 的变更同步过来。分词器按 `FTS_TOKENIZERS` 优先级尝试：先用 `trigram`（支持中文子串匹配，对中文语料至关重要），失败则退回 `unicode61`；两者都不可用时把 `fts_tokenizer` 留为 `None`，让仓储照常工作、检索退化为纯向量，而不是让整个仓储打不开。若表已存在，它会从 `sqlite_master` 里读回建表 SQL 判断当初用的是哪个分词器，避免把 `unicode61` 误报成 `trigram`；若是首次建立，则执行一次 `rebuild` 把已存在的 chunks 行补进索引。
- **参数**：`connection: sqlite3.Connection` 必填，由 `_initialize` 传入的、处于事务作用域内的连接；`row_factory` 必须是 `sqlite3.Row`，因为下面用 `existing["sql"]` 按列名取值。
- **返回**：无返回值（`None`），副作用是把 `self.fts_tokenizer` 设为 `"trigram"`、`"unicode61"` 或保持 `None`，并在库里创建虚拟表与触发器。
- **内部流程**：第一步查询 `sqlite_master` 里 `type = 'table' AND name = 'chunks_fts'` 的 `sql`，得到 `existing`，并把 `rebuild` 设为 `existing is None`。第二步分支：若表已存在，从 `existing["sql"]` 里查找子串 `"trigram"`，命中则 `self.fts_tokenizer = "trigram"`，否则为 `"unicode61"`；若不存在，则遍历 `FTS_TOKENIZERS`，对每个分词器尝试执行 `CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='rowid', tokenize='<tokenizer>')`，成功则记录该分词器并 `break`，抛 `sqlite3.OperationalError` 则先 `DROP TABLE IF EXISTS chunks_fts` 清理残留再 `continue` 试下一个。第三步：若 `self.fts_tokenizer is None` 直接 `return`，跳过触发器与回填。第四步：依次创建三个触发器（均为 `IF NOT EXISTS`）——`chunks_ai` 在 `chunks` 插入后向 `chunks_fts` 插入 `(new.rowid, new.text)`；`chunks_ad` 在删除后用 FTS5 约定的 `'delete'` 指令把旧行从索引移除；`chunks_au` 在更新时先按旧值 `'delete'` 再插入新值。第五步：若 `rebuild` 为真，执行 `INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')` 为 FTS 建立之前就已存在的 chunks 行一次性建索引。
- **异常/边界**：建虚拟表时只捕获 `sqlite3.OperationalError`（覆盖「no such module: fts5」等情形）并降级到下一个分词器；若两个分词器都失败，`fts_tokenizer` 保持 `None`，仓储继续可用。其它异常（如磁盘错误）会向外传播，导致构造失败。注意 `DROP TABLE IF EXISTS chunks_fts` 会连带删除该虚拟表上的触发器，但后续 `CREATE TRIGGER IF NOT EXISTS` 会重新建立。触发器创建失败（例如权限或 SQLite 版本问题）不会被捕获。`rebuild` 仅在首次建表时为真，因此后续启动不会重复全量回填。
- **同文件关系**：被 `_initialize` 调用；它读取 `FTS_TOKENIZERS` 常量、写 `self.fts_tokenizer`（该字段被 `search_keywords` 读取）；不调用其它方法。

### `DocumentRepository.upsert_document(self, doc: DocumentRecord) -> None` （第 254 行）
- **作用**：把一篇文档写入或更新到 `documents` 表，是整条入库流水线的第一步落库动作。之所以用 upsert 而不是 insert，是因为同一份文档可能在解析失败后重跑、或正文被重新归一化后再写一次，用 `ON CONFLICT(document_id) DO UPDATE` 可以安全地覆盖而不会主键冲突。它特别处理了 `created_at`：更新分支里刻意不覆盖该列，只在首次插入时落库，这样重跑 ingest 不会把「文档首次进入系统的时间」刷成当前时间。`tags` 在这里被 `json.dumps(..., ensure_ascii=False)` 序列化成 JSON 文本存进单列，中文标签保持原样不转义，读回时由 `_decode_document` 反序列化。`error` 允许写 `None`，表示当前没有错误。
- **参数**：`doc: DocumentRecord` 必填，待写入的文档记录。要求 `permission` 属于 `PERMISSIONS`（`"private"`/`"shared"`/`"public"`），`status` 属于 `DOCUMENT_STATUSES`（`"uploaded"`/`"parsed"`/`"vectorized"`/`"extracted"`/`"failed"`），`document_id` 为非空字符串；`created_at`/`updated_at` 为空串时自动用当前 UTC 时间兜底。
- **返回**：无返回值（`None`）；调用方通过返回值之外的方式（如 `get_document`）确认结果。
- **内部流程**：先做类型与取值校验：`not isinstance(doc, DocumentRecord)` 抛 `TypeError`；`self._check_choice("permission", doc.permission, PERMISSIONS)` 与 `self._check_choice("status", doc.status, DOCUMENT_STATUSES)` 校验枚举；`doc.document_id` 非字符串或 `strip()` 后为空抛 `ValueError`。然后 `now = utc_now().isoformat()` 取当前时间。接着进入 `with self._connection_scope() as connection:`，执行带 `ON CONFLICT(document_id) DO UPDATE SET` 的 INSERT，更新列表里包含 `title`、`raw_text`、`source`、`tags`、`permission`、`status`、`error`、`updated_at`，唯独没有 `created_at`。参数元组按列顺序给出，`tags` 用 `json.dumps(list(doc.tags), ensure_ascii=False)`，`created_at`/`updated_at` 用 `doc.created_at or now`、`doc.updated_at or now` 兜底。
- **异常/边界**：非法类型抛 `TypeError`，非法枚举值或空 `document_id` 抛 `ValueError`，均在写库前发生。`doc.tags` 若为 `None`，`list(None)` 会抛 `TypeError`（本函数未做兜底）。`raw_text` 为 `None` 时因列是 `NOT NULL` 会抛 `sqlite3.IntegrityError`。事务失败时 `_connection_scope` 负责回滚。无特殊超时处理。
- **同文件关系**：调用 `_check_choice` 与 `_connection_scope`（间接调用 `_connect`）；消费 `DocumentRecord`；被上层入库流水线调用，本文件内部不被其它方法调用。

### `DocumentRepository.get_document(self, document_id: str) -> DocumentRecord | None` （第 288 行）
- **作用**：按主键精确取回一篇文档，是详情页、抽取阶段读取原文、以及判断某文档是否已入库的基础查询。它把行对象交给 `_decode_document` 统一还原成 `DocumentRecord`，因此调用方拿到的是已经反序列化好 `tags` 的业务对象，而不必自己处理 JSON。查不到时返回 `None` 而不是抛异常，让调用方可以用 `if doc is None` 做「不存在」分支，这在 Web 层映射为 404 非常自然。
- **参数**：`document_id: str` 必填，文档主键；未做类型校验，传非字符串时会作为参数绑定给 SQLite（可能匹配不到任何行而返回 `None`）。
- **返回**：命中时返回 `DocumentRecord`；无匹配行时返回 `None`。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行 `SELECT * FROM documents WHERE document_id = ?`（参数化查询防注入），取 `fetchone()` 得到单行或 `None`；退出作用域（提交/关闭临时连接）后，用 `self._decode_document(row) if row else None` 返回结果。注意解码发生在作用域之外，此时行数据已完整取回内存，是安全的。
- **异常/边界**：无特殊异常处理；数据库层面错误（如表不存在、库被锁）会向外抛出。空字符串或不存在的 `document_id` 都只返回 `None`。若 `tags` 列里存了非法 JSON，`_decode_document` 中的 `json.loads` 会抛 `json.JSONDecodeError`。
- **同文件关系**：调用 `_connection_scope` 与 `_decode_document`；不被本文件其它方法调用，供外部使用。

### `DocumentRepository.list_documents(self, *, tag: str = "", status: str = "", page: int = 1, page_size: int = 20) -> tuple[list[DocumentRecord], int]` （第 295 行）
- **作用**：为文档列表页提供「分页 + 过滤 + 总数」的一站式查询。它的返回值设计成 `(items, total)` 二元组，其中 `total` 是**忽略分页但保留过滤条件**的总命中数，这样前端既能渲染当前页数据，又能算出总页数、显示「共 N 条」，不必再发一次计数请求。过滤有两个维度：按 `tag` 过滤时用 SQLite 的 `json_each(documents.tags)` 把 JSON 数组展开逐项比对，所以标签虽然存在一个文本列里，仍然可以精确匹配；按 `status` 过滤则直接比较状态列。两个过滤都写成 `? = '' OR ...` 的形式，用一个参数同时表达「不过滤」和「过滤某值」，避免拼接 SQL。排序按 `updated_at DESC`，让最近改动的文档排在前面。
- **参数**：全部为关键字参数。`tag: str = ""` 标签过滤，空串表示不过滤，非空时要求文档 `tags` 数组中存在完全相等的元素；`status: str = ""` 状态过滤，空串表示不过滤，非空时与 `documents.status` 相等匹配；`page: int = 1` 页码，从 1 开始，必须是正整数（布尔值也被拒绝）；`page_size: int = 20` 每页条数，必须是正整数。
- **返回**：返回 `tuple[list[DocumentRecord], int]`，第一个元素是当前页的 `DocumentRecord` 列表（可能为空列表），第二个元素是满足过滤条件的总行数（`int`）。注意本函数**没有**对 `status` 做枚举校验，传非法状态值只会查不到数据而不会报错。
- **内部流程**：先校验分页参数：`isinstance(page, bool) or not isinstance(page, int) or page < 1` 抛 `ValueError("page must be a positive integer")`，`page_size` 同理抛 `ValueError("page_size must be a positive integer")`。然后构造 `params = (tag, tag, status, status)`（每个值出现两次，分别对应 `? = ''` 与比较条件）。进入 `_connection_scope`，先执行 count 查询取 `["n"]` 作为 `total`，SQL 为带 `json_each` 子查询与 status 条件的 `SELECT count(*) AS n FROM documents WHERE (...) AND (...)`；再执行数据查询，SQL 相同但追加 `ORDER BY updated_at DESC LIMIT ? OFFSET ?`，参数展开为 `(*params, page_size, (page - 1) * page_size)`。最后用列表推导 `[self._decode_document(row) for row in rows]` 与 `int(total)` 组成元组返回。
- **异常/边界**：`page`/`page_size` 非正整数或为布尔值抛 `ValueError`；`tag`/`status` 传 `None` 会作为绑定参数与 `''` 比较失败，可能匹配不到行（本函数不校验）。页码超出总页数时返回空列表但 `total` 仍正确。依赖 SQLite 的 JSON1 扩展（`json_each`），若运行环境缺少该扩展会抛 `sqlite3.OperationalError`。
- **同文件关系**：调用 `_connection_scope` 与 `_decode_document`；不被本文件其它方法调用。

### `DocumentRepository.count_documents(self) -> int` （第 322 行）
- **作用**：返回文档表的总行数，用于总览面板、健康检查或「知识库是否为空」的判断。它与 `list_documents` 返回的 `total` 不同：这里不带任何过滤，是纯粹的全局计数，因此实现更简单、也不需要 JSON1 扩展。它用 `count(*)` 在数据库侧完成计数，而不是把所有行取回再 `len()`，避免大库时把内存撑爆。
- **参数**：无参数（除 `self`）。
- **返回**：返回 `int`，即 `documents` 表的当前总行数，空库时为 `0`。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行 `SELECT count(*) AS n FROM documents` 并 `fetchone()`，取 `["n"]` 字段，用 `int(...)` 包装后直接 `return`。
- **异常/边界**：无特殊处理；表不存在或库不可用时会抛 `sqlite3.OperationalError`。空表返回 `0`，不会返回 `None`。
- **同文件关系**：调用 `_connection_scope`；不被本文件其它方法调用。

### `DocumentRepository.set_status(self, document_id: str, status: str, *, error: str | None = None) -> None` （第 326 行）
- **作用**：只更新一篇文档的状态与错误信息，是流水线各阶段推进的轻量开关。解析完成调 `parsed`、向量化完成调 `vectorized`、抽取完成调 `extracted`、任一步失败调 `failed` 并附带 `error` 说明。把它单独做成一个方法而不是让调用方读回整个 `DocumentRecord` 再 upsert，是为了避免并发下「读—改—写」覆盖掉其它字段的更新，也让状态流转成为一次原子 UPDATE。它会同时刷新 `updated_at`，因此列表页按更新时间排序时，刚推进状态的文档会自动冒到最前面。
- **参数**：`document_id: str` 必填，目标文档主键，未做类型校验；`status: str` 必填，必须属于 `DOCUMENT_STATUSES`，否则抛 `ValueError`；`error: str | None = None` 仅关键字，失败原因，默认 `None`，成功时通常传 `None` 把之前的错误清空。
- **返回**：无返回值（`None`）。注意：即使 `document_id` 不存在，UPDATE 影响 0 行也不会报错，调用方无法从返回值判断是否命中。
- **内部流程**：先调用 `self._check_choice("status", status, DOCUMENT_STATUSES)` 做枚举校验；然后进入 `_connection_scope`，执行 `UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE document_id = ?`，参数依次是新的 `status`、`error`、`utc_now().isoformat()`、`document_id`。
- **异常/边界**：非法 `status` 抛 `ValueError`；目标不存在时静默无操作（`rowcount` 为 0，但方法不返回该值）；数据库错误向外抛出。`error` 允许 `None`（列本身可空），传入空串与 `None` 语义由调用方约定，方法本身不区分。
- **同文件关系**：调用 `_check_choice` 与 `_connection_scope`；不被本文件其它方法调用。

### `DocumentRepository.delete_document(self, document_id: str) -> int` （第 334 行）
- **作用**：删除一篇文档及其全部分块，用于用户主动删除或清理失败任务。它先删 `chunks` 再删 `documents`，顺序不能反——虽然表上没有声明外键约束，但先删子表能保证任何中间失败都不会留下「有文档却没分块」的歧义状态，也符合逻辑上的依赖方向。返回值是**被删掉的 chunk 数量**而不是布尔值，让调用方能顺便向用户报告「同时清理了 N 个分块」，也便于上层做审计。整段操作在同一个事务作用域内，`_connection_scope` 的 `with connection:` 保证两次 DELETE 原子提交或一起回滚。因为 `chunks` 上装了 FTS 同步触发器，删除分块时全文索引会自动跟着清理。
- **参数**：`document_id: str` 必填，要删除的文档主键；未做类型校验，不存在时不会报错。
- **返回**：返回 `int`，即被删除的 `chunks` 行数（来自 `cursor.rowcount`）。文档不存在时返回 `0`。
- **内部流程**：进入 `with self._connection_scope() as connection:`，先 `cursor = connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))`，把 `cursor.rowcount` 存入局部变量 `removed`；再执行 `DELETE FROM documents WHERE document_id = ?`（其 rowcount 被忽略）；退出作用域后 `return int(removed)`。此处利用了 `chunks_ad` 触发器自动维护 FTS 索引。
- **异常/边界**：无特殊参数校验；`document_id` 为 `None` 时匹配不到任何行，返回 `0`。事务中途异常时两次 DELETE 一起回滚，`removed` 不会被返回。若 FTS 触发器执行失败（例如虚拟表被手工删除但触发器残留），删除会抛 `sqlite3.OperationalError`。不处理并发删除同一文档的情形，第二次调用返回 `0`。
- **同文件关系**：调用 `_connection_scope`；不被本文件其它方法调用。与 `_initialize_fts` 中创建的 `chunks_ad` 触发器存在隐式协作关系。

### `DocumentRepository.upsert_chunk(self, chunk: ChunkRecord) -> None` （第 343 行）
- **作用**：写入或更新单个分块，适用于流式切分、逐块入库或只补写某一块的场景。校验后它把真正的 SQL 委托给静态方法 `_write_chunk`，从而与批量写入共享同一段 upsert 逻辑，避免两处 SQL 不一致。`ON CONFLICT(chunk_id) DO UPDATE` 让重复入库（例如同一文档重新切分但 chunk_id 稳定）变成幂等覆盖，而不会因为主键冲突中断整批任务。因为 `chunks` 上有插入/更新触发器，这一写操作会同步刷新 FTS 索引，使新块立刻可被关键词检索到。
- **参数**：`chunk: ChunkRecord` 必填，待写入的分块记录；要求 `vector_status` 属于 `CHUNK_VECTOR_STATUSES`（`"pending"`/`"indexed"`/`"failed"`），其余字段不做校验。
- **返回**：无返回值（`None`）。
- **内部流程**：先判断 `not isinstance(chunk, ChunkRecord)` 则抛 `TypeError("chunk must be a ChunkRecord")`；再 `self._check_choice("vector_status", chunk.vector_status, CHUNK_VECTOR_STATUSES)` 校验枚举；随后进入 `with self._connection_scope() as connection:`，调用 `self._write_chunk(connection, chunk)` 执行 upsert。整个校验在开启连接之前完成，避免白开一次事务。
- **异常/边界**：类型错误抛 `TypeError`，非法 `vector_status` 抛 `ValueError`。空 `chunk_id`、`char_end < char_start` 等不做校验，直接写库。若 `text` 为 `None`，因列 `NOT NULL` 抛 `sqlite3.IntegrityError`。
- **同文件关系**：调用 `_check_choice`、`_connection_scope`、`_write_chunk`；不被本文件其它方法调用。

### `DocumentRepository.upsert_chunks(self, chunks: list[ChunkRecord]) -> None` （第 350 行）
- **作用**：批量写入一篇文档的全部分块，是入库主路径上真正被频繁调用的方法。它把「先全部校验、再全部写入」分成两阶段：第一阶段遍历整个列表做类型与枚举检查，任何一个元素不合法就立刻抛异常，此时**一个字节都还没写库**；第二阶段才在同一个事务里逐条执行 upsert。这样设计避免了「写了前 50 块、第 51 块非法导致半成品入库」的脏数据问题。所有写入共享一次连接与一个事务，既减少了开连接与提交的开销，也保证了整篇文档分块要么全成、要么全回滚。
- **参数**：`chunks: list[ChunkRecord]` 必填，分块记录列表；空列表是合法输入，等价于什么都不做；列表内每个元素都必须是 `ChunkRecord` 且 `vector_status` 合法。
- **返回**：无返回值（`None`）。
- **内部流程**：第一个 `for chunk in chunks:` 循环只做校验：`not isinstance(chunk, ChunkRecord)` 抛 `TypeError("chunks must contain ChunkRecord instances")`，`self._check_choice("vector_status", chunk.vector_status, CHUNK_VECTOR_STATUSES)` 校验枚举。校验全部通过后进入 `with self._connection_scope() as connection:`，第二个 `for chunk in chunks:` 循环对每块调用 `self._write_chunk(connection, chunk)`。事务在作用域退出时统一提交。
- **异常/边界**：元素类型错误抛 `TypeError`，枚举非法抛 `ValueError`（都在写入前发生，不会留下部分数据）。写入阶段若某条触发 `sqlite3.IntegrityError`，整个事务回滚，之前已执行的那些 upsert 一并撤销。传入 `None` 会在第一个 `for` 处抛 `TypeError`（不可迭代）。空列表不报错、不写库。大量分块时循环内逐条 execute 性能一般，但没有分批或上限控制。
- **同文件关系**：调用 `_check_choice`、`_connection_scope`、`_write_chunk`；不被本文件其它方法调用。

### `DocumentRepository._write_chunk(connection: sqlite3.Connection, chunk: ChunkRecord) -> None` （第 359 行）
- **作用**：这是分块写入的唯一 SQL 出口，被单条写入与批量写入共同复用。它被声明为 `@staticmethod`，因为除了执行一条预编译好的 upsert 之外不需要访问 `self` 的任何状态；这样既能被实例方法方便地调用，也明确表达「这里没有隐藏状态依赖」。它执行 `INSERT ... ON CONFLICT(chunk_id) DO UPDATE SET`，把分块的七个字段全部写入（冲突时更新除主键外的六个字段），因此重复写同一 chunk_id 是幂等覆盖而非报错。调用方必须自行保证校验已经做过，并且传入的连接处于打开的事务作用域内。
- **参数**：`connection: sqlite3.Connection` 必填，已打开的连接（由 `_connection_scope` 提供）；`chunk: ChunkRecord` 必填，已经通过类型与枚举校验的分块记录。
- **返回**：无返回值（`None`），副作用是插入或更新 `chunks` 表一行，并可能经由触发器更新 `chunks_fts`。
- **内部流程**：单条 `connection.execute(...)`，SQL 为 `INSERT INTO chunks (chunk_id, document_id, chunk_index, char_start, char_end, text, vector_status) VALUES (?, ?, ?, ?, ?, ?, ?)` 加 `ON CONFLICT(chunk_id) DO UPDATE SET document_id=excluded.document_id, chunk_index=excluded.chunk_index, char_start=excluded.char_start, char_end=excluded.char_end, text=excluded.text, vector_status=excluded.vector_status`；参数元组按字段顺序直接取自 `chunk`。没有 `fetchone`/`fetchall`，不读取返回行。
- **异常/边界**：不做任何参数校验（调用方负责），传非 `ChunkRecord` 会在取属性时抛 `AttributeError`。`text` 为 `None` 触发 `NOT NULL` 约束抛 `sqlite3.IntegrityError`；`chunk_id` 为 `None` 同样因主键非空抛错。若 `chunks_fts` 相关触发器存在缺陷，异常会从这里冒出并回滚整个事务。无超时与重试处理。
- **同文件关系**：被 `upsert_chunk` 与 `upsert_chunks` 调用；它不调用本文件其它函数，但写操作会触发 `_initialize_fts` 里创建的 `chunks_ai`/`chunks_au` 触发器。

### `DocumentRepository.get_chunk(self, chunk_id: str) -> ChunkRecord | None` （第 382 行）
- **作用**：按主键取回单个分块，用于向量库命中后回查正文、或在重试索引前确认某块是否仍然存在。返回的是 `_decode_chunk` 还原后的 `ChunkRecord`，因此 `char_start`/`char_end` 可以直接用来在原文里做高亮定位。查不到返回 `None` 而非抛异常，方便调用方处理「向量库里有、但原文块已被删除」这种不一致情况——这是重建投影时必须考虑的场景。
- **参数**：`chunk_id: str` 必填，分块主键；未做类型校验，不存在的 id 返回 `None`。
- **返回**：命中返回 `ChunkRecord`；无匹配行返回 `None`。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行参数化的 `SELECT * FROM chunks WHERE chunk_id = ?` 并 `fetchone()`；退出作用域后用 `self._decode_chunk(row) if row else None` 返回。
- **异常/边界**：无特殊处理；数据库错误向外抛出。空串或非法类型只导致 `None`。`vector_status` 列若存了不在枚举内的值，本函数仍会照原样返回，不做校验。
- **同文件关系**：调用 `_connection_scope` 与 `_decode_chunk`；不被本文件其它方法调用。

### `DocumentRepository.list_chunks(self, document_id: str) -> list[ChunkRecord]` （第 387 行）
- **作用**：取回某篇文档的全部有序分块，是「展示文档分块」「重建该文档向量」「重新抽取」等操作的主要数据来源。SQL 里带 `ORDER BY chunk_index`，保证返回顺序与原文顺序一致——这一点很重要，因为分块顺序错了会导致拼接上下文语义颠倒。它不做分页，返回该文档的所有块；对于被切得很碎的文档，调用方需要自行评估内存占用。返回的是解码后的 `ChunkRecord` 列表，文档不存在或没有分块时返回空列表而不是 `None`，让调用方可以直接迭代。
- **参数**：`document_id: str` 必填，目标文档主键；未做类型校验，不匹配时返回空列表。
- **返回**：返回 `list[ChunkRecord]`，按 `chunk_index` 升序排列；无数据时为空列表。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行 `SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index` 并 `fetchall()`；退出作用域后返回 `[self._decode_chunk(row) for row in rows]`。
- **异常/边界**：无参数校验与异常捕获。`document_id` 为 `None` 时匹配不到行，返回空列表。若某行 `tags` 之类的数据损坏不影响本函数（chunks 表无 JSON 列）。没有数量上限或分页，超大文档可能返回很多行。
- **同文件关系**：调用 `_connection_scope` 与 `_decode_chunk`；不被本文件其它方法调用。

### `DocumentRepository.set_chunk_vector_status(self, chunk_id: str, status: str) -> None` （第 394 行）
- **作用**：更新单个分块的向量化状态，是向量化 worker 标记进度的最小粒度操作。它让「哪些块已经进了向量库、哪些失败需要重试」变得可查询：索引成功写 `indexed`，嵌入接口报错写 `failed`，配合 `chunk_ids(vector_status=...)` 就能捞出重试集合。之所以单独开一个方法而不是复用 `upsert_chunk`，是因为重写整行既多余又可能在并发下覆盖掉别人刚更新的正文或偏移；一条窄 UPDATE 更安全也更便宜。它不做任何时间戳更新，因为 `chunks` 表本身没有 `updated_at` 列。
- **参数**：`chunk_id: str` 必填，目标分块主键，未做类型校验；`status: str` 必填，必须属于 `CHUNK_VECTOR_STATUSES`（`"pending"`/`"indexed"`/`"failed"`），否则抛 `ValueError`。
- **返回**：无返回值（`None`）；目标不存在时静默无操作，调用方无法从返回值判断是否命中。
- **内部流程**：先 `self._check_choice("vector_status", status, CHUNK_VECTOR_STATUSES)` 校验；再进入 `with self._connection_scope() as connection:`，执行 `UPDATE chunks SET vector_status = ? WHERE chunk_id = ?`，参数为 `(status, chunk_id)`。
- **异常/边界**：非法 `status` 抛 `ValueError`；`chunk_id` 不存在时无操作且不报错；数据库错误向外抛出。该 UPDATE 会触发 `chunks_au` 触发器，导致 FTS 索引对该行做一次「删除 + 重新插入」——功能上无害（正文没变），但会带来额外写放大。
- **同文件关系**：调用 `_check_choice` 与 `_connection_scope`；与 `_initialize_fts` 创建的 `chunks_au` 触发器隐式协作；不被本文件其它方法调用。

### `DocumentRepository._fts_query(query: str) -> str` （第 402 行）
- **作用**：把用户的原始查询串转换成安全的 FTS5 短语，是所有关键词检索的必经关口。它的核心动作是给整串加双引号，并把串内已有的双引号翻倍转义，从而让 FTS5 把输入当作**一个完整的短语**而不是查询语法。文档字符串特别强调这不是理论洁癖：像 `abc-123` 这样的裸串会被 FTS5 把 `-` 解析成列语法并抛出 `no such column: 123`，`型号: X200` 里的冒号同理，在这个函数存在之前这类输入会直接导致接口 500。它被声明为 `@staticmethod`，因为不依赖任何实例状态，纯粹是字符串变换。
- **参数**：`query: str` 必填，用户原始查询串；函数本身不校验类型，若传入非字符串，第 411 行的 `.replace` 会抛 `AttributeError`（调用方 `search_keywords` 已提前挡掉非字符串）。
- **返回**：返回 `str`，形如 `"原始内容"` 的合法 FTS5 短语字符串，内部双引号已翻倍。
- **内部流程**：一行表达式 `return '"' + query.replace('"', '""') + '"'`：先把所有 `"` 替换成 `""`（FTS5 里短语内的双引号转义方式），再在两端各补一个 `"`。不做分词、不做大小写处理、不做特殊字符剥离——因为整串被引号包住后，`-`、`:`、`*` 等都只是普通字符。
- **异常/边界**：不做类型校验，非字符串输入抛 `AttributeError`；空串输入会返回 `""`（两个引号），这种空短语交由 SQLite 处理（实际上调用方 `search_keywords` 已用 `query.strip()` 过滤掉空串）。不处理超长输入或超深引号嵌套，但引号翻倍本身已经防止语法注入。
- **同文件关系**：被 `search_keywords` 调用；不被其它方法调用。

### `DocumentRepository.search_keywords(self, query: str, *, limit: int = 10) -> list[tuple[str, float]]` （第 413 行）
- **作用**：在分块上执行 BM25 关键词检索，是混合召回中「关键词一路」的实现，也是嵌入服务不可用时的兜底检索通道。它返回 `(chunk_id, score)` 列表，分数做了符号翻转，使**分数越大越相关**，与向量相似度的方向保持一致，方便上层直接融合排序。查询通过 `chunks_fts MATCH` 走 FTS5 倒排索引，速度远快于 `LIKE '%...%'` 全表扫描；再用 `JOIN chunks c ON c.rowid = chunks_fts.rowid` 把命中的行映射回真实 chunk_id。它有多重防御：FTS 不可用（`fts_tokenizer is None`）、查询不是字符串、查询去掉空白后为空——这三种情况一律返回空列表而不是报错，从而让「没有 FTS 的环境」或「用户没输入」都不会把整条检索链路打断。只有 `limit` 非法时才抛异常。
- **参数**：`query: str` 必填，用户查询串；`limit: int = 10` 仅关键字，返回条数上限，必须是正整数（布尔值也被拒绝），默认 10。
- **返回**：返回 `list[tuple[str, float]]`，每个元素是 `(chunk_id, 分数)`，分数为 `-bm25(...)`，因此越大越相关；按分数降序（SQL 里 `ORDER BY rank`，FTS5 的 `bm25()` 是越小越相关，取负后即越大越相关）。无 FTS 支持、查询为空或没有命中时返回空列表。
- **内部流程**：第一步短路判断：`self.fts_tokenizer is None or not isinstance(query, str) or not query.strip()` 成立则直接 `return []`。第二步校验 `limit`：`isinstance(limit, bool) or not isinstance(limit, int) or limit < 1` 抛 `ValueError("limit must be a positive integer")`。第三步进入 `with self._connection_scope() as connection:`，执行 `SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS rank FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?`，参数为 `(self._fts_query(query), limit)`，并 `fetchall()`。第四步退出作用域后返回列表推导 `[(row["chunk_id"], -float(row["rank"])) for row in rows]`。
- **异常/边界**：`limit` 非法抛 `ValueError`；`query` 为空/非字符串/无 FTS 时静默返回 `[]`，不抛异常。若 `chunks_fts` 表被外部删除但 `fts_tokenizer` 仍非空，MATCH 查询会抛 `sqlite3.OperationalError`（本函数不捕获）。`_fts_query` 的引号转义保证畸形输入不会引发 FTS5 语法错误。分数被 `float()` 转换，`bm25()` 返回 `NULL` 的极端情况下会抛 `TypeError`。
- **同文件关系**：调用 `_connection_scope` 与 `_fts_query`；依赖 `_initialize_fts` 建立的 `chunks_fts` 表与 `self.fts_tokenizer`；不被本文件其它方法调用。

### `DocumentRepository.chunk_counts(self) -> dict[str, int]` （第 429 行）
- **作用**：一次性返回「每个文档各有多少分块」的映射，专为文档列表页设计——列表每一行都要显示块数，如果逐行去 `list_chunks` 再 `len()`，就会变成 N+1 次查询；这里用一条 `GROUP BY` 把全部计数一次拿回，再用字典按 `document_id` 查，代价从 N+1 降到 1。文档注释直接点明了这个用途（`list view needs it per row`）。没有分块的文档不会出现在字典里，调用方取不到键时按 0 处理即可。
- **参数**：无参数（除 `self`）。
- **返回**：返回 `dict[str, int]`，键是 `document_id`，值是该文档的 `chunks` 行数；没有任何分块时返回空字典 `{}`（不是 `None`）。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行 `SELECT document_id, count(*) AS n FROM chunks GROUP BY document_id` 并 `fetchall()`；退出作用域后返回字典推导 `{row["document_id"]: int(row["n"]) for row in rows}`，把 SQLite 返回的整数显式转成 Python `int`。
- **异常/边界**：无参数可校验；数据库错误向外抛出。空表返回 `{}`。该查询会扫描整张 `chunks` 表（无 WHERE），超大库时开销随分块总数增长；`document_id` 索引 `idx_chunks_document` 对 `GROUP BY` 有一定帮助。
- **同文件关系**：调用 `_connection_scope`；与 `stats`、`list_all_chunks` 同属报表区块但互不调用。

### `DocumentRepository.chunk_ids(self, *, vector_status: str = "") -> list[str]` （第 438 行）
- **作用**：返回分块 id 列表，可按向量化状态过滤，是「对账（reconcile）」流程的输入来源。典型用法是拿 `vector_status="pending"` 的 id 去补做索引、拿 `"failed"` 的去重试、或拿全量 id 与向量库里实际存在的 id 做差集，从而发现孤儿向量并清理。它只返回 id 而不是完整记录，因为对账阶段通常只需要标识符，避免把大量正文读进内存。结果按 `chunk_id` 排序，保证同一状态下多次调用返回顺序稳定，便于比较与去重。
- **参数**：`vector_status: str = ""` 仅关键字，过滤条件，默认空串表示不过滤、返回全部；非空时必须属于 `CHUNK_VECTOR_STATUSES`（`"pending"`/`"indexed"`/`"failed"`），否则抛 `ValueError`。
- **返回**：返回 `list[str]`，按 `chunk_id` 升序排列；无匹配时为空列表。
- **内部流程**：先判断 `if vector_status:`，为真则调用 `self._check_choice("vector_status", vector_status, CHUNK_VECTOR_STATUSES)` 校验。进入 `with self._connection_scope() as connection:` 后按条件分支：`vector_status` 非空时执行 `SELECT chunk_id FROM chunks WHERE vector_status = ? ORDER BY chunk_id` 并传参；否则执行不带 WHERE 的 `SELECT chunk_id FROM chunks ORDER BY chunk_id`。最后返回 `[row["chunk_id"] for row in rows]`。
- **异常/边界**：非法 `vector_status` 抛 `ValueError`；空串合法（走全量分支）。注意本函数用 `if vector_status:` 判断真假，因此传入 `None` 会被当作「不过滤」而不是报错。无分页与上限，全量模式下可能返回非常大的列表。
- **同文件关系**：调用 `_check_choice` 与 `_connection_scope`；不被本文件其它方法调用。

### `DocumentRepository.list_all_chunks(self) -> list[ChunkRecord]` （第 453 行）
- **作用**：按文档与块序返回整库所有分块，用于「全量重建投影」——当嵌入模型或图结构变更时，需要把所有块重新算一遍，此时就需要一次性遍历完整语料。与 `chunk_ids` 不同，它返回完整的 `ChunkRecord`（含正文与字符区间），因为重建时既要用正文算向量，也要用偏移更新图里的引用。排序是 `ORDER BY document_id, chunk_index`，让同一文档的块连续排列且内部有序，便于调用方按文档分批处理、或顺序拼接上下文。它不做分页，调用方在大库场景下需要考虑内存与耗时。
- **参数**：无参数（除 `self`）。
- **返回**：返回 `list[ChunkRecord]`，先按 `document_id` 升序、同文档内按 `chunk_index` 升序排列；无数据时为空列表。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行 `SELECT * FROM chunks ORDER BY document_id, chunk_index` 并 `fetchall()`；退出作用域后返回 `[self._decode_chunk(row) for row in rows]`。
- **异常/边界**：无参数校验；数据库错误向外抛出。空表返回 `[]`。全表读取无上限，超大语料可能造成明显内存占用与长事务。`document_id` 为字符串排序，因此是字典序而非入库顺序。
- **同文件关系**：调用 `_connection_scope` 与 `_decode_chunk`；不被本文件其它方法调用。

### `DocumentRepository.get_embedding_lock(self) -> EmbeddingLockRecord | None` （第 462 行）
- **作用**：读取「本库绑定哪个嵌入空间」的单行锁记录，用于启动自检与写入前比对。典型用法是：算出当前配置的 `(model, dimension)` 后与本函数结果比较，一致才允许往向量库写入；不一致则说明需要全量重建，避免新旧向量混用导致相似度失真。因为 `embedding_lock` 表用 `CHECK (id = 1)` 强制最多一行，这里按 `id = 1` 精确取即可。全新库还没写过锁时返回 `None`，表示「尚未锁定」，此时通常允许首次写入并由 `set_embedding_lock` 落锁。
- **参数**：无参数（除 `self`）。
- **返回**：有锁记录时返回 `EmbeddingLockRecord`（字段 `model`、`dimension`、`updated_at`）；表内无 `id = 1` 行时返回 `None`。所有字段都用 `str()`/`int()` 做了显式类型转换，`updated_at` 为 `NULL` 时用 `or ""` 兜成空串。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行 `SELECT model, dimension, updated_at FROM embedding_lock WHERE id = 1` 并 `fetchone()`。退出作用域后判断 `row is None`，是则 `return None`；否则构造并返回 `EmbeddingLockRecord(model=str(row["model"]), dimension=int(row["dimension"]), updated_at=str(row["updated_at"] or ""))`。
- **异常/边界**：无参数校验；数据库错误向外抛出。若 `dimension` 列被外部写成非数字文本，`int(...)` 会抛 `ValueError`。`updated_at` 为 `NULL` 时安全降级为空串（尽管该列声明为 `NOT NULL`）。
- **同文件关系**：调用 `_connection_scope`；构造并返回 `EmbeddingLockRecord`；与 `set_embedding_lock` 成对使用，但互不调用。

### `DocumentRepository.set_embedding_lock(self, model: str, dimension: int) -> EmbeddingLockRecord` （第 475 行）
- **作用**：写入或更新嵌入空间锁，把「本库的向量是哪个模型、多少维」固化下来。它用 `INSERT ... ON CONFLICT(id) DO UPDATE` 实现单行 upsert，因此无论首次落锁还是模型更换后重新落锁，都是同一条 `id = 1` 的记录被覆盖，不会出现多行锁。它会把 `model` 去掉首尾空白后再存（`model.strip()`），避免配置里的空格导致后续比对莫名不等；`updated_at` 用当前 UTC 时间。返回值直接构造一个新的 `EmbeddingLockRecord`，让调用方无需再查库就能拿到落锁后的权威值。
- **参数**：`model: str` 必填，嵌入模型标识，必须是非空字符串（`strip()` 后非空），否则抛 `ValueError`；`dimension: int` 必填，向量维度，必须是正整数（布尔值被显式排除），否则抛 `ValueError`。
- **返回**：返回新构造的 `EmbeddingLockRecord`，其 `model` 为去空白后的值、`dimension` 为传入值、`updated_at` 为本次写入时间。
- **内部流程**：先校验 `not isinstance(model, str) or not model.strip()` 抛 `ValueError("embedding lock model must be a non-empty string")`；再校验 `isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1` 抛 `ValueError("embedding lock dimension must be a positive integer")`。接着 `now = utc_now().isoformat()`。进入 `with self._connection_scope() as connection:`，执行 `INSERT INTO embedding_lock (id, model, dimension, updated_at) VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET model=excluded.model, dimension=excluded.dimension, updated_at=excluded.updated_at`，参数为 `(model.strip(), dimension, now)`。退出作用域后返回 `EmbeddingLockRecord(model=model.strip(), dimension=dimension, updated_at=now)`。
- **异常/边界**：空/非字符串 `model` 或非正 `dimension` 抛 `ValueError`，均在写库前。传入 `model="  "`（纯空白）同样被拒。`True`/`False` 因被显式排除而抛 `ValueError`（否则 Python 里 `bool` 是 `int` 子类会悄悄通过）。数据库写入失败时异常向外抛出，不返回记录。注意方法不做「与旧锁是否一致」的判断，更换模型时直接覆盖——需要检测变更的调用方应先用 `get_embedding_lock` 比对。
- **同文件关系**：调用 `_connection_scope`；构造并返回 `EmbeddingLockRecord`；与 `get_embedding_lock` 成对使用但互不调用。

### `DocumentRepository.stats(self) -> dict[str, int]` （第 493 行）
- **作用**：汇总知识库的三个核心计数——文档总数、分块总数、已索引分块数，供总览面板、健康检查与「向量化进度」展示使用。`chunks_indexed` 只统计 `vector_status = 'indexed'` 的行，因此 `chunks - chunks_indexed` 天然就是「尚未索引或索引失败」的数量，运维据此判断是否需要触发对账或重试。三次计数都在数据库侧用 `count(*)` 完成，不会把行读进内存；它们在同一个连接作用域内执行，得到的是同一时刻的一致快照。
- **参数**：无参数（除 `self`）。
- **返回**：返回 `dict[str, int]`，固定包含三个键：`"documents"`（文档总数）、`"chunks"`（分块总数）、`"chunks_indexed"`（`vector_status = 'indexed'` 的分块数）。空库时三个值都是 `0`。
- **内部流程**：进入 `with self._connection_scope() as connection:`，依次执行三条查询并各取 `fetchone()["n"]`：`SELECT count(*) AS n FROM documents` 存入 `documents`；`SELECT count(*) AS n FROM chunks` 存入 `chunks`；`SELECT count(*) AS n FROM chunks WHERE vector_status = 'indexed'` 存入 `indexed`。退出作用域后返回 `{"documents": int(documents), "chunks": int(chunks), "chunks_indexed": int(indexed)}`，三个值都显式转成 `int`。
- **异常/边界**：无参数校验；数据库错误向外抛出。空库返回全 0，不会出现缺失键。`vector_status` 为 `NULL` 或其它值的行不计入 `chunks_indexed`。没有缓存，频繁调用会产生三次查询。
- **同文件关系**：调用 `_connection_scope`；不被本文件其它方法调用（`chunk_counts` 是另一种粒度的报表，二者互不调用）。

### `DocumentRepository.close(self) -> None` （第 502 行）
- **作用**：释放仓储持有的固定连接，用于应用关停、测试收尾或需要重开数据库的场景。它只对「自己持有的连接」做关闭——也就是 `:memory:` 库的连接或外部注入的连接；对文件库而言 `self._connection` 本来就是 `None`，因为每次操作都临时开关连接，所以本方法是空操作。关闭后把 `self._connection` 重置为 `None`，这会让后续的 `_connect` 转为按路径新建连接：对文件库来说等于「重新打开」，对 `:memory:` 来说则会得到一个全新的空库（因为内存库随连接销毁）。整个过程在 `self._lock` 保护下进行，避免与正在执行查询的线程冲突。
- **参数**：无参数（除 `self`）。
- **返回**：无返回值（`None`）。可重复调用，第二次及以后因 `self._connection` 已是 `None` 而什么都不做（幂等）。
- **内部流程**：进入 `with self._lock:`，判断 `if self._connection is not None:`，成立则调用 `self._connection.close()` 并 `self._connection = None`；不成立则直接结束。注意这里没有使用 `_connection_scope`，因为不需要开事务，也避免在关闭时又去 `_connect` 新建连接。
- **异常/边界**：若连接已被外部关闭，重复 `close()` 在 SQLite 上通常不报错；若关闭时有未提交事务，未提交的改动会丢失。关闭后继续调用查询方法：文件库会自动重开新连接（数据仍在），`:memory:` 库则相当于换了一个空库（原有数据不可见，但表结构会在下次 `_connect` 后……实际上 `_initialize` 不会重跑，因此后续查询可能报 `no such table`）——这是使用时需要留意的边界。不处理多线程同时调用 `close` 与查询的语义竞争，仅靠锁串行化。
- **同文件关系**：调用 `self._lock` 与 `self._connection`；不调用本文件其它方法；与 `_connect` 通过 `self._connection` 状态间接耦合。

### `DocumentRepository.create_ingest_job(self, text: str, *, kind: str = "sentence", event_at: str = "") -> IngestJobRecord` （第 509 行）
- **作用**：把一句话入库任务写进队列表并立刻返回，是「提交即返回、后台慢慢入库」这条异步链路的入口。它只做一件事：生成唯一 `job_id`、填好初始状态 `pending`、`attempts = 0`、`error` 与 `result` 为空串，然后 INSERT 落库；真正耗时的分块、向量化、抽取由后台 worker 之后去做。因为任务持久化在 SQLite 里，进程重启后仍能被 worker 重新捞起（配合 `restart_stale_ingest_jobs` 与 `list_ingest_jobs(status="pending")`）。返回完整的 `IngestJobRecord` 让 Web 层可以马上把 `job_id` 与创建时间回给前端，用于轮询状态。`event_at` 与 `created_at` 分开：前者是这句话描述的事件时间，后者是任务创建时间。
- **参数**：`text: str` 必填，待入库的一句话正文，必须是非空字符串（`strip()` 后非空），否则抛 `ValueError`；`kind: str = "sentence"` 仅关键字，任务种类，默认 `"sentence"`，不做枚举校验（数据库列也无约束）；`event_at: str = ""` 仅关键字，事件发生时间，默认空串。
- **返回**：返回新建的 `IngestJobRecord`，其 `job_id` 形如 `job_<32位十六进制>`、`status` 为默认的 `"pending"`、`attempts` 为 0、`error` 与 `result` 为空串、`created_at` 与 `updated_at` 均为本次 `utc_now().isoformat()`、`text`/`kind`/`event_at` 为传入值。
- **内部流程**：先校验 `not isinstance(text, str) or not text.strip()` 抛 `ValueError("ingest job text must be a non-empty string")`。然后 `now = utc_now().isoformat()`，构造 `record = IngestJobRecord(job_id=f"job_{uuid4().hex}", text=text, event_at=event_at, kind=kind, created_at=now, updated_at=now)`（`status`、`attempts`、`error`、`result` 走数据类默认值）。进入 `with self._connection_scope() as connection:`，执行 `INSERT INTO ingest_jobs (job_id, kind, text, event_at, status, attempts, error, result, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', 0, '', '', ?, ?)`，参数为 `(record.job_id, record.kind, record.text, record.event_at, now, now)`——状态与计数直接写在 SQL 字面量里。退出作用域后返回 `record`。
- **异常/边界**：非字符串或纯空白 `text` 抛 `ValueError`（注意：`text` 不会被 `strip()` 后再存，前导/尾随空白原样保留）。`kind` 与 `event_at` 不校验，传 `None` 会因列 `NOT NULL` 抛 `sqlite3.IntegrityError`（`event_at` 列声明为 `NOT NULL DEFAULT ''`）。`job_id` 用 UUID4 生成，碰撞概率可忽略。数据库错误向外抛出。
- **同文件关系**：调用 `_connection_scope`；构造并返回 `IngestJobRecord`；与 `get_ingest_job`、`set_ingest_job_status`、`list_ingest_jobs`、`restart_stale_ingest_jobs`、`reset_failed_ingest_job` 共同构成一句话入库队列，但彼此不直接调用（只有 `reset_failed_ingest_job` 会调 `get_ingest_job`）。

### `DocumentRepository.get_ingest_job(self, job_id: str) -> IngestJobRecord | None` （第 532 行）
- **作用**：按 `job_id` 查询单个入库任务的当前状态，是前端轮询「我提交的那句话入库完了吗」的直接支撑。返回的 `IngestJobRecord` 里既有状态（`pending`/`running`/`done`/`failed`）也有 `result` 摘要与 `error` 原因，因此一次查询就能驱动整个结果展示。它把行解码委托给模块级函数 `_ingest_job_from_row`（而不是像文档/分块那样用类内静态方法），这是本文件里唯一采用该写法的解码路径。查不到返回 `None`，让 Web 层自然映射为 404。
- **参数**：`job_id: str` 必填，任务主键；未做类型校验，不存在时返回 `None`。
- **返回**：命中返回 `IngestJobRecord`；无匹配行返回 `None`。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行参数化的 `SELECT * FROM ingest_jobs WHERE job_id = ?` 并 `fetchone()`；退出作用域后返回 `_ingest_job_from_row(row) if row is not None else None`。
- **异常/边界**：无特殊异常处理；数据库错误向外抛出。空串或不存在 id 返回 `None`。若行内 `attempts` 被外部写成非整数文本，`_ingest_job_from_row` 会原样透传（不做 `int()` 转换）。
- **同文件关系**：调用 `_connection_scope` 与模块级 `_ingest_job_from_row`；被 `reset_failed_ingest_job` 调用（在 UPDATE 命中后回读最新记录），也被外部查询接口调用。

### `DocumentRepository.set_ingest_job_status(self, job_id: str, status: str, *, error: str = "", result: str = "") -> bool` （第 539 行）
- **作用**：推进一句话入库任务的状态，是 worker 更新进度的唯一写入口。它有两个精巧之处：一是「移动到 `running` 时同时把 `attempts` 加一」，因此尝试次数不需要调用方自己维护，worker 只要在开工时置 `running` 就自动计数，配合 `attempts` 上限就能实现「失败重试 N 次后放弃」；二是 `error` 与 `result` 采用「非空才覆盖」的语义（SQL 里 `CASE WHEN :error != '' THEN :error ELSE error END`），因此可以在只更新状态时不慎清空已有的错误信息或结果摘要。返回 `bool` 表示是否真的命中了某行，让 worker 能发现「任务已被删除」这类情况，而不是无声地继续跑。
- **参数**：`job_id: str` 必填，目标任务主键；`status: str` 必填，必须属于 `INGEST_JOB_STATUSES`（`"pending"`/`"running"`/`"done"`/`"failed"`），否则抛 `ValueError`；`error: str = ""` 仅关键字，新的失败信息，空串表示保留原值，默认空串；`result: str = ""` 仅关键字，新的结果摘要（通常是 JSON 文本），空串表示保留原值，默认空串。
- **返回**：返回 `bool`，`True` 表示 UPDATE 影响了至少一行（任务存在且被更新），`False` 表示没有匹配的 `job_id`（任务不存在）。
- **内部流程**：先 `self._check_choice("status", status, INGEST_JOB_STATUSES)` 校验枚举；再 `now = utc_now().isoformat()`。进入 `with self._connection_scope() as connection:`，执行带命名参数的 UPDATE：`SET status = :status, error = CASE WHEN :error != '' THEN :error ELSE error END, result = CASE WHEN :result != '' THEN :result ELSE result END, attempts = attempts + :attempt, updated_at = :now WHERE job_id = :job_id`，参数字典为 `{"status": status, "error": error, "result": result, "attempt": 1 if status == "running" else 0, "now": now, "job_id": job_id}`。退出作用域后返回 `cursor.rowcount > 0`。
- **异常/边界**：非法 `status` 抛 `ValueError`；`job_id` 不存在时返回 `False` 且不报错。因为 `error`/`result` 用「非空才覆盖」，**无法通过本方法把已有的 error 或 result 主动清空**——需要清空时必须走 `reset_failed_ingest_job`（它直接 `SET error = ''`、`result = ''`）。非 `running` 的状态不会增加 `attempts`，所以「失败重试计数」只在置 `running` 时累加，若 worker 直接置 `failed` 而不经过 `running`，计数不会增长。`error`/`result` 传 `None` 时 SQL 里 `NULL != ''` 为真，会写入 `NULL`，而两列声明为 `NOT NULL DEFAULT ''`，从而抛 `sqlite3.IntegrityError`。
- **同文件关系**：调用 `_check_choice` 与 `_connection_scope`；与 `create_ingest_job`、`get_ingest_job`、`list_ingest_jobs`、`restart_stale_ingest_jobs`、`reset_failed_ingest_job` 同属队列区块，但互不调用。

### `DocumentRepository.list_ingest_jobs(self, *, status: str | None = None, limit: int = 50) -> list[IngestJobRecord]` （第 573 行）
- **作用**：列出入库任务，可按状态过滤，是「历史记录页」与「worker 领任务」共用的查询。历史页调用时不传 `status`，拿到最近的任务列表（含结果摘要与失败原因）直接展示；worker 则传 `status="pending"` 捞出待处理任务。它支持动态拼 WHERE 子句：只有传入 `status` 时才追加 `WHERE status = ?`，参数放进 `params` 列表按顺序绑定，既避免 SQL 注入也避免多余的占位符。排序是 `ORDER BY created_at DESC, rowid DESC`——先按创建时间倒序，再按 `rowid` 倒序作为稳定兜底，这样同一秒内创建的多条任务也能保持确定的先后顺序，不会因为时间戳精度相同而乱序。
- **参数**：`status: str | None = None` 仅关键字，状态过滤，默认 `None` 表示不过滤；非 `None` 时必须属于 `INGEST_JOB_STATUSES`，否则抛 `ValueError`。`limit: int = 50` 仅关键字，返回条数上限，默认 50；内部会做 `max(1, min(int(limit), 200))` 的夹取，因此小于 1 会被抬到 1、大于 200 会被压到 200。
- **返回**：返回 `list[IngestJobRecord]`，按 `created_at` 倒序（同秒按 `rowid` 倒序）；无数据时为空列表。列表长度不超过夹取后的 `limit`。
- **内部流程**：先判断 `if status is not None:`，成立则 `self._check_choice("status", status, INGEST_JOB_STATUSES)`。初始化 `query = "SELECT * FROM ingest_jobs"` 与空列表 `params: list[Any] = []`。若 `status is not None`，把 `query` 追加 `" WHERE status = ?"` 并把 `status` 加入 `params`。接着追加 `" ORDER BY created_at DESC, rowid DESC LIMIT ?"`，并把 `max(1, min(int(limit), 200))` 加入 `params`。进入 `with self._connection_scope() as connection:`，执行 `connection.execute(query, params).fetchall()`；退出作用域后返回 `[_ingest_job_from_row(row) for row in rows]`。
- **异常/边界**：`status` 非 `None` 且非法时抛 `ValueError`。`limit` 的处理较宽松：非整数（如字符串）会在 `int(limit)` 处抛 `ValueError`，`None` 会抛 `TypeError`，布尔值 `True` 会被 `int(True)=1` 静默接受。`limit` 越界被夹取而不报错。数据库错误向外抛出。
- **同文件关系**：调用 `_check_choice`、`_connection_scope` 与模块级 `_ingest_job_from_row`；不被本文件其它方法调用。

### `DocumentRepository.restart_stale_ingest_jobs(self) -> int` （第 587 行）
- **作用**：这是崩溃恢复的清理动作：把库里所有卡在 `running` 的任务重置回 `pending`，让它们在本次进程启动后能被 worker 重新领走。之所以需要它，是因为任务状态是持久化的——如果进程在处理某条任务时被强杀或崩溃，那条任务会永远停在 `running`，既没人继续做也不会被 `pending` 的查询捞到，成为僵尸任务。它在应用启动时调用一次即可。返回受影响行数，便于日志里记录「上次异常退出遗留了 N 条任务」。注意它不增加 `attempts`，因此反复崩溃不会导致任务被重试计数淘汰。
- **参数**：无参数（除 `self`）。
- **返回**：返回 `int`，即被从 `running` 改回 `pending` 的任务条数（来自 `cursor.rowcount`）；没有僵尸任务时返回 `0`。
- **内部流程**：进入 `with self._connection_scope() as connection:`，执行 `UPDATE ingest_jobs SET status = 'pending', updated_at = ? WHERE status = 'running'`，参数为 `utc_now().isoformat()`；退出作用域后 `return cursor.rowcount`。
- **异常/边界**：无参数可校验；数据库错误向外抛出。该方法「无条件」地把所有 `running` 任务重置——如果同一个库被多个进程同时使用，新进程启动时会误伤另一个进程正在跑的任务（本模块没有加租约或心跳机制来区分）。不修改 `attempts`、`error`、`result`，因此上次失败留下的错误信息与摘要会被保留，直到任务重新跑完覆盖。
- **同文件关系**：调用 `_connection_scope`；与 `reset_failed_ingest_job` 是互补的恢复手段（一个处理 `running` 僵尸，一个处理 `failed` 重试），但二者互不调用。

### `DocumentRepository.reset_failed_ingest_job(self, job_id: str) -> IngestJobRecord | None` （第 597 行）
- **作用**：这是「用户手动重试」的入口：把一条**失败**任务重置回 `pending` 并清空现场（`error`、`result` 归零，`attempts` 清零），让它像新任务一样重新排队。它特意在 WHERE 里加了 `AND status = 'failed'` 这个守卫，因此只能对失败任务生效——不能把正在跑的任务打断、也不能把已成功的任务重跑，避免用户误操作造成重复入库或状态倒退。清空 `attempts` 是为了让重试次数重新计算，否则一个曾多次失败的任务会立刻又耗尽重试额度。返回重置后的记录（重新查库读取，保证返回值与库内真实状态一致）或 `None`（表示任务不存在或状态不是 `failed`），调用方据此返回 404 或 409。
- **参数**：`job_id: str` 必填，目标任务主键；未做类型校验，不存在或状态不符时返回 `None`。
- **返回**：重置成功时返回重新查库得到的 `IngestJobRecord`（`status` 为 `"pending"`、`error` 与 `result` 为空串、`attempts` 为 0、`updated_at` 为本次时间）；若没有匹配行（任务不存在，或存在但状态不是 `failed`）返回 `None`。
- **内部流程**：先 `now = utc_now().isoformat()`。进入 `with self._connection_scope() as connection:`，执行 `UPDATE ingest_jobs SET status = 'pending', error = '', result = '', attempts = 0, updated_at = ? WHERE job_id = ? AND status = 'failed'`，参数为 `(now, job_id)`；随后 `if cursor.rowcount == 0: return None`（此时在 `with` 块内部直接返回，`_connection_scope` 的 `finally` 仍会正常提交/关闭）。若命中则退出作用域，最后 `return self.get_ingest_job(job_id)` 重新查询并返回最新记录。
- **异常/边界**：无参数校验；数据库错误向外抛出。任务不存在、或状态为 `pending`/`running`/`done` 时都返回 `None`（三种情况不做区分，调用方无法从返回值分辨原因）。理论上存在极小竞态：UPDATE 成功后、`get_ingest_job` 之前任务被别的进程改状态或删除，此时可能返回被改后的状态或 `None`。这里是本文件唯一「一个公开方法调用另一个公开方法」的地方，因为 `RLock` 可重入，不会死锁。
- **同文件关系**：调用 `_connection_scope` 与 `self.get_ingest_job`；与 `restart_stale_ingest_jobs`、`set_ingest_job_status` 同属队列恢复区块，但只调用 `get_ingest_job`。

### `DocumentRepository._check_choice(field_name: str, value: Any, allowed: tuple[str, ...]) -> None` （第 619 行）
- **作用**：这是全类共用的枚举校验小工具，把「字段值必须在白名单里」这一重复逻辑收敛到一处。它被用在所有状态类字段上：`permission`、文档 `status`、分块 `vector_status`、入库任务 `status`，以及 `chunk_ids` 的过滤参数。之所以要统一校验，是因为这些值最终会写进数据库并驱动业务分支（例如 `vector_status = 'indexed'` 是 `stats` 统计的依据），一旦写进非法值，后续查询会静默漏数据而很难排查。报错信息里带上字段名、允许集合与实际值，方便定位是哪个调用点传错了。它被声明为 `@staticmethod`，不依赖实例状态。
- **参数**：`field_name: str` 必填，出错的字段名，仅用于拼错误消息；`value: Any` 必填，待校验的值，类型不限；`allowed: tuple[str, ...]` 必填，允许取值的元组，本文件里传入 `PERMISSIONS`、`DOCUMENT_STATUSES`、`CHUNK_VECTOR_STATUSES`、`INGEST_JOB_STATUSES` 之一。
- **返回**：无返回值（`None`）；校验通过时静默返回，失败时抛异常。
- **内部流程**：只有一步判断 `if value not in allowed:`，成立则 `raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}")`，用 `!r` 让实际值以 `repr` 形式呈现，便于区分 `""` 与 `None`、`"1"` 与 `1`。不做类型转换、不做大小写归一化、不做去空白处理——传 `"PENDING"` 或 `" pending"` 都会被判为非法。
- **异常/边界**：值不在白名单时抛 `ValueError`，消息包含字段名、允许集合与实际值。若 `value` 是自定义对象且其 `__eq__` 抛异常，异常会从这里冒出（不做捕获）。`allowed` 传空元组时任何值都会失败。无特殊空值处理：`None` 与 `""` 都只是普通的不匹配值。
- **同文件关系**：被 `upsert_document`、`set_status`、`upsert_chunk`、`upsert_chunks`、`set_chunk_vector_status`、`chunk_ids`、`set_ingest_job_status`、`list_ingest_jobs` 调用；它不调用本文件其它函数。

### `DocumentRepository._decode_document(row: sqlite3.Row) -> DocumentRecord` （第 624 行）
- **作用**：把 `documents` 表的一行原始数据还原成 `DocumentRecord` 对象，是「数据库行 → 业务对象」的转换边界。它承担了本文件里唯一的 JSON 反序列化职责：库里的 `tags` 是 `json.dumps` 出来的文本（如 `["财务","2024"]`），这里用 `json.loads` 还原成 Python 列表，让上层能直接迭代标签。它还对 `tags` 做了空值兜底——`json.loads(row["tags"]) if row["tags"] else []`——因为该列虽然有 `NOT NULL DEFAULT '[]'`，但外部工具或旧数据仍可能写入空串，空串会让 `json.loads` 抛异常。它被声明为 `@staticmethod`，因为只需传入行、不需实例状态。`error` 字段原样透传，允许为 `None`。
- **参数**：`row: sqlite3.Row` 必填，来自 `SELECT * FROM documents` 的单行结果；要求连接的 `row_factory` 是 `sqlite3.Row`，因为函数用列名取值。
- **返回**：返回一个字段齐全的 `DocumentRecord` 实例：`document_id`、`title`、`raw_text`、`source`、`tags`（已反序列化的 `list[str]`）、`permission`、`status`、`error`、`created_at`、`updated_at` 全部按列取值填充。
- **内部流程**：单个 `return DocumentRecord(...)` 构造表达式，逐字段用 `row["列名"]` 取值；其中 `tags=json.loads(row["tags"]) if row["tags"] else []` 是唯一带条件的分支；其余字段直接赋值，不做类型转换（SQLite 的 TEXT 列返回 `str`，可空列返回 `None`）。
- **异常/边界**：若 `tags` 列存的是非法 JSON 文本（如 `"[未闭合"`），`json.loads` 抛 `json.JSONDecodeError`（`ValueError` 的子类），本函数不捕获，异常会一路传到调用方。`tags` 为 `None` 或空串时安全返回 `[]`。若 `row` 缺少某列（例如调用方用了自定义 SELECT），`row["列名"]` 会抛 `IndexError`。传入非 `sqlite3.Row`（如普通元组）会抛 `TypeError`。
- **同文件关系**：被 `get_document` 与 `list_documents` 调用；它构造并返回 `DocumentRecord`；不调用本文件其它函数。

### `DocumentRepository._decode_chunk(row: sqlite3.Row) -> ChunkRecord` （第 639 行）
- **作用**：把 `chunks` 表的一行还原成 `ChunkRecord`，与 `_decode_document` 对称，构成「行 → 对象」的第二个转换边界。分块表没有 JSON 列，所以它比文档解码更简单：七个字段逐一映射即可，没有反序列化与兜底分支。它保证了所有读取路径（`get_chunk`、`list_chunks`、`list_all_chunks`）返回的对象结构完全一致，上层不必区分数据来自哪个查询。`char_start`/`char_end` 原样透传，调用方可以据此在 `documents.raw_text` 上切片还原上下文。它同样是 `@staticmethod`。
- **参数**：`row: sqlite3.Row` 必填，来自 `SELECT * FROM chunks` 的单行结果；要求 `row_factory` 为 `sqlite3.Row`。
- **返回**：返回字段齐全的 `ChunkRecord`：`chunk_id`、`document_id`、`chunk_index`、`char_start`、`char_end`、`text`、`vector_status`。
- **内部流程**：单个 `return ChunkRecord(...)` 表达式，七个关键字参数各自用 `row["列名"]` 取值，无任何条件分支、无类型转换、无异常捕获。
- **异常/边界**：不做任何校验与兜底。若 `vector_status` 列存了枚举外的值，会原样进入对象（不做 `_check_choice`）。若 `row` 缺少某列，`row["列名"]` 抛 `IndexError`；传入非 `sqlite3.Row` 抛 `TypeError`。因为 `chunks` 表各列都是 `NOT NULL`，正常路径不会出现 `None`。
- **同文件关系**：被 `get_chunk`、`list_chunks`、`list_all_chunks` 调用；它构造并返回 `ChunkRecord`；不调用本文件其它函数。

### `_ingest_job_from_row(row: sqlite3.Row) -> IngestJobRecord` （第 652 行）
- **作用**：这是模块级（不属于任何类）的行解码函数，把 `ingest_jobs` 表的一行还原成 `IngestJobRecord`。之所以放在模块顶层而不是像文档、分块那样做成类内 `@staticmethod`，是因为它被 `get_ingest_job` 与 `list_ingest_jobs` 使用，而这两个方法的调用点都在类内部——写成模块级函数后无需 `self` 或类名前缀即可直接调用，读起来更短。它保证队列查询返回的对象结构统一：状态、尝试次数、错误信息、结果摘要、时间戳全部就位，前端展示历史记录时无需再做二次查询。它不做任何类型转换与校验，因为 `ingest_jobs` 所有列都有 `NOT NULL DEFAULT`，正常写入路径不会产生 `NULL`。
- **参数**：`row: sqlite3.Row` 必填，来自 `SELECT * FROM ingest_jobs` 的单行结果；要求连接的 `row_factory` 为 `sqlite3.Row`，因为函数用列名取值。
- **返回**：返回字段齐全的 `IngestJobRecord`：`job_id`、`text`、`event_at`、`kind`、`status`、`attempts`、`error`、`result`、`created_at`、`updated_at`。
- **内部流程**：单个 `return IngestJobRecord(...)` 表达式，十个关键字参数各用 `row["列名"]` 取值。注意构造时参数的书写顺序与数据类字段顺序不同（`text`、`event_at`、`kind` 的先后被打乱），但因为全部使用关键字传参，语义不受影响。没有条件分支、没有循环、没有异常处理。
- **异常/边界**：不做校验与兜底。若 `row` 缺少某列抛 `IndexError`，传入非 `sqlite3.Row` 抛 `TypeError`。`attempts` 若被外部写成非整数文本，会原样成为对象属性而不报错。不处理 `row` 为 `None` 的情况——调用方（`get_ingest_job`）已提前用 `if row is not None` 挡掉，`list_ingest_jobs` 则只传入真实行。
- **同文件关系**：被 `DocumentRepository.get_ingest_job` 与 `DocumentRepository.list_ingest_jobs` 调用；它构造并返回 `IngestJobRecord`；不调用本文件其它函数。它是 `__all__` 之外的私有模块级辅助函数，不对外导出。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `DocumentRecord` | 一篇已入库文档的内存记录，承载标题、归一化全文、来源、标签、权限与状态等全部字段。 |
| `ChunkRecord` | 一个分块及其在原文中的字符区间，并带向量化状态以支持断点续跑与重试。 |
| `EmbeddingLockRecord` | 不可变的单行锁记录，声明本 SQLite 文件绑定的嵌入模型与向量维度。 |
| `IngestJobRecord` | 一条持久化的一句话后台入库任务，含状态、尝试次数、错误与结果摘要。 |
| `DocumentRepository` | 四张语料相关表在共享 memory SQLite 上的完整仓储，负责建表、CRUD、FTS 检索、统计与入库队列。 |
| `DocumentRepository.__init__` | 规范化路径、创建目录、初始化锁与连接策略，并立即建表建索引建 FTS。 |
| `DocumentRepository._connect` | 按需返回连接：固定连接直接复用，文件库每次新建并统一设置 `sqlite3.Row`。 |
| `DocumentRepository._connection_scope` | 加锁、开事务、交出连接、自动提交或回滚并关闭临时连接的上下文管理器。 |
| `DocumentRepository._initialize` | 幂等创建 `documents`/`chunks`/`ingest_jobs`/`embedding_lock` 四表及其索引，并调用 FTS 初始化。 |
| `DocumentRepository._initialize_fts` | 建 FTS5 外部内容虚拟表与三个同步触发器，按 trigram→unicode61 降级并回填已有数据。 |
| `DocumentRepository.upsert_document` | 校验后插入或更新文档，更新时保留原始 `created_at`，`tags` 序列化为 JSON。 |
| `DocumentRepository.get_document` | 按主键取回单篇文档并解码，不存在返回 `None`。 |
| `DocumentRepository.list_documents` | 按标签与状态过滤、分页返回文档列表，并同时给出忽略分页的总数。 |
| `DocumentRepository.count_documents` | 返回文档表总行数。 |
| `DocumentRepository.set_status` | 单条 UPDATE 推进文档状态与错误信息并刷新 `updated_at`。 |
| `DocumentRepository.delete_document` | 事务内先删分块再删文档，返回被清理的分块数量。 |
| `DocumentRepository.upsert_chunk` | 校验单个分块后调用 `_write_chunk` 写入或更新。 |
| `DocumentRepository.upsert_chunks` | 先整体校验再在同一事务内批量写入分块，保证全成或全回滚。 |
| `DocumentRepository._write_chunk` | 分块写入的唯一 SQL 出口，执行 `ON CONFLICT(chunk_id)` 的幂等 upsert。 |
| `DocumentRepository.get_chunk` | 按主键取回单个分块并解码，不存在返回 `None`。 |
| `DocumentRepository.list_chunks` | 按 `chunk_index` 顺序返回某文档的全部分块。 |
| `DocumentRepository.set_chunk_vector_status` | 更新单个分块的向量化状态，供索引 worker 标记进度与失败。 |
| `DocumentRepository._fts_query` | 把用户原始查询转成安全 FTS5 短语，引号包裹并把内部引号翻倍。 |
| `DocumentRepository.search_keywords` | 用 BM25 在分块上做关键词检索，返回按相关度降序的 `(chunk_id, 分数)`。 |
| `DocumentRepository.chunk_counts` | 一条 `GROUP BY` 查询返回 `{document_id: 块数}`，避免列表页 N+1 查询。 |
| `DocumentRepository.chunk_ids` | 返回分块 id 列表，可按 `vector_status` 过滤，供对账与重试使用。 |
| `DocumentRepository.list_all_chunks` | 按文档与块序返回整库所有分块，供全量重建投影使用。 |
| `DocumentRepository.get_embedding_lock` | 读取单行嵌入空间锁，未落锁时返回 `None`。 |
| `DocumentRepository.set_embedding_lock` | 校验后 upsert 单行嵌入空间锁，并把去空白后的模型名与新记录返回。 |
| `DocumentRepository.stats` | 汇总文档数、分块数与已索引分块数三个计数。 |
| `DocumentRepository.close` | 在锁保护下关闭并清空持有的固定连接，可重复调用。 |
| `DocumentRepository.create_ingest_job` | 生成 `job_<uuid>` 并以 `pending` 状态把一句话入库任务落库后立即返回。 |
| `DocumentRepository.get_ingest_job` | 按 `job_id` 查询单条入库任务，不存在返回 `None`。 |
| `DocumentRepository.set_ingest_job_status` | 推进任务状态，置 `running` 时自增尝试次数，`error`/`result` 非空才覆盖，返回是否命中。 |
| `DocumentRepository.list_ingest_jobs` | 按状态可选过滤、按创建时间倒序列出入库任务，`limit` 夹取在 1–200。 |
| `DocumentRepository.restart_stale_ingest_jobs` | 把崩溃遗留的 `running` 任务重置为 `pending`，返回重置条数。 |
| `DocumentRepository.reset_failed_ingest_job` | 仅对 `failed` 任务清空错误与计数并回到 `pending`，成功后回读返回最新记录。 |
| `DocumentRepository._check_choice` | 通用枚举白名单校验，失败时抛带字段名与允许集合的 `ValueError`。 |
| `DocumentRepository._decode_document` | 把 `documents` 行还原为 `DocumentRecord`，并反序列化 `tags` JSON。 |
| `DocumentRepository._decode_chunk` | 把 `chunks` 行还原为 `ChunkRecord`，逐字段直传不做转换。 |
| `_ingest_job_from_row` | 模块级辅助函数，把 `ingest_jobs` 行还原为 `IngestJobRecord`。 |
