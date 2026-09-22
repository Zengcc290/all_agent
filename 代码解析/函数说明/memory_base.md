# memory/base.py

## 一、这个文件是干什么的

这个文件是四层记忆系统的「地基」：它定义了贯穿整个记忆子系统的数据结构（记忆类型枚举、记忆条目、检索结果、运行时配置），以及所有记忆类型共用的 `BaseMemory` 基类实现。

它刻意不依赖任何具体存储厂商：`MemoryItem` 既可以在 SQLite、Qdrant 还是自研后端之间流转，而不需要改动上层业务代码；`BaseMemory` 则把增删改查（CRUD）与语义检索这两类通用操作一次写好，供工作记忆、情景记忆、语义记忆、感知记忆四种记忆类型继承复用。

文件里还收拢了两个关键的「构造工厂」：`default_sqlite_path()` 负责确定默认持久化路径（唯一保留的环境变量覆盖入口），`make_default_embedding()` 负责按配置决定使用云端 OpenAI 兼容嵌入服务还是离线确定性哈希嵌入——它在云端配置不全时选择静默退回离线兜底而不是抛异常，从而保证记忆层、Agent 工具和 Web 应用在完全离线时依然可用。

配置部分由 `MemoryConfig` 承担：它集中描述 SQLite 路径、TTL、工作记忆容量、检索条数、相似度阈值、嵌入模型/端点/密钥/维度/超时/批大小，以及 Qdrant、Neo4j、代理等连接开关；`from_config()` 从 `config/services.toml` 补齐未显式指定的字段，形成「显式构造参数 > services.toml」的两层优先级。

序列化辅助函数 `_json_safe` / `_json_restore` 负责把 bytes、Mapping、集合等不可直接 JSON 化的负载安全地转成可落库形式并能还原回来，`utc_now` / `ensure_datetime` 则统一了全文件的时间语义（一律 UTC、一律时区感知）。

对外暴露的接口由文件末尾的 `__all__` 声明，因此上层的 manager、存储实现、Agent 工具与 Web 路由都可以只 import 这一个模块就拿到全部公共数据结构与基类。

---

## 二、函数与类逐条详解

### `class MemoryType(StrEnum)` （第 54 行）

- **作用**：定义记忆系统支持的四种记忆类型的枚举，是整个记忆层的「类型标签」总开关。它继承 `StrEnum`，意味着每个成员的取值本身就是字符串（`"working"`、`"episodic"`、`"semantic"`、`"perceptual"`），既可以直接参与字符串比较、拼接到日志里，也能直接序列化成 JSON 字段。四种类型分别对应工作记忆（短时、有 TTL 的临时上下文）、情景记忆（带时间戳的经历事件）、语义记忆（沉淀下来的事实知识，可关联知识图谱）和感知记忆（带图像等多模态负载的感知片段）。`BaseMemory` 通过 `memory_type` 属性来隔离各类型的数据：同一条记录只能属于一种类型，跨类型的读取会被拒绝。数据库中该字段以 `.value` 的字符串形式存储，反序列化时再通过 `MemoryType(...)` 转回枚举，从而保证类型安全。
- **参数**：无（枚举类，不接收构造参数；成员为 `WORKING`、`EPISODIC`、`SEMANTIC`、`PERCEPTUAL`）。
- **返回**：无（类定义本身；实例化时返回对应的枚举成员）。
- **内部流程**：声明四个成员并绑定字符串值；由于继承 `StrEnum`，成员可直接当作字符串使用。文件内 `MemoryItem.__post_init__` 调用 `MemoryType(self.memory_type)` 把传入的字符串或枚举统一归一化为枚举成员；`BaseMemory.__init__` 也用同样方式归一化自身类型；`add`、`get`、`delete`、`list`、`clear` 等方法用 `self.memory_type` 做过滤；`add` 中还会用 `existing.memory_type != self.memory_type` 判断 id 冲突，`get` 中通过 `item.memory_type == self.memory_type` 判断归属。
- **异常/边界**：本身不抛异常；但 `MemoryType(非法字符串)` 会抛出 `ValueError`，这是 `MemoryItem.__post_init__` 与 `BaseMemory.__init__` 校验非法类型值的实际来源。
- **同文件关系**：被 `MemoryItem`（字段类型与归一化）、`BaseMemory.__init__`、`BaseMemory.add`、`BaseMemory.get`、`BaseMemory.delete`、`BaseMemory.search`、`BaseMemory.list`、`BaseMemory.clear` 引用；不被本文件任何函数调用其方法。

### `default_sqlite_path() -> str` （第 61 行）

- **作用**：给出面向 Agent 的记忆工具默认使用的持久化 SQLite 文件路径。它是整个项目里唯一保留的路径覆盖入口，做了「单点收拢」：只要设置环境变量 `MEMORY_DB_PATH`，所有没有自己注入 manager 的调用方都会跟着改到新路径。未设置时返回项目根目录下的 `memory.sqlite3`（文件名来自 `DEFAULT_MEMORY_DB_FILENAME` 常量），也就是与 `memory/` 包的父目录同级。之所以放在 `memory/base.py` 而不是各调用点，是为了避免路径拼接逻辑在多处重复、出现「有的写相对路径、有的写绝对路径」的不一致。它返回的是字符串而非 `Path`，方便直接塞进 `MemoryConfig.sqlite_path`（该字段接受 `str | Path`）或传给 SQLite 驱动。
- **参数**：无参数。
- **返回**：`str`。若环境变量 `MEMORY_DB_PATH` 有值（非空字符串）则原样返回其值；否则返回 `Path(__file__).resolve().parent.parent / DEFAULT_MEMORY_DB_FILENAME` 的字符串形式，即项目根目录下的默认数据库文件名。
- **内部流程**：先用 `os.getenv("MEMORY_DB_PATH")` 读取环境变量，若结果非空（真值判断，空串会被跳过）则直接返回；否则用 `Path(__file__).resolve()` 取得本文件绝对路径，`parent` 得到 `memory/` 目录，再 `parent` 得到项目根目录，用 `/` 运算符拼接 `DEFAULT_MEMORY_DB_FILENAME`，最后 `str()` 转成字符串返回。
- **异常/边界**：无特殊处理。`Path(__file__)` 在正常模块导入场景下始终有效；环境变量为空串时按未设置处理，回退到项目默认路径。
- **同文件关系**：不调用本文件任何函数；本文件内也没有其它函数调用它（`MemoryConfig` 的默认值直接用的是常量 `MEMORY_SQLITE_DEFAULT`），它是给外部调用方（Agent 工具、Web 应用装配层）使用的公共入口，并通过 `__all__` 导出。

### `make_default_embedding(config: MemoryConfig | None = None) -> BaseEmbedding` （第 71 行）

- **作用**：根据配置构建默认的嵌入（embedding）服务实例，是记忆层「文本转向量」能力的装配工厂。它实现了一个三级选择策略：配置里显式要求 `hash` 时强制走离线兜底；否则如果云端端点（`[embedding].base_url`）和 `api_key` 都配齐了，就启用 OpenAI 兼容的 `APIEmbedding`（同时支持纯文本与图文 VL 输入）；两者都不满足时，不抛异常，而是退回确定性的离线 `HashEmbedding`，让记忆层、Agent 工具和 Web 应用在完全离线环境下依然可跑。这样设计的另一个好处是 `/api/health` 的 `embedding_mode` 能如实显示为 hash，运维一眼能看出当前实际用的是哪种嵌入。文件注释还特别提醒：云端向量空间之间不可互换，换模型必须重建投影（维度的唯一开关是 `[embedding].model`，配套 `QdrantVectorStore` 的维度守卫与 `scripts/migrate_to_cloud.py`）。
- **参数**：
  - `config: MemoryConfig | None`，默认 `None`。为 `None` 时内部调用 `MemoryConfig.from_config()` 自动从 `config/services.toml` 构建配置；显式传入时直接使用该配置对象，不再读取配置文件。
- **返回**：`BaseEmbedding` 实例，实际类型为以下三种之一：
  - `HashEmbedding`（当 `embedding_provider` 归一化后等于 `"hash"`）；
  - `APIEmbedding`（当 provider 不是 hash，且 `embedding_base_url` 与 `embedding_api_key` 去空白后都非空）；
  - `HashEmbedding`（其余情况，即云端配置不完整时的离线兜底）。
- **内部流程**：第一步，`config = config if config is not None else MemoryConfig.from_config()`，保证后续字段访问安全。第二步，取 `config.embedding_provider`，为空则用常量 `MEMORY_EMBEDDING_PROVIDER_DEFAULT` 兜底，然后 `.strip().casefold()` 归一化，赋值给 `provider`。第三步，若 `provider == "hash"`，立即返回 `HashEmbedding(dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION)`（注意 `embedding_dimension` 允许为 `None`，此时用常量 1024 兜底）。第四步，分别对 `config.embedding_base_url` 与 `config.embedding_api_key` 做 `strip()`（api_key 先 `str()` 再 strip，避免 `None` 报错）。第五步，若 `base_url and api_key` 同时成立，构造并返回 `APIEmbedding(api_key=..., model=config.embedding_model, base_url=..., dimension=config.embedding_dimension, timeout=config.embedding_timeout, batch_size=config.embedding_batch_size)`。第六步，否则返回离线 `HashEmbedding`，维度同样用 `config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION` 兜底。
- **异常/边界**：本函数自身不显式抛异常。`config` 为 `None` 时依赖 `MemoryConfig.from_config()`，该过程可能因 services.toml 中的非法值在 `MemoryConfig.__post_init__` 抛 `ValueError`/`TypeError`；`config.embedding_provider` 非法值同样会在 `MemoryConfig.__post_init__` 阶段就被拒绝，因此这里拿到的是已校验过的值。`embedding_dimension` 为 `None` 属于正常情况（表示「不预设，首次响应自动识别」），用常量兜底。provider 显式要求云端但配置不全时，不抛错、静默退回离线兜底。
- **同文件关系**：调用 `MemoryConfig.from_config()`（当 `config` 为 `None`）。被 `BaseMemory.__init__` 在未注入 `embedding` 时调用。文件内的 `__all__` 将其导出。

### `_merge_services_into(values: dict[str, object], services: ServicesConfig) -> None` （第 110 行）

- **作用**：这是一个原地修改的辅助函数，用于把 `config/services.toml` 里的配置值补进一个尚不完整的字典 `values`。它是 `MemoryConfig.from_config()` 的实现细节：`from_config` 先造一个空字典，再由本函数把 services 配置里的嵌入、Qdrant、Neo4j、代理等字段填进去，最后用 `cls(**values)` 构造配置对象。关键语义是「只在字段缺席时补值」——只有 `field_name not in values` 时才写入，因此未来若 `from_config` 想支持显式参数覆盖，显式值永远优先于共享配置文件，形成严格的「显式构造参数 > services.toml」两层优先级。
- **参数**：
  - `values: dict[str, object]`：待补齐的字段字典，函数会**原地修改**它，不返回新对象。当前调用方传入的是空字典。
  - `services: ServicesConfig`：从 `config/services.toml` 加载出来的服务配置对象，包含 `embedding`、`qdrant`、`neo4j`、`proxy` 四个子配置块。
- **返回**：`None`。结果通过原地修改 `values` 体现。
- **内部流程**：第一步，`embedding = services.embedding` 取出嵌入子配置，减少后续属性访问长度。第二步，构造一个 `merged` 元组序列，把「目标字段名 → 候选值」成对列出，共 14 项：`embedding_provider`、`embedding_base_url`、`embedding_model`、`embedding_api_key`、`embedding_dimension`、`embedding_batch_size`、`embedding_timeout` 来自 `services.embedding`；`qdrant_url`、`qdrant_collection`、`qdrant_api_key` 来自 `services.qdrant`；`neo4j_uri`、`neo4j_username`、`neo4j_password` 来自 `services.neo4j`；`proxy_url` 来自 `services.proxy`。第三步，`for field_name, value in merged:` 逐项遍历，若 `value is None`（该字段在 toml 中未配置）或 `field_name in values`（调用方已经显式给了值）就 `continue` 跳过，否则 `values[field_name] = value` 写入。
- **异常/边界**：无特殊处理。`services` 为 `None` 或缺少 `embedding`/`qdrant`/`neo4j`/`proxy` 子属性时会抛 `AttributeError`，但调用方 `from_config` 传入的是 `load_services_config()` 的返回值，类型上保证存在这些块。注意它只跳过 `None`，不会跳过空字符串——空字符串会照常写入 `values`，再由 `MemoryConfig.__post_init__` 按各自规则校验（例如 `embedding_base_url` 允许空串表示「未配置云端」）。
- **同文件关系**：不调用本文件其它函数；被 `MemoryConfig.from_config()` 唯一调用，并与 `load_services_config()`、`MemoryConfig.__post_init__` 协作完成配置装配。

### `utc_now() -> datetime` （第 140 行）

- **作用**：返回当前 UTC 时间，是文件内唯一的时间「取当前时刻」入口。它把 `datetime.now(UTC)` 包成函数，好处是所有调用点都使用同一时区语义（时区感知的 UTC 时间），避免出现有的地方用 `datetime.utcnow()`（无时区）而有的地方用本地时间导致的比较错误。它既被用作 `MemoryItem.created_at` / `updated_at` 的默认工厂，也被 `MemoryItem.is_expired` 用来判断是否过期，还被 `BaseMemory.add` 用来计算 `expires_at = utc_now() + timedelta(...)`。
- **参数**：无参数。
- **返回**：`datetime`，当前 UTC 时间，带 `UTC` 时区信息（`tzinfo` 非空）。
- **内部流程**：单行 `return datetime.now(UTC)`。其中 `UTC` 是从 `datetime` 模块导入的时区常量（Python 3.11+ 提供）。
- **异常/边界**：无特殊处理。依赖系统时钟，时钟回拨等极端情况不做补偿。
- **同文件关系**：被 `MemoryItem` 的字段默认工厂（`created_at`、`updated_at`）、`MemoryItem.__post_init__`（`self.updated_at = ensure_datetime(self.updated_at) or self.created_at` 中不直接调用，但 `created_at` 兜底时调用）、`MemoryItem.is_expired`、`BaseMemory.add` 调用；它自身不调用本文件其它函数。通过 `__all__` 导出给外部使用。

### `ensure_datetime(value: datetime | str | None) -> datetime | None` （第 144 行）

- **作用**：把「可能是 `datetime`、可能是 ISO 格式字符串、也可能是 `None`」的时间输入统一规范成「带 UTC 时区的 `datetime`」或 `None`。记忆数据可能来自数据库（时间常以字符串形式存储）、来自 JSON 反序列化、来自外部工具调用（用户可能传字符串），也可能根本不给（`None`），如果各处自行解析就会出现「有的有时区有的没时区」而无法比较的隐患。本函数用一处逻辑收拢：`None` 直接透传，`datetime` 原样接收，字符串用 `fromisoformat` 解析，随后统一补齐 UTC 时区并转换到 UTC。`MemoryItem.__post_init__` 用它处理 `created_at`、`updated_at`、`expires_at`、`timestamp` 四个时间字段。
- **参数**：
  - `value: datetime | str | None`：待规范化的时间值。`None` → 返回 `None`；`datetime` 实例 → 直接进入后续时区处理（无论是否带时区）；`str` → 用 ISO 8601 格式解析，Python 3.11+ 的 `fromisoformat` 可直接接受末尾的 `"Z"`；其它类型 → 抛 `TypeError`。
- **返回**：`datetime | None`。输入为 `None` 时返回 `None`；否则返回带 `UTC` 时区的 `datetime`，且已经 `astimezone(UTC)` 转换到 UTC 时刻。原本无时区的输入会被假定为 UTC。
- **内部流程**：第一步，`if value is None: return None` 提前返回。第二步，分支判断类型：`isinstance(value, datetime)` 则 `result = value`；`isinstance(value, str)` 则 `result = datetime.fromisoformat(value)`；其它类型 `raise TypeError("datetime values must be datetime, ISO string, or None")`。第三步，`if result.tzinfo is None: result = result.replace(tzinfo=UTC)`——把「裸时间」直接贴上 UTC 标签（不做本地时区换算）。第四步，`return result.astimezone(UTC)`，把带其它时区的时间换算成 UTC 表示。
- **异常/边界**：输入为 `None` 返回 `None`；非 `datetime`/`str`/`None` 类型抛 `TypeError`；字符串不是合法 ISO 格式时，`datetime.fromisoformat` 会抛 `ValueError`，本函数不做捕获，由调用方（`MemoryItem.__post_init__`）自然向上抛出。
- **同文件关系**：不调用本文件其它函数；被 `MemoryItem.__post_init__` 四次调用（处理 `created_at`、`updated_at`、`expires_at`、`timestamp`）。通过 `__all__` 导出。

### `@dataclass class MemoryItem` （第 159 行）

- **作用**：这是整个记忆系统的「规范记忆记录」数据类，也是四条记忆链路之间流转的唯一载体。它的 `content` 是可检索的文本表示，`payload` 用来承载可选的多模态数据（例如图像字节或一个 URI），两者刻意分开存放，这样向量库只需要索引文本，不必理解二进制负载。除了内容，它还携带类型、重要性、创建/更新时间、过期时间、业务时间戳、嵌入向量、模态标记和关系列表等元数据，因此单条记录就能完整描述「这段记忆是什么、属于哪类、多重要、什么时候过期、和谁有关」。`BaseMemory.add()` 会构造它、填充嵌入向量后先写文档库再写向量库；`BaseMemory.get/list/search` 返回的也是它；`MemorySearchResult` 则把它和相似度分数打包在一起。`to_dict()` 提供了 JSON 友好的序列化形式供存储层和 Web 层使用。
- **参数**：作为 dataclass，其字段即构造参数，顺序如下：
  - `content: str`（必填，位置参数）：可检索的文本内容。非字符串会被 `__post_init__` 用 `str()` 强制转换。
  - `memory_type: MemoryType | str = MemoryType.WORKING`：记忆类型，可传枚举或字符串，`__post_init__` 会归一化为 `MemoryType`。
  - `id: str = field(default_factory=lambda: str(uuid4()))`：唯一标识，默认随机 UUID4 字符串；必须是非空字符串。
  - `metadata: dict[str, Any] = field(default_factory=dict)`：任意附加元数据字典，可用于检索时的等值过滤。
  - `importance: float = 0.5`：重要性，取值必须落在 `[0, 1]` 闭区间，必须是数字且不能是 `bool`。
  - `created_at: datetime = field(default_factory=utc_now)`：创建时间，默认当前 UTC。
  - `updated_at: datetime = field(default_factory=utc_now)`：更新时间，默认当前 UTC。
  - `expires_at: datetime | None = None`：过期时刻，`None` 表示永不过期。
  - `timestamp: datetime | None = None`：业务语义上的时间戳（例如事件实际发生时间），与创建时间解耦。
  - `embedding: list[float] | None = None`：嵌入向量，`None` 表示尚未计算；提供时会被逐元素转成 `float`。
  - `payload: Any = None`：多模态负载（图像字节、URI 等），不参与向量索引。
  - `modality: str | None = None`：模态标记（如 `"text"`、`"image"`），提供时必须是非空字符串。
  - `relations: list[dict[str, Any]] = field(default_factory=list)`：关系列表，每个元素必须是 `Mapping`（通常描述与其它记忆或实体的边）。
- **返回**：构造返回 `MemoryItem` 实例；`__post_init__` 无返回值（见下条）。
- **内部流程**：dataclass 装饰器根据字段声明生成 `__init__`，其中 `id`、`metadata`、`created_at`、`updated_at`、`relations` 使用 `field(default_factory=...)` 实现「每个实例独立默认值」，避免可变默认值共享。构造完成后自动调用 `__post_init__` 做归一化与校验（见下条）。文件内 `MemoryItem` 的三个成员为 `__post_init__`、`is_expired`（property）、`to_dict`。
- **异常/边界**：合法性完全由 `__post_init__` 把关：非法类型、越界重要性、空 id、非法嵌入等都会在构造时抛 `TypeError` 或 `ValueError`。`payload` 不做类型限制（可为任意对象），只在序列化时由 `_json_safe` 兜底。
- **同文件关系**：被 `MemorySearchResult.item`（字段类型）、`BaseMemory.add`（构造）、`BaseMemory.get`（返回）、`BaseMemory.list`（返回）、`BaseMemory.delete`（内部取回比较类型）引用；其 `to_dict()` 被 `MemorySearchResult.to_dict()` 调用；`__post_init__` 调用 `ensure_datetime`、`utc_now`、`MemoryType`；`to_dict` 调用 `_json_safe`。

#### `MemoryItem.__post_init__(self) -> None` （第 182 行）

- **作用**：这是 `MemoryItem` 的构造后校验与归一化钩子，由 dataclass 自动在 `__init__` 末尾调用。它承担三件事：把宽容的输入「洗」成规范形式（内容转字符串、类型转枚举、元数据转 dict、时间统一为 UTC、嵌入转 float 列表、关系转 list），对确实非法的值坚决报错（空 id、越界重要性、非有限数或空的嵌入向量、非映射的关系项、空白 modality），以及为时间字段补默认（`created_at` 缺失时用当前 UTC，`updated_at` 缺失时回落到 `created_at`）。有了它，记忆层其余代码都可以假定拿到的 `MemoryItem` 已经是自洽的，不需要在每条读路径上重复做防御性检查。
- **参数**：`self`（隐式），即正在初始化的 `MemoryItem` 实例。
- **返回**：`None`。所有效果都体现为对 `self` 字段的原地修改。
- **内部流程**：按顺序执行以下步骤：
  1. `if not isinstance(self.content, str): self.content = str(self.content)`——宽容地把非字符串内容转成字符串。
  2. `self.memory_type = MemoryType(self.memory_type)`——把字符串或枚举统一为 `MemoryType`，非法值在此抛 `ValueError`。
  3. `if not isinstance(self.id, str) or not self.id.strip(): raise ValueError("memory id must be a non-empty string")`——id 必须是非空字符串（纯空白也算空）。
  4. `if not isinstance(self.metadata, dict): self.metadata = dict(self.metadata or {})`——非 dict 的元数据尝试用 `dict()` 转换，`None` 转成空 dict。
  5. `if isinstance(self.importance, bool) or not isinstance(self.importance, (int, float)): raise TypeError("importance must be a number")`——显式排除 `bool`（因为 `bool` 是 `int` 子类，若不排除 `True` 会被当成 1）。
  6. `if not math.isfinite(float(self.importance)) or not 0 <= float(self.importance) <= 1: raise ValueError("importance must be between 0 and 1")`——拒绝 `nan`/`inf` 与越界值。
  7. `self.importance = float(self.importance)`——统一成 `float`。
  8. 时间字段处理：`self.created_at = ensure_datetime(self.created_at) or utc_now()`；`self.updated_at = ensure_datetime(self.updated_at) or self.created_at`；`self.expires_at = ensure_datetime(self.expires_at)`；`self.timestamp = ensure_datetime(self.timestamp)`。
  9. `if self.modality is not None and (not isinstance(self.modality, str) or not self.modality.strip()): raise ValueError("modality must be a non-empty string when provided")`——允许 `None`，但给了就必须是非空字符串。
  10. 嵌入向量校验：若 `self.embedding is not None`，先拒绝 `str`/`bytes`（避免把字符串误当成数字序列），再用列表推导 `[float(v) for v in self.embedding]` 逐元素转换，转换失败抛 `TypeError("embedding must be an iterable of numbers")` 并保留 `from exc` 链；随后用 `any(not math.isfinite(value) ...)` 拒绝 `nan`/`inf`（抛 `ValueError`）；最后拒绝空向量（抛 `ValueError("embedding must not be empty")`）。
  11. 关系列表处理：`if not isinstance(self.relations, list): self.relations = list(self.relations)` 做宽容转换；再用 `any(not isinstance(relation, Mapping) ...)` 校验每一项都是映射，否则抛 `TypeError("relations must contain mappings")`。
- **异常/边界**：`ValueError`：id 为空、importance 越界或非有限、embedding 含非有限值、embedding 为空、modality 为空白字符串、`memory_type` 非法。`TypeError`：importance 非数字（含 `bool`）、embedding 为 `str`/`bytes` 或元素无法转 `float`、relations 含非映射元素；`ensure_datetime` 对非法类型/非法 ISO 字符串抛出的 `TypeError`/`ValueError` 也会向上传播。`content` 为 `None` 等会被 `str()` 转成 `"None"` 而不是报错；`metadata` 为 `None` 转空 dict。
- **同文件关系**：调用 `MemoryType`（归一化）、`ensure_datetime`（四次）、`utc_now`（兜底创建时间）；被 dataclass 生成的 `__init__` 自动调用，因此也被 `BaseMemory.add` 间接调用（它构造 `MemoryItem` 时触发）。

#### `MemoryItem.is_expired` （property，第 217 行）

- **作用**：一个只读属性，回答「这条记忆是否已经过期」。判断规则是：必须有 `expires_at` 且该时刻小于等于当前 UTC 时间。没有设置 `expires_at` 的记录永远不过期，因此工作记忆之外的类型（默认不带 TTL）不会被误删。它被 `BaseMemory.get()` 用来在读取时顺手清理过期条目：如果查到的记录已过期且类型匹配，就调用 `delete` 并返回 `None`，实现「惰性过期」——不需要后台定时任务扫表，只在真正被访问时才回收。用属性而非方法，是为了让调用点写成 `item.is_expired` 更直观。
- **参数**：无（`self` 隐式）。
- **返回**：`bool`。`expires_at is not None and expires_at <= utc_now()` 成立时返回 `True`，否则 `False`。
- **内部流程**：单行表达式求值：先判断 `self.expires_at is not None`（短路，未设置直接 `False`），再调用 `utc_now()` 取当前 UTC 时间做 `<=` 比较。比较要求两侧都是时区感知的 `datetime`，这一点由 `__post_init__` 中 `ensure_datetime` 的归一化保证。
- **异常/边界**：无特殊处理。`expires_at` 为 `None` 时不比较、直接返回 `False`；不会修改任何状态，也不删除记录（删除动作在 `BaseMemory.get` 里）。
- **同文件关系**：调用 `utc_now()`；被 `BaseMemory.get()` 调用。

#### `MemoryItem.to_dict(self) -> dict[str, Any]` （第 221 行）

- **作用**：把一条记忆记录转换成 JSON 友好的普通字典，是记忆数据「离开 Python 对象世界」的出口。它被 `MemorySearchResult.to_dict()` 复用（先拿条目字典再补 `score` 字段），进而被存储层写入 SQLite 文档库、被 Web API 序列化成响应、被 Agent 工具打印或回传给模型。转换过程中时间字段统一用 `isoformat()` 输出字符串（`expires_at`、`timestamp` 为 `None` 时输出 `None` 而不是字符串 `"None"`），类型字段用 `self.memory_type.value` 输出纯字符串，而 `metadata`、`relations`、`payload` 三个可能含 bytes 或任意对象的字段则先过 `_json_safe` 保证可 JSON 化。`embedding` 直接原样输出（已经是 `list[float]` 或 `None`），因为浮点列表本身就可 JSON 序列化。
- **参数**：无（`self` 隐式）。
- **返回**：`dict[str, Any]`，固定包含 14 个键：`id`、`content`、`memory_type`、`metadata`、`importance`、`created_at`、`updated_at`、`expires_at`、`timestamp`、`embedding`、`modality`、`relations`、`payload`（共 13 个键，其中时间相关占 4 个）。各键的取值类型：`memory_type` 为字符串，`created_at`/`updated_at` 为 ISO 字符串，`expires_at`/`timestamp` 为 ISO 字符串或 `None`，其余按字段原值或经 `_json_safe` 处理后的值。
- **内部流程**：第一步，`result: dict[str, Any] = {...}` 用一个字典字面量一次性组装全部键值：`self.memory_type.value` 取枚举字符串；`_json_safe(self.metadata)`、`_json_safe(self.relations)`、`_json_safe(self.payload)` 做安全转换；`self.created_at.isoformat()`、`self.updated_at.isoformat()` 直接格式化（这两个字段在 `__post_init__` 中保证非 `None`）；`self.expires_at.isoformat() if self.expires_at else None`、`self.timestamp.isoformat() if self.timestamp else None` 做条件格式化。第二步，`return result`。
- **异常/边界**：无特殊处理。`_json_safe` 已把不可序列化对象降级为 `str(value)`，因此本方法几乎不会抛异常；`created_at`/`updated_at` 在 `__post_init__` 中已被兜底，正常构造的实例不会为 `None`。
- **同文件关系**：调用 `_json_safe`（三次）；被 `MemorySearchResult.to_dict()` 调用。

### `_json_safe(value: Any) -> Any` （第 240 行）

- **作用**：把任意 Python 值递归地转换成「能被 `json.dumps` 序列化」的形式，是记忆数据落库/上网前的安全阀。它专门处理四类情况：`bytes`（例如图像负载）编码成 `{"__bytes__": "<base64>"}` 这个带标记的字典，从而既能进 JSON 又能被 `_json_restore` 精确识别还原；`Mapping` 递归转换键值（键统一 `str()`）；`list`/`tuple`/`set` 统一转成列表并递归；其余值先试着 `json.dumps` 一次，能过就原样返回，不能过就退化为 `str(value)`。这种「能用原值就用原值、不能就字符串化」的策略保证了序列化永不因为某个奇怪对象而整体失败。
- **参数**：
  - `value: Any`：待转换的任意值，可能是 `bytes`、字典、列表、元组、集合、基本类型，或任意自定义对象。
- **返回**：`Any`。`bytes` → `{"__bytes__": str}`；`Mapping` → `dict`（键已 `str()`）；`list`/`tuple`/`set` → `list`；可通过 `json.dumps` 的值 → 原值；其它 → `str(value)`。
- **内部流程**：第一步，`if isinstance(value, bytes): return {"__bytes__": base64.b64encode(value).decode("ascii")}`。第二步，`if isinstance(value, Mapping): return {str(key): _json_safe(item) for key, item in value.items()}`（递归）。第三步，`if isinstance(value, (list, tuple, set)): return [_json_safe(item) for item in value]`（递归，集合因此变成无序列表）。第四步，`try: json.dumps(value); return value except (TypeError, ValueError): return str(value)`——注意这里只做「试序列化」探针，返回值本身是原对象而非 JSON 文本。
- **异常/边界**：无特殊处理，函数本身不抛异常：`json.dumps` 的 `TypeError`/`ValueError` 被捕获并降级为 `str(value)`；`bytes` 分支的 base64 编码不会失败；循环引用会导致递归无限展开并最终 `RecursionError`（未做防护）；`bytearray` 不在 `bytes` 分支内，会走 `str()` 降级路径。
- **同文件关系**：递归调用自身；被 `MemoryItem.to_dict()`（处理 `metadata`、`relations`、`payload`）调用。与 `_json_restore` 构成一对逆操作（`_json_restore` 识别它生成的 `{"__bytes__": ...}` 标记）。

### `_json_restore(value: Any) -> Any` （第 254 行）

- **作用**：`_json_safe` 的逆操作，负责在从存储读回数据后把 `{"__bytes__": "<base64>"}` 标记还原成真正的 `bytes`，并把嵌套的字典/列表递归还原。它是「读取不应失败」原则的体现：即使负载里的 base64 内容损坏、无法解码，也只是原样返回那个标记字典，而不会让整条记忆读取失败。它保证了 `payload` 中的图像字节等二进制数据能完整地往返（写入 → 落库 → 读回 → 还原）。
- **参数**：
  - `value: Any`：从 JSON/存储读回的任意值，可能是带 `__bytes__` 标记的字典、普通字典、列表，或标量。
- **返回**：`Any`。识别为字节标记且解码成功时返回 `bytes`；解码失败时返回原字典；普通 `dict` 返回递归还原后的新 `dict`；`list` 返回递归还原后的新 `list`；其它值原样返回。
- **内部流程**：第一步，`if isinstance(value, dict) and set(value) == {"__bytes__"}`——用键集合精确匹配，确保只有当字典「有且仅有」`__bytes__` 这一个键时才按字节标记处理（避免误伤恰好含该键的业务数据）。匹配后进入 `try: return base64.b64decode(value["__bytes__"])`，`except Exception` 时 `return value`（带 `# noqa: BLE001` 注释，说明是刻意捕获所有异常，因为无法解码的负载按原样返回，读取不应失败）。第二步，`if isinstance(value, dict): return {key: _json_restore(item) for key, item in value.items()}`（递归，键不转换）。第三步，`if isinstance(value, list): return [_json_restore(item) for item in value]`（递归）。第四步，`return value` 原样返回。
- **异常/边界**：`base64.b64decode` 的任何异常都被 `except Exception` 吞掉并返回原字典；注意 `set(value) == {"__bytes__"}` 对非字符串键的字典同样成立，但取值传给 `b64decode` 时若类型不对会被上述 `except` 兜住。`tuple` 不会被还原成元组（`_json_safe` 已把它变成列表，JSON 本身也不区分）。循环引用同样会导致 `RecursionError`（未做防护）。
- **同文件关系**：递归调用自身；与 `_json_safe` 互为逆操作。在本文件内没有被其它函数直接调用（供存储层/外部反序列化使用），因此也未出现在 `__all__` 中。

### `@dataclass(frozen=True) class MemorySearchResult` （第 267 行）

- **作用**：把「一条命中的记忆」和「它的相似度分数」打包成一个不可变的小对象，作为 `BaseMemory.search()` 的返回元素。`frozen=True` 表示实例创建后字段不可修改，这带来两个好处：一是检索结果可以安全地传给多个消费方而不用担心被就地篡改，二是它天然可哈希、语义上明确是「一次性产出的结果快照」。分数与条目分离存放，而不是塞进 `MemoryItem.metadata`，是为了保持记忆记录本身干净（分数只对本次检索有意义，不应污染持久化数据）。`to_dict()` 提供把「条目字典 + score」合并成一个扁平字典的便捷方法，方便 Web API 直接返回。
- **参数**：作为 dataclass，字段即构造参数：
  - `item: MemoryItem`（必填）：命中的记忆条目。
  - `score: float`（必填）：相似度分数，由向量库返回；`BaseMemory.search` 中会先过滤掉低于阈值或 `<= 0` 的分数再构造本对象。
- **返回**：构造返回 `MemorySearchResult` 实例（不可变）。
- **内部流程**：`frozen=True` 使 dataclass 生成 `__init__` 后冻结赋值（`__setattr__` 会抛 `FrozenInstanceError`），并生成 `__hash__`；由于未声明默认值，两个字段都必须显式传入。文件内唯一的成员方法是 `to_dict`。
- **异常/边界**：字段缺失会抛 `TypeError`（缺少必需参数）；构造后对 `item` 或 `score` 赋值会抛 `dataclasses.FrozenInstanceError`。不对 `score` 的取值范围做校验（过滤逻辑在 `BaseMemory.search` 中）。
- **同文件关系**：字段类型引用 `MemoryItem`；被 `BaseMemory.search` 构造并作为列表元素返回；`to_dict` 调用 `MemoryItem.to_dict`。

#### `MemorySearchResult.to_dict(self) -> dict[str, Any]` （第 272 行）

- **作用**：把检索结果拍平成单个字典：先取记忆条目的完整字典表示，再在同一层加上 `score` 键。这样调用方（例如 Web 路由、Agent 工具）不需要自己处理「结果对象 → 条目字典 → 再插分数」的三步，直接 `json.dumps` 即可返回给前端或模型。之所以选择扁平结构而不是嵌套成 `{"item": {...}, "score": x}`，是为了让前端和工具层访问字段更直接。
- **参数**：无（`self` 隐式）。
- **返回**：`dict[str, Any]`，内容等于 `self.item.to_dict()` 的全部键，外加一个 `"score"` 键（值为 `self.score`）。注意 `score` 是后写入的，若条目字典里本来就有 `score` 键（`MemoryItem.to_dict` 不产生该键，所以正常情况下不会）会被覆盖。
- **内部流程**：第一步，`data = self.item.to_dict()` 取得条目字典。第二步，`data["score"] = self.score` 就地插入分数。第三步，`return data`。
- **异常/边界**：无特殊处理。`self.item` 一定是 `MemoryItem`（字段类型约束），其 `to_dict` 内部已由 `_json_safe` 保证可序列化。
- **同文件关系**：调用 `MemoryItem.to_dict()`；不被本文件其它函数调用。

### `@dataclass class MemoryConfig` （第 278 行）

- **作用**：这是记忆层的运行时配置聚合对象，集中描述「记忆数据存哪、留多久、检索多少条、用哪种嵌入、连不连 Qdrant/Neo4j、走不走代理」。默认值刻意选成本地化、无外部依赖：SQLite 路径用常量默认值，TTL 有默认秒数，工作记忆容量、检索条数、相似度阈值都有常量默认值；把 `sqlite_path` 设成 `":memory:"` 就能得到一个适合测试和短生命周期 Agent 的纯内存库。注释里还专门说明了两套连接开关的语义：出厂 `None` 表示**不连接**，走内存回退，Qdrant/Neo4j 都不启动也能跑；端点与端口的单一事实来源是 `constants.py` 的「连接与端点」小节，要连真服务就在 `config/services.toml` 里集中配置，而最终选型发生在 `memory/manager.py`（`qdrant_url` 非空才建 `QdrantVectorStore`）。嵌入维度字段留空（`None`）表示不预设、首次响应里自动识别；离线兜底不受影响，仍用常量 1024。`BaseMemory.__init__` 未注入 `config` 时会 `MemoryConfig()` 取全默认值，`make_default_embedding` 与 `from_config` 也都围绕它工作。
- **参数**：作为 dataclass，字段即构造参数：
  - `sqlite_path: str | Path = MEMORY_SQLITE_DEFAULT`：SQLite 文档库路径；传 `":memory:"` 走内存库。
  - `default_ttl_seconds: float | None = MEMORY_DEFAULT_TTL_SECONDS`：默认 TTL 秒数，仅在工作记忆未显式指定 `ttl_seconds` 时生效；`None` 表示不设默认过期。
  - `working_memory_capacity: int = MEMORY_WORKING_CAPACITY`：工作记忆容量，必须为正整数。
  - `search_limit: int = MEMORY_SEARCH_LIMIT`：默认检索返回条数，必须为正整数。
  - `similarity_threshold: float = MEMORY_SIMILARITY_THRESHOLD`：默认相似度阈值，必须落在 `[0, 1]`。
  - `embedding_dimension: int | None = MEMORY_EMBEDDING_DIMENSION_REMOTE`：远端嵌入的期望维度；`None` = 不预设、首次响应自动识别；否则必须为正整数。
  - `embedding_provider: str = MEMORY_EMBEDDING_PROVIDER_DEFAULT`：嵌入提供方；`auto` = 配了端点+密钥就走云端，否则离线兜底；取值必须属于 `MEMORY_EMBEDDING_PROVIDERS`。
  - `embedding_model: str = DEFAULT_EMBEDDING_MODEL`：嵌入模型名，必填非空字符串。
  - `embedding_base_url: str = ""`：云端端点（OpenAI 兼容），空串 = 未配置云端。
  - `embedding_api_key: str | None = None`：云端密钥，非空字符串或 `None`。
  - `embedding_timeout: float = MEMORY_EMBEDDING_TIMEOUT`：嵌入请求超时秒数，必须为正数且有限。
  - `embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE`：嵌入批大小，必须为正数且有限。
  - `qdrant_url: str | None = None`：Qdrant 端点；非空才会在 manager 中启用 `QdrantVectorStore`。
  - `qdrant_collection: str = MEMORY_QDRANT_COLLECTION`：Qdrant 集合名，必须非空。
  - `qdrant_api_key: str | None = None`：Qdrant 密钥，非空字符串或 `None`。
  - `neo4j_uri: str | None = None`：Neo4j 连接串（如 `bolt+s://...`）。
  - `neo4j_username: str | None = None`：Neo4j 用户名。
  - `neo4j_password: str | None = None`：Neo4j 密码。
  - `proxy_url: str | None = None`：本地转发代理（`http://host:port`）；显式配置覆盖默认值，云端 Qdrant/Neo4j 未配置时由 `MemoryManager` 使用 `constants.DEFAULT_PROXY_URL`（7890）。
- **返回**：构造返回 `MemoryConfig` 实例；`__post_init__` 无返回值（见下条）。
- **内部流程**：dataclass 生成 `__init__`（全部字段都有默认值，可零参构造），构造末尾自动调用 `__post_init__` 做类型与范围校验，并把 `embedding_provider` 归一化为小写去空白形式。文件内成员为 `__post_init__` 与类方法 `from_config`。
- **异常/边界**：所有字段校验集中在 `__post_init__`；不合法会抛 `ValueError` 或 `TypeError`。注意 `neo4j_uri`/`neo4j_username`/`neo4j_password`/`qdrant_url` 未做类型校验（可传任意值），它们的启用判断发生在 `memory/manager.py`。
- **同文件关系**：被 `make_default_embedding`（参数与 `from_config` 调用）、`BaseMemory.__init__`（默认配置来源）使用；`from_config` 是它的类方法；`__post_init__` 与 `_merge_services_into` 协作完成从 services.toml 的装配。

#### `MemoryConfig.__post_init__(self) -> None` （第 322 行）

- **作用**：`MemoryConfig` 的构造后校验器，是整个记忆配置的「守门人」。它把所有非法配置挡在启动阶段，避免错误值在运行期才以奇怪的方式暴露（例如容量为 0 导致工作记忆永远装不下东西、阈值大于 1 导致检索永远为空、维度为负数导致向量库报底层错误）。它的校验风格统一：先用 `isinstance(x, bool)` 显式排除布尔值（因为 `bool` 是 `int` 的子类，`True` 会被当作 1 通过数值检查），再检查类型，再检查范围/有限性。同时它顺手完成一次归一化：把 `embedding_provider` 去空白并转小写，这样后续 `make_default_embedding` 里的 `casefold()` 比较和 `MEMORY_EMBEDDING_PROVIDERS` 成员判断都基于规范形式。
- **参数**：无（`self` 隐式），读取并校验实例上已赋值的各字段。
- **返回**：`None`。校验通过时无副作用（除了把 `self.embedding_provider` 规范化为去空白小写）。
- **内部流程**：按顺序执行：
  1. `working_memory_capacity`：排除 `bool`、要求 `int`、要求 `>= 1`，否则 `raise ValueError("working_memory_capacity must be a positive integer")`。
  2. `search_limit`：同样规则，错误信息 `"search_limit must be a positive integer"`。
  3. 遍历 `for name in ("similarity_threshold",)`（用元组循环，便于将来扩展）：取 `getattr(self, name)`，排除 `bool`、要求 `(int, float)`、要求 `0 <= float(value) <= 1`，否则 `raise ValueError(f"{name} must be between 0 and 1")`。
  4. `embedding_dimension`：允许 `None`；非 `None` 时排除 `bool`、要求 `int`、要求 `>= 1`，否则 `raise ValueError("embedding_dimension must be a positive integer or None (learn from the response)")`。
  5. `embedding_provider`：要求是 `str` 且 `strip().casefold()` 属于 `MEMORY_EMBEDDING_PROVIDERS`，否则 `raise ValueError(f"embedding_provider must be one of {', '.join(MEMORY_EMBEDDING_PROVIDERS)}")`；通过后执行 `self.embedding_provider = self.embedding_provider.strip().casefold()` 归一化。
  6. `embedding_model`：要求是非空字符串（`isinstance(str)` 且 `strip()` 非空），否则 `raise ValueError("embedding_model must be a non-empty string")`。
  7. `embedding_base_url`：要求是 `str`（空串合法，表示未配置云端），否则 `raise TypeError("embedding_base_url must be a string (empty = cloud not configured)")`。
  8. `embedding_api_key`：要求是 `str` 或 `None`，且若为字符串则 `strip()` 非空，否则 `raise ValueError("embedding_api_key must be a non-empty string or None")`。
  9. 遍历 `for name in ("embedding_timeout", "embedding_batch_size")`：排除 `bool`、要求 `(int, float)`、要求 `math.isfinite(float(value))`、要求 `> 0`，否则 `raise ValueError(f"{name} must be positive")`。
  10. `default_ttl_seconds`：允许 `None`；非 `None` 时排除 `bool`、要求 `(int, float)`、要求有限、要求 `> 0`，否则 `raise ValueError("default_ttl_seconds must be positive or None")`。
  11. `qdrant_collection`：要求非空字符串，否则 `raise ValueError("qdrant_collection must be non-empty")`。
  12. `qdrant_api_key`：`str` 或 `None`，字符串时必须 `strip()` 非空，否则 `raise ValueError("qdrant_api_key must be a non-empty string or None")`。
  13. `proxy_url`：`str` 或 `None`，字符串时必须 `strip()` 非空，否则 `raise ValueError("proxy_url must be a non-empty string or None")`。
- **异常/边界**：`ValueError`：容量/检索条数非正整数、阈值越界、维度非正整数、provider 不在允许集合、model 为空、api_key 为空字符串、timeout/batch_size 非正或非有限、ttl 非正或非有限、collection 为空、qdrant_api_key 为空字符串、proxy_url 为空字符串。`TypeError`：`embedding_base_url` 非字符串。空值处理：`embedding_dimension`、`default_ttl_seconds`、`embedding_api_key`、`qdrant_api_key`、`proxy_url` 显式允许 `None`；`embedding_base_url` 允许空串。注意 `qdrant_url`、`neo4j_*` 不在此校验。
- **同文件关系**：调用常量 `MEMORY_EMBEDDING_PROVIDERS` 与 `math.isfinite`；被 dataclass 生成的 `__init__` 自动调用，因此 `MemoryConfig()` 与 `MemoryConfig.from_config()` 都会经过它。

#### `MemoryConfig.from_config() -> MemoryConfig` （第 365 行，`@classmethod`）

- **作用**：从 `config/services.toml` 构建一份配置对象，是「配置只认 config/ 与 constants.py」这一原则的落地入口。它只实现两层优先级：**显式构造参数 > services.toml**；历史上存在的 `HELLOAGENTS_MEMORY_*` 环境变量层与 `.env` 装载已经被删除。services.toml 被定位为「所有外部 API 调用」的集中配置（嵌入/搜索/Qdrant/Neo4j），由 `core/services_config.py` 负责解析。实现方式很简洁：先造一个空字典，用 `_merge_services_into` 把 services 里的值补进去，再 `cls(**values)` 构造——因此补进来的值同样要过 `__post_init__` 的完整校验，不会绕过守门人。`make_default_embedding` 在未传 config 时就是调用它。
- **参数**：无显式参数（`cls` 隐式，由 `@classmethod` 注入，指向 `MemoryConfig` 或其子类）。
- **返回**：`MemoryConfig` 实例（严格说是 `cls` 的实例，支持子类继承时返回子类）。所有字段值来自 `config/services.toml` 中已配置的项，未配置的项使用 dataclass 声明的默认值。
- **内部流程**：第一步，`values: dict[str, object] = {}` 建立空字典。第二步，`_merge_services_into(values, load_services_config())`——`load_services_config()` 读并解析 services.toml 返回 `ServicesConfig`，`_merge_services_into` 把 embedding/qdrant/neo4j/proxy 共 14 个字段中「有值且尚未在 values 中」的项写进字典。第三步，`return cls(**values)` 用关键字展开构造配置对象，触发 `__post_init__` 校验与 `embedding_provider` 归一化。
- **异常/边界**：`load_services_config()` 读取或解析 services.toml 失败时，其自身抛出的异常（文件缺失、TOML 语法错误等）会向上传播；services.toml 里写了非法值（例如 provider 不在允许集合、维度为 0）会在 `cls(**values)` 阶段由 `__post_init__` 抛 `ValueError`/`TypeError`。services.toml 完全缺失时，`values` 为空字典，构造出的是全默认配置。
- **同文件关系**：调用 `_merge_services_into` 与外部函数 `load_services_config`；被 `make_default_embedding`（当 `config` 为 `None` 时）调用。`MemoryConfig` 自身被 `BaseMemory.__init__` 用作默认配置。

### `class BaseMemory` （第 380 行）

- **作用**：这是四种记忆类型共用的抽象基类（实际是「可实例化的公共实现」），把记忆操作拆成三个协作者：`document_store` 作为「事实来源」负责持久化与按类型列举，`vector_store` 负责向量索引与相似度检索，`embedding` 负责把文本（以及可选的多模态负载）转成向量。它提供统一的 CRUD（`add`/`get`/`delete`/`list`/`clear`）与语义检索（`search`）语义，并通过类属性 `memory_type` 让每个子类声明自己归属哪一层记忆。所有写操作都遵循「先写文档库、再写向量库，向量库失败则回滚文档库」的顺序，保证不会出现「向量索引里有、事实来源里没有」的幽灵记录。读操作（`get`/`list`）刻意不触碰向量索引，索引重建被明确划归 manager 在启动时负责，避免每次读取都产生 O(n) 的向量 upsert。子类通过覆盖 `delete` 等方法可以扩展副作用（例如 `SemanticMemory` 还会删除图谱边），而 `clear` 特意走 `self.delete` 以触发这些覆盖。
- **参数**：无构造签名在此处（见 `__init__`）；类属性 `memory_type: MemoryType` 为类型声明，子类必须给出具体值（基类自身未赋默认值，若直接实例化且不传 `memory_type` 参数会因属性缺失而报错）。
- **返回**：类定义本身；实例化由 `__init__` 完成。
- **内部流程**：类体只包含类属性 `memory_type` 的类型注解和 9 个方法：`__init__`、`_embed_item`、`_validate_embedding_dimension`、`add`、`get`、`delete`、`search`、`list`、`clear`。实例状态在 `__init__` 中建立（config、三个协作者、自身记忆类型），其余方法都基于这些状态工作。
- **异常/边界**：直接实例化 `BaseMemory()` 而不传 `memory_type` 会因为在 `__init__` 中执行 `MemoryType(memory_type or self.memory_type)` 而读取不存在的类属性，抛 `AttributeError`；传了合法 `memory_type` 则可正常实例化。非法类型字符串抛 `ValueError`（来自 `MemoryType(...)`）。
- **同文件关系**：构造时调用 `make_default_embedding`（未注入 embedding 时）；`add` 调用 `_embed_item` 与 `_validate_embedding_dimension`；`search` 调用 `_validate_embedding_dimension` 与 `self.get`；`clear` 调用 `self.delete`；`get` 调用 `self.delete`。它是本文件对外的核心出口，并被 `__all__` 导出。

#### `BaseMemory.__init__(self, *, document_store: BaseDocumentStore | None = None, vector_store: BaseVectorStore | None = None, embedding: BaseEmbedding | None = None, config: MemoryConfig | None = None, memory_type: MemoryType | str | None = None) -> None` （第 385 行）

- **作用**：为一种记忆类型装配运行时依赖，是「依赖注入 + 合理默认」的组合。三个存储/嵌入协作者都可以外部注入（便于测试用内存实现替换、或在 manager 中共享同一个 Qdrant 向量库与嵌入服务），也都可以省略，省略时给出可用的本地默认值：`SQLiteDocumentStore` + `InMemoryVectorStore` + `make_default_embedding`。`config` 省略时使用全默认 `MemoryConfig()`。`memory_type` 必须最终确定，它既可由参数传入（覆盖类属性），也可由子类通过类属性声明。之所以存储实现要在方法体内 `from .storage import ...` 延迟导入，是因为存储模块需要 import 本模块的数据结构，模块顶部导入会形成循环依赖。
- **参数**（全部为关键字参数，`*` 之后）：
  - `document_store: BaseDocumentStore | None = None`：文档存储（事实来源）。`None` 时自建 `SQLiteDocumentStore(self.config.sqlite_path)`。
  - `vector_store: BaseVectorStore | None = None`：向量存储（索引）。`None` 时自建 `InMemoryVectorStore()`（纯内存，重启即丢）。
  - `embedding: BaseEmbedding | None = None`：嵌入服务。`None` 时调用 `make_default_embedding(self.config)` 按配置选择云端或离线实现。
  - `config: MemoryConfig | None = None`：运行时配置。`None` 时使用 `MemoryConfig()` 全默认值。
  - `memory_type: MemoryType | str | None = None`：本实例的记忆类型。`None` 时回落到类属性 `self.memory_type`（由子类声明）；非 `None` 时覆盖类属性。
- **返回**：`None`（构造器）。
- **内部流程**：第一步，`from .storage import InMemoryVectorStore, SQLiteDocumentStore` 延迟导入，规避循环导入。第二步，`self.config = config if config is not None else MemoryConfig()`。第三步，`self.document_store = document_store if document_store is not None else SQLiteDocumentStore(self.config.sqlite_path)`。第四步，`self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()`。第五步，`self.embedding = embedding if embedding is not None else make_default_embedding(self.config)`。第六步，`self.memory_type = MemoryType(memory_type or self.memory_type)`——注意 `memory_type` 传空字符串时会被 `or` 判为假而回落到类属性，传合法值则覆盖并归一化为枚举。
- **异常/边界**：未传 `memory_type` 且子类/基类没有类属性 `memory_type` 时抛 `AttributeError`；`memory_type` 为非法字符串时 `MemoryType(...)` 抛 `ValueError`。`config.sqlite_path` 非法（例如目录不存在、无写权限）不会在此抛错，而是在 `SQLiteDocumentStore` 内部或首次使用时才暴露。`make_default_embedding` 在云端配置不全时静默退回离线，不会在此抛错。
- **同文件关系**：调用 `MemoryConfig()`（默认配置）、`make_default_embedding`（默认嵌入）；延迟导入 `memory/storage.py` 的 `InMemoryVectorStore`、`SQLiteDocumentStore`。被本类的所有实例方法依赖其建立的四个属性。

#### `BaseMemory._embed_item(self, content: str, *, payload: Any = None, modality: str | None = None) -> list[float]` （第 400 行）

- **作用**：为「一次写入」计算嵌入向量，并允许支持多模态的嵌入后端把 `payload`（例如图像字节）一起折进向量。设计意图是兼容两种嵌入实现：`BaseEmbedding.embed_item` 在基类里是纯文本实现，而云端 VL 模型（`APIEmbedding`，模型名带 `vl`）会覆盖它，从而支持「图像」以及「图像+文本融合」的嵌入；如果注入的 embedding 对象根本不是 `BaseEmbedding` 的子类（例如测试里的轻量替身），则退回调用普通的 `embed(content)`。用 `getattr(..., None)` + `callable` 探测而不是直接假设方法存在，使这个鸭子类型分支既安全又不需要引入额外抽象。
- **参数**：
  - `content: str`：要嵌入的文本内容（位置参数）。
  - `payload: Any = None`（仅关键字）：可选的多模态负载，透传给支持它的嵌入实现；`None` 表示纯文本写入。
  - `modality: str | None = None`（仅关键字）：模态标记，透传给支持它的嵌入实现；`None` 表示不指定。
- **返回**：`list[float]`，即嵌入向量；长度由所用嵌入实现的维度决定（离线 `HashEmbedding` 与云端模型可能不同）。
- **内部流程**：第一步，`embed_item = getattr(self.embedding, "embed_item", None)` 尝试取 `embed_item` 属性。第二步，`if callable(embed_item): return embed_item(content, payload=payload, modality=modality)`——把三个参数一并交给它。第三步，否则 `return self.embedding.embed(content)`，退化为纯文本嵌入（丢弃 `payload`/`modality`）。
- **异常/边界**：不做异常处理；底层嵌入实现抛出的异常（网络超时、鉴权失败、维度不匹配等）会原样向上传播。若 embedding 既没有 `embed_item` 也没有 `embed`，会抛 `AttributeError`。`payload` 为 `None` 时对纯文本实现无影响。
- **同文件关系**：被 `BaseMemory.add` 在构造条目后调用；自身调用 `self.embedding` 上的方法，不调用本文件其它函数。

#### `BaseMemory._validate_embedding_dimension(self, vector: list[float]) -> None` （第 413 行）

- **作用**：一个轻量的维度守卫，用来在把向量交给存储/检索之前检查它的长度是否与嵌入服务声明的期望维度一致。维度不一致是记忆系统里最隐蔽也最致命的错误之一：写入维度错了会在向量库里造成混合维度数据，检索维度错了会得到毫无意义的相似度。它通过 `getattr(self.embedding, "dimension", 0)` 读取期望维度，取不到（或为 0/假值）时直接跳过校验，因此对没有声明维度的自定义嵌入实现保持宽容。写入路径（`add`）和检索路径（`search`）都会调用它。
- **参数**：
  - `vector: list[float]`：待校验的嵌入向量（由 `_embed_item` 或 `embedding.embed` 产出）。
- **返回**：`None`。校验通过时无返回值、无副作用。
- **内部流程**：第一步，`expected = getattr(self.embedding, "dimension", 0)` 取期望维度，缺失时用 `0`。第二步，`if expected and len(vector) != expected: raise ValueError(f"embedding dimension {len(vector)} does not match expected dimension {expected}")`——`expected` 为 `0`/`None`/空值时短路跳过。
- **异常/边界**：维度不匹配时抛 `ValueError`，错误信息里带实际长度与期望长度，便于定位。`expected` 为假值（0、`None`）时不校验。不检查 `vector` 是否为列表（传 `None` 会在 `len()` 处抛 `TypeError`）。
- **同文件关系**：被 `BaseMemory.add`（写入前）与 `BaseMemory.search`（检索前）调用；自身只读取 `self.embedding.dimension`，不调用本文件其它函数。

#### `BaseMemory.add(self, content: str, *, metadata: Mapping[str, Any] | None = None, importance: float = 0.5, ttl_seconds: float | None = None, timestamp: datetime | str | None = None, item_id: str | None = None, payload: Any = None, modality: str | None = None, relations: list[dict[str, Any]] | None = None) -> MemoryItem` （第 418 行）

- **作用**：向本记忆类型写入一条新记忆（或按 `item_id` 覆盖更新一条已有记忆），是记忆系统最主要的写入口。它把「参数校验 → TTL 计算 → id 冲突检查 → 构造 `MemoryItem` → 计算嵌入 → 维度校验 → 双写持久化 → 失败回滚」这一整套流程封装在一处，调用方只需给出内容与少量可选元信息。TTL 语义被明确设计为唯一的过期入口：`expires_at` 由它派生，没有直传的调用方；且当调用方没给 `ttl_seconds` 而本实例又是工作记忆时，会自动套用 `config.default_ttl_seconds`，让工作记忆天然带过期时间。`item_id` 参数让它兼具「新增」和「按 id 覆盖」两种用法，覆盖时会校验原记录的类型归属，避免把工作记忆的 id 拿去覆盖情景记忆。持久化顺序是刻意设计的：先写文档库（事实来源）再写向量库，因为「向量后端失败但事实来源已写」可以用回滚修复，而反过来会产生索引里有、事实来源里没有的幽灵记录。
- **参数**（`content` 为位置参数，其余全部为关键字参数）：
  - `content: str`：记忆的文本内容。非字符串会被 `str()` 强制转换。
  - `metadata: Mapping[str, Any] | None = None`：附加元数据，内部用 `dict(metadata or {})` 复制成普通字典，`None` 视为空。
  - `importance: float = 0.5`：重要性，最终由 `MemoryItem.__post_init__` 校验必须为 `[0, 1]` 内的有限数字。
  - `ttl_seconds: float | None = None`：存活秒数。`None` 且类型为工作记忆时取 `config.default_ttl_seconds`；显式给出时必须是正的 `int`/`float`（非 `bool`），否则抛 `ValueError`。
  - `timestamp: datetime | str | None = None`：业务时间戳，交给 `MemoryItem.__post_init__` 里的 `ensure_datetime` 归一化。
  - `item_id: str | None = None`：指定 id。`None` 时生成新的 UUID4；给出时先查库判断是否为覆盖更新。
  - `payload: Any = None`：多模态负载，写入 `MemoryItem.payload` 并透传给 `_embed_item`（供 VL 模型使用）。
  - `modality: str | None = None`：模态标记，写入 `MemoryItem.modality` 并透传。
  - `relations: list[dict[str, Any]] | None = None`：关系列表，用 `list(relations or [])` 复制，`None` 视为空列表。
- **返回**：`MemoryItem`，即最终写入（或覆盖后）的记忆对象，其 `embedding` 已填充、`expires_at` 已按 TTL 计算、`created_at`/`updated_at` 由构造时生成。
- **内部流程**：按顺序：
  1. `if not isinstance(content, str): content = str(content)` 宽容转换内容。
  2. 初始化 `expires_at: datetime | None = None`。
  3. `if ttl_seconds is None and self.memory_type == MemoryType.WORKING: ttl_seconds = self.config.default_ttl_seconds`——工作记忆自动套默认 TTL。
  4. `if ttl_seconds is not None:` 校验 `isinstance(ttl_seconds, bool)` 为假、类型为 `(int, float)`、`> 0`，否则 `raise ValueError("ttl_seconds must be positive")`；通过后 `expires_at = utc_now() + timedelta(seconds=float(ttl_seconds))`。
  5. `existing = self.document_store.get(item_id) if item_id else None`——只有指定 id 时才查库。
  6. `if existing is not None and existing.memory_type != self.memory_type: raise ValueError(f"item id already belongs to {existing.memory_type.value} memory")`——禁止跨类型覆盖。
  7. 构造 `item = MemoryItem(id=item_id if item_id is not None else str(uuid4()), content=content, memory_type=self.memory_type, metadata=dict(metadata or {}), importance=importance, expires_at=expires_at, timestamp=timestamp, payload=payload, modality=modality, relations=list(relations or []))`；注意 `created_at`/`updated_at` 未传，由 dataclass 默认工厂取当前 UTC。
  8. `item.embedding = self._embed_item(item.content, payload=payload, modality=modality)` 计算向量。
  9. `self._validate_embedding_dimension(item.embedding)` 校验维度。
  10. `self.document_store.upsert(item)` 先写事实来源。
  11. `try: self.vector_store.upsert(item) except Exception:`——向量写入失败时进入回滚分支：若 `existing is None`（本次是新增）则 `self.document_store.delete(item.id)` 删除刚写入的记录；否则（本次是覆盖更新）用 `self.document_store.upsert(existing)` 恢复旧记录，并进一步恢复旧向量——若 `existing.embedding is not None` 则 `self.vector_store.upsert(existing)`，否则 `self.vector_store.delete(existing.id)`；最后 `raise` 把原始异常抛出（不吞掉，让调用方知道写入失败）。
  12. `return item`。
- **异常/边界**：`ValueError`：`ttl_seconds` 非正或非数字（含 `bool`）、`item_id` 已被其它记忆类型占用；`TypeError`/`ValueError`：来自 `MemoryItem.__post_init__` 的各项校验（重要性越界、embedding 非法、modality 空白、relations 非映射等）与 `_validate_embedding_dimension`（维度不匹配）；嵌入服务或存储后端的异常（网络、磁盘、连接）原样向上传播。空值处理：`metadata`/`relations`/`payload`/`timestamp`/`modality` 为 `None` 都有明确兜底；`content` 为 `None` 会被转成字符串 `"None"`。回滚分支只处理向量写入异常，文档库写入异常不触发回滚（此时尚未产生不一致）。
- **同文件关系**：调用 `MemoryType`（比较类型）、`utc_now`（计算过期时刻）、`MemoryItem`（构造）、`self._embed_item`、`self._validate_embedding_dimension`、`self.document_store`（`get`/`upsert`/`delete`）、`self.vector_store`（`upsert`/`delete`）。被外部（manager、Agent 工具、Web 路由）调用；本文件内没有其它函数调用它。

#### `BaseMemory.get(self, item_id: str) -> MemoryItem | None` （第 475 行）

- **作用**：按 id 读取一条记忆，并顺带完成「惰性过期清理」与「类型归属过滤」。它的行为可以概括为三段：先从文档库取记录；如果记录存在、已过期、且类型属于本实例，就调用 `self.delete(item_id)` 把它删掉并返回 `None`（不需要后台清理任务，访问即回收）；最后只在记录类型与本实例一致时返回，否则返回 `None`（让不同记忆类型之间的 id 空间彼此隔离，跨类型读取不会泄漏数据）。它同时被 `search` 复用为「候选 id → 记录对象」的解析步骤，因此它不触碰向量索引这一点也让检索路径保持轻量。
- **参数**：
  - `item_id: str`：要读取的记忆 id。
- **返回**：`MemoryItem | None`。命中且类型匹配、未过期时返回该对象；记录不存在、类型不匹配、或已过期（此时记录已被删除）时返回 `None`。
- **内部流程**：第一步，`item = self.document_store.get(item_id)` 取记录。第二步，`if item is not None and item.is_expired and item.memory_type == self.memory_type:`——三个条件同时成立时执行 `self.delete(item_id)` 并 `return None`。第三步，`return item if item is not None and item.memory_type == self.memory_type else None`——类型不匹配一律返回 `None`。
- **异常/边界**：无特殊处理。`item_id` 为 `None` 或不存在时，`document_store.get` 通常返回 `None`（具体行为取决于存储实现），本函数按 `None` 处理并返回 `None`。过期清理时 `delete` 的返回值被忽略；若删除失败（例如向量库报错），异常会向上传播。
- **同文件关系**：调用 `self.document_store.get`、`self.delete`（过期清理）、读取 `MemoryItem.is_expired`；被 `BaseMemory.search` 在遍历候选时对每个候选 id 调用。

#### `BaseMemory.delete(self, item_id: str) -> bool` （第 482 行）

- **作用**：删除一条记忆，同时清理向量索引，并保证「只能删自己类型的记录」。它先读回记录做归属校验：不存在或类型不属于本实例时直接返回 `False`，不做任何删除动作，这样即使传入别的记忆类型的 id 也不会误删。校验通过后先 `vector_store.delete(item_id)` 再 `document_store.delete(item_id)`，并把后者的布尔结果作为返回值——先删索引再删事实来源，与 `add` 的写入顺序相反但目的一致：宁可短暂出现「索引已删、记录还在」（可被后续重建或重复删除修复），也不要留下「记录已删、索引还在」的脏索引。子类可以覆盖它来追加副作用（`clear` 特意通过 `self.delete` 调用以保证覆盖生效）。
- **参数**：
  - `item_id: str`：要删除的记忆 id。
- **返回**：`bool`。类型匹配且文档库删除成功时返回 `True`；记录不存在、类型不匹配、或文档库删除未命中时返回 `False`。
- **内部流程**：第一步，`item = self.document_store.get(item_id)` 读回记录。第二步，`if item is None or item.memory_type != self.memory_type: return False` 提前返回。第三步，`self.vector_store.delete(item_id)` 删向量索引（返回值忽略）。第四步，`return self.document_store.delete(item_id)` 删文档并把结果作为返回值。
- **异常/边界**：无特殊处理（无显式 try/except）。若向量库删除抛异常，文档库删除不会执行，异常向上传播；若文档库删除失败返回 `False`，向量索引可能已被删除（下次重建可修复）。`item_id` 不存在时安全返回 `False`。
- **同文件关系**：调用 `self.document_store`（`get`/`delete`）与 `self.vector_store.delete`；被 `BaseMemory.get`（过期清理）与 `BaseMemory.clear`（逐条清理）调用，也可能被子类覆盖。

#### `BaseMemory.search(self, query: str, *, limit: int | None = None, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[MemorySearchResult]` （第 489 行）

- **作用**：语义检索入口：把查询文本嵌入成向量，在向量库里找同类型的相似候选，再逐条回到事实来源取记录、做过期与元数据过滤，最后按分数组装成 `MemorySearchResult` 列表返回。它包含若干务实的工程决策：查询为空或纯空白时直接返回空列表（不做无意义的嵌入调用）；参数默认值来自 `config.search_limit` 与 `config.similarity_threshold`；向向量库请求的候选数是 `max(limit * 4, limit)`，即多取几倍，因为后续的过期/类型/元数据过滤会在向量库排名之后剔除一部分，多取才能凑够目标条数；过滤时还额外要求 `score > 0`，因为零向量或无关向量即使在阈值默认为 0（最宽松）时也不该算命中；元数据过滤采用「所有给定键值都必须相等」的 AND 语义；凑满 `limit` 条立即 `break`，不做多余遍历。
- **参数**：
  - `query: str`（位置参数）：查询文本，必须为字符串（否则 `TypeError`），纯空白视为无查询。
  - `limit: int | None = None`（仅关键字）：返回条数上限。`None` 时用 `config.search_limit`；显式给出时必须是非 `bool` 的正整数，否则 `ValueError`。
  - `threshold: float | None = None`（仅关键字）：相似度下限。`None` 时用 `config.similarity_threshold`；显式给出时必须是 `[0, 1]` 内的数字（非 `bool`），否则 `ValueError`。
  - `metadata: Mapping[str, Any] | None = None`（仅关键字）：元数据等值过滤条件，`None` 或空映射表示不过滤；非空时要求候选条目的 `metadata` 在每一个给定键上都与给定值相等。
- **返回**：`list[MemorySearchResult]`，按向量库给出的候选顺序排列（通常是相似度从高到低），长度不超过 `limit`；无命中、查询为空、或全部候选被过滤时返回空列表 `[]`。
- **内部流程**：按顺序：
  1. `if not isinstance(query, str): raise TypeError("query must be a string")`。
  2. `if not query.strip(): return []`——空白查询短路返回。
  3. `limit = self.config.search_limit if limit is None else limit`；随后 `if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1: raise ValueError("limit must be a positive integer")`。
  4. `threshold = self.config.similarity_threshold if threshold is None else threshold`；随后 `if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1: raise ValueError("threshold must be between 0 and 1")`。
  5. `vector = self.embedding.embed(query)` 计算查询向量（注意这里用的是 `embed` 而非 `_embed_item`，因为查询没有 payload/modality）。
  6. `self._validate_embedding_dimension(vector)` 校验查询向量维度。
  7. `candidates = self.vector_store.search(vector, limit=max(limit * 4, limit), memory_type=self.memory_type)` 取候选 `(item_id, score)` 列表，超取 4 倍并限制在本记忆类型内。
  8. `results: list[MemorySearchResult] = []` 初始化结果列表。
  9. `for item_id, score in candidates:` 遍历候选：`item = self.get(item_id)` 解析记录（`get` 内部会顺手清理过期项并过滤类型）；`if item is None or score < threshold or (query.strip() and score <= 0): continue` 过滤掉空记录、低于阈值、以及非正分数的候选；`if metadata and any(item.metadata.get(key) != value for key, value in metadata.items()): continue` 做元数据 AND 过滤；`results.append(MemorySearchResult(item=item, score=score))`；`if len(results) >= limit: break` 凑满即停。
  10. `return results`。
- **异常/边界**：`TypeError`：`query` 非字符串。`ValueError`：`limit` 非正整数（含 `bool`）、`threshold` 非数字或不在 `[0, 1]`、`_validate_embedding_dimension` 维度不匹配。嵌入服务与向量库的异常（网络、连接、集合不存在）原样向上传播。空值：`query` 为空白返回 `[]`；`metadata` 为 `None`/空映射不过滤；`candidates` 为空则循环不执行、返回 `[]`。注意 `score` 的比较是直接与 `threshold` 数值比较，未做 `NaN` 处理（若向量库返回 `NaN`，`NaN < threshold` 为 `False`、`NaN <= 0` 为 `False`，该候选不会被这两条过滤掉）。
- **同文件关系**：调用 `self.embedding.embed`、`self._validate_embedding_dimension`、`self.vector_store.search`、`self.get`、`MemorySearchResult`（构造）；被外部（manager、Agent 工具、Web 路由）调用；本文件内没有其它函数调用它。

#### `BaseMemory.list(self, *, include_expired: bool = False) -> list[MemoryItem]` （第 520 行）

- **作用**：列举本记忆类型下的记录，是一个纯读操作，刻意不触碰向量索引。文档字符串明确说明了理由：索引重建是 manager 在启动时的职责，读取操作不应在每次调用时做 O(n) 的向量 upsert（否则一次列举就会把整个记忆库重新嵌入一遍，既慢又可能触发大量外部 API 调用）。它把类型过滤与「是否包含已过期记录」这两个条件直接下推给文档存储，让存储层用 SQL/查询条件高效完成，而不是把全量记录拉到 Python 里再筛。默认 `include_expired=False`，即面向业务展示时过期记录不出现；需要审计、清理或统计时传 `True`。
- **参数**：
  - `include_expired: bool = False`（仅关键字）：是否把已过期（`expires_at` 早于当前时间）的记录也包含在结果中。默认 `False` 表示只列未过期的。
- **返回**：`list[MemoryItem]`，本记忆类型下的记录列表；无记录时返回空列表。排序由底层文档存储决定，本函数不重排。
- **内部流程**：单行 `return self.document_store.list(memory_type=self.memory_type, include_expired=include_expired)`——把两个过滤条件交给文档存储实现。
- **异常/边界**：无特殊处理；不做参数类型校验（传非布尔值会被存储层按真值处理）。若文档存储查询失败（磁盘、连接问题），异常原样向上传播。注意它不会像 `get` 那样惰性删除过期记录，只是按条件过滤。
- **同文件关系**：调用 `self.document_store.list`；不被本文件其它函数调用（`clear` 用的是 `document_store.list` 且强制 `include_expired=True`，未经过本方法）。

#### `BaseMemory.clear(self) -> int` （第 528 行）

- **作用**：清空本记忆类型下的所有记录（包括已过期的），返回实际删除的条数。它实现上的关键细节是「不直接调用文档存储的批量删除，而是逐条走 `self.delete(item.id)`」——这样做是为了让子类的覆盖生效：例如 `SemanticMemory` 覆盖了 `delete`，除了删记录还会删除知识图谱里的边，如果 `clear` 绕过 `delete` 直接清表，就会留下悬空的图边。因此 `clear` 牺牲一点效率换取副作用的一致性。它在列举时显式传 `include_expired=True`，保证过期记录也被纳入清理范围，不会出现「列表看不到、但又没被删掉」的残留。
- **参数**：无参数。
- **返回**：`int`，成功删除的记录条数。计数只在 `self.delete(item.id)` 返回真值时递增，因此类型不匹配或删除未命中的条目不会被计入。
- **内部流程**：第一步，`count = 0`。第二步，`for item in self.document_store.list(memory_type=self.memory_type, include_expired=True):` 取本类型全部记录（含过期）。第三步，`if self.delete(item.id): count += 1`——通过多态调用 `delete`（可能被子类覆盖），成功才计数。第四步，`return count`。
- **异常/边界**：无特殊处理（无 try/except）。若某条记录的 `delete` 抛异常（例如向量库连接失败），循环中断、异常向上传播，此前已删除的记录不会回滚，返回值也拿不到。空库时循环不执行、返回 `0`。
- **同文件关系**：调用 `self.document_store.list` 与 `self.delete`；不被本文件其它函数调用。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `MemoryType` | 定义 working / episodic / semantic / perceptual 四种记忆类型的字符串枚举，作为记忆数据的类型标签与隔离依据。 |
| `default_sqlite_path()` | 返回默认 SQLite 持久化路径，唯一保留的覆盖入口是环境变量 `MEMORY_DB_PATH`，否则用项目根目录下的默认文件名。 |
| `make_default_embedding(config=None)` | 按配置构建嵌入服务：显式 hash 走离线、云端端点+密钥齐全走 `APIEmbedding`、配置不全则静默退回离线 `HashEmbedding`。 |
| `_merge_services_into(values, services)` | 把 services.toml 的嵌入/Qdrant/Neo4j/代理字段原地补进字典，且只在字段缺席时补值（显式参数优先）。 |
| `utc_now()` | 返回带 UTC 时区的当前时间，统一全文件的时间取值语义。 |
| `ensure_datetime(value)` | 把 `datetime`/ISO 字符串/`None` 统一规范成带 UTC 时区的 `datetime` 或 `None`，非法类型抛 `TypeError`。 |
| `MemoryItem` | 记忆系统的规范记录数据类，承载内容、类型、元数据、重要性、时间、嵌入、多模态负载与关系。 |
| `MemoryItem.__post_init__()` | 构造后校验并归一化各字段（类型转换、时间补 UTC、importance 范围、embedding 有限非空、relations 必须为映射）。 |
| `MemoryItem.is_expired` | 只读属性，判断是否设置了 `expires_at` 且已早于当前 UTC 时间。 |
| `MemoryItem.to_dict()` | 把记忆记录转成 JSON 友好字典（时间转 ISO 字符串、类型取 `.value`、metadata/relations/payload 过 `_json_safe`）。 |
| `_json_safe(value)` | 递归把任意值转成可 JSON 序列化的形式，`bytes` 编码为 `{"__bytes__": base64}`，其余不可序列化对象降级为字符串。 |
| `_json_restore(value)` | `_json_safe` 的逆操作，递归把 `{"__bytes__": ...}` 标记还原为 `bytes`，解码失败时原样返回而不报错。 |
| `MemorySearchResult` | 不可变的检索结果数据类，把命中的 `MemoryItem` 与相似度 `score` 打包在一起。 |
| `MemorySearchResult.to_dict()` | 把条目字典与 `score` 合并成一个扁平字典，便于直接序列化返回。 |
| `MemoryConfig` | 记忆层运行时配置聚合对象，涵盖 SQLite 路径、TTL、容量、检索条数、阈值、嵌入参数与 Qdrant/Neo4j/代理连接开关。 |
| `MemoryConfig.__post_init__()` | 逐字段校验配置合法性（正整数、`[0,1]` 阈值、有限正数超时、非空 model/collection 等）并归一化 `embedding_provider`。 |
| `MemoryConfig.from_config()` | 类方法，从 `config/services.toml` 构建配置，实现「显式构造参数 > services.toml」的两层优先级。 |
| `BaseMemory` | 四种记忆类型共用的基类，组合文档存储、向量存储与嵌入服务，提供统一 CRUD 与语义检索。 |
| `BaseMemory.__init__()` | 以依赖注入 + 合理默认的方式装配 config、document_store、vector_store、embedding 与 memory_type（延迟导入避免循环依赖）。 |
| `BaseMemory._embed_item(content, payload, modality)` | 计算写入用的嵌入向量，优先调用支持多模态的 `embed_item`，否则退回纯文本 `embed`。 |
| `BaseMemory._validate_embedding_dimension(vector)` | 维度守卫：嵌入服务声明了期望维度而向量长度不匹配时抛 `ValueError`，未声明则跳过。 |
| `BaseMemory.add(...)` | 写入或按 id 覆盖一条记忆：校验 TTL、检查 id 类型归属、算嵌入、先写文档库再写向量库，向量失败则回滚。 |
| `BaseMemory.get(item_id)` | 按 id 读取记录，顺手惰性删除已过期项，并只返回属于本记忆类型的记录。 |
| `BaseMemory.delete(item_id)` | 删除记录及其向量索引，仅当记录存在且类型匹配时执行，返回是否删除成功。 |
| `BaseMemory.search(query, limit, threshold, metadata)` | 语义检索：嵌入查询、超取候选、按阈值/过期/类型/元数据过滤后组装成带分数的结果列表。 |
| `BaseMemory.list(include_expired=False)` | 列举本类型的记录，纯读操作、不触碰向量索引，是否包含过期项由参数决定。 |
| `BaseMemory.clear()` | 清空本类型全部记录（含过期），逐条走 `self.delete` 以触发子类覆盖的额外副作用，返回删除条数。 |
