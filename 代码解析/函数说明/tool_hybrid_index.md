# tool/hybrid_index.py

## 一、这个文件是干什么的

本文件是「混合索引」这条双写路径的唯一实现：把**一个**文本分块同时写进两套检索投影，并保证写入顺序与状态标记一致。第一套是**关键词真值源**，即 SQLite 的 `documents`/`chunks` 表，通过 `DocumentRepository.upsert_chunk` 落行，`chunks` 上的 FTS5 触发器会顺带维护关键词索引，因此真值行写进去就等于关键词可召回。第二套是**向量投影**，通过 `MemoryManager.add` 写入 `memories` 行并 upsert 到向量库。文件顶部注释明确了顺序不可颠倒：先写真值源、再写向量；反过来的话，向量写成功而真值行失败就会留下无法解释的孤立向量。写入成功后还会把该分块的 `vector_status` 置为 `indexed`，让 `/api/reconcile` 与重嵌入脚本能区分「已投影 / 待投影」。

对外暴露的核心函数是 `index_chunk`，它被 `memory.rag.pipeline.RAGPipeline.ingest` 调用，取代了原先内联的三步写入；此外还提供了一个可被 Agent 调用的工具类 `HybridIndexTool`（工具名 `knowledge.hybrid_index`），把同一套逻辑包装成带 Pydantic 输入输出模型的标准工具，供运行时发现与执行。文件中还包含 `repository_for`（从 manager 解析出真值源仓库）、`_default_manager`（懒加载默认 manager）、`_metadata_dict`（元数据列表转字典）、`create_tool`（工具工厂）等辅助件，以及 `IndexMetadataEntry`、`HybridIndexInput`、`HybridIndexOutput` 三个数据模型。

模块级还有两个常量：`TOOL_ENABLED = True` 表示该工具默认启用（供工具注册/发现机制判断）；`MAX_INDEX_CHARS = 200_000` 是单次索引允许的最大字符数，与 `WEB_KNOWLEDGE_MAX_CHARS` 同量级，用来防止无界输入。文件末尾的 `__all__` 列出了对外公开的名字。

## 二、函数与类逐条详解

### `class IndexMetadataEntry(BaseModel)` （第 37 行）
- **作用**：这是调用方传入的单条「元数据键值对」模型。之所以要专门定义一个类、而不是在输入模型里用开放的 `dict[str, str]`，是为了让工具的输入结构完全封闭、可被 schema 校验器严格检查：注释里写明「keeps the Input free of open dicts」。因为很多工具协议/函数调用框架对开放式字典支持不好，用键值对列表表达元数据既保持了灵活性，又能给每个字段加上长度约束和描述。它只在 `HybridIndexInput.metadata` 字段里被使用，是输入链路的最内层结构。
- **参数**：作为 Pydantic 模型没有显式 `__init__` 参数，其字段即构造参数。`key: str` 必填，长度 1 到 100，描述为「元数据键」，不能为空串；`value: str` 必填，长度上限 2000（没有下限，所以允许空串），描述为「元数据值」。两者都是 `str` 类型。
- **返回**：无返回值，构造出的是模型实例；后续由 `_metadata_dict` 把它的 `.key` 与 `.value` 取出来组装成普通字典。
- **内部流程**：类体里先声明 `model_config = ConfigDict(extra="forbid", strict=True)`。`extra="forbid"` 表示构造时出现未声明字段会直接报校验错误，`strict=True` 表示不做宽松类型强转（例如不会把数字 1 自动当成字符串 "1"）。接着用 `Field(...)` 声明 `key` 和 `value` 两个字段及其约束与 `description`。整个类没有定义任何方法，是纯数据模型。
- **异常/边界**：字段校验失败时由 Pydantic 抛出 `ValidationError`（例如 `key` 为空串、`key` 超过 100 字符、`value` 超过 2000 字符、传入未声明字段、类型不是 `str`）。本文件内部不做额外的异常捕获。
- **同文件关系**：被 `HybridIndexInput.metadata` 作为元素类型引用，被 `_metadata_dict` 读取其 `.key`/`.value`；不调用本文件任何函数。

### `_metadata_dict(entries: list[IndexMetadataEntry] | None) -> dict[str, str]` （第 46 行）
- **作用**：把工具输入里的「元数据键值对列表」压平成普通的 `dict[str, str]`，供后续写入向量侧元数据使用。`MemoryManager.add` 需要的是字典形态的 metadata，而工具输入出于 schema 封闭性的考虑用的是列表形态，这个小函数就是两者之间的适配层。它同时也是空值兜底点：调用方传 `None` 时不会崩，而是得到空字典。逻辑极短，但被 `HybridIndexTool.execute` 依赖，是每次工具调用的必经环节。
- **参数**：`entries`：`list[IndexMetadataEntry] | None`，调用方提供的元数据条目列表；允许为 `None`，也允许为空列表。
- **返回**：`dict[str, str]`。若 `entries` 为 `None` 或空列表，返回空字典 `{}`；否则返回按列表顺序构建的键值字典。
- **内部流程**：一行字典推导式：`(entries or [])` 先做空值兜底——`None` 和空列表都会被替换成空列表；然后遍历每一项，用 `entry.key` 作键、`entry.value` 作值。由于是字典推导，若列表中出现重复的 `key`，后面的值会覆盖前面的值（静默去重）。
- **异常/边界**：本身不抛异常；若列表元素不是 `IndexMetadataEntry`（缺少 `key`/`value` 属性）会抛 `AttributeError`，但正常情况下元素在进入本函数前已经过 Pydantic 校验。重复键静默覆盖，不做告警。
- **同文件关系**：读取 `IndexMetadataEntry` 的属性；被 `HybridIndexTool.execute` 调用；不调用本文件其他函数。

### `_default_manager() -> MemoryManager` （第 50 行）
- **作用**：构建（或取得）项目共享的、落盘的默认 `MemoryManager` 实例。之所以包一层函数而不是在模块顶层直接构造，是因为文档字符串点明了原因：`_memory` 模块会拉起 `memory.rag`，如果在模块导入期就引入会产生循环导入或提前初始化；因此这里改成函数内**延迟导入**（lazy import）。同时延迟构造也意味着「导入/发现这个工具」不会顺带打开 SQLite 连接，这对工具注册阶段的轻量性很重要。它只在工具被真正执行、且调用方没有注入 manager 时才被触发。
- **参数**：无参数。
- **返回**：`MemoryManager`，即 `build_default_manager()` 的返回值——一个已经配置好存储与向量后端的共享管理器。
- **内部流程**：第一步在函数体内执行 `from ._memory import build_default_manager`（相对导入，说明 `_memory` 与本文件同属 `tool` 包）；第二步直接 `return build_default_manager()`。没有任何缓存逻辑，因此每次调用都会让 `_memory` 那边决定是否复用单例。
- **异常/边界**：若 `tool._memory` 模块不存在或 `build_default_manager` 缺失，会抛 `ImportError`；若底层存储初始化失败（路径不可写、依赖缺失等），异常由 `build_default_manager` 向上抛，本函数不做捕获。无空值处理（无参数）。
- **同文件关系**：被 `HybridIndexTool.manager` 属性调用；调用本文件之外的 `tool._memory.build_default_manager`；不调用本文件其他函数。

### `repository_for(manager: MemoryManager) -> DocumentRepository | None` （第 58 行）
- **作用**：从给定的 `MemoryManager` 反推出承载 `documents`/`chunks` 真值源的 `DocumentRepository`，也就是关键词索引那一侧的入口。它的行为刻意与 `RAGPipeline.document_repo` 保持一致：当文档存储不是 SQLite、或者路径是内存库 `:memory:` 时返回 `None`。文档字符串解释了原因——再开一个内存数据库会是一个**私有连接**，它与被镜像的那个 store 之间不共享任何数据，写进去的关键词索引根本不会被真正使用，所以这种情况下宁可明确返回 `None`（表示关键词侧不可用），让上层降级为只写向量。
- **参数**：`manager`：`MemoryManager`，提供 `document_store` 属性的管理器实例；本函数通过 `getattr(manager.document_store, "path", None)` 读取其路径，因此 `document_store` 没有 `path` 属性也不会报错。
- **返回**：`DocumentRepository | None`。当路径存在且不等于字符串 `":memory:"` 时，返回 `DocumentRepository(str(path))` 新实例；否则（路径缺失、为空、为 `None`、或为 `:memory:`）返回 `None`。
- **内部流程**：先 `path = getattr(manager.document_store, "path", None)` 做安全取值；再用 `if not path or str(path) == ":memory:": return None` 判断——注意 `not path` 同时覆盖了 `None`、空字符串以及其他假值，`str(path)` 则兼容路径对象（如 `pathlib.Path`）并做字符串比较；最后用 `str(path)` 构造 `DocumentRepository` 返回。每次调用都会新建一个仓库对象，不做缓存。
- **异常/边界**：如果 `manager.document_store` 本身取不到（例如 manager 为 `None`），会在访问 `.document_store` 时抛 `AttributeError`；如果路径非空但 `DocumentRepository` 构造失败（如目录不存在、无权限），异常直接向上抛。空路径与内存库这两种边界被显式转为 `None` 而不是异常。
- **同文件关系**：被 `HybridIndexTool.execute` 调用；使用本文件导入的 `DocumentRepository`；不调用本文件其他函数。

### `index_chunk(manager, repository, *, chunk_id, document_id, chunk_index, char_start, char_end, text, metadata=None) -> MemoryItem` （第 72 行）
- **作用**：本模块的核心，也是整个项目「混合索引」双写路径的唯一实现。它把一个分块同时写进关键词真值源与向量投影，并返回向量侧产出的 `MemoryItem` 作为调用方唯一的事实来源。文档字符串强调了两点：其一，`repository=None` 表示关键词侧不可用（非 SQLite 或内存文档库），此时仍然照常写向量侧，这样调用方不会被环境差异打断；其二，返回的 item 就是调用方后续需要引用的对象。`memory.rag.pipeline.RAGPipeline.ingest` 不再内联这三步，而是调用它，从而保证全项目只有一处双写逻辑、只有一种顺序约定。
- **参数**：`manager: MemoryManager`，必填位置参数，负责向量侧的写入（`manager.add`）；`repository: DocumentRepository | None`，必填位置参数，关键词侧仓库，`None` 表示关键词侧不可用，此时跳过真值写入与状态标记；以下均为**仅关键字参数**（因为签名里有 `*` 分隔符）：`chunk_id: str` 分块唯一 id，同时被用作真值行主键与向量条目 id；`document_id: str` 所属文档 id；`chunk_index: int` 分块在文档内序号；`char_start: int` 分块在归一化正文中的起始偏移；`char_end: int` 结束偏移；`text: str` 分块原文，必须是非空字符串（会被 `strip()` 检查）；`metadata: Mapping[str, Any] | None = None` 附加元数据，默认 `None`，任意只读映射类型均可。
- **返回**：`MemoryItem`——`manager.add` 的返回值，携带新记忆条目的 id、类型、文本与元数据，供调用方继续使用（例如工具层取 `item.id` 作为 `memory_id`）。
- **内部流程**：第一步做入参校验：`if not isinstance(text, str) or not text.strip(): raise ValueError("text must be a non-empty string")`，即非字符串或纯空白字符串一律拒绝。第二步是**真值源写入**：仅当 `repository is not None` 时，构造 `ChunkRecord`（字段 `chunk_id`、`document_id`、`chunk_index=int(chunk_index)`、`char_start=int(char_start)`、`char_end=int(char_end)`、`text=text`），并调用 `repository.upsert_chunk(...)`；这里对三个偏移/序号字段显式 `int(...)` 转换，兼容传入数字字符串或布尔之类可转整型的值。第三步是**向量投影写入**：调用 `manager.add(text, memory_type=MemoryType.SEMANTIC, metadata=dict(metadata or {}), item_id=chunk_id)`，即固定语义记忆类型、把元数据拷贝成普通字典（`dict(...)` 避免持有调用方可变对象）、并把 `chunk_id` 作为记忆 id，从而实现「同一 chunk_id 重复索引即覆盖」。第四步是**状态标记**：仍然只在 `repository is not None` 时执行 `repository.set_chunk_vector_status(chunk_id, "indexed")`，把该分块标记为已投影。最后返回 `item`。整个顺序严格遵循模块文档的约定：真值源在前、向量在后，最后才打状态。
- **异常/边界**：`text` 非字符串或全空白时抛 `ValueError`；真值写入失败（`upsert_chunk` 抛异常）时向量侧**尚未写入**，不会产生孤立向量，异常向上抛；向量写入失败时真值行已存在但 `vector_status` 保持原值（不会被误标为 `indexed`），从而可被 `/api/reconcile` 或重嵌入脚本识别为待投影；`metadata` 为 `None` 时用空字典兜底；`repository` 为 `None` 时关键词侧整体跳过且不抛异常；`chunk_index`/`char_start`/`char_end` 若不可转整型会抛 `TypeError`/`ValueError`。
- **同文件关系**：被 `HybridIndexTool.execute` 调用，也被项目中的 `memory.rag.pipeline.RAGPipeline.ingest` 调用；使用本文件导入的 `ChunkRecord`、`MemoryType`；不调用本文件其他函数。

### `class HybridIndexInput(BaseModel)` （第 115 行）
- **作用**：`HybridIndexTool` 的输入契约模型，定义了 Agent 调用 `knowledge.hybrid_index` 时能传什么、每个字段的默认值和边界。它把「一段原文 + 文档定位信息 + 可选元数据」完整描述出来，让运行时能在真正执行前做参数校验并生成工具 schema。与 `IndexMetadataEntry` 一样，它用 `extra="forbid"` 与 `strict=True` 把输入收紧，避免多余字段或类型强转带来的隐蔽错误。
- **参数**：字段即参数。`text: str` 必填，长度 1 到 `MAX_INDEX_CHARS`（200000），是要入库的原文分块，同时成为关键词索引与向量投影的内容；`document_id: str | None = None` 所属文档 id，为空时会用 `source` 或 `'manual'` 生成稳定 id；`chunk_index: int = 0`，取值 0 到 1,000,000，分块序号从 0 开始；`char_start: int = 0`，`ge=0`，分块在文档归一化正文中的起始字符偏移；`char_end: int = 0`，`ge=0`，结束字符偏移（注意没有校验 `char_end >= char_start`）；`source: str = ""`，最长 500，来源标识（文件名或 URL），可为空串；`filename: str = ""`，最长 500，原始文件名，可为空串；`metadata: list[IndexMetadataEntry] | None = None`，附加元数据键值对，没有就传 `null`。
- **返回**：无返回值；构造出的实例被传入 `HybridIndexTool.execute`。
- **内部流程**：类体先设 `model_config = ConfigDict(extra="forbid", strict=True)`，再用 `Field(...)` 逐个声明字段与约束。`text` 用 `min_length=1`、`max_length=MAX_INDEX_CHARS` 限制体量；`chunk_index` 用 `ge=0`、`le=1_000_000` 限制范围；两个偏移字段只做 `ge=0` 下限约束；字符串字段用 `max_length` 限长；`metadata` 以列表形式承载键值对。类内没有方法，纯数据模型。
- **异常/边界**：任何字段越界（`text` 为空或超过 20 万字符、`chunk_index` 为负或超百万、偏移为负、字符串超长、出现未知字段、类型不匹配）都会由 Pydantic 抛 `ValidationError`；`document_id` 为 `None` 是合法输入，由 `execute` 负责兜底生成；`source`/`filename` 允许空串。本模型不校验 `char_start`/`char_end` 的先后关系。
- **同文件关系**：被 `HybridIndexTool.spec` 的 `input_model` 引用、被 `HybridIndexTool.execute` 的 `arguments` 形参标注使用；引用 `IndexMetadataEntry` 与 `MAX_INDEX_CHARS`；不调用本文件任何函数。

### `class HybridIndexOutput(BaseModel)` （第 145 行）
- **作用**：`HybridIndexTool` 的输出契约模型，规定工具执行完成后返回给调用方/运行时的结构化结果。它把「这次索引到底成功了哪几侧」显式暴露出来：`keyword_indexed` 说明关键词真值源是否写入，`vector_indexed` 说明向量投影是否写入，`vector_status` 给出该分块的投影状态标记。这样上层（尤其是 Agent 的后续推理和 `/api/reconcile` 类的对账逻辑）无需猜测环境差异带来的降级行为。它同样用 `extra="forbid"`、`strict=True` 收紧结构。
- **参数**：字段即构造参数，且**全部必填**（没有默认值）。`chunk_id: str` 实际使用的分块 id；`document_id: str` 实际使用的文档 id；`memory_id: str` 向量侧记忆条目 id；`keyword_indexed: bool` 真值源（FTS5 关键词索引）是否写入成功；`vector_indexed: bool` 向量投影是否写入成功；`vector_status: Literal["indexed", "pending"]` 只能是这两个字面量之一；`character_count: int` 入库文本的字符数。
- **返回**：无返回值；实例作为 `execute` 的返回值交给工具框架序列化。
- **内部流程**：类体只做两件事——设置 `model_config = ConfigDict(extra="forbid", strict=True)`，以及声明上述字段与 `description`（其中 `keyword_indexed`、`vector_indexed` 带中文/英文描述说明）。`vector_status` 用 `Literal["indexed", "pending"]` 做枚举约束，保证不会出现第三种状态字符串。类内没有方法。
- **异常/边界**：缺字段、多字段、类型不符或 `vector_status` 传入 `"indexed"`/`"pending"` 之外的值都会抛 `ValidationError`；本模型自身不做任何业务判断，语义正确性由 `execute` 保证。
- **同文件关系**：被 `HybridIndexTool.spec` 的 `output_model` 引用、被 `HybridIndexTool.execute` 构造并返回；不调用本文件任何函数。

### `class HybridIndexTool(BaseTool)` （第 157 行）
- **作用**：把 `index_chunk` 这条双写路径包装成运行时/Agent 可调用的标准工具。类体上声明了 `spec = ToolSpec(...)`，即工具的元信息：名字 `knowledge.hybrid_index`，描述说明「把一个文本分块同时索引进关键词真值源（SQLite chunks + FTS5）与向量投影，适用于原文必须同时可按精确词与按语义召回的场景，且它不跑 LLM 抽取、不创建图谱事实」；版本 `1.0.0`；输入输出模型分别是 `HybridIndexInput`/`HybridIndexOutput`；`side_effect="write"` 表明是写操作；`permissions=()` 表示不需要额外权限声明；`timeout_seconds=60.0`；`idempotent=True`（同一 chunk_id 重复索引是覆盖写）；`parallel_safe=False`（涉及数据库写，不宣称可并行）；`tags` 为 `("memory", "index", "hybrid", "fts5", "vector", "write")`；`guidance` 给出使用边界——需要让原文同时可被关键词与语义检索时用它，不要用它写记忆条目或抽知识（走 `memory.rag` 与 `knowledge.add_fact`），也不要用它改图节点属性（走 `knowledge.graph_node_update`），并提示写操作需要人工确认钥匙。
- **参数**：类本身无构造签名（继承 `BaseTool`），实例化走下方 `__init__`。
- **返回**：无（类定义）。
- **内部流程**：类体先构造并赋值类属性 `spec`，随后定义 `__init__`、`manager` 属性与 `execute` 方法。除 `spec` 外没有其他类级状态，实例状态只有 `self._manager`。
- **异常/边界**：类定义阶段若 `ToolSpec` 参数不合法会在导入时报错；除此之外无特殊处理。
- **同文件关系**：继承导入的 `BaseTool`；使用 `ToolSpec`、`HybridIndexInput`、`HybridIndexOutput`；其方法调用 `_metadata_dict`、`repository_for`、`index_chunk`、`_default_manager`；被 `create_tool` 实例化。

#### `__init__(self, manager: MemoryManager | None = None) -> None` （第 182 行）
- **作用**：工具构造函数，只做一件事——把外部可选注入的 `MemoryManager` 记到实例上，其余什么都不做。注释写明「Lazily built so importing/discovering the tool never opens SQLite」：延迟构建是为了让工具在导入/发现阶段不打开 SQLite 连接，避免注册工具时就产生副作用与性能开销。因此注入的 manager 会被原样保存，真正的默认 manager 要等到 `manager` 属性第一次被访问时才构造。这让测试可以传入替身 manager，也让生产环境保持惰性。
- **参数**：`manager: MemoryManager | None = None`，可选注入的管理器；传 `None`（默认）表示稍后懒加载默认 manager。
- **返回**：`None`（构造函数）。
- **内部流程**：唯一语句 `self._manager = manager`，不做校验、不做连接、不做缓存预热。
- **异常/边界**：无特殊处理；传入任意对象都不会在此处报错（错误会推迟到实际使用 manager 时暴露）。
- **同文件关系**：不调用本文件其他函数；被 `create_tool` 以无参形式调用。

#### `@property manager(self) -> MemoryManager` （第 186 行）
- **作用**：以只读属性形式对外提供 `MemoryManager`，并实现懒初始化。调用方（`execute`）只需要读 `self.manager`，不必关心是否注入过；如果没有注入，这里会在首次访问时通过 `_default_manager()` 构造共享默认管理器并把结果缓存回 `self._manager`，从而后续访问不再重复构造。这个属性是把「构造零副作用」与「使用时可获得 manager」两个目标缝合起来的关键。
- **参数**：无（`self` 由属性访问隐式传入）。
- **返回**：`MemoryManager`。若 `self._manager` 非空直接返回它；否则构造默认 manager 后返回，并已写回 `self._manager`。
- **内部流程**：`if self._manager is None: self._manager = _default_manager()`；随后 `return self._manager`。缓存逻辑就写在这个分支赋值上，没有额外锁（并发首次访问可能构造两次，但只有一次结果被保留）。
- **异常/边界**：若 `_default_manager()` 内部导入或存储初始化失败，异常会从属性访问处抛出（即 `execute` 中 `manager = self.manager` 那一行）；属性本身不做捕获。
- **同文件关系**：调用 `_default_manager`；被 `HybridIndexTool.execute` 通过 `self.manager` 访问。

#### `execute(self, arguments: HybridIndexInput) -> HybridIndexOutput` （第 192 行）
- **作用**：工具的执行入口，把校验过的输入翻译成 `index_chunk` 所需的参数，然后调用双写核心，最后把结果包装成结构化输出。它承担了「默认值推导」的职责：`document_id` 为空时依次回退到 `source`、`filename`、字面量 `"manual"`；`chunk_id` 由 `document_id` 与 `chunk_index` 拼接成 `"{document_id}:{chunk_index}"`，因此同一文档同一序号天然幂等。它还负责补齐元数据（把 source、filename、document_id、chunk_index、char_start、char_end 塞进去，但都用 `setdefault`，尊重调用方显式传入的值），并解析出关键词侧仓库。最后根据 `repository is not None` 决定 `keyword_indexed` 与 `vector_status` 的取值。
- **参数**：`arguments: HybridIndexInput`，已完成 schema 校验的输入对象，携带 `text`、`document_id`、`chunk_index`、`char_start`、`char_end`、`source`、`filename`、`metadata`。
- **返回**：`HybridIndexOutput`，字段为：`chunk_id`（拼接得到的 id）、`document_id`（回退后的 id）、`memory_id`（取自 `item.id`）、`keyword_indexed`（`repository is not None`）、`vector_indexed=True`（能走到这里说明 `manager.add` 未抛异常）、`vector_status`（有仓库时 `"indexed"`，否则 `"pending"`）、`character_count`（`len(arguments.text)`）。
- **内部流程**：第一步推导 `document_id`：`arguments.document_id or arguments.source or arguments.filename or "manual"`，逐级回退，全空则用 `"manual"`。第二步拼 `chunk_id = f"{document_id}:{arguments.chunk_index}"`。第三步构造 `metadata: dict[str, Any] = dict(_metadata_dict(arguments.metadata))`，把调用方元数据列表转成字典；随后依次 `setdefault`：`source` 为 `arguments.source or document_id`（source 为空串时用 document_id 顶替）、`filename` 仅在非空时写入、`document_id`、`chunk_index`、`char_start`、`char_end`。第四步 `manager = self.manager` 取管理器，`repository = repository_for(manager)` 取真值源（可能为 `None`）。第五步调用 `index_chunk(manager, repository, chunk_id=..., document_id=..., chunk_index=..., char_start=..., char_end=..., text=arguments.text, metadata=metadata)`，拿到 `item`。第六步构造并返回 `HybridIndexOutput`，其中 `vector_status` 用三元表达式 `"indexed" if repository is not None else "pending"` 表达关键词侧缺失时的降级状态。
- **异常/边界**：`text` 全空白会由 `index_chunk` 抛 `ValueError`（Pydantic 的 `min_length=1` 只挡空串，挡不住纯空格）；真值源写入失败或向量写入失败都会让异常直接向上抛，不返回部分成功的输出；`arguments.metadata` 为 `None` 时 `_metadata_dict` 返回空字典；`document_id`、`source`、`filename` 全空时回退到 `"manual"`；`repository` 为 `None`（非 SQLite 或内存库）时不报错，而是返回 `keyword_indexed=False`、`vector_status="pending"`。注意 `vector_indexed` 被硬编码为 `True`，属于「走到这一步即视为成功」的约定，而不是查询真实状态。
- **同文件关系**：调用 `_metadata_dict`、`repository_for`、`index_chunk`，并通过 `self.manager` 间接调用 `_default_manager`；构造 `HybridIndexOutput`；被工具运行时/Agent 调用。

### `create_tool() -> BaseTool` （第 232 行）
- **作用**：工具工厂函数，供项目里的工具注册/发现机制以统一约定实例化工具。它返回一个**未注入 manager** 的 `HybridIndexTool`，也就是说 manager 会保持懒加载，注册阶段不打开任何存储，真正执行时才构造。把它单独抽出来是因为很多工具加载器按「模块提供 `create_tool()`」的约定扫描，这样 `tool/hybrid_index.py` 无需暴露类的构造细节即可被挂载。
- **参数**：无参数。
- **返回**：`BaseTool`，实际类型是 `HybridIndexTool`（以基类类型标注，便于调用方统一处理）。
- **内部流程**：单行 `return HybridIndexTool()`；不传 manager，也不做任何配置或注册副作用。
- **异常/边界**：无特殊处理；`HybridIndexTool()` 的构造本身几乎不会失败。
- **同文件关系**：实例化 `HybridIndexTool`；不调用本文件其他函数；在 `__all__` 中导出。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `IndexMetadataEntry` | 单条元数据键值对模型，用封闭字段替代开放字典并限制 key/value 长度。 |
| `_metadata_dict` | 把元数据条目列表压平成 `dict[str, str]`，`None` 或空列表返回空字典。 |
| `_default_manager` | 延迟导入并构造共享的默认 `MemoryManager`，避免导入期打开 SQLite。 |
| `repository_for` | 从 manager 的文档存储路径解析出 `DocumentRepository`，非 SQLite 或 `:memory:` 时返回 `None`。 |
| `index_chunk` | 核心双写函数：先写真值源 `chunks`，再写向量投影，最后标记 `vector_status=indexed`，返回 `MemoryItem`。 |
| `HybridIndexInput` | 工具输入模型，定义原文、文档定位字段、来源信息与元数据的校验约束。 |
| `HybridIndexOutput` | 工具输出模型，回报 chunk/document/memory id、两侧是否索引成功与投影状态。 |
| `HybridIndexTool` | 把双写逻辑包装成名为 `knowledge.hybrid_index` 的写类工具，含 `spec` 元信息与 guidance 边界说明。 |
| `HybridIndexTool.__init__` | 仅保存可选注入的 manager，保持构造零副作用与懒加载。 |
| `HybridIndexTool.manager` | 只读属性，首次访问时懒加载并缓存默认 `MemoryManager`。 |
| `HybridIndexTool.execute` | 推导 document_id/chunk_id 与元数据，调用 `index_chunk` 双写并返回结构化结果。 |
| `create_tool` | 工具工厂，返回未注入 manager 的 `HybridIndexTool` 实例供注册发现。 |
