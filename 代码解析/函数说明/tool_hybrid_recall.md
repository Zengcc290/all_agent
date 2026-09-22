# tool/hybrid_recall.py

## 一、这个文件是干什么的

这个文件实现了一条「混合召回」检索路径，也就是把**向量语义检索**和 **FTS5 关键词检索**两条通路的结果用 **RRF（Reciprocal Rank Fusion，倒数名次融合）** 合并成一份统一的分块（chunk）命中列表。之所以要两路并行，是因为精确词（型号、编号、代码标识符）在向量空间里区分度很差，而纯转述的问句又只有向量路能召回，两路天然互补；同时任何一路不可用都不允许让整条检索链路失败——向量路异常时本工具会降级为纯关键词（FTS5），并把降级原因写进返回值的 `note` 字段，供 UI 和健康检查如实展示。融合之所以用 RRF，是因为余弦相似度和 SQLite FTS5 的 bm25 分属于两套完全不同的量纲，直接相加没有意义，RRF 只看名次、不看绝对分值，因此天然可比。

文件内容由四块组成：一是几个纯函数形式的检索原语（`rrf_fuse` 做名次融合、`chunk_metadata` 抽取分块元数据、`hybrid_enabled` 读开关、`vector_hits` 走向量路、`fuse_hits` 做融合并回查正文、`hybrid_recall` 是编排整条链路的主入口）；二是数据载体（冻结 dataclass `HybridRecallResult`，以及三个 Pydantic 模型 `HybridRecallInput` / `HybridHit` / `HybridRecallOutput`）；三是把上面这些包装成 Agent 可调用工具的 `HybridRecallTool`（内含 `__init__`、`pipeline` 属性、`execute` 方法）；四是给工具注册表用的工厂函数 `create_tool` 与模块导出清单 `__all__`。

运行期它是被这样用到的：Agent 在回答「知识库里是怎么说的」这类问题时调用名为 `knowledge.hybrid_recall` 的工具（`web/app.py` 的聊天溯源面板也走同一条路），`HybridRecallTool.execute` 把入参交给模块级的 `hybrid_recall`，后者拿 `pipeline` 里的文档仓储去跑双路检索并融合，最后把结果重新包装成 `HybridRecallOutput` 返回给上层。模块头注释还特别声明：本模块是这条召回路径的**唯一实现**，`memory.rag.pipeline.RAGPipeline` 已经不再持有 `hybrid_retrieve` / `_vector_hits` / `_rrf_fuse` / `last_retrieval_note` 这些成员，调用方统一改为调用 `hybrid_recall`。模块级常量有两个：`TOOL_ENABLED = True`（工具默认启用标记）和 `RRF_K = 60`（RRF 名次平滑常数，取值与信息检索领域常用值一致）。

## 二、函数与类逐条详解

### `_default_pipeline() -> Any` （第 38 行）
- **作用**：构造「默认管线」对象，也就是当 `HybridRecallTool` 没有被外部注入 `pipeline` 时，用它去惰性获取一个可用的 `RAGPipeline`（或等价的管线对象）。之所以要单独抽成一个函数并做**惰性导入**，是因为 `._memory` 模块会去拉 `memory.rag`，而 `memory.rag` 的导入链较重、可能反过来牵扯本模块，如果在文件顶层直接 `import` 会造成循环导入或拖慢模块加载；放进函数体里就只有在真正需要默认管线时才付出这份导入代价。它只在 `HybridRecallTool.pipeline` 属性发现 `self._pipeline is None` 时被调用，属于「兜底装配」路径；测试或上层显式传入管线时根本不会走到这里。
- **参数**：无参数。
- **返回**：返回 `build_default_pipeline()` 的返回值，类型标注为 `Any`，实际应是项目里默认构造出来的检索管线对象（具备 `manager`、`document_repo()`、`retrieve()` 等成员，因为后续 `hybrid_recall` 会按这些接口使用它）。
- **内部流程**：函数体只有两步：先执行 `from ._memory import build_default_pipeline` 这个相对导入（延迟到调用时才解析），然后直接 `return build_default_pipeline()` 把构造结果交出去。没有任何缓存逻辑，也就是说每次被调用都会重新走一遍导入语句和构造函数（导入本身由 Python 的模块缓存兜住，重复开销主要在构造上）。
- **异常/边界**：本身不做任何 try/except；如果 `tool._memory` 模块缺失、`build_default_pipeline` 不存在，或者默认管线构造过程中依赖的配置/资源不可用，异常会原样向上抛出（典型为 `ImportError` / `ModuleNotFoundError` 或构造阶段抛出的运行时异常）。没有空值兜底，也不会返回 `None`。
- **同文件关系**：调用了本文件之外的 `tool._memory.build_default_pipeline`（通过惰性导入）。被本文件里的 `HybridRecallTool.pipeline` 属性调用；同文件其它函数都不调用它。

### `HybridRecallResult` （第 46 行，`@dataclass(frozen=True)` 数据类）
- **作用**：混合召回的**结果载体**，把「融合后的分块列表」「向量路降级原因」「向量路是否可用」三件事打包成一个不可变对象，供 `hybrid_recall` 返回、供调用方读取。它存在的意义是让函数返回值带上下文：调用方不只知道召回了哪些块，还知道这次是不是降级跑的、为什么降级，从而能在回答里如实告知用户检索已降级，而不是把纯关键词结果冒充成混合结果。因为是 `frozen=True` 的 dataclass，实例创建后字段不可被改写，可以安全地跨线程/跨协程传递，也不会被下游误改。
- **参数（字段）**：字段共三个。`chunks: list[RetrievedChunk]`，默认由 `field(default_factory=list)` 生成空列表（用 `default_factory` 而不是可变默认值，避免所有实例共享同一个 list）；`note: str`，默认空串 `""`，用来装向量路失败的降级说明；`vector_available: bool`，默认 `True`，表示向量路是否正常。类的构造方法 `__init__` 由 dataclass 自动生成，按 `(chunks, note, vector_available)` 顺序接受这三个关键字/位置参数。
- **返回**：这是类，不返回值；实例化后得到的结果对象由 `hybrid_recall` 返回给调用方。
- **内部流程**：`@dataclass(frozen=True)` 装饰器在类定义时自动生成 `__init__`、`__repr__`、`__eq__`，并禁止属性赋值（赋值会抛 `dataclasses.FrozenInstanceError`）。类体里只做字段声明，没有自定义方法、没有 `__post_init__`、没有校验逻辑，因此不做任何字段类型或取值范围检查。
- **异常/边界**：构造时对字段不做校验，传 `None` 进 `chunks` 也会被接受（只是后续遍历会出错）；尝试修改已有实例的字段会抛 `FrozenInstanceError`。无其它特殊处理。
- **同文件关系**：不调用本文件任何函数；被 `hybrid_recall` 构造并返回，被 `HybridRecallTool.execute` 间接消费（读取其 `chunks` / `note` / `vector_available`），也出现在模块 `__all__` 导出清单里。

### `rrf_fuse(rank_lists: list[list[str]], *, k: int = RRF_K) -> list[tuple[str, float]]` （第 55 行）
- **作用**：实现 Reciprocal Rank Fusion 融合算法，把若干条「只含 chunk_id 的排名列表」合并成一条统一排名。它的核心价值在于**只按名次计分**：每个 chunk 在每一路里的贡献是 `1/(k + rank)`，rank 从 1 开始，所以排得越靠前的块得分越高、排到很后面的块得分衰减得很平缓；这样就不必把余弦相似度和 bm25 两套量纲不同的原始分数放在一起相加。当某个 chunk 同时被向量路和关键词路命中时，它的得分是两路贡献之和，自然被顶到前面，这正是混合召回想要的「双路都认它就更可信」的效果。它被 `fuse_hits` 在每次融合时调用。
- **参数**：`rank_lists: list[list[str]]`，外层列表的每一项是一路检索的命中顺序，内层是 chunk_id 字符串；顺序即名次，列表可以长度不同，也可以为空列表，甚至可以整体传空列表。关键字参数 `k: int = RRF_K`（默认 60）是名次平滑常数，取值越大则名次之间的差距被压得越平（高分与低分更接近），取值越小则头部名次的优势越突出；约定为正整数，函数本身不校验。
- **返回**：返回 `list[tuple[str, float]]`，每个元素是 `(chunk_id, 融合得分)`，按「得分降序、chunk_id 升序」排序（即 `key=lambda kv: (-kv[1], kv[0])`，第二排序键保证得分相同时结果稳定可复现）。没有任何命中时返回空列表。
- **内部流程**：先建一个空字典 `scores: dict[str, float]`；然后对 `rank_lists` 里每一路 `hits` 做遍历，用 `enumerate(hits, start=1)` 拿到从 1 开始的 `rank` 和 `chunk_id`，执行累加 `scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)`；最后 `sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))` 返回排序后的 `(id, 分)` 列表。注意同一个 chunk_id 在同一路列表里出现多次会被重复计分（不去重）。
- **异常/边界**：不做入参校验。`k` 传成负值或使 `k + rank` 为 0 时会触发 `ZeroDivisionError`；`rank_lists` 传 `None` 会在遍历时抛 `TypeError`；空列表输入返回空列表；同一个 id 重复出现会累加分数；排序在得分完全相同时靠 id 字典序兜底，因此结果确定。除上述外无特殊处理。
- **同文件关系**：调用了本文件之外的 Python 内置 `sorted`。被本文件里的 `fuse_hits` 调用（`fuse_hits` 只传 `rank_lists`，`k` 用默认的 `RRF_K`）；同时出现在 `__all__` 里对外导出，方便测试与复用。

### `chunk_metadata(chunk: ChunkRecord) -> dict[str, Any]` （第 65 行）
- **作用**：把存储层的 `ChunkRecord`（分块真值记录）里跟「定位」有关的四个字段抽成一个普通字典，供上层（尤其是 `HybridRecallTool.execute` 以及前端展示）读取。之所以要做这层「翻译」，是因为检索结果对象 `RetrievedChunk` 的 `metadata` 是一个宽松的 `Mapping`，把 ORM/记录对象的属性显式映射成固定键名的字典，可以让下游不必依赖 `ChunkRecord` 的内部结构，键名也统一成对外契约（`document_id` / `chunk_index` / `char_start` / `char_end`），前端据此就能实现「跳到原文某一段」的高亮。它被 `fuse_hits` 在构造每条 `RetrievedChunk` 时调用。
- **参数**：`chunk: ChunkRecord`，一个分块记录对象，要求具备 `document_id`、`chunk_index`、`char_start`、`char_end` 四个属性（分别表示所属文档 ID、块在文档内的序号、该块在原文中的起始字符偏移、结束字符偏移）。函数不校验它是否为 `None`。
- **返回**：返回 `dict[str, Any]`，恰好四个键：`"document_id"`、`"chunk_index"`、`"char_start"`、`"char_end"`，值直接取自入参对象的同名属性（保持原类型，通常是字符串/整数/可能为 `None`）。
- **内部流程**：没有任何分支或循环，直接构造并返回一个字面量字典，四个键的值分别读 `chunk.document_id`、`chunk.chunk_index`、`chunk.char_start`、`chunk.char_end`。纯读取，无副作用。
- **异常/边界**：如果传入 `None` 或对象缺少上述任一属性，会抛 `AttributeError`；不做键名过滤，属性值为 `None` 时也会照原样放进字典；无特殊处理。
- **同文件关系**：不调用本文件其它函数。被本文件里的 `fuse_hits` 调用（把结果塞进 `RetrievedChunk` 的 metadata 位置参数），其结果随后被 `HybridRecallTool.execute` 通过 `chunk.metadata.get("document_id")` / `chunk.metadata.get("chunk_index")` 读取；同时出现在 `__all__` 导出清单里。

### `hybrid_enabled() -> bool` （第 74 行）
- **作用**：读取「是否启用混合检索」的全局开关。它把 `constants.MEMORY_HYBRID` 这个模块级配置包成一个函数调用点，好处是调用方（`hybrid_recall`）不必直接依赖 `constants` 模块、也让开关在运行期具备被替换/打桩（monkeypatch）的能力。语义是：默认开启，走「向量 + FTS5 双路融合」；一旦关闭，`hybrid_recall` 就退回纯向量检索（直接调 `pipeline.retrieve`），此时 FTS5 关键词路完全不参与。它只在 `hybrid_recall` 的入口判断里被调用一次。
- **参数**：无参数。
- **返回**：返回 `bool`，即 `MEMORY_HYBRID` 的值本身。按设计它是布尔量，但函数不做 `bool()` 强制转换，若配置被写成非布尔真值（例如 `1` 或非空字符串）会原样返回，在 `if not hybrid_enabled()` 处仍按真值语义判断。
- **内部流程**：单行实现，直接 `return MEMORY_HYBRID`，没有缓存、没有默认值兜底、没有异常捕获。`MEMORY_HYBRID` 在文件顶层通过 `from constants import MEMORY_HYBRID, RAG_RETRIEVE_LIMIT` 导入。
- **异常/边界**：若 `constants` 模块缺少 `MEMORY_HYBRID`，问题会在模块导入阶段就以 `ImportError` 暴露，而不是在调用本函数时；函数本身无特殊处理。
- **同文件关系**：不调用本文件任何函数；被本文件里的 `hybrid_recall` 调用（用于决定走融合路径还是纯向量回退路径）；同时出现在 `__all__` 导出清单里。

### `vector_hits(pipeline: Any, query: str, *, limit: int, threshold: float | None, metadata: Mapping[str, Any] | None) -> tuple[list[tuple[str, float]], str]` （第 80 行）
- **作用**：执行混合召回的**向量路**：拿查询串去语义记忆里做相似度检索，把命中结果规整成 `(chunk_id, 相似度)` 的二元组列表返回；如果向量端点不可用或检索过程出错，则返回空列表并附上一段中文降级说明，让整条链路可以无缝切到纯关键词。它把「失败」当成一种正常结果而不是异常抛出，这是本文件「任一投影不可用都不能让检索整体失败」这一设计原则的落点。注释还特别说明：这里不再做请求前的 TCP 可达性探测（那是历史隧道网关的产物），因为云端端点一次 `urlopen` 的代价和探测一样，失败时异常本身就是最好的降级信号。它由 `hybrid_recall` 在启用混合检索且仓储可用时调用。
- **参数**：`pipeline: Any`，检索管线对象，函数只用到它的 `manager.search(...)` 接口。`query: str`，检索问句。关键字参数三个：`limit: int`，向量路要取的条数（`hybrid_recall` 会传 `limit * 2`，为融合多留候选）；`threshold: float | None`，相似度阈值，`None` 表示不设阈值、交给下游默认；`metadata: Mapping[str, Any] | None`，元数据过滤条件（如限定某文档/某类记忆），`None` 表示不过滤。
- **返回**：返回二元组 `(hits, note)`。`hits` 是 `list[tuple[str, float]]`，元素为 `(result.item.id, float(result.score))`，即分块 ID 与浮点相似度；`note` 是 `str`，成功时为空串 `""`，失败时是形如「向量检索失败，本次检索降级为纯关键词（FTS5）：<异常类名>: <异常信息>」的说明文本。
- **内部流程**：整体包在 `try` 里调用 `pipeline.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=limit, threshold=threshold, metadata=metadata)`，也就是明确指定只检索 `MemoryType.SEMANTIC`（语义记忆）类型；成功则用列表推导把每个 `result` 拆成 `(result.item.id, float(result.score))` 并配空 `note` 返回。若捕获到异常，则用 `type(exc).__name__` 和 `str(exc)` 拼出降级说明字符串，返回 `([], note)`。
- **异常/边界**：只捕获 `ConnectionError`、`OSError`、`RuntimeError` 三类（覆盖网络连接失败、文件/套接字层错误、以及运行时类错误，例如底层客户端抛出的通用运行时异常）；其它异常（如 `AttributeError`、`KeyError`、`TypeError`、`ValueError`）不在捕获范围内，会向上冒泡并可能让整次工具调用失败。返回的 `note` 非空即代表降级，这一点被 `hybrid_recall` 用来设置 `vector_available`。没有超时参数，超时行为取决于底层 `search` 实现。
- **同文件关系**：调用了本文件之外的 `MemoryType.SEMANTIC` 与 `pipeline.manager.search`。被本文件里的 `hybrid_recall` 调用；它自身不调用本文件其它函数；名字出现在 `__all__` 导出清单里。

### `fuse_hits(repository: Any, rank_lists: list[list[str]], scores: tuple[dict[str, float], dict[str, float]], *, limit: int) -> list[RetrievedChunk]` （第 111 行）
- **作用**：把两路的名次做 RRF 融合，并按**真值源回查正文**，组装成最终的 `RetrievedChunk` 列表。这里有一个重要的安全/一致性约束：向量库（向量投影）里存在的分块，如果在真值源（文档仓储）里查不到，就**直接跳过、不返回**，避免「孤立向量」把已经不存在的旧内容泄露出去。同时它会把两路的原始分数（向量相似度、FTS5 bm25）作为 `detail` 一起带上，这样对外暴露的 `score` 虽然是 RRF 分，但贡献来源仍然可见、可解释。它由 `hybrid_recall` 在双路结果都拿到之后调用。
- **参数**：`repository: Any`，文档仓储对象，需要提供 `get_chunk(chunk_id)` 用于按 ID 回查分块。`rank_lists: list[list[str]]`，两路命中的 chunk_id 排名列表（约定第一路是向量、第二路是关键词，顺序只影响可读性不影响 RRF 计分）。`scores: tuple[dict[str, float], dict[str, float]]`，与 `rank_lists` 对应的两路原始分字典，元组解包为 `(vector_scores, keyword_scores)`，仅用于填充 `detail`。关键字参数 `limit: int`，最终返回条数上限。
- **返回**：返回 `list[RetrievedChunk]`。每个元素的构造位置参数依次是：分块正文 `chunk.text`、RRF 得分 `score`、分块 ID `chunk.chunk_id`、元数据 `chunk_metadata(chunk)`、以及关键字参数 `detail`（含 `rrf_score`、`vector_score`、`keyword_score` 三个键）。真值源查不到或全部被跳过的极端情况下返回空列表。
- **内部流程**：先解包 `vector_scores, keyword_scores = scores`；初始化空列表 `results`；调用 `rrf_fuse(rank_lists)`（`k` 用默认 `RRF_K`）拿到全局排序，切片 `[:limit]` 只取前 `limit` 条；对每一条 `(chunk_id, score)` 调 `repository.get_chunk(chunk_id)`，若返回 `None` 就 `continue` 跳过；否则 append 一个 `RetrievedChunk`，其中 `detail["rrf_score"]` 写 `float(score)`（注释说明它与对外 `score` 同源同值，这里不做四舍五入，展示精度交给前端），`detail["vector_score"]` 和 `detail["keyword_score"]` 分别用 `vector_scores.get(chunk_id)` / `keyword_scores.get(chunk_id)` 取值——某一路没命中该块时对应的就是 `None`。
- **异常/边界**：不做入参校验，`scores` 不是两元组会在解包处抛 `ValueError`，`repository.get_chunk` 抛出的异常不捕获；真值源缺失分块的处理方式是静默跳过（不报错、不补位，因此最终条数可能少于 `limit`）；`limit` 传 0 或负数时切片结果为空列表。无其它特殊处理。
- **同文件关系**：调用了本文件里的 `rrf_fuse`（做名次融合）和 `chunk_metadata`（抽元数据）。被本文件里的 `hybrid_recall` 调用；它构造出来的 `RetrievedChunk.detail` 会被 `HybridRecallTool.execute` 读取（取 `rrf_score` / `vector_score` / `keyword_score`）；名字出现在 `__all__` 导出清单里。

### `hybrid_recall(pipeline: Any, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> HybridRecallResult` （第 143 行）
- **作用**：这是整条混合召回链路的**主编排函数**，也是模块对外声明的唯一实现入口（`web/app.py` 的聊天溯源面板就是直接调它）。它负责决定走哪条路径：如果混合开关被关掉、或者拿不到文档仓储，就退回纯向量检索；否则并发走「向量路 + FTS5 关键词路」，把两路名次用 RRF 融合、按真值源回查正文，最后连同「向量路是否可用/为什么降级」一起打包返回。它让上层只面对一次调用、一种返回结构，不必关心两路检索的差异与降级细节。
- **参数**：`pipeline: Any`，检索管线对象，需要具备 `document_repo()`（返回文档仓储或 `None`）、`retrieve(...)`（纯向量回退用）以及向量路所需的 `manager`。`query: str`，检索问句。关键字参数：`limit: int = RAG_RETRIEVE_LIMIT`（来自 `constants`，返回条数上限）；`threshold: float | None = None`（相似度阈值，`None` 表示不设）；`metadata: Mapping[str, Any] | None = None`（元数据过滤条件）。
- **返回**：返回 `HybridRecallResult`。三种情形：纯向量回退时 `chunks` 为 `pipeline.retrieve(...)` 的结果、`note=""`、`vector_available=True`；正常双路融合时 `chunks` 为 `fuse_hits` 的结果、`note` 为 `vector_hits` 返回的降级说明（成功时是空串）、`vector_available=not note`；向量路失败但仓储可用时，仍然会把关键词路的命中融合后返回，只是 `note` 非空、`vector_available=False`。
- **内部流程**：第一步 `repository = pipeline.document_repo()`；第二步判断 `if not hybrid_enabled() or repository is None:`，命中则直接返回纯向量结果（`chunks=pipeline.retrieve(query, limit=limit, threshold=threshold, metadata=metadata)`，`note=""`，`vector_available=True`）；第三步调 `vector_hits(pipeline, query, limit=limit * 2, threshold=threshold, metadata=metadata)`，把返回解包成 `hits, note`——注意这里刻意把候选数放大一倍，给融合留出更多候选；第四步调 `repository.search_keywords(query, limit=limit * 2)` 拿关键词路结果，并用列表推导转成 `(chunk_id, float(score))` 形式的 `keyword_hits`；第五步调 `fuse_hits(repository, [[chunk_id for chunk_id, _ in hits], [chunk_id for chunk_id, _ in keyword_hits]], (dict(hits), dict(keyword_hits)), limit=limit)`——把两路拆成两个纯 ID 排名列表喂给 RRF，同时把两路原始分转成字典留作 `detail`（注释标为 U4：两路原始分数在融合前留一份，否则 RRF 只留下名次、贡献不可见）；第六步返回 `HybridRecallResult(chunks=chunks, note=note, vector_available=not note)`。
- **异常/边界**：向量路自身的网络/运行时异常已被 `vector_hits` 内部吞掉并转成 `note`，所以这条路径不会因为向量端点挂掉而失败；但 `repository.search_keywords`、`repository.get_chunk`、`pipeline.document_repo()`、`pipeline.retrieve(...)` 抛出的异常都没有捕获，会向上冒泡。关键词路返回空时 `keyword_hits` 为空列表，RRF 仍能只用向量路名次完成排序；两路都空时返回空的 `chunks`。`limit` 未做下界校验（传 0 会得到空结果），`threshold` / `metadata` 为 `None` 时按底层默认处理。
- **同文件关系**：调用了本文件里的 `hybrid_enabled`、`vector_hits`、`fuse_hits`，并构造本文件里的 `HybridRecallResult`；被本文件里的 `HybridRecallTool.execute` 调用；名字出现在 `__all__` 导出清单里，也是模块头注释指定的对外唯一入口。

### `HybridRecallInput` （第 179 行，Pydantic 入参模型）
- **作用**：定义 `knowledge.hybrid_recall` 工具的**输入契约**，即 Agent/上层调用这个工具时允许传什么、必须传什么、边界在哪。它在 `HybridRecallTool.spec` 里通过 `input_model=HybridRecallInput` 挂到 `ToolSpec` 上，运行期由工具框架负责把外部参数（通常来自模型生成的 JSON）按这份模型解析和校验，校验失败就不会进入 `execute`，因此 `execute` 里可以放心地假定参数已经合法。它是纯声明式模型，没有任何方法。
- **参数（字段）**：两个字段。`query: str`，必填，`Field(min_length=1, max_length=2000)`，说明为「检索问句；精确词（型号/编号）与转述句都能召回」，即空串和超过 2000 字符的串会被拒绝。`limit: int`，有默认值 `5`，约束 `ge=1, le=50`，说明为「返回条数上限」，即必须是 1 到 50 之间的整数。类配置 `model_config = ConfigDict(extra="forbid", strict=True)` 表示：出现模型未声明的多余字段直接报错，且开启严格模式（不做 `"5"` → `5` 之类的宽松强制转换）。
- **返回**：这是类，不返回值；实例化后得到校验通过的入参对象，被 `HybridRecallTool.execute` 读取 `arguments.query` 与 `arguments.limit`。
- **内部流程**：类体只有 `model_config` 声明与两个 `Field` 声明，校验逻辑全部由 Pydantic 在实例化/解析时自动执行（长度、范围、类型、多余字段检查），没有自定义 validator、没有 `__init__` 覆写。
- **异常/边界**：构造或校验失败时抛 Pydantic 的 `ValidationError`（例如 `query` 为空串、`limit` 为 0 或 51、传了未声明的额外字段、`limit` 传字符串等），异常由工具框架处理；模型本身不做额外兜底。边界值 1 与 50、长度 1 与 2000 是允许的。
- **同文件关系**：不调用本文件任何函数；被本文件里的 `HybridRecallTool.spec` 引用为 `input_model`，被 `HybridRecallTool.execute` 用作参数类型标注；名字出现在 `__all__` 导出清单里。

### `HybridHit` （第 190 行，Pydantic 出参单条模型）
- **作用**：定义单条召回命中的**输出结构**，把内部 `RetrievedChunk` 里对使用者有用的信息摊平成一层扁平字段，方便模型阅读和前端渲染。它刻意把三种分数都暴露出来：`score` 是最终排序依据（RRF 分），`vector_score` 与 `keyword_score` 是两路各自的原始分，这样上层可以解释「为什么这条排在前面」，也便于排查「是不是只有关键词路命中」。它由 `HybridRecallTool.execute` 在列表推导里逐条构造。
- **参数（字段）**：`chunk_id: str`（必填，分块唯一 ID，来自 `chunk.memory_id`）；`document_id: str | None = None`（所属文档 ID，可能为空）；`chunk_index: int | None = None`（块在文档内的序号，可能为空）；`snippet: str`（必填，正文片段，`execute` 里截取前 200 字符）；`score: float`（必填，字段说明为「RRF 融合分，同时是排序依据」）；`rrf_score: float`（必填）；`vector_score: float | None`（字段说明为「向量路原始相似度；降级时为 null」）；`keyword_score: float | None`（字段说明为「FTS5 bm25 原始分；无命中时为 null」）。类配置同样是 `ConfigDict(extra="forbid", strict=True)`，即禁止多余字段、严格类型。
- **返回**：这是类，不返回值；实例化后作为 `HybridRecallOutput.hits` 列表的元素返回给调用方。
- **内部流程**：只有字段声明，无方法、无自定义校验；`extra="forbid"` 与 `strict=True` 由 Pydantic 在构造时强制执行，`float` 字段在严格模式下要求传入的确实是数值类型（`execute` 里已用 `float(...)` 显式转换后再传入）。
- **异常/边界**：字段缺失或类型不符时抛 `ValidationError`；`document_id`、`chunk_index`、`vector_score`、`keyword_score` 允许为 `None`，而 `chunk_id`、`snippet`、`score`、`rrf_score` 不允许缺失。无其它特殊处理。
- **同文件关系**：不调用本文件任何函数；被本文件里的 `HybridRecallTool.execute` 逐条构造，被 `HybridRecallOutput` 作为列表元素类型引用；名字出现在 `__all__` 导出清单里。

### `HybridRecallOutput` （第 203 行，Pydantic 出参模型）
- **作用**：定义 `knowledge.hybrid_recall` 工具的**整体输出契约**，也就是工具返回给 Agent/前端的完整响应结构。它把「问了什么（`query`）」「命中多少条（`count`）」「这次检索有没有降级（`note` / `vector_available`）」「命中了哪些块（`hits`）」放在同一个对象里，让模型既能引用检索内容，又能如实转述降级状态。它在 `HybridRecallTool.spec` 里通过 `output_model=HybridRecallOutput` 挂到 `ToolSpec` 上，由工具框架负责序列化。
- **参数（字段）**：`query: str`（必填，回显本次检索问句）；`count: int`（必填，命中条数，`execute` 里取 `len(result.chunks)`）；`note: str`（必填，字段说明为「向量路降级说明；空串表示两路都正常」）；`vector_available: bool`（必填，向量路是否可用）；`hits: list[HybridHit]`，用 `Field(default_factory=list)` 给默认空列表，避免共享可变默认值。类配置为 `ConfigDict(extra="forbid", strict=True)`。
- **返回**：这是类，不返回值；实例化后作为工具 `execute` 的返回值。
- **内部流程**：只有字段声明，无方法、无自定义校验；`hits` 的元素类型由 Pydantic 在构造时按 `HybridHit` 递归校验。
- **异常/边界**：缺字段或类型不符抛 `ValidationError`；`hits` 省略时默认为空列表；`count` 与 `hits` 的实际长度不强制一致（`execute` 里两者同源，所以自然一致）。无其它特殊处理。
- **同文件关系**：不调用本文件任何函数；被本文件里的 `HybridRecallTool.execute` 构造并返回，被 `HybridRecallTool.spec` 引用为 `output_model`，并以列表元素类型引用本文件里的 `HybridHit`；名字出现在 `__all__` 导出清单里。

### `HybridRecallTool` （第 213 行，继承 `BaseTool`）
- **作用**：把上面的混合召回能力包装成一个符合项目工具框架规范的 **Agent 工具**，名字是 `knowledge.hybrid_recall`。它声明了工具的自然语言描述、版本、入参/出参模型、副作用等级、权限、超时、幂等性、并行安全性和标签，还写了一段给模型看的使用指引（`guidance`），告诉模型在回答「知识库里是怎么说的」这类问题时应**首先**用它、不要凭模型记忆作答，需要看整篇原文时接着用 `knowledge.document_get`，以及向量端点不可用降级时必须在回答里如实告知。类里还负责持有（或惰性构造）检索管线，并把框架传入的入参翻译成模块级 `hybrid_recall` 的调用。
- **参数（类属性）**：`spec` 是一个类属性，取值为 `ToolSpec(...)`，字段包括：`name="knowledge.hybrid_recall"`；`description`（英文，说明它结合向量相似度与 FTS5 关键词匹配、用 RRF 融合，适用于精确标识符和转述问句，只读、不写入，向量端点不可用时降级为纯关键词而不是失败）；`version="1.0.0"`；`input_model=HybridRecallInput`；`output_model=HybridRecallOutput`；`side_effect="read"`（只读）；`permissions=()`（不需要额外权限）；`timeout_seconds=30.0`；`idempotent=True`；`parallel_safe=True`；`tags=("memory", "recall", "hybrid", "rrf", "fts5", "read")`；`guidance`（中文指引文本）。
- **返回**：这是类，不返回值；通过 `create_tool()` 实例化后交给工具注册表。
- **内部流程**：类体只做 `spec` 声明与三个成员（`__init__`、`pipeline` 属性、`execute` 方法）的定义，没有其它类级逻辑；`BaseTool` 基类负责按 `spec` 做参数解析、校验、超时与结果序列化。
- **异常/边界**：类本身不抛异常；实际行为约束（30 秒超时、只读、可并行、幂等）由框架按 `spec` 施加。无特殊处理。
- **同文件关系**：引用了本文件里的 `HybridRecallInput`、`HybridRecallOutput` 作为入参/出参模型；`pipeline` 属性调用本文件里的 `_default_pipeline`，`execute` 调用本文件里的 `hybrid_recall`；被本文件里的 `create_tool` 实例化；名字出现在 `__all__` 导出清单里。

### `HybridRecallTool.__init__(self, pipeline: Any = None) -> None` （第 238 行）
- **作用**：工具实例的构造方法，唯一职责是**记录外部注入的检索管线**。默认允许不传（`pipeline=None`），这时实例处于「未装配」状态，真正的管线会在第一次访问 `pipeline` 属性时通过 `_default_pipeline()` 惰性构造；这种「可注入 + 惰性兜底」的设计让单元测试或上层（如 `web/app.py`）可以把已配置好的管线直接塞进来，避免重复构造和全局状态依赖。它不做任何校验、不做任何 I/O。
- **参数**：`pipeline: Any = None`，外部传入的检索管线对象；`None` 表示稍后惰性构造。
- **返回**：返回 `None`（构造方法）。
- **内部流程**：单行 `self._pipeline = pipeline`，把入参原样存到私有属性上；注意它并没有在此时判断 `None` 或提前构造，构造开销完全推迟到属性访问时。
- **异常/边界**：无任何异常处理；传任意对象都会被接受（包括传错类型的对象，错误会在真正调用 `execute` 时才暴露）；`None` 是合法取值，表示走默认管线。
- **同文件关系**：不调用本文件任何函数；其设置的 `self._pipeline` 被本文件里的 `HybridRecallTool.pipeline` 属性读取；由本文件里的 `create_tool` 以无参形式调用。

### `HybridRecallTool.pipeline` （第 241 行，`@property`）
- **作用**：以只读属性形式对外提供检索管线，并实现**惰性初始化 + 缓存**：第一次访问时如果 `_pipeline` 还是 `None`，就调用 `_default_pipeline()` 构造一个默认管线并写回 `self._pipeline`，之后所有访问都直接复用同一个对象，不会重复构造。它把「管线从哪来」这件事完全封装在工具内部，`execute` 只需写 `self.pipeline` 即可，不必关心是注入的还是默认的。
- **参数**：无显式参数（`self` 为实例）。
- **返回**：返回 `Any`，即当前的管线对象（可能是构造之初注入的，也可能是本次惰性构造出来的）。因为它是 `@property`，调用方用 `tool.pipeline` 取值而不是 `tool.pipeline()`；且没有配套 setter，外部不能直接给属性赋值。
- **内部流程**：判断 `if self._pipeline is None:`，成立则执行 `self._pipeline = _default_pipeline()`；随后 `return self._pipeline`。没有加锁，也没有并发保护。
- **异常/边界**：若 `_default_pipeline()` 内部抛异常（例如 `tool._memory` 导入失败），异常会从这个属性访问处向上抛出，且由于赋值发生在调用之后，`self._pipeline` 仍保持 `None`，下次访问会再次尝试构造（不会留下坏状态）。没有并发保护，多线程首次同时访问理论上可能构造两次，但最终都会指向同一个被缓存的实例之一。无其它特殊处理。
- **同文件关系**：调用了本文件里的 `_default_pipeline`；被本文件里的 `HybridRecallTool.execute` 通过 `self.pipeline` 使用。

### `HybridRecallTool.execute(self, arguments: HybridRecallInput) -> HybridRecallOutput` （第 247 行）
- **作用**：工具的实际执行体，把框架已经校验过的入参翻译成一次混合召回，并把内部结果**转换成对外契约**。它负责三件事：调用模块级 `hybrid_recall` 拿到 `HybridRecallResult`；把内部 `RetrievedChunk` 列表逐条映射成扁平、可序列化的 `HybridHit`（取 `memory_id` 当 `chunk_id`、从 `metadata` 取文档 ID 与块序号、正文截前 200 字符当 `snippet`、三种分数都带上）；最后连同 `query`、`count`、`note`、`vector_available` 一起打包成 `HybridRecallOutput`。当向量路降级时，`note` 会随输出一起回到模型面前，从而支撑 `guidance` 里「必须如实告知已降级」的要求。
- **参数**：`arguments: HybridRecallInput`，已经过 Pydantic 校验的入参对象，函数只读取 `arguments.query` 和 `arguments.limit`（因此 `threshold`、`metadata` 都保持 `hybrid_recall` 的默认值 `None`，即不设相似度阈值、不做元数据过滤）。
- **返回**：返回 `HybridRecallOutput` 实例，字段为：`query=arguments.query`；`count=len(result.chunks)`；`note=result.note`；`vector_available=result.vector_available`；`hits` 是 `HybridHit` 列表，每个元素由 `chunk.memory_id`、`chunk.metadata.get("document_id")`、`chunk.metadata.get("chunk_index")`、`(chunk.content or "")[:200]`、`float(chunk.score)`、`float(chunk.detail.get("rrf_score") or chunk.score)`、`chunk.detail.get("vector_score")`、`chunk.detail.get("keyword_score")` 组成。
- **内部流程**：第一步 `result = hybrid_recall(self.pipeline, arguments.query, limit=arguments.limit)`（`self.pipeline` 触发本文件里的惰性管线构造）；第二步构造 `HybridRecallOutput`，其中 `hits` 用列表推导遍历 `result.chunks`，对每条 `chunk` 依次取值：`chunk_id` 取 `chunk.memory_id`（内部 ID 在检索结果上叫 `memory_id`，对外统一叫 `chunk_id`），`document_id` / `chunk_index` 从 `chunk.metadata` 里 `get` 出来（缺失即 `None`），`snippet` 用 `(chunk.content or "")[:200]` 兜住 `content` 为 `None` 的情况并硬截 200 字符，`score` 与 `rrf_score` 都做 `float(...)` 转换（`rrf_score` 还带 `or chunk.score` 的兜底，防止 `detail` 里没有该键或值为 0/空时取不到分）；`vector_score`、`keyword_score` 直接透传，某一路未命中即为 `None`。
- **异常/边界**：`hybrid_recall` 或 `repository` 抛出的异常不在这里捕获，会向上交给框架（由 `spec.timeout_seconds=30.0` 约束超时）；`chunk.content` 为 `None` 时用空串兜底，`chunk.metadata` 缺少键时用 `None` 兜底；`detail` 缺少 `rrf_score` 时回退到 `chunk.score`；`snippet` 固定截断到 200 字符，超长正文只返回开头部分（要看全文需另调 `knowledge.document_get`）。`count` 用 `result.chunks` 的长度，与 `hits` 长度一致。
- **同文件关系**：调用了本文件里的 `hybrid_recall`（以及经由 `self.pipeline` 间接调用 `_default_pipeline`），并构造本文件里的 `HybridRecallOutput` 与 `HybridHit`；不被本文件其它函数调用，由工具框架在 Agent 调用 `knowledge.hybrid_recall` 时触发。

### `create_tool() -> BaseTool` （第 270 行）
- **作用**：工具工厂函数，供项目工具注册表/插件发现机制调用，返回一个可直接注册的 `HybridRecallTool` 实例。之所以用工厂函数而不是模块级单例，是为了让注册表每次拿到的是独立实例（避免跨会话共享工具状态），同时也保持了「注册入口是一个无参可调用对象」的统一约定。它不做任何参数校验，也不读取配置。
- **参数**：无参数。
- **返回**：返回 `BaseTool`（实际类型是 `HybridRecallTool`），内部未注入 `pipeline`，因此该实例的管线会在首次执行时惰性构造。
- **内部流程**：单行 `return HybridRecallTool()`，即无参构造（`pipeline` 取默认 `None`），其余装配工作交给 `pipeline` 属性的惰性逻辑。
- **异常/边界**：本身不抛异常；只有在 `HybridRecallTool()` 构造失败时才会异常，而当前 `__init__` 只做一次属性赋值，实际不会失败。无特殊处理。
- **同文件关系**：调用了本文件里的 `HybridRecallTool`（无参构造）；不被本文件其它函数调用，由外部工具注册机制调用；名字出现在 `__all__` 导出清单里。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_default_pipeline` | 惰性导入 `tool._memory` 并构造默认检索管线，作为未注入管线时的兜底。 |
| `HybridRecallResult` | 冻结 dataclass 结果载体，装融合后的分块列表、降级说明与向量路可用标记。 |
| `rrf_fuse` | 用倒数名次 `1/(k+rank)` 把多路 chunk_id 排名融合成一个带分数的统一排名。 |
| `chunk_metadata` | 从 `ChunkRecord` 抽取文档 ID、块序号、字符起止偏移四个定位字段成字典。 |
| `hybrid_enabled` | 返回混合检索开关 `constants.MEMORY_HYBRID`，关闭即退回纯向量。 |
| `vector_hits` | 走语义记忆向量路取 `(chunk_id, 相似度)`，失败时返回空表并附降级说明而不抛异常。 |
| `fuse_hits` | RRF 融合两路名次并按真值源回查正文，真值源缺失的孤立向量直接丢弃。 |
| `hybrid_recall` | 主编排入口：按开关/仓储决定纯向量回退还是双路融合，返回 `HybridRecallResult`。 |
| `HybridRecallInput` | 工具入参模型：`query`（1–2000 字符）与 `limit`（1–50，默认 5），禁止多余字段。 |
| `HybridHit` | 单条命中出参模型：块 ID、文档 ID、块序号、正文片段与三种分数。 |
| `HybridRecallOutput` | 工具出参模型：回显 query、命中条数、降级 note、向量可用标记与 hits 列表。 |
| `HybridRecallTool` | 把混合召回包装成只读、幂等、可并行的 Agent 工具 `knowledge.hybrid_recall`。 |
| `HybridRecallTool.__init__` | 记录外部注入的检索管线，`None` 表示稍后惰性构造。 |
| `HybridRecallTool.pipeline` | 只读属性，首次访问时惰性构造并缓存默认管线。 |
| `HybridRecallTool.execute` | 调用 `hybrid_recall` 并把结果逐条映射为 `HybridHit` 后包装成 `HybridRecallOutput`。 |
| `create_tool` | 工厂函数，无参构造并返回一个 `HybridRecallTool` 实例供注册。 |
