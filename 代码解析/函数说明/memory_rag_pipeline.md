# memory/rag/pipeline.py

## 一、这个文件是干什么的

这个文件是四层记忆系统里「RAG（检索增强生成）」这一层的总装配点，职责是把「文档」变成「可检索的记忆」，再把「检索结果」变成「能塞进提示词的上下文」。它对外只暴露两个名字：`RetrievedChunk`（检索结果的数据载体）和 `RAGPipeline`（执行入库与检索的门面类），文件末尾用 `__all__` 明确锁定了这两个导出。

它的上游是 `memory/rag/document.py`（负责解析文件、切块、给块标注字符区间）和 `memory/rag/knowledge.py`（负责实体/关系抽取与图谱上下文构建），下游是 `memory/manager.py` 的 `MemoryManager`（向量/记忆的读写入口）以及 `memory/storage/document_repo.py` 的 `DocumentRepository`（documents/chunks 真值源）。

运行时它主要在两条链路上被用到：一条是「入库链路」——`ingest()` / `ingest_source()` 把解析后的 `Document` 切成块，逐块写入真值源 `chunks` 表并写入语义记忆，然后在 `auto_extract` 打开时调用知识抽取器把块内容抽成实体与关系，最后把抽取统计写进 `self.last_ingest_report` 供 Web 层展示；另一条是「检索链路」——`retrieve()` / `build_context()` 做混合检索并把正文按真值源修正后拼成上下文，`graph_retrieve()` / `graph_context()` 则委托给 `GraphRAGPipeline` 做多跳图谱检索。

文件里还包含几个模块级小工具：`accepts_parameter()`（用 `inspect` 自省抽取器的 `extract` 签名，判断它是否接受某个关键字参数）、`_accepts_graph_context()`（它的特化封装，用来兼容旧的两参数抽取器契约）、`_document_tags()` 和 `_document_permission()`（从调用方给的 metadata 里安全地读出标签与权限，权限采取「失败即私有」的保守策略）。配置项（块大小、重叠、检索条数、图谱跳数、上下文最大字符数）全部来自 `constants`，本文件不硬编码数值。

## 二、函数与类逐条详解

### `RetrievedChunk` （第 38 行）
- **作用**：这是一个 `@dataclass(frozen=True)` 的不可变数据类，用来承载「一条被检索到的文本块」。为什么需要它：`MemoryManager.search()` 返回的是 `MemorySearchResult`（记忆条目 + 分数），但记忆条目里的正文可能已经过时（同一 id 的正文真值可能已被修正到 `chunks` 表里），而且调用方通常还需要混合检索的分数明细。所以这里定义了一个独立的、只描述「检索结果」的结构，把正文、分数、记忆 id、元数据和分数明细四样东西绑在一起，供上层拼上下文或做溯源面板展示。它会被 `retrieve()` 构造并返回，是 `RAGPipeline` 检索链路的对外产物。字段 `detail` 用 `field(default_factory=dict)` 而不是直接写 `= {}`，是为了避免可变默认值在实例之间共享。由于是 frozen，实例创建后不能改字段，可以安全地跨线程/跨函数传递。
- **参数**：作为 dataclass，参数就是四个字段加一个带默认值的字段：`content: str` 是块的正文文本；`score: float` 是检索得分（通常是融合后的 RRF 分数或相似度）；`memory_id: str` 是这条记忆在记忆系统里的唯一 id；`metadata: Mapping[str, Any]` 是随块携带的元数据（来源、文档 id、块序号、标签、权限等）；`detail: Mapping[str, Any] = field(default_factory=dict)` 是混合检索的分数明细，注释里说明约定包含 `rrf_score` / `vector_score` / `keyword_score` 三个键，默认是空字典，也就是「不提供明细」。
- **返回**：类本身，实例化后即为一个检索结果对象；没有其它返回形式。
- **内部流程**：由 `@dataclass(frozen=True)` 装饰器在导入时自动生成 `__init__`、`__repr__`、`__eq__` 等方法（因为 frozen，还会生成 `__hash__`），本文件没有手写任何方法体。`detail` 的默认值走 `field(default_factory=dict)`，每次实例化都会新建一个空字典。
- **异常/边界**：本类自身不抛异常。边界上不校验类型：`content` 传空字符串、`score` 传负数或 `nan` 都能构造成功，是否合法由调用方负责；`metadata` 与 `detail` 只标注为 `Mapping`，传入 `None` 也不会在这里被拦下，但后续 `.get()` 调用会失败。
- **同文件关系**：它被 `RetrievedChunk.from_result()`（同类方法）构造，被 `retrieve()` 直接构造并返回；`build_context()` 消费它的 `content` 字段，`delete_document()` 之外的地方不涉及它。

### `RetrievedChunk.from_result(cls, result: MemorySearchResult) -> RetrievedChunk` （第 47 行）
- **作用**：这是一个 `@classmethod` 工厂方法，作用是把记忆系统原生的 `MemorySearchResult` 转换成 `RetrievedChunk`。为什么需要它：检索链路有两条分支——一条是能在真值源 `chunks` 表里查到该 id（正文以真值源为准，走另一条手工构造的分支），另一条是查不到（例如记忆不是通过文档入库来的，或者真值源不可用），此时只能退回使用记忆条目自带的正文，这条退路就统一走这个工厂方法。它把「从记忆条目里取哪些字段」这件事收敛到一处，避免在 `retrieve()` 里散落字段访问代码。`detail` 不在这里传递，因此走这条分支的结果分数明细为空字典。
- **参数**：`cls` 是类方法隐式传入的类对象（即 `RetrievedChunk`，通过它可以正确支持子类继承）；`result: MemorySearchResult` 是记忆检索返回的单条结果，需要它有 `item` 属性，且 `item` 上有 `content`、`id`、`metadata`，结果对象自身有 `score`。
- **返回**：返回一个新的 `RetrievedChunk` 实例，字段依次取自 `result.item.content`、`result.score`、`result.item.id`、`result.item.metadata`，`detail` 保持默认空字典。
- **内部流程**：只有一步——直接用位置参数调用 `cls(...)` 构造并返回。没有条件分支、没有循环、没有中间变量。
- **异常/边界**：如果 `result` 为 `None`，或 `result.item` 缺失 / 没有 `content`、`id`、`metadata` 属性，会抛 `AttributeError`；本方法不做任何判空或类型兜底。空字符串正文会被原样接受。
- **同文件关系**：它调用本文件的 `RetrievedChunk` 构造器；被本文件的 `retrieve()` 在「真值源查不到该块」的分支里调用。

### `accepts_parameter(extractor: KnowledgeExtractor, name: str) -> bool` （第 52 行）
- **作用**：这是一个模块级的公开工具函数，用来判断某个知识抽取器的 `extract` 方法是否接受名为 `name` 的关键字参数。为什么需要它：抽取器是一个协议（protocol），项目里既有内置抽取器，也允许外部实现自定义抽取器；新版本给 `extract` 增加了可选参数（例如 `graph_context`），但老的自定义实现只有两个参数，如果直接传新参数就会 `TypeError`。文档字符串明确说明图片入库（`tool/ingest_image.py`）与文本入库共用这一处判定逻辑，所以它是公开的，不写第二份 `inspect.signature` 逻辑。它只在入库流程开始前被调用一次，用来决定后续按哪种签名去调用抽取器。
- **参数**：`extractor: KnowledgeExtractor` 是抽取器实例，必须具有可被 `inspect.signature` 解析的 `extract` 属性；`name: str` 是要探测的参数名，例如 `"graph_context"`，大小写敏感，需与目标参数名完全一致。
- **返回**：返回 `bool`。当 `extract` 的签名里显式包含名为 `name` 的参数时返回 `True`；当签名里存在 `*args` 风格的 `**kwargs`（即 `inspect.Parameter.VAR_KEYWORD`）时也返回 `True`，因为这类实现能吞下任意关键字参数；当 `inspect.signature` 抛 `TypeError` 或 `ValueError`（对象不可内省、是内置函数等）时返回 `False`。
- **内部流程**：第一步用 `inspect.signature(extractor.extract).parameters` 取出参数字典，这一步被 `try/except` 包住，只捕获 `TypeError` 与 `ValueError`。第二步先做一次 `name in parameters` 的快速判断，命中就直接 `True`。第三步用 `any(...)` 遍历所有参数，检查是否存在 `parameter.kind is inspect.Parameter.VAR_KEYWORD` 的参数（即 `**kwargs`），存在则整体返回 `True`，否则返回 `False`。整个函数是纯函数，无副作用。
- **异常/边界**：函数内部已经把 `TypeError` 与 `ValueError` 吞掉并转为 `False`，因此自身不向外抛异常。边界情况：`extractor` 为 `None` 会抛 `AttributeError`（未被捕获，因为访问 `extractor.extract` 发生在 `try` 内部——实际上这一句在 `try` 里，`AttributeError` 不在捕获列表中，所以会向上抛）；`name` 传空字符串时不会匹配任何真实参数名，通常返回 `False`；用 `functools.partial` 或 C 实现包装的 `extract` 可能内省失败，同样归入 `False`。
- **同文件关系**：它被本文件的 `_accepts_graph_context()` 直接调用（那是它在文件内的唯一调用方）；它本身不调用本文件里任何其它函数。

### `_accepts_graph_context(extractor: KnowledgeExtractor) -> bool` （第 69 行）
- **作用**：这是一个私有辅助函数，专门回答「这个抽取器能不能消费抽取前构建好的子图」。为什么需要它：`ingest()` 在抽取前会先用 `build_graph_context()` 算出与当前块相关的既有子图，把已有实体名喂给模型，好让模型复用规范名、正确地把旧值标记为失效，而不是又造一个新实体；但只有显式声明接受 `graph_context` 的抽取器才应该收到这个额外参数。文档字符串点明了兼容性意图：按旧的两参数契约写的自定义抽取器继续照常工作，只有主动「opt in」的实现才拿到子图上下文。它在每次 `ingest()` 调用开始时被求值一次，结果缓存在局部变量里供循环中所有块复用。
- **参数**：`extractor: KnowledgeExtractor` 是待检查的抽取器实例，会原样转交给 `accepts_parameter()`。
- **返回**：返回 `bool`——`True` 表示抽取器接受 `graph_context` 参数（显式声明或有 `**kwargs`），`False` 表示不接受，调用方应退回两参数调用方式。
- **内部流程**：函数体只有一行，直接把 `extractor` 和固定字符串 `"graph_context"` 传给 `accepts_parameter()` 并返回其结果。没有任何分支、循环或本地状态。
- **异常/边界**：自身不抛异常，异常行为完全继承自 `accepts_parameter()`（例如 `extractor` 为 `None` 时会因属性访问失败而抛 `AttributeError`）。对 `**kwargs` 型抽取器一律判定为接受，这是有意的宽松处理。
- **同文件关系**：它调用本文件的 `accepts_parameter()`；只被本文件的 `ingest()` 调用（把结果存进局部变量 `accepts_context`）。

### `_document_tags(metadata: Mapping[str, Any]) -> list[str]` （第 79 行）
- **作用**：这是一个私有工具函数，作用是从调用方提供的 metadata 里把 `tags` 读成一个规整的字符串列表。为什么需要它：入库时标签可能来自不同的调用方，有人给列表 `["a", "b"]`，有人图省事给单个字符串 `"a"`，也可能干脆不给；而真值源 `DocumentRecord.tags` 需要的是统一的 `list[str]`。这个函数把三种形态归一化，避免在 `ingest()` 主体里写类型判断。它只在 `ingest()` 写 `DocumentRecord` 时被调用一次（每个文档一次），属于纯粹的入库前数据清洗。
- **参数**：`metadata: Mapping[str, Any]` 是文档级元数据字典，只需要可能包含 `"tags"` 键；缺失该键、值为 `None`、空列表或空字符串都会走默认分支。
- **返回**：返回 `list[str]`。当 `tags` 缺失或为假值（`None`、`[]`、`""`）时返回空列表 `[]`；当 `tags` 是字符串时返回只含该字符串的单元素列表 `[tags]`（不做拆分，`"a,b"` 会原样成为一个标签）；其它情况按可迭代对象处理，对每个元素调用 `str()` 后组成新列表，因此数字、枚举等都会被转成字符串，`None` 元素会变成字符串 `"None"`。
- **内部流程**：第一步 `metadata.get("tags") or []`，用 `or` 把 `None`、空列表、空字符串等假值统一替换为空列表。第二步 `isinstance(tags, str)` 判断是否单字符串，是则直接包成单元素列表返回。第三步用列表推导 `[str(tag) for tag in tags]` 遍历并逐项字符串化。没有异常捕获，没有日志。
- **异常/边界**：`metadata` 为 `None` 会抛 `AttributeError`（在 `try` 之外）；`tags` 是整数等不可迭代对象时会抛 `TypeError`（`'int' object is not iterable`）；`tags` 是字典时会迭代它的键并转成字符串，属于宽松但可能不符合预期的行为。函数不修改传入的 `metadata`，返回的是新列表。
- **同文件关系**：它不调用本文件任何其它函数；被本文件的 `ingest()` 调用，用于填充 `DocumentRecord` 的 `tags` 字段。

### `_document_permission(metadata: Mapping[str, Any]) -> str` （第 88 行）
- **作用**：这是一个私有工具函数，用来从 metadata 里解析文档的权限级别，策略是「失败即关闭（fail closed）」——任何无法识别的取值都退回 `private`，也就是数据留在这台机器上不外传。为什么需要它：权限决定了文档能否被共享/上传，是一个安全相关字段，不能因为调用方拼错字（例如把 `"public"` 写成 `"publc"`）就意外放开。文档字符串把这个保守策略写得很直白。它只在 `ingest()` 写 `DocumentRecord` 时被调用，每个文档一次。
- **参数**：`metadata: Mapping[str, Any]` 是文档级元数据字典，可能带 `"permission"` 键；该键缺失时取默认值 `"private"`。
- **返回**：返回 `str`。当 `metadata["permission"]` 存在且其字符串形式落在 `PERMISSIONS` 集合（从 `memory/storage/document_repo.py` 导入的合法权限白名单）中时，返回该字符串；否则一律返回 `"private"`。
- **内部流程**：第一步 `str(metadata.get("permission", "private"))`，先取默认值再强制转成字符串，这样即便调用方传了非字符串（例如某个枚举）也会先字符串化再去比对白名单。第二步用 `permission in PERMISSIONS` 做白名单校验，命中就返回 `permission`，否则返回字面量 `"private"`。整个函数没有循环、没有副作用。
- **异常/边界**：`metadata` 为 `None` 会抛 `AttributeError`；`metadata.get` 返回的对象若 `__str__` 抛异常则会向上传播。空字符串 `""` 不在白名单里，因此会被安全地归为 `"private"`；大小写不匹配（如 `"Public"`）同样会被降级为 `"private"`。
- **同文件关系**：它不调用本文件任何其它函数；被本文件的 `ingest()` 调用，用于填充 `DocumentRecord` 的 `permission` 字段。

### `RAGPipeline` （第 95 行）
- **作用**：这是本文件的核心门面类，把「文档入库 + 知识抽取 + 混合检索 + 图谱检索 + 文档删除 + 资源释放」这一整套 RAG 能力封装成一个对象。为什么需要它：上层（Web 层、工具层）不应该自己去拼装 `MemoryManager`、`DocumentProcessor`、抽取器、`GraphRAGPipeline` 这些部件，也不应该关心真值源仓库怎么按需懒加载；`RAGPipeline` 提供一个稳定的构造入口和一组语义清晰的方法。它持有一个惰性创建的真值源仓库缓存 `_repository`，以及一份最近一次入库的统计报告 `last_ingest_report`，后者是 Web 界面展示「抽了多少实体/关系、有没有报错」的数据来源。它内部还持有一个 `GraphRAGPipeline` 实例，图谱相关的方法都只是转发给它。
- **参数**：类本身无构造参数（参数在 `__init__` 里）。
- **返回**：类对象，用于实例化。
- **内部流程**：类体只包含方法定义，没有任何类级属性、类变量或装饰器；实例状态全部在 `__init__` 中建立。方法分为四组：入库（`ingest`、`ingest_source`）、检索（`retrieve`、`build_context`、`graph_retrieve`、`graph_context`）、维护（`delete_document`、`document_repo`）、生命周期（`close`）。
- **异常/边界**：类定义阶段不抛异常；实例化时的失败风险来自 `__init__` 内部默认构造的各个部件。
- **同文件关系**：它使用本文件的 `RetrievedChunk`、`_accepts_graph_context()`、`_document_tags()`、`_document_permission()`；被 `__all__` 导出，是外部唯一的使用入口。

### `RAGPipeline.__init__(self, manager: MemoryManager | None = None, *, processor: DocumentProcessor | None = None, extractor: KnowledgeExtractor | None = None, auto_extract: bool = True) -> None` （第 96 行）
- **作用**：构造一个 RAG 流水线实例，并完成所有依赖的装配。为什么需要它：项目里既有「生产路径」（什么都不传，自动创建真实的记忆管理器、文档处理器和空抽取器），也有「测试/定制路径」（注入假的 manager、自定义 processor、带模型的自定义 extractor）。构造器用「传了就用手上的，没传就造默认的」这种显式判空写法（而不是 `x or default`），避免传入的合法假值被误替换。它还会立刻创建 `GraphRAGPipeline`、初始化入库报告字典和真值源仓库缓存为 `None`（表示尚未懒加载）。构造完成后实例即可直接调用 `ingest()` 或 `retrieve()`。
- **参数**：`manager: MemoryManager | None = None` 是记忆管理器，`None` 时新建 `MemoryManager()`，它同时是向量/记忆读写入口和真值源 `document_store` 的持有者；`processor: DocumentProcessor | None = None` 是文档处理器（负责解析、归一化文本、切块、切句），`None` 时新建 `DocumentProcessor()`；`extractor: KnowledgeExtractor | None = None` 是知识抽取器，`None` 时使用 `NullKnowledgeExtractor()`，即「不抽取」，这也是默认行为；`auto_extract: bool = True` 是关键字参数，控制 `ingest()` 是否对每个块执行抽取（注意默认为 `True`，但默认抽取器是空实现，所以默认组合下相当于快速入库）。前三个参数中只有 `manager` 是位置参数，其余三个都是 keyword-only（`*` 之后）。
- **返回**：无返回值（`None`）。
- **内部流程**：依次给五个实例属性赋值——`self.manager`、`self.processor`、`self.extractor`（三者都用 `if x is not None else 默认值` 的三元写法）、`self.auto_extract`；然后 `self.graph = GraphRAGPipeline(self.manager)` 把记忆管理器交给图谱流水线；再初始化 `self.last_ingest_report = {}`；最后把 `self._repository = None` 置为未加载状态，表示真值源仓库会在首次 `document_repo()` 调用时才建立。整个过程没有 I/O、没有校验、没有异常捕获。
- **异常/边界**：`manager`/`processor` 为 `None` 时会真正构造默认对象，如果这些构造器内部需要访问磁盘或配置，异常会从这里抛出并向上传播；传入 `auto_extract` 非布尔值（如字符串）不会被校验，只会按真值语义在 `ingest()` 里被 `if not self.auto_extract` 使用。`NullKnowledgeExtractor` 的构造不涉及模型，所以默认实例化很轻。
- **同文件关系**：它构造了本文件的 `RAGPipeline` 实例状态，并调用外部的 `GraphRAGPipeline`；被外部代码（Web 层/工具层）以及可能的测试直接调用。

### `RAGPipeline.document_repo(self) -> DocumentRepository | None` （第 112 行）
- **作用**：返回 `documents`/`chunks` 真值源的仓库对象，如果不可用则返回 `None`。为什么需要它：真值源是「正文的最终真相」，检索时要拿它校正正文、入库时要往里写行、删除文档时要同步清行，但并非所有运行环境都能提供它。文档字符串明确列出两种返回 `None` 的情形：一是被注入的文档存储不是 SQLite 支撑的（没有 `path` 属性），二是路径是 `:memory:`（再开一个内存库会得到一个私有连接，跟它本应镜像的那个库共享不到任何数据，反而制造出两个互不可见的库）。这个方法同时承担「懒加载 + 缓存 + 路径变更时重建」的职责：它把创建出来的仓库缓存在 `self._repository`，避免每次检索都重新打开数据库。
- **参数**：只有 `self`，无其它参数。
- **返回**：返回 `DocumentRepository | None`。路径不可用（`path` 为假值或等于字符串 `":memory:"`）时返回 `None`；否则返回一个 `DocumentRepository` 实例，且该实例一定满足 `self._repository.path == str(path)`。
- **内部流程**：第一步 `path = getattr(self.manager.document_store, "path", None)`，用 `getattr` 带默认值的方式安全取路径，这样即使 `document_store` 没有 `path` 属性也不会报错。第二步判断 `not path or str(path) == ":memory:"`，命中就返回 `None`。第三步判断缓存是否可复用：`self._repository is None` 或 `self._repository.path != str(path)` 时重建，即 `self._repository = DocumentRepository(path)`；否则直接返回已有缓存。这里比较的是字符串形式的路径，所以缓存不会因为换了数据库文件而错用旧连接。
- **异常/边界**：`self.manager` 为 `None` 会抛 `AttributeError`；`DocumentRepository(path)` 构造失败（例如目录不可写、文件被占用）时异常直接向上抛，不会退化为 `None`——也就是说「打不开库」是硬错误，而「本来就不该有库」才是 `None`。`path` 是 `Path` 对象或字符串都能工作，因为比较前统一 `str()` 化。
- **同文件关系**：它不调用本文件其它函数；被本文件的 `ingest()`、`retrieve()`、`delete_document()`、`close()` 调用，是这几处访问真值源的统一入口。

### `RAGPipeline.ingest(self, documents: Document | Iterable[Document], *, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP, granularity: str = "chunk") -> list[MemoryItem]` （第 128 行）
- **作用**：这是整个文件最重的函数，负责把一批文档完整地「吃进」记忆系统。它做四件事：把文档登记到真值源 `documents` 表；把文档切成块（默认按字符窗口，可选逐句）并逐块写入真值源 `chunks` 表与语义记忆；在 `auto_extract` 打开时对每块做知识抽取，把实体、关系、失效、撤回等结果物化进图谱；最后把统计与错误汇总成一份报告存进 `self.last_ingest_report`，并把文档状态置为 `extracted`、`vectorized` 或 `failed`。为什么需要它：这是「文档 → 可检索记忆」的唯一通道，注释里特别强调它同时是混合索引（先写真值源、再写向量、再置 `vector_status`）的调用点，而混合索引的唯一实现在 `tool/hybrid_index.py`，这里用函数内导入避免 `memory.rag` 与 `tool` 之间的模块级初始化环。它的错误处理策略很讲究：抽取失败只记错误、不丢源文本；而嵌入/真值写入失败必须回滚该文档的真值行后原样抛出，避免留下半吊子记录。
- **参数**：`documents: Document | Iterable[Document]` 是单个 `Document` 或一批 `Document`，单个会被自动包成单元素列表；`chunk_size: int = RAG_CHUNK_SIZE`（keyword-only）是字符窗口块大小，来自 `constants`，仅在 `granularity == "chunk"` 时生效；`overlap: int = RAG_CHUNK_OVERLAP`（keyword-only）是相邻块的重叠字符数，同样只在字符窗口模式下生效；`granularity: str = "chunk"`（keyword-only）是切块粒度，只允许 `"chunk"` 或 `"sentences"` 两个取值，其它值会直接报错。
- **返回**：返回 `list[MemoryItem]`，按处理顺序包含本次入库成功写入的每一条块级语义记忆条目。注意返回的是「块」条目，抽取出来的实体/关系记忆不在此列表中；它们的数量体现在 `self.last_ingest_report` 里。当 `documents` 为空可迭代对象时返回空列表，报告里的计数保持为 0。
- **内部流程**：第一步归一化输入——`[documents] if isinstance(documents, Document) else list(documents)`，把生成器等一次性可迭代对象先物化，防止后面重复遍历时被耗尽。第二步校验 `granularity`，不在 `{"chunk", "sentences"}` 中就抛 `ValueError`。第三步初始化 `items` 空列表和 `report` 字典（键包括 `chunks`、`domains`、`entities`、`relations`、`superseded`、`retracted`、`skipped_relations`、`errors`）。第四步做一次性准备：创建 `EntityResolver(self.manager)`（注释说明每次 ingest 只建一个解析器，这样第 1 块创建的实体能被第 2 块复用和加别名，不必每块重新全量加载）；调用 `_accepts_graph_context(self.extractor)` 得到 `accepts_context`；调用 `self.document_repo()` 得到 `repository`；调用 `apply_embedding_lock(self.manager, repository)` 施加嵌入锁。第五步进入 `for document in values` 主循环：先算 `source = str(document.metadata.get("source", document.id))`；若 `repository` 可用，则 `upsert_document(DocumentRecord(...))`，其中 `title` 取 `metadata["title"]`（默认空串）、`raw_text` 用 `self.processor.normalized_text(document)`、`tags` 与 `permission` 分别由 `_document_tags()` 和 `_document_permission()` 解析、`status` 固定为 `"parsed"`；随后置 `document_error = None`。第六步按粒度取 `spans`：`granularity == "sentences"` 时用 `self.processor.sentences_with_spans(document)`，否则用 `self.processor.chunks_with_spans(document, chunk_size=..., overlap=...)`（注释说明句级也写真值源 `chunks`，不新建表）。第七步进入 `for span in spans` 内层循环：从 `span.chunk` 取块，把 `chunk.metadata` 复制成新字典并 `setdefault("source", source)`；函数内 `from tool.hybrid_index import index_chunk`；调用 `index_chunk(...)` 传入 manager、repository、`chunk_id`、`document_id`、`chunk_index`（从 `chunk.metadata["chunk_index"]` 强转 `int`）、`char_start`/`char_end`（取自 span）、`text` 与 `metadata`，返回的 `item` 追加进 `items` 并把 `report["chunks"]` 加一。若 `auto_extract` 为假则 `continue` 跳过抽取。否则进入抽取 `try` 块：先用 `build_graph_context(self.manager, chunk.content, resolver=resolver)` 算出抽取前子图；再按 `accepts_context` 决定调用 `self.extractor.extract(chunk.content, metadata=metadata, graph_context=graph_context)` 还是 `self.extractor.extract(chunk.content, metadata=metadata)`；然后 `materialize_extraction(...)` 物化抽取结果；最后把 `materialized` 里的 `domain` 追加到 `report["domains"]`，并把 `entities`、`relations`、`superseded`、`retracted`、`skipped_relations` 五个计数累加进报告。抽取异常被 `except Exception as exc` 捕获（注释说明抽取失败绝不能丢掉源文本），把 `"{类型名}: {消息}"` 追加到 `report["errors"]`，并用 `document_error = document_error or message` 保留该文档的第一条错误。第八步外层循环还有一个 `except Exception:` 包裹整个块循环：一旦嵌入或真值写入失败（注释举例云端端点不可达），若 `repository` 可用就先 `repository.delete_document(document.id)` 回滚该文档的真值行，然后 `raise` 原样抛出，交给调用方转成明确的错误响应。第九步循环收尾：若 `repository` 可用，`document_error is None` 时按 `auto_extract` 把状态置为 `"extracted"` 或 `"vectorized"`，否则调用 `repository.set_status(document.id, "failed", error=document_error)`。第十步所有文档处理完后做报告收尾：用 `list(dict.fromkeys(report["domains"]))` 给 domain 列表去重且保持首次出现顺序；写入 `report["extractor"] = type(self.extractor).__name__`；写入 `report["extraction_skipped"] = isinstance(self.extractor, NullKnowledgeExtractor)`；把报告赋给 `self.last_ingest_report`；最后返回 `items`。
- **异常/边界**：`granularity` 非法抛 `ValueError`；`documents` 是 `None` 时 `isinstance` 判断为假、`list(None)` 抛 `TypeError`；块循环内的嵌入/真值写入异常会先删除该文档真值行再重新抛出（回滚是尽力而为的：`repository` 为 `None` 时不做回滚）；抽取异常被吞掉并只记入报告，不会中断整批入库；`chunk.metadata["chunk_index"]` 缺失会抛 `KeyError`（属于嵌入分支，会走回滚并抛出）；`report["domains"]` 可能包含重复值，最后才去重；单条错误只保留第一条（`document_error or message`），但 `report["errors"]` 会累积全部错误。若 `values` 为空，报告仍会被写入 `self.last_ingest_report`（各计数为 0，`extractor` 与 `extraction_skipped` 已填好）并返回空列表。
- **同文件关系**：它调用本文件的 `document_repo()`、`_accepts_graph_context()`、`_document_tags()`、`_document_permission()`；被本文件的 `ingest_source()` 调用（转发 `**kwargs`）。它本身不调用 `RetrievedChunk`。

### `RAGPipeline.ingest_source(self, source: str | Path, *, base_dir: str | Path | None = None, **kwargs: Any) -> list[MemoryItem]` （第 247 行）
- **作用**：这是按「文件路径」入库的便捷入口，把「解析文件」和「入库」两步串起来。为什么需要它：调用方（尤其是模型驱动的调用方）手上往往是一个路径而不是已经解析好的 `Document`；这个函数负责把路径交给 `DocumentProcessor.parse()` 变成 `Document`，再把其余参数原样转发给 `ingest()`。文档字符串强调两件事：一是这里的 `source` 是路径，字符串会被转成 `Path`，避免短文本被误当成文件名；二是当给了 `base_dir` 时，解析后的路径必须落在该目录内，这正是让模型驱动的调用方能够显式声明「只能读这个边界内的文件」的机制。它是一次调用的薄封装，不保留任何状态。
- **参数**：`source: str | Path` 是待入库的文件路径，字符串会被 `Path(source)` 转换；`base_dir: str | Path | None = None`（keyword-only）是可选的包含边界，`None` 表示不做边界限制；`**kwargs: Any` 收集其余关键字参数，其中 `metadata` 会被 `pop` 出来交给 `processor.parse()`，剩下的（例如 `chunk_size`、`overlap`、`granularity`）原样转发给 `ingest()`。
- **返回**：返回 `list[MemoryItem]`，即 `ingest()` 的返回值，含义与 `ingest()` 完全一致（本次入库写入的块级语义记忆条目）。
- **内部流程**：第一步 `path = Path(source)` 把入参统一成 `Path`。第二步若 `base_dir is not None`，则 `path = resolve_within(base_dir, path)`——这一步在 `memory/rag/document.py` 里实现，负责解析路径并校验它没有越出 `base_dir`。第三步调用 `self.processor.parse(path, metadata=kwargs.pop("metadata", None))` 得到 `Document`，注意 `metadata` 用 `pop` 取出（默认 `None`），因此它不会残留在 `kwargs` 里被重复传给 `ingest()`（`ingest()` 本身也不接受 `metadata` 参数）。第四步 `return self.ingest(document, **kwargs)` 把剩余参数转发出去。
- **异常/边界**：路径越界或非法由 `resolve_within()` 抛错（例如符号链接逃逸、路径不存在），本函数不捕获；`source` 是空字符串会得到 `Path(".")` 这样的结果，行为取决于后续解析；`kwargs` 里带了 `ingest()` 不认识的键会抛 `TypeError`；`metadata` 显式传 `None` 与不传等价。
- **同文件关系**：它调用本文件的 `ingest()`；不被本文件其它函数调用，是外部调用的入口之一。

### `RAGPipeline.retrieve(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[RetrievedChunk]` （第 269 行）
- **作用**：执行一次语义检索并把结果规整成 `RetrievedChunk` 列表。为什么需要它：它承担一个关键的「正文校正」职责——注释写明正文以 `chunks` 真值源为准，因为同一 id 对应的内容可能已经被修正过，记忆里存的可能是旧版本；只有在真值源查不到该 id 时才退回使用记忆条目自带的正文。这个校正让「检索到的正文」和「真值源里的正文」保持一致，避免上层拿到过期内容。它同时是 `build_context()` 的数据来源。整个方法只做查询与映射，不写任何数据。
- **参数**：`query: str` 是检索查询文本，直接交给 `MemoryManager.search()`；`limit: int = RAG_RETRIEVE_LIMIT`（keyword-only）是返回条数上限，默认取自 `constants`；`threshold: float | None = None`（keyword-only）是可选的相关度阈值，`None` 表示不设阈值、由检索层默认策略决定；`metadata: Mapping[str, Any] | None = None`（keyword-only）是可选元数据过滤条件，用来把检索范围限制在符合该元数据的记忆上。
- **返回**：返回 `list[RetrievedChunk]`，顺序与 `manager.search()` 返回的顺序一致（通常按分数降序）。真值源可用且命中该 id 时，`content` 用 `chunk.text`；否则用记忆条目的正文。两种情况下 `score`、`memory_id`、`metadata` 都来自原检索结果，`detail` 一律为空字典（本方法不填充分数明细）。检索无结果时返回空列表。
- **内部流程**：第一步 `repository = self.document_repo()` 拿真值源（可能为 `None`）。第二步初始化空列表 `results`。第三步 `for result in self.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=limit, threshold=threshold, metadata=metadata)` 遍历检索结果——注意检索被硬编码限定在 `MemoryType.SEMANTIC` 语义记忆上。第四步在循环体内：若 `repository` 不为 `None` 则 `repository.get_chunk(result.item.id)` 按记忆 id 查真值块；`chunk is None`（包括 repository 为 `None` 或真的查不到）时用 `RetrievedChunk.from_result(result)` 追加；否则用 `RetrievedChunk(chunk.text, float(result.score), result.item.id, result.item.metadata)` 手工构造（`score` 显式 `float()` 化，`detail` 保持默认空）。第五步返回 `results`。
- **异常/边界**：`self.manager` 为 `None` 或 `search` 抛错时异常向上传播；`document_repo()` 可能因数据库打不开而抛错（此时不会静默退化）；`get_chunk()` 内部异常同样向上抛。`limit` 传 0 或负数时的行为由检索层决定（本方法不校验）；`query` 传空字符串会原样交给检索层；`threshold` 传非数值类型不会被本方法拦截。
- **同文件关系**：它调用本文件的 `document_repo()` 与 `RetrievedChunk.from_result()`，并构造 `RetrievedChunk`；被本文件的 `build_context()` 调用。

### `RAGPipeline.build_context(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, separator: str = "\n\n") -> str` （第 283 行）
- **作用**：把检索结果拼成一段可直接塞进提示词的纯文本上下文。为什么需要它：上层（对话/生成链路）需要的是字符串而不是结构化对象，这个方法把「检索 → 取正文 → 用分隔符连接」三步合成一行，并且用一个显式类型检查挡住传错的 `separator`（例如误传列表），这样拼接失败会以清晰的错误信息暴露而不是产出畸形上下文。它不做长度截断、不做去重，只负责拼接；需要图谱版上下文时应改用 `graph_context()`。
- **参数**：`query: str` 是检索查询，原样传给 `retrieve()`；`limit: int = RAG_RETRIEVE_LIMIT`（keyword-only）是参与拼接的块数上限；`separator: str = "\n\n"`（keyword-only）是块之间的连接字符串，默认两个换行（即空行分段）。
- **返回**：返回 `str`。当 `retrieve()` 返回空列表时返回空字符串 `""`（因为 `join` 空序列的结果就是空串）。正常情况返回各块 `content` 用 `separator` 连接后的单个字符串。
- **内部流程**：第一步 `if not isinstance(separator, str): raise TypeError("separator must be a string")` 做类型防御。第二步 `return separator.join(chunk.content for chunk in self.retrieve(query, limit=limit))`——用生成器表达式惰性取出每个块的 `content` 再交给 `str.join`。没有缓存、没有截断、没有异常兜底。
- **异常/边界**：`separator` 非字符串抛 `TypeError`（注意 `bool` 是 `int` 不是 `str`，传 `True` 也会被拒）；`self.retrieve()` 抛出的异常（如数据库打不开）直接向上传播；`chunk.content` 为 `None` 时 `join` 会抛 `TypeError`；上下文总长度不做限制，可能非常长，是否截断由调用方决定。
- **同文件关系**：它调用本文件的 `retrieve()`；不被本文件其它函数调用，是外部调用的入口之一。

### `RAGPipeline.graph_retrieve(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, hops: int = RAG_GRAPH_HOPS, at: str | None = None) -> GraphRAGResult` （第 288 行）
- **作用**：执行图谱检索（多跳），返回结构化的 `GraphRAGResult`。为什么需要它：普通向量检索只能命中与查询字面/语义相近的块，而图谱检索会沿着实体之间的关系边走若干跳，能找出「没有直接提到关键词但通过关系链相关」的内容。这个方法本身不含逻辑，只是把调用转发给构造时创建的 `GraphRAGPipeline`，从而让 `RAGPipeline` 对外呈现统一的检索接口，调用方不必自己持有图谱流水线对象。`at` 参数让它能按某个时间点做「时间旅行」式的图谱查询，用于查看历史状态。
- **参数**：`query: str` 是检索查询文本；`limit: int = RAG_RETRIEVE_LIMIT`（keyword-only）是返回结果条数上限；`hops: int = RAG_GRAPH_HOPS`（keyword-only）是关系图上的扩展跳数，默认取自 `constants`；`at: str | None = None`（keyword-only）是可选的时间点过滤（形如时间戳字符串），`None` 表示按当前状态检索。
- **返回**：返回 `GraphRAGResult`（从 `memory/rag/graph_rag.py` 导入的类型），内容由被委托的 `GraphRAGPipeline.retrieve()` 决定，本方法不做任何加工或包装。
- **内部流程**：函数体只有一行 `return self.graph.retrieve(query, limit=limit, hops=hops, at=at)`，参数全部按关键字传递，无任何分支、校验或状态变更。
- **异常/边界**：自身不做任何校验，异常完全来自 `self.graph.retrieve()`（例如图谱存储不可用、`hops` 为负数导致遍历参数非法等），会原样向上抛。`at` 传非法时间格式的后果由图谱层决定。
- **同文件关系**：它调用实例属性 `self.graph`（外部 `GraphRAGPipeline`）；不被本文件其它函数调用；与 `graph_context()` 是「同一能力的不同返回形态」（结构化 vs 纯文本）。

### `RAGPipeline.graph_context(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, hops: int = RAG_GRAPH_HOPS, max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str` （第 298 行）
- **作用**：执行图谱检索并把结果渲染成一段受长度限制的纯文本上下文。为什么需要它：图谱检索的结果是结构化对象，但提示词需要文本；而且图谱上下文很容易膨胀（一跳可能就是几十个实体），所以这里专门暴露 `max_chars` 让调用方控制预算，默认值来自 `constants` 的 `RAG_CONTEXT_MAX_CHARS`。它与 `build_context()` 的区别在于：`build_context()` 是「向量检索结果的直接拼接、不截断」，而本方法是「图谱检索结果的渲染、带字符预算」。它同样只是转发，不做自己的拼装逻辑。
- **参数**：`query: str` 是检索查询；`limit: int = RAG_RETRIEVE_LIMIT`（keyword-only）是图谱检索的种子结果条数上限；`hops: int = RAG_GRAPH_HOPS`（keyword-only）是关系扩展跳数；`max_chars: int = RAG_CONTEXT_MAX_CHARS`（keyword-only）是渲染文本的最大字符数预算，由图谱层负责裁剪。
- **返回**：返回 `str`，即被委托的 `GraphRAGPipeline.build_context()` 的输出；没有结果或全部被裁剪时可能是空字符串（具体由图谱层实现决定）。
- **内部流程**：函数体只有一行 `return self.graph.build_context(query, limit=limit, hops=hops, max_chars=max_chars)`，四个参数全部按关键字传递，无本地变量、无分支、无异常处理。
- **异常/边界**：自身不校验任何参数；`max_chars` 传 0 或负数、`hops` 传负数的具体后果由图谱层决定，异常原样向上抛。不做缓存，每次调用都会重新检索并重新渲染。
- **同文件关系**：它调用实例属性 `self.graph`（外部 `GraphRAGPipeline`）；不被本文件其它函数调用；与 `graph_retrieve()` 共享同一底层能力。

### `RAGPipeline.delete_document(self, document_id: str) -> int` （第 301 行）
- **作用**：按文档 id 删除一份已入库文档及其派生记忆，返回删除的语义记忆条数。为什么需要它：一份文档入库后会产生一批块级语义记忆（以及抽取出的实体/关系记忆），如果只删真值源就会留下「永远查不到来源的孤儿行」，只删记忆又会留下孤儿文档记录；所以这个方法两边都清：先在记忆层按元数据里的 `document_id` 匹配并逐条删除，再同步删除真值源里的 `documents`/`chunks` 行。注意它匹配的是记忆条目的 `metadata["document_id"]` 字段，而不是用文档 id 去拼记忆 id，所以它删的是「块级条目」这一类带该元数据的记忆。删除真值源那一步的失败不会阻止返回已删除的记忆条数。
- **参数**：`document_id: str` 是要删除的文档 id，必须与入库时 `DocumentRecord.document_id` / 块元数据里的 `document_id` 一致，否则匹配不到任何记忆（但真值源删除仍会执行）。
- **返回**：返回 `int`，表示成功从记忆系统中删除的条目数量（即 `self.manager.delete(item.id)` 返回真值的次数）。注意这个数字只统计记忆条目的删除，不包含真值源里被删掉的行数。
- **内部流程**：第一步 `items = self.manager.list(memory_type=MemoryType.SEMANTIC, include_expired=True)`，列出全部语义记忆，并且 `include_expired=True` 以确保已过期的条目也能被清理掉（否则会残留）。第二步初始化计数器 `removed = 0`，进入 `for item in items` 循环：用 `item.metadata.get("document_id") == document_id` 判断归属，命中且 `self.manager.delete(item.id)` 返回真值时 `removed += 1`——注意这里用 `and` 短路，所以删除失败（返回假值）不会计数，但循环继续处理下一条。第三步调用 `self.document_repo()` 取真值源，若不为 `None` 则 `repository.delete_document(document_id)` 删除该文档及其块行（注释说明这是为了避免留下查不到来源的孤儿行）。第四步返回 `removed`。
- **异常/边界**：`self.manager.list()` 或 `delete()` 抛错时异常向上传播，`removed` 的中间状态会丢失（不会返回部分计数）；`document_id` 不存在时不会报错，返回 0（若真值源可用仍会尝试删一次，通常是空操作）；`item.metadata` 为 `None` 会抛 `AttributeError`（本方法不做判空）；`document_repo()` 抛错会打断流程，此时记忆已删但真值源未删，存在不一致窗口。方法本身不做事务，也不删除抽取出的实体/关系记忆（只按 `document_id` 元数据匹配）。
- **同文件关系**：它调用本文件的 `document_repo()`；不被本文件其它函数调用，是外部调用的维护入口。

### `RAGPipeline.close(self) -> None` （第 313 行）
- **作用**：释放这个流水线持有的资源，主要是真值源仓库连接和记忆管理器。为什么需要它：`document_repo()` 懒加载出来的 `DocumentRepository` 会持有一个 SQLite 连接，如果不显式关闭，进程退出或长时间运行后可能留下未释放的文件句柄/锁；而 `MemoryManager` 自己也有需要收尾的资源。这个方法把两处清理按顺序做掉，并把仓库缓存重置为 `None`，这样如果实例之后又被使用，`document_repo()` 会重新建立连接而不是继续用一个已关闭的连接。它适合在应用关闭（例如 FastAPI 的 shutdown 钩子）时调用。
- **参数**：只有 `self`，无其它参数。
- **返回**：无返回值（`None`）。
- **内部流程**：第一步 `if self._repository is not None:` 判断缓存是否存在，存在则调用 `self._repository.close()` 关闭连接，紧接着把 `self._repository = None` 置空（先关闭再置空，避免关闭失败时缓存里留着半死的对象——不过如果 `close()` 抛异常，置空这一步不会执行）。第二步无条件调用 `self.manager.close()` 关闭记忆管理器。
- **异常/边界**：`self._repository.close()` 抛异常会导致 `self._repository` 不被置空，且 `self.manager.close()` 不会被执行（没有 `try/finally` 保护）；重复调用是安全的，因为第二次 `self._repository` 已是 `None`，只会再次调用 `self.manager.close()`，其是否幂等取决于管理器实现；`self.manager` 为 `None` 会抛 `AttributeError`。
- **同文件关系**：它使用 `self._repository`（由本文件的 `document_repo()` 创建）；不被本文件其它函数调用，是外部调用的生命周期收尾入口。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `RetrievedChunk` | 不可变的检索结果数据类，承载正文、分数、记忆 id、元数据与分数明细。 |
| `RetrievedChunk.from_result` | 把记忆系统的 `MemorySearchResult` 转换成 `RetrievedChunk` 的工厂类方法。 |
| `accepts_parameter` | 用 `inspect` 自省抽取器 `extract` 签名，判断它是否接受某个关键字参数或 `**kwargs`。 |
| `_accepts_graph_context` | 判断抽取器能否消费抽取前的子图上下文，用于兼容旧的两参数抽取器契约。 |
| `_document_tags` | 把 metadata 里的 `tags` 归一化成字符串列表（缺失/空为 `[]`，单字符串包成一项）。 |
| `_document_permission` | 从 metadata 解析权限，白名单外一律失败即私有，退回 `"private"`。 |
| `RAGPipeline` | RAG 门面类，封装文档入库、知识抽取、混合检索、图谱检索、删除与资源释放。 |
| `RAGPipeline.__init__` | 装配记忆管理器、文档处理器、抽取器、图谱流水线，并初始化报告与仓库缓存。 |
| `RAGPipeline.document_repo` | 懒加载并缓存真值源仓库；非 SQLite 或 `:memory:` 时返回 `None`。 |
| `RAGPipeline.ingest` | 把文档切块写入真值源与语义记忆，按需抽取实体关系，并汇总入库报告。 |
| `RAGPipeline.ingest_source` | 按文件路径解析后入库，支持 `base_dir` 包含边界，其余参数转发给 `ingest`。 |
| `RAGPipeline.retrieve` | 语义检索并按 `chunks` 真值源校正正文，返回 `RetrievedChunk` 列表。 |
| `RAGPipeline.build_context` | 把检索到的块正文用分隔符拼成一段提示词上下文（不截断）。 |
| `RAGPipeline.graph_retrieve` | 转发给图谱流水线做多跳检索，返回结构化的 `GraphRAGResult`。 |
| `RAGPipeline.graph_context` | 转发给图谱流水线渲染带 `max_chars` 预算的图谱上下文文本。 |
| `RAGPipeline.delete_document` | 按文档 id 删除其语义记忆条目并同步清理真值源，返回删除条数。 |
| `RAGPipeline.close` | 关闭真值源仓库连接并置空缓存，再关闭记忆管理器。 |
