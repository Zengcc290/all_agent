# tool/document_revectorize.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时「知识星云」记忆系统里的一把**单文档重嵌入专用写工具**，职责范围被刻意收得很窄：只重建**一篇文档**的向量投影，并顺手把这篇文档的 chunk 向量状态和文档状态写成"已索引/已向量化"。它存在的原因是：当云端嵌入服务当时不可达、或者进程被 kill，导致"真值源里有分块、但向量投影缺失或落后"时，最省事的修复方式不是全量重建整个向量库，而是"只重灌这一篇"。这样做的副作用面被限制在一篇文档内——不碰其它文档，不碰知识图谱。

文件里包含五类东西：一个模块级常量 `TOOL_ENABLED = True`（标记本工具处于启用状态，供工具注册/发现机制读取）；一个纯函数 `revectorize_document`（真正的业务逻辑实现）；两个 Pydantic 数据模型 `DocumentRevectorizeInput` 与 `DocumentRevectorizeOutput`（分别描述工具的入参与出参契约）；一个继承自 `core.BaseTool` 的工具类 `DocumentRevectorizeTool`（把纯函数包装成 Agent 可调用的工具，带 spec、超时、权限、幂等性等元信息）；以及一个工厂函数 `create_tool`，供工具装载器以统一方式实例化。

从注释可以看出它的历史位置：这段逻辑原本内联在 `web/app.py` 的重嵌入端点里，后来被抽出来，本模块成为这段逻辑的**唯一实现**，Web 端点只保留 409/404/422/502 的错误码映射。它依赖外部的记忆管理器 `MemoryManager`、文档真值源仓库 `DocumentRepository`、嵌入锁闸门 `apply_embedding_lock`、以及同目录 `hybrid_index` 里的 `repository_for`（用来从 manager 反查出文档仓库，并识别"内存模式"这种没有真值源的情况）。

最关键的运行约束是**闸门顺序**：先过嵌入锁闸门，再检查文档是否存在，最后才检查有没有分块。顺序绝对不能换——当嵌入空间不一致时，必须优先报出"锁不一致"（上层映射成 409），而不是报"文档不存在"（404），因为往错误的向量空间里写数据比"文档没找到"严重得多。并且 `confirm_rebuild` 默认是 false，锁不一致时明确抛 `EmbeddingLockMismatch` 失败，只有调用方显式确认才会"重建投影并全量重灌"，绝不静默切换向量空间。

## 二、函数与类逐条详解

### `revectorize_document(manager: MemoryManager, repository: DocumentRepository, document_id: str, *, confirm_rebuild: bool = False) -> dict[str, Any]` （第 41 行）

- **作用**：这是整个文件的业务核心，负责把一篇文档的全部分块重新送入嵌入模型算出向量，再逐条写回向量库，并把每个分块的向量状态标记为 `indexed`，最后更新文档本身的状态。它被设计成一个不依赖 FastAPI、不依赖工具框架的纯业务函数，因此既能被工具类 `DocumentRevectorizeTool.execute` 调用，也能被其它内部代码（例如运维脚本、修复流程）直接调用。它需要 `manager` 和 `repository` 两个依赖都由调用方注入，而不是自己去构造，这样便于测试和复用。它的第一行就是调用嵌入锁闸门，说明"安全校验"是这段逻辑不可分割的一部分，而不是可选的前置步骤。函数末尾返回一个朴素字典（而不是 Pydantic 模型），把结果结构的组装责任留给上层工具类，从而保持本函数对数据契约零耦合。当文档不存在或没有分块时它会主动抛异常终止，因为"重嵌入一个没有内容的东西"在语义上是无意义的，静默返回 0 会让调用方误以为修复成功。

- **参数**：
  - `manager: MemoryManager`：记忆管理器，必须由调用方传入。它同时提供两样东西：`manager.embedding.embed_batch(...)`（批量嵌入能力）和 `manager.vector_store.upsert_chunk(...)`（向量写入能力）。同时它还会被透传给 `apply_embedding_lock` 做嵌入空间一致性校验。没有默认值，不允许为 None。
  - `repository: DocumentRepository`：文档真值源仓库，必须由调用方传入。负责读取文档记录、列出分块、写分块向量状态、写文档状态。没有默认值，不允许为 None。
  - `document_id: str`：要重嵌入的文档 id，必填位置参数。函数本身不对它做长度或格式校验（校验在上层 `DocumentRevectorizeInput` 里做），只把它原样传给仓库查询和错误信息拼接。
  - `confirm_rebuild: bool`：关键字限定参数（`*` 之后），默认 `False`。它的语义是"嵌入锁不一致时，是否确认重建整个向量投影并全量重灌（破坏性操作）"。默认 false 意味着锁不一致就明确失败；只有显式传 True 才允许切换向量空间。它不被本函数直接判断，而是原样透传给 `apply_embedding_lock`。

- **返回**：返回一个 `dict[str, Any]`，固定包含三个键：`"document_id"`（原样回传传入的文档 id）、`"chunks_reindexed"`（本次重新嵌入的分块数量，取 `len(chunks)`）、`"status"`（写入后的文档状态字符串）。`status` 的取值逻辑是：如果文档原本的状态就是 `"extracted"`，则保持 `"extracted"`，否则一律写成 `"vectorized"`。这个字典随后会被 `DocumentRevectorizeTool.execute` 用 `**result` 展开成 `DocumentRevectorizeOutput`。

- **内部流程**：
  1. 第一件事是调用 `apply_embedding_lock(manager, repository, confirm_rebuild=confirm_rebuild)`，完成嵌入锁闸门校验；这是全函数唯一的安全前置检查，注释明确强调顺序不能调换。
  2. 通过 `repository.get_document(document_id)` 取文档记录，结果赋给 `document`。
  3. 如果 `document is None`，抛 `LookupError(f"文档不存在：{document_id}")`，错误信息里带上文档 id 便于定位。
  4. 通过 `repository.list_chunks(document_id)` 取出该文档的全部分块，赋给 `chunks`。
  5. 如果 `chunks` 为假值（None 或空列表），抛 `ValueError("该文档没有分块，无法重嵌入")`。
  6. 用列表推导 `[chunk.text for chunk in chunks]` 抽取所有分块文本，一次性交给 `manager.embedding.embed_batch(...)` 得到 `vectors`。这里刻意做批量嵌入而不是循环单条嵌入，以减少网络往返。
  7. 用 `zip(chunks, vectors, strict=True)` 成对遍历分块与向量。`strict=True` 表示如果两者长度不一致会直接抛错，而不是静默截断——这是一个防止"嵌入结果缺失导致错位写库"的保险。
  8. 循环体内先调 `manager.vector_store.upsert_chunk(...)` 写入向量，参数包括 `chunk.chunk_id`、向量本体，以及关键字参数 `document_id=chunk.document_id`、`chunk_index=chunk.chunk_index`、`source=document.source`（来源取文档级字段）、`memory_type=MemoryType.SEMANTIC.value`（固定标记为语义记忆类型）。
  9. 写完向量紧接着调 `repository.set_chunk_vector_status(chunk.chunk_id, "indexed")`，把该分块的向量状态标记为已索引。写入与状态标记在同一个循环迭代内相邻执行，保证"写了向量的分块才被标记为 indexed"。
  10. 循环结束后计算 `status = "extracted" if document.status == "extracted" else "vectorized"`，即保留 extracted 这一特殊前置状态，其余一律置为 vectorized。
  11. 调 `repository.set_status(document_id, status)` 落库。
  12. 返回上面描述的三键字典。

- **异常/边界**：
  - `apply_embedding_lock` 在嵌入锁不一致且 `confirm_rebuild` 为 False 时会抛 `EmbeddingLockMismatch`（文件头注释明确说明），本函数不做捕获，直接向上冒泡。
  - 文档不存在时抛 `LookupError`，消息为"文档不存在：{document_id}"。
  - 文档存在但没有任何分块时抛 `ValueError("该文档没有分块，无法重嵌入")`。
  - 嵌入结果数量与分块数量不一致时，`zip(..., strict=True)` 抛 `ValueError`。
  - `manager.embedding.embed_batch` 或 `manager.vector_store.upsert_chunk` 内部的网络/超时/后端异常不做捕获，原样向上抛。
  - 函数不做事务回滚：如果循环中途某次 `upsert_chunk` 失败，此前已写入的分块会保持已写状态，属于"部分完成"。文件本身没有补偿逻辑。
  - 对 `document_id` 为 None、空串或非法格式不做校验，由上层 Pydantic 模型负责。

- **同文件关系**：它不调用本文件里的任何其它函数；被本文件的 `DocumentRevectorizeTool.execute` 调用（在拿到 repository 之后）。它依赖的 `apply_embedding_lock`、`MemoryType`、`MemoryManager`、`DocumentRepository` 均来自其它模块的导入，不属于本文件。

### `DocumentRevectorizeInput` （第 77 行）

- **作用**：这是一个 Pydantic 输入契约模型，用来描述"重嵌入一篇文档"这个工具调用允许携带哪些参数、每个参数的类型和约束是什么。它存在的意义是把校验从业务函数里剥离出来：业务函数假设输入已经合法，而模型层负责把非法输入挡在门外，这样工具被 Agent（尤其是 LLM 生成的参数）调用时不会把脏数据带进写流程。它的 `model_config` 使用 `ConfigDict(extra="forbid", strict=True)`：`extra="forbid"` 表示出现任何未声明的字段就直接报错（防止模型瞎编参数被默默忽略），`strict=True` 表示不做宽松类型强转（比如字符串 "true" 不会被当作布尔 True）。它被 `DocumentRevectorizeTool.spec` 通过 `input_model=DocumentRevectorizeInput` 引用，从而由工具框架在调用 `execute` 之前自动完成实例化与校验。本类没有定义任何方法，全部行为来自 Pydantic 基类和字段声明。

- **参数**：作为类本身没有函数参数；其字段即构造参数：
  - `document_id: str`：必填字段，`Field(min_length=1, max_length=200, description="要重嵌入的文档 id。")`。约束是长度必须在 1 到 200 之间（含边界），因此空字符串会被拒绝，超长 id 也会被拒绝。
  - `confirm_rebuild: bool`：可选字段，默认 `False`，描述为"嵌入锁不一致时是否确认重建整个向量投影并全量重灌（破坏性）。默认 false 表示明确失败，绝不静默切换向量空间。"在 `strict=True` 下必须是真正的布尔值。

- **返回**：类本身不"返回"值；实例化后得到一个不可变语义的校验结果对象（Pydantic 模型实例），其 `.document_id` 与 `.confirm_rebuild` 属性供 `execute` 读取。校验失败时 Pydantic 抛 `ValidationError`，而不是返回 None。

- **内部流程**：没有自定义方法体。实例化时 Pydantic 按字段声明顺序读取传入的映射/关键字参数，先应用 `extra="forbid"` 拒绝未知键，再对 `document_id` 做 `str` 类型与长度区间校验，对 `confirm_rebuild` 做严格布尔校验并在缺省时填入默认值 `False`，全部通过后构造出实例。

- **异常/边界**：缺 `document_id`、传空串、传超过 200 字符的字符串、传非字符串类型、传 `confirm_rebuild="yes"` 这类非布尔值、或传入任何额外字段，都会抛 Pydantic 的 `ValidationError`（在 FastAPI 端点层通常映射为 422）。`confirm_rebuild` 缺失时不会报错，取默认 False。

- **同文件关系**：被 `DocumentRevectorizeTool.spec` 的 `input_model` 引用；被 `DocumentRevectorizeTool.execute` 的类型注解引用（其入参就是本类实例）；被 `__all__` 导出。它不调用本文件任何函数。

### `DocumentRevectorizeOutput` （第 90 行）

- **作用**：这是与输入配对的 Pydantic 输出契约模型，用来规定"重嵌入"这个工具调用必须返回什么样的结构化结果。工具框架用它来校验 `execute` 的返回值、生成给 Agent 看的结构化输出，以及保证上层（例如 Web 端点）拿到的字段名和类型稳定。同样采用 `ConfigDict(extra="forbid", strict=True)`，即输出里多出任何字段都算错误，类型也必须严格匹配，这相当于对业务函数返回值的一道回归保险。它被 `DocumentRevectorizeTool.spec` 通过 `output_model` 声明，并在 `execute` 末尾用 `DocumentRevectorizeOutput(**result)` 从字典构造。本类不定义任何方法。

- **参数**：作为类本身没有函数参数；其字段即构造参数：
  - `document_id: str`：必填，无额外约束，原样回传被处理的文档 id。
  - `chunks_reindexed: int`：必填，描述"本次重新嵌入的分块数。"，来自业务函数返回的 `len(chunks)`。
  - `status: str`：必填，描述"写入后的文档状态：vectorized，或保持 extracted。"，取值由业务函数按原状态决定。

- **返回**：类本身不返回值；构造成功得到一个包含上述三个字段的模型实例，供工具框架序列化返回。字段缺失或类型不符时抛 `ValidationError`。

- **内部流程**：无自定义方法体。构造时 Pydantic 依次校验三个必填字段存在且类型严格正确（`str`/`int`/`str`），并拒绝任何额外键，然后生成实例。

- **异常/边界**：如果 `revectorize_document` 的返回值字典缺少任一字段、字段名拼写不符、或 `chunks_reindexed` 不是整数（例如传了字符串），构造会抛 `ValidationError`。严格模式下 `bool` 不被当作 `int` 接受（Pydantic 严格模式对 bool/int 有区分），这是它的边界之一。

- **同文件关系**：被 `DocumentRevectorizeTool.spec` 的 `output_model` 引用；被 `DocumentRevectorizeTool.execute` 用于构造返回值；被 `__all__` 导出。它不调用本文件任何函数。

### `DocumentRevectorizeTool` （第 98 行）

- **作用**：这是把 `revectorize_document` 这个纯业务函数包装成 Agent 可调用工具的类，继承自 `core.BaseTool`。它的类属性 `spec` 是一个 `ToolSpec`，集中声明了工具的对外元信息：名称 `knowledge.document_revectorize`、一段英文描述（说明它把一篇文档的分块重新嵌入向量投影并标记为 indexed，嵌入空间不一致时会大声失败，除非设置了 `confirm_rebuild`，因为静默混用嵌入空间会污染检索）、版本 `1.0.0`、输入输出模型、副作用类型 `write`、权限元组为空 `()`、超时 600 秒、幂等 `idempotent=True`、并行不安全 `parallel_safe=False`、标签 `("knowledge", "document", "embedding", "write")`，以及一段中文 `guidance`（指导 Agent 只在真值源有分块而向量投影缺失或落后时使用，通常由 `knowledge.reconcile` 报告触发；嵌入空间不一致时会明确失败，只有确认要重建向量空间才传 `confirm_rebuild=true`，不要用它掩盖配置错误；没有分块的文档会被拒绝）。这些元信息让上层调度器知道该工具是写操作、要独占执行、可以重试（幂等）、且需要较长超时。类本身只定义了三个成员：`__init__`、`manager` 属性和 `execute`。

- **参数**：类本身没有构造参数（由 `__init__` 定义，见下条）；类属性 `spec` 的各字段如上所述，其中 `side_effect="write"` 表示它会修改持久状态，`parallel_safe=False` 表示不应与其它操作并发执行，`idempotent=True` 表示重复调用同一文档不会造成额外的语义变化（因为 `upsert_chunk` 是覆盖写、状态设置是幂等的），`timeout_seconds=600.0` 表示单次调用允许最多 10 分钟（批量嵌入可能较慢）。

- **返回**：类本身不返回值；实例化得到工具对象，供工具注册表登记并调用其 `execute`。

- **内部流程**：类体在定义时即执行 `spec = ToolSpec(...)`，完成元信息构建；随后定义 `__init__`、`manager`（property）与 `execute` 三个成员。运行期由框架读取 `spec` 做注册与校验，再调用 `execute` 完成实际工作。

- **异常/边界**：类定义阶段若 `ToolSpec` 的构造参数不合法（例如名称格式或字段类型不符），会在导入本模块时立即抛错，属于启动期失败。`permissions=()` 表示不声明任何额外权限要求。其余异常处理在下属方法中说明。

- **同文件关系**：它引用本文件的 `DocumentRevectorizeInput`、`DocumentRevectorizeOutput` 和 `revectorize_document`；被本文件的 `create_tool` 实例化并返回；被 `__all__` 导出。它内部通过 `from ._memory import build_default_manager`（在 `manager` 属性里延迟导入）和 `from .hybrid_index import repository_for`（模块级导入）与同目录其它模块协作。

### `DocumentRevectorizeTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 122 行）

- **作用**：这是工具类的构造函数，唯一职责是接收一个可选的记忆管理器并把它存到实例私有属性 `self._manager` 上。之所以把 manager 设计成"可选注入"，是为了让本工具既能被正常运行时使用（不传 manager，等真正执行时惰性构建默认管理器），也能在测试或特殊装配场景下被注入一个现成的、可替换的 manager。它不做任何构建、校验或 IO，因此构造本身永远不会因为记忆库不可用而失败——这一点很重要，因为工具通常在应用启动时就被注册，此时不应该强迫初始化重量级的记忆后端。惰性构建的逻辑被放在 `manager` 属性里，而不是构造函数里。

- **参数**：
  - `self`：实例自身。
  - `manager: MemoryManager | None`：可选，默认 `None`。传入一个 `MemoryManager` 实例则直接使用；传 `None`（或不传）表示"稍后惰性构建默认管理器"。不做类型运行时校验。

- **返回**：返回 `None`（构造函数语义，返回新实例）。

- **内部流程**：唯一一步是 `self._manager = manager`，把参数原样保存为私有属性。

- **异常/边界**：无特殊处理。传任何对象（哪怕是错误类型的对象）都不会在构造时报错，类型问题会推迟到 `execute` 时以属性访问错误的形式暴露。重复构造互不影响。

- **同文件关系**：它设置的 `self._manager` 被本文件的 `manager` 属性和 `execute` 方法读取。它不调用本文件任何函数。它被 `create_tool` 间接触发（`create_tool` 调用 `DocumentRevectorizeTool()`，即使用默认的 `manager=None`）。

### `DocumentRevectorizeTool.manager` （第 125 行，`@property`）

- **作用**：这是一个只读属性，作用是"按需提供记忆管理器"。它实现了惰性初始化：如果 `self._manager` 还是 `None`，就先从同目录 `_memory` 模块导入 `build_default_manager` 并调用它构建一个默认管理器，缓存到 `self._manager`，然后返回。这样做的收益是：工具对象可以在启动时零成本注册，直到第一次真正执行重嵌入时才去构建记忆后端；同时构建结果被缓存，后续调用不会重复构建。它把"从哪拿 manager"这个环境耦合点集中到一处，让 `execute` 只需要写 `self.manager` 而不必关心是否被注入过。如果构造时已经注入过 manager，这里直接返回注入的实例，保证测试可替换性。

- **参数**：无（除隐式 `self`）。属性的"取值"不需要参数。

- **返回**：返回一个 `MemoryManager` 实例。若已有缓存（无论来自构造注入还是上次惰性构建）就返回该缓存；否则构建一个默认管理器并返回它。类型注解声明为 `MemoryManager`。

- **内部流程**：
  1. 判断 `if self._manager is None:`。
  2. 条件成立时执行函数内延迟导入 `from ._memory import build_default_manager`（放在函数体内是为了避免模块级循环导入，也避免在不需要时加载重量级依赖）。
  3. 调 `build_default_manager()` 并把返回值赋给 `self._manager`。
  4. 返回 `self._manager`。

- **异常/边界**：如果 `_memory` 模块不可导入或 `build_default_manager()` 内部失败（例如配置缺失、后端不可达），异常会从属性访问处抛出，不做捕获，也不缓存失败结果——因此下次访问会重试构建。如果构造时传入的是错误类型的对象（非 None），本属性不会察觉，会把错误对象原样返回，问题推迟到 `execute` 使用其属性时爆发。

- **同文件关系**：它读取 `__init__` 写入的 `self._manager`；被本文件的 `execute` 方法通过 `self.manager` 访问（`execute` 里出现三次：`repository_for(self.manager)` 和传给 `revectorize_document` 的那次以及 `repository_for` 内部的调用）。它不调用本文件的其它函数。

### `DocumentRevectorizeTool.execute(self, arguments: DocumentRevectorizeInput) -> DocumentRevectorizeOutput` （第 133 行）

- **作用**：这是工具的实际执行入口，由工具框架在完成输入校验后调用。它承担三件事：把"管理器"翻译成"文档真值源仓库"、在内存模式下提前拒绝、调用纯业务函数并把朴素字典结果转换成强类型的输出模型。它之所以需要先做仓库探测，是因为 `revectorize_document` 要求一个 `DocumentRepository` 来读文档和分块，而当记忆库运行在 `:memory:` 内存模式时根本不存在 documents/chunks 真值源，此时继续执行只会得到无意义的失败；因此这里显式抛 `LookupError` 并给出清晰的中文原因，让上层能映射成合适的错误码。它也是把异常类型（`EmbeddingLockMismatch`、`LookupError`、`ValueError`）留给上层做 409/404/422/502 映射的那一层——注释说明 Web 端点只保留错误码映射，本方法不吞异常、不做降级。

- **参数**：
  - `self`：实例自身。
  - `arguments: DocumentRevectorizeInput`：已通过 Pydantic 校验的输入模型实例，本方法从中读取 `arguments.document_id`（字符串，长度 1–200）和 `arguments.confirm_rebuild`（布尔，默认 False）。

- **返回**：返回一个 `DocumentRevectorizeOutput` 实例，由业务函数返回的字典 `**result` 展开构造，包含 `document_id`、`chunks_reindexed`、`status` 三个字段。正常路径下一定会返回该模型，不会返回 None。

- **内部流程**：
  1. 调 `repository_for(self.manager)` 从记忆管理器中取出文档真值源仓库，结果赋给 `repository`。这次 `self.manager` 访问会触发惰性构建（如果尚未构建）。
  2. 判断 `if repository is None:`，成立则抛 `LookupError("当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源")`。
  3. 调本文件的 `revectorize_document(self.manager, repository, arguments.document_id, confirm_rebuild=arguments.confirm_rebuild)`，把两个依赖和两个参数原样传下去，得到 `result` 字典。
  4. 调 `DocumentRevectorizeOutput(**result)` 把字典转成输出模型并返回。

- **异常/边界**：
  - `repository_for(self.manager)` 返回 `None` 时抛 `LookupError`（消息指出是 `:memory:` 内存模式）。
  - 透传 `revectorize_document` 的所有异常：`EmbeddingLockMismatch`（嵌入锁不一致且未确认重建）、`LookupError`（文档不存在）、`ValueError`（没有分块或 zip 长度不匹配）、以及嵌入/写库过程中的底层异常。本方法不做捕获、不做重试、不做日志。
  - 若 `result` 的键与 `DocumentRevectorizeOutput` 字段不匹配，构造输出模型时会抛 Pydantic `ValidationError`。
  - 没有对 `arguments` 为 None 的防护，传 None 会在访问 `.document_id` 时抛 `AttributeError`。

- **同文件关系**：它调用本文件的 `revectorize_document`，并构造本文件的 `DocumentRevectorizeOutput`；它通过 `self.manager` 使用本文件 `manager` 属性提供的管理器；它被工具框架（外部）调用，本文件内没有其它函数调用它。

### `create_tool() -> BaseTool` （第 148 行）

- **作用**：这是一个零参数的工厂函数，作用是提供一个统一的、无参的工具构造入口。工具装载/发现机制通常要求每个工具模块导出一个无参可调用对象来创建工具实例，这样注册表就不必知道每个工具类各自的构造签名（例如 `DocumentRevectorizeTool` 的 `manager` 参数）。这里它简单地返回 `DocumentRevectorizeTool()`，即不注入 manager，让工具在首次执行时惰性构建默认管理器。它不缓存实例，每次调用都产生一个新的工具对象，因此调用方需要自行决定是否复用。

- **参数**：无。

- **返回**：返回一个 `BaseTool`（具体类型是 `DocumentRevectorizeTool`）的新实例。总是返回实例，不会返回 None。

- **内部流程**：唯一一步是 `return DocumentRevectorizeTool()`，触发 `DocumentRevectorizeTool.__init__` 并以 `manager=None` 完成构造；`spec` 是类属性，在模块导入时就已构建好，此处不会重新构建。

- **异常/边界**：无特殊处理。只有当 `DocumentRevectorizeTool` 的构造过程本身抛错时才会失败，而当前构造逻辑只做一次属性赋值，实际不会失败。

- **同文件关系**：它调用本文件的 `DocumentRevectorizeTool`（构造函数，进而执行 `__init__`）；被外部工具装载器调用，本文件内没有其它函数调用它；它被列在 `__all__` 中对外导出。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `revectorize_document` | 先过嵌入锁闸门，再校验文档存在与有分块，批量重算向量并逐条 upsert、标记 indexed，最后把文档状态置为 vectorized（原为 extracted 则保持），返回结果字典。 |
| `DocumentRevectorizeInput` | Pydantic 输入契约，声明 `document_id`（1–200 字符）与默认 False 的 `confirm_rebuild`，并禁止额外字段与宽松类型转换。 |
| `DocumentRevectorizeOutput` | Pydantic 输出契约，声明 `document_id`、`chunks_reindexed`、`status` 三个必填字段，并禁止额外字段与宽松类型转换。 |
| `DocumentRevectorizeTool` | 继承 `BaseTool` 的工具类，通过 `spec` 声明名称、描述、版本、输入输出模型、写副作用、600 秒超时、幂等、非并行安全等元信息。 |
| `DocumentRevectorizeTool.__init__` | 接收可选的 `MemoryManager` 并存入 `self._manager`，不做任何构建或校验，支持注入与惰性初始化两种模式。 |
| `DocumentRevectorizeTool.manager` | 只读属性，`self._manager` 为空时延迟导入并调用 `build_default_manager()` 构建并缓存默认管理器，否则直接返回已缓存的实例。 |
| `DocumentRevectorizeTool.execute` | 从管理器取出文档仓库（内存模式则抛 `LookupError`），调用 `revectorize_document`，再把返回字典构造为 `DocumentRevectorizeOutput` 返回。 |
| `create_tool` | 无参工厂函数，返回一个以默认参数构造的 `DocumentRevectorizeTool` 实例，供工具装载器统一调用。 |
