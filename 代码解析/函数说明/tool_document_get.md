# tool/document_get.py

## 一、这个文件是干什么的

这个文件实现的是「文档详情」这一项独立的只读工具能力，工具全名为 `knowledge.document_get`，是知识库工具族中的一员。它解决的问题是：当向量检索命中了一个 `chunk_id`、需要回到真值源（truth source）核对原文时，必须有一个地方能一次读到某篇文档的元数据、完整原文以及它全部分块的边界与投影状态（`vector_status`）。整个文件的逻辑分为三层：最底层是模块级函数 `get_document()`，负责从 `DocumentRepository` 取文档与分块并拼装成字典；中间层是三个 Pydantic 模型 `DocumentGetInput`、`DocumentChunk`、`DocumentGetOutput`，分别描述入参、单个分块和出参的形状，全部启用 `extra="forbid"` 与 `strict=True` 的严格校验；最上层是继承 `BaseTool` 的 `DocumentGetTool`，通过类属性 `spec` 声明工具元信息（名称、版本、输入输出模型、只读副作用、超时、幂等、并发安全、标签、给模型的使用指引），并在 `execute()` 中完成真正的调用编排。此外文件还提供 `create_tool()` 工厂函数供注册器创建实例，并用 `__all__` 显式导出公开符号。

由于工具是只读的（`side_effect="read"`）、幂等的（`idempotent=True`）且并发安全的（`parallel_safe=True`），它可以被 Agent 在任意时刻反复调用而不会污染状态。文档不存在这种「真值缺失」的情形用 `LookupError` 表达，把「该返回 404 还是别的错误码」的决定权交给上层调用方，而不是在这个文件里硬编码 HTTP 语义。文件头部注释特别声明：本模块是这段逻辑的**唯一实现**，`web/app.py` 里原先内联的详情构造代码已经被删除，因此它是文档详情能力的单一事实来源。

依赖方面，它从 `core` 引入 `BaseTool` 与 `ToolSpec`，从 `memory.manager` 引入 `MemoryManager`，从 `memory.storage.document_repo` 引入 `DocumentRepository`，并从同包内引入 `_documents.document_detail`（真正负责把 ORM 对象摊平成字典）与 `hybrid_index.repository_for`（从记忆管理器里取出文档仓储）。值得注意的是，`manager` 属性内部对 `_memory.build_default_manager` 采用了延迟导入，避免模块导入期的循环依赖。文件用 `from __future__ import annotations` 让类型注解延迟求值，因此 `MemoryManager | None` 这类新式联合写法在旧解释器上也能通过解析。

## 二、函数与类逐条详解

### `get_document(repository: DocumentRepository, document_id: str) -> dict[str, Any]` （第 30 行）

- **作用**：这是本文件最核心的数据读取函数，负责「拿到一篇文档的完整视图」：既包含元数据，也包含原文，还包含这篇文档的全部切片。它被设计成一个不依赖工具框架的纯函数，只要求调用方给一个仓储对象和一个文档 id，因此既可以被 `DocumentGetTool.execute()` 使用，也可以被其它需要详情的地方直接复用。之所以要单独抽出来，是因为「取文档 → 判空 → 取分块 → 组装」这四步在语义上属于同一个原子动作，散落各处会导致错误处理不一致。函数把「文档不存在」这种情况统一翻译成 `LookupError`，让上层只面对一种失败信号。它本身不做权限判断、不做截断、不做字段裁剪，那些都交给上层的模型层处理。
- **参数**：
  - `repository: DocumentRepository`：文档仓储对象，必须已经连接到真实的持久化后端（如 SQLite），它需要同时具备 `get_document(document_id)` 与 `list_chunks(document_id)` 两个能力。传 `None` 或传入内存模式下的空仓储会在调用其方法时失败，因此调用方（`execute`）在更早的步骤里已经做了 `None` 判断。
  - `document_id: str`：要查询的文档 id。类型为字符串，没有在本函数内部做长度或格式校验（长度约束 `min_length=1, max_length=200` 由 `DocumentGetInput` 在更外层保证）。按业务约定它来自 `knowledge.document_list` 的输出。
- **返回**：返回一个 `dict[str, Any]`，内容是 `_documents.document_detail` 的产物，即包含 `document_id`、`title`、`raw_text`、`source`、`tags`、`permission`、`status`、`error`、`created_at`、`updated_at` 以及 `chunks` 列表的扁平字典。该字典的键与 `DocumentGetOutput` 的字段一一对应，`execute()` 会逐键取出后构造输出模型。
- **内部流程**：第一步调用 `repository.get_document(document_id)` 取文档对象，结果可能是 `None`；第二步用 `if document is None` 判断，为 `None` 时抛出 `LookupError(f"文档不存在：{document_id}")`，消息里带上原始 id 便于排查；第三步在文档确实存在的前提下，调用 `repository.list_chunks(document_id)` 取回该文档的全部分块；第四步把文档对象与分块列表一起交给 `document_detail(document, repository.list_chunks(document_id))`，由它完成字段提取与序列化，并把返回值直接 `return` 出去。整个函数没有循环、没有分支嵌套，是典型的「查—判—装」线性结构。
- **异常/边界**：文档不存在时主动抛 `LookupError`（消息为中文「文档不存在：<id>」）。`document_id` 为空字符串时不会在这里被拦下，而是由 `DocumentGetInput` 的 `min_length=1` 在 Pydantic 校验阶段拒绝。若仓储自身在访问底层数据库时抛错（例如连接断开、表缺失），本函数不做捕获与包装，异常原样向上传播。若文档存在但没有任何分块，`list_chunks` 返回空列表，函数正常返回、`chunks` 为空，不视为错误。
- **同文件关系**：它调用了本文件之外的两个依赖：`DocumentRepository` 的两个方法（外部对象）与同包 `_documents.document_detail`（外部函数）；在本文件内部，它被 `DocumentGetTool.execute()` 调用。它自身不调用本文件里的任何其它函数。

### `class DocumentGetInput(BaseModel)` （第 42 行）

- **作用**：这是 `knowledge.document_get` 工具的输入模型，用来在真正执行业务逻辑之前把模型（LLM）给出的参数做一次强校验和规范化。它存在的意义是把「参数非法」和「业务失败」两类问题彻底分开：参数问题由 Pydantic 在校验期直接拒绝，业务问题才进入 `execute()` 变成 `LookupError` 等异常。通过把 `model_config` 设为 `ConfigDict(extra="forbid", strict=True)`，任何多余字段都会被拒绝，类型也不会被隐式转换（例如把数字字符串当整数用会被拒），这在与 LLM 交互时非常重要，可以尽早暴露模型瞎编参数的问题。该模型还被 `ToolSpec(input_model=...)` 引用，用于自动生成给模型看的 JSON Schema 描述。
- **参数**（即模型的字段）：
  - `document_id: str`：必填，`Field(min_length=1, max_length=200, description="文档 id（来自 knowledge.document_list）。")`。长度为 1 到 200 个字符，通常形如 `doc_xxx` 的标识符；描述里明确提示它的来源是文档列表工具的输出。
  - `max_chunks: int`：可选，默认值 `50`，约束 `ge=1, le=1000`，描述为「最多返回多少个分块（chunk_count 始终是完整数量）」。它只影响返回列表的裁剪长度，不影响总数统计。
- **返回**：类本身不返回值；它的实例代表一次合法调用，实例属性 `document_id` 与 `max_chunks` 会被 `execute()` 读取使用。
- **内部流程**：类体只做两件事——声明 `model_config` 打开「禁止额外字段 + 严格类型」；声明两个字段及其约束与描述。真正的校验逻辑由 Pydantic 的 `BaseModel` 在实例化时执行：先检查是否有多余键（有则报错），再逐字段检查类型与数值范围，最后把结果填入实例。
- **异常/边界**：校验失败时由 Pydantic 抛出 `ValidationError`（本文件不捕获，交由工具框架统一转成参数错误）。`document_id` 缺失、为空串、超过 200 字符，或 `max_chunks` 小于 1、大于 1000、不是整数（严格模式下 `"50"` 也会被拒）都会触发该异常。
- **同文件关系**：被 `DocumentGetTool.spec` 通过 `input_model=DocumentGetInput` 引用，也被 `DocumentGetTool.execute()` 的类型注解使用；它是 `DocumentGetOutput` 的入参对应物。它不调用本文件里的任何函数。

### `class DocumentChunk(BaseModel)` （第 54 行）

- **作用**：这个模型描述单个分块（chunk）在工具输出里的形状，是详情结果中「分块列表」的元素类型。它把分块的关键信息压缩成六个字段：身份、序号、字符区间、正文以及向量投影状态，使 Agent 既能看到原文片段，也能判断这段文字在原文里的位置，还能知道它有没有被成功投影到向量库。把它独立成模型而不是直接透传仓储返回的字典，是为了保证输出结构稳定、字段可预期，并在 `extra="forbid"` 的保护下防止底层多出来的列泄漏到对外契约里。它同时被 `DocumentGetOutput.chunks` 用作列表元素类型，从而在最终序列化时获得统一的字段顺序与默认值。
- **参数**（即模型的字段）：
  - `chunk_id: str`：必填，无默认值，分块的唯一标识，通常正是向量检索命中时返回的那个 id。
  - `chunk_index: int`：默认 `0`，分块在文档内的顺序号，用于按原文顺序排列与定位。
  - `char_start: int`：默认 `0`，分块在原文中的起始字符偏移。
  - `char_end: int`：默认 `0`，分块在原文中的结束字符偏移（与 `char_start` 一起界定边界，便于核对切片是否切坏）。
  - `text: str`：默认 `""`，分块正文，即真正要被引用或核对的内容。
  - `vector_status: str`：默认 `""`，向量投影状态字符串，用来判断该分块是否已入向量库。
- **返回**：类本身不返回值；其实例表示一个分块，实例会被放进 `DocumentGetOutput.chunks` 列表返回给调用方。
- **内部流程**：声明严格配置后依次声明六个字段。实际构造在 `DocumentGetTool.execute()` 中通过 `DocumentChunk(**chunk)` 完成，即把 `document_detail` 产出的每个分块字典按关键字展开传入；由于配置了 `extra="forbid"` 与 `strict=True`，如果 `document_detail` 返回了本模型未声明的键，构造会直接报错，从而暴露出契约不一致。
- **异常/边界**：构造时若字段类型不匹配（例如 `chunk_index` 传了字符串）或出现未声明字段，Pydantic 抛 `ValidationError`；缺少必填的 `chunk_id` 同样报错。其余字段均有默认值，缺省时退化为 `0` 或空串，不会抛错。
- **同文件关系**：被 `DocumentGetOutput.chunks` 引用为元素类型，并在 `DocumentGetTool.execute()` 中被实例化。它不调用本文件里的任何函数。

### `class DocumentGetOutput(BaseModel)` （第 65 行）

- **作用**：这是工具的输出模型，定义了 `knowledge.document_get` 成功返回时给调用方（最终是给 LLM）看到的完整数据结构。它把「文档元数据 + 原文 + 分块列表 + 截断标记」打包成一份自描述的结果，其中 `chunk_count` 与 `truncated` 这一对字段是刻意设计的：前者始终是分块的真实总数，后者表明返回的 `chunks` 是否被 `max_chunks` 截断，这样模型即使只拿到前 50 个分块也能知道文档到底有多少块、自己是否读全了。`error` 字段用来承载入库或抽取阶段的失败原因，成功时为 `null`，使「文档存在但内容处理失败」这种中间状态也能被如实表达。该模型被 `ToolSpec(output_model=...)` 引用，用于生成输出 Schema。
- **参数**（即模型的字段）：
  - `document_id: str`：必填，文档唯一标识，回显输入以便调用方对齐。
  - `title: str`：默认 `""`，文档标题。
  - `raw_text: str`：默认 `""`，文档完整原文，是核对引用时的真值来源。
  - `source: str`：默认 `""`，文档来源描述（例如文件路径或导入渠道）。
  - `tags: list[str]`：默认空列表（`default_factory=list`），文档标签集合。
  - `permission: str`：默认 `""`，权限标记。
  - `status: str`：默认 `""`，文档处理状态。
  - `error: str | None`：默认 `None`，描述为「入库/抽取失败原因；成功时为 null」。
  - `created_at: str`：默认 `""`，创建时间（字符串形式）。
  - `updated_at: str`：默认 `""`，更新时间（字符串形式）。
  - `chunk_count: int`：必填（无默认值），描述为「分块总数（不受 max_chunks 影响）」。
  - `chunks: list[DocumentChunk]`：默认空列表，实际返回的分块明细，长度最多为 `max_chunks`。
  - `truncated: bool`：必填（无默认值），描述为「true 表示分块被 max_chunks 截断」。
- **返回**：类本身不返回值；其实例即工具的最终返回值，会被工具框架序列化成 JSON 交给上层。
- **内部流程**：声明严格配置与全部字段后，由 `DocumentGetTool.execute()` 用关键字参数一次性构造：把 `payload` 字典里的同名字段逐个取出填入，`tags` 用 `list(...)` 复制一份防止外部字典被共享引用，`chunk_count` 传入分块总数，`chunks` 传入切片后的列表，`truncated` 传入「总数是否大于上限」的布尔结果。
- **异常/边界**：构造时缺字段或类型不符会抛 `ValidationError`；`error` 允许显式为 `None`。注意 `chunk_count` 与 `truncated` 没有默认值，构造时必须提供，这是一种「强制显式声明是否截断」的设计。
- **同文件关系**：被 `DocumentGetTool.spec` 通过 `output_model=DocumentGetOutput` 引用，被 `DocumentGetTool.execute()` 的返回类型注解使用，并在 `execute()` 中被实例化；它内部引用 `DocumentChunk` 作为列表元素类型。它不调用本文件里的任何函数。

### `class DocumentGetTool(BaseTool)` （第 83 行）

- **作用**：这是对外暴露的工具类本体，把前面那些函数和模型组装成一个符合项目工具协议的、可被注册与调度的能力。它通过类属性 `spec`（一个 `ToolSpec` 实例）声明元信息：名称 `knowledge.document_get`；英文描述说明「从真值源读一篇文档：元数据、原文与分块（每个带 vector_status），用于在引用检索命中之前先核对它到底说了什么」；版本 `1.0.0`；输入输出模型分别指向 `DocumentGetInput` 与 `DocumentGetOutput`；`side_effect="read"` 表明只读；`permissions=()` 表示不需要额外权限；`timeout_seconds=60.0` 给出 60 秒超时；`idempotent=True` 与 `parallel_safe=True` 表示可以安全重复调用与并发调用；`tags=("knowledge", "document", "detail", "read")` 用于分类检索；`guidance` 用中文给出给模型的使用建议——引用或核对原文时使用、先用 `knowledge.document_list` 拿 id、`max_chunks` 只裁剪返回列表而 `chunk_count` 仍是全量、判断是否读完看 `truncated`、文档不存在会明确报错且不要反复重试同一个 id。这些声明是工具能被框架正确调度、被模型正确使用的关键，因此 `spec` 虽然是一个类属性而不是方法，也属于本类最重要的组成部分。
- **参数**：类本身无构造参数语义（构造由下面的 `__init__` 定义）；`spec` 中的字段含义如上所述。
- **返回**：类不返回值；实例化后由 `execute()` 产生 `DocumentGetOutput`。
- **内部流程**：类体先定义 `spec`，再定义 `__init__`、`manager` 属性和 `execute` 三个成员。框架读取 `spec` 完成注册、参数校验（用 `input_model`）与结果序列化（用 `output_model`），运行时再调用 `execute()`。
- **异常/边界**：`spec` 本身在类定义时构造，若 `ToolSpec` 的字段名或类型写错会在导入期就报错，属于「早失败」设计。运行期异常见 `execute()` 条目。
- **同文件关系**：它引用 `DocumentGetInput`、`DocumentGetOutput`；被 `create_tool()` 实例化；它调用了本文件的 `execute`、`manager` 属性（间接使用 `get_document`）。它也是 `__all__` 的导出项之一。

### `DocumentGetTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 106 行）

- **作用**：构造函数，只做一件事——把外部可选注入的记忆管理器保存到实例的私有属性 `self._manager` 上。之所以允许注入，是为了让工具在测试或嵌入到已有运行时环境时能复用调用方已经建好的 `MemoryManager`，避免每个工具各自新建一套管理器、连到不同的库上。默认传 `None` 时它不会立刻创建管理器，而是把这个动作推迟到 `manager` 属性第一次被访问时（懒加载），这样仅仅实例化工具对象并不会触发任何 IO 或配置解析，导入与注册阶段的开销因此保持极低。这种「注入优先、懒加载兜底」的写法在工具族里是统一的模式。
- **参数**：
  - `self`：实例自身。
  - `manager: MemoryManager | None`：可选，默认 `None`。传入时应是一个已构造好的 `MemoryManager` 实例；传 `None` 表示「先不指定，等真正用到时再按默认配置构建」。
- **返回**：无返回值（`None`），仅产生副作用——设置 `self._manager`。
- **内部流程**：单条赋值语句 `self._manager = manager`，没有任何校验、日志或分支。
- **异常/边界**：无特殊处理。传入任意对象（哪怕是类型不对的）也不会在这里报错，错误会在后续使用 `self._manager` 时暴露。
- **同文件关系**：只写 `self._manager`，被 `manager` 属性读取；不调用本文件里的任何函数。

### `DocumentGetTool.manager` （`@property`，第 109 行）

- **作用**：这是一个只读属性，作为访问记忆管理器的统一入口，实现了「按需构建、构建一次」的懒加载语义。所有需要管理器的代码（本文件里是 `execute()`）都通过 `self.manager` 取值，而不是直接碰 `self._manager`，这样就把「为空时怎么办」的逻辑集中在一处。懒加载在这里尤其重要，因为构建默认管理器意味着读取配置、打开数据库连接、可能还要初始化向量后端，如果在 `__init__` 里做，会让工具的实例化变得沉重且可能在导入期就触发副作用。属性返回类型标注为 `MemoryManager`（非可选），即保证调用方拿到的永远是一个可用对象。
- **参数**：无显式参数（`self` 由属性机制隐式传入）。
- **返回**：返回 `MemoryManager` 实例。若 `self._manager` 已存在则原样返回该实例（同一实例会被反复复用，不会重复构建）；若为 `None` 则先构建再返回。
- **内部流程**：判断 `if self._manager is None`；成立时执行延迟导入 `from ._memory import build_default_manager`，这一步把导入放在函数内部，避免模块顶层出现循环依赖；随后调用 `build_default_manager()` 并把结果赋给 `self._manager`；最后 `return self._manager`。整个逻辑是典型的「缓存 + 兜底构建」。
- **异常/边界**：若 `build_default_manager()` 因配置缺失、依赖未安装或后端不可用而抛异常，异常会原样向上传播，属性不会被部分赋值（`self._manager` 仍为 `None`，下次访问会再次尝试）。若已注入管理器，则永不触发导入与构建，也不会有任何异常。
- **同文件关系**：它调用了同包 `_memory.build_default_manager`（外部函数，延迟导入）；在本文件内部被 `DocumentGetTool.execute()` 通过 `self.manager` 调用，并读取 `__init__` 写入的 `self._manager`。

### `DocumentGetTool.execute(self, arguments: DocumentGetInput) -> DocumentGetOutput` （第 117 行）

- **作用**：这是工具真正的执行入口，框架在参数校验通过后调用它。它负责把「取仓储 → 兜底检查 → 读文档 → 构造分块模型 → 组装输出 → 按 max_chunks 截断」这一整条链路串起来，是模型层与数据层之间的粘合点。它特别处理了内存模式（`:memory:`）这种没有 `documents/chunks` 真值源的场景：此时 `repository_for` 会返回 `None`，工具不会静默返回空结果，而是抛出 `LookupError` 并给出明确原因，避免 Agent 把「查不到」误判成「文档不存在」。截断逻辑也在这里实现：先用完整分块列表算 `chunk_count`，再对列表做切片，并用长度比较得出 `truncated`，从而保证「总数永远是全量、列表可能被裁剪」这一契约。因为工具声明了 `idempotent=True` 与 `parallel_safe=True`，这个函数被设计成纯读取、不写任何状态。
- **参数**：
  - `self`：实例自身。
  - `arguments: DocumentGetInput`：已经通过 Pydantic 严格校验的输入模型实例，其中 `document_id` 是 1 到 200 字符的非空字符串，`max_chunks` 是 1 到 1000 的整数（默认 50）。
- **返回**：返回一个 `DocumentGetOutput` 实例，包含文档元数据、`raw_text`、完整总数 `chunk_count`、被裁剪后的 `chunks` 列表以及 `truncated` 标记。正常路径下一定返回该模型，不会返回 `None`。
- **内部流程**：第一步 `repository = repository_for(self.manager)`，从记忆管理器里取出文档仓储；第二步判空，`if repository is None` 时抛 `LookupError("当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源")`；第三步 `payload = get_document(repository, arguments.document_id)` 拿到完整详情字典（这一步可能因文档不存在抛 `LookupError`）；第四步用列表推导 `chunks = [DocumentChunk(**chunk) for chunk in payload["chunks"]]` 把每个分块字典转成分块模型，得到**完整**的分块列表；第五步构造 `DocumentGetOutput`，逐字段从 `payload` 取同名值，`tags` 用 `list(payload["tags"])` 拷贝，`chunk_count=len(chunks)` 取完整数量，`chunks=chunks[: arguments.max_chunks]` 做切片截断，`truncated=len(chunks) > arguments.max_chunks` 用布尔比较得出是否截断；最后返回该对象。
- **异常/边界**：仓储为 `None`（内存模式）时抛 `LookupError`；文档 id 不存在时由 `get_document` 抛 `LookupError`，本函数不捕获、直接透传；`payload` 缺少某个期望的键会抛 `KeyError`；分块字典含未声明字段或类型不符时由 `DocumentChunk(**chunk)` 抛 `ValidationError`；`max_chunks` 超出 1 到 1000 的范围在更早的输入校验阶段就被拒绝，这里不会看到非法值。若 `max_chunks` 大于等于分块总数，`chunks` 全量返回且 `truncated` 为 `False`；若分块总数为 0，则 `chunk_count` 为 0、`chunks` 为空列表、`truncated` 为 `False`。超时由 `spec` 的 `timeout_seconds=60.0` 在外层控制，本函数内部没有超时处理。
- **同文件关系**：它调用了本文件的 `manager` 属性（进而可能调用 `_memory.build_default_manager`）、`get_document()` 以及 `DocumentChunk`、`DocumentGetOutput` 两个模型；它被 `create_tool()` 间接使用（工厂创建实例后由框架调用）；它依赖外部函数 `repository_for`（来自 `hybrid_index`）。

### `create_tool() -> BaseTool` （第 142 行）

- **作用**：这是工具工厂函数，供工具注册/发现机制调用，用来创建并返回一个 `DocumentGetTool` 实例。之所以不直接让注册表引用类、而是提供一个无参工厂，是为了把「如何构造这个工具」的知识留在本模块内部：注册器只需要知道「调用 `create_tool()` 就能拿到一个可用的 BaseTool」，将来若构造方式改变（比如需要注入管理器、需要读配置），也只需改这一个函数而不用改注册代码。它返回类型标注为基类 `BaseTool`，对外只暴露工具协议而不暴露具体实现类，从而降低耦合。`TOOL_ENABLED = True` 这个模块级常量与它配套，供加载器判断本工具是否启用。
- **参数**：无参数。
- **返回**：返回 `BaseTool`（实际运行时类型是 `DocumentGetTool`），是一个未指定 `manager` 的全新实例，其管理器会在首次访问 `manager` 属性时按默认配置懒加载构建。
- **内部流程**：函数体只有一行 `return DocumentGetTool()`，使用默认参数构造，因此 `self._manager` 初始为 `None`。
- **异常/边界**：无特殊处理。由于 `__init__` 不做任何校验与 IO，正常情况下不会抛异常；若 `DocumentGetTool` 的类定义本身有误，则在模块导入期就已失败。
- **同文件关系**：它调用了本文件的 `DocumentGetTool` 类；不被本文件里的其它函数调用，而是由外部注册机制调用。它是 `__all__` 的导出项之一。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `get_document(repository, document_id)` | 从文档仓储读取指定文档的元数据、原文与全部分块，文档不存在时抛 `LookupError`。 |
| `DocumentGetInput` | `knowledge.document_get` 的严格输入模型，定义 `document_id`（1–200 字符）与 `max_chunks`（默认 50，范围 1–1000）。 |
| `DocumentChunk` | 单个分块的输出模型，携带 `chunk_id`、序号、字符区间、正文与 `vector_status`。 |
| `DocumentGetOutput` | 工具输出模型，打包文档元数据、原文、全量 `chunk_count`、裁剪后的 `chunks` 与 `truncated` 标记。 |
| `DocumentGetTool` | 继承 `BaseTool` 的只读工具本体，用 `spec` 声明 `knowledge.document_get` 的元信息与使用指引。 |
| `DocumentGetTool.__init__(manager)` | 构造函数，把可选的 `MemoryManager` 存入 `self._manager`，为懒加载做准备。 |
| `DocumentGetTool.manager`（property） | 只读属性，首次访问时按需构建默认记忆管理器并缓存复用。 |
| `DocumentGetTool.execute(arguments)` | 执行入口：取仓储、拒内存模式、读文档、构造分块、按 `max_chunks` 截断并组装输出模型。 |
| `create_tool()` | 工厂函数，无参创建并返回 `DocumentGetTool` 实例供注册机制使用。 |
