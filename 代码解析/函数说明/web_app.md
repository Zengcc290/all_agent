# web/app.py

## 一、这个文件是干什么的

这个文件是整个「知识星云」Web 应用的 HTTP 入口层，也是真实运行入口 `python -m web.app` 最终落到的地方。它用一个应用工厂 `create_app()` 组装出 FastAPI 实例，并在工厂内部把全部 REST 路由注册进去：星云图数据（`/api/graph`、`/api/graph-rag`）、与知识管家对话（`/api/chat`）、文档导入（`/api/ingest`）、手工三元组（`/api/facts`）、一句话入库及其后台任务队列（`/api/knowledge`、`/api/knowledge/jobs`、`/api/knowledge/image`）、种子播种（`/api/seed`）、导出与导入（`/api/export`、`/api/import`）、文档中心与三库对账（`/api/documents*`、`/api/stats`、`/api/reconcile`）、嵌入重建（`/api/embedding/rebuild`）以及健康检查（`/api/health`），最后把 `web/static` 目录挂到根路径 `/` 提供前端静态页。

它本身不实现任何业务算法：向量检索、图检索、抽取、切块、导出导入、对账修复这些真正的逻辑都在 `memory/`、`tool/`、`core` 等模块里，本文件只负责「HTTP 边界」——解析请求体、做参数与权限校验、调用对应的唯一实现、把异常翻译成合适的 HTTP 状态码、维护进程级的星云图缓存与失效。

文件顶部还有一个模块级文档字符串，逐条列出了所有路由与用途，相当于本文件的接口清单。模块级语句 `app = create_app()` 让 `uvicorn web.app:app` 这种写法也能直接拿到实例；文件末尾的 `if __name__ == "__main__"` 分支则在直接运行本模块时调用 uvicorn，监听地址取自 `constants.LOCALHOST` 与 `constants.DEFAULT_WEB_PORT`。

本文件包含 6 个 Pydantic 请求体模型类、1 个模块级辅助异步函数、1 个应用工厂函数，以及工厂内部定义的大量嵌套函数（生命周期钩子、若干内部辅助闭包、以及全部路由处理函数），下文按出现顺序逐个讲解。

## 二、函数与类逐条详解

### `class ReconcileBody(BaseModel)` （第 101 行）

- **作用**：这是 `POST /api/reconcile` 端点的请求体模型，用来描述「三库对账修复」这一动作要修复哪些东西。它的存在是为了让 FastAPI 能自动做 JSON → Python 对象的解析与类型校验，避免在路由函数里手写 `request.json()` 和字段检查。`repair` 字段是一个字符串列表，列表里放的是要执行的修复项名称；当它是空列表（默认值）时，语义是「只做对账报告，不做任何修复」，也就是一次只读检查。它只在 `/api/reconcile` 这一个端点上被用到，由 `reconcile_repair()` 消费后原样转交给 `tool/repair_drift.py` 的 `repair_drift` 函数。
- **参数**：作为 Pydantic 模型，其构造参数只有一个字段 `repair`，类型为 `list[str]`，默认值由 `Field(default_factory=list)` 生成，即默认空列表。没有长度上限、没有元素取值约束，元素语义由下游 `repair_drift` 解释；传入非字符串元素会被 Pydantic 校验拒绝，返回 422。字段的文档字符串说明「`repair` 为空表示只报告不修」。
- **返回**：它不是函数而是类，构造时返回一个 `ReconcileBody` 实例；实例的 `.repair` 属性即上述列表。作为 FastAPI 请求体模型，校验失败时框架会直接返回 422 响应，不会进入路由函数。
- **内部流程**：本身没有方法体逻辑，全部行为来自 Pydantic 的 `BaseModel`：FastAPI 在解析请求时调用它的校验器，把 JSON 中的 `repair` 键映射到字段，缺失时用 `default_factory` 生成空列表，多余字段默认被忽略（Pydantic 默认行为 `ignore`），类型不符则抛校验错误。
- **异常/边界**：Pydantic 校验失败由 FastAPI 转成 422 响应；字段缺失不是异常而是走默认值空列表；没有自定义校验器，因此不做语义合法性检查（例如某个修复项名字是否有效，要等 `repair_drift` 抛 `ValueError`，再由 `reconcile_repair` 映射成 422）。
- **同文件关系**：被本文件的 `reconcile_repair()` 作为参数类型注解使用，除此之外不被其它函数调用。

### `embedding_config_hint(manager: MemoryManager) -> str` （第 107 行）

- **作用**：生成「云端嵌入未配置」时给用户看的配置指引文案，已配置时返回空字符串。历史上这个位置曾经是「网关不可达就返回 503 门禁」的预检逻辑（隧道时代），进入云端时代后端点失败会在真正请求时自然报错，预检与门禁都删掉了，只剩下这段提示文案。它唯一的用途是被 `/api/health` 放进 `degraded.embedding_hint` 字段，向调用方解释「为什么检索退化成了关键词而不是向量」。它不发起任何网络请求，只读一个属性做判断，因此是纯函数式的轻量调用。
- **参数**：`manager`，类型 `MemoryManager`，是本进程的记忆管理器实例（共享单例或测试注入的实例）。函数只用它取嵌入实现，不修改它。没有默认值；传入 `None` 会在 `getattr` 链上安全返回 `None`（因为整条链都用了 `getattr` 带默认值），从而走「未配置」分支。
- **返回**：返回 `str`。当 `manager.embedding.base_url` 存在且为真值（非空字符串）时返回空字符串 `""`，表示配置正常、无需提示；否则返回一段固定中文提示，内容是说明云端嵌入未配置、向量检索退化为关键词、离线兜底向量与云端不兼容不会被静默混用，并指示用户去 `config/services.toml` 的 `[embedding]` 段填写 `base_url` / `api_key` / `model` 后重启服务。
- **内部流程**：第一步用嵌套的 `getattr` 取值：先 `getattr(manager, "embedding", None)` 拿到嵌入对象，再对它取 `base_url`，任一环节为 `None` 都不会抛异常。第二步对结果做真值判断：`if base_url: return ""`。第三步直接返回拼接好的多行字符串字面量（相邻字符串自动拼接）。整个函数没有循环、没有分支嵌套。
- **异常/边界**：无特殊处理。因为全程使用带默认值的 `getattr`，即使 `manager` 为 `None`、没有 `embedding` 属性、或嵌入对象没有 `base_url` 属性，都只会返回提示文案而不会抛 `AttributeError`。它不区分「base_url 配了但 key 错了」这种情况——那种情况由真实请求时报错。
- **同文件关系**：只被本文件的 `health()` 调用；它自己不调用本文件里的任何其它函数。

### `class ConfirmedChatToolCall(BaseModel)` （第 125 行）

- **作用**：描述「人类在上一轮回复中明确接受的那一次破坏性工具调用」，用于聊天端点的二次确认闭环。知识管家在需要删除、清空、入库这类危险操作时，会先返回一个待确认的工具调用提案，前端展示给用户；用户点确认后，前端把这个被接受的调用的工具名和参数原样回传，服务端据此放行，而不是让模型自己「记住」用户同意过。类名与字段的英文注释 `Exact destructive call the human accepted in the previous response` 正说明了「必须精确一致」这一设计意图。它只在 `/api/chat` 的请求体 `ChatBody.confirmation` 字段里出现。
- **参数**：只有一个字段 `tool_name`，类型被 `Literal["memory.manage"]` 收窄，也就是只允许字面量字符串 `"memory.manage"` 这一个取值（其它任何工具名都会被 Pydantic 拒绝）。另一个字段 `arguments`，类型 `dict[str, Any]`，即任意键值对形式的工具参数，没有默认值，属于必填。
- **返回**：作为类，构造返回实例；实例提供 `.tool_name` 与 `.arguments` 两个属性。在路由里通过 `model_dump(mode="json")` 被序列化成普通字典后交给下游。
- **内部流程**：无自定义方法，全部依赖 Pydantic 的字段校验。`Literal` 约束保证只有 `memory.manage` 能通过；`arguments` 被原样保留为字典，服务端不做逐字段白名单校验，一致性校验交给 `chat_confirmed_side_effects` 与 `ExecutionContext`。
- **异常/边界**：工具名不是 `memory.manage`、缺少 `arguments`、或 `arguments` 不是对象，都会由 FastAPI 转成 422。空字典 `{}` 是合法值，是否安全由下游判断。
- **同文件关系**：被 `ChatBody` 作为字段类型引用；在 `chat()` 中通过 `body.confirmation.model_dump(mode="json")` 被使用。

### `class ChatBody(BaseModel)` （第 132 行）

- **作用**：`POST /api/chat` 的请求体模型，承载一次问答所需的全部输入。它把「问什么」「用哪种模式回答」「是否带上了用户确认的危险调用」三件事打包在一起。`message` 的长度上下界由常量控制，防止空消息或超长消息打穿模型上下文；`mode` 用 `Literal` 限定只有离线和联网两种；`confirmation` 可选，用来完成上一轮危险操作的确认。这个模型是聊天链路上第一道也是最外层的输入校验，任何不合法的请求在进入 agent 之前就被挡掉。
- **参数**：`message`，`str`，必填，`min_length=1`、`max_length=WEB_CHAT_MAX_CHARS`（上限来自 `constants`）。`mode`，`Literal["offline", "online"]`，默认 `"offline"`，注释说明 offline 只靠本地记忆、online 额外允许 `web.search` 联网搜索。`confirmation`，类型 `ConfirmedChatToolCall | None`，默认 `None`，为 `None` 表示这是普通的一轮对话而不是对上一轮提案的确认。
- **返回**：构造返回 `ChatBody` 实例，属性为上述三者。校验失败由 FastAPI 返回 422，不会进入 `chat()`。
- **内部流程**：无自定义逻辑。Pydantic 负责长度与枚举校验，`confirmation` 嵌套模型的字段也一并递归校验。
- **异常/边界**：空字符串或超过 `WEB_CHAT_MAX_CHARS` 的消息返回 422；`mode` 传其它值返回 422；`confirmation` 内部字段非法同样返回 422。注意 `message` 只做长度校验，不做空白裁剪——纯空白字符串能通过校验，是否要拒绝由后续流程决定（本文件未对纯空白做特殊处理）。
- **同文件关系**：只被 `chat()` 使用，并通过字段引用 `ConfirmedChatToolCall`。

### `class FactBody(BaseModel)` （第 139 行）

- **作用**：`POST /api/facts` 的请求体模型，用于手工添加一条三元组知识（主语-谓语-宾语），并可附带领域、备注与置信度。它的存在让「人肉录一条事实」这件事有明确的字段契约：主语/谓语/宾语三者必填且有各自长度上限，避免超长字符串直接写进图库；`confidence` 用 0~1 的区间约束保证打分语义一致。字段长度上限全部取自 `constants` 里的 `WEB_FACT_*` 常量，便于统一调参。
- **参数**：`subject`，`str`，必填，`min_length=1`、`max_length=WEB_FACT_SUBJECT_MAX`。`predicate`，`str`，必填，`min_length=1`、`max_length=WEB_FACT_PREDICATE_MAX`。`object`，`str`，必填，`min_length=1`、`max_length=WEB_FACT_OBJECT_MAX`。`domain`，`str | None`，默认 `None`，`max_length=WEB_FACT_DOMAIN_MAX`（为 `None` 时不校验长度）。`note`，`str | None`，默认 `None`，`max_length=WEB_FACT_NOTE_MAX`。`confidence`，`float`，默认 `1.0`，约束 `ge=0, le=1`，即必须落在闭区间 [0,1]。
- **返回**：构造返回 `FactBody` 实例；其字段在 `add_fact()` 中被逐一取出（`body.domain or ""`、`body.note or ""` 把 `None` 归一成空串）后传给工具层。
- **内部流程**：无自定义逻辑，校验全部由 Pydantic 完成。特别注意 `ge`/`le` 对 `confidence` 的数值边界检查，以及 `None` 与缺失的等价处理：字段给了默认 `None`，所以 JSON 里不传这两个键也能通过。
- **异常/边界**：三元组任一为空字符串（`min_length=1`）或超长返回 422；`confidence` 超出 [0,1] 返回 422；`confidence` 传字符串等非数值类型返回 422。`domain`/`note` 传空字符串是合法的（长度 0 满足 `max_length`），最终会被当作空值写入。
- **同文件关系**：只被 `add_fact()` 使用。

### `class GraphRAGBody(BaseModel)` （第 148 行）

- **作用**：`POST /api/graph-rag` 的请求体模型，描述一次「向量证据 + 图关系路径」的混合检索请求。它要回答的问题是：用哪个查询词、召回多少条证据、在图上扩展几跳、以及是否要按某个历史时间点做快照检索。默认值刻意取 `constants` 里的 `RAG_RETRIEVE_LIMIT` 与 `RAG_GRAPH_HOPS`，保证不传参时和聊天链路内部使用的检索口径一致；同时用 `WEB_GRAPH_RAG_LIMIT_MAX` 与 `RAG_GRAPH_MAX_HOPS` 给上限兜底，防止有人传一个巨大的 `limit` 把全库拖出来。
- **参数**：`query`，`str`，必填，`min_length=1`、`max_length=WEB_GRAPH_RAG_QUERY_MAX`。`limit`，`int`，默认 `RAG_RETRIEVE_LIMIT`，约束 `ge=1, le=WEB_GRAPH_RAG_LIMIT_MAX`。`hops`，`int`，默认 `RAG_GRAPH_HOPS`，约束 `ge=0, le=RAG_GRAPH_MAX_HOPS`，`0` 表示不沿图扩展、只做向量证据召回。`at`，`str | None`，默认 `None`，`max_length=80`，语义是历史时刻的 ISO-8601 字符串，为空表示检索当前最新状态。
- **返回**：构造返回 `GraphRAGBody` 实例。字段在 `graph_rag()` 中按名传给 `pipeline.graph_retrieve(query, limit=..., hops=..., at=...)`。
- **内部流程**：无自定义逻辑。`at` 的字符串格式在此处不做校验（只限长度），非法时间格式会在 `graph_retrieve` 内部或图构建时以 `TypeError`/`ValueError` 形式暴露——这一点从同文件 `graph()` 端点对 `TypeError, ValueError` 的处理可以印证。
- **异常/边界**：`query` 为空或超长返回 422；`limit` 小于 1 或超过上限、`hops` 为负或超过最大跳数、`at` 超长均返回 422。`hops=0` 是合法的边界值，表示仅向量检索。
- **同文件关系**：只被 `graph_rag()` 使用。

### `class KnowledgeBody(BaseModel)` （第 155 行）

- **作用**：`POST /api/knowledge` 的请求体模型，也就是「一句话入库」的输入契约。它描述一段要被向量化并自动抽取实体/关系的原文、一个可选的业务发生时间、以及一个控制同步/异步行为的关键开关 `wait`。`wait=True` 是默认值，保持旧的同步语义（提交后阻塞等结果，兼容既有测试与前端旧行为）；`wait=False` 则走后台队列，提交一条任务记录后立即返回，前端随后轮询任务状态。这个开关是同一端点两种工作模式的唯一入口。
- **参数**：`text`，`str`，必填，`min_length=1`、`max_length=WEB_KNOWLEDGE_MAX_CHARS`。`event_at`，`str | None`，默认 `None`，`max_length=80`，语义是这条知识对应的业务时间（不是入库时间），为空表示未指定。`wait`，`bool`，默认 `True`，代码注释明确说明其兼容意图。
- **返回**：构造返回 `KnowledgeBody` 实例。在 `add_knowledge()` 中，`text` 会先被 `.strip()` 再判空，`event_at` 用 `body.event_at or ""` 归一成字符串，`wait` 决定走异步队列还是同步回退分支。
- **内部流程**：无自定义逻辑，只有 Pydantic 的长度与类型校验。
- **异常/边界**：`text` 为空或超长返回 422；`event_at` 超长返回 422；`wait` 传非布尔值返回 422。注意 `min_length=1` 允许纯空白字符串通过模型校验，但 `add_knowledge()` 里额外的 `text.strip()` 判空会把它挡成 422——这是模型层与路由层两道校验配合的例子。
- **同文件关系**：只被 `add_knowledge()` 使用。

### `async def _save_upload(file: UploadFile, *, prefix: str, suffix: str | None = None) -> Path` （第 162 行）

- **作用**：把一次上传流式写入临时文件，同时在写入过程中强制 `MAX_UPLOAD_BYTES` 上限。它被所有上传型端点共用（文档导入、图片入库、JSON 导入），原因是历史实现直接 `await file.read()` 把整个 body 读进内存，单个大文件就能把进程内存打爆，而且当时只有 `/api/ingest` 做了大小限制。现在统一走这个函数：超限的输入在任何解析动作之前就以 413 拒绝，空输入以 400 拒绝。返回的临时文件路径由调用方负责删除，本函数不做清理（失败路径除外）。它是模块级函数而非工厂内闭包，因为不依赖 `app.state`。
- **参数**：`file`，类型 `fastapi.UploadFile`，是 FastAPI 解析出的上传文件对象，提供 `.filename`、`.content_type` 与异步 `.read(size)`。`prefix`，关键字限定参数（`*` 之后的参数必须用关键字传），`str`，是临时文件名前缀，调用方分别传 `"nebula-ingest-"`、`"nebula-image-"`、`"nebula-import-"` 以便排查。`suffix`，关键字限定参数，`str | None`，默认 `None`；为 `None` 时从原文件名后缀推导，原文件名没有后缀则回退 `".txt"`。
- **返回**：返回 `pathlib.Path`，指向已落盘的临时文件。文件内容是上传的原始字节，大小在 1 到 `MAX_UPLOAD_BYTES` 之间（含边界）。调用方拿到后必须自行 `unlink`。
- **内部流程**：先算文件名 `filename = file.filename or "untitled"`（兼容没有文件名的客户端），再决定后缀 `extension`（显式 `suffix` 优先，否则 `Path(filename).suffix or ".txt"`）。接着用 `tempfile.NamedTemporaryFile(delete=False, suffix=extension, prefix=prefix)` 打开临时文件（`delete=False` 是为了关闭后文件仍存在、可被返回并再次打开），进入 `while chunk := await file.read(1024 * 1024)` 循环，每次读 1MB：累加 `total_size`，若超过 `MAX_UPLOAD_BYTES` 则先 `tmp.close()` 再 `os.unlink(tmp.name)` 删掉半成品，然后抛 413，错误文案里把字节上限换算成 MB（`MAX_UPLOAD_BYTES // (1024 * 1024)`）；否则 `tmp.write(chunk)` 落盘。循环结束（读到空字节）后检查 `total_size == 0`，是则同样关闭并删除临时文件，抛 400「上传内容为空」。最后 `return Path(tmp.name)`。
- **异常/边界**：会抛 `fastapi.HTTPException`：413 表示超限（detail 为「文件超过上传上限 NMB」），400 表示空内容。`file.filename` 为 `None` 时用 `"untitled"` 兜底，不会因此崩溃。`Path(filename).suffix` 为空时用 `.txt`。使用 `while ... :=` 海象运算符逐块读取，因此超大文件不会一次性进内存；但极端情况（单次 `read` 返回超过上限的大块）仍会在写盘前被判定并清理。函数本身不捕获 `NamedTemporaryFile`/`os.unlink` 的系统级异常（如磁盘满、权限不足），这类错误会向上冒泡。
- **同文件关系**：被本文件的 `ingest()`、`add_image_knowledge()`、`import_file()` 三个路由调用；它自己不调用本文件里的任何其它函数。

### `def create_app(manager: MemoryManager | None = None) -> FastAPI` （第 198 行）

- **作用**：应用工厂，负责把一个完整的 FastAPI 应用组装出来并返回。它做三件事：注册 `lifespan` 生命周期（启动时准备 manager、RAG 管道、后台入库队列、聊天锁、可选播种，关闭时停队列并释放 manager）、在函数内部定义并注册全部 API 路由闭包、最后把静态前端挂到根路径。参数 `manager` 支持依赖注入：测试时可以传一个用内存库构造的 `MemoryManager`，默认（`None`）则使用 `support.get_manager()` 返回的共享单例。这个「可注入」设计让同一份代码既能跑真实服务也能被测试驱动。
- **参数**：`manager`，类型 `MemoryManager | None`，默认 `None`。为 `None` 时表示「本应用自己拥有 manager」，启动时取共享单例、关闭时调用 `close_manager()` 释放；传入实例时表示「外部拥有 manager」，关闭时不会释放它（由 `lifespan` 内的 `owns_manager` 标志控制）。
- **返回**：返回配置完成的 `fastapi.FastAPI` 实例，标题为「知识星云 · 个人知识库」、版本 `0.1.0`、绑定了 `lifespan`。返回值在模块末尾被赋给全局 `app`。
- **内部流程**：第一步定义 `lifespan` 异步上下文管理器（见下条）。第二步用 `FastAPI(title=..., version=..., lifespan=lifespan)` 创建实例。第三步定义内部闭包 `the_manager()`、`guard_embedding()`、`raise_embedding_http()`、`invalidate_graph()` 以及模块级的 `graph_cache` 字典（含 `external` 与 `payload` 两个键）。第四步用装饰器依次注册所有 `@app.get` / `@app.post` 路由，每个路由函数都作为闭包捕获 `app`、`the_manager`、`invalidate_graph` 等内部符号。第五步判断 `STATIC_DIR.is_dir()`，为真则 `app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")`——注释说明挂载必须放在 API 路由之后，否则会吞掉 `/api/*` 请求。最后 `return app`。
- **异常/边界**：函数体本身基本不做异常处理；`STATIC_DIR` 不存在时静默跳过静态挂载（只有 API 可用），这是刻意的边界处理。`FastAPI(...)` 构造、路由注册、`mount` 都可能因参数非法抛异常（例如目录不存在），但这里已用 `is_dir()` 前置判断规避了静态目录的情况。
- **同文件关系**：它是本文件所有嵌套函数的定义处与调用者（通过路由注册间接触发），并在模块末尾被 `app = create_app()` 调用一次；它内部调用的外部符号包括 `get_manager`、`RAGPipeline`、`IngestJobQueue`、`seed`、`close_manager`、`build_knowledge_extractor` 等（均来自其它模块）。

#### `async def lifespan(app: FastAPI)` （第 201 行，`create_app` 内嵌，带 `@asynccontextmanager`）

- **作用**：FastAPI 的生命周期钩子，在应用启动时完成全部「进程级资源」的准备，在应用关闭时做对称清理。之所以需要它，是因为 manager、RAG 管道、后台队列、聊天锁这些对象要么必须在事件循环已经存在之后才能创建（`asyncio.Lock` 会绑定当前事件循环），要么需要在启动时做一次幂等播种，要么需要在退出时优雅停机。它用 `owns_manager` 区分 manager 的所有权，避免测试注入的 manager 被应用关闭时误释放。
- **参数**：`app`，类型 `FastAPI`，由框架注入的应用实例。函数通过往 `app.state` 上挂属性来把资源暴露给路由闭包（`app.state.manager`、`app.state.pipeline`、`app.state.ingest_queue`、`app.state.chat_lock`）。
- **返回**：作为 `@asynccontextmanager` 装饰的异步生成器，返回异步上下文管理器；`yield` 之前是启动阶段，`yield` 之后是关闭阶段。`yield` 本身不产出值给框架使用。
- **内部流程**：① `owns_manager = manager is None` 记录所有权；② `app.state.manager = manager if manager is not None else get_manager()`，即注入优先、否则取共享单例；③ 构造 `app.state.pipeline = RAGPipeline(app.state.manager, extractor=build_knowledge_extractor())`，把知识抽取器注入管道；④ 构造 `app.state.ingest_queue = IngestJobQueue(app.state.manager, app.state.pipeline.extractor, on_progress=lambda _job_id: invalidate_graph())`，`on_progress` 回调在任务进入终态时让星云图缓存失效，于是前端用 `since` 轮询就能自动看到新图；⑤ `app.state.ingest_queue.start()` 启动队列（注释说明异常退出后由 `start()` 复位续跑，`:memory:` 测试库下 `available=False`）；⑥ `app.state.chat_lock = asyncio.Lock()`，注释解释知识管家是进程级单例、对话历史是共享可变状态，并发问答会互相覆盖，因此串行化（在 lifespan 内创建以保证锁绑定当前事件循环）；⑦ 若 `WEB_AUTOSEED` 为真则调用 `seed(app.state.manager)` 首次启动自动播种，让星云图一打开就有内容；⑧ `yield` 让应用开始对外服务；⑨ 关闭阶段先 `app.state.ingest_queue.shutdown()`，再在 `owns_manager` 为真时 `close_manager()`。
- **异常/边界**：无 try/except 包裹。如果 `get_manager()`、`RAGPipeline(...)`、`IngestJobQueue(...)`、`start()` 或 `seed()` 抛异常，启动会失败并阻止应用对外服务——这是刻意的「启动即暴露问题」策略。`on_progress` 回调使用 lambda 忽略 `job_id`，只做缓存失效。
- **同文件关系**：调用本文件内嵌定义的 `invalidate_graph()`（通过 lambda 延迟调用）；其创建的对象 `app.state.pipeline`、`app.state.ingest_queue`、`app.state.chat_lock`、`app.state.manager` 被本文件几乎所有路由函数读取。被 `create_app()` 在构造 `FastAPI` 时传入。

#### `def the_manager() -> MemoryManager` （第 232 行，`create_app` 内嵌）

- **作用**：极简的取值辅助闭包，用来从 `app.state` 上取出当前进程的记忆管理器。之所以不直接写 `app.state.manager` 而包一层函数，是为了让路由与其它内部闭包有一个统一的访问点，也让 `guard_embedding`、`raise_embedding_http`、`health` 等函数体更短、更一致。它假设 `lifespan` 已经执行过，因此不做存在性检查。
- **参数**：无参数。
- **返回**：返回 `MemoryManager`，即 `app.state.manager`。若 `lifespan` 尚未运行（例如直接在未启动的应用上调用路由），会因为 `app.state` 上还没有 `manager` 属性而抛 `AttributeError`。
- **内部流程**：只有一行 `return app.state.manager`。
- **异常/边界**：无特殊处理；未启动场景会抛 `AttributeError`，但正常请求路径下不会出现。
- **同文件关系**：被 `guard_embedding()`、`raise_embedding_http()`、`graph()`、`ingest()`、`add_fact()`、`add_knowledge()`、`revectorize_document()`、`stats()`、`reconcile()`、`reconcile_repair()`、`rebuild_embedding()`、`health()`、`export()`、`import_file()` 等大量函数调用。

#### `def guard_embedding(*, confirm_rebuild: bool = False) -> None` （第 235 行，`create_app` 内嵌）

- **作用**：在所有写向量投影的操作（文档导入、一句话入库、图片入库、重试入库、重建向量、文档重嵌入）之前执行的「嵌入锁闸门」。它的职责是保证当前使用的嵌入实现与已锁定的嵌入配置、以及 Qdrant 中现有向量维度一致；如果 SQLite 里记录的锁与运行中的嵌入不匹配，或者线上 Qdrant 的维度与当前嵌入不匹配，就拒绝这次写操作，避免把两套不兼容的向量混进同一个集合。用户可以通过 `confirm_rebuild=true` 明确表示「我知道会重建，继续」。
- **参数**：`confirm_rebuild`，`bool`，关键字限定参数，默认 `False`。为 `False` 时一旦发现不匹配就报 409 要求确认；为 `True` 时把确认意图透传给 `apply_embedding_lock`，允许它执行重建/改写锁的动作。
- **返回**：返回 `None`。成功即代表闸门放行，调用方继续后续写入。失败不返回而是抛异常。
- **内部流程**：① `manager = the_manager()` 取管理器；② `repository = app.state.pipeline.document_repo()` 取文档仓储（作为锁信息的持久化位置）；③ 在 `try` 中调用 `apply_embedding_lock(manager, repository, confirm_rebuild=confirm_rebuild)`；④ 捕获 `EmbeddingLockMismatch`，用 `exc.to_detail()` 作为 detail 抛 `HTTPException(409)`，并用 `from exc` 保留原始异常链。
- **异常/边界**：抛 `fastapi.HTTPException(409)`，detail 是嵌入锁不匹配的结构化详情；原始异常通过 `raise ... from exc` 保留。除 `EmbeddingLockMismatch` 之外的异常（例如仓储不可用）不在此处捕获，会继续向上冒泡。
- **同文件关系**：调用 `the_manager()`；被 `ingest()`、`add_knowledge()`、`retry_knowledge_job()`、`add_image_knowledge()`、`rebuild_embedding()` 调用。

#### `def raise_embedding_http(exc: BaseException) -> None` （第 245 行，`create_app` 内嵌）

- **作用**：异常翻译辅助函数。当某个写操作抛出任意异常时，先用 `mismatch_from_exception` 判断这个异常本质上是不是「嵌入锁不匹配」造成的（例如底层库抛出的维度错误），如果是，就把它翻译成带结构化详情的 409 响应；如果不是，就什么都不做、静默返回，让调用方继续走自己原本的错误处理分支（通常是抛 422 或 502）。这样做的价值是把「同一个不匹配问题可能以多种底层异常形式出现」这件事收敛成统一的可读 HTTP 错误。
- **参数**：`exc`，类型 `BaseException`，是刚刚被捕获到的原始异常对象，用来做特征识别。它只被读取，不被修改或重新抛出。
- **返回**：返回 `None`——当 `mismatch_from_exception` 判定「不是锁不匹配」时返回 `None`，控制权回到调用方。若判定是锁不匹配，则不返回，直接抛 `HTTPException(409)`。
- **内部流程**：① 调用 `mismatch_from_exception(exc, the_manager(), app.state.pipeline.document_repo())`，把异常、当前 manager 与仓储一起交给它做映射，得到 `mapped`；② `if mapped is not None:` 成立时用 `mapped.to_detail()` 抛 `HTTPException(409)` 并以 `from exc` 链接原异常；③ 否则函数自然结束、返回 `None`。
- **异常/边界**：抛 `HTTPException(409)`（条件性）。`mapped is None` 时无副作用、无返回值。注意它是「可能抛异常也可能静默返回」的函数，调用方必须在它之后继续写自己的兜底 `raise`，本文件中的 `ingest()`、`add_image_knowledge()`、`revectorize_document()` 都是这样用的。
- **同文件关系**：调用 `the_manager()`；被 `ingest()`、`add_image_knowledge()`、`revectorize_document()` 调用。

#### `def invalidate_graph() -> None` （第 263 行，`create_app` 内嵌）

- **作用**：星云图缓存的失效函数，任何会改变图内容的写操作完成后都要调用它。它做两件事：递增进程级的图修订号（`support.GRAPH_REVISION`，通过 `bump_graph_revision()`），以及把本地缓存的 payload 置为 `None`。之所以要同时动这两处，是因为本地写入与后台问答抽取线程共用同一个进程级计数：如果后台线程只递增计数而不清本地缓存，或本地只清缓存而不递增计数，`/api/graph?since=` 就会出现「只有一个真相来源」被破坏的情况，典型症状是刚通过 API 写完却被 `since` 判成「无变化」。
- **参数**：无参数。
- **返回**：返回 `None`。
- **内部流程**：① 调用 `bump_graph_revision()` 递增全局修订号；② `graph_cache["payload"] = None` 让下一次 `/api/graph` 强制重建。注意它不修改 `graph_cache["external"]`，那个字段由 `graph()` 在读请求时与最新修订号比对并更新。
- **异常/边界**：无特殊处理；若 `bump_graph_revision` 内部出错会直接冒泡。
- **同文件关系**：被 `lifespan()`（作为 `IngestJobQueue` 的 `on_progress` 回调）、`graph()` 无关但被 `chat()`、`ingest()`、`add_fact()`、`add_knowledge()`、`add_image_knowledge()`、`reseed()`、`import_file()`、`rebuild_embedding()` 调用。

#### `def graph(since: int = -1, at: str | None = None) -> dict[str, Any]` （第 273 行，路由 `GET /api/graph`）

- **作用**：星云图的主数据端点，返回全图的 `nodes` + `edges`（外加 `stats`），是前端星云图的数据源。它额外承担两个优化职责：一是进程内缓存（按「修订号 + as-of 时间」作为缓存键），避免每次打开或刷新都做一遍 `build_graph` 的全库 O(N) 遍历；二是「无变化探测」——前端带上上次的 `since` 修订号，若与当前修订号相同且没有指定历史时间，就只回一个 `unchanged: True` 的空图，省掉整包传输。注释还说明写操作会递增 `GRAPH_REVISION`，因此连 Neo4j Aura 侧来自本应用的更新也能让缓存失效。
- **参数**：`since`，`int`，查询参数，默认 `-1`；`-1` 表示「我没有已知修订号，请给全量」，`>= 0` 时表示客户端上次看到的修订号，与当前修订号相等则命中「无变化」分支。`at`，`str | None`，查询参数，默认 `None`，语义是历史时刻的 ISO-8601 字符串，用于取某个时间点的图快照；为 `None`/空串时表示取当前最新图，且只有这种情况下才允许走「无变化」短路。
- **返回**：返回 `dict[str, Any]`。三种形态：① 无变化时返回 `{"revision": revision, "unchanged": True, "nodes": [], "edges": [], "stats": payload.get("stats", {})}`——注意 `stats` 仍来自已构建的 payload；② 正常时返回 `{**payload, "revision": revision, "unchanged": False}`，即在图数据上叠加修订号与标志位；③ 时间参数非法时抛 422。
- **内部流程**：① `revision = graph_revision()` 取当前修订号；② `cache_at = at or ""` 把 `None` 归一成空串，保证缓存键可比较；③ 在 `try` 中判断三个条件之一成立就重建：`graph_cache["payload"] is None`（缓存被写操作清空）、`graph_cache["external"] != revision`（外部/后台修订号变了）、`graph_cache.get("at") != cache_at`（请求的历史时刻变了）；重建时调用 `build_graph(the_manager(), at=at)`，并回填 `payload`、`external`、`at` 三个键；④ 把结果赋给 `payload`；⑤ 若 `build_graph` 抛 `TypeError` 或 `ValueError`，捕获后抛 `HTTPException(422, detail=f"时间参数无效：{exc}")`——把非法时间字符串翻译成 422 而不是 500；⑥ 判断 `not cache_at and since >= 0 and since == revision`，三个条件同时成立才返回「无变化」的轻量响应；⑦ 否则返回完整 payload 加修订号与 `unchanged: False`。
- **异常/边界**：抛 `HTTPException(422)`，仅在 `build_graph` 抛 `TypeError`/`ValueError` 时（典型来源是 `at` 格式非法）。`since` 为负数或与当前修订号不等都走全量返回。`at` 为空串与 `None` 等价（都让 `cache_at` 为空串）。缓存字典的 `at` 键在首次访问时用 `.get("at")` 读取，因此不会因缺少键而报 `KeyError`。
- **同文件关系**：调用 `the_manager()` 与外部 `build_graph`、`graph_revision`；读取并写入本文件 `create_app()` 中定义的闭包变量 `graph_cache`；与 `invalidate_graph()` 构成「写失效 / 读重建」的配对关系。

#### `def graph_rag(body: GraphRAGBody) -> dict[str, Any]` （第 305 行，路由 `POST /api/graph-rag`）

- **作用**：图 RAG 检索端点，把一次「向量证据召回 + 图关系路径扩展」的混合检索结果直接暴露给外部调用方（前端「依据/路径」面板、调试工具或其它 Agent 都能用）。它把检索逻辑完全委托给 `RAGPipeline.graph_retrieve`，自己只做请求体到调用参数的映射，以及把结果对象同时序列化成结构化数据与一段可读上下文文本。没有额外的鉴权或限流逻辑，参数边界由 `GraphRAGBody` 负责。
- **参数**：`body`，类型 `GraphRAGBody`，由 FastAPI 从 JSON 请求体解析。其字段 `query`（检索词）、`limit`（召回条数）、`hops`（图上扩展跳数）、`at`（历史时刻）分别被透传。
- **返回**：返回 `dict[str, Any]`，内容是 `result.to_dict()` 与 `{"context": result.build_context()}` 的字典合并（`|` 运算符，右侧键覆盖左侧同名键）。即既包含结构化检索结果（证据、路径等），也包含一段拼装好的上下文文本，方便直接喂给模型。
- **内部流程**：唯一一步是调用 `app.state.pipeline.graph_retrieve(body.query, limit=body.limit, hops=body.hops, at=body.at)` 得到 `result`，然后 `return result.to_dict() | {"context": result.build_context()}`。
- **异常/边界**：本函数不做任何异常捕获。`graph_retrieve` 内部抛出的异常（例如 `at` 时间非法）会向上冒泡为 500；这一点与 `graph()` 端点显式把 `TypeError`/`ValueError` 映射成 422 的做法不同——本端点没有这层映射。`app.state.pipeline` 不存在（未启动）会抛 `AttributeError`。
- **同文件关系**：读取 `create_app()` 内 `lifespan()` 设置的 `app.state.pipeline`；不调用本文件里的其它函数（`to_dict`/`build_context` 属于外部 `GraphRAGResult` 的方法）。

#### `async def chat(body: ChatBody) -> dict[str, Any]` （第 318 行，路由 `POST /api/chat`）

- **作用**：与知识管家对话的主端点，是整个应用里逻辑最重的一个路由。它负责：检查聊天模型是否就绪（未配置直接 503）、按请求模式决定是否开放联网搜索工具、把共享单例 agent 的同步阻塞调用丢到线程池并加锁串行化、处理「危险工具调用的二次确认」闭环、在完成一轮问答后写入 episodic 记忆并调度后台图抽取、最后再补一份图检索证据与混合召回明细返回给前端做「依据」展示。可以说它把一次问答涉及的模型调用、记忆留痕、图更新与溯源报告全部串起来了。
- **参数**：`body`，类型 `ChatBody`，从 JSON 请求体解析，含 `message`（用户消息）、`mode`（`"offline"` 或 `"online"`）、`confirmation`（可选，上一轮被人类接受的破坏性调用）。
- **返回**：返回 `dict[str, Any]`，两种形态：① 当 agent 返回了待确认项时，返回 `{"answer", "mode", "sources": [], "paths": [], "retrieval": {"note": "", "hits": []}, "confirmations": confirmations}`——这是「确认提案」响应，刻意不写 QA 记录、不跑抽取，等用户接受精确调用后才继续；② 正常完成时返回 `{"answer", "mode", "sources", "paths", "retrieval", "confirmations": []}`，其中 `sources` 是图检索证据（含 `memory_id`、`score`、来源文件名、块 id），`paths` 是关系路径的 `to_dict()` 列表，`retrieval` 是混合召回的明细报告。失败时抛 503 或 502。
- **内部流程**：① `ready, reason = chat_ready()`，`not ready` 就抛 503（detail 为 `reason`）；② `agent = get_agent()` 取共享 agent；③ `online = body.mode == "online"`；④ `tool_names = chat_tool_names(agent, online=online)` 取本次允许的工具名，注释说明联网模式只有在 AnySearch 已配置时才真正开放 `web.search`；⑤ `effective_mode = "online" if SEARCH_TOOL_NAME in tool_names else "offline"`，把「请求的联网」修正为「实际生效的模式」；⑥ 构造确认上下文：`body.confirmation is None` 时用 `chat_confirmed_side_effects(agent)`（只为 `memory.add` 预置写确认，即用户这一轮明确要求「记住」时模型才能落库），否则用 `chat_confirmed_side_effects(agent, destructive_call=body.confirmation.model_dump(mode="json"))` 把用户接受的那次精确调用带上；⑦ `context = ExecutionContext(confirmed_side_effects=confirmations)`；⑧ 在 `async with app.state.chat_lock:` 中 `await asyncio.to_thread(agent.run, body.message, tool_names=tool_names, context=context)`——`agent.run` 是同步阻塞调用，用 `to_thread` 避免卡住事件循环，用锁串行化避免并发问答互相污染共享历史（后到请求排队而不是并发改写）；⑨ 整个 try 块捕获任意 `Exception` 并归一为 `HTTPException(502, detail=f"聊天模型调用失败：{type(exc).__name__}: {exc}")`，注释注明这是刻意把任意上游失败归一为可读错误；⑩ `confirmations = list(getattr(agent, "pending_confirmations", []))`，若非空则直接返回确认提案响应（不写 QA、不抽取）；⑪ 否则 `record_qa(the_manager(), body.message, answer, mode=effective_mode)` 写入问答留痕；⑫ `schedule_qa_extraction(body.message, answer, manager=the_manager())` 把问答交给 LLM 转成图补丁并在后台执行（抽取是第二次模型往返，不让用户多等）；⑬ `invalidate_graph()` 让本地缓存立即失效；⑭ 在 `try` 中调 `app.state.pipeline.graph_retrieve(body.message, limit=RAG_RETRIEVE_LIMIT, hops=RAG_GRAPH_HOPS)` 取图证据，若抛异常则局部 `from memory.rag.graph_rag import GraphRAGResult` 并构造空的 `GraphRAGResult(query=body.message)` 兜底——注释强调图检索是「答后补充」，失败不能让聊天从 200 变成错误；⑮ `hybrid = hybrid_recall(app.state.pipeline, body.message, limit=RAG_RETRIEVE_LIMIT)` 做同一句问话的混合召回（向量 × FTS5 RRF），注释说明网关不可用时这里退化为纯关键词，正好让降级原因对用户可见；⑯ 组装 `retrieval_report`，对 `hybrid.chunks` 逐条产出字典（`chunk_id`、`document_id`、`chunk_index`、截断到 200 字的 `snippet`、`score`、以及 detail 里的 `rrf_score`/`vector_score`/`keyword_score`）；⑰ 组装并返回最终响应，其中 `sources` 是对 `retrieval.evidence` 的列表推导，`source` 字段优先取元数据里的 `filename`、其次 `source`，`chunk_id` 优先取元数据 `chunk_id`、其次 `document_id`。
- **异常/边界**：① 聊天模型未就绪抛 503；② `agent.run` 及确认上下文构造阶段的任意异常统一抛 502（`noqa: BLE001` 标注这是有意的宽泛捕获）；③ 图检索失败被吞掉并用空结果兜底（第二个 `noqa: BLE001`），保证聊天仍然 200；④ `getattr(agent, "pending_confirmations", [])` 用默认值兼容没有该属性的 agent；⑤ 混合召回的失败没有单独捕获，若它抛异常会向上冒泡（注释只说明了它会退化为关键词而不是抛错）；⑥ `snippet` 用 `(hit.content or "")[:200]` 对 `None` 内容做兜底。确认提案路径下刻意不写记忆、不跑抽取，避免「用户还没同意就落库」。
- **同文件关系**：调用本文件的 `invalidate_graph()`；读取 `lifespan()` 设置的 `app.state.chat_lock` 与 `app.state.pipeline`；大量依赖外部符号 `chat_ready`、`get_agent`、`chat_tool_names`、`SEARCH_TOOL_NAME`、`chat_confirmed_side_effects`、`ExecutionContext`、`record_qa`、`schedule_qa_extraction`、`hybrid_recall`、`RAG_RETRIEVE_LIMIT`、`RAG_GRAPH_HOPS`。

#### `async def ingest(file: UploadFile, confirm_rebuild: bool = Query(default=False)) -> dict[str, Any]` （第 425 行，路由 `POST /api/ingest`）

- **作用**：文档导入端点，把用户上传的文档落盘成临时文件后交给 RAG 管道切块入库，让星云图长出新星星。它完整演示了本文件对上传型端点的标准套路：先流式落盘（带大小上限）、再做嵌入锁闸门、再调业务实现、再把解析类异常翻译成 422、最后无论成败都删临时文件。导入成功后还会往 episodic 记忆里写一条「上传并导入了文档《X》（N 个知识块）」的记录，并让图缓存失效。
- **参数**：`file`，`UploadFile`，表单上传的文件，文件名缺失时用 `"untitled"` 兜底。`confirm_rebuild`，`bool`，查询参数，默认 `False`（`Query(default=False)`），语义是「确认在嵌入不匹配时重建向量投影」，会透传给 `guard_embedding`。
- **返回**：返回 `dict[str, Any]`，含三个键：`filename`（原文件名）、`chunks`（切出的知识块数量，即 `len(items)`）、`extraction`（`dict(app.state.pipeline.last_ingest_report)`，即本次导入的抽取报告副本）。
- **内部流程**：① `filename = file.filename or "untitled"`；② `tmp_path = await _save_upload(file, prefix="nebula-ingest-")`，这一步可能抛 413/400；③ `try` 块内先 `guard_embedding(confirm_rebuild=confirm_rebuild)`，再 `items = app.state.pipeline.ingest_source(tmp_path, metadata={"source": filename, "filename": filename}, chunk_size=WEB_INGEST_CHUNK_SIZE, overlap=RAG_CHUNK_OVERLAP)`——注意文档导入用的是 `WEB_INGEST_CHUNK_SIZE` 而不是 `RAG_CHUNK_SIZE`；④ `except HTTPException: raise` 原样放行闸门抛出的 409；⑤ `except Exception as exc:` 先调 `raise_embedding_http(exc)` 尝试把它识别成嵌入锁不匹配并抛 409，若不是则继续抛 `HTTPException(422, detail=f"文档解析失败：{type(exc).__name__}: {exc}")`；⑥ `finally: tmp_path.unlink(missing_ok=True)` 保证临时文件总被清理（`missing_ok=True` 容忍已被删的情况）；⑦ `the_manager().episodic.record(f"上传并导入了文档《{filename}》（{len(items)} 个知识块）", metadata={"title": "导入文档", "filename": filename})`；⑧ `invalidate_graph()`；⑨ 返回结果字典。
- **异常/边界**：可能抛 413（文件超限）、400（空文件，来自 `_save_upload`）、409（嵌入锁不匹配，来自 `guard_embedding` 或 `raise_embedding_http`）、422（解析失败，附带异常类型名与消息）。临时文件在 `finally` 中无条件清理，即使中途抛异常也不会残留。`items` 为空（文档解析出 0 块）不会报错，会照常记录一条「0 个知识块」的记忆并返回 `chunks: 0`。
- **同文件关系**：调用 `_save_upload()`、`guard_embedding()`、`raise_embedding_http()`、`the_manager()`、`invalidate_graph()`；被 FastAPI 在 `POST /api/ingest` 上调用。

#### `def add_fact(body: FactBody) -> dict[str, Any]` （第 463 行，路由 `POST /api/facts`）

- **作用**：手工添加一条三元组知识的端点，给用户一个「不经模型、直接写图」的入口（例如录入一条确定性很高的常识）。它把实际写库动作完全委托给 `tool/add_fact.py` 的 `add_fact`（即工具名 `knowledge.add_fact`），保证 Web 端点和 Agent 工具走的是同一份实现，行为不会分叉。写入后让图缓存失效，并把新条目的 id 返回给调用方。
- **参数**：`body`，类型 `FactBody`，从 JSON 请求体解析，含主语、谓语、宾语、可选的领域与备注、以及 0~1 的置信度。函数内部对可选字段做 `or ""` 归一，把 `None` 变成空字符串再交给工具层。
- **返回**：返回 `dict[str, Any]`，固定为 `{"ok": True, "item_id": item.id}`，`item_id` 是新建事实条目的标识。
- **内部流程**：① `item = write_fact(the_manager(), subject=body.subject, predicate=body.predicate, object=body.object, domain=body.domain or "", note=body.note or "", confidence=body.confidence)`，六个参数全部以关键字传递；② `invalidate_graph()` 让星云图下次请求重建；③ 返回成功字典。
- **异常/边界**：本函数不捕获异常，工具层的失败会直接冒泡为 500。字段长度与取值边界在 `FactBody` 层已校验（422）。`domain`/`note` 为 `None` 或空串都会以空字符串写入，不做额外区分。
- **同文件关系**：调用 `the_manager()` 与 `invalidate_graph()`；外部调用 `tool.add_fact.add_fact`（导入时别名为 `write_fact`）。

#### `def add_knowledge(body: KnowledgeBody, confirm_rebuild: bool = Query(default=False)) -> dict[str, Any]` （第 483 行，路由 `POST /api/knowledge`）

- **作用**：「一句话入库」端点：把一段原文向量化，同时让 LLM 自动抽取实体与关系写进图结构。它有三条执行路径——后台队列的「立即返回」模式、后台队列的「同步等待」模式、以及没有持久化 SQLite 时的同步回退模式。之所以设计成三种，是因为正常部署下（真实文件库）应该用持久化队列以便查看历史与重启续跑，而 `:memory:` 测试库没有 `ingest_jobs` 表，必须保留旧的同步行为才能兼容测试。它是把「异步任务队列」这个较重的机制引入 HTTP 层的核心位置。
- **参数**：`body`，类型 `KnowledgeBody`，含 `text`（要入库的原文）、`event_at`（可选业务时间）、`wait`（默认 `True`，为 `False` 时提交后台队列后立即返回）。`confirm_rebuild`，`bool`，查询参数，默认 `False`，透传给嵌入锁闸门。
- **返回**：返回 `dict[str, Any]`，按路径不同而形态不同：① 异步提交返回 `{"ok": True, "job_id": ..., "status": job.status, "label": "排队中", "async": True}`；② 同步等待成功返回 `{"ok": True, "job_id": finished.job_id, "chunks": result.get("chunks", 0), "items": [], "extraction": report}`；③ 无队列回退返回 `{"ok": True, "chunks": len(items), "items": [item.to_dict() for item in items], "extraction": report}`。失败时抛 422/409/500。
- **内部流程**：① `text = body.text.strip()` 并 `if not text: raise HTTPException(422, detail="一句话内容不能为空")`——模型层只保证长度 ≥1，这里额外挡掉纯空白；② `guard_embedding(confirm_rebuild=confirm_rebuild)` 做嵌入锁闸门；③ `queue = getattr(app.state, "ingest_queue", None)` 用 `getattr` 容忍属性缺失；④ 若 `queue is not None and queue.available`：调 `job = queue.submit(text, event_at=body.event_at or "")`；若 `not body.wait` 立即返回「排队中」响应；否则 `finished = queue.wait(job.job_id)` 阻塞等终态，若 `finished is None or finished.status != "done"` 则用 `finished.error`（或「等待入库超时」）拼出 500「入库失败：...」；接着在 `try` 中 `result = json.loads(finished.result) if finished.result else {}`，`except ValueError` 时抛 500「入库结果解析失败：...」并 `from exc`（注释说明后台任务写入的 `result` 必须是合法 JSON，对端损坏时给可读错误而不是让解析异常变成 500 内部错误）；然后 `report = result.get("report") if isinstance(result.get("report"), dict) else {}` 做类型兜底；最后返回同步成功响应（注意此路径 `items` 固定为空列表，因为结果以任务记录形式返回）。⑤ 队列不可用时走同步回退：`pipeline = app.state.pipeline`；函数内局部 `from memory.rag import Document`（延迟导入避免循环依赖）；`preview = text.splitlines()[0][:40]` 取首行前 40 字作为展示用文件名；调 `pipeline.ingest(Document(text, metadata={"source": "一句话入库", "filename": preview, "note": text[:400], "event_at": body.event_at or "", "reference_time": datetime.now(UTC).isoformat()}), chunk_size=RAG_CHUNK_SIZE, overlap=RAG_CHUNK_OVERLAP)`——注意回退路径用的是 `RAG_CHUNK_SIZE` 而非 `WEB_INGEST_CHUNK_SIZE`；`report = pipeline.last_ingest_report`；`the_manager().episodic.record(f"添加了一条知识：{text[:80]}", metadata={"title": "一句话入库", "source": "一句话入库"})`；`invalidate_graph()`；返回含 `items` 的响应。
- **异常/边界**：422（空白文本）、409（嵌入锁不匹配，来自闸门）、500（入库失败或等待超时、结果 JSON 损坏）。`body.event_at` 为 `None` 时用 `""`；`finished.result` 为空时把 `result` 当空字典处理，于是 `chunks` 取默认 0、`extraction` 取空字典。`text.splitlines()[0]` 在 `text` 非空时安全（前面已判空），`text[:400]` 与 `text[:80]` 是简单截断。`getattr(app.state, "ingest_queue", None)` 的写法让本函数在队列未初始化时也能安全回退。
- **同文件关系**：调用 `guard_embedding()`、`the_manager()`、`invalidate_graph()`；读取 `lifespan()` 建立的 `app.state.pipeline` 与 `app.state.ingest_queue`；外部依赖 `json.loads`、`datetime.now(UTC)`、`RAGPipeline.ingest`、`Document`。

#### `def list_knowledge_jobs(status: str | None = None, limit: int = 20) -> dict[str, Any]` （第 560 行，路由 `GET /api/knowledge/jobs`）

- **作用**：一句话入库的历史/状态查询端点，返回每条任务的持久化状态（排队中/正在入库/成功/失败），供前端轮询展示进度。它同时承担「队列是否可用」的探测职责：在 `:memory:` 之类的无持久化场景下不报错，而是返回 `available: False` 与空列表，让前端知道该功能不可用而不是看到一个错误。它还对 `limit` 做了 API 层夹紧，避免超大的步进直接打到 SQL 层。
- **参数**：`status`，`str | None`，查询参数，默认 `None`，用于按状态过滤；取值必须落在 `pending`/`running`/`done`/`failed` 四个之一，否则抛 422。`limit`，`int`，查询参数，默认 `20`，实际生效值会被夹在 1 到 200 之间。
- **返回**：返回 `dict[str, Any]`。队列不可用时返回 `{"items": [], "available": False}`；可用时返回 `{"items": [job_to_dict(job) for job in jobs], "available": True}`，其中每个元素是任务的字典化表示。
- **内部流程**：① `queue = getattr(app.state, "ingest_queue", None)`；② `if queue is None or not queue.available:` 则直接返回不可用响应；③ 校验 `status`：非 `None` 且不在四值元组内就抛 422，detail 明确列出允许的取值；④ `jobs = queue.list(status=status, limit=max(1, min(int(limit), 200)))`——`int(limit)` 是防御性转换，`min(..., 200)` 是硬上限，`max(1, ...)` 保证至少 1，注释说明仓储层内部已有 200 条硬上限，这里是 API 层再夹一层并让前端拿到的 `limit` 就是实际生效值；⑤ 返回带 `job_to_dict` 映射结果的响应。
- **异常/边界**：抛 `HTTPException(422)` 当 `status` 取值非法。`limit` 为 0、负数或极大值都被夹紧到 [1, 200]，不报错。队列缺失或不可用时静默返回 `available: False`，不抛异常。若 `limit` 传入无法转成整数的字符串，`int(limit)` 会抛 `ValueError`——本函数未捕获它，实际会被 FastAPI 当作 500 处理（这是本端点的一个隐含边界）。
- **同文件关系**：读取 `lifespan()` 建立的 `app.state.ingest_queue`；使用外部函数 `job_to_dict`（从 `.ingest_queue` 导入）；不调用本文件里的其它函数。

#### `def retry_knowledge_job(job_id: str, confirm_rebuild: bool = Query(default=False)) -> dict[str, Any]` （第 577 行，路由 `POST /api/knowledge/jobs/{job_id}/retry`）

- **作用**：把一条失败的一句话入库任务重新入队的端点。它先过嵌入锁闸门（注释说明是「after the embedding gate」，因为重试往往发生在用户刚修好嵌入配置之后），再检查队列可用性，最后调用队列的 `retry`。它把队列层的三种失败原因精确映射成不同的 HTTP 状态码，让前端能区分「任务不存在」「状态不允许重试」「功能不可用」。
- **参数**：`job_id`，`str`，路径参数，来自 URL 的 `{job_id}` 段，指定要重试的任务。`confirm_rebuild`，`bool`，查询参数，默认 `False`，透传给 `guard_embedding`。
- **返回**：返回 `dict[str, Any]`，形式为 `{"ok": True, "async": True, **job_to_dict(job)}`——在任务字典上叠加成功标志与异步标志（若任务字典里恰好有同名键会被覆盖，这里的顺序保证了 `ok`/`async` 优先）。
- **内部流程**：① `guard_embedding(confirm_rebuild=confirm_rebuild)`；② `queue = getattr(app.state, "ingest_queue", None)`，若 `queue is None or not queue.available` 则抛 503「入库队列不可用」；③ 在 `try` 中 `job = queue.retry(job_id)`；④ `except LookupError` 抛 404「入库任务不存在」，用 `from None` 抑制原始异常链（避免把内部查找细节暴露给客户端）；⑤ `except ValueError as exc` 抛 400 并把异常消息作为 detail（`from exc` 保留链路），典型场景是任务当前状态不允许重试；⑥ 返回合并后的成功响应。
- **异常/边界**：409（闸门不匹配）、503（队列不可用）、404（任务不存在）、400（状态不允许重试）。`from None` 的使用是一个刻意的边界处理：404 时不暴露 `LookupError` 的原始堆栈上下文。
- **同文件关系**：调用 `guard_embedding()`；读取 `lifespan()` 建立的 `app.state.ingest_queue`；使用外部 `job_to_dict`；不调用本文件里的其它函数。

#### `async def add_image_knowledge(file: UploadFile, text: str = Form(default=""), captured_at: str = Form(default=""), confirm_rebuild: bool = Query(default=False)) -> dict[str, Any]` （第 596 行，路由 `POST /api/knowledge/image`）

- **作用**：图片/相机观测入库端点，接收一张图片（可附带一段文字说明与拍摄时间），做视觉 embedding 并调用视觉模型抽取实体、时间与多元关系，最终写进图结构。它比文本入库多了几道输入检查：MIME 类型必须是 `image/*`（否则 415）、文字说明不能超过 `WEB_KNOWLEDGE_MAX_CHARS`（否则 422）。它同时演示了「表单字段 + 文件 + 查询参数」混合的 FastAPI 参数写法。
- **参数**：`file`，`UploadFile`，表单里的图片文件，文件名缺失时用 `"camera.jpg"` 兜底。`text`，`str`，表单字段（`Form(default="")`），默认空串，是给视觉模型/检索用的图片说明。`captured_at`，`str`，表单字段，默认空串，语义是拍摄时间（由客户端提供，服务端只做 `strip()`）。`confirm_rebuild`，`bool`，查询参数，默认 `False`，透传给闸门。
- **返回**：返回 `dict[str, Any]`，形式为 `{"ok": True, **result}`，`result` 是 `tool/ingest_image.py` 的 `ingest_image` 返回的字典（包含入库与抽取的结果信息）。
- **内部流程**：① `filename = file.filename or "camera.jpg"`；② `mime_type = (file.content_type or "").split(";", 1)[0].strip().lower()`——先兜底空串、再剥掉形如 `; charset=utf-8` 的参数、去空白、转小写；③ `if not mime_type.startswith("image/"): raise HTTPException(415, detail="只接受图片文件")`；④ `if len(text) > WEB_KNOWLEDGE_MAX_CHARS: raise HTTPException(422, detail="图片说明过长")`；⑤ `tmp_path = await _save_upload(file, prefix="nebula-image-")`（可能 413/400）；⑥ 在 `try` 中 `image = tmp_path.read_bytes()` 把整张图读进内存，`finally` 中 `tmp_path.unlink(missing_ok=True)` 立即清理临时文件（与 `ingest()` 不同，这里读取后马上删除，因为后续不再需要路径）；⑦ 组装 `metadata` 字典：`source`/`filename` 为文件名、`captured_at` 为 `captured_at.strip()`、`reference_time` 为 `datetime.now(UTC).isoformat()`、`modality` 固定为 `"image"`；⑧ 在 `try` 中先 `guard_embedding(confirm_rebuild=confirm_rebuild)`，再 `result = ingest_image(app.state.pipeline, image=image, text=text, mime_type=mime_type, metadata=metadata)`（注释说明唯一实现在 `tool/ingest_image.py`，工具名 `knowledge.ingest_image`）；⑨ `except HTTPException: raise` 放行闸门的 409；⑩ `except Exception as exc:` 先 `raise_embedding_http(exc)` 尝试映射成 409，若 `isinstance(exc, (TypeError, ValueError))` 则抛 422「图片入库参数无效：...」，否则 `raise` 原样重抛（变成 500）；⑪ `invalidate_graph()`；⑫ 返回 `{"ok": True, **result}`。
- **异常/边界**：415（非图片 MIME，包括 `content_type` 为空的情况，因为空串不以 `image/` 开头）、422（说明过长或参数类型/取值无效）、413/400（来自 `_save_upload`）、409（嵌入锁不匹配）、以及其它异常重抛成 500。`file.content_type` 为 `None` 时安全降级为空串并被 415 拒绝。`captured_at` 不做格式校验，只去空白。`text` 只校验长度、不校验空白。临时文件在读取后立即删除，异常时也由 `finally` 保证清理。
- **同文件关系**：调用 `_save_upload()`、`guard_embedding()`、`raise_embedding_http()`、`invalidate_graph()`；读取 `lifespan()` 建立的 `app.state.pipeline`；外部调用 `ingest_image`。

#### `def reseed() -> dict[str, Any]` （第 646 行，路由 `POST /api/seed`）

- **作用**：（重新）播种 Aetheria 种子数据的端点，对应课设里的演示需求：让系统在初始状态或用户清空后能一键恢复一套可展示的样例知识。它是幂等的——重复调用不会产生重复数据（幂等性由 `tool/seed_knowledge.py` 的 `seed` 保证，本端点只负责调用与缓存失效）。播种会改变图内容，因此必须让星云图缓存失效。
- **参数**：无参数（没有请求体、没有查询参数）。
- **返回**：返回 `dict[str, Any]`，即 `seed(the_manager())` 的原始返回值（通常包含播种了哪些内容、数量等统计信息）。
- **内部流程**：① `result = seed(the_manager())`；② `invalidate_graph()` 让图缓存失效；③ 返回 `result`。
- **异常/边界**：无特殊处理，`seed` 抛出的异常会直接冒泡为 500。重复调用依赖下游的幂等实现，本函数不做去重判断。
- **同文件关系**：调用 `the_manager()` 与 `invalidate_graph()`；外部调用 `tool.seed_knowledge.seed`（导入时别名为 `seed`）；`seed` 还被 `lifespan()` 在 `WEB_AUTOSEED` 为真时用于启动播种。

#### `def export(request: Request) -> JSONResponse` （第 652 行，路由 `GET /api/export`）

- **作用**：导出端点，把全部记忆导出成一个 JSON 文件下载，对应课设「库 → 文件」的硬性要求。它返回的是 `JSONResponse` 而不是普通字典，原因是需要设置 `Content-Disposition: attachment` 响应头，让浏览器把它当文件下载而不是在页面里展示。导出文件名由 `tool/export_knowledge.py` 的 `export_filename()` 生成，注释说明用本地时间戳（面向用户、非持久化时间语义）。
- **参数**：`request`，类型 `fastapi.Request`，是 FastAPI 注入的请求对象。函数体实际并未使用它——保留该参数是为了拿到请求上下文（例如将来需要读取头信息）以及让签名更明确；本实现里它是一个未使用的参数。
- **返回**：返回 `fastapi.responses.JSONResponse`，body 是 `export_payload` 生成的完整导出载荷，响应头里带 `Content-Disposition: attachment; filename="<export_filename()>"`。
- **内部流程**：① `payload = export_payload(the_manager(), limit=0)`——注释说明载荷构造的唯一实现在 `tool/export_knowledge.py`，`limit=0` 表示全量导出；② 构造 `JSONResponse(payload, headers={"Content-Disposition": f'attachment; filename="{export_filename()}"'})` 并返回。
- **异常/边界**：无特殊处理，`export_payload` 的异常会冒泡为 500。`limit=0` 是全量导出的约定值。文件名直接内插到响应头里，未做转义处理（如果文件名含引号会破坏头部语法，属于隐含边界）。
- **同文件关系**：调用 `the_manager()`；外部调用 `export_payload` 与 `export_filename`；不调用本文件里的其它函数。

#### `async def import_file(file: UploadFile) -> dict[str, Any]` （第 664 行，路由 `POST /api/import`）

- **作用**：导入端点，把此前 `/api/export` 导出的 JSON 文件重新写回库里，对应课设「文件 → 库」的硬性要求。它先按上传流式落盘（强制 `.json` 后缀、带大小上限），读取字节后立即删临时文件，然后交给 `tool/import_knowledge.py` 解析并逐条写入。非法 JSON 被翻译成 400 而不是 500，并在确实有导入内容时让图缓存失效。
- **参数**：`file`，`UploadFile`，表单上传的 JSON 文件；函数本身不使用其文件名，但 `_save_upload` 会读取。
- **返回**：返回 `dict[str, Any]`，即 `import_items(the_manager(), entries)` 的返回值（通常包含 `imported` 计数等统计字段，因为函数内部用 `result["imported"]` 做判断，说明该键必然存在）。
- **内部流程**：① `tmp_path = await _save_upload(file, prefix="nebula-import-", suffix=".json")`——显式指定后缀，保证临时文件是 `.json`（可能抛 413/400）；② 在 `try` 中 `raw = tmp_path.read_bytes()` 读全部字节，`finally` 中 `tmp_path.unlink(missing_ok=True)` 立即清理；③ 在 `try` 中 `entries = parse_import_payload(raw)`，`except ValueError as exc` 抛 `HTTPException(400, detail=str(exc))` 并 `from exc`——注释说明解析与逐条写入的唯一实现在 `tool/import_knowledge.py`，非法 JSON 仍是 400；④ `result = import_items(the_manager(), entries)`；⑤ `if result["imported"]:` 为真则 `invalidate_graph()`（没有真正导入任何东西时不做无谓的缓存失效）；⑥ 返回 `result`。
- **异常/边界**：413（文件超限）、400（空文件，来自 `_save_upload`）、400（JSON 非法或结构不合法，来自 `parse_import_payload` 的 `ValueError`）。临时文件在读取后立即删除并由 `finally` 兜底。`result["imported"]` 采用直接下标访问，若下游返回的字典缺少该键会抛 `KeyError`（属于对下游契约的隐式依赖）。
- **同文件关系**：调用 `_save_upload()`、`the_manager()`、`invalidate_graph()`；外部调用 `parse_import_payload` 与 `import_items`。

#### `def the_repository() -> DocumentRepository` （第 686 行，`create_app` 内嵌）

- **作用**：文档仓储的取值辅助闭包，返回 RAG 管道缓存的那个 `DocumentRepository`。注释说明了它的两个特性：与管道使用同一个 sqlite 文件、自带锁、每个作用域使用独立连接——因此复用管道缓存的那个实例是最安全的做法，而不是新建一个。当记忆库是 `:memory:` 内存模式时，不存在 `documents/chunks` 真值源，此时它抛 400 并给出明确中文提示，让「文档中心」相关端点有一个统一的、可读的失败方式。
- **参数**：无参数。
- **返回**：返回 `DocumentRepository` 实例（`app.state.pipeline.document_repo()` 的结果）。当该结果为 `None` 时抛异常而不返回。
- **内部流程**：① `repository = app.state.pipeline.document_repo()`；② `if repository is None:` 则抛 `HTTPException(400, detail="当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源")`；③ `return repository`。
- **异常/边界**：抛 `HTTPException(400)`，条件是仓储为 `None`（内存模式）。`app.state.pipeline` 缺失会抛 `AttributeError`（未启动场景）。
- **同文件关系**：被 `list_documents()`、`get_document()`、`revectorize_document()` 调用；它自己不调用本文件里的其它函数。

#### `def list_documents(tag: str = "", status: str = "", page: int = 1, page_size: int = 20) -> dict[str, Any]` （第 698 行，路由 `GET /api/documents`）

- **作用**：文档中心的列表端点，支持按标签与状态过滤、以及分页。它是「documents/chunks 真值源 → 向量/图投影」这套三库模型里读取真值源的入口。列表逻辑本身（含 `page_size` 上限校验）已收敛到 `tool/document_list.py`，本端点只保留 HTTP 边界与错误码映射：把下游的 `ValueError`（典型是分页参数越界）翻译成 422。
- **参数**：`tag`，`str`，查询参数，默认空串，按标签过滤；空串表示不过滤。`status`，`str`，查询参数，默认空串，按状态过滤；空串表示不过滤。`page`，`int`，查询参数，默认 `1`。`page_size`，`int`，查询参数，默认 `20`，其上限校验在下游实现里。
- **返回**：返回 `dict[str, Any]`，即 `list_documents_payload(...)` 的原始返回值（通常包含文档条目列表与分页元信息）。参数非法时抛 422 而不返回。
- **内部流程**：① 在 `try` 中调用 `list_documents_payload(the_repository(), tag=tag, status=status, page=page, page_size=page_size)` 并直接返回；② `except ValueError as exc` 抛 `HTTPException(422, detail=str(exc))` 并 `from exc`。
- **异常/边界**：抛 400（内存模式下 `the_repository()` 抛出的）、422（下游 `ValueError`）。注意 `the_repository()` 的 400 是在 `try` 内部调用的，但 `HTTPException` 不是 `ValueError`，因此不会被误吞、会正常向外传播。`tag`/`status` 传空串是「不过滤」的约定值，不做额外校验。
- **同文件关系**：调用 `the_repository()`；外部调用 `list_documents_payload`（从 `tool.document_list` 导入并别名）。

#### `def get_document(document_id: str) -> dict[str, Any]` （第 708 行，路由 `GET /api/documents/{document_id}`）

- **作用**：单个文档详情端点，按 id 返回一份文档的完整信息（含其切块等）。详情逻辑的唯一实现在 `tool/document_get.py`，本端点只做两件事：拿到仓储、把「找不到」翻译成 404。这样 Web 端点与 Agent 工具再次共享同一份实现，避免两处查询口径不一致。
- **参数**：`document_id`，`str`，路径参数，来自 URL 的 `{document_id}` 段，指定要查看的文档标识。
- **返回**：返回 `dict[str, Any]`，即 `document_payload(the_repository(), document_id)` 的返回值。文档不存在时抛 404 而不返回。
- **内部流程**：① 在 `try` 中 `return document_payload(the_repository(), document_id)`；② `except LookupError as exc` 抛 `HTTPException(404, detail=str(exc))` 并 `from exc`。
- **异常/边界**：抛 404（下游 `LookupError`，即文档不存在）、400（`the_repository()` 在内存模式下抛出，`HTTPException` 不被 `LookupError` 捕获所以正常传播）。`document_id` 为空串不会被本层拒绝，会直接交给下游查询并很可能得到 404。
- **同文件关系**：调用 `the_repository()`；外部调用 `document_payload`（从 `tool.document_get` 导入并别名）。

#### `def revectorize_document(document_id: str, confirm_rebuild: bool = Query(default=False)) -> dict[str, Any]` （第 716 行，路由 `POST /api/documents/{document_id}/revectorize`）

- **作用**：重建指定文档的向量投影端点。它体现了设计约束 D8：网关不可达时明确失败，绝不悄悄切换到别的向量空间去写，否则同一集合里会混进两套不兼容的向量。它把重嵌入的唯一实现委托给 `tool/document_revectorize.py`（注释指出那里的锁闸门顺序与本文件一致），自己负责把下游的各类异常精确映射成 409/404/422/502 等状态码，是本文件里异常映射最完整的一个端点。
- **参数**：`document_id`，`str`，路径参数，要重建向量的文档标识。`confirm_rebuild`，`bool`，查询参数，默认 `False`，作为「确认重建」的显式意图传给下游。
- **返回**：返回 `dict[str, Any]`，即 `revectorize_document_payload(...)` 的返回值（重建结果统计）。失败时按异常类型抛对应 HTTP 错误。
- **内部流程**：① 在 `try` 中 `return revectorize_document_payload(the_manager(), the_repository(), document_id, confirm_rebuild=confirm_rebuild)`，四个实参分别是管理器、仓储、文档 id 与确认标志；② 依次捕获并映射异常：`EmbeddingLockMismatch` → 409（detail 用 `exc.to_detail()`）；`LookupError` → 404；`ValueError` → 422；`HTTPException` → 原样 `raise`（避免把已经成型的 HTTP 错误再包一层）；其它 `Exception` → 先 `raise_embedding_http(exc)` 尝试识别成嵌入锁不匹配并抛 409，然后抛 `HTTPException(502, detail=f"重嵌入失败：{type(exc).__name__}: {exc}")`。
- **异常/边界**：409（嵌入锁不匹配，可能来自 `EmbeddingLockMismatch` 也可能来自 `raise_embedding_http` 的映射）、404（文档不存在）、422（参数无效）、502（其它失败，附异常类型名）、400（内存模式下 `the_repository()` 抛出，因为它是 `HTTPException` 会命中 `except HTTPException: raise` 分支被原样放行）。捕获顺序很重要：`HTTPException` 分支放在宽泛的 `Exception` 之前，保证已成型的状态码不被 502 覆盖。
- **同文件关系**：调用 `the_manager()`、`the_repository()`、`raise_embedding_http()`；外部调用 `revectorize_document_payload`（从 `tool.document_revectorize` 导入并别名）。

#### `def stats() -> dict[str, Any]` （第 743 行，路由 `GET /api/stats`）

- **作用**：知识库统计端点，返回三库（文档真值源、向量投影、图）的规模等指标，供前端展示「知识星云有多大」。注释强调统计口径的唯一实现在 `tool/knowledge_stats.py`（即 `knowledge.stats` 工具）：同一个函数既服务这个 HTTP 端点，也能被 LLM 直接当作工具调用，保证人看的数字和模型看的数字完全一致。
- **参数**：无参数。
- **返回**：返回 `dict[str, Any]`，即 `knowledge_stats(...)` 返回的 Pydantic 模型经 `.model_dump()` 得到的字典。
- **内部流程**：只有一步：`return knowledge_stats(the_manager(), app.state.pipeline.document_repo()).model_dump()`。注意这里直接调 `document_repo()` 而**没有**经过 `the_repository()`，因此内存模式下不会抛 400，而是把 `None` 交给 `knowledge_stats` 自行处理。
- **异常/边界**：无特殊处理；`knowledge_stats` 内部若对 `None` 仓储不兼容会抛异常并冒泡为 500。`app.state.pipeline` 缺失会抛 `AttributeError`。
- **同文件关系**：调用 `the_manager()`；读取 `lifespan()` 建立的 `app.state.pipeline`；外部调用 `knowledge_stats`；与 `the_repository()` 不同，它不走那个 400 检查。

#### `def reconcile() -> dict[str, Any]` （第 749 行，路由 `GET /api/reconcile`）

- **作用**：三库对账的只读端点，报告「documents/chunks 真值源」与「向量投影」「图投影」之间的差异（例如真值源里有但向量库缺失的块）。它是漂移自愈能力的前半段：先看清楚差在哪，再决定是否修。对账逻辑本身收敛在 `tool/reconcile.py`（可被 Agent 调用的独立工具），本端点只做 HTTP 边界。
- **参数**：无参数。
- **返回**：返回 `dict[str, Any]`，即 `reconcile_report(the_manager(), app.state.pipeline.document_repo())` 的返回值，描述各库之间的一致性状态与差异明细。
- **内部流程**：只有一步：直接返回 `reconcile_report(the_manager(), app.state.pipeline.document_repo())`。与 `stats()` 一样，这里直接调 `document_repo()` 而不是 `the_repository()`，所以内存模式下不会主动抛 400。
- **异常/边界**：无特殊处理，下游异常冒泡为 500。无请求体、无参数校验。
- **同文件关系**：调用 `the_manager()`；读取 `app.state.pipeline`；外部调用 `reconcile_report`。

#### `def reconcile_repair(body: ReconcileBody) -> dict[str, Any]` （第 753 行，路由 `POST /api/reconcile`）

- **作用**：三库对账的修复端点，按请求体里指定的修复项执行实际修补（例如把缺失的向量补回来）。它与 `GET /api/reconcile` 构成「报告 / 修复」一对：`body.repair` 为空时下游只报告不修，非空时按列表执行。修复逻辑收敛在 `tool/repair_drift.py`，本端点只做请求体解析与错误码映射。
- **参数**：`body`，类型 `ReconcileBody`，从 JSON 请求体解析，其唯一字段 `repair`（`list[str]`，默认空列表）是要执行的修复项名称列表。
- **返回**：返回 `dict[str, Any]`，即 `repair_drift(the_manager(), app.state.pipeline.document_repo(), body.repair)` 的返回值，包含实际执行的修复动作与结果。
- **内部流程**：① 在 `try` 中 `return repair_drift(the_manager(), app.state.pipeline.document_repo(), body.repair)`，第三个位置参数就是修复项列表；② `except ValueError as exc` 抛 `HTTPException(422, detail=str(exc))` 并 `from exc`——典型场景是修复项名称不认识。
- **异常/边界**：抛 422（下游 `ValueError`，例如非法修复项）；其它异常冒泡为 500。`body.repair` 为空列表不会报错，会走下游的「只报告不修」语义。
- **同文件关系**：使用本文件的 `ReconcileBody` 模型与 `the_manager()`；读取 `app.state.pipeline`；外部调用 `repair_drift`。

#### `def rebuild_embedding(confirm_rebuild: bool = Query(default=False)) -> dict[str, Any]` （第 762 行，路由 `POST /api/embedding/rebuild`）

- **作用**：嵌入投影重建端点。当用户更换了嵌入模型（维度可能变化）或从离线哈希切到云端嵌入时，需要显式确认并重建向量投影，让锁与 Qdrant 维度对齐。它的执行顺序很讲究：先过 `guard_embedding` 闸门（这一步在 `confirm_rebuild=True` 时会执行实际的锁更新/重建），再让图缓存失效，最后读取一份嵌入锁快照返回给调用方确认结果。注释明确它需要「确认并重建」的语义。
- **参数**：`confirm_rebuild`，`bool`，查询参数，默认 `False`。为 `False` 时闸门会因不匹配而抛 409；为 `True` 时表示用户确认重建，闸门放行并执行重建。
- **返回**：返回 `dict[str, Any]`，含四个键：`ok`（固定 `True`）、`rebuilt`（`bool(confirm_rebuild)`，如实反映调用方是否确认了重建）、`embedding_lock`（快照里的 `locked`，即当前锁定的嵌入标识）、`qdrant_dimension`（快照里的 `qdrant_dimension`，即线上集合维度）。
- **内部流程**：① `guard_embedding(confirm_rebuild=confirm_rebuild)`——这一步可能抛 409；② `invalidate_graph()` 让图缓存失效；③ `snapshot = inspect_embedding_lock(the_manager(), app.state.pipeline.document_repo())` 读取锁与维度快照；④ 组装并返回结果字典，其中 `rebuilt` 用 `bool(confirm_rebuild)` 而不是从下游返回，说明它是「调用方意图」的如实回显。
- **异常/边界**：抛 409（嵌入锁不匹配且未确认重建）。`snapshot` 字典的键采用直接下标访问（`snapshot["locked"]`、`snapshot["qdrant_dimension"]`），若下游返回缺键会抛 `KeyError`。`confirm_rebuild=False` 且本来就一致时不会报错，返回 `rebuilt: False`。
- **同文件关系**：调用 `guard_embedding()`、`invalidate_graph()`、`the_manager()`；读取 `app.state.pipeline`；外部调用 `inspect_embedding_lock`。

#### `def health() -> dict[str, Any]` （第 776 行，路由 `GET /api/health`）

- **作用**：健康检查端点，也是整个应用信息最密集的只读端点。它报告：聊天是否就绪、实际生效的嵌入实现模式（云端 API 还是本地哈希）、当前嵌入配置的字典、视觉模型名、联网搜索是否可用、三个存储各自的实际实现（注释注明对应设计约束 D1：全部在本机）、出网点的降级状态（对应 D8/D9，云端嵌入未配置时必须如实提示检索只能走 FTS5）、以及嵌入锁的当前值/投影值/Qdrant 维度/是否存在不匹配、还有知识抽取器的类名。它的设计原则是「报告实际生效的实现，而不是猜配置」——所以全部通过实例属性判断（云端实现带 `base_url`，离线 `HashEmbedding` 没有）。
- **参数**：无参数。
- **返回**：返回 `dict[str, Any]`，键包括：`ok`（固定 `True`）、`chat_ready`（`chat_ready()` 的第一个返回值）、`embedding_mode`（`"api"` 或 `"local-hash"`）、`embedding_reachable`（`bool(base_url)`）、`embedding`（`getattr(embedding, "to_dict", dict)()`，即嵌入对象自己的字典化结果，没有 `to_dict` 就返回空字典）、`vision_model`（管道抽取器的 `vision_model`，通过 `and` 短路取到 `None` 或值）、`search_available`、`store_modes`（含 `document`/`vector`/`graph` 三个子键）、`degraded`（含 `embedding`、`embedding_endpoint`、`chat_ready`、`keyword_fallback`、`embedding_hint`）、`embedding_lock`、`embedding_current`、`embedding_projection`、`qdrant_dimension`、`embedding_mismatch`、`knowledge_extractor`。
- **内部流程**：① `ready, _ = chat_ready()`，只要就绪布尔值、忽略原因（原因已经在 `/api/chat` 里用）；② `embedding = getattr(the_manager(), "embedding", None)`，`base_url = getattr(embedding, "base_url", None)`；③ 由 `base_url` 推导 `embedding_mode`（`"api"` / `"local-hash"`）与 `embedding_state`（`"api"` / `"hash"`）——两个变量取值不同但同源，分别服务于不同字段；④ `manager = the_manager()` 再取一次（与第 ① 步的 `the_manager()` 是同一个对象）；⑤ `repository = app.state.pipeline.document_repo()`；⑥ `snapshot = inspect_embedding_lock(manager, repository)`、`lock = snapshot["locked"]`；⑦ 返回大字典。其中 `embedding` 字段用 `getattr(embedding, "to_dict", dict)()` 这一惯用法：拿不到 `to_dict` 时回退到内置 `dict` 并用无参调用得到 `{}`；`vision_model` 用 `getattr(getattr(app.state, "pipeline", None), "extractor", None) and getattr(app.state.pipeline.extractor, "vision_model", None)`，即管道或抽取器缺失时短路得到 `None`；`store_modes.document` 判断 `str(getattr(manager.document_store, "path", "")) == ":memory:"`，是内存模式则为 `"memory"` 否则 `"sqlite"`；`store_modes.vector` 直接取向量存储的类名；`store_modes.graph` 看 `manager.graph_store.driver` 是否为 `None`，非空则 `"neo4j"`，否则 `"inmemory"`；`degraded.keyword_fallback` 为 `not bool(base_url)`；`degraded.embedding_hint` 由 `embedding_config_hint(manager)` 生成；`knowledge_extractor` 取 `type(app.state.pipeline.extractor).__name__`。
- **异常/边界**：本函数不捕获异常；`inspect_embedding_lock` 或 `document_repo()` 抛错会让健康检查变成 500（这是一个隐含边界：健康检查本身依赖了仓储可用）。用 `getattr` 默认值处理 `embedding` 缺失、`to_dict` 缺失、`pipeline`/`extractor` 缺失、`document_store.path` 缺失等情况，因此这些退化场景不会崩。`snapshot` 的键采用直接下标访问，缺键会抛 `KeyError`。
- **同文件关系**：调用 `embedding_config_hint()` 与 `the_manager()`；读取 `lifespan()` 建立的 `app.state.pipeline`；外部调用 `chat_ready`、`search_available`、`inspect_embedding_lock`。

#### 模块级 `app = create_app()` （第 829 行）

- **作用**：模块导入时立即执行的应用工厂调用，把 `create_app()` 的返回值绑定到模块级名字 `app`。这样无论是 `uvicorn web.app:app` 这种「按导入路径找属性」的启动方式，还是其它模块 `from web.app import app` 的引用方式，都能拿到同一个 FastAPI 实例。它不是函数，但在本文件的运行时语义里承担了「装配入口」的角色。
- **参数**：无（作为赋值语句）。
- **返回**：绑定 `FastAPI` 实例到全局名 `app`。
- **内部流程**：调用 `create_app()`（不传 manager，因此使用共享单例且由应用自己拥有），并把结果赋值给 `app`。注意此时 `lifespan` 尚未执行，`app.state.manager` 等属性要等应用启动才存在。
- **异常/边界**：`create_app()` 若在导入期抛异常，整个模块导入失败。没有其它特殊处理。
- **同文件关系**：调用本文件的 `create_app()`；被 `if __name__ == "__main__"` 分支中的 `uvicorn.run(app, ...)` 使用。

#### 模块级 `if __name__ == "__main__"` 运行分支 （第 832 行）

- **作用**：让 `python -m web.app` 直接把这个模块当脚本运行时启动 HTTP 服务，这也是项目背景里说明的真实运行入口。它刻意把 `import uvicorn` 与 `from constants import DEFAULT_WEB_PORT` 放在分支内部做延迟导入，避免仅仅导入本模块（例如被测试或其它模块引用）时就引入 uvicorn 依赖。注释强调监听端口的唯一来源是 `constants.DEFAULT_WEB_PORT`，历史上的 `NEBULA_PORT` 环境变量已经删除。
- **参数**：无（作为条件分支）。
- **返回**：无返回值；调用 `uvicorn.run` 后进程进入服务循环，直到被终止。
- **内部流程**：① 判断 `__name__ == "__main__"`；② 局部 `import uvicorn`；③ 从 `constants` 导入 `DEFAULT_WEB_PORT`；④ `uvicorn.run(app, host=LOCALHOST, port=DEFAULT_WEB_PORT)`，其中 `LOCALHOST` 在文件顶部已从 `constants` 导入，因此服务默认只监听本机地址（配合文档字符串里的默认 `http://127.0.0.1:8765`）。
- **异常/边界**：端口被占用等错误由 uvicorn 自行报错并退出；本分支不做异常处理。被当作模块导入时该分支完全不执行。
- **同文件关系**：使用模块级 `app`（由 `create_app()` 产生）；依赖文件顶部导入的 `LOCALHOST` 与分支内导入的 `DEFAULT_WEB_PORT`。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `ReconcileBody` | `POST /api/reconcile` 的请求体模型，用 `repair` 字符串列表指定要执行的修复项，空列表表示只报告不修。 |
| `embedding_config_hint` | 根据 manager 的嵌入对象是否带 `base_url`，返回「云端嵌入未配置」的配置指引文案或空串，供 `/api/health` 的降级提示使用。 |
| `ConfirmedChatToolCall` | 描述人类在上一轮明确接受的破坏性工具调用（只允许 `memory.manage`），用于聊天的二次确认闭环。 |
| `ChatBody` | `POST /api/chat` 的请求体模型，承载消息文本、离线/联网模式与可选的确认调用。 |
| `FactBody` | `POST /api/facts` 的请求体模型，约束三元组主语/谓语/宾语的长度以及 0~1 的置信度。 |
| `GraphRAGBody` | `POST /api/graph-rag` 的请求体模型，描述查询词、召回条数、图上扩展跳数与历史时刻。 |
| `KnowledgeBody` | `POST /api/knowledge` 的请求体模型，承载要入库的原文、业务时间与「同步等待还是后台排队」的开关。 |
| `_save_upload` | 把上传流式写入临时文件并在写入过程中强制 `MAX_UPLOAD_BYTES` 上限，超限 413、空文件 400，返回临时文件路径。 |
| `create_app` | 应用工厂：注册 lifespan、全部 API 路由与静态前端挂载，支持注入 manager，返回配置完成的 FastAPI 实例。 |
| `lifespan`（内嵌） | 启动时准备 manager、RAG 管道、后台入库队列、聊天锁并按需播种，关闭时停队列并按所有权释放 manager。 |
| `the_manager`（内嵌） | 从 `app.state.manager` 取出当前记忆管理器的统一访问点。 |
| `guard_embedding`（内嵌） | 写向量投影前的嵌入锁闸门，不匹配时抛 409，`confirm_rebuild=True` 时放行重建。 |
| `raise_embedding_http`（内嵌） | 把捕获到的异常识别为嵌入锁不匹配并翻译成 409，否则静默返回交由调用方兜底。 |
| `invalidate_graph`（内嵌） | 递增全局图修订号并把本地图缓存 payload 置空，保证 `since` 轮询只有一个真相来源。 |
| `graph`（路由） | `GET /api/graph`：按「修订号 + as-of 时间」缓存返回全图数据，并在 `since` 命中时回 `unchanged` 轻量响应。 |
| `graph_rag`（路由） | `POST /api/graph-rag`：调用管道做向量证据 + 图路径混合检索，同时返回结构化结果与可读上下文。 |
| `chat`（路由） | `POST /api/chat`：串行化调用知识管家、处理危险操作确认、写问答留痕、后台调度图抽取并附上溯源报告。 |
| `ingest`（路由） | `POST /api/ingest`：上传文档流式落盘后经嵌入锁闸门切块入库，记录 episodic 记忆并失效图缓存。 |
| `add_fact`（路由） | `POST /api/facts`：把手工三元组交给 `knowledge.add_fact` 写入图并失效缓存。 |
| `add_knowledge`（路由） | `POST /api/knowledge`：一句话入库，按队列可用性与 `wait` 选择异步排队、同步等待或无队列同步回退三条路径。 |
| `list_knowledge_jobs`（路由） | `GET /api/knowledge/jobs`：查询入库任务历史与状态，队列不可用时返回 `available: False`，并把 `limit` 夹在 1~200。 |
| `retry_knowledge_job`（路由） | `POST /api/knowledge/jobs/{job_id}/retry`：过嵌入锁闸门后把失败任务重新入队，映射 404/400/503。 |
| `add_image_knowledge`（路由） | `POST /api/knowledge/image`：校验图片 MIME 与说明长度后做视觉 embedding 与多元关系抽取入库。 |
| `reseed`（路由） | `POST /api/seed`：幂等地（重新）播种 Aetheria 种子数据并失效图缓存。 |
| `export`（路由） | `GET /api/export`：全量导出记忆为 JSON 并带 `Content-Disposition` 头触发浏览器下载。 |
| `import_file`（路由） | `POST /api/import`：上传 JSON 落盘解析后逐条写回库，非法 JSON 映射为 400，有导入才失效图缓存。 |
| `the_repository`（内嵌） | 复用管道缓存的文档仓储；内存模式（`:memory:`）下抛 400 说明没有真值源。 |
| `list_documents`（路由） | `GET /api/documents`：按标签/状态过滤并分页列出文档，把下游 `ValueError` 映射为 422。 |
| `get_document`（路由） | `GET /api/documents/{document_id}`：返回单个文档详情，不存在时映射为 404。 |
| `revectorize_document`（路由） | `POST /api/documents/{document_id}/revectorize`：重建该文档向量投影，网关不可达时明确失败而不换向量空间。 |
| `stats`（路由） | `GET /api/stats`：用 `knowledge_stats` 的统计口径返回知识库规模指标。 |
| `reconcile`（路由） | `GET /api/reconcile`：只读报告 documents/chunks 真值源与向量、图投影之间的差异。 |
| `reconcile_repair`（路由） | `POST /api/reconcile`：按请求体给出的修复项执行漂移修补，非法修复项映射为 422。 |
| `rebuild_embedding`（路由） | `POST /api/embedding/rebuild`：确认后重建向量投影，并回显嵌入锁与 Qdrant 维度快照。 |
| `health`（路由） | `GET /api/health`：汇总聊天可用性、嵌入实现与锁状态、三个存储的实现、降级提示与抽取器类型。 |
| 模块级 `app = create_app()` | 导入期即装配好 FastAPI 实例，供 `uvicorn web.app:app` 等方式引用。 |
| 模块级 `__main__` 分支 | 直接运行本模块时用 uvicorn 在 `LOCALHOST:DEFAULT_WEB_PORT` 上启动服务。 |
