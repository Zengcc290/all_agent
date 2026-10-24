# tool/ingest_image.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时里「图片入库」这一条独立写入能力的唯一实现，对外暴露成一个名为 `knowledge.ingest_image` 的工具（`side_effect="write"`）。它解决的问题是：把一张本地图片作为**感知记忆（perceptual memory）的原始载荷**写进记忆系统，并在嵌入锁保护下让支持 VL（视觉语言）的嵌入后端把图片字节向量化，然后再调用结构化视觉抽取器，把图上识别出的实体、关系、时序观察物化进知识图谱。它刻意与纯文本入库（`memory.rag`）分开：图片的图谱边只能来自视觉抽取器，绝不来自向量相似度，这条规则决定了它是一个独立动作而不是 `ingest` 的一个参数。文件内部大致分成四块：模块级常量（`TOOL_ENABLED`、`MIME_BY_SUFFIX`、`TEXT_ONLY_WARNING`）、两个纯函数（`guess_mime_type`、`ingest_image`）、三个 Pydantic 数据模型（`IngestImageInput`、`IngestExtractionReport`、`IngestImageOutput`）、以及工具类 `IngestImageTool` 和工厂函数 `create_tool`。整个入库流程固定三步：过嵌入锁闸门 → 写一条带图片字节的 perceptual 记忆 → 视觉抽取并物化成 n-ary 观察；抽取失败**不**回滚已入库的图片，源句必须留下，错误只写进报告。Web 层与工具协议层都直接调用本模块的 `ingest_image`，不再经过 `RAGPipeline.ingest_media` 那层薄委托。

## 二、函数与类逐条详解

### `guess_mime_type(path: Path) -> str` （第 59 行）
- **作用**：根据图片文件的扩展名推断出 MIME 媒体类型，供后续把文件字节按正确媒体类型交给视觉抽取器。它存在的意义是：用户在工具入参里可以不填 `mime_type`，此时需要一个兜底推断，否则无法告诉抽取器这是什么格式的图片。推断只依据扩展名，不看文件内容，所以即使扩展名与实际内容不一致，它也只会给出一个「按扩展名看合理」的答案，真正的解释权仍归抽取器。它被 `IngestImageTool.execute` 在 `mime_type` 留空时调用，属于「输入补全」这一步的默认值来源。
- **参数**：`path`，类型 `pathlib.Path`，无默认值，代表要推断类型的图片文件路径。函数只读取它的 `suffix` 属性，因此路径本身不需要真实存在，也不会触发任何磁盘 I/O；传入目录路径、不存在的路径都能正常返回。
- **返回**：返回 `str` 类型的 MIME 字符串。若 `path.suffix.lower()` 命中 `MIME_BY_SUFFIX` 字典（`.jpg`/`.jpeg` → `image/jpeg`，`.png` → `image/png`，`.webp` → `image/webp`，`.gif` → `image/gif`，`.bmp` → `image/bmp`），返回对应值；未命中（含没有扩展名、扩展名未知如 `.tiff`）时统一返回默认值 `"image/jpeg"`。
- **内部流程**：第一步取 `path.suffix` 得到扩展名字符串（含前导点，如 `".PNG"`）；第二步对它调用 `.lower()` 做大小写归一，保证 `.JPG`、`.Png` 这类大写写法也能命中；第三步以归一后的键去查模块级字典 `MIME_BY_SUFFIX`，用 `dict.get(key, "image/jpeg")` 一步完成查找与兜底，直接返回结果。整个函数没有分支语句、没有循环、没有副作用。
- **异常/边界**：无特殊处理。`path.suffix` 对 `Path` 对象总是可用的字符串（可能为空串），`.lower()` 与字典查找都不会抛异常；没有扩展名或扩展名不在表中时静默回退到 `image/jpeg`，不会报错也不会提示「未知格式」。因此它对非法扩展名是「宽容」而非「拒绝」的，真正的合法性检查由上层 `IngestImageTool.execute` 的 `startswith("image/")` 判断承担。
- **同文件关系**：它读取本文件模块级常量 `MIME_BY_SUFFIX`；被本文件的 `IngestImageTool.execute` 调用（当入参 `mime_type` 去掉空白后为空时作为默认推断）；不调用本文件里任何其他函数。

### `ingest_image(pipeline: RAGPipeline, *, image: bytes, text: str = "", mime_type: str = "image/jpeg", metadata: dict[str, Any] | None = None) -> dict[str, Any]` （第 63 行）
- **作用**：这是整个图片入库能力的核心纯函数，完成「校验入参 → 过嵌入锁 → 写感知记忆 → 视觉抽取 → 物化进图谱 → 组装报告」的全部工作。它把图片字节当作规范化的感知载荷交给记忆管理器嵌入，同时把知识图谱的边完全交给结构化视觉抽取器生成，绝不使用向量相似度。它被设计成「抽取失败不回滚」的语义：只要图片已经写入成功，即使后面的抽取环节抛异常，也只在报告里追加错误字符串，绝不删除或撤销那条感知记忆。`IngestImageTool.execute` 是它的主要调用方，Web 层也直接调用它，因此它不依赖工具类即可使用。它是本文件里唯一会真正修改记忆系统状态的函数。
- **参数**：
  - `pipeline`：位置参数，类型 `RAGPipeline`，无默认值，代表已经构建好的检索增强入库管线。函数会访问它的 `manager`（记忆管理器）、`document_repo()`（文档仓库，用于嵌入锁）、`auto_extract`（是否自动抽取的开关）、`extractor`（抽取器）、`last_ingest_report`（最近一次入库报告字段）。传入 `None` 或缺少这些属性的对象会在运行中报属性错误。
  - `image`：关键字参数，类型 `bytes`，无默认值，图片的原始字节内容。约束是非空且必须是 `bytes` 类型（`bytearray` 也会被拒绝），否则抛 `ValueError`。
  - `text`：关键字参数，类型 `str`，默认空字符串 `""`，图片的文字说明或上下文。约束上只做「是否为字符串」的宽容判断：非字符串会被当成空处理；字符串会先 `.strip()` 去空白。如果去空白后为空，则用元数据里的 `filename` 或固定串 `"图片观察"` 兜底，保证写入内容非空。
  - `mime_type`：关键字参数，类型 `str`，默认 `"image/jpeg"`，图片媒体类型。约束是必须为 `str` 且以 `"image/"` 开头，否则抛 `ValueError`。
  - `metadata`：关键字参数，类型 `dict[str, Any] | None`，默认 `None`，附加元数据。函数会做 `dict(metadata or {})` 浅拷贝，因此不会修改调用方传入的字典；`None`、空字典都等价于「无元数据」。
- **返回**：返回一个 `dict[str, Any]`，固定包含四个键：`"item_id"`（写入的感知记忆项 id，来自 `item.id`）、`"modality"`（固定字符串 `"image"`）、`"extraction"`（抽取报告字典，见下）、`"warning"`（字符串，当报告里的 `multimodal_embedding` 为假时返回模块级常量 `TEXT_ONLY_WARNING` 的诚实提示，否则返回空字符串）。`extraction` 报告字典包含 `chunks`（固定为 1）、`domains`（抽取命中的领域列表，未抽取时为空列表）、`entities`、`relations`、`superseded`、`retracted`、`skipped_relations`（整数计数）、`errors`（错误字符串列表）、`modality`（固定 `"image"`）、`multimodal_embedding`（布尔）。函数无论抽取成功与否都会返回这个字典（除非前面的校验就抛异常），不会返回 `None`。
- **内部流程**：
  1. 入参校验：`image` 不是 `bytes` 或为空 → 抛 `ValueError("image must be non-empty bytes")`；`mime_type` 不是 `str` 或不以 `image/` 开头 → 抛 `ValueError("mime_type must be an image media type")`。
  2. 复制元数据：`details = dict(metadata or {})`，然后用 `setdefault` 在缺失时补 `"source"`（优先取 `details.get("filename")`，否则用中文串 `"图片入库"`），再强制把 `details["modality"] = "image"` 覆盖写入。
  3. 规范化正文：`content = text.strip() if isinstance(text, str) else ""`；若 `content` 为空，则取 `str(details.get("filename") or "图片观察")` 作为内容。
  4. 嵌入锁闸门：调用 `apply_embedding_lock(pipeline.manager, pipeline.document_repo())`，在嵌入/写入期间持有锁，避免并发写入互相干扰。
  5. 写感知记忆：`pipeline.manager.add(content, memory_type=MemoryType.PERCEPTUAL, metadata=details, payload=image, modality="image", timestamp=details.get("captured_at") or None)`，把图片字节作为载荷，时间戳取元数据里的 `captured_at`，缺失则传 `None` 让管理器自行决定。返回的记忆项记为 `item`。
  6. 初始化报告 `report`：`chunks=1`、`domains=[]`、各计数为 0、`errors=[]`、`modality="image"`，`multimodal_embedding` 通过 `getattr(pipeline.manager.embedding, "multimodal", False)` 探测当前嵌入后端是否支持多模态并转成布尔。
  7. 自动抽取分支：仅当 `pipeline.auto_extract` 为真时进入。构造 `EntityResolver(pipeline.manager)`，用 `build_graph_context(pipeline.manager, content, resolver=resolver)` 取图谱上下文；随后用 `accepts_parameter` 做抽取器协议自省，按能力拼 `kwargs`：总是带 `metadata=details`，若抽取器接受 `graph_context` 参数就加上，若接受 `image` 参数就同时加上 `image=image` 与 `mime_type=mime_type`；接着 `pipeline.extractor.extract(content, **kwargs)` 得到 `extraction`，再调用 `materialize_extraction(...)` 把抽取结果物化成 n-ary 观察，传入 `source_item=item`、`source_metadata=details`、`resolver=resolver`。
  8. 回填报告：遍历 `("entities", "relations", "superseded", "retracted", "skipped_relations")` 五个键，从 `materialized` 中取出对应值覆盖到 `report`；再把 `report["domains"]` 设为单元素列表 `[materialized["domain"]]`。
  9. 异常兜底：整个抽取块被 `try/except Exception` 包住，捕获到任何异常时只做 `report["errors"].append(f"{type(exc).__name__}: {exc}")`，不重新抛出、不回滚第 5 步已写入的图片。
  10. 收尾：把报告挂到 `pipeline.last_ingest_report = report`，然后按前述结构返回结果字典，`warning` 依据 `report.get("multimodal_embedding")` 的真假决定是空串还是 `TEXT_ONLY_WARNING`。
- **异常/边界**：主动抛出的异常只有两种 `ValueError`（空/非 bytes 图片、非法 MIME）。`auto_extract` 为假时完全跳过抽取，报告保持零值与空 `domains`，此时 `errors` 为空但也没有任何实体关系。抽取阶段的任何异常（模型报错、网络超时、抽取器不接受参数等）都被就地吞掉并转成 `errors` 里的一条字符串，函数仍然正常返回，图片保持已入库。`metadata` 为 `None` 时按空字典处理；`text` 非字符串时按空串处理；`captured_at` 缺失或为空串时向 `manager.add` 传 `None`。`getattr(..., "multimodal", False)` 保证嵌入对象缺少该属性时不报错，只是报告为 `False` 并触发降级提示。
- **同文件关系**：它使用模块级常量 `TEXT_ONLY_WARNING`；它不调用本文件里其他函数，但被本文件的 `IngestImageTool.execute` 调用，是本文件对外的两个主要入口之一（另一个是 `create_tool`）。

### `class IngestImageInput(BaseModel)` （第 150 行）
- **作用**：这是 `knowledge.ingest_image` 工具的输入契约模型，用 Pydantic 描述「调用这个工具时允许传什么」。它的存在让工具协议层能在真正读磁盘之前就把参数形状、长度、多余字段都校验一遍，避免非法参数深入到入库流程里。它把「路径」「说明文字」「媒体类型」「拍摄时间」四件事拆成四个字段，并把媒体类型与拍摄时间都设计成可留空的字符串，让调用方可以只给一个 `path` 就完成一次最简入库。该模型的实例由工具协议框架在解析 JSON 参数后构造，再交给 `IngestImageTool.execute` 使用。
- **参数**：类本身不接收运行时参数，它是 Pydantic 模型类；其字段即「构造参数」：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：禁止传入未声明的多余字段（`extra="forbid"`），并开启严格模式（`strict=True`，不做宽松类型强转）。
  - `path: str`：必填，`min_length=1`、`max_length=1000`，说明为「图片文件路径（相对工作区根，或工作区内绝对路径）」。
  - `text: str`：可选，默认 `""`，`max_length=20_000`，说明为「图片说明/上下文文字；留空则用文件名当观察内容」。
  - `mime_type: str`：可选，默认 `""`，`max_length=80`，说明为「图片媒体类型；留空按扩展名推断（jpg/png/webp/gif/bmp）」。
  - `captured_at: str`：可选，默认 `""`，`max_length=80`，说明为「拍摄时间（ISO 8601）；留空表示未知」。
- **返回**：作为类，它没有函数式返回值；实例化后得到一个携带上述四个字段的模型对象，供 `IngestImageTool.execute` 以属性方式读取（`arguments.path`、`arguments.text`、`arguments.mime_type`、`arguments.captured_at`）。
- **内部流程**：没有自定义方法或校验器，全部行为由 Pydantic 的声明式配置驱动。构造时 Pydantic 先检查传入字段是否都在声明范围内（多余字段直接报错），再对每个字段做严格类型与长度约束校验，校验通过后把值挂到实例属性上；`text`、`mime_type`、`captured_at` 未传时填入默认空字符串。注意这里只做长度与类型约束，不做「路径是否在工作区内」这类业务校验，那属于 `execute` 的职责。
- **异常/边界**：字段缺失（如没给 `path`）、类型不符（严格模式下给非字符串）、超长（`path` 超过 1000 字符、`text` 超过 20000 字符、`mime_type` 或 `captured_at` 超过 80 字符）、传入未声明字段，都会由 Pydantic 抛出校验异常（`ValidationError`），在工具协议层被转成参数错误，根本不会进入 `execute`。空字符串对 `path` 是非法的（`min_length=1`），但对另外三个字段是合法默认值。
- **同文件关系**：它被本文件 `IngestImageTool.spec` 的 `input_model` 引用，作为 `IngestImageTool.execute` 的入参类型注解；不调用本文件任何函数。

### `class IngestExtractionReport(BaseModel)` （第 163 行）
- **作用**：这是抽取结果的输出契约模型，把 `ingest_image` 内部那个自由形态的 `report` 字典固化成有类型、有默认值的结构，作为工具输出的一部分返回给调用方。它的价值在于：让「抽取成功了多少实体、多少关系、有没有被取代或撤回、有没有跳过关系、抽取是否失败、向量是不是真的用了图片」这些信息以稳定字段的形式呈现，前端和 Agent 都能可靠读取。它同时承担「诚实降级」的载体：`multimodal_embedding` 为假时上层据此给出文字降级提示。
- **参数**：类本身不接收运行时参数，其字段即构造参数：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：禁止多余字段，严格类型校验。
  - `chunks: int = 0`：文本块数，默认 0（图片入库时实际会被填成 1）。
  - `domains: list[str] = Field(default_factory=list)`：命中的领域列表，用工厂避免可变默认值共享。
  - `entities: int = 0`：抽取到的实体数。
  - `relations: int = 0`：抽取到的关系数。
  - `superseded: int = 0`：被新观察取代的旧观察数。
  - `retracted: int = 0`：被撤回的观察数。
  - `skipped_relations: int = 0`：被跳过的关系数。
  - `errors: list[str] = Field(default_factory=list, description="抽取失败原因；图片本身仍然已入库。")`：错误原因列表。
  - `modality: str = "image"`：模态标识，默认图片。
  - `multimodal_embedding: bool`：必填无默认值，说明为「false 表示当前 embedding 不是 VL 模型，向量只用了文字」。
- **返回**：作为类没有函数式返回值；实例化后得到承载上述十个字段的报告对象，被放进 `IngestImageOutput.extraction`。
- **内部流程**：同样没有自定义方法，行为全部来自 Pydantic 声明。构造时它接收 `ingest_image` 返回字典里 `result["extraction"]` 的键值对（`IngestImageTool.execute` 用 `IngestExtractionReport(**result["extraction"])` 展开），逐个字段做类型与存在性校验；除 `multimodal_embedding` 外全部可省略并取默认值，`domains` 与 `errors` 通过 `default_factory` 生成独立空列表，避免多个实例共享同一个列表对象。因为 `extra="forbid"`，如果传入字典里出现了模型未声明的键就会直接报错。
- **异常/边界**：缺少必填字段 `multimodal_embedding`、字段类型不符、出现未声明字段时抛 Pydantic 校验异常。字段本身对空列表、零值都是合法的；`errors` 非空并不意味着图片没入库，描述里已明确说明「图片本身仍然已入库」。
- **同文件关系**：被本文件 `IngestImageOutput.extraction` 字段引用，并在 `IngestImageTool.execute` 中被实例化；不调用本文件任何函数。

### `class IngestImageOutput(BaseModel)` （第 178 行）
- **作用**：这是 `knowledge.ingest_image` 工具的输出契约模型，规定工具执行成功后必须返回哪几项信息：写入的记忆项 id、模态、抽取报告、以及降级提示。它的作用是让返回值结构化，方便工具协议层序列化给 Agent，也方便前端直接渲染「图片已入库 + 抽取统计 + 诚实提示」这一组信息。它与 `IngestImageInput` 一进一出，共同构成这个工具的自描述接口。
- **参数**：类本身不接收运行时参数，其字段即构造参数：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：禁止多余字段，严格类型校验。
  - `item_id: str`：必填，说明为「写入的感知记忆项 id」。
  - `modality: str = "image"`：模态，默认图片。
  - `extraction: IngestExtractionReport`：必填，嵌套的抽取报告模型。
  - `warning: str = Field(default="", description="非空时表示向量只用了文字说明（诚实降级提示）。")`：降级提示，默认空串。
- **返回**：作为类没有函数式返回值；实例化后得到工具最终输出对象，由 `IngestImageTool.execute` 返回给工具协议层。
- **内部流程**：行为由 Pydantic 声明驱动。构造时要求 `item_id` 与 `extraction` 必须提供，`modality` 缺省为 `"image"`，`warning` 缺省为空串；`extraction` 若是字典，Pydantic 会按 `IngestExtractionReport` 的定义做嵌套校验（在本文件里 `execute` 已经先手工构造好该模型再传入）。所有字段校验通过后实例即为最终输出。
- **异常/边界**：缺少 `item_id` 或 `extraction`、类型不符、传入未声明字段时抛 Pydantic 校验异常。`warning` 为空串是正常状态（表示嵌入后端支持多模态）；非空表示当前不是 VL 模型、向量只用了文字说明。
- **同文件关系**：被本文件 `IngestImageTool.spec` 的 `output_model` 引用，并在 `IngestImageTool.execute` 中构造返回；它引用本文件的 `IngestExtractionReport`；不调用本文件任何函数。

### `class IngestImageTool(BaseTool)` （第 187 行）
- **作用**：这是把图片入库能力接入 Agent 工具协议的工具类，继承 `core.BaseTool`。它通过类属性 `spec`（`ToolSpec`）向运行时声明自己是谁：工具名 `knowledge.ingest_image`、版本 `1.0.0`、输入输出模型、副作用类型 `write`、权限为空、超时 300 秒、非幂等、非并行安全、标签 `("knowledge", "image", "multimodal", "write")`，并附带一段给模型看的使用指引（何时用、path 必须在工作区内、非 VL 嵌入时必须如实告知 warning、抽取失败不回滚图片要在回答里说明）。它把「参数校验 → 路径解析与越界防护 → 文件存在性与大小上限检查 → 元数据组装 → 调用 `ingest_image` → 组装输出模型」这条链路串起来，是 `ingest_image` 纯函数之上的一层协议适配。它还通过可注入的 `pipeline` 支持测试替换，生产环境则延迟构建默认管线。
- **参数**：类本身在实例化时接收参数（见 `__init__`）；类属性 `spec` 中声明的关键约束为：`name="knowledge.ingest_image"`、`side_effect="write"`、`permissions=()`（不需要额外权限）、`timeout_seconds=300.0`、`idempotent=False`（同一张图重复入库会产生新的记忆项，不保证幂等）、`parallel_safe=False`（不应与其他写操作并行）、`tags` 为四个字符串的元组。
- **返回**：类本身无返回值；实例是一个可被工具注册表登记、可被协议层调用 `execute` 的工具对象。
- **内部流程**：类体在导入时构建 `spec` 常量；实例化时只保存 `pipeline`（可能为 `None`）；真正的执行逻辑在 `execute` 中，通过 `pipeline` 属性惰性拿到管线。类本身不直接读写记忆系统，一切副作用都发生在 `execute` 调用链里。
- **异常/边界**：类定义阶段不抛异常；`spec` 是纯声明，字段不合法会在导入时报错。运行期的异常处理全部由 `execute` 与 `ingest_image` 承担。
- **同文件关系**：它引用本文件的 `IngestImageInput`、`IngestExtractionReport`、`IngestImageOutput`、`ingest_image`、`guess_mime_type`；它调用本文件的 `ingest_image` 与 `guess_mime_type`，并在内部属性 `pipeline` 中调用模块外的 `build_default_pipeline`；它被本文件的 `create_tool` 实例化。

### `IngestImageTool.__init__(self, pipeline: RAGPipeline | None = None) -> None` （第 211 行）
- **作用**：这是 `IngestImageTool` 的构造方法，职责极其克制：只把外部传入的管线对象（或 `None`）存到实例私有属性 `self._pipeline` 上，不做任何校验、不做任何 IO、不构建默认管线。这样设计是为了让调用方可以注入一个现成的 `RAGPipeline`（例如测试里用假管理器、或在 Web 应用里复用全局管线），而把「没有注入时怎么拿到管线」推迟到真正要用的时候，避免仅仅构造一个工具对象就触发昂贵的管线初始化。它被 `create_tool` 以无参形式调用。
- **参数**：`self` 为实例本身；`pipeline`，类型 `RAGPipeline | None`，默认 `None`，表示要使用的入库管线；传 `None` 表示「暂不指定，等首次访问 `pipeline` 属性时再构建默认管线」。
- **返回**：返回 `None`，仅产生 `self._pipeline` 这一个副作用。
- **内部流程**：一行赋值 `self._pipeline = pipeline`。没有分支、没有循环、没有类型检查，也没有调用父类 `BaseTool` 的初始化逻辑。
- **异常/边界**：无特殊处理。传入任何对象（包括类型不符的对象）都会被原样保存，错误会在后续 `pipeline` 属性访问或 `execute` 使用该对象时以属性错误的形式暴露。
- **同文件关系**：被本文件的 `create_tool` 间接调用（`IngestImageTool()` 使用默认 `pipeline=None`）；不调用本文件任何其他函数。

### `IngestImageTool.pipeline` （第 214 行，`@property`）
- **作用**：这是一个只读属性（getter），把「当前可用的入库管线」暴露给类内部与外部使用，并实现惰性初始化：第一次访问时如果 `self._pipeline` 还是 `None`，就调用 `build_default_pipeline()` 构建一个默认管线并缓存回 `self._pipeline`，之后再次访问直接返回缓存对象。它存在的意义是让工具在「无注入」场景下也能自己拿到生产管线，同时保证默认管线只构建一次，避免每次 `execute` 都重新初始化嵌入后端与仓库。`execute` 通过 `self.pipeline` 读取它。
- **参数**：只有 `self`（属性 getter 不接受额外参数，外部以 `tool.pipeline` 的形式读取，不能赋值）。
- **返回**：返回 `RAGPipeline` 类型的管线对象。若构造时注入了管线则返回注入的那个；否则返回首次访问时构建并缓存的默认管线。
- **内部流程**：判断 `self._pipeline is None`；为真时执行 `self._pipeline = build_default_pipeline()` 完成构建与缓存；最后 `return self._pipeline`。没有锁保护，也没有异常包裹，构建失败会直接向上抛。
- **异常/边界**：无特殊处理。`build_default_pipeline()` 内部的任何失败（缺少配置、依赖不可用等）都会原样抛出，且因为赋值发生在调用之后，构建失败时 `self._pipeline` 仍为 `None`，下一次访问会再次尝试构建。没有针对并发访问的同步措施，理论上两个线程同时首次访问可能各自构建一次。
- **同文件关系**：它调用模块外导入的 `build_default_pipeline`（来自 `._memory`）；被本文件的 `IngestImageTool.execute` 通过 `self.pipeline` 使用；不调用本文件其他函数。

### `IngestImageTool.execute(self, arguments: IngestImageInput) -> IngestImageOutput` （第 220 行）
- **作用**：这是工具的实际执行入口，协议层校验完入参后会调用它。它负责把「工作区内的一个文件路径」变成一次完整的图片入库：先解析并约束路径防止越界访问，再确认文件确实存在、不是目录、大小不超过上传上限，然后推断或校验 MIME 类型，组装包含来源文件名、拍摄时间、参考时间与模态的元数据，读取文件字节并交给 `ingest_image` 完成真正的写入与抽取，最后把返回字典转成 `IngestImageOutput` 模型。它是本文件里唯一直接触碰文件系统的函数，也是把「业务校验」与「协议模型」连接起来的一层。
- **参数**：`self` 为工具实例；`arguments`，类型 `IngestImageInput`，必填，已经通过 Pydantic 校验的入参对象，提供 `path`（必填非空字符串）、`text`（默认空串）、`mime_type`（默认空串）、`captured_at`（默认空串）四个字段。
- **返回**：返回 `IngestImageOutput` 实例，字段为 `item_id`（来自 `ingest_image` 结果的 `item_id`）、`modality`（来自结果的 `modality`，值为 `"image"`）、`extraction`（用 `IngestExtractionReport(**result["extraction"])` 从结果字典的 `extraction` 构造）、`warning`（来自结果的 `warning`，非 VL 嵌入时为诚实提示）。抽取失败不会改变返回值形状，错误只会出现在 `extraction.errors` 里。
- **内部流程**：
  1. 路径解析与越界防护：`path = resolve_path(workspace_root(), arguments.path)`，把用户给的相对路径或工作区内绝对路径解析成真实 `Path`，越界路径会在这里被拒绝（由 `resolve_path` 抛出异常）。
  2. 存在性检查：`if not path.is_file()` 则 `raise LookupError(f"图片文件不存在：{path}")`；这同时排除了目录和不存在两种情况。
  3. 大小上限检查：`size = path.stat().st_size`，若 `size > MAX_UPLOAD_BYTES` 则 `raise ValueError` 并给出「图片超过上限：X 字节 > Y 字节」的明确信息。
  4. MIME 处理：`mime_type = arguments.mime_type.strip() or guess_mime_type(path)`，即用户填了就用用户填的（去空白后），留空则按扩展名推断；随后 `if not mime_type.startswith("image/")` 抛 `ValueError("mime_type must be an image media type")`，拒绝非图片类型。
  5. 元数据组装：构造字典 `metadata`，包含 `source`（`path.name`）、`filename`（`path.name`）、`captured_at`（入参去空白后的字符串，可能为空串）、`reference_time`（`datetime.now(UTC).isoformat()`，即当前 UTC 时间的 ISO 8601 字符串）、`modality`（固定 `"image"`）。
  6. 调用核心函数：`result = ingest_image(self.pipeline, image=path.read_bytes(), text=arguments.text, mime_type=mime_type, metadata=metadata)`；注意 `self.pipeline` 的访问会触发惰性构建默认管线，`path.read_bytes()` 会把整张图片读进内存。
  7. 组装输出：用结果的四个键构造并返回 `IngestImageOutput`，其中 `extraction` 通过展开字典构造 `IngestExtractionReport`。
- **异常/边界**：路径越界（工作区外）由 `resolve_path` 拒绝并抛异常；文件不存在或不是普通文件抛 `LookupError`；超过 `MAX_UPLOAD_BYTES` 抛 `ValueError`；MIME 不是 `image/` 开头抛 `ValueError`；读取字节失败（权限、文件被删等）由 `path.read_bytes()` 抛 `OSError`；惰性构建默认管线失败也会向上抛。注意 `mime_type` 为空且扩展名未知时会被 `guess_mime_type` 兜底成 `image/jpeg`，因此不会因为「推断不出来」而失败。抽取阶段的异常不会传到这里，因为 `ingest_image` 内部已捕获并写进报告。
- **同文件关系**：它调用本文件的 `guess_mime_type`（推断 MIME）与 `ingest_image`（核心入库）；它读取本文件的 `IngestImageInput` 字段、构造 `IngestExtractionReport` 与 `IngestImageOutput`；它使用 `self.pipeline` 属性间接调用模块外的 `build_default_pipeline`，并使用模块外导入的 `resolve_path`、`workspace_root`、`MAX_UPLOAD_BYTES`、`datetime`；它被工具协议层调用，是本文件运行期最外层的一环。

### `create_tool() -> BaseTool` （第 254 行）
- **作用**：这是工具注册用的工厂函数，供工具加载器在发现本模块时调用，返回一个可注册的工具实例。它存在的意义是把「工具类怎么构造」这件事收敛到一个无参函数上，让自动发现机制（通常按模块扫描并调用同名工厂）不需要了解 `IngestImageTool` 的构造细节，也方便测试用同样的入口拿到工具对象。它不做任何配置读取，因此返回的工具会在首次执行时才惰性构建默认管线。
- **参数**：无参数。工厂故意不接受任何配置，保证自动发现时可以无条件调用。
- **返回**：返回 `BaseTool` 类型（实际运行期是 `IngestImageTool` 实例），其中 `self._pipeline` 为 `None`，真正使用的管线在第一次访问 `pipeline` 属性或调用 `execute` 时构建。
- **内部流程**：单步执行 `return IngestImageTool()`，即用默认参数实例化工具类，`__init__` 把 `_pipeline` 设为 `None` 后返回对象。
- **异常/边界**：无特殊处理。构造过程本身不涉及 IO，唯一可能的异常来自 `IngestImageTool` 类体（导入期已执行），函数调用阶段不会因为环境缺配置而失败；缺配置的问题会推迟到首次 `execute` 时暴露。
- **同文件关系**：它调用本文件的 `IngestImageTool` 构造方法；不被本文件其他函数调用，是被外部工具加载器使用的导出入口；它的名字也在本文件末尾的 `__all__` 中声明。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `guess_mime_type` | 按文件扩展名查表推断图片 MIME 类型，未命中时兜底为 `image/jpeg`。 |
| `ingest_image` | 图片入库核心：校验入参、过嵌入锁、写入感知记忆（含图片字节）、视觉抽取并物化进图谱、返回含降级提示的报告。 |
| `IngestImageInput` | 工具入参模型，声明 `path`（必填）与 `text`/`mime_type`/`captured_at`（可选）四个字段及长度约束。 |
| `IngestExtractionReport` | 抽取结果模型，承载块数、领域、实体、关系、取代/撤回/跳过计数、错误列表、模态与多模态嵌入标志。 |
| `IngestImageOutput` | 工具输出模型，返回记忆项 id、模态、嵌套抽取报告与诚实降级提示。 |
| `IngestImageTool` | 图片入库工具类，用 `ToolSpec` 声明协议（写入、非幂等、非并行安全、300 秒超时）并承载执行逻辑。 |
| `IngestImageTool.__init__` | 仅把可注入的管线（或 `None`）保存到 `self._pipeline`，不做校验与 IO。 |
| `IngestImageTool.pipeline` | 只读属性，惰性构建并缓存默认入库管线，供 `execute` 使用。 |
| `IngestImageTool.execute` | 解析并校验工作区内路径、检查文件存在与大小、确定 MIME、组装元数据、调用 `ingest_image` 并转成输出模型。 |
| `create_tool` | 无参工厂，返回一个 `IngestImageTool` 实例供工具自动发现与注册。 |
