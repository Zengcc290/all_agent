# tool/knowledge_stats.py

## 一、这个文件是干什么的

这个文件是知识星云项目里的一个**只读统计工具**，职责非常单一：把「知识库里现在到底存了多少东西」聚合成一组计数返回。具体来说，它把四层记忆系统与真值源（文档库）的规模信息压缩成五个整数：文档数 `documents`、分块总数 `chunks`、已就绪索引的分块数 `chunks_indexed`、语义事实条数 `facts`、以及四层记忆条目总数 `memories_total`。

文件顶部的模块文档明确交代了它的来历：这段统计逻辑原本写在 `web/app.py` 的 `/api/stats` 端点里，现在被**搬运**到本文件，端点依然保留（因为前端仪表盘要用），但端点退化成「调用本工具 + 把结果转成字典」。这样搬运的意义在于：同一份统计逻辑既能被 HTTP 端点复用，也能被 LLM 当作工具直接调用——当用户问「我库里现在有多少内容」时，模型不必先把全部记忆列出来再自己数，而是直接调这个工具拿到现成计数。

文件内部由四块组成：两个 Pydantic 模型（`KnowledgeStatsInput` 输入模型、`KnowledgeStatsOutput` 输出模型）负责约束工具契约；一个模块级纯函数 `knowledge_stats()` 承担真正的聚合计算；一个继承 `BaseTool` 的工具类 `KnowledgeStatsTool` 把函数包装成带 `ToolSpec` 声明的可注册工具；最后是一个工厂函数 `create_tool()` 供注册表实例化。文件开头还定义了模块级常量 `TOOL_ENABLED = True`，作为该工具是否对外启用的开关标志。整个文件不写任何数据、不改任何状态，属于典型的读侧聚合工具。

## 二、函数与类逐条详解

### `KnowledgeStatsInput` （第 23 行）
- **作用**：这是一个 Pydantic 输入模型，用来声明「调用 knowledge.stats 这个工具时，调用方不需要提供任何参数」。它的存在不是因为要校验什么复杂数据，而是为了让工具契约保持统一：项目里所有工具都必须有输入模型，运行时框架据此生成 JSON Schema、校验模型传进来的 `arguments`，并在模型多传字段时报错。因为本工具是纯统计、无查询条件，所以这个模型体内一个业务字段都没有，只放了一条配置来收紧校验行为。任何试图给它塞参数的调用都会被拒绝，这本身就是一种保护——避免模型误以为可以用它按条件过滤统计。
- **参数**：类本身没有 `__init__` 参数（由 Pydantic 自动生成），实例化时若不传任何关键字参数则得到一个合法空实例。
- **返回**：作为类，它返回实例；此处永远是「没有业务字段」的空模型实例。
- **内部流程**：类体只包含一行类属性赋值 `model_config = ConfigDict(extra="forbid", strict=True)`。`extra="forbid"` 表示出现未声明字段时直接报校验错误，而不是静默忽略；`strict=True` 表示启用严格模式，禁止 Pydantic 的宽松类型强制转换。除此之外没有任何字段声明、没有校验器、没有方法。
- **异常/边界**：当调用方传入任何额外字段时，Pydantic 会抛出 `ValidationError`（由 `extra="forbid"` 触发）。类型不匹配时在严格模式下同样抛 `ValidationError`。由于没有字段，正常路径下不会出现缺字段错误。
- **同文件关系**：被 `KnowledgeStatsTool.spec` 通过 `input_model=KnowledgeStatsInput` 引用，作为该工具的输入契约；不被本文件里的任何函数主动调用。它没有调用本文件里的其它函数。

### `KnowledgeStatsOutput` （第 29 行）
- **作用**：这是本工具的输出契约模型，定义了统计结果的五个计数字段及其约束与语义说明。它保证无论底层存储是什么形态，工具对外返回的结构永远一致、字段永远存在，从而让 HTTP 端点、LLM 调用方和前端都能稳定地按字段名取值。每个字段都带 `ge=0`（大于等于零）约束和中文 `description`，前者防止出现负计数的荒谬结果，后者既服务于自动生成的 JSON Schema，也直接告诉模型每个数字是什么意思。模型顶部的类文档特别强调「只返回计数，不返回任何内容正文」，这是本工具的核心安全边界。
- **参数**：类本身无 `__init__` 参数；实例化时必须提供全部五个计数字段（`documents`、`chunks`、`chunks_indexed`、`facts`、`memories_total`）。
- **返回**：作为类，返回实例；实例上可读取上述五个整数字段。
- **内部流程**：类体先写 `model_config = ConfigDict(extra="forbid", strict=True)`，与输入模型保持同样的严格策略（多余字段报错、不做宽松类型转换）。随后依次声明五个字段：`documents: int = Field(ge=0, description="真值源里的文档数。")`、`chunks: int = Field(ge=0, ...)`、`chunks_indexed: int = Field(ge=0, description="已标记为 indexed（向量投影就绪）的分块数。")`、`facts: int = Field(ge=0, description="语义记忆里的事实条数（不含备注行）。")`、`memories_total: int = Field(ge=0, description="四层记忆的条目总数，包含已过期条目（用于展示真实存量）。")`。五个字段都没有默认值，因此都属于必填。
- **异常/边界**：缺少任一字段、字段为负数、传入额外字段或类型不严格匹配时，Pydantic 抛 `ValidationError`。注意 `strict=True` 下字符串 `"3"` 不会被自动转成整数 `3`。构造成功后的实例是只读语义上的数据载体（Pydantic 模型默认允许赋值，但本文件不修改它）。
- **同文件关系**：被模块级函数 `knowledge_stats()` 作为返回值类型构造；被 `KnowledgeStatsTool.spec` 通过 `output_model=KnowledgeStatsOutput` 引用；被 `KnowledgeStatsTool.execute()` 标注为返回类型。它不调用本文件里的其它函数。

### `knowledge_stats(manager: Any, repository: Any = None) -> KnowledgeStatsOutput` （第 44 行）
- **作用**：这是本文件真正的业务核心，也是唯一做计算的函数。它把两类来源的规模信息合并成一份统一统计：一类是「真值源」——即文档库（`repository`），提供文档数、分块总数、已索引分块数；另一类是「记忆层」——即 `manager` 管理的四层记忆，提供事实条数和记忆条目总数。函数被设计成对 `repository` 缺省友好：当 `repository` 为 `None` 时（例如底层不是 SQLite 文档库，或者用的是内存库），文档与分块相关计数一律按 0 返回，函数文档里特意注明这一行为与重构前 `/api/stats` 的行为**逐字一致**，属于刻意保持的向后兼容，而不是疏漏。它在 HTTP 端点（`/api/stats`）和 LLM 工具调用两条路径上都会被用到，是本文件被外部依赖的主要入口。
- **参数**：
  - `manager`（`Any`，必填）：记忆管理器，实际期望是 `MemoryManager` 实例。函数会在它上面调用 `document_store.list(include_expired=True)` 来统计记忆条目总数，并把它交给 `fact_items()` 去统计事实条数。类型标注写成 `Any` 而不是 `MemoryManager`，是为了降低耦合、方便测试时传入替身对象。
  - `repository`（`Any`，默认 `None`）：文档库/真值源仓储对象。不为 `None` 时函数会调用它的 `stats()` 方法，并期望该方法返回一个字典，字典里可能有 `documents`、`chunks`、`chunks_indexed` 三个键。为 `None` 时跳过调用，直接用三个 0 兜底。
- **返回**：返回一个 `KnowledgeStatsOutput` 实例，五个字段均为 `int`：`documents`、`chunks`、`chunks_indexed` 来自 `counts` 字典（用 `int()` 强制转换），`facts` 等于 `fact_items(manager)` 返回列表的长度，`memories_total` 等于 `manager.document_store.list(include_expired=True)` 返回列表的长度。函数不返回字典、不返回 None，任何成功路径都返回完整模型。
- **内部流程**：
  1. 先用一个条件表达式计算 `counts`：若 `repository is not None`，调用 `repository.stats()` 取其返回值；否则直接用字面量字典 `{"documents": 0, "chunks": 0, "chunks_indexed": 0}` 作为兜底。
  2. 接着构造 `KnowledgeStatsOutput`，字段逐一填充：三个文档相关计数通过 `counts.get("documents", 0)`、`counts.get("chunks", 0)`、`counts.get("chunks_indexed", 0)` 读取，并各自用 `int(...)` 包一层转换——这意味着即使 `repository.stats()` 返回的是浮点数或数字字符串（在 Pydantic 严格模式下），也能先被规整为 `int` 再进入模型校验。
  3. 第四个字段 `facts` 通过 `len(fact_items(manager))` 计算：`fact_items` 是本文件从 `tool.reconcile` 导入的辅助函数，负责把语义记忆里的事实条目提取成列表，取长度即得事实条数（按输出模型描述，该计数不含备注行，这一过滤逻辑在 `fact_items` 内部完成）。
  4. 第五个字段 `memories_total` 通过 `len(manager.document_store.list(include_expired=True))` 计算：显式传 `include_expired=True`，表示统计包含已过期条目，目的是展示真实存量而不是当前有效量。
  5. 把构造好的模型作为唯一返回值返回。
- **异常/边界**：函数体内没有 `try/except`，因此不做异常兜底。若 `repository` 不为 `None` 但其 `stats()` 方法不存在或抛错，异常会直接向上传播；若 `stats()` 返回的不是支持 `.get()` 的对象（例如返回列表或整数），会抛 `AttributeError`；若 `manager` 为 `None` 或缺 `document_store`，`fact_items(manager)` 或 `manager.document_store.list(...)` 会抛 `AttributeError`/`TypeError`。计数缺键时有 `get(..., 0)` 兜底为 0；计数为负数时不会在这里被拦住，而是在 `KnowledgeStatsOutput` 的 `ge=0` 校验处抛 `ValidationError`。没有任何超时或重试处理。
- **同文件关系**：它调用了本文件内定义的 `KnowledgeStatsOutput`（作为构造与返回类型），并间接使用了同为模块级导入的 `fact_items`。它被本文件里的 `KnowledgeStatsTool.execute()` 调用；自身不调用本文件里的其它函数。

### `KnowledgeStatsTool` （第 64 行）
- **作用**：这是把上面的纯函数包装成「可被 Agent 运行时注册和调度」的工具类，继承自 `core.BaseTool`。它的价值在于声明式契约：类属性 `spec` 用 `ToolSpec` 完整描述了工具名、给模型看的英文说明、版本、输入输出模型、副作用等级、权限、超时、幂等性、并行安全性、标签以及给模型的中文使用指引。运行时会读取这份 spec 来生成函数调用 schema、决定能否并发执行、以及在模型选错工具时给出提示。类文档一句话概括其定位：「只读：统计知识库规模。」它内部还持有一个可选的记忆管理器，并采用懒加载方式在第一次真正需要时才构造默认管理器，避免仅注册工具就产生初始化开销。
- **参数**：类无显式构造参数（由 `__init__` 定义，见下条）。
- **返回**：作为类，实例化后返回 `KnowledgeStatsTool` 实例，该实例具备 `spec`、`manager` 属性和 `execute()` 方法。
- **内部流程**：类体首先定义 `spec = ToolSpec(...)`，其中：`name="knowledge.stats"`（对外暴露的工具名）；`description` 是一段英文说明，讲清「返回真值源的文档/分块计数、已索引分块数、事实数与记忆条目总数，只读、只给计数不给内容，可用于回答『我有多少内容』或确认一次入库是否真的落库，要看内容请用 recall 类工具」；`version="1.0.0"`；`input_model=KnowledgeStatsInput`；`output_model=KnowledgeStatsOutput`；`side_effect="read"`（声明只读副作用）；`permissions=()`（不需要任何额外权限）；`timeout_seconds=30.0`（30 秒超时）；`idempotent=True`（重复调用结果一致）；`parallel_safe=True`（可安全并发）；`tags=("knowledge", "stats", "counts", "read")`；`guidance` 是一段中文指引，强调它只给计数不给内容、回答具体内容要用 `knowledge.hybrid_recall` / `knowledge.multi_recall` / `memory.rag_search`，并且特别说明 `chunks_indexed` 小于 `chunks` 意味着向量投影落后，此时应先用 `knowledge.reconcile` 确认，再决定是否 `knowledge.repair_drift`。之后类体依次定义 `__init__`、`manager` 属性、`execute` 方法（各自详见下文条目）。
- **异常/边界**：类定义本身不执行任何可能失败的操作；`ToolSpec` 若被传入非法参数会在导入/类定义阶段就报错。实例化本身不做校验。具体异常行为见各方法条目。
- **同文件关系**：它引用了本文件里的 `KnowledgeStatsInput`、`KnowledgeStatsOutput`；`execute()` 调用本文件里的 `knowledge_stats()`；它被本文件末尾的 `create_tool()` 实例化。

### `KnowledgeStatsTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 95 行）
- **作用**：构造工具实例，并允许调用方从外部注入一个记忆管理器。之所以把 `manager` 做成可选参数，是为了两件事：一是在生产路径上可以什么都不传，让工具自己懒加载默认管理器；二是在测试或需要复用已有管理器（例如同一个 Web 应用里已经构建好的那个）时，直接注入现成实例，避免重复构建、也便于替换替身。方法体只做一件事——把传入的值（可能是 `None`）原样保存到实例属性 `self._manager` 上，真正的构造推迟到 `manager` 属性被访问时。因此构造这个对象本身是零副作用的、非常轻量。
- **参数**：
  - `self`：实例本身。
  - `manager`（`MemoryManager | None`，默认 `None`）：外部注入的记忆管理器。为 `None` 时表示「暂不提供，等真正要用时再懒加载默认管理器」；传入实例时会被直接采用，且不会被本文件再覆盖。
- **返回**：返回 `None`（构造函数语义，返回新建的实例给调用方）。
- **内部流程**：仅执行 `self._manager = manager` 一条赋值语句。没有校验、没有日志、没有资源申请。
- **异常/边界**：无特殊处理。传入任何类型（包括类型标注之外的对象）都不会在这里报错，非法类型会在后续 `execute()` 真正使用它时暴露。
- **同文件关系**：被 `create_tool()` 以无参形式调用（等价于 `manager=None`）；它设置的状态被 `manager` 属性读取。它不调用本文件里的其它函数。

### `KnowledgeStatsTool.manager` （property，第 98 行）
- **作用**：这是一个只读属性（带 `@property` 装饰器），对外暴露「本工具使用的记忆管理器」，同时承担懒加载职责。第一次访问它时，如果 `self._manager` 还是 `None`，它会就地导入并调用 `build_default_manager()` 构建一个默认管理器并缓存回 `self._manager`；之后再访问就直接返回缓存实例，不会重复构建。这样做的好处是把重量级的记忆系统初始化推迟到真正要执行统计的那一刻，让工具注册、schema 生成、模型看到工具列表这些早期阶段保持廉价。返回类型标注为 `MemoryManager`，表示调用方可以认为拿到的永远是一个可用的管理器，而不是 `None`。
- **参数**：只有 `self`（属性访问形式，无显式调用参数）。
- **返回**：返回 `MemoryManager` 实例。若构造时注入了管理器则返回注入的那个；否则返回本次或此前由 `build_default_manager()` 构建并缓存的那个。
- **内部流程**：
  1. 判断 `self._manager is None`。
  2. 若为 `None`，执行函数内的延迟导入 `from ._memory import build_default_manager`（相对导入，指向同包下的 `_memory` 模块，避免模块级循环导入）。
  3. 调用 `build_default_manager()`，把返回值赋给 `self._manager`。
  4. 最后 `return self._manager`，无论走的是懒加载分支还是缓存分支。
- **异常/边界**：如果 `_memory` 模块不可导入或 `build_default_manager()` 构造失败，异常会直接从属性访问处抛出，没有捕获与降级。若并发地从多个线程首次访问同一实例，理论上可能重复构建（本文件未加锁），但最终缓存的是其中一个实例，不影响结果正确性。无超时处理。
- **同文件关系**：它被本文件里的 `execute()` 通过 `self.manager` 读取，并且在 `execute()` 中还会把同一个 `self.manager` 传给 `repository_for(...)` 与 `knowledge_stats(...)`，保证三处用的是同一个管理器实例。它自身调用的是本文件之外导入的 `build_default_manager`。

### `KnowledgeStatsTool.execute(self, arguments: KnowledgeStatsInput) -> KnowledgeStatsOutput` （第 106 行）
- **作用**：这是工具被运行时调用时真正执行的入口方法，也是整个文件对外行为的落地点。它做的事情非常薄：把「自己持有的管理器」和「从该管理器解析出的文档库仓储」一起交给模块级函数 `knowledge_stats()`，然后把结果原样返回。它体现了这个工具「只读、无参」的特性——虽然形参里接收 `arguments`，但由于输入模型没有任何字段，实际没有任何条件需要解析，因此方法体里完全没有读取 `arguments`。调用它的时机是：LLM 决定调用 `knowledge.stats`，运行时完成输入校验并构造出 `KnowledgeStatsInput` 实例之后。
- **参数**：
  - `self`：工具实例，提供 `manager` 属性。
  - `arguments`（`KnowledgeStatsInput`，必填）：运行时校验过的输入模型实例。由于 `KnowledgeStatsInput` 不含任何业务字段，该参数在实现中是「被接受但未使用」的，它的存在只为满足统一的工具调用签名。
- **返回**：返回 `knowledge_stats()` 的返回值，即 `KnowledgeStatsOutput` 实例，包含五个非负整数计数。
- **内部流程**：
  1. 先访问 `self.manager`，触发（或复用）记忆管理器的懒加载。
  2. 调用 `repository_for(self.manager)`（从 `tool.hybrid_index` 导入），由它根据管理器解析出对应的文档库仓储对象；如果底层不是 SQLite 文档库或使用内存库，这里可能返回 `None`。
  3. 把这两个对象作为位置参数传给 `knowledge_stats(self.manager, repository_for(self.manager))`——注意 `self.manager` 被求值了两次，但因为属性已缓存，两次拿到的是同一实例。
  4. 直接把该函数返回的 `KnowledgeStatsOutput` 作为本方法返回值返回，不做任何再加工。
- **异常/边界**：本方法不捕获任何异常。若管理器构造失败、`repository_for` 抛错、或 `knowledge_stats` 内部的计数超出 `ge=0` 约束导致 `ValidationError`，异常都会向上传播给运行时，由运行时决定如何回报给模型。`repository_for` 返回 `None` 属于被正常处理的边界（统计里文档类计数记 0）。没有任何显式的超时控制，超时由 `ToolSpec.timeout_seconds=30.0` 在运行时层面施加。
- **同文件关系**：它调用本文件里的 `knowledge_stats()` 与 `manager` 属性；它被运行时框架（外部）调用，本文件内没有其它函数调用它。

### `create_tool() -> BaseTool` （第 110 行）
- **作用**：这是工具注册使用的工厂函数，作用是把「如何构造这个工具」这件事收敛到一个零参入口上。工具注册表通常只约定「给我一个工厂，我需要时自己调」，而不关心构造函数签名，因此即便 `KnowledgeStatsTool.__init__` 带有一个可选参数，这里也统一以无参方式实例化，让工具走懒加载管理器的默认路径。需要注入自定义管理器的场景不会走这个工厂，而是直接 `KnowledgeStatsTool(manager=...)`。函数体只有一行，是典型的薄封装。
- **参数**：无参数。
- **返回**：返回 `KnowledgeStatsTool()` 新建实例，声明类型为 `BaseTool`（父类类型），调用方按 `BaseTool` 接口使用即可。
- **内部流程**：唯一一步是 `return KnowledgeStatsTool()`。不传 `manager`，因此新实例的 `self._manager` 为 `None`，记忆管理器将在第一次执行统计时通过 `manager` 属性懒加载构建。
- **异常/边界**：无特殊处理。若 `KnowledgeStatsTool` 的类定义或 `ToolSpec` 构造有问题，异常会在导入阶段而非本函数内出现；本函数自身不抛异常。
- **同文件关系**：它调用本文件里的 `KnowledgeStatsTool` 构造实例；本文件内没有任何函数调用它，它是供外部注册表使用的对外入口。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `KnowledgeStatsInput` | 声明该工具不需要任何调用参数的严格 Pydantic 输入模型（禁止多余字段、禁用宽松类型转换）。 |
| `KnowledgeStatsOutput` | 定义统计结果的五个非负整数字段（文档数、分块数、已索引分块数、事实数、记忆条目总数）并只返回计数不含内容。 |
| `knowledge_stats(manager, repository=None)` | 聚合真值源与记忆层的规模：取 `repository.stats()` 或全 0 兜底，再用 `fact_items()` 与 `document_store.list(include_expired=True)` 的长度补齐事实数与记忆总数。 |
| `KnowledgeStatsTool` | 把上述聚合函数包装成名为 `knowledge.stats` 的只读、幂等、可并发的 `BaseTool`，并以 `ToolSpec` 声明其契约与使用指引。 |
| `KnowledgeStatsTool.__init__(manager=None)` | 轻量构造，仅把可选注入的记忆管理器存到 `self._manager`，其余推迟到首次使用。 |
| `KnowledgeStatsTool.manager` | 只读属性，首次访问时懒加载 `build_default_manager()` 并缓存，保证调用方总能拿到可用的 `MemoryManager`。 |
| `KnowledgeStatsTool.execute(arguments)` | 工具执行入口：用 `self.manager` 与 `repository_for(self.manager)` 调用 `knowledge_stats()` 并原样返回统计结果。 |
| `create_tool()` | 零参工厂，返回无注入、走懒加载路径的 `KnowledgeStatsTool` 实例供注册表使用。 |
