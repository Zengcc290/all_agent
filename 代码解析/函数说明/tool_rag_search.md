# tool/rag_search.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时里「记忆检索」这一类内置工具中的 RAG（检索增强生成）检索工具实现。它把自己注册成一个名为 `memory.rag_search` 的工具，让 Agent 可以在回答问题之前，先去已存储的知识里取回向量命中片段、图事实（实体与关系）、关系路径，或者直接取回一段已经拼装好的上下文文本块，用来给回答提供事实依据。

它是 `memory.rag`（负责「写入 / 摄入知识」的那一半）的只读搭档：文件头注释明确写了这是 ingest 模块的 read-only companion。之所以把 `side_effect` 声明成 `"read"`，是为了让运行时的「写操作需要用户确认」这道闸门不要挡在检索路径上，这样 Agent 可以在不需要用户批准的情况下，把回答建立在已有知识之上。

文件里主要包含四块内容：两个 Pydantic 输入/输出模型（`RAGSearchInput`、`RAGSearchOutput`），一个继承自 `core.BaseTool` 的工具类 `RAGSearchTool`，一个模块级工厂函数 `create_tool()`，以及一个模块级开关常量 `TOOL_ENABLED` 和导出清单 `__all__`。

默认情况下，工具内部使用的管道是惰性创建的：`__init__` 只保存传入的 `pipeline`（可能是 `None`），只有在真正访问 `pipeline` 属性、也就是第一次执行检索时，才会调用 `tool._memory.build_default_pipeline()` 去打开默认数据库。默认管道读取 `MEMORY_DB_PATH` 环境变量指定的路径，如果没有就用项目旁边的 `memory.sqlite3`；也可以在建工具时注入自定义的 `RAGPipeline` 来对接别的后端。

使用方式上，运行时会先按 `ToolSpec` 发现并注册这个工具，模型按 `RAGSearchInput` 的结构给出参数（`action` / `query` / `limit` / `hops`），`execute()` 根据 `action` 走四条不同分支，最后统一包成 `RAGSearchOutput` 返回。整个文件没有任何写库、摄入或修改状态的逻辑，纯粹是查询侧。

---

## 二、函数与类逐条详解

### 模块级常量 `TOOL_ENABLED` （第 23 行）

- **作用**：这是一个模块级布尔开关，值为 `True`，用来告诉工具发现/加载机制「本模块提供的工具是启用的」。运行时（或工具注册表）在扫描 `tool/` 目录下的各个模块时，会读这个变量来决定是否把 `RAGSearchTool` 纳入可用工具集合；把它改成 `False` 就能在不删除代码、不改动别处的情况下整体下线这个工具。它不是函数也不是类，因此没有参数与返回值，但按「一个都不能漏」的要求在此单列说明。
- **参数**：无。
- **返回**：无（模块级变量，值为 `True`）。
- **内部流程**：模块导入时直接绑定为 `True`，不涉及任何计算或条件判断。
- **异常/边界**：无特殊处理。
- **同文件关系**：与 `RAGSearchTool`（第 46 行）配套，是它被外部发现的门槛；文件内部没有代码读取它。

### `class RAGSearchInput(BaseModel)` （第 26 行）

- **作用**：这是 `memory.rag_search` 工具的输入契约模型，基于 Pydantic 的 `BaseModel`。它规定了模型（LLM）在调用这个工具时必须提供的字段、字段类型以及取值范围，运行时会在真正执行 `execute()` 之前用它做参数校验与反序列化，从而把非法调用挡在业务逻辑之外。它存在的意义是让工具接口自描述：工具说明、JSON Schema 校验和类型提示都从这一个类派生，避免手写校验。类里没有定义任何自己的方法，只有三个字段声明和一个模型配置；实例只在运行时内部构造，业务代码通常不会手动 `RAGSearchInput(...)`，但测试或直接调用 `execute()` 时需要自己构造。
- **参数**：无（构造函数由 Pydantic 依据字段生成，接受关键字参数 `action`、`query`、`limit`、`hops`）。
- **返回**：构造时返回 `RAGSearchInput` 实例。
- **内部流程**：类体先设置 `model_config = ConfigDict(extra="forbid", strict=True)`，意思是拒绝任何未声明的多余字段（多传字段直接报错），并且开启严格模式（不做隐式的类型强转，例如不会把字符串 `"5"` 悄悄转成整数 5）。随后声明三个字段：`action` 是 `Literal["retrieve", "context", "graph_retrieve", "graph_context"]`，即只允许这四种取值，没有默认值，属于必填；`query` 是 `str`，通过 `Field(min_length=1)` 约束不能为空字符串，必填；`limit` 是 `int`，默认 5，且 `ge=1, le=50` 限制在 1 到 50 之间；`hops` 是 `int`，默认 1，且 `ge=0, le=3` 限制在 0 到 3 之间。
- **异常/边界**：实例化时如果缺少 `action` 或 `query`、`action` 不在四个字面量之内、`query` 是空串、`limit` 不在 [1, 50]、`hops` 不在 [0, 3]、传入了未声明字段，或者开启了严格模式后类型不匹配（例如 `limit` 传字符串），Pydantic 都会抛出 `ValidationError`。模型自身没有对 `query` 做去空格、长度上限或注入过滤，这些都不在本文件处理范围内。
- **同文件关系**：被 `RAGSearchTool.spec`（第 54 行，作为 `input_model`）引用，也被 `RAGSearchTool.execute`（第 78 行）作为入参类型使用，并在 `__all__` 中导出。

### `class RAGSearchOutput(BaseModel)` （第 35 行）

- **作用**：这是 `memory.rag_search` 工具的输出契约模型，同样基于 Pydantic 的 `BaseModel`。它把四种 `action` 的返回结果统一成同一个结构，使得运行时可以把结果稳定地序列化给模型或写进调用日志，而不用为每种 action 定义不同形状的返回值。它的设计特点是「字段全集 + 默认值」：无论哪条分支，返回的都是同一个类，只是填充的字段不同——`retrieve` 分支主要填 `items` 和 `count`，`graph_retrieve` 分支填 `items`、`entities`、`paths`、`context` 和 `count`，`context` 与 `graph_context` 分支只填 `context` 和 `count`。类里没有自定义方法，字段构造与校验全部由 Pydantic 完成。
- **参数**：无（构造函数由 Pydantic 依据字段生成，接受关键字参数 `action`、`context`、`items`、`count`、`entities`、`paths`）。
- **返回**：构造时返回 `RAGSearchOutput` 实例。
- **内部流程**：类体先设置 `model_config = ConfigDict(extra="forbid", strict=True)`，禁止未声明字段并启用严格模式。然后声明六个字段：`action` 是 `str`，必填无默认值；`context` 是 `str`，默认空字符串 `""`；`items` 是 `list[dict[str, Any]]`，通过 `Field(default_factory=list)` 让每次实例化都拿到一个新的空列表（避免可变默认值共享）；`count` 是 `int`，默认 0；`entities` 是 `list[str]`，同样用 `default_factory=list`；`paths` 是 `list[dict[str, Any]]`，也用 `default_factory=list`。
- **异常/边界**：构造时字段类型不符（例如 `count` 传字符串、`items` 传非列表）、传入了未声明的额外字段，都会触发 Pydantic 的 `ValidationError`。注意 `count` 的语义在不同分支下不一致（`retrieve` 是向量命中条数，`graph_retrieve` 是证据条数，`context` / `graph_context` 是「上下文非空则为 1，否则为 0」），这是本文件有意为之的复用方式，模型侧需要结合 `action` 解读。
- **同文件关系**：被 `RAGSearchTool.spec`（第 55 行，作为 `output_model`）引用，被 `RAGSearchTool.execute`（第 78 行）作为返回类型并在四条分支里实际构造，并在 `__all__` 中导出。

### `class RAGSearchTool(BaseTool)` （第 46 行）

- **作用**：这是本文件的核心类，实现了名为 `memory.rag_search` 的内置 Agent 工具。它继承 `core.BaseTool`，通过类属性 `spec`（一个 `ToolSpec`）向运行时声明工具的名字、说明、版本、输入输出模型、副作用等级、所需权限、超时、幂等性、并行安全性、标签和给模型的使用指引。运行时据此把它暴露给 LLM，并在调用时把参数按 `RAGSearchInput` 校验后交给 `execute()`。它本身不实现任何检索算法，而是把请求转交给 `memory.rag.RAGPipeline` 的对应方法，属于「薄适配层」：负责协议对接、结果整形和管道惰性初始化。类中定义了 `__init__`、`pipeline`（property）和 `execute` 三个成员，检索的具体逻辑全在被委托的 pipeline 里。
- **参数**：无（类定义本身不接收参数；实例化见 `__init__`）。
- **返回**：无（类对象；实例化后返回 `RAGSearchTool` 实例）。
- **内部流程**：类体第一步构造 `ToolSpec`，逐项赋值为：`name="memory.rag_search"`；`description` 说明它可以取回向量匹配、图事实或现成上下文块，并且是只读的、永不摄入；`version="1.0.0"`；`input_model=RAGSearchInput`；`output_model=RAGSearchOutput`；`side_effect="read"`（关键：让检索不被写确认闸门拦截）；`permissions=("memory.read",)`；`timeout_seconds=30.0`；`idempotent=True`；`parallel_safe=True`；`tags=("memory", "rag", "retrieval", "read")`；`guidance` 用中文提示「需要图事实、路径或现成的上下文块来支撑回答时用它；action 决定返回向量命中、图检索结果还是拼好的上下文。纯分块召回用 knowledge.hybrid_recall。回答用户问题前应先用它检索，检索不到再如实说明」。随后定义三个成员方法，由运行时按「构造实例 → 校验输入 → 调用 execute → 校验输出」的顺序使用。
- **异常/边界**：类本身在定义阶段不抛异常；运行期异常主要来自 `pipeline` 惰性创建（例如默认数据库不可用时由 `build_default_pipeline` 抛出）以及被委托的 pipeline 方法。超时由 `ToolSpec.timeout_seconds=30.0` 在运行时层面约束，类内部没有自己的超时或重试逻辑。
- **同文件关系**：引用了 `RAGSearchInput`（第 26 行）与 `RAGSearchOutput`（第 35 行）；被 `create_tool()`（第 120 行）实例化，并在 `__all__` 中导出。

#### `__init__(self, pipeline: RAGPipeline | None = None) -> None` （第 68 行）

- **作用**：构造工具实例，只做一件事——把外部注入的 `RAGPipeline` 记到实例属性 `self._pipeline` 上。之所以把它设计成「可注入 + 可省略」，是因为不同的部署可能想接不同的后端（例如另一个向量库或另一个 SQLite 路径），测试时也可以塞一个假管道进来。注释里写明了这里刻意采用惰性创建：构造工具时不去碰数据库，这样在导入模块、扫描工具目录、生成工具清单这些高频且不需要检索的阶段，永远不会打开 SQLite 连接，避免无谓的 I/O 和文件句柄占用。
- **参数**：
  - `pipeline: RAGPipeline | None`，默认 `None`。传 `None` 表示「不指定，等真正检索时用默认管道」；传入一个 `RAGPipeline` 实例则表示使用该自定义管道，后续所有检索都走它。
- **返回**：`None`（构造函数）。
- **内部流程**：单条语句 `self._pipeline = pipeline`，不做校验、不做类型检查、不建立任何连接、不读环境变量。此时 `self._pipeline` 可能是 `None`。
- **异常/边界**：无特殊处理；传入任何对象都会被原样保存，若类型不对，问题会推迟到 `pipeline` 属性被访问、实际调用其方法时才暴露（`AttributeError` 之类）。
- **同文件关系**：被 `create_tool()`（第 120 行）以无参形式调用；写入的 `self._pipeline` 被同类的 `pipeline` property（第 72 行）读取。

#### `pipeline(self) -> RAGPipeline` （第 72 行，`@property`）

- **作用**：这是一个只读属性，作为工具访问底层 `RAGPipeline` 的唯一入口。它实现了「惰性单例」：第一次被访问时，如果 `self._pipeline` 还是 `None`，就调用 `tool._memory.build_default_pipeline()` 创建一个默认管道并缓存到实例上；之后再访问就直接复用，不会重复创建。这样既保证了构造工具时零 I/O，又保证了同一次会话里检索共享同一个管道对象（连接、索引等资源不会反复重建）。`execute()` 的每一条分支都是通过这个属性拿到管道再调用其方法的。
- **参数**：无（`self` 除外）。
- **返回**：`RAGPipeline` 实例——要么是 `__init__` 注入的那个，要么是 `build_default_pipeline()` 新建并缓存的默认管道。正常情况下不会返回 `None`（除非 `build_default_pipeline()` 返回了 `None`，本文件没有对此做防护）。
- **内部流程**：第一步判断 `if self._pipeline is None:`；成立则执行 `self._pipeline = build_default_pipeline()` 完成创建与缓存；随后无条件 `return self._pipeline`。`build_default_pipeline` 来自同包内的 `tool._memory` 模块，它按文件头注释的说明读取 `MEMORY_DB_PATH`（缺省时用项目旁的 `memory.sqlite3`）来构造管道。
- **异常/边界**：如果默认数据库路径不可用、文件损坏、依赖缺失等，异常会从 `build_default_pipeline()` 抛出并原样向上传播，本文件不做捕获、不回退到其他后端。若 `build_default_pipeline()` 返回 `None`，属性会把 `None` 返回给调用方，随后 `execute()` 里的 `self.pipeline.retrieve(...)` 会抛 `AttributeError`——本文件未做空值检查。该属性没有加锁，多线程并发首次访问时理论上可能重复创建管道（虽然 `parallel_safe=True` 描述的是工具调用语义，不保证此处初始化互斥）。
- **同文件关系**：调用了外部函数 `build_default_pipeline`（本文件之外，来自 `._memory`）；被本文件的 `execute()`（第 78 行）在四个分支中反复访问。

#### `execute(self, arguments: RAGSearchInput) -> RAGSearchOutput` （第 78 行）

- **作用**：这是工具的实际执行入口，运行时在输入校验通过后调用它。它根据 `arguments.action` 把请求分发到 `RAGPipeline` 的四个不同方法之一，并把管道返回的对象「翻译」成统一的 `RAGSearchOutput` 结构：`retrieve` 走向量检索、返回命中片段列表；`graph_retrieve` 走图检索、返回证据、实体、路径以及拼好的上下文；`graph_context` 只返回拼好的图上下文文本；其余情况（即 `context`）返回普通的向量上下文文本。它承担的是协议适配与结果整形职责，本身不实现任何相似度计算或图遍历，因此改动检索算法不需要动这个文件。
- **参数**：
  - `arguments: RAGSearchInput`，必填。是已经通过 Pydantic 校验的输入模型实例，携带 `action`（四种字面量之一）、`query`（非空字符串）、`limit`（1–50，默认 5）、`hops`（0–3，默认 1）。函数内部直接读取这些字段，不重复校验。
- **返回**：`RAGSearchOutput` 实例，四条分支各自返回不同填充：
  - `action == "retrieve"`：`action="retrieve"`、`count=len(values)`、`items` 为每个命中项转成的字典（含 `content`、`score`、`memory_id`、`metadata`），其余字段保持默认（`context=""`、`entities=[]`、`paths=[]`）。
  - `action == "graph_retrieve"`：`action=arguments.action`、`count=len(result.evidence)`、`items` 为 `result.evidence` 中每项的 `to_dict()`、`entities=result.entities`、`paths` 为 `result.paths` 中每项的 `to_dict()`、`context=result.build_context()`。
  - `action == "graph_context"`：`action=arguments.action`、`count=1 if context else 0`、`context=context`。
  - 其他（即 `action == "context"`）：`action="context"`、`count=1 if context else 0`、`context=context`。
- **内部流程**：
  1. 先把 `query = arguments.query` 取到局部变量，四条分支共用。
  2. 判断 `arguments.action == "retrieve"`：调用 `self.pipeline.retrieve(query, limit=arguments.limit)` 得到 `values`；用列表推导把每个 `item` 映射成字典，字段取自 `item.content`、`item.score`、`item.memory_id`，以及 `dict(item.metadata)`（显式复制成普通字典，避免把 pipeline 的元数据对象直接暴露出去）；`count` 用 `len(values)`；然后 `return`。
  3. 判断 `arguments.action == "graph_retrieve"`：调用 `self.pipeline.graph_retrieve(query, limit=arguments.limit, hops=arguments.hops)` 得到 `result`；`items` 用 `[item.to_dict() for item in result.evidence]` 序列化证据，`entities` 直接透传 `result.entities`，`paths` 用 `[path.to_dict() for path in result.paths]` 序列化路径，`context` 通过 `result.build_context()` 让结果对象自己拼装上下文；`count` 取证据条数；然后 `return`。
  4. 判断 `arguments.action == "graph_context"`：调用 `self.pipeline.graph_context(query, limit=arguments.limit, hops=arguments.hops)` 得到 `context` 字符串；返回时用真值判断把 `count` 记为 `1 if context else 0`（空串或 `None` 记 0）。
  5. 以上都不匹配（在输入模型约束下只可能是 `"context"`）：调用 `self.pipeline.build_context(query, limit=arguments.limit)` 得到 `context`，返回 `action="context"`、`count=1 if context else 0`、`context=context`。注意这里 `action` 是硬编码的 `"context"`，而不是回显 `arguments.action`。
- **异常/边界**：函数自身不 try/except，也不做参数再校验——`query` 为空、`limit` 越界等都由 `RAGSearchInput` 在更早阶段拦下。首次调用会触发 `pipeline` 属性惰性创建默认管道，因此「数据库打不开」这类错误会在第一条分支取 `self.pipeline` 时抛出。若 pipeline 返回空列表，`retrieve` 分支返回 `count=0` 且 `items=[]`，不会报错；`context` / `graph_context` 分支对空字符串用 `1 if context else 0` 记 0 并原样返回空串。若 pipeline 返回的对象缺少 `content` / `score` / `memory_id` / `metadata` / `to_dict` / `build_context` 等属性，会抛 `AttributeError`；若 `item.metadata` 不能被 `dict()` 转换，会抛 `TypeError` 或 `ValueError`。没有对 `hops=0` 做特殊分支，直接透传给 pipeline。
- **同文件关系**：通过 `self.pipeline`（第 72 行）取得管道，构造并返回 `RAGSearchOutput`（第 35 行），入参类型为 `RAGSearchInput`（第 26 行）；本文件内没有其他函数调用它，它由运行时（`core.BaseTool` 的调用流程）在工具被调用时触发。

### `create_tool() -> BaseTool` （第 120 行）

- **作用**：这是模块级的工厂函数，是工具发现机制创建本工具实例的标准入口。运行时/注册表通常按约定在工具模块里查找一个无参工厂，用它拿到实例，因此这个函数的存在让注册流程不必知道 `RAGSearchTool` 的构造细节。它刻意不接受任何参数，意味着用这种方式创建出来的工具总是使用默认管道（`pipeline=None`，首次检索时才惰性建默认管道），与文件头「默认管道读 `MEMORY_DB_PATH` 或 `memory.sqlite3`」的说明一致；需要注入自定义后端的调用方则应直接 `RAGSearchTool(pipeline=...)`。
- **参数**：无。
- **返回**：`BaseTool` 类型标注，实际返回 `RAGSearchTool()` 新建的实例（即 `RAGSearchTool` 是 `BaseTool` 的子类，符合标注）。
- **内部流程**：单条语句 `return RAGSearchTool()`。由于 `RAGSearchTool.__init__` 只做属性赋值，这里不会触发任何数据库访问或 I/O，可以安全地在导入/扫描阶段反复调用。
- **异常/边界**：正常情况下不抛异常；只有 `RAGSearchTool` 的类体（`ToolSpec` 构造）在导入时出问题才会失败，那属于模块导入阶段的问题，与本函数无关。它没有做缓存或单例，每次调用都返回一个新实例（各自持有独立的 `_pipeline` 缓存）。
- **同文件关系**：调用本文件的 `RAGSearchTool`（第 46 行，进而触发其 `__init__`，第 68 行）；本文件内没有其他函数调用它，它供外部工具加载器使用，并在 `__all__` 中导出。

### 模块级变量 `__all__` （第 124 行）

- **作用**：声明本模块的公开导出清单，包含四个名字：`"RAGSearchInput"`、`"RAGSearchOutput"`、`"RAGSearchTool"`、`"create_tool"`。它告诉 `from tool.rag_search import *` 以及文档/静态检查工具，哪些是模块的对外接口；同时也隐含说明 `TOOL_ENABLED`、`pipeline` 之类的内部细节不是主要 API。它同样不是函数或类，按要求在此单列。
- **参数**：无。
- **返回**：无（模块级列表常量）。
- **内部流程**：模块导入时把四个字符串字面量组成列表绑定到 `__all__`，不涉及计算。
- **异常/边界**：无特殊处理。
- **同文件关系**：列出本文件的 `RAGSearchInput`（第 26 行）、`RAGSearchOutput`（第 35 行）、`RAGSearchTool`（第 46 行）、`create_tool`（第 120 行）。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `TOOL_ENABLED`（第 23 行） | 模块级开关常量，值为 `True`，供工具发现机制判断是否启用本模块提供的工具。 |
| `RAGSearchInput`（第 26 行） | 工具输入契约模型，严格校验 `action`、`query`、`limit`、`hops` 四个字段的取值范围。 |
| `RAGSearchOutput`（第 35 行） | 工具输出契约模型，用统一结构承载上下文、命中项、数量、实体与路径。 |
| `RAGSearchTool`（第 46 行） | 名为 `memory.rag_search` 的只读检索工具类，用 `ToolSpec` 声明元信息并把请求转发给 RAGPipeline。 |
| `RAGSearchTool.__init__`（第 68 行） | 只保存注入的 pipeline（可为 `None`），保证构造阶段不打开数据库。 |
| `RAGSearchTool.pipeline`（第 72 行） | 只读属性，首次访问时惰性创建并缓存默认 RAGPipeline。 |
| `RAGSearchTool.execute`（第 78 行） | 按 `action` 分发到向量检索、图检索、图上下文或普通上下文，并把结果整形为输出模型。 |
| `create_tool`（第 120 行） | 无参工厂，返回一个使用默认管道的 `RAGSearchTool` 实例供工具加载器注册。 |
| `__all__`（第 124 行） | 模块公开导出清单，声明四个对外名字。 |
