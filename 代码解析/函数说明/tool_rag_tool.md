# tool/rag_tool.py

## 一、这个文件是干什么的

这个文件实现的是 Agent 运行时里的一个内置工具 `memory.rag`，职责非常单一：把「一段文字」或者「工作区里的一个文件」写入 Agent 的记忆系统（记忆库 / 向量库），并在写入过程中完成知识抽取与切分，让这些原始资料变成后续可以被检索的知识条目。

它来源于一次拆分：文件开头的模块文档说明，它原本和只读检索工具合并在 `memory.rag` 一个工具里，拆分之后，只读检索走 `memory.rag_search`，而写入动作（`side_effect="write"`）单独留在本文件中，避免只读检索被迫携带写确认逻辑。

文件里包含的东西很少但结构完整：两个 Pydantic 模型（`RAGToolInput` 输入模型、`RAGToolOutput` 输出模型）、一个工具类 `RAGTool`（继承 `core.BaseTool`，内部持有惰性创建的 `RAGPipeline`），以及一个工厂函数 `create_tool()`；另外还有一个模块级开关常量 `TOOL_ENABLED = True` 和导出列表 `__all__`。

被用到的方式是：工具注册/发现机制导入本模块，读取 `RAGTool.spec` 拿到工具名、描述、输入输出模型、权限、超时等元数据，然后把模型给出的 JSON 参数校验成 `RAGToolInput`，再调用 `RAGTool.execute()` 真正落库。真正做重活的 `RAGPipeline`（切分、抽取、持久化）来自 `memory.rag`，本文件只做「参数校验 + 路径沙箱解析 + 调用管线 + 结果包装」。

安全上有一条重要约束：模型给出的 `source` 不会被直接当成文件路径使用，而是通过 `resolve_path(workspace_root(), source)` 走和 `fs.*` 工具完全相同的沙箱解析，因此无法被诱导读取工作区之外的任意路径。默认管线落库位置是 `MEMORY_DB_PATH`，没有配置时落到项目旁边的 `memory.sqlite3`；如果需要别的后端，可以在构造 `RAGTool` 时注入自定义的 `RAGPipeline`。

---

## 二、函数与类逐条详解

### `RAGToolInput` （第 29 行）

- **作用**：这是 `memory.rag` 工具的输入参数模型，用 Pydantic 定义模型被调用时允许出现的字段、类型、默认值和取值范围。它的存在是为了让「模型自由生成的 JSON」在进入真正的入库逻辑之前先被严格校验一次，避免 `text` 和 `source` 同时给、避免把 `chunk_size` 设成 0 或负数、避免传入未知字段。工具注册时通过 `ToolSpec(input_model=RAGToolInput)` 把它交给运行时，运行时据此生成工具的 JSON Schema 并做入参校验。它本身不执行任何业务逻辑，只是数据载体。
- **参数**：这里没有函数参数，只有四个模型字段。
  - `action: Literal["ingest"]`，默认 `"ingest"`：动作名，用 `Literal` 锁死只能取 `"ingest"` 这一个值，等价于声明「本工具只有入库这一个动作」，其它值会在校验阶段直接报错。
  - `text: str | None`，默认 `None`：要入库的字面文本内容。
  - `source: str | None`，默认 `None`：要入库的文件路径，必须是相对于工作区的路径；字段描述里明确写了「`text` 和 `source` 二选一，不能同时提供，工作区之外的路径会被拒绝」。
  - `chunk_size: int`，默认 `1000`：切块大小，约束为 `ge=1, le=100000`，即最小 1、最大 100000。
  - `overlap: int`，默认 `100`：相邻块之间的重叠长度，约束为 `ge=0`，即不允许为负，没有上界约束。
  - 类级配置 `model_config = ConfigDict(extra="forbid", strict=True)`：`extra="forbid"` 表示出现任何未声明字段就报错，`strict=True` 表示开启严格模式，不做宽松的类型强转。
- **返回**：类本身不返回值；实例化后返回一个 `RAGToolInput` 对象，其字段即上面四项（加上 `action`）。在 `RAGTool.execute()` 中，`arguments.text`、`arguments.source`、`arguments.chunk_size`、`arguments.overlap` 会被逐项读取。
- **内部流程**：类体只做三件事——设置 `model_config`（禁止多余字段、严格模式）、声明 `action` 字段并限定为 `"ingest"`、声明 `text` / `source` / `chunk_size` / `overlap` 四个字段及其默认值与约束。所有校验（类型、范围、枚举、多余字段）由 Pydantic 在构造实例时自动完成，类内部没有任何手写方法或钩子。`Field(...)` 的 `description` 会被导出到工具 schema 里，成为给模型看的字段说明。
- **异常/边界**：字段类型不符、`chunk_size` 不在 `1..100000` 内、`overlap < 0`、`action` 不是 `"ingest"`、出现未声明字段，都会由 Pydantic 抛出校验异常（`pydantic.ValidationError`）。注意「`text` 与 `source` 至少给一个、且不能同时给」这条业务规则**不在**本模型里校验，而是在 `RAGTool.execute()` 里手写判断并抛 `ValueError`；`source` 是否越出工作区也不在这里校验，由 `resolve_path` 负责。因此本模型可以合法地构造出 `text=None, source=None` 的实例。
- **同文件关系**：被 `RAGTool.spec` 通过 `input_model=RAGToolInput` 引用，被 `RAGTool.execute()` 作为 `arguments` 的类型标注与实际入参使用；它本身不调用本文件里的任何东西。

---

### `RAGToolOutput` （第 45 行）

- **作用**：这是 `memory.rag` 工具的输出模型，规定工具执行成功后回给 Agent 运行时的结果结构。入库是一个「写」操作，模型调用后需要知道「写了几条、写了哪些、这次抽取的报告是什么」，这个模型就是把这三类信息固定成机器可读的形状。它同时用于工具 schema 的输出声明（`ToolSpec(output_model=RAGToolOutput)`），让调用方对返回结构有稳定预期。它只承载数据，不含逻辑。
- **参数**：没有函数参数，只有四个模型字段。
  - `action: str`：动作名，无默认值，必填；`execute()` 里固定填 `"ingest"`。
  - `count: int`，默认 `0`：本次入库产生的条目数量，`execute()` 里填 `len(values)`。
  - `items: list[dict[str, Any]]`，默认由 `default_factory=list` 生成空列表：本次入库产生的每个条目的字典形式（`item.to_dict()`）。
  - `report: dict[str, Any]`，默认由 `default_factory=list` 生成空字典：管线暴露的本次入库报告（`self.pipeline.last_ingest_report`）。
  - 类级配置 `model_config = ConfigDict(extra="forbid", strict=True)`：禁止额外字段，严格类型模式。
- **返回**：类本身不返回值；实例化后返回一个 `RAGToolOutput` 对象，作为 `execute()` 的最终返回值交回运行时。三个带默认值的字段都使用 `default_factory`，因此不同实例之间不会共享同一个可变的 list / dict。
- **内部流程**：类体只声明 `model_config` 与四个字段，其中 `items` 和 `report` 用 `Field(default_factory=...)` 提供独立的默认容器。所有实例化与校验由 Pydantic 完成，没有自定义方法、没有 `__init__` 覆写、没有序列化钩子。
- **异常/边界**：构造时若传入类型不符的值（例如 `count` 传字符串）或未声明字段，Pydantic 会抛校验异常。因为 `items` 与 `report` 有默认工厂，只传 `action` 也能成功构造。本文件不负责把该模型转成 JSON 文本，序列化由运行时统一处理。
- **同文件关系**：被 `RAGTool.spec` 通过 `output_model=RAGToolOutput` 引用，并在 `RAGTool.execute()` 末尾被实例化并返回；它本身不调用本文件里的任何函数。

---

### `RAGTool` （第 54 行）

- **作用**：这是 `memory.rag` 工具的实体类，继承自 `core.BaseTool`，是本文件的核心。它通过类属性 `spec` 向运行时声明自己是谁：工具名 `memory.rag`、描述「把文字或工作区文件写入 Agent 记忆，抽取知识供后续检索」、版本 `2.0.0`、输入输出模型、副作用 `write`、所需权限 `memory.write`、超时 30 秒、非幂等、非并行安全、标签 `("memory", "rag", "ingest", "write")`，以及一段中文 `guidance` 提示模型「这是让资料变成可检索知识的主入口，回答检索问题要用 `memory.rag_search` 或 `knowledge.hybrid_recall`，同一资料重复入库会重复抽取，先确认是否已存在」。它把「参数校验 → 路径沙箱解析 → 调用 RAG 管线 → 包装输出」这条链路串起来，本身不做切分、抽取或持久化。运行时（以及 `create_tool()`）实例化它，并把模型传来的参数交给它的 `execute()`。
- **参数**：类没有构造参数（`__init__` 的参数见下一条）；类属性 `spec` 是 `ToolSpec` 实例，其字段即上段列出的那一组元数据，取值都是本文件里写死的常量。
- **返回**：类不是函数，本身没有返回值；它的实例是一个可被工具注册表持有、可被调用的工具对象。真正对外产出结果的是它的 `execute()` 方法，返回 `RAGToolOutput`。
- **内部流程**：类体内先定义类属性 `spec`（一个完整的 `ToolSpec`，包含 `name`、`description`、`version`、`input_model`、`output_model`、`side_effect`、`permissions`、`timeout_seconds`、`idempotent`、`parallel_safe`、`tags`、`guidance`），然后依次定义 `__init__`、只读属性 `pipeline`、方法 `execute`。执行时的路径是：运行时读取 `spec` 做工具发现与 schema 生成 → 用 `RAGToolInput` 校验入参 → 调用 `execute()` → `execute()` 按需访问 `pipeline` 属性拿到管线 → 返回 `RAGToolOutput`。
- **异常/边界**：类定义阶段不抛异常。实例化时不打开数据库（管线惰性创建）。执行期的异常都发生在 `execute()` 和 `pipeline` 属性内部：缺少 `text`/`source`、两者同时提供、路径越界、管线落库失败等，详见各自条目。元数据里声明了 `idempotent=False`，意味着运行时不应把它当幂等操作重试；`parallel_safe=False` 意味着不应并发调用同一实例。
- **同文件关系**：它引用同文件的 `RAGToolInput`、`RAGToolOutput` 作为 `spec` 的输入输出模型；`execute()` 内部调用同文件的 `resolve_path`、`workspace_root`（来自 `._shared`，在本文件被导入使用）以及自己的 `pipeline` 属性；`create_tool()` 会实例化它。它被 `create_tool()` 调用（构造），是 `RAGToolInput` / `RAGToolOutput` 的主要使用方。

---

### `RAGTool.__init__(self, pipeline: RAGPipeline | None = None) -> None` （第 76 行）

- **作用**：构造 `RAGTool` 实例，并把可选的 `RAGPipeline` 存到实例属性 `self._pipeline` 上。它刻意**不做**任何重活：注释里写明「Created lazily so importing/discovering the tool never opens SQLite」——工具被导入、被工具注册表发现枚举时不会去连 SQLite 或打开记忆库，只有真正执行入库、访问 `pipeline` 属性时才会创建默认管线。这样保证了启动阶段和工具列表展示阶段的开销与副作用都为零，也让单元测试或上层应用可以注入自己的假管线或别的后端实现。
- **参数**：
  - `self`：实例本身，隐式参数。
  - `pipeline: RAGPipeline | None`，默认 `None`：外部注入的 RAG 管线。传 `None` 时表示「不注入」，此时实例内部仍保存 `None`，真正需要时再由 `pipeline` 属性惰性创建默认管线；传入一个 `RAGPipeline` 实例时，后续所有入库都走这个注入对象，用于替换默认存储后端或做测试替身。类型标注允许 `None`，没有其它取值范围约束。
- **返回**：`None`。构造函数不返回值，只在实例上写入 `_pipeline` 属性。
- **内部流程**：整个方法体只有一行：`self._pipeline = pipeline`。没有校验、没有日志、没有对传入对象的类型检查、没有建立连接、没有读取配置；`build_default_pipeline` 也**不**在这里调用。
- **异常/边界**：正常情况下不抛异常。传 `None` 是合法且默认的行为（表示稍后惰性创建）。传一个不是 `RAGPipeline` 的对象也不会在这里报错——类型标注只是提示，Python 不做运行时强制，错误会在后续 `execute()` 真正调用其 `ingest` / `ingest_source` 方法时以 `AttributeError` 等形式暴露。不涉及超时或空值特殊处理。
- **同文件关系**：它是 `RAGTool` 的构造入口，被 `create_tool()` 间接调用（`RAGTool()` 不传参），也可能被上层显式带管线调用。它不调用本文件里任何其它函数；它写入的 `self._pipeline` 被同文件的 `pipeline` 属性读取，而 `pipeline` 属性又被 `execute()` 使用。

---

### `RAGTool.pipeline` （第 80 行，`@property`，定义于第 81–84 行）

- **作用**：这是一个只读属性（用 `@property` 装饰），对外暴露一个「一定可用」的 `RAGPipeline` 实例。它实现的是惰性初始化：第一次被访问时，如果 `self._pipeline` 还是 `None`，就调用 `build_default_pipeline()` 创建默认管线（默认落库到 `MEMORY_DB_PATH`，未配置时落到项目旁边的 `memory.sqlite3`），把结果缓存回 `self._pipeline`，然后返回；后续访问直接返回缓存对象，不会重复创建。之所以需要它，是因为构造 `RAGTool` 时不能保证配置已就绪、也不希望工具发现阶段就打开 SQLite，把「创建」推迟到真正要写入的那一刻最安全。`execute()` 就是通过它拿到管线的。
- **参数**：只有隐式参数 `self`。没有显式参数，也不接受任何参数（属性形式，不能像方法那样传参）。
- **返回**：返回 `RAGPipeline` 类型对象。第一次访问时返回刚由 `build_default_pipeline()` 建好的新实例；之后返回同一个被缓存的对象。它不会返回 `None`——除非 `build_default_pipeline()` 本身抛出异常。
- **内部流程**：第一步判断 `if self._pipeline is None:`；成立则执行 `self._pipeline = build_default_pipeline()`，把默认管线赋给实例属性；随后无论是否新建，都执行 `return self._pipeline`。整个过程没有锁、没有并发保护、没有日志。
- **异常/边界**：若 `build_default_pipeline()` 因配置错误、依赖缺失或 SQLite 打开失败而抛异常，异常会原样向上传播，并且此时 `self._pipeline` 仍为 `None`，下次访问会再次尝试创建。因为 `self._pipeline` 是「先赋值后返回」，一旦赋值成功就不会重复创建。该属性不是线程安全的：多个线程同时首次访问可能各自创建一次管线（受 `parallel_safe=False` 的声明约束，运行时本就不应并发调用本工具）。注入的管线为假对象时，异常会在真正调用其方法时才出现。
- **同文件关系**：它读取 `__init__` 写入的 `self._pipeline`，内部调用从 `._memory` 导入的 `build_default_pipeline`（本文件外部依赖，但在本文件中被使用），并被同文件的 `RAGTool.execute()` 调用（`self.pipeline.ingest` 与 `self.pipeline.ingest_source`、`self.pipeline.last_ingest_report`）。它不调用 `RAGToolInput` / `RAGToolOutput`。

---

### `RAGTool.execute(self, arguments: RAGToolInput) -> RAGToolOutput` （第 86 行）

- **作用**：这是工具的真正的执行入口，由 Agent 运行时在入参通过 schema 校验后调用。它把一次入库请求完整落地：先做业务级互斥校验（`text` 和 `source` 必须二选一），再按来源分流——纯文本走 `pipeline.ingest(Document(text))`，文件路径先经 `resolve_path(workspace_root(), source)` 做工作区沙箱解析再走 `pipeline.ingest_source(resolved)`，最后把返回的条目列表和管线上的本次入库报告一起打包成 `RAGToolOutput` 返回。它是把「工具层参数」翻译成「记忆层调用」的胶水层，也是防止模型同时给两个来源或都不给的关键闸门。
- **参数**：
  - `self`：实例本身，隐式参数。
  - `arguments: RAGToolInput`，必填、无默认值：已经由 Pydantic 校验过的输入模型实例。其中 `arguments.text` 为字面文本（可为 `None`）、`arguments.source` 为工作区相对路径（可为 `None`）、`arguments.chunk_size` 为切块大小（默认 1000，合法范围 1–100000）、`arguments.overlap` 为块间重叠（默认 100，`>=0`）。注意 `arguments.action` 在本方法中并未被读取使用，因为 `Literal` 已经把它限定为 `"ingest"`，输出里的 `action` 是写死的字符串。
- **返回**：返回 `RAGToolOutput` 实例，包含四个字段：`action="ingest"`（固定字符串）、`count=len(values)`（本次入库产生的条目数）、`items=[item.to_dict() for item in values]`（每个条目转成字典，便于 JSON 序列化）、`report=self.pipeline.last_ingest_report`（管线记录的本次入库报告字典）。当管线返回空列表时，`count` 为 `0`、`items` 为空列表，但仍会正常返回一个输出对象，而不是报错。
- **内部流程**：按代码顺序分四步。第一步，两个互斥检查：`if arguments.text is None and arguments.source is None:` 抛 `ValueError("text or source is required for ingest")`；`if arguments.text is not None and arguments.source is not None:` 抛 `ValueError("provide either text or source, not both")`。第二步分流：`if arguments.text is not None:` 时构造 `Document(arguments.text)` 并调用 `self.pipeline.ingest(...)`，把 `chunk_size=arguments.chunk_size`、`overlap=arguments.overlap` 透传下去，结果存进局部变量 `values`；否则进入 `else` 分支，先写 `assert arguments.source is not None  # narrowed by the check above` 做类型收窄断言（让静态类型检查器满意），再 `resolved = resolve_path(workspace_root(), arguments.source)` 得到沙箱内的绝对/规范路径，然后调用 `self.pipeline.ingest_source(resolved, chunk_size=..., overlap=...)`，同样把结果存进 `values`。第三步（隐式）：`self.pipeline` 的首次访问会触发惰性创建默认管线。第四步：构造并返回 `RAGToolOutput`，其中 `items` 用列表推导逐条调用 `item.to_dict()`，`report` 直接取 `self.pipeline.last_ingest_report`。
- **异常/边界**：
  - `text` 与 `source` 都为 `None` → 抛 `ValueError("text or source is required for ingest")`。
  - `text` 与 `source` 同时非 `None` → 抛 `ValueError("provide either text or source, not both")`。
  - `source` 越出工作区或不存在 → `resolve_path` 抛出相应异常（沙箱拒绝 / 路径解析失败），本方法不做捕获，直接向上传播。
  - `text` 为空字符串（`""`）不算 `None`，会走文本分支并交给 `pipeline.ingest`，是否接受由管线决定。
  - 管线内部抛出的抽取、切分、落库异常（例如 SQLite 打开失败、嵌入模型不可用）不会被吞掉，原样向上抛；`self.pipeline` 若因默认管线创建失败而抛异常，同样向上传播。
  - 超时方面本方法没有自己的超时逻辑，工具级 `timeout_seconds=30.0` 由运行时依据 `spec` 统一控制。
  - 本方法不检查重复入库，重复调用会重复抽取（`spec` 的 `guidance` 中已提醒模型先确认资料是否已存在）。
- **同文件关系**：它调用同文件的 `RAGTool.pipeline` 属性（`self.pipeline.ingest` / `self.pipeline.ingest_source` / `self.pipeline.last_ingest_report`），构造并返回同文件的 `RAGToolOutput`，接收同文件的 `RAGToolInput` 作为参数类型；内部还调用了从 `._shared` 导入的 `resolve_path` 与 `workspace_root`，以及从 `memory.rag` 导入的 `Document`。它本身被运行时（工具调用链）和可能的上层代码调用，在本文件内没有其它函数调用它。

---

### `create_tool() -> BaseTool` （第 113 行）

- **作用**：这是模块级的工厂函数，作用是创建一个 `memory.rag` 工具的实例并返回。工具注册 / 发现机制通常约定每个工具模块提供一个 `create_tool()` 工厂，由它统一产出工具对象，这样注册表不需要了解各工具类各自的构造签名，也方便将来在工厂里加入配置读取、依赖注入或条件启停等逻辑。当前实现是最简形式：直接返回 `RAGTool()`，即使用默认的惰性管线，不注入自定义 `RAGPipeline`。返回类型标注为基类 `BaseTool`，对调用方隐藏具体实现类型。
- **参数**：无参数。
- **返回**：返回 `BaseTool` 类型（运行时实际是 `RAGTool` 实例）。返回值一定非 `None`；构造失败时以异常形式表现，而不是返回 `None`。因为 `RAGTool.__init__` 不做重活，这个函数调用本身不会打开数据库或建立任何连接。
- **内部流程**：函数体只有一行 `return RAGTool()`：调用 `RAGTool` 的构造函数（不传 `pipeline`，于是 `self._pipeline` 被置为 `None`），随即把新实例返回。没有条件分支、没有异常捕获、没有缓存或单例逻辑——每次调用都会得到一个全新的工具实例。
- **异常/边界**：正常路径不抛异常。若 `RAGTool` 类定义或 `ToolSpec` 构造阶段出现问题（例如元数据非法），异常会在此处首次实例化时向上抛出。没有空值、超时或非法参数需要处理（它不接受参数）。
- **同文件关系**：它调用同文件的 `RAGTool` 构造出实例；在本文件内没有被任何其它函数调用，属于对外导出（`__all__` 中列出）的入口。它不调用 `RAGToolInput` / `RAGToolOutput`。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `RAGToolInput` | `memory.rag` 的输入 Pydantic 模型，严格校验 `action` / `text` / `source` / `chunk_size` / `overlap` 五个字段并禁止多余字段。 |
| `RAGToolOutput` | `memory.rag` 的输出 Pydantic 模型，固定返回动作名、入库条目数、条目字典列表与本次入库报告。 |
| `RAGTool` | 工具主体类，通过 `spec` 声明工具元数据，并把参数校验、路径沙箱解析、RAG 管线调用、结果包装串成一条执行链。 |
| `RAGTool.__init__(self, pipeline=None)` | 只把可选注入的管线存进 `self._pipeline`，不创建连接、不做任何重活，保证工具发现阶段无副作用。 |
| `RAGTool.pipeline` | 只读属性，首次访问时用 `build_default_pipeline()` 惰性创建并缓存默认管线，之后返回同一实例。 |
| `RAGTool.execute(self, arguments)` | 校验 `text`/`source` 二选一，按来源调用 `pipeline.ingest` 或 `pipeline.ingest_source` 完成入库，并打包成 `RAGToolOutput` 返回。 |
| `create_tool()` | 模块工厂函数，无参构造并返回一个使用默认惰性管线的 `RAGTool` 实例。 |
