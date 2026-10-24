# tool/memory_add.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时里的一个**内置工具定义文件**，专门负责「往记忆系统里新增一条记忆」这一件事，工具在运行时对外暴露的名字是 `memory.add`。它原本和删除、清空记忆的功能打包在同一个 `memory.manage` 工具里，后来被拆出来，目的是让调用方可以只授予「记住这件事」这一项窄权限，而不必连带把「删除 / 清空记忆」的破坏性权限一起给出去。文件里主要包含四样东西：一个输入模型 `MemoryAddInput`（约束并规范化调用方传来的参数）、一个输出模型 `MemoryAddOutput`（统一返回结构）、一个工具类 `MemoryAddTool`（承载工具元信息 `ToolSpec` 并真正执行写入）、以及一个工厂函数 `create_tool()`（供工具发现/注册机制实例化工具）。写入动作最终委托给 `MemoryManager.add()` 完成，而这个 `MemoryManager` 是**懒加载**的：只有在第一次真正要用到管理器时才去构建默认管理器，因此导入或扫描这个工具文件本身永远不会打开 SQLite 数据库文件。工具被标记为 `side_effect="write"`、`idempotent=False`、`parallel_safe=False`，意味着运行时会要求显式的确认键之后才允许执行，且不保证重复执行幂等、也不允许与其他工具并行跑。默认管理器写入 `MEMORY_DB_PATH` 指定的位置（或项目旁的 `memory.sqlite3`），需要换后端的应用可以自行注入一个 `MemoryManager` 实例。

## 二、函数与类逐条详解

### `class MemoryAddInput(BaseModel)` （第 33 行）

- **作用**：这是 `memory.add` 工具的输入契约模型，基于 pydantic 的 `BaseModel` 定义。它的职责是把「调用方（通常是 LLM 生成的工具调用参数）想要写入的那条记忆」描述清楚，并在进入真正的写入逻辑之前，把参数做一层类型与取值范围的校验和格式规范化。它存在的意义在于：工具是暴露给模型调用的，模型给出的参数经常形态不规范（比如元数据写成 dict 而不是列表、把数字写成字符串、带上多余字段），所以需要一个严格的入口模型来兜底。运行时框架会根据这个模型的字段生成 JSON Schema 交给模型，模型返回的参数再被解析回这个类的实例，然后传给 `MemoryAddTool.execute()` 使用。它本身不执行任何写入，只负责「描述 + 校验 + 规范化」。
- **参数**：类本身没有构造参数，它的字段即参数，全部通过 pydantic 的类属性声明：`content`（`str`，必填，`min_length=1`，含义是要记住的正文文本，不允许空字符串）；`memory_type`（`MemoryScope` 类型，默认值 `"working"`，表示写到哪一层记忆，取值受 `MemoryScope` 这个类型别名约束，说明里明确建议持久化的用户事实与偏好应写入 `episodic` 或 `semantic`，而 `working` 会随会话过期）；`item_id`（`str | None`，默认 `None`，可选的记忆条目 ID，用于指定/覆盖条目标识）；`metadata`（`list[MemoryMetadata] | None`，默认 `None`，可选的元数据列表，每个元素是 `MemoryMetadata`，实际传入时允许是别种形态并由校验器转换）；`importance`（`float`，默认 `0.5`，取值范围 `ge=0, le=1`，即 0 到 1 之间的重要度）；`ttl_seconds`（`float | None`，默认 `None`，存活秒数，约束 `gt=0`，即一旦给出就必须是正数，为 `None` 表示不过期）。此外类级别配置 `model_config = ConfigDict(extra="forbid", strict=True)` 也是行为约束的一部分：`extra="forbid"` 表示多传未知字段直接报错，`strict=True` 表示不做宽松的隐式类型强转。
- **返回**：作为模型类，它「返回」的是被构造出来的实例；成功构造后得到的是一个字段齐备、类型正确的 `MemoryAddInput` 对象，供 `MemoryAddTool.execute()` 读取 `content` / `memory_type` / `metadata` / `importance` / `ttl_seconds` / `item_id` 这些属性。构造失败（缺 `content`、`content` 为空串、`importance` 越界、`ttl_seconds` 非正、出现未知字段等）时不会返回对象，而是由 pydantic 抛出校验异常。
- **内部流程**：字段声明阶段，pydantic 依据 `Field(...)` 收集每个字段的默认值、描述与数值/长度约束，并生成供模型使用的 schema；实例化阶段，pydantic 先按 `model_config` 检查是否有多余字段（`extra="forbid"` 会直接拒绝）以及是否允许非严格类型；随后执行被 `@model_validator(mode="before")` 修饰的 `normalize_metadata`，在字段级校验**之前**对整份原始输入做一次预处理，其中调用本文件导入的 `normalize_metadata_payload(value)` 把元数据部分整理成规范形态（例如把 dict 形式的元数据统一成列表）；预处理完成后才逐字段校验类型与约束，最终产出实例。
- **异常/边界**：`content` 缺失或为空字符串会触发校验错误；`importance` 小于 0 或大于 1 会触发校验错误；`ttl_seconds` 传入 0 或负数（`gt=0` 要求严格大于 0）会触发校验错误；传入未声明的额外字段会因 `extra="forbid"` 报错；`strict=True` 使得类型不匹配时不会做隐式宽松转换。空值方面，`item_id`、`metadata`、`ttl_seconds` 明确允许为 `None`，`metadata` 的规范化由 `normalize_metadata` 负责兜底。本文件内部没有对校验异常做 try/except 捕获，异常会向上冒泡给工具调用框架。
- **同文件关系**：它调用了本文件里的方法 `normalize_metadata`（由 pydantic 在校验阶段自动调用），并把结果字段交给 `MemoryAddTool.execute()` 消费；`normalize_metadata` 又依赖本文件从 `._memory` 导入的 `normalize_metadata_payload` 与字段类型 `MemoryMetadata`。反过来，它被 `MemoryAddTool.spec` 以 `input_model=MemoryAddInput` 的方式引用，并出现在 `__all__` 导出清单中。

### `normalize_metadata(cls, value: Any) -> Any` （第 49 行，`@model_validator(mode="before")` + `@classmethod`）

- **作用**：这是 `MemoryAddInput` 上的一个「前置」模型校验器，作用是在 pydantic 正式开始逐字段校验之前，把调用方传来的整份原始输入先过一遍规范化流程。它存在的意义在于：模型（LLM）在生成工具参数时，`metadata` 字段的形态非常不稳定——可能写成单个对象、可能写成键值对映射、也可能缺省不写——如果不先统一成列表形态，后面的字段校验会因为类型不符而误报失败。因此它扮演的是「输入清洗入口」的角色，把各种宽松写法收敛成 `list[MemoryMetadata]` 能接受的形态。它只在构造 `MemoryAddInput` 时被 pydantic 自动触发，不需要（也不能）由业务代码手动调用。
- **参数**：`cls` 是类方法约定注入的类对象，即 `MemoryAddInput` 本身，用于满足 `@classmethod` 的签名要求；`value`（`Any` 类型）是即将被校验的原始输入，通常是调用方传入的 dict（也可能已是模型实例或其他可映射对象），里面可能包含 `content`、`memory_type`、`metadata` 等键。该参数没有默认值，pydantic 会固定把待校验数据传进来。
- **返回**：返回处理后的输入数据（`Any` 类型），实际就是把原始 `value` 交给 `normalize_metadata_payload` 处理后的结果，再交回 pydantic 继续走后续的字段级校验。它不返回布尔值、不返回 `MemoryAddInput` 实例，只是原样透传「清洗过的输入」。
- **内部流程**：整个方法体只有一步——直接把 `value` 传给本文件从 `._memory` 导入的 `normalize_metadata_payload(value)`，并把它的返回值原样 `return` 出去。没有条件判断、没有循环、没有异常捕获，所有关于「元数据该长什么样」的具体规则都封装在被调用的那个辅助函数里，这里只负责在正确的时机（`mode="before"`，即字段校验之前）挂上这个钩子。
- **异常/边界**：本方法自身不做任何异常处理，也没有对 `None`、空 dict、非法结构做特殊判断；如果 `value` 的形态让 `normalize_metadata_payload` 无法处理，异常会从那里直接向上抛出，最终表现为模型校验失败。由于是 `mode="before"` 校验器，它拿到的是原始数据而非已校验字段，因此对 `None` 之类的边界输入是否安全，完全取决于被调用函数的实现。
- **同文件关系**：它被本文件 `MemoryAddInput` 类定义中通过 `@model_validator(mode="before")` 注册、由 pydantic 在构造该类实例时自动调用；它自己调用了从 `._memory` 导入的 `normalize_metadata_payload`（外部模块函数）。本文件内没有其他函数调用它。

### `class MemoryAddOutput(BaseModel)` （第 55 行）

- **作用**：这是 `memory.add` 工具的输出契约模型，用来规定工具执行完成后返回给调用方（模型 / 上层框架）的结构。它存在的价值是让工具结果有稳定、可预测的形态：无论写入的是哪一层记忆、内容是什么，返回值都固定包含「做了什么动作」「写了几条」「具体是哪些条目」。同时它也承担一层出参校验作用，防止工具实现意外返回结构不符的对象。对于只写一条记忆的 `memory.add` 而言，这个结构保持了与同类记忆工具（比如批量操作类工具）一致的形状，便于上层统一解析。
- **参数**：类本身不接收构造参数，只有三个字段：`action`（`str`，默认 `"add"`，表示本次执行的动作名，本工具固定为新增）；`count`（`int`，默认 `0`，表示实际写入的条目数量，本工具在成功时写 1）；`items`（`list[dict[str, Any]]`，通过 `Field(default_factory=list)` 声明，默认是空列表，存放被写入条目的序列化字典，使用 `default_factory` 而非可变默认值是为了避免多个实例共享同一个列表对象）。类配置 `model_config = ConfigDict(extra="forbid", strict=True)` 同样是行为约束：禁止额外字段、要求严格类型。
- **返回**：它返回的是被构造出的 `MemoryAddOutput` 实例，实例上可读取 `action`、`count`、`items` 三个属性；该实例即 `MemoryAddTool.execute()` 的最终返回值，会被运行时序列化后交回给调用方。校验不通过（比如 `count` 传了字符串）时会抛 pydantic 校验异常而不是返回实例。
- **内部流程**：定义阶段由 pydantic 收集三个字段的默认值与描述；实例化时按 `extra="forbid"` 检查是否有未知字段、按 `strict=True` 检查类型是否严格匹配，然后为未传的字段填默认值——`action` 填 `"add"`、`count` 填 `0`、`items` 调用 `default_factory` 生成一个新的空列表。本类没有定义任何自定义校验器或方法，所有行为都来自 pydantic 基类。
- **异常/边界**：传入未知字段会因 `extra="forbid"` 报错；字段类型不符合 `str` / `int` / `list[dict[str, Any]]` 时会因严格模式报错；不传字段则一律取默认值，因此 `items` 至少是空列表而不会是 `None`，调用方无需做空值判断。本文件内部没有 try/except 包裹它的构造过程。
- **同文件关系**：它被 `MemoryAddTool.spec` 以 `output_model=MemoryAddOutput` 引用，并在 `MemoryAddTool.execute()` 末尾被实例化返回（传入 `action="add"`、`count=1`、`items=[item.to_dict()]`）；同时出现在 `__all__` 导出清单中。它不调用本文件里的任何函数。

### `class MemoryAddTool(BaseTool)` （第 63 行）

- **作用**：这是本文件的核心工具类，继承自框架基类 `BaseTool`，代表运行时可被发现、可被模型调用的 `memory.add` 工具。它把三件事捆在一起：一是通过类属性 `spec`（`ToolSpec`）声明工具的元信息——名称 `memory.add`、描述、版本 `1.0.0`、输入输出模型、副作用类型、所需权限、超时、幂等性、并行安全性、标签与给模型的使用指引；二是持有并懒加载一个 `MemoryManager` 实例作为真正干活的后端；三是通过 `execute()` 把经过校验的输入翻译成一次 `MemoryManager.add()` 调用并包装成输出模型。它被标记为 `side_effect="write"`、`idempotent=False`、`parallel_safe=False`、`permissions=("memory.write",)`、`timeout_seconds=10.0`，因此运行时会在执行前要求显式确认、授予 `memory.write` 权限，并施加 10 秒超时；`guidance` 字段还用中文提示模型：只有用户明确要求「记住这件事」时才写入（默认 `episodic`），它只写内容不抽知识，需要抽取实体与关系应改用 `memory.rag`，需要断言结构化事实应改用 `knowledge.add_fact`，并且写入前应先把用户的话整理成一句自洽的陈述。
- **参数**：类本身在定义时不需要参数；它的行为参数来自类属性 `spec` 中的各项设置，以及构造时的可选依赖注入（见 `__init__`）。`spec` 的关键取值：`name="memory.add"`、`version="1.0.0"`、`input_model=MemoryAddInput`、`output_model=MemoryAddOutput`、`side_effect="write"`、`permissions=("memory.write",)`、`timeout_seconds=10.0`、`idempotent=False`、`parallel_safe=False`、`tags=("memory", "storage", "write")`。
- **返回**：类本身作为可调用对象被实例化后使用；实例化由 `create_tool()` 完成，返回类型标注为 `BaseTool`。调用其实例的 `execute()` 会返回 `MemoryAddOutput`。
- **内部流程**：类定义时先构造 `ToolSpec` 对象并绑定为类属性 `spec`，这一步只做静态元信息登记，不产生任何 I/O；随后定义 `__init__` 用于接收可选的 `manager` 依赖；定义 `manager` 属性用于按需构建默认管理器；定义 `execute` 用于执行写入。运行时框架读取 `spec` 生成对外工具声明、检查权限与确认要求，然后在用户确认后把参数校验为 `MemoryAddInput` 并调用 `execute()`。
- **异常/边界**：类本身不抛异常；异常发生在它的方法与 `ToolSpec` 构造阶段——若 `ToolSpec` 参数不合法会在类定义（导入模块）时即报错。权限缺失、缺少确认键、执行超过 10 秒等情况由运行时框架依据 `spec` 中的声明处理，本类内部没有额外兜底逻辑。
- **同文件关系**：它引用本文件的 `MemoryAddInput` 与 `MemoryAddOutput` 作为输入输出模型；它的 `execute()` 调用了本文件的 `manager` 属性以及从 `._memory` 导入的 `metadata_dict`；它被本文件的 `create_tool()` 实例化；`MemoryAddTool` 本身出现在 `__all__` 导出清单中。

### `__init__(self, manager: MemoryManager | None = None) -> None` （第 82 行）

- **作用**：这是 `MemoryAddTool` 的构造函数，唯一职责是把「记忆管理器」这个依赖保存到实例上，并刻意保持懒加载语义——构造时不创建数据库连接、不打开 SQLite、不做任何 I/O，只把传进来的对象（或 `None`）记下来。这样做的意义是：工具在框架启动时会被导入、扫描、注册，如果构造函数里就去建立存储后端，那么仅仅是「发现工具」这个动作就会产生副作用（打开数据库文件），既慢又可能出错；把真正的构建推迟到第一次使用，可以让工具发现过程完全无副作用。同时，通过允许外部传入 `manager`，需要换后端（例如使用别的数据库或测试替身）的应用可以注入自己的 `MemoryManager`。
- **参数**：`self` 是实例本身；`manager`（`MemoryManager | None`，默认 `None`）是可选的记忆管理器实例。传入一个现成的 `MemoryManager` 时，工具将直接使用它；传 `None` 时表示不注入，后续由 `manager` 属性按需调用 `build_default_manager()` 创建默认实现。该参数没有其他取值约束，也不做类型运行时校验。
- **返回**：返回 `None`（标注为 `-> None`），只产生副作用——在实例上设置 `_manager` 属性。
- **内部流程**：方法体只有一行注释加一行赋值：注释说明「懒创建，使导入/发现工具永不打开 SQLite」，赋值语句 `self._manager = manager` 把参数原样存到私有属性 `_manager` 上。没有条件分支、没有循环、不调用任何函数，也不做默认值的即时求值——默认后端留待 `manager` 属性去构建。
- **异常/边界**：无特殊处理。传入 `None` 是正常且预期的用法；传入任意对象也不会在此报错（类型错误会在真正调用 `manager.add()` 时暴露）。不做参数校验、不抛自定义异常。
- **同文件关系**：它被本文件的 `create_tool()` 间接调用（`MemoryAddTool()` 无参实例化，因此 `_manager` 为 `None`）；它设置的 `_manager` 被本文件的 `manager` 属性读取并在其为 `None` 时替换为默认管理器，进而被 `execute()` 使用。

### `manager(self) -> MemoryManager` （第 86 行，`@property`）

- **作用**：这是一个只读属性（由 `@property` 装饰），作用是向外提供「当前可用的记忆管理器」，并在第一次被访问时把默认管理器补上，从而实现懒初始化。它的存在解决了 `__init__` 刻意不建后端留下的空缺：任何需要真正操作记忆的代码（本文件中就是 `execute()`）只要写 `self.manager` 就能拿到一个可用的 `MemoryManager`，而不必关心它是被注入的还是默认构建的。由于是属性而非普通方法，调用方按属性方式访问，语义上更像「读取一个已经准备好的依赖」。
- **参数**：只有 `self`（实例本身）。不接受其他参数，属性访问时也不需要传参。
- **返回**：返回一个 `MemoryManager` 实例（标注 `-> MemoryManager`）。具体来说：如果实例的 `_manager` 已经是非 `None`，就原样返回它（注入的管理器优先）；如果是 `None`，则先调用 `build_default_manager()` 构建默认管理器并赋值给 `self._manager`（缓存下来，后续访问不再重复构建），再返回该实例。任何情况下都不会返回 `None`。
- **内部流程**：第一步判断 `self._manager is None`；若成立，则执行 `self._manager = build_default_manager()` 完成构建与缓存——`build_default_manager` 是本文件从 `._memory` 导入的辅助函数，负责按默认配置（读取 `MEMORY_DB_PATH` 环境变量或退回到项目旁的 `memory.sqlite3`）创建 `MemoryManager`；第二步无条件 `return self._manager`，把（可能是刚构建的）管理器交出去。整个流程没有循环、没有异常捕获。
- **异常/边界**：本方法自身不做异常处理。如果 `build_default_manager()` 在构建默认后端时失败（例如路径不可写、依赖缺失），异常会直接向上抛出并冒泡到调用方（`execute()`，再往上是工具调用框架）。对 `_manager` 为 `None` 与不为 `None` 两种情况都做了明确分支处理，不会出现「返回空管理器」的边界情况；但它不校验注入对象的真实类型。
- **同文件关系**：它读取 `__init__` 设置的 `_manager`，并调用从 `._memory` 导入的 `build_default_manager`（外部模块函数）；它被本文件的 `execute()` 通过 `self.manager` 调用。本文件内没有其他函数调用它。

### `execute(self, arguments: MemoryAddInput) -> MemoryAddOutput` （第 92 行）

- **作用**：这是工具真正的执行入口，把一次已经通过校验的调用翻译成一次实际写入，并把结果包装成统一的输出模型。它的具体工作是：从输入模型中取出正文、记忆层级、元数据、重要度、TTL 与可选条目 ID，调用记忆管理器的 `add()` 完成落库，然后把返回的记忆条目对象序列化成字典，连同动作名与计数一起打包返回。它是整个文件里唯一产生持久化副作用的地方，因此运行时会依据 `spec` 中的 `side_effect="write"` 在调用它之前要求确认。之所以要把 `MemoryType(arguments.memory_type)` 显式转换一次，是因为输入模型里 `memory_type` 用的是工具层面对外的 `MemoryScope` 类型，而存储层需要的是 `MemoryType` 枚举，这一步完成了两侧的类型对齐。
- **参数**：`self` 是工具实例；`arguments`（`MemoryAddInput` 类型，无默认值）是已经由 pydantic 校验并规范化过的调用参数。它内部会被读取的字段包括：`arguments.content`（要记住的正文，必填非空）、`arguments.memory_type`（记忆层级，默认 `"working"`，此处会转成 `MemoryType` 枚举）、`arguments.metadata`（`list[MemoryMetadata] | None`，此处会经 `metadata_dict` 转换）、`arguments.importance`（0 到 1 的浮点重要度）、`arguments.ttl_seconds`（正浮点数或 `None`）、`arguments.item_id`（字符串或 `None`）。
- **返回**：返回一个 `MemoryAddOutput` 实例，构造参数固定为 `action="add"`、`count=1`、`items=[item.to_dict()]`。也就是说，无论输入内容长短、写入哪一层记忆，成功路径下都返回「动作是 add、数量是 1、items 里是刚写入条目的字典表示」。它不返回 `None`，也没有其他分支返回值。
- **内部流程**：第一步通过 `self.manager` 取到记忆管理器（这一步可能触发本文件 `manager` 属性的懒初始化，从而按需构建默认后端）；第二步调用 `self.manager.add(...)` 并逐项传入参数——位置参数传 `arguments.content`，关键字参数依次传 `memory_type=MemoryType(arguments.memory_type)`（把输入层的 scope 值转成存储层枚举）、`metadata=metadata_dict(arguments.metadata)`（把元数据模型列表转成普通字典结构，`None` 也交由该函数处理）、`importance=arguments.importance`、`ttl_seconds=arguments.ttl_seconds`、`item_id=arguments.item_id`；第三步接收 `add()` 返回的条目对象，赋值给局部变量 `item`；第四步构造并返回 `MemoryAddOutput(action="add", count=1, items=[item.to_dict()])`，其中 `item.to_dict()` 把条目序列化成可 JSON 化的字典。整个过程没有条件分支、没有循环、没有异常捕获。
- **异常/边界**：本方法内部没有 try/except，所有异常都会向上冒泡。可能出现的异常来源包括：`MemoryType(arguments.memory_type)` 转换失败（取值不在枚举范围内）、`self.manager` 触发默认后端构建失败（如数据库路径不可写）、`manager.add()` 因存储层错误（写库失败、磁盘问题等）抛错、以及 `item.to_dict()` 在返回对象结构异常时抛错。执行耗时受 `spec.timeout_seconds=10.0` 约束，超时由运行时框架中止。对空值/非法值的处理发生在进入本方法之前的 `MemoryAddInput` 校验阶段（例如 `content` 不允许为空），本方法不再重复校验；`item_id`、`metadata`、`ttl_seconds` 为 `None` 时按原样透传给存储层，由存储层决定语义（如 `ttl_seconds=None` 表示不过期）。
- **同文件关系**：它调用了本文件的 `manager` 属性（进而间接依赖 `__init__` 设置的 `_manager`）以及本文件的 `MemoryAddOutput` 构造；它使用了本文件 `MemoryAddInput` 实例作为参数；它依赖从 `._memory` 导入的 `metadata_dict`，并依赖从 `memory` 导入的 `MemoryType`。本文件内没有其他函数调用它——它由运行时框架在工具被调用时执行。

### `create_tool() -> BaseTool` （第 104 行）

- **作用**：这是本文件对外暴露的工厂函数，供工具发现/注册机制在加载该模块时调用，用来产出一个可以注册进运行时的工具实例。它存在的意义是给框架一个稳定、统一、无参数的构造入口：框架不需要知道工具类叫什么、构造签名是什么，只要按约定调用模块里的 `create_tool()` 就能拿到 `BaseTool` 实例。由于 `MemoryAddTool.__init__` 的 `manager` 参数有默认值 `None`，这里无需传入任何依赖即可构造，默认后端会在真正执行写入时才被懒加载创建。与模块级常量 `TOOL_ENABLED = True` 配合，构成了「这个模块是否启用、以及如何实例化工具」的标准约定。
- **参数**：无参数（签名里只有返回标注 `-> BaseTool`）。它不接受配置、不接受依赖注入；需要自定义 `MemoryManager` 的调用方应绕过本函数，直接实例化 `MemoryAddTool(manager=...)`。
- **返回**：返回一个新构造的 `MemoryAddTool` 实例，静态类型标注为基类 `BaseTool`。每次调用都会返回一个**新的**工具对象（不是单例，也没有缓存），因此多次调用之间不共享 `_manager` 状态；只有在外部自行注入同一个管理器时才会共享后端。
- **内部流程**：函数体只有一行 `return MemoryAddTool()`——以无参方式实例化工具类，触发本文件 `MemoryAddTool.__init__`，把 `_manager` 设为 `None`，随后把实例返回给调用方。没有条件判断、循环、异常捕获，也没有任何 I/O。
- **异常/边界**：无特殊处理。在正常条件下不会抛异常（构造过程只做一次赋值）；若 `MemoryAddTool` 类本身在导入期就因 `ToolSpec` 定义问题而失败，那属于模块导入阶段的错误，与本函数调用无关。不校验调用次数、不做幂等保证。
- **同文件关系**：它调用了本文件的 `MemoryAddTool` 类（进而触发该类的 `__init__`）。本文件内没有其他函数调用它；它被外部工具注册机制调用，且 `create_tool` 本身出现在 `__all__` 导出清单中。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `MemoryAddInput` | `memory.add` 的输入契约模型，声明并严格校验要写入记忆的正文、层级、元数据、重要度、TTL 与可选 ID。 |
| `normalize_metadata` | `MemoryAddInput` 的前置模型校验器，在字段校验前把各种形态的元数据输入交给 `normalize_metadata_payload` 统一规范化。 |
| `MemoryAddOutput` | `memory.add` 的输出契约模型，固定返回动作名、写入条数与条目字典列表。 |
| `MemoryAddTool` | `memory.add` 工具类，用 `ToolSpec` 声明元信息与权限/副作用约束，并持有懒加载的记忆管理器来执行写入。 |
| `__init__` | `MemoryAddTool` 的构造函数，仅把可选注入的管理器存入 `_manager`，保证导入与工具发现不产生任何 I/O。 |
| `manager` | 只读属性，首次访问时用 `build_default_manager()` 懒构建并缓存默认管理器，之后直接返回可用管理器。 |
| `execute` | 工具执行入口，把校验后的参数转成 `MemoryManager.add()` 调用，并把写入结果包装成 `MemoryAddOutput`。 |
| `create_tool` | 无参工厂函数，供框架按约定实例化并返回一个新的 `MemoryAddTool`。 |
