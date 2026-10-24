# tool/update_log.py

## 一、这个文件是干什么的

这个文件实现了一个「只写不读」的项目更新日志工具，是整个 Agent 运行时工具箱中的一个具体工具模块，工具名为 `system.update_log`。它把每次对项目做出的真实修改，以结构化记录的形式追加写入由 `core.update_log.UpdateLogRepository` 提供的 SQLite 审计日志中，并只把新生成的数字 ID 返回给调用方（AI 模型或人）。文件刻意不提供任何读取历史记录的操作，这样历史日志不会占用模型的上下文窗口，同时审计信息仍然完整落库。文件内部主要由四部分组成：三个 Pydantic 数据模型（`UpdateLogFileChange` 描述单个被改动文件、`UpdateLogInput` 描述一次更新的完整入参、`UpdateLogOutput` 描述写库后的返回结果）；一个继承 `core.BaseTool` 的工具类 `UpdateLogTool`，负责声明工具元信息（ToolSpec）并执行写入；一个模块级工厂函数 `create_tool()`，供工具注册表按约定实例化该工具；以及一个模块级开关常量 `TOOL_ENABLED` 和 `__all__` 导出清单。在项目运行过程中，Agent 每次改完代码或配置后调用该工具，工具把参数交给仓储层落库，返回记录 ID、下一个可用 ID、UTC 时间戳、检测到的操作系统名以及写入成功标记。由于该工具声明了写副作用（`side_effect="write"`）、非幂等（`idempotent=False`）、非并行安全（`parallel_safe=False`）且最大并发为 1，运行时会把它当作需要串行化执行的写操作来处理。

## 二、函数与类逐条详解

### `UpdateLogFileChange` （第 19 行）
- **作用**：这是一个 Pydantic 模型类，用来描述「本次更新中某一个被改动的文件」这一最小粒度的记录单元。它存在的意义是把「改了哪个文件、属于哪种变更动作、具体改了什么」三件事绑定在一起，避免在日志里只写一个文件路径而丢失变更语义。它被 `UpdateLogInput.files` 列表复用，因此一次更新可以携带多个文件条目，每个条目结构完全一致。由于模型配置为严格模式且禁止多余字段，任何拼错的字段名或类型不符的值都会在参数校验阶段直接失败，从而保证落库的数据形状稳定。它本身不包含任何业务方法，只承担数据结构与校验的职责。它出现在工具的输入模型中，是调用方必须提供的信息之一。
- **参数**：无构造参数（作为 Pydantic 模型，通过关键字字段实例化）。字段如下：`path`（`str`，必填，长度 1 到 500，含义是项目相对路径的已改动文件路径）；`action`（`str`，必填，长度 1 到 32，含义是变更动作，文档描述取值应为 added、modified、deleted、renamed、generated 之一，但代码层面只做长度约束、未用枚举强制，因此其它短字符串在语法上也能通过校验）；`description`（`str`，必填，长度 1 到 2000，含义是该文件具体改了什么）。所有字段都没有默认值，即全部必填。
- **返回**：类本身不返回值；实例化后得到一个不可变语义的校验后对象（Pydantic v2 模型），其 `.model_dump()` 方法可在后续被调用以转换成普通字典。
- **内部流程**：类定义时先设置 `model_config = ConfigDict(extra="forbid", strict=True)`，声明「禁止额外字段」和「严格类型」两条约束；随后按顺序声明三个 `Field`，每个 `Field` 都带有 `min_length` / `max_length` 长度约束和 `description` 说明文本。实例化时 Pydantic 会逐字段做类型与长度校验，全部通过才生成对象；不通过则抛出校验错误。
- **异常/边界**：字段缺失、类型不是字符串、长度越界（空字符串或超长）以及出现未声明的额外字段时，Pydantic 会抛出 `ValidationError`。对空值（`None`）没有额外兜底，因为 `str` 且无默认值，传入 `None` 会被严格模式判定为类型错误。
- **同文件关系**：不调用本文件中的任何函数；被 `UpdateLogInput` 通过 `files: list[UpdateLogFileChange]` 引用，并在 `UpdateLogTool.execute` 中通过 `item.model_dump()` 被消费。

### `UpdateLogInput` （第 35 行）
- **作用**：这是工具 `system.update_log` 的输入契约模型，定义了一次「项目更新记录」必须包含的全部字段。它的作用是强制调用方（通常是 AI 模型）在写日志时提供完整、可核查的信息，而不是只写一句「改了代码」。它覆盖了审计所需的各个维度：谁改的、属于哪类改动、标题、任务背景、实现细节、新增能力、涉及文件清单、行为影响、验证方式、风险与回滚、后续工作。工具执行时会把它作为 `execute` 的入参类型，运行时框架会依据 `ToolSpec.input_model` 指向它来做参数校验与 JSON Schema 生成。因为每个字段都有最小长度限制，调用方无法用空字符串敷衍过关，这保证了日志的可追溯质量。它把文件清单交给 `UpdateLogFileChange` 列表统一约束，从而在保持单一模型的同时支持一次记录多个文件。它是整个文件里字段最多、约束最密的模型。
- **参数**：无构造参数（Pydantic 模型，按字段关键字实例化）。字段如下：`executor`（`str`，必填，长度 1 到 200，含义是执行本次改动的 AI/模型或人）；`update_type`（`str`，必填，长度 1 到 64，含义是变更类别，如 feature、fix、refactor、config、docs）；`title`（`str`，必填，长度 1 到 300，含义是简短的更新标题）；`task_background`（`str`，必填，长度 1 到 4000，含义是为什么提出该改动、目标是什么）；`update_details`（`str`，必填，长度 1 到 12000，含义是具体实现细节与重要决策）；`added_features`（`str`，必填，长度 1 到 6000，含义是新增能力，无新增时按约定填 `none`）；`files`（`list[UpdateLogFileChange]`，必填，元素个数 1 到 100，含义是所有被新增、修改、删除或重命名的项目文件，至少要有一个）；`behavior_impact`（`str`，必填，长度 1 到 6000，含义是兼容性、API、配置、数据、部署或用户影响说明）；`validation`（`str`，必填，长度 1 到 6000，含义是实际运行过的测试/检查及其真实结果）；`risks`（`str`，必填，长度 1 到 4000，含义是已知风险与回滚说明，无风险时填 `none`）；`follow_up`（`str`，必填，长度 1 到 4000，含义是剩余待办，没有则填 `none`）。全部字段无默认值，即都是必填项。
- **返回**：类本身不返回值；实例化后得到校验通过的对象，供 `UpdateLogTool.execute` 读取各字段。其字段结构同时被运行时用于生成工具的输入 JSON Schema。
- **内部流程**：定义时先声明 `model_config = ConfigDict(extra="forbid", strict=True)`，禁止额外字段并启用严格类型检查；然后逐个声明字段，字符串字段用 `min_length` / `max_length` 约束，`files` 字段用列表形式的 `min_length=1, max_length=100` 约束元素个数，元素类型是 `UpdateLogFileChange`。实例化时 Pydantic 递归校验：先校验标量字段，再逐个校验 `files` 里的每个元素模型；任何一层失败都会整体抛错。
- **异常/边界**：字段缺失、空字符串、超长文本、`files` 为空列表、`files` 超过 100 项、`files` 元素不是合法的 `UpdateLogFileChange`（例如缺少 `path`）、或出现额外字段时，都会抛出 Pydantic `ValidationError`。严格模式下传入 `None` 或类型不符的值（如把数字传给字符串字段）同样报错。代码本身不做截断或兜底填充。
- **同文件关系**：不调用本文件中的任何函数；通过 `files: list[UpdateLogFileChange]` 依赖 `UpdateLogFileChange`；被 `UpdateLogTool.spec` 作为 `input_model` 引用，并被 `UpdateLogTool.execute` 作为参数类型注解与 `isinstance` 校验目标。

### `UpdateLogOutput` （第 91 行）
- **作用**：这是工具 `system.update_log` 的输出契约模型，定义了写入成功后返回给调用方的最小信息集合。它刻意只包含 ID、下一个 ID、时间戳、系统名和成功标记这几项，不含任何历史日志正文，这正是本文件「只写不读、历史不占模型上下文」设计意图的落地方式。调用方拿到 `update_id` 后可以在后续对话中引用这条记录，拿到 `next_update_id` 后可以知道下一次写入应使用的编号。`system_name` 记录了实际写入时检测到的操作系统名，便于审计时区分运行环境。`recorded` 字段用 `= True` 直接写成默认值，语义上表示该条记录已成功落库，因为工具只有在仓储写入成功返回后才会构造这个对象。它由 `execute` 在拿到仓储返回的字典后通过 `**result` 解包构造。
- **参数**：无构造参数（Pydantic 模型，按字段关键字实例化）。字段如下：`update_id`（`int`，必填，`ge=1`，含义是本次更新被分配到的 ID）；`next_update_id`（`int`，必填，`ge=1`，含义是下一次写入时应展示/使用的 ID）；`timestamp`（`str`，必填，长度至少 1，含义是实际写入的 UTC 时间戳）；`system_name`（`str`，必填，长度至少 1，含义是检测到的操作系统名）；`recorded`（`bool`，默认 `True`，含义是写入成功标记）。
- **返回**：类本身不返回值；实例化后得到输出对象，运行时框架会把它序列化后回给调用方。
- **内部流程**：定义时声明 `model_config = ConfigDict(extra="forbid", strict=True)`；然后声明 `update_id`、`next_update_id` 两个带 `ge=1` 下界约束的整数字段，`timestamp`、`system_name` 两个带最小长度约束的字符串字段，以及带默认值的 `recorded` 布尔字段。构造时 Pydantic 校验各字段类型与约束；注意由于 `extra="forbid"`，`**result` 中若含有本模型未声明的键会直接抛错。
- **异常/边界**：`update_id` 或 `next_update_id` 小于 1、类型不是整数（严格模式下不接受字符串数字）、`timestamp` / `system_name` 为空字符串、传入额外字段、或缺少必填字段时，都会抛出 Pydantic `ValidationError`。`recorded` 不传时自动为 `True`。
- **同文件关系**：不调用本文件中的任何函数；被 `UpdateLogTool.spec` 作为 `output_model` 引用，并在 `UpdateLogTool.execute` 末尾以 `UpdateLogOutput(**result)` 的形式被构造和返回。

### `UpdateLogTool` （第 101 行）
- **作用**：这是本文件的核心类，继承自 `core.BaseTool`，实现了名为 `system.update_log` 的具体工具。它把「一次结构化更新写入」封装成一个运行时可直接调度的能力：类属性 `spec` 声明工具的元信息与治理策略，`__init__` 负责准备仓储依赖，`execute` 负责真正的落库动作。它被声明为强制工具（描述里写明 MANDATORY after every project modification），即项目每次真实修改后都应调用一次。它被设计成有写副作用、非幂等、非并行安全、最大并发为 1，因此运行时不会把它并发调度，避免日志写入乱序。它不申请任何权限门（`permissions=()`），因为它被定位为公开的项目工具，靠写确认来保护副作用。超时被设为 10 秒，属于轻量快速的本地 SQLite 写入操作。整个文件对外暴露的主要就是这个类以及它的工厂函数。
- **参数**：类属性 `spec` 为 `ToolSpec` 实例，构造参数包括：`name="system.update_log"`；`description` 为一段英文说明，强调「每次项目修改后必须追加一条完整、真实的结构化记录到 SQLite 审计日志，本工具只写记录并返回其 ID，从不读取历史条目」；`version="1.0"`；`input_model=UpdateLogInput`；`output_model=UpdateLogOutput`；`side_effect="write"`；`permissions=()`（空元组，即不施加权限门）；`timeout_seconds=10.0`；`idempotent=False`；`parallel_safe=False`；`max_concurrency=1`；`tags=("update-log", "audit", "project", "mandatory")`；`guidance` 为一段中文提示，说明每次真实修改项目后必须调用一次、要写完整可核查的记录（改了哪些文件、为什么改、怎么验证），它只写记录不读历史，并且不要用它记录「打算做什么」，因为它记的是已经发生的事。类实例化参数见 `__init__`。
- **返回**：类本身不返回值；实例化后得到一个可被运行时注册和调用的工具对象，其 `execute` 返回 `UpdateLogOutput`。
- **内部流程**：类体先定义类属性 `spec`（在导入时即完成 `ToolSpec` 构造，包含上述所有治理字段），然后定义 `__init__` 与 `execute` 两个方法。运行时通常会读取该类的 `spec` 来注册工具、生成参数 schema、决定调度与超时策略，然后在需要时调用 `execute` 完成写入。类定义本身不执行任何 I/O。
- **异常/边界**：类体在导入时构造 `ToolSpec`，若 `ToolSpec` 对参数有额外约束且不满足，导入阶段即会报错；本文件内部对这类情况没有 try/except 兜底。`permissions=()` 表示不做权限校验，这是有意的设计而非疏漏。
- **同文件关系**：它引用本文件中的 `UpdateLogInput`、`UpdateLogOutput` 两个模型；其 `execute` 方法被 `create_tool` 返回的实例在运行时调用；`__init__` 默认构造 `UpdateLogRepository`（来自 `core.update_log`，不属于本文件）。

### `UpdateLogTool.__init__(self, repository: UpdateLogRepository | None = None) -> None` （第 128 行）
- **作用**：这是 `UpdateLogTool` 的构造函数，唯一的职责是准备数据仓储依赖。它支持依赖注入：如果调用方传入了一个现成的 `UpdateLogRepository`，就直接使用它；如果传入 `None`（默认情况），就现场创建一个新的仓储实例。这样设计的好处是测试时可以把仓储替换成假实现或指向临时数据库，而生产运行时的默认路径则无需任何参数即可工作。构造函数里没有任何 I/O，`UpdateLogRepository()` 的构造本身由 `core.update_log` 决定（本文件看不到其内部行为）。该依赖随后被 `execute` 通过 `self.repository.append(...)` 使用。由于它只做一次赋值，开销极小，可以安全地在工具注册阶段被调用。
- **参数**：`self` 为工具实例本身；`repository`（`UpdateLogRepository | None`，默认 `None`）为可选的数据仓储对象，传 `None` 时表示「使用默认仓储」，传入实例时表示「使用外部注入的仓储」。类型上要求是 `UpdateLogRepository` 或其兼容对象，但代码未做 `isinstance` 检查，传入任意具有 `append` 方法的鸭子类型对象在运行时同样可用。
- **返回**：无返回值（`None`），仅设置实例属性 `self.repository`。
- **内部流程**：第一步判断 `repository is not None`；条件为真时把传入对象赋给 `self.repository`；条件为假时执行 `UpdateLogRepository()` 创建默认仓储并赋给 `self.repository`。整个函数只有这一条条件表达式赋值语句。
- **异常/边界**：本函数自身不抛异常；若 `UpdateLogRepository()` 在构造时失败（例如数据库文件无法创建或路径不可写），异常会向上传播，不会被吞掉。对 `None` 的处理是「走默认分支」，这是显式设计的边界行为。未对传入对象做类型校验，因此传入不兼容对象时错误会延迟到 `execute` 调用 `append` 时才暴露。
- **同文件关系**：调用了外部类 `UpdateLogRepository`（来自 `core.update_log`），不调用本文件中的其它函数；它被 `UpdateLogTool` 的实例化过程（包括 `create_tool`）调用，赋值的 `self.repository` 被 `UpdateLogTool.execute` 使用。

### `UpdateLogTool.execute(self, arguments: UpdateLogInput) -> UpdateLogOutput` （第 131 行）
- **作用**：这是工具真正干活的方法，负责把一次更新写入 SQLite 审计日志并返回紧凑的结果对象。它首先做一次防御性的类型检查，确保传入的是 `UpdateLogInput` 实例而不是裸字典或别的对象，避免后续属性访问出错。随后它把输入模型里的每个字段显式拆开，逐个作为关键字参数传给仓储的 `append` 方法，其中文件列表被转换成「字典列表」后再传下去，这样可以跨层传递而不用让仓储层依赖 Pydantic 模型。它还会在调用时通过 `platform.system()` 读取当前操作系统名一并写入，保证日志里记录的是真实运行环境。仓储返回的结果字典被直接解包构造为 `UpdateLogOutput`，从而由 Pydantic 完成输出侧的校验与形状收敛。整个方法不读取任何历史记录，只做一次写入。
- **参数**：`self` 为工具实例，提供 `self.repository`；`arguments`（`UpdateLogInput`）为本次更新的完整结构化参数，包含 `executor`、`update_type`、`title`、`task_background`、`update_details`、`added_features`、`files`（元素为 `UpdateLogFileChange`）、`behavior_impact`、`validation`、`risks`、`follow_up` 共 11 个必填字段。该参数必须已经是校验通过的模型实例。
- **返回**：返回一个 `UpdateLogOutput` 实例，其中包含 `update_id`、`next_update_id`、`timestamp`、`system_name` 与 `recorded` 字段，字段值来自仓储 `append` 返回的字典。只有在仓储成功返回后才会构造并返回该对象。
- **内部流程**：第一步用 `isinstance(arguments, UpdateLogInput)` 做类型校验，不是该类型就立刻 `raise TypeError("arguments must be an UpdateLogInput instance")`；第二步调用 `self.repository.append(...)`，把 `executor`、`update_type`、`title`、`task_background`、`update_details`、`added_features` 原样传入，把 `files` 用列表推导 `[item.model_dump() for item in arguments.files]` 转换为字典列表后传入，再传入 `behavior_impact`、`validation`、`risks`、`follow_up`，最后把 `platform.system()` 的返回值作为 `system_name` 传入；第三步把 `append` 返回的字典用 `**result` 解包，构造 `UpdateLogOutput` 并作为返回值返回。
- **异常/边界**：传入非 `UpdateLogInput` 实例时抛 `TypeError`；仓储 `append` 可能因数据库错误、磁盘满、表结构缺失等原因抛出异常，本方法不做捕获，异常直接向上传播（此时不会构造输出对象，也不会返回 `recorded=True`）；`UpdateLogOutput(**result)` 阶段若 `result` 缺少必需键或含有 `UpdateLogOutput` 未声明的额外键，会抛 Pydantic `ValidationError`。时间戳不由本方法生成，而是由仓储负责写入并回传；操作系统名在每次调用时实时读取，不缓存。
- **同文件关系**：调用了本文件中的 `UpdateLogInput`（用于 `isinstance` 判断）、`UpdateLogFileChange` 的实例方法 `model_dump()`（通过 `arguments.files` 的元素）以及 `UpdateLogOutput`（用于构造返回值）；同时依赖外部库函数 `platform.system()` 和外部类 `UpdateLogRepository.append`（来自 `core.update_log`）。它被 `create_tool` 返回的 `UpdateLogTool` 实例在运行时调用，本文件内部没有其它函数调用它。

### `create_tool() -> BaseTool` （第 151 行）
- **作用**：这是模块级的工具工厂函数，按项目约定向工具注册表提供该工具的实例。它的存在让注册流程可以统一用「调用 `create_tool()`」的方式获取工具对象，而不需要知道具体类名或构造参数。它内部固定以无参方式实例化 `UpdateLogTool`，因此会走 `__init__` 的默认分支，自动创建一个默认的 `UpdateLogRepository`。返回类型注解写成基类 `BaseTool`，表示调用方只需按基类接口（`spec`、`execute`）使用它。由于工具没有任何必需的构造配置，这个工厂函数极其简单，只负责「造一个能用默认配置运行的工具」。它被列在文件末尾的 `__all__` 中对外导出。
- **参数**：无参数。
- **返回**：返回一个 `BaseTool` 类型的对象，实际运行时类型为 `UpdateLogTool` 实例，其 `spec.name` 为 `system.update_log`，可直接被运行时注册与调度。
- **内部流程**：直接执行 `return UpdateLogTool()`：调用 `UpdateLogTool` 的构造函数（`__init__` 收到 `repository=None`），在构造函数内部创建默认仓储并赋值给 `self.repository`，然后把实例返回。函数体内没有条件分支、循环或异常捕获。
- **异常/边界**：无特殊处理。若 `UpdateLogTool()` 构造过程中（即默认 `UpdateLogRepository()` 创建时）失败，异常会原样向上抛出，调用方需自行处理。
- **同文件关系**：调用了本文件中的类 `UpdateLogTool`（进而间接调用其 `__init__`，默认创建外部仓储 `UpdateLogRepository`）；本文件内部没有其它函数调用它，它属于对外出口函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `UpdateLogFileChange` | 描述本次更新中单个被改动文件（路径、变更动作、具体改动说明）的严格校验模型。 |
| `UpdateLogInput` | 工具 `system.update_log` 的输入契约，强制调用方提供背景、细节、文件清单、验证、风险等完整审计字段。 |
| `UpdateLogOutput` | 工具的输出契约，只回传更新 ID、下一个 ID、UTC 时间戳、系统名与写入成功标记，不含历史日志正文。 |
| `UpdateLogTool` | 继承 `BaseTool` 的更新日志工具类，声明 `system.update_log` 的元信息、写副作用与串行调度策略。 |
| `UpdateLogTool.__init__` | 准备数据仓储依赖，未注入时自动创建默认的 `UpdateLogRepository`。 |
| `UpdateLogTool.execute` | 校验入参类型后把整条更新记录（含文件字典列表与实时系统名）写入仓储，并返回紧凑的输出对象。 |
| `create_tool` | 无参工厂函数，返回一个使用默认仓储的 `UpdateLogTool` 实例供工具注册表使用。 |
