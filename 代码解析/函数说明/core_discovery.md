# core/discovery.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时的「工具自动发现（tool discovery）」层，负责把一个 Python 包（默认是 `tool` 包）下面所有直接的子模块扫描一遍，判断哪些模块是启用的工具插件，然后把它们实例化、注册进 `ToolRegistry`，并把工具的元数据（`ToolSpec`）持久化到 `ToolSpecRepository`。它定义了「单文件工具协议」：一个模块必须暴露布尔量 `TOOL_ENABLED`，并且提供一个零参数的 `create_tool()` 工厂函数，该工厂返回 `BaseTool` 的实例，实例上必须带一个 `ToolSpec`。除了真正实例化并注册可执行工具的常规模式，它还支持 `metadata_only=True` 的「只登记元数据」模式：不调用工厂、不往注册表里塞可执行对象，只把模块里那唯一一个带类级 `ToolSpec` 的 `BaseTool` 子类写进目录仓库。扫描结果不是直接抛出异常，而是收敛成不可变的数据结构 `ToolDiscoveryRecord` 与 `ToolDiscoveryReport`，让调用方既能拿到逐模块的成败明细，也能在 `strict=True` 时选择性地用 `ToolDiscoveryError` 一次性失败。文件里还包含几个纯粹的工具函数：加载包、推断模块文件路径、拼装错误记录、把 spec 落库。整体设计原则是「一个坏插件不能拖垮整轮扫描」，所以除了参数校验和 `strict` 模式之外，导入失败、工厂抛异常、注册冲突都被捕获并转写成 `status="error"` 的记录。

## 二、函数与类逐条详解

### `DiscoveryStatus` （第 17 行）
- **作用**：这是一个模块级的类型别名，用 `Literal` 把工具发现过程中一个模块可能落入的终态固定成五个字符串字面量：`"registered"`（本轮成功注册进注册表）、`"already_registered"`（注册表里已经有一个等价的实现，本轮只是幂等确认并顺手补存元数据）、`"disabled"`（模块里写了 `TOOL_ENABLED = False`，属于被主动关闭的插件）、`"ignored"`（模块名以下划线开头、名字叫 `base`，或者本身是个子包，属于协议规定不扫描的对象）、`"error"`（导入失败、缺少 `TOOL_ENABLED`、工厂不可调用、注册冲突等一切失败情形）。它的意义在于给 `ToolDiscoveryRecord.status` 一个静态可检查的取值域，任何写错状态字符串的地方都能被类型检查器（以及阅读代码的人）立刻发现。运行期它不产生任何对象，也不做任何校验，纯粹是给类型注解用的。它被 `ToolDiscoveryRecord` 的字段声明直接引用，因此可以看作整个报告体系的状态字典。
- **参数**：无（它是类型别名，不是可调用对象）。
- **返回**：无返回值；在类型层面等价于 `Literal["registered", "already_registered", "disabled", "ignored", "error"]`。
- **内部流程**：无运行期流程，仅在 `from __future__ import annotations` 生效的前提下被当作注解字符串使用，同时由于它是显式赋值的模块级变量，运行时它确实存在，值为 `typing.Literal[...]` 这个特殊对象。
- **异常/边界**：无特殊处理；它不做取值校验，传入 `Literal` 之外的字符串在运行期不会报错，只有静态检查工具会提示。
- **同文件关系**：被 `ToolDiscoveryRecord.status` 字段引用，进而影响 `ToolDiscoveryReport.errors`、`ToolDiscoveryReport.registered`、`ToolDiscoveryReport.ok` 三个属性的过滤逻辑，以及 `_error_record` 里固定写入的 `"error"`。

### `ToolDiscoveryRecord` （第 26 行）
- **作用**：这是一个用 `@dataclass(frozen=True)` 声明的不可变数据类，代表「一个 Python 模块」这一轮被发现、被判断、被注册的全部结果。它把分散在扫描循环里的信息统一成一条记录：模块名、磁盘路径、是否启用、最终状态，以及在成功或失败时附带的工具名、版本、注册代次和错误文本。之所以需要它，是因为扫描过程里出错的地方非常多（导入、缺变量、工厂签名不对、注册冲突、落库失败），如果每一种都单独抛异常，调用方就无法知道「到底还有哪些模块是好的」；把它做成记录后，报告可以完整保留所有模块的结局。它同时被用作成功路径和失败路径的统一返回类型，`_error_record` 就是专门生产 `status="error"` 版本的工厂。由于是 `frozen=True`，它一旦创建就不可修改，天然适合在报告里跨函数传递和做缓存键，也避免了调用方误改扫描结果。
- **参数**：构造参数依次为——`module: str`（模块的完整点号名，例如 `tool.echo`，必填，无默认值）；`path: str | None`（模块文件路径，成功导入时优先取 `module.__file__`，导入失败时用 `_candidate_path` 猜出来的路径，无法确定时为 `None`）；`enabled: bool | None`（模块声明的启用状态，`True` 表示启用，`False` 表示显式关闭，`None` 表示还没读到或该模块被跳过/导入失败，所以无从判断）；`status: DiscoveryStatus`（终态，必填）；`tool_name: str | None = None`（工具名，来自 `spec.name`，只有走到注册阶段才有）；`version: str | None = None`（工具版本，来自 `spec.version`）；`generation: int | None = None`（工具在注册表中的代次号，注册成功时由 `registry.resolve` 返回，元数据模式或注册前失败时为 `None`）；`error: str | None = None`（错误描述，格式由 `_error_record` 统一成 `前缀: 异常类名: 异常消息`）。
- **返回**：它本身是类；构造时返回一个 `ToolDiscoveryRecord` 实例。字段全部只读（frozen），比较与哈希按字段值进行，因此两条内容相同的记录彼此相等。
- **内部流程**：`dataclass` 装饰器在类定义时自动生成 `__init__`、`__repr__`、`__eq__`；`frozen=True` 再额外生成带写保护的 `__setattr__`/`__delattr__`，任何 `record.status = ...` 的赋值都会抛 `FrozenInstanceError`。类体里只有字段声明和一行 docstring，没有任何自定义方法或 `__post_init__` 校验。
- **异常/边界**：构造本身不校验取值，`status` 传非法字符串、`module` 传空串都不会报错；唯一的异常来源是对已创建实例做属性赋值或删除时抛出的 `dataclasses.FrozenInstanceError`。空值方面，除 `module`、`status` 外的字段都允许为 `None`，并被下游属性（如 `for_tool` 的 `record.tool_name == name`）当作正常情况处理。
- **同文件关系**：被 `_error_record` 直接构造；在 `discover_tools` 里被构造十余次（ignored、disabled、各类 error、以及成功路径的返回值）；被 `_load_package`、`_register_discovered_tool`、`_register_discovered_metadata` 作为返回值构造；被 `ToolDiscoveryReport` 以元组形式聚合，并被 `ToolDiscoveryReport.errors`、`ToolDiscoveryReport.registered`、`ToolDiscoveryReport.for_tool` 读取；`ToolDiscoveryError.__init__` 也遍历它来拼错误信息。

### `ToolDiscoveryReport` （第 40 行）
- **作用**：这是一个 `frozen=True` 的数据类，代表「一次包扫描」的完整、不可变报告，由被扫描包的名称和该包下所有模块的记录元组组成。它存在的意义是把裸列表升级成一个带查询能力的领域对象：调用方不需要自己写 `for` 循环去筛错误或筛成功项，直接用 `.errors`、`.registered`、`.ok` 就能拿到想要的视图。因为记录被存成 `tuple` 而不是 `list`，报告在创建后结构上完全冻结，既能安全地挂在异常对象上（`ToolDiscoveryError.report`），也能作为日志或缓存的内容。`for_tool` 则提供了按工具名反查模块的便捷入口，方便上层在注册失败后定位是哪个文件的问题。整轮 `discover_tools` 的最后一步就是构造它、用它记一条汇总日志，并在 `strict` 模式下决定是否据此抛异常。
- **参数**：构造参数为——`package: str`（被扫描包的名称，可能是用户传入的字符串，也可能是 `ModuleType.__name__`，必填）；`records: tuple[ToolDiscoveryRecord, ...]`（该包下所有被检查模块的记录，顺序与 `discover_tools` 中按模块名排序后的扫描顺序一致，必填）。
- **返回**：类本身；构造返回 `ToolDiscoveryReport` 实例。对外暴露 `errors`、`registered`、`ok` 三个只读属性和一个 `for_tool` 方法。
- **内部流程**：`dataclass` 生成 `__init__`/`__eq__`/`__repr__`，`frozen=True` 阻止后续赋值；类体内定义了三个 `@property` 和一个普通方法，它们都只读 `self.records`，不做任何写入或缓存，因此每次访问都会重新做一次元组推导。构造时不做任何校验，`records` 传列表也能通过（类型注解不强制），但后续属性仍能工作。
- **异常/边界**：构造无异常；属性访问无异常（空 `records` 时 `errors` 和 `registered` 都返回空元组，`ok` 返回 `True`）；唯一会抛异常的是 `for_tool` 对非法入参的处理。属性赋值会抛 `FrozenInstanceError`。
- **同文件关系**：被 `discover_tools` 构造（包加载失败时的早退分支和扫描完成后的正常分支各一次）；被 `ToolDiscoveryError.__init__` 接收并读取 `report.errors`；其内部属性方法读取 `ToolDiscoveryRecord`，其 `for_tool` 也被设计为供外部按工具名查询本文件产出的记录。

### `ToolDiscoveryReport.errors` （第 47 行，property）
- **作用**：这是一个只读属性，把报告里所有 `status == "error"` 的记录筛出来，按原顺序组成一个元组返回。它存在的价值在于把「这轮扫描有没有坏模块」这个最常用的判断变成一行代码，`discover_tools` 末尾的 `if strict and report.errors` 以及 `ToolDiscoveryError.__init__` 拼错误详情都依赖它。它只做过滤，不改写记录，也不对错误做去重或聚合，因此同一模块若产生多条错误记录会原样全部返回（当前实现每个模块最多一条）。因为每次访问都重新计算，调用方如果在循环里反复访问，会重复做线性扫描，但记录数量级很小，无需缓存。
- **参数**：无（`self` 由属性机制隐式传入）。
- **返回**：`tuple[ToolDiscoveryRecord, ...]`；没有任何错误记录时返回空元组 `()`，绝不会返回 `None`。
- **内部流程**：使用生成器表达式遍历 `self.records`，逐个判断 `record.status == "error"`，命中则保留，最后用 `tuple(...)` 物化成不可变元组。
- **异常/边界**：无特殊处理；`self.records` 为空时自然返回空元组。
- **同文件关系**：读取 `ToolDiscoveryRecord.status`；被 `ToolDiscoveryReport.ok` 通过 `not self.errors` 间接调用；被 `ToolDiscoveryError.__init__` 在 `discover_tools` 抛异常时调用。

### `ToolDiscoveryReport.registered` （第 51 行，property）
- **作用**：这是一个只读属性，把「本轮最终处于已注册状态」的记录筛出来，判定条件是 `status` 属于 `{"registered", "already_registered"}` 两个值。它把「幂等重复注册」和「首次注册」视为同一类成功结果，这样上层统计工具数量时不会因为重复扫描而漏算。`discover_tools` 在扫描结束后用它来收集所有工具名，传给 `log_discovery_summary` 打一条活动日志，因此它是「本轮扫描登记了哪些工具」的唯一数据来源。它同样只做过滤，不排序、不去重，返回顺序即扫描顺序。
- **参数**：无。
- **返回**：`tuple[ToolDiscoveryRecord, ...]`；没有成功记录时返回空元组。
- **内部流程**：生成器表达式遍历 `self.records`，用集合字面量 `{"registered", "already_registered"}` 做成员判断（每次访问都会重新构造该集合），命中即保留，最后 `tuple(...)` 物化。
- **异常/边界**：无特殊处理；空记录返回空元组。
- **同文件关系**：读取 `ToolDiscoveryRecord.status`；被 `discover_tools` 在收尾处调用，用于向 `log_discovery_summary` 提供工具名序列。

### `ToolDiscoveryReport.ok` （第 59 行，property）
- **作用**：这是一个只读布尔属性，语义是「本轮扫描没有任何错误记录」。它是给调用方做整体判断用的便捷开关：拿到报告后先看 `report.ok`，为真说明所有被扫描的模块要么注册成功、要么被正常跳过、要么被显式关闭，为假说明至少有一个模块出了错。它的实现直接复用 `errors` 属性并取反，避免重复写一遍过滤逻辑，也保证了 `ok` 与 `errors` 永远一致。注意它只反映「有没有 error」，不反映「有没有注册到工具」——一个所有模块都被 `TOOL_ENABLED = False` 关闭的包，`ok` 依然为 `True`。
- **参数**：无。
- **返回**：`bool`；`errors` 为空元组时为 `True`，否则为 `False`。
- **内部流程**：调用 `self.errors` 得到错误元组，再对其取 `not`，利用空元组为假值、非空元组为真值的规则得到布尔结果。
- **异常/边界**：无特殊处理；依赖 `errors` 不会抛异常。
- **同文件关系**：调用 `ToolDiscoveryReport.errors`；本文件内部没有直接调用者，属于提供给外部调用方的便捷判断入口。

### `ToolDiscoveryReport.for_tool(name: str) -> ToolDiscoveryRecord | None` （第 63 行）
- **作用**：按工具名在报告里反查对应的模块记录，返回第一条 `tool_name` 等于 `name` 的记录，找不到则返回 `None`。它的使用场景是：上层只知道某个工具（例如从注册表或配置里读到的名字）出了问题，想快速定位它是哪个模块、路径在哪、错误文本是什么，就不必自己遍历 `records`。它只返回第一条匹配，因为同一个工具名在一轮扫描里理论上只应由一个模块提供；如果因为异常情况出现重复，后面的记录会被忽略。入参校验比较严格：不是字符串或空串都会立刻抛 `ValueError`，属于「把错误尽早暴露」的设计，而不是静默返回 `None`。
- **参数**：`name: str`——要查找的工具名，必须是非空字符串；空串 `""` 和非字符串（例如 `None`、整数）都会触发 `ValueError`。
- **返回**：`ToolDiscoveryRecord | None`——命中时返回该记录对象（可能是成功记录也可能是错误记录，只要 `tool_name` 匹配）；没有任何记录的 `tool_name` 等于 `name` 时返回 `None`。
- **内部流程**：先做入参类型与空值校验，不通过则抛 `ValueError`；通过后用 `next((record for record in self.records if record.tool_name == name), None)` 做惰性线性扫描，一旦找到即短路返回，否则由 `next` 的默认值返回 `None`。比较时直接用 `==`，`None == name` 为假，所以那些 `tool_name` 为 `None` 的记录（ignored、disabled、导入失败等）天然不会误命中。
- **异常/边界**：`name` 为空串或非字符串时抛 `ValueError("tool name must be a non-empty string")`；找不到时返回 `None` 而不抛异常；`self.records` 为空时同样返回 `None`。
- **同文件关系**：读取 `ToolDiscoveryRecord.tool_name`；本文件内部没有调用者，是对外暴露的查询接口。

### `ToolDiscoveryError` （第 71 行）
- **作用**：这是继承自 `RuntimeError` 的自定义异常，专门表示「在 `strict=True` 的严格扫描下，有一个或多个模块加载/注册失败」。它的关键设计是携带完整的 `report` 对象：不仅把错误信息拼进异常消息里供人阅读，还把结构化的报告挂在异常实例上，让捕获方可以继续按模块、按工具名做程序化处理（例如只重试失败的那几个模块）。异常消息由所有错误记录的 `模块名: 错误详情` 用 `"; "` 连接而成，便于日志一行看清全部问题。它只在 `strict` 为真时被 `discover_tools` 抛出：包本身加载失败时抛一次，扫描结束且报告含错误时再抛一次。非严格模式下永远不会抛出，失败信息只留在报告里。
- **参数**：`report: ToolDiscoveryReport`——必填，本轮扫描的完整报告；`__init__` 会从中读取 `errors` 来拼消息，因此传入的报告应至少包含错误记录，否则消息会退化成 `"tool discovery failed: "`（不会报错，但信息无意义）。
- **返回**：类本身；实例化后得到一个 `ToolDiscoveryError`，可通过 `str(exc)` 得到汇总消息，通过 `exc.report` 拿到原始报告。
- **内部流程**：`__init__` 先把 `report` 存到实例属性 `self.report`，然后用生成器表达式遍历 `report.errors`，对每条记录格式化成 `f"{record.module}: {record.error}"`，再用 `"; ".join(...)` 拼接成 `details`，最后以 `f"tool discovery failed: {details}"` 调用父类 `RuntimeError.__init__` 设置异常消息。
- **异常/边界**：构造过程本身不会抛异常（即使 `report` 类型不对也只是属性访问失败时的 `AttributeError`）；`report.errors` 为空时消息为 `"tool discovery failed: "` 加空串，不报错。
- **同文件关系**：读取 `ToolDiscoveryReport.errors` 与 `ToolDiscoveryRecord.module`、`ToolDiscoveryRecord.error`；被 `discover_tools` 在 `strict=True` 的两个分支中抛出。

### `ToolDiscoveryError.__init__(self, report: ToolDiscoveryReport) -> None` （第 74 行）
- **作用**：这是 `ToolDiscoveryError` 的构造函数，负责把报告存进实例并生成人类可读的异常消息。之所以要在构造函数里就把消息拼好，是因为异常消息一旦抛出就应当自包含，日志或上层包装不必再回头查报告就能看到「哪些模块、什么错」。它把每条错误记录渲染成 `模块名: 错误详情` 的形式，多条之间用分号加空格分隔，这样既紧凑又能逐条辨认。消息统一以 `tool discovery failed: ` 开头，方便日志检索和断言。整个实现只读不写，除了 `self.report` 之外不修改任何外部状态。
- **参数**：`self`——异常实例本身；`report: ToolDiscoveryReport`——本轮扫描报告，必填，会被原样保存到 `self.report`，并立即被遍历以生成消息。
- **返回**：`None`（构造函数不返回值，符合 Python 约定；实例由 `__new__` 返回）。
- **内部流程**：第一步 `self.report = report` 保存引用；第二步用生成器表达式遍历 `report.errors`，对每条错误记录取 `record.module` 和 `record.error` 拼成一段文本；第三步 `"; ".join(...)` 把所有片段连成一个字符串 `details`；第四步调用 `super().__init__(f"tool discovery failed: {details}")`，把最终消息交给 `RuntimeError` 保存为 `args`。
- **异常/边界**：`report` 为 `None` 或缺少 `errors` 属性时会在遍历处抛 `AttributeError`；`report.errors` 为空时消息只剩前缀，不抛异常。对 `record.error` 为 `None` 的记录，格式化结果会显示为 `模块名: None`，属于可接受的降级表现。
- **同文件关系**：调用 `ToolDiscoveryReport.errors`（间接读取 `ToolDiscoveryRecord.status`、`module`、`error`）；由 `discover_tools` 在严格模式下实例化并抛出。

### `discover_tools(registry: ToolRegistry, *, package: str | ModuleType = "tool", repository: ToolSpecRepository | None = None, replace: bool = False, strict: bool = False, reload_modules: bool = False, metadata_only: bool = False) -> ToolDiscoveryReport` （第 82 行）
- **作用**：这是整个文件的主入口，也是唯一一个被外部调用的函数。它加载目标包，枚举包目录下的所有直接子模块，逐个判断是否符合单文件工具协议，并把合格的工具注册进 `ToolRegistry`、把元数据存进 `ToolSpecRepository`，最后返回一份不可变的扫描报告。它有两种工作模式：默认模式会真的调用 `create_tool()` 拿到 `BaseTool` 实例并注册可执行工具；`metadata_only=True` 时只校验工厂可调用性，然后从模块里找出唯一一个带类级 `ToolSpec` 的 `BaseTool` 子类，把 spec 落库而不往注册表里塞任何东西，用于「只同步工具目录、不加载实现」的场景。它贯彻「插件即隔离边界」的理念：导入异常、工厂异常、注册冲突都被捕获并转成 `status="error"` 的记录，让一个坏插件不影响同目录其它插件的注册。只有在 `strict=True` 时才会把失败升级成 `ToolDiscoveryError`。无论成败，扫描结束都会通过 `log_discovery_summary` 记一条活动日志。
- **参数**：`registry: ToolRegistry`——必填、位置参数，接收扫描到的工具的注册表；不是 `ToolRegistry` 实例时抛 `TypeError`。`package: str | ModuleType = "tool"`——关键字参数，默认 `"tool"`，可以传包名字符串（内部 `importlib.import_module` 加载）也可以直接传已导入的模块对象；传空串、非字符串非模块的值会抛 `TypeError`（校验在 `_load_package` 内完成）；传字符串但导入失败会被转成错误记录。`repository: ToolSpecRepository | None = None`——关键字参数，元数据仓库，为 `None` 时跳过落库；非 `None` 且不是 `ToolSpecRepository` 时抛 `TypeError`。`replace: bool = False`——关键字参数，是否允许覆盖注册表/仓库里已有的不同实现或不同 schema，必须是 `bool`，否则抛 `TypeError`；为 `False` 时遇到冲突会抛 `ValueError`（被本函数捕获成错误记录）。`strict: bool = False`——关键字参数，严格模式开关，必须是 `bool`；为 `True` 时包加载失败或最终报告含错误都会抛 `ToolDiscoveryError`。`reload_modules: bool = False`——关键字参数，是否为已在 `sys.modules` 里的模块执行 `importlib.reload`，必须是 `bool`；用于开发期热重载工具实现。`metadata_only: bool = False`——关键字参数，元数据模式开关，必须是 `bool`；为 `True` 时要求 `repository` 不为 `None`，否则抛 `ValueError`，并且不会调用工厂、不会注册可执行对象。
- **返回**：`ToolDiscoveryReport`——包含 `package` 名称和按模块名排序的 `records` 元组。包加载失败且 `strict=False` 时返回只含一条错误记录的报告；正常情况下返回含全部模块结局的报告。当 `strict=True` 且存在错误时不返回，而是抛异常。
- **内部流程**：第一步做参数校验——`registry` 类型、`metadata_only` 与 `repository` 的搭配关系、`repository` 类型、四个布尔开关的类型，任何一项不合法立刻抛 `TypeError`/`ValueError`。第二步调用 `_load_package(package)` 拿到 `(package_name, package_module, package_error)`；若 `package_error` 不为空，就构造只含这一条记录的报告，`strict` 时抛 `ToolDiscoveryError`，否则直接返回。第三步 `importlib.invalidate_caches()` 清掉导入缓存，然后用 `pkgutil.iter_modules(package_module.__path__, prefix=f"{package_module.__name__}.")` 枚举子模块，并用 `sorted(..., key=lambda item: item.name)` 按完整模块名排序，保证扫描顺序稳定可复现。第四步进入逐模块循环：先取 `short_name`（点号名最后一段）和 `candidate_path = _candidate_path(module_info)`；若 `short_name` 以下划线开头、等于 `"base"`，或 `module_info.ispkg` 为真（是子包），就追加一条 `status="ignored"`、`enabled=None` 的记录并 `continue`。第五步尝试 `importlib.import_module(module_info.name)`，若 `reload_modules` 为真且模块已在 `sys.modules` 中则再 `importlib.reload(module)`；这一步的任何 `Exception` 都被捕获，通过 `_error_record(..., "module import failed")` 记错后 `continue`。第六步取 `module_path = getattr(module, "__file__", None) or candidate_path`，读 `enabled = getattr(module, "TOOL_ENABLED", None)`：不是 `bool` 就记 `status="error"`、错误文本 `"TOOL_ENABLED must be defined as a boolean"`；为 `False` 就记 `status="disabled"`、`enabled=False`；两种情况都 `continue`。第七步取 `factory = getattr(module, "create_tool", None)`，不是可调用对象就记错误 `"enabled module must define callable create_tool()"`；然后用 `inspect.signature(factory).bind()` 试探性地无参绑定，抛 `TypeError`/`ValueError` 就通过 `_error_record(..., "create_tool must be callable without arguments")` 记错。第八步分叉：若 `metadata_only` 为真，调用 `_register_discovered_metadata(...)`，其抛出的任何异常被转成 `"metadata registration failed"` 错误记录，否则把返回记录追加进去，然后 `continue`。第九步（默认模式）：把 `tool` 初始化为 `None`，在 `try` 里先 `tool = factory()` 再调用 `_register_discovered_tool(...)` 得到记录；一旦抛异常，就从 `tool` 上尽力取 `spec`（用 `getattr(tool, "spec", None)` 并检查 `isinstance(spec, ToolSpec)`）得到 `tool_name` 和 `version`，再通过 `registry.maybe_resolve(tool_name)` 查出当前活跃实现以取到代次号，最后用 `_error_record(..., "create or register failed", ...)` 生成带工具名、版本、代次的错误记录；没有异常则直接追加成功记录。第十步循环结束后构造 `ToolDiscoveryReport(package_name, tuple(records))`，调用 `log_discovery_summary(package_name, (record.tool_name for record in report.registered if record.tool_name))` 记汇总日志（这里过滤掉 `tool_name` 为 `None` 的记录）；最后若 `strict and report.errors` 为真则 `raise ToolDiscoveryError(report)`，否则 `return report`。
- **异常/边界**：会主动抛出的异常有——`TypeError`（`registry` 非 `ToolRegistry`、`repository` 非 `ToolSpecRepository`、四个开关非 `bool`、`package` 非法）、`ValueError`（`metadata_only=True` 但 `repository is None`）、`ToolDiscoveryError`（`strict=True` 且包加载失败，或 `strict=True` 且扫描后存在错误记录）。对第三方代码的异常采取「一律捕获并记录」策略：模块导入失败、`create_tool` 抛任意异常、注册/落库失败都不会中断扫描。边界情况：包里没有任何子模块时 `modules` 为空、`records` 为空、报告 `ok` 为 `True`；`TOOL_ENABLED` 缺失或非布尔一律算错误（不会默认为启用或关闭）；元数据模式下即使模块里工具子类数量不等于 1，也只是被记成错误记录而不会中断其它模块；`replace=False` 遇到同名不同实现会抛 `ValueError`，被捕获成该模块的错误记录，且注册表保持原样。
- **同文件关系**：调用 `_load_package`（加载并校验目标包）、`_candidate_path`（推断候选文件路径）、`_error_record`（四处生成错误记录）、`_register_discovered_metadata`（元数据模式注册）、`_register_discovered_tool`（默认模式注册）；构造 `ToolDiscoveryReport` 与 `ToolDiscoveryRecord`，并在严格模式下抛出 `ToolDiscoveryError`。它调用外部依赖 `importlib.invalidate_caches`、`pkgutil.iter_modules`、`importlib.import_module`、`importlib.reload`、`inspect.signature`、`log_discovery_summary`，以及 `registry.maybe_resolve`。本文件内没有其它函数调用它。

### `_load_package(package: str | ModuleType) -> tuple[str, ModuleType | None, ToolDiscoveryRecord | None]` （第 279 行）
- **作用**：这是一个私有辅助函数，负责把 `discover_tools` 收到的 `package` 参数统一成「一个已导入的、具备 `__path__` 的包模块」，同时产出用于报告的包名和可能的错误记录。它把两种合法输入（模块对象、非空包名字符串）收敛到同一套后续逻辑上，让主流程不必区分用户传的是名字还是对象。当传入的是字符串时，它用 `importlib.import_module` 真正导入，并把导入过程中的任何异常（包不存在、包内部语法错误、依赖缺失等）捕获成一条错误记录而不是向上抛，因为包导入本身也属于「扩展代码」，其失败应当出现在报告里。导入成功后它还额外校验模块是否有 `__path__`，以此确认它确实是个包而不是普通模块——没有 `__path__` 就没法枚举子模块，因此直接产出 `"tool package must define __path__"` 的错误记录。返回值设计成三元组而不是抛异常，是为了让 `discover_tools` 能用同一套 `strict` 分支处理所有失败。
- **参数**：`package: str | ModuleType`——必填。若为 `ModuleType` 实例，直接采用它的 `__name__` 作为包名并把它本身当作包模块（不做重新导入）；若为非空字符串，则以其为包名调用 `importlib.import_module` 导入；若既不是模块也不是非空字符串（例如 `None`、空串、`""`、整数、空白字符串），抛 `TypeError`。
- **返回**：`tuple[str, ModuleType | None, ToolDiscoveryRecord | None]`，即 `(package_name, package_module, package_error)`。成功时 `package_module` 是包模块、`package_error` 为 `None`；字符串导入失败时 `package_module` 为 `None`、`package_error` 是一条 `status="error"`、`error` 以 `"package import failed: "` 开头的记录；模块没有 `__path__` 时 `package_module` 为 `None`、`package_error` 是一条 `error="tool package must define __path__"` 的记录，其 `path` 取该模块的 `__file__`（可能为 `None`）。
- **内部流程**：第一步用 `isinstance(package, ModuleType)` 判断是否为模块对象，是则取 `__name__` 并直接复用。第二步否则判断 `isinstance(package, str) and package.strip()`，成立则把原始字符串当作 `package_name`（注意保留原始写法，不做 `strip` 归一化），在 `try` 里 `importlib.import_module(package)`；捕获到任意 `Exception` 时立即返回 `(package_name, None, _error_record(package_name, None, None, exc, "package import failed"))`，其中 `path` 和 `enabled` 都传 `None`，因为此时还不知道文件路径、也无从谈启用状态。第三步两者都不是则 `raise TypeError("package must be a non-empty module name or ModuleType")`。第四步对已确定的 `package_module` 用 `hasattr(package_module, "__path__")` 检查，缺失则返回带 `__path__` 缺失错误的记录。第五步全部通过则返回 `(package_name, package_module, None)`。
- **异常/边界**：唯一的主动异常是入参既非模块也非非空字符串时的 `TypeError`；字符串导入异常被内部吞掉并转成记录，不会向外抛；`hasattr` 检查对任何模块对象都安全。边界细节：`package` 传 `""` 或全空白字符串时，`package.strip()` 为假会走到 `TypeError` 分支，而不是尝试导入空名；模块对象的 `__file__` 可能不存在（如命名空间包或内建模块），此时记录里的 `path` 为 `None`。
- **同文件关系**：调用 `_error_record` 生成导入失败记录，并构造 `ToolDiscoveryRecord`（`__path__` 缺失分支）；被 `discover_tools` 在参数校验之后立即调用。

### `_register_discovered_tool(registry: ToolRegistry, module: ModuleType, module_path: str | None, tool: object, *, repository: ToolSpecRepository | None, replace: bool) -> ToolDiscoveryRecord` （第 319 行）
- **作用**：这是一个私有辅助函数，负责把默认模式下一次 `create_tool()` 的产物真正落地：校验类型、判断是否与注册表里的现有实现等价、必要时执行替换注册，并把 `ToolSpec` 存进仓库。它把「幂等」做成了显式分支：如果注册表里已有同名工具，且当前实现的类型与 spec 都与新工具完全一致，就认为这轮扫描是重复的，只补存一次元数据并返回 `status="already_registered"`，代次号沿用旧的，不会无谓地产生新一代。如果同名但实现或 spec 不同，则根据 `replace` 决定是报错还是覆盖——`replace=False` 时抛 `ValueError` 让调用方显式选择，避免插件同名冲突被静默吞掉。真正注册时它用 `replace=current is not None` 告诉注册表「这是覆盖还是新增」，注册后再用 `registry.resolve(spec.name)` 取回代次号写进记录，最后统一通过 `_save_spec` 落库。它把所有校验失败都交给调用方（`discover_tools`）捕获并转成错误记录。
- **参数**：`registry: ToolRegistry`——目标注册表，位置参数，必填，用于查询现有实现并执行注册。`module: ModuleType`——工具所在的模块对象，位置参数，必填；只用于取 `module.__name__` 作为记录里的模块名以及拼 `implementation_ref`。`module_path: str | None`——模块文件路径，位置参数，必填，可为 `None`，原样写进记录。`tool: object`——工厂返回的对象，位置参数，必填，必须能通过 `isinstance(tool, BaseTool)`，否则抛 `TypeError`。`repository: ToolSpecRepository | None`——关键字参数，必填（无默认值），元数据仓库；为 `None` 时落库被 `_save_spec` 直接跳过。`replace: bool`——关键字参数，必填，冲突时是否允许覆盖；为假时冲突抛 `ValueError`。
- **返回**：`ToolDiscoveryRecord`——两种可能：`status="already_registered"`（等价实现已存在，带旧代次号）或 `status="registered"`（新增或覆盖成功，带 `registry.resolve` 返回的新代次号）。两种情况下 `enabled` 都是 `True`，`tool_name` 与 `version` 都取自 `spec`。
- **内部流程**：第一步校验 `isinstance(tool, BaseTool)`，不通过抛 `TypeError("create_tool() must return a BaseTool instance")`。第二步用 `getattr(tool, "spec", None)` 取 spec 并校验 `isinstance(spec, ToolSpec)`，不通过抛 `TypeError("created tool must define a ToolSpec instance")`。第三步 `current = registry.maybe_resolve(spec.name)`；若非 `None`，解包出 `current_tool, generation`，先用 `type(current_tool) is type(tool) and current_tool.spec == spec` 判断是否为完全等价实现——成立则调用 `_save_spec(repository, spec, module, tool)` 并返回 `already_registered` 记录；不等价且 `not replace` 则抛 `ValueError`，消息中带上工具名并提示 `pass replace=True to replace it`。第四步调用 `registry.register(tool, replace=current is not None)`，用 `current is not None` 精确区分「新增」与「覆盖」。第五步 `_, generation = registry.resolve(spec.name)` 取回权威代次号。第六步 `_save_spec(...)` 落库。第七步构造并返回 `status="registered"` 的记录。
- **异常/边界**：`TypeError`——工具不是 `BaseTool`、工具没有 `ToolSpec`；`ValueError`——同名不同实现且 `replace=False`；注册表自身的方法可能抛出的异常（例如重复注册）不会被本函数捕获，直接向上冒泡给 `discover_tools`。边界：`repository is None` 时落库静默跳过但注册照常进行；`registry.maybe_resolve` 返回 `None` 表示名字未占用，直接走新增路径；等价判断使用 `type(...) is type(...)` 的严格类型同一性，子类与父类即使 spec 相同也不算等价，会走冲突分支。
- **同文件关系**：调用 `_save_spec`；构造 `ToolDiscoveryRecord`；被 `discover_tools` 在默认模式的 `try` 块中调用，其抛出的异常由 `discover_tools` 捕获成 `"create or register failed"` 错误记录。

### `_register_discovered_metadata(module: ModuleType, module_path: str | None, registry: ToolRegistry, *, repository: ToolSpecRepository | None, replace: bool) -> ToolDiscoveryRecord` （第 368 行）
- **作用**：这是元数据模式（`metadata_only=True`）下的注册函数，它的职责是「不执行工厂，只把模块里静态声明的 `ToolSpec` 登记到目录仓库」。之所以需要它，是因为有些场景（例如列出工具清单、生成前端目录、跨进程同步 schema）只关心 spec 元数据，而执行 `create_tool()` 可能会建立网络连接、加载模型或做其它有副作用的初始化，代价高且不必要。它的实现方式是遍历模块命名空间，找出所有「是 `BaseTool` 子类、定义在本模块内、且带有类级 `ToolSpec` 实例」的类，并要求这样的类恰好只有一个，否则抛 `TypeError`，因为多于一个就无法确定该模块代表哪个工具。找到后它先看注册表里是否已有同名工具：若 spec 相同则幂等返回 `already_registered`；若不同且 `replace=False` 则抛 `ValueError`。接着它还会检查仓库里同名的同版本条目，如果已存条目的 `schema_hash` 与新 spec 不一致且不允许替换，同样抛 `ValueError`，防止 schema 静默漂移。全部通过后落库，并返回一条 `status="registered"` 但 `generation=None` 的记录，明确表示「目录里有，但可执行注册表是空的」。
- **参数**：`module: ModuleType`——待扫描的模块对象，位置参数，必填；用于 `vars(module)` 遍历、`module.__name__` 记录和 `__module__` 归属判断。`module_path: str | None`——模块文件路径，位置参数，必填，可为 `None`，写入记录。`registry: ToolRegistry`——注册表，位置参数，必填；本函数只用它的 `maybe_resolve` 做冲突判断，绝不写入。`repository: ToolSpecRepository | None`——关键字参数，必填；为 `None` 时跳过所有落库与仓库 schema 冲突检查（但注册表冲突检查仍会执行）。`replace: bool`——关键字参数，必填；为假时遇到「注册表同名不同 spec」或「仓库同名同版本但 schema_hash 不同」都会抛 `ValueError`。
- **返回**：`ToolDiscoveryRecord`——可能是 `status="already_registered"`（注册表已有同 spec 工具，代次号取自现有实现）或 `status="registered"`（`generation` 固定为 `None`，因为本模式不向注册表注册可执行对象）。两种情况下 `enabled=True`，`tool_name`、`version` 取自 spec。
- **内部流程**：第一步初始化空列表 `candidates`。第二步遍历 `vars(module).values()`：跳过非类（`not inspect.isclass(value)`）和 `BaseTool` 本身；用 `issubclass(value, BaseTool)` 判断是否工具类，并把 `TypeError`（某些对象传给 `issubclass` 会报错）兜住置为 `False`；再取 `getattr(value, "spec", None)`；只有同时满足「是工具类」「`value.__module__ == module.__name__`（类确实定义在本模块，排除 import 进来的工具类）」「`spec` 是 `ToolSpec` 实例」三个条件才加入 `candidates`。第三步若 `len(candidates) != 1`（0 个或多个）抛 `TypeError`，消息为 `"metadata-only discovery requires exactly one BaseTool subclass with a class-level ToolSpec"`。第四步解包 `tool_class, spec = candidates[0]`。第五步 `current = registry.maybe_resolve(spec.name)`：若非 `None`，比较 `current_tool.spec == spec`，相等则调用 `_save_metadata(repository, spec, module, tool_class, replace=True)` 并返回 `already_registered`（沿用旧代次号）；不相等且 `not replace` 抛 `ValueError`。第六步若 `repository is not None`：用 `repository.get(spec.name, spec.version)` 取同名同版本的已存条目，若存在且 `stored["schema_hash"] != spec.schema_hash` 且 `not replace`，抛 `ValueError`（消息提到 `different catalog schema`）；否则调用 `_save_metadata(repository, spec, module, tool_class, replace=True)` 落库——注意这里恒传 `replace=True`，因为前面的冲突检查已经把关。第七步返回 `generation=None` 的 `status="registered"` 记录。
- **异常/边界**：`TypeError`——模块内符合「本模块定义 + BaseTool 子类 + 类级 ToolSpec」条件的类不是恰好一个；`ValueError`——注册表同名不同 spec 且 `replace=False`，或仓库同名同版本 schema 不同且 `replace=False`。边界：`repository is None` 时仓库检查整段跳过，函数仍能正常返回记录（只是没落库）；`vars(module)` 只反映模块命名空间里可见的名字，被 `del` 掉的类不会被看到；`issubclass` 的 `TypeError` 被静默当作「不是工具类」，不会中断扫描；`stored` 为 `None`（仓库里没有该版本）时直接进入落库分支。所有抛出的异常都会在 `discover_tools` 中被捕获成 `"metadata registration failed"` 错误记录。
- **同文件关系**：调用 `_save_metadata`；构造 `ToolDiscoveryRecord`；被 `discover_tools` 在 `metadata_only` 分支的 `try` 块中调用。

### `_save_spec(repository: ToolSpecRepository | None, spec: ToolSpec, module: ModuleType, tool: BaseTool) -> None` （第 440 行）
- **作用**：这是一个极小的私有辅助函数，专门负责把「运行时实例出来的工具」的 spec 存进仓库，并顺手生成实现引用字符串。它把实现引用统一成 `模块点号名:类限定名` 的格式（例如 `tool.echo:EchoTool`），这个字符串让目录仓库能反向定位到具体是哪个模块里的哪个类提供了该工具，便于后续诊断或按需导入。之所以要单独抽一个函数，是因为默认模式的两条成功路径（`already_registered` 与 `registered`）都需要落库，抽出来可以保证两处写入格式完全一致。它对 `repository is None` 做静默短路，因此调用方不必在每个调用点重复写空判断，让注册逻辑保持干净。
- **参数**：`repository: ToolSpecRepository | None`——位置参数，必填；为 `None` 时函数立即返回，不做任何事。`spec: ToolSpec`——位置参数，必填，要保存的工具规格，原样传给 `repository.save`。`module: ModuleType`——位置参数，必填，只用于取 `module.__name__` 拼实现引用。`tool: BaseTool`——位置参数，必填，只用于取 `type(tool).__qualname__` 拼实现引用（用 `qualname` 而非 `name`，以便嵌套类也能被准确定位）。
- **返回**：`None`；不返回任何值，落库成功与否也不反馈（`repository.save` 的返回值被丢弃）。
- **内部流程**：第一步判断 `repository is None`，是则 `return`。第二步用 f-string 拼出 `implementation_ref = f"{module.__name__}:{type(tool).__qualname__}"`。第三步调用 `repository.save(spec, implementation_ref=implementation_ref, replace=True)`，注意这里恒传 `replace=True`，即重复保存同一工具时直接覆盖，因为冲突判断已经在上游 `_register_discovered_tool` 里做过了。
- **异常/边界**：本函数自身不抛异常，也不捕获异常；`repository.save` 抛出的任何异常（磁盘写入失败、schema 冲突等）会原样冒泡，最终由 `discover_tools` 捕获成错误记录。边界：`repository is None` 时完全空操作；`module` 缺少 `__name__` 会抛 `AttributeError`（正常模块不会发生）。
- **同文件关系**：被 `_register_discovered_tool` 调用（在 `already_registered` 分支和 `registered` 分支各一次）；本文件内没有其它调用者。

### `_save_metadata(repository: ToolSpecRepository | None, spec: ToolSpec, module: ModuleType, tool_class: type[BaseTool], *, replace: bool) -> None` （第 456 行）
- **作用**：这是元数据模式专用的落库辅助函数，与 `_save_spec` 几乎对称，区别有两点：它接收的是工具「类」而不是「实例」（因为元数据模式从不实例化），并且 `replace` 由调用方显式传入而不是恒为 `True`，以便保留调用方对覆盖行为的控制。它同样生成 `模块点号名:类限定名` 形式的实现引用，让仓库条目能指回定义该工具的文件与类。抽成独立函数的价值在于让 `_register_discovered_metadata` 的两处保存点（幂等分支与正常分支）共享同一套引用格式与空值短路逻辑，避免实现引用格式在两处漂移。
- **参数**：`repository: ToolSpecRepository | None`——位置参数，必填；为 `None` 时直接返回。`spec: ToolSpec`——位置参数，必填，要保存的工具规格。`module: ModuleType`——位置参数，必填，提供 `__name__` 用于拼实现引用。`tool_class: type[BaseTool]`——位置参数，必填，`BaseTool` 的子类对象，取其 `__qualname__` 拼实现引用。`replace: bool`——关键字参数（`*` 之后），必填无默认值；原样透传给 `repository.save`，决定同名条目能否被覆盖。
- **返回**：`None`；不返回任何值，也不反馈保存结果。
- **内部流程**：第一步 `if repository is None: return` 短路。第二步拼 `implementation_ref = f"{module.__name__}:{tool_class.__qualname__}"`。第三步调用 `repository.save(spec, implementation_ref=implementation_ref, replace=replace)`，把 `replace` 原样传递。
- **异常/边界**：自身不抛不捕；`repository.save` 的异常向上冒泡到 `discover_tools` 并被记为 `"metadata registration failed"`。边界：`repository is None` 时空操作；`replace` 传非布尔值不会被本函数校验（调用方恒传 `True`），最终由 `repository.save` 自行决定如何处理。
- **同文件关系**：被 `_register_discovered_metadata` 调用（幂等分支传 `replace=True`，正常分支也传 `replace=True`）；本文件内没有其它调用者。

### `_candidate_path(module_info: pkgutil.ModuleInfo) -> str | None` （第 470 行）
- **作用**：这是一个私有工具函数，用来在「模块还没成功导入、拿不到 `__file__`」的情况下，根据 `pkgutil.iter_modules` 给出的 `ModuleInfo` 猜出这个模块大概对应的磁盘文件路径。它的存在是为了让错误记录尽量带上可点击/可定位的路径：导入失败时我们无法读取模块对象，但 `module_info.module_finder.path` 通常就是包所在目录，再拼上模块短名即可得到 `xxx.py` 或 `xxx/__init__.py`。它用 `isinstance(root, str)` 做了保守判断，因为 `module_finder` 可能是 zipimporter 等对象，其 `path` 未必是普通字符串路径，这时宁可返回 `None` 也不给出错误路径。这个函数只做字符串拼接与路径构造，不触碰文件系统，不会因为文件不存在而报错。
- **参数**：`module_info: pkgutil.ModuleInfo`——必填，`pkgutil.iter_modules` 产出的具名元组，含 `module_finder`、`name`、`ispkg` 三个字段；本函数只读 `module_finder.path`、`name` 和 `ispkg`。
- **返回**：`str | None`——正常返回拼接后的路径字符串（子包为 `<root>/<short_name>/__init__.py`，普通模块为 `<root>/<short_name>.py`）；当 `module_finder` 没有 `path` 属性或其值不是 `str` 时返回 `None`。
- **内部流程**：第一步 `root = getattr(module_info.module_finder, "path", None)`，用 `getattr` 默认值兜住缺失属性。第二步 `isinstance(root, str)` 不成立就 `return None`。第三步取 `short_name = module_info.name.rsplit(".", 1)[-1]`，即去掉 `iter_modules` 加上的包前缀，只保留模块短名。第四步按 `module_info.ispkg` 分支：为真用 `Path(root, short_name, "__init__.py")`，为假用 `Path(root, f"{short_name}.py")`，最后都 `str(...)` 化后返回。注意路径用 `Path` 拼接因此是平台原生的分隔符。
- **异常/边界**：无特殊处理；不做文件存在性检查，返回的路径可能指向不存在的文件；`module_info` 缺少 `name` 或 `ispkg` 属性时会抛 `AttributeError`（`pkgutil.ModuleInfo` 保证有这两个字段）。`root` 为 `bytes`、`pathlib.Path` 等非 `str` 类型时按 `None` 处理。
- **同文件关系**：被 `discover_tools` 在循环开头为每个模块调用一次，结果既用于 ignored 记录，也作为导入失败时的错误记录路径以及 `module.__file__` 缺失时的回退路径；本文件内没有其它调用者。

### `_error_record(module: str, path: str | None, enabled: bool | None, error: Exception, prefix: str, *, tool_name: str | None = None, version: str | None = None, generation: int | None = None) -> ToolDiscoveryRecord` （第 480 行）
- **作用**：这是全文件统一的「错误记录工厂」，把一次失败的事件（异常对象 + 人类可读前缀 + 上下文信息）规范化成一条 `status="error"` 的 `ToolDiscoveryRecord`。它最重要的价值是统一错误文本格式：最终写进记录的 `error` 字段固定为 `前缀: 异常类名: 异常消息`，例如 `create or register failed: ValueError: tool 'x' has a different active implementation`。这种格式让日志既能一眼看出失败发生在哪个阶段（前缀），又能看出异常类型和具体原因，便于按异常类型做统计或断言。它同时允许携带工具名、版本、代次号，使得「工厂已经跑起来、spec 也拿到了，但注册阶段失败」这类半成功场景的信息不会丢失。整个函数没有副作用，只做字符串拼接和对象构造，因此可以在扫描循环的各个 `except` 分支里安全、重复地调用。
- **参数**：`module: str`——位置参数，必填，出错的模块名，写进记录的 `module` 字段。`path: str | None`——位置参数，必填，模块路径，可为 `None`。`enabled: bool | None`——位置参数，必填，该模块的启用状态：导入阶段失败传 `None`，走到工厂/注册阶段失败传 `True`，包级失败传 `None`。`error: Exception`——位置参数，必填，被捕获的异常对象，用于取 `type(error).__name__` 和 `str(error)`。`prefix: str`——位置参数，必填，阶段说明前缀，本文件里使用的取值有 `"package import failed"`、`"module import failed"`、`"create_tool must be callable without arguments"`、`"metadata registration failed"`、`"create or register failed"`。`tool_name: str | None = None`——关键字参数，工具名，默认 `None`，仅注册阶段失败时提供。`version: str | None = None`——关键字参数，工具版本，默认 `None`。`generation: int | None = None`——关键字参数，注册表中的代次号，默认 `None`。
- **返回**：`ToolDiscoveryRecord`——一条 `status="error"` 的记录，`error` 字段为 `f"{prefix}: {type(error).__name__}: {error}"`，其余字段按传入值原样填充（`enabled`、`tool_name`、`version`、`generation` 可能为 `None`）。
- **内部流程**：唯一的一步就是构造并返回 `ToolDiscoveryRecord`：`module`、`path`、`enabled` 直接透传，`status` 硬编码为 `"error"`，`tool_name`、`version`、`generation` 透传，`error` 用 f-string 把 `prefix`、`type(error).__name__`、`error`（触发其 `__str__`）三段用冒号加空格连接。没有分支、循环或缓存。
- **异常/边界**：本函数自身几乎不抛异常；极端情况下若 `error` 的 `__str__` 实现抛异常，异常会从格式化处冒泡（正常异常对象不会发生）。边界：`prefix` 传空串时错误文本会以 `": "` 开头；`error` 的消息为空时文本以异常类名结尾；`error` 的消息本身含换行时会原样保留在记录里。
- **同文件关系**：构造 `ToolDiscoveryRecord`；被 `discover_tools` 在四处调用（模块导入失败、工厂签名不可无参调用、元数据注册失败、创建或注册失败），并被 `_load_package` 在包导入失败分支调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `DiscoveryStatus` | 用 `Literal` 固定工具发现结果的五种状态取值（registered / already_registered / disabled / ignored / error）。 |
| `ToolDiscoveryRecord` | 不可变数据类，记录单个模块这一轮被发现、判断、注册的全部结果与错误信息。 |
| `ToolDiscoveryReport` | 不可变数据类，聚合一整轮包扫描的所有模块记录，并提供按状态筛选与按工具名反查的能力。 |
| `ToolDiscoveryReport.errors` | 只读属性，筛出所有 `status == "error"` 的记录元组。 |
| `ToolDiscoveryReport.registered` | 只读属性，筛出 `registered` 与 `already_registered` 两种成功状态的记录元组。 |
| `ToolDiscoveryReport.ok` | 只读属性，返回「本轮没有任何错误记录」的布尔判断。 |
| `ToolDiscoveryReport.for_tool(name)` | 按工具名反查第一条匹配的模块记录，找不到返回 `None`，入参非法抛 `ValueError`。 |
| `ToolDiscoveryError` | 严格模式下抛出的异常，携带完整报告并把所有错误拼成一条消息。 |
| `ToolDiscoveryError.__init__(report)` | 保存报告到 `self.report` 并生成 `tool discovery failed: 模块: 错误...` 形式的异常消息。 |
| `discover_tools(...)` | 主入口：扫描指定包的直接子模块，按单文件工具协议注册工具或仅登记元数据，返回扫描报告。 |
| `_load_package(package)` | 把包名字符串或模块对象统一成已导入的包模块，失败时返回错误记录而非抛异常。 |
| `_register_discovered_tool(...)` | 校验工厂产出的 `BaseTool`/`ToolSpec`，处理幂等与冲突后注册进注册表并落库 spec。 |
| `_register_discovered_metadata(...)` | 不调用工厂，只从模块中找出唯一带类级 `ToolSpec` 的工具类并写入目录仓库。 |
| `_save_spec(repository, spec, module, tool)` | 以 `模块:类` 作为实现引用，把运行时工具的 spec 以覆盖方式存入仓库。 |
| `_save_metadata(repository, spec, module, tool_class, *, replace)` | 以 `模块:类` 作为实现引用，把工具类的 spec 按调用方指定的 replace 策略存入仓库。 |
| `_candidate_path(module_info)` | 由 `pkgutil.ModuleInfo` 推断模块的磁盘文件路径，无法确定时返回 `None`。 |
| `_error_record(...)` | 统一的错误记录工厂，把异常与阶段前缀规范成 `前缀: 异常类名: 消息` 形式的 error 记录。 |
