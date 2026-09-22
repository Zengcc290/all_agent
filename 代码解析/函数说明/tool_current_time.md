# tool/current_time.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时工具集中的一个「最小只读工具」实现，职责非常单一：让运行中的 Agent 能够拿到**真实机器的当前系统本地时间**。它的存在意义在于纠正大模型的固有缺陷——模型内部对「现在几点、今天几号」没有可靠感知，只能靠训练数据里的时间瞎猜，因此必须提供一个可调用的外部工具来读取操作系统时钟。文件整体结构遵循项目的工具规范：先定义输入模型、再定义输出模型、然后定义继承自 `core.BaseTool` 的工具类并声明一份 `ToolSpec` 元数据（工具名 `system.current_time`、版本、输入输出模型、副作用等级 `read`、超时 5 秒、幂等、可并行等），最后提供一个零参数工厂函数 `create_tool()` 供自动发现器实例化。整个文件不含任何写操作、不访问网络、不读写磁盘，纯粹调用标准库 `datetime` 读取一次系统时钟并格式化为可序列化的结构。它被用到的时机是：Agent 在推理中需要「现在时间」这类事实（例如判断近两天、本周、今天日期）时，由工具调度层按 `ToolSpec` 校验参数后调用 `execute`，拿到 ISO 8601 字符串、时区名和 Unix 时间戳三段结果，再拼进上下文。文件中还包括两个 Pydantic 模型类（`CurrentTimeInput`、`CurrentTimeOutput`）和一个模块级常量 `TOOL_ENABLED = True`，后者是工具加载开关，告诉发现器「这个工具默认启用」。

## 二、函数与类逐条详解

### `CurrentTimeInput(BaseModel)` （第 14 行）
- **作用**：定义 `system.current_time` 工具的输入参数结构。这个工具刻意设计成「不需要任何参数」——时间是一个全局环境事实，调用方提供任何东西都没有意义，因此该模型里只声明了配置项、一个字段都不加。它被用作 `ToolSpec.input_model`，让工具调度层在调用 `execute` 之前先用 Pydantic 校验传入的参数字典：由于配置了 `extra="forbid"`，模型如果擅自编造出 `{"timezone": "Asia/Shanghai"}` 之类的多余参数，校验会直接失败并报错，从而避免模型产生「可以指定时区/可以指定时间」的幻觉。这个类本身没有业务逻辑，是纯粹的模式声明，属于「用类型系统约束 LLM 行为」的防御性设计。
- **参数**：无自定义字段。类体内仅设置 `model_config = ConfigDict(extra="forbid", strict=True)`：`extra="forbid"` 表示禁止任何未声明字段；`strict=True` 表示严格类型校验模式，不做隐式类型转换。因此该模型的合法输入只有空字典 `{}`（或省略参数）。
- **返回**：类不是函数，没有返回值；其实例化结果是 `CurrentTimeInput` 对象，代表一次「无参数」的调用请求。
- **内部流程**：继承自 `pydantic.BaseModel`，由 Pydantic 的元类在类创建时读取 `model_config` 与字段声明，构建出校验器。因为没有任何字段，校验逻辑退化为「只检查是否有多余键」。运行时由 `core` 层的工具调用框架负责实例化，本文件内部并不主动构造它。
- **异常/边界**：传入含多余键的字典会抛 Pydantic 的 `ValidationError`（因为 `extra="forbid"`）。除此之外无特殊处理；空值、`None` 是否被接受取决于外层框架如何调用，本文件未做额外兜底。
- **同文件关系**：被本文件的 `CurrentTimeTool.spec` 通过 `input_model=CurrentTimeInput` 引用；`CurrentTimeTool.execute` 的类型标注 `arguments: CurrentTimeInput` 也指向它。它不调用本文件任何函数。

### `CurrentTimeOutput(BaseModel)` （第 20 行）
- **作用**：定义 `system.current_time` 工具的输出结构，把一次时间读取的结果固定成三个字段，保证输出「稳定、可序列化、可被 LLM 与上层代码共同消费」。它同时承担文档职责：每个字段的 `description` 会被序列化进工具 schema，模型据此知道 `local_time` 是带 UTC 偏移的 ISO 8601 字符串、`timezone_name` 可能退化为 `UTC`、`unix_timestamp` 是整数秒。这样设计的好处是结果既能给人看（本地时间字符串），也能给程序做算术（Unix 时间戳），还能让模型知道本地时区。它被用作 `ToolSpec.output_model`，`execute` 的返回值就是它的实例。
- **参数**：无自定义字段以外的入参。类内三个字段及其约束：`local_time: str`，`min_length=1`，不允许空串，语义为「系统本地时间，ISO 8601 格式，并包含 UTC 偏移量」；`timezone_name: str`，`min_length=1`，语义为「系统本地时区名称；无法获取名称时为 UTC」；`unix_timestamp: int`，`ge=0`，语义为「当前时刻的 Unix 时间戳（自 1970-01-01 UTC 起的整数秒）」。类级别配置 `model_config = ConfigDict(extra="forbid", strict=True)`，即禁止额外字段、严格类型校验（例如给 `unix_timestamp` 传字符串 `"123"` 会失败）。
- **返回**：类不是函数，没有返回值；实例即一次时间读取的最终结果对象。
- **内部流程**：由 Pydantic 依据字段声明与 `Field(...)` 约束生成校验器。构造时依次校验 `local_time` 非空字符串、`timezone_name` 非空字符串、`unix_timestamp` 为不小于 0 的整数，任一不满足即整体失败。实际填充工作发生在 `CurrentTimeTool.execute` 的 `return CurrentTimeOutput(...)` 处。
- **异常/边界**：字段约束不满足时抛 Pydantic `ValidationError`。特别注意 `unix_timestamp` 的 `ge=0` 约束在正常系统时钟下不会触发，只有当系统时间被设置到 1970 年之前（异常环境）才会校验失败。其他空值/非法值由 Pydantic 统一处理，本文件无额外处理。
- **同文件关系**：被 `CurrentTimeTool.spec` 通过 `output_model=CurrentTimeOutput` 引用，并在 `CurrentTimeTool.execute` 中被构造为返回值。它不调用本文件任何函数。

### `CurrentTimeTool(BaseTool)` （第 39 行）
- **作用**：这是本文件的核心工具类，代表「读取当前系统本地时间」这一能力，继承项目 `core` 模块的 `BaseTool` 基类以获得统一的调用、校验与元数据协议。它本身只承载类级 `spec` 元数据，真正的读取逻辑放在 `execute` 方法里。`spec` 用 `ToolSpec` 描述了工具的全部对外契约：名字 `system.current_time`、英文描述（告诉模型「需要真实当前时间时用它，它不接受参数、不做写操作」）、版本 `1.0.0`、输入/输出模型、`side_effect="read"`（只读副作用，权限审批层可据此放行）、`permissions=()`（不需要任何权限）、`timeout_seconds=5.0`（读取系统时钟几乎瞬时，5 秒是宽松上限）、`idempotent=True`（同一时刻重复调用语义一致）、`parallel_safe=True`（可与其他工具并发执行）、`max_concurrency=None`（不额外限制并发数）、标签 `("system", "time", "clock", "datetime")` 以及一段中文 `guidance`，明确提示模型「需要现在几点、今天几号或做时间推理（近两天、本周）时先调用它拿到真实时间，不要凭模型内部的时间作答」。这个类在运行时由 `create_tool()` 实例化，然后被工具注册表收录，最终由 Agent 在一次工具调用中触发 `execute`。
- **参数**：类定义本身没有构造参数（未定义 `__init__`，沿用 `BaseTool` 的默认构造）。类体内声明了一个类属性 `spec`，其构造参数含义为：`name` 工具唯一标识；`description` 给模型看的自然语言说明；`version` 版本号；`input_model` 输入校验模型；`output_model` 输出校验模型；`side_effect` 副作用类别，此处为只读 `read`；`permissions` 所需权限元组，此处为空；`timeout_seconds` 单次调用超时秒数 5.0；`idempotent` 是否幂等；`parallel_safe` 是否可并行；`max_concurrency` 并发上限，`None` 表示不限制；`tags` 检索标签；`guidance` 面向模型的使用指引。
- **返回**：类不是函数；实例化后得到一个可注册、可调用的工具对象，其能力通过 `execute` 暴露。
- **内部流程**：类创建时，`ToolSpec(...)` 被求值并绑定为类属性 `spec`，把输入输出模型类对象、超时、权限等一次性固化下来。运行时 `BaseTool` 框架读取该 `spec` 生成工具 schema 供 LLM 选择，并在真正调用时负责参数校验、超时控制与结果序列化；本类自身不覆写这些流程，只提供 `execute` 实现。
- **异常/边界**：类定义阶段若 `ToolSpec` 参数类型不符会在导入时抛错；`spec` 声明本身不做运行时兜底。由于没有任何可变实例状态，实例化不会因环境问题失败。
- **同文件关系**：引用了本文件的 `CurrentTimeInput` 与 `CurrentTimeOutput` 作为输入输出模型；它定义了方法 `execute`，并被本文件的 `create_tool()` 实例化返回。

### `CurrentTimeTool.execute(self, arguments: CurrentTimeInput) -> CurrentTimeOutput` （第 64 行）
- **作用**：真正执行「读一次系统时间」的方法，是工具被 Agent 调用时唯一会跑到的业务代码。它先取当前本地时间（带时区信息的 aware datetime），再从中抽取出时区名、把同一时刻换算成 UTC 后取整数 Unix 时间戳，最后把三者打包成 `CurrentTimeOutput` 返回。之所以要同时给出「本地时间字符串 + 时区名 + Unix 时间戳」，是因为三者用途不同：本地时间字符串方便模型直接读懂「现在是几点、几号」；时区名让模型知道这个本地时间属于哪个时区，避免跨时区推理错误；Unix 时间戳是无歧义的绝对时刻，便于上层程序做时间差计算或持久化。方法对入参完全不做使用（第一行就 `del arguments`），这是刻意的：工具没有参数，但仍必须接受框架传入的参数字典，删除它可明确表达「这里不依赖任何调用方输入」，同时避免静态检查工具报「未使用参数」的告警。它会在每一次工具调用时执行一次，通常耗时在微秒级。
- **参数**：`self` 为工具实例本身（`CurrentTimeTool` 对象），无实例状态被读取；`arguments: CurrentTimeInput` 是框架按 `input_model` 校验后传入的输入对象，类型为 `CurrentTimeInput`，其合法内容为空（不允许任何额外字段）。该方法不读取它的任何字段，第一行即 `del arguments` 显式丢弃。
- **返回**：返回一个 `CurrentTimeOutput` 实例，包含三个字段：`local_time`（`str`，由 `datetime.now().astimezone()` 得到的本地 aware datetime 经 `isoformat(timespec="seconds")` 格式化，形如 `2025-01-02T15:04:05+08:00`，精确到秒、省略微秒，并带 UTC 偏移）；`timezone_name`（`str`，取 `local_now.tzname()`，若为 `None` 则回退为字面量 `"UTC"`，保证非空以满足 `min_length=1`）；`unix_timestamp`（`int`，由 `local_now.astimezone(UTC).timestamp()` 取整得到，即同一时刻的 Unix 秒数）。正常路径下必然返回该对象，没有其他返回分支。
- **内部流程**：第一步 `del arguments` 丢弃入参；第二步 `local_now = datetime.now().astimezone()` 获取当前本地时间——`datetime.now()` 返回不带时区的 naive 本地时间，`.astimezone()` 在无参数调用时按系统本地时区补上 `tzinfo`，得到 aware 的本地时间，这一步同时决定了后面 ISO 字符串里的 UTC 偏移量；第三步 `timezone_name = local_now.tzname() or "UTC"` 取时区缩写名（如 `China Standard Time`、`CST`、`UTC`），若系统无法提供名称则用 `or` 兜底成 `"UTC"`；第四步 `unix_timestamp = int(local_now.astimezone(UTC).timestamp())` 把本地 aware 时间转换到 UTC 时区，再调用 `timestamp()` 得到浮点秒数并用 `int()` 截断为整数秒（截断而非四舍五入，结果恒为向下取整的当前秒）；第五步用这三个局部变量构造并 `return CurrentTimeOutput(...)`，其中 `local_time` 通过 `local_now.isoformat(timespec="seconds")` 生成，`timespec="seconds"` 保证输出不带微秒、只到秒。整个流程没有循环、没有条件分支（除 `or` 短路外），也没有调用任何本文件其他函数。
- **异常/边界**：入参被立即丢弃，因此不存在因参数非法而失败的分支（非法参数在进入本方法前已由 Pydantic 校验拦截）。可能的环境级异常包括：极端受限环境中 `astimezone()` 无法确定本地时区时，CPython 会退化为使用 UTC 偏移 0 而非抛错；`tzname()` 返回 `None` 时由 `or "UTC"` 兜底，不会产生空字符串从而不会触发输出模型的 `min_length=1` 校验失败；若系统时钟被设置到 1970-01-01 之前，`unix_timestamp` 会为负数，从而触发 `CurrentTimeOutput` 中 `ge=0` 的校验并抛 `ValidationError`，本文件对此没有额外处理；5 秒超时由框架依据 `spec.timeout_seconds` 控制，本方法自身没有超时逻辑。总体而言无特殊异常处理代码。
- **同文件关系**：调用关系上它不调用本文件任何函数或方法（只使用标准库 `datetime`/`UTC` 与外部 `core` 基类提供的能力）；被调用关系上，它由 `BaseTool` 框架在工具被 Agent 触发时调用，而承载它的 `CurrentTimeTool` 实例由本文件的 `create_tool()` 创建。

### `create_tool() -> BaseTool` （第 77 行）
- **作用**：这是本文件对外暴露的唯一模块级工厂函数，也是自动发现器（tool auto-discovery）约定的「唯一零参数入口」。项目在启动时扫描 `tool/` 目录下的模块，找到名为 `create_tool` 的函数并以零参数方式调用它，从而拿到该模块提供的工具实例并注册进工具表；同时模块级常量 `TOOL_ENABLED = True` 作为开关，让发现器判断是否加载本模块。之所以不直接暴露 `CurrentTimeTool` 类，而要用工厂函数包一层，是为了给所有工具模块一个统一、稳定的实例化契约（发现器无需知道每个模块里具体类叫什么名字，也便于将来在工厂里注入配置或做条件构造），并且每次调用都返回一个全新实例，避免多个注册方共享同一个对象带来的状态耦合。
- **参数**：无参数。函数签名为零参数，发现器只能以 `create_tool()` 形式调用。
- **返回**：返回类型标注为 `BaseTool`（来自 `core`），实际返回的是 `CurrentTimeTool()` 新建实例。因为 `CurrentTimeTool` 未定义 `__init__` 且无实例状态，返回对象是一个可直接使用的只读时间工具。任何情况下都返回实例，没有提前返回或返回 `None` 的分支。
- **内部流程**：只有一步——执行 `CurrentTimeTool()` 并作为返回值。构造过程中 `BaseTool` 的初始化会读取类属性 `spec`（若基类有相关初始化逻辑），把工具元数据挂到实例上；本函数不做任何参数校验、缓存、单例控制或异常捕获。
- **异常/边界**：若 `CurrentTimeTool` 的构造（含基类 `__init__`）抛错，异常会原样向上传播给发现器，本函数不捕获、不兜底。无其他边界情况。
- **同文件关系**：它直接实例化并返回本文件定义的 `CurrentTimeTool` 类，因此依赖该类及其 `spec`；它不调用本文件的 `execute`，`execute` 由框架在后续工具调用时触发。本文件中没有其他函数调用 `create_tool()`，调用方是模块外的工具自动发现机制。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `CurrentTimeInput` | 定义 `system.current_time` 工具的输入模型，不允许任何参数，用 `extra="forbid"` 阻止模型乱传参。 |
| `CurrentTimeOutput` | 定义工具输出模型，固定为本地 ISO 8601 时间字符串、时区名和整数 Unix 时间戳三个字段。 |
| `CurrentTimeTool` | 继承 `BaseTool` 的时间工具类，通过类属性 `spec` 声明工具名、描述、只读副作用、5 秒超时、幂等可并行等元数据。 |
| `CurrentTimeTool.execute` | 实际读取一次系统本地时间，换算 UTC 时间戳并打包成 `CurrentTimeOutput` 返回，忽略全部入参。 |
| `create_tool` | 供工具自动发现器调用的零参数工厂，返回一个新的 `CurrentTimeTool` 实例。 |
