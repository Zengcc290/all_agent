# tool/read_update_logs.py

## 一、这个文件是干什么的

这个文件实现了 Agent 运行时里的一个「批量读取项目更新日志」工具，工具全名是 `system.read_update_logs`。它的核心职责是：让模型用**一次工具调用**就把一段连续的更新日志（update log）读回来，而不是像 `system.read_update_log` 那样一次只能读一行、需要反复多轮对话。

底层存储仍然是 SQLite，仓库（Repository）层依然只能按「一个不可变行 / 一个 ID」来查询；这个文件把「逐条循环」的逻辑搬进了工具内部，所以对外暴露的是「范围读取」，对内仍然是「一条一条地取」。这样做的动机在文件开头的模块 docstring 里写得很清楚：审计完整历史时，不应该每一行都消耗一轮 LLM 交互。

文件里包含四类东西：两个 Pydantic 模型（输入模型 `ReadUpdateLogsInput`、输出模型 `ReadUpdateLogsOutput`，后者复用了单条读取工具的输出模型作为元素类型）、一个继承自 `BaseTool` 的工具类 `ReadUpdateLogsTool`（带 `ToolSpec` 元数据声明和一个 `execute` 执行体）、以及一个工厂函数 `create_tool`。另外还有模块级开关 `TOOL_ENABLED = True` 和 `__all__` 导出清单。

在项目运行时，工具注册机制会导入这个模块、读取 `ToolSpec`（名称、描述、输入输出模型、超时、并发上限、标签、给模型看的使用指引），由框架负责把 JSON 参数校验成 `ReadUpdateLogsInput`、调用 `execute`、再把 `ReadUpdateLogsOutput` 序列化回模型。文件里不做任何 I/O 之外的花活：校验范围、查最新 ID、取一段记录、拼装返回，仅此而已。

## 二、函数与类逐条详解

### `ReadUpdateLogsInput` （第 20 行）

- **作用**：这是批量读取工具的**入参模型**，用 Pydantic 定义模型可以传入的两个字段：起始 ID `start_id` 和结束 ID `end_id`。它存在的意义是把「参数形状」和「参数合法性」从 `execute` 的业务逻辑里剥离出来，由框架在调用工具前统一做类型与范围校验。模型配置 `extra="forbid"` 意味着模型多传任何未声明字段都会直接报错，避免拼错参数名时被静默忽略；`strict=True` 意味着不做宽松类型转换（例如字符串 `"5"` 不会被自动当成整数 5）。它还挂了一个 `model_validator`，用于做「跨字段」的联合校验，这是单个 `Field` 约束做不到的。
- **参数**：类本身没有构造参数；它的两个字段是：`start_id: int`，默认值 `1`，约束 `ge=1`（必须是大于等于 1 的整数），语义是「要读取的第一个更新 ID，闭区间包含」；`end_id: int | None`，默认值 `None`，约束 `ge=1`，语义是「要读取的最后一个更新 ID，闭区间包含；不传时表示一直读到当前最新 ID」。
- **返回**：作为模型类，实例化后返回一个 `ReadUpdateLogsInput` 实例，其 `start_id` 为整数、`end_id` 为整数或 `None`。
- **内部流程**：Pydantic 在构造时先按 `ConfigDict(extra="forbid", strict=True)` 检查字段名与类型，再对每个字段套用 `Field` 上的 `ge` 约束，最后执行 `mode="after"` 的模型校验器 `validate_range`（见下一条）。任何一步失败都会抛出 `pydantic.ValidationError`。
- **异常/边界**：字段缺失时用默认值（`start_id=1`、`end_id=None`），不算错误；`start_id` 或 `end_id` 小于 1、类型不是整数（严格模式）、或出现未声明的额外字段时抛 `ValidationError`；跨字段的非法组合由 `validate_range` 抛 `ValidationError`（内部是 `ValueError`）。
- **同文件关系**：被 `ReadUpdateLogsTool.execute` 用作 `arguments` 的类型并做 `isinstance` 检查；被 `ReadUpdateLogsTool.spec` 里的 `input_model` 引用；内部调用了自己的校验器 `validate_range`。它也出现在模块的 `__all__` 中。

### `ReadUpdateLogsInput.validate_range(self) -> ReadUpdateLogsInput` （第 34 行）

- **作用**：这是输入模型的跨字段校验器，负责保证「请求的范围是合理的」。它做两件事：第一，如果调用方同时给了 `start_id` 和 `end_id`，必须满足 `end_id >= start_id`，否则范围是倒着的，没有意义；第二，如果给了 `end_id`，那么一次请求的行数 `end_id - start_id + 1` 不能超过常量 `UPDATE_LOG_READ_RANGE_MAX`（单次上限，从 `constants` 导入，工具描述里写明是 100）。这个上限是保护措施：防止模型一次请求几万行把上下文、内存或 SQLite 查询拖垮。
- **参数**：`self`，即已经完成字段级校验的 `ReadUpdateLogsInput` 实例；无其它参数。
- **返回**：校验通过时返回 `self`，即同一个实例（`mode="after"` 校验器的约定：返回的对象会成为最终构造结果）。
- **内部流程**：第一步判断 `self.end_id is not None and self.end_id < self.start_id`，成立就抛 `ValueError("end_id must be greater than or equal to start_id")`；第二步判断 `self.end_id is not None and self.end_id - self.start_id + 1 > UPDATE_LOG_READ_RANGE_MAX`，成立就抛 `ValueError`，消息里用 f-string 拼出实际上限值（"requested range exceeds the maximum of {N} records per call"）。两个分支都不成立时直接 `return self`。注意 `end_id is None` 时两个检查都跳过，因为「不传结束 ID」的语义是「读到最新」，长度由 `execute` 在运行时才知道，无法在这里校验。
- **异常/边界**：抛 `ValueError`（会被 Pydantic 包装成 `ValidationError`）的条件就是上面两种：范围倒置、范围超长。`end_id=None` 时不做长度检查，是刻意的边界放行；`start_id` 本身小于 1 的情况已经在字段层被 `ge=1` 拦下，这里不再重复判断。
- **同文件关系**：由 Pydantic 在构造 `ReadUpdateLogsInput` 时自动调用（被本类的模型配置触发），不调用本文件里的其它函数。

### `ReadUpdateLogsOutput` （第 46 行）

- **作用**：这是批量读取工具的**出参模型**，描述工具成功执行后返回给框架（最终给模型看）的结构。它包含四部分：读到的记录列表 `records`、本次实际读取的起始 ID `start_id`、本次实际读取的结束 ID `end_id`、以及当前库里最新的更新 ID `latest_update_id`。把「实际范围」和「最新 ID」一起返回，是为了让模型知道「我这次读了哪一段」以及「还有没有更新」，方便它决定要不要继续往下读，而不必再额外调一次工具问最新 ID。列表元素类型是单条读取工具的输出模型 `ReadUpdateLogOutput`（从 `tool.read_update_log` 导入并复用），这样两条工具在「一条记录长什么样」上保持完全一致，模型看到的字段结构不会分裂。
- **参数**：类本身没有构造参数；字段为：`records: list[ReadUpdateLogOutput]`，约束 `min_length=1`（至少一条，空列表非法），描述为「按 ID 升序读取的更新记录」；`start_id: int`，约束 `ge=1`；`end_id: int`，约束 `ge=1`；`latest_update_id: int`，约束 `ge=0`，描述为「当前已存储的最大更新 ID」。注意 `latest_update_id` 允许为 0（空库时最新 ID 为 0 是合理表示），而 `start_id`/`end_id` 不允许为 0。
- **返回**：实例化后返回一个 `ReadUpdateLogsOutput` 实例，其 `records` 是一个非空的 `ReadUpdateLogOutput` 列表，其余三个字段为整数。
- **内部流程**：Pydantic 在构造时先做 `extra="forbid"` 与 `strict=True` 检查，再逐个字段套用 `ge` / `min_length` 约束；由于 `records` 的元素类型是另一个 BaseModel，Pydantic 还会对列表中每一项递归校验（通常元素已经是模型实例，直接复用）。
- **异常/边界**：`records` 为空列表会因 `min_length=1` 抛 `ValidationError`；`start_id`/`end_id` 小于 1 或 `latest_update_id` 小于 0 抛 `ValidationError`；出现未声明字段抛 `ValidationError`。
- **同文件关系**：在 `ReadUpdateLogsTool.execute` 中被构造并返回，是 `execute` 的返回类型；同时被 `ReadUpdateLogsTool.spec` 的 `output_model` 引用；也出现在模块 `__all__` 中。

### `ReadUpdateLogsTool` （第 61 行）

- **作用**：这是工具本体，继承自 `core.BaseTool`，把「范围读取更新日志」这个能力暴露给 Agent 运行时。类上定义了一个类属性 `spec`（`ToolSpec` 实例），里面声明了工具的全部元数据：名字 `system.read_update_logs`、给模型看的长描述（说明单次最多 `UPDATE_LOG_READ_RANGE_MAX` 条、内部逐 ID 循环、升序返回、因此完整审计只需少量工具调用）、版本 `1.0`、输入输出模型、副作用类型 `read`（只读，不改状态）、空权限元组、超时 `10.0` 秒、`idempotent=True`（同样参数重复调用结果一致）、`parallel_safe=True`（可并发执行）、`max_concurrency=4`、标签 `("update-log", "audit", "read", "batch", "project")`，以及一段中文 `guidance`：需要连续审计一段历史时用它，单次最多 100 条、升序返回，比逐条调用省轮次，只看一条时用 `system.read_update_log`。这个 `guidance` 是直接喂给模型的行为提示，属于本文件区别于单条工具的关键设计。
- **参数**：类定义本身无参数；实例化时的参数见 `__init__`。
- **返回**：类本身不返回值；实例化返回 `ReadUpdateLogsTool` 对象，其行为由 `execute` 提供。
- **内部流程**：类体只做两件事——定义 `spec` 类属性、定义 `__init__` 与 `execute` 两个方法。框架在注册工具时读取 `spec`，在调用工具时构造（或复用）实例并调用 `execute`。
- **异常/边界**：`spec` 在导入期就被求值，`UPDATE_LOG_READ_RANGE_MAX` 若不存在会在导入阶段就 `ImportError`/`NameError` 暴露出来，属于启动期失败而非运行期失败；类本身不抛业务异常。
- **同文件关系**：包含 `__init__` 与 `execute` 两个方法；被 `create_tool` 实例化；其 `input_model`/`output_model` 指向本文件的 `ReadUpdateLogsInput` / `ReadUpdateLogsOutput`；出现在模块 `__all__` 中。

### `ReadUpdateLogsTool.__init__(self, repository: UpdateLogRepository | None = None) -> None` （第 87 行）

- **作用**：构造工具实例，并决定它使用哪个数据源。它接受一个可选的 `repository`（类型是 `core.update_log` 里的 `UpdateLogRepository`），如果调用方传了就用调用方的，没传（`None`）就自己新建一个默认仓库。这个「依赖注入 + 默认兜底」的写法让生产环境零配置即可用，同时让测试或其它调用方可以塞入替身对象（文件里 `execute` 的一段注释就明确提到「兼容轻量级的 repository 替身」）。
- **参数**：`self`，实例本身；`repository: UpdateLogRepository | None`，默认 `None`，含义是「外部传入的更新日志仓库；为 `None` 时自行创建默认仓库」。约束是它至少要能响应 `latest_id()` 和 `get()`（可选地响应 `get_range()`）。
- **返回**：`None`（构造函数不返回值），副作用是把选定的仓库对象存到实例属性 `self.repository` 上。
- **内部流程**：一行三元表达式：`self.repository = repository if repository is not None else UpdateLogRepository()`。注意判断用的是 `is not None` 而不是真值判断，所以传入一个「假值但非 None」的替身对象也不会被替换掉。
- **异常/边界**：如果传入了非 `None` 但不具备所需方法的对象，本方法不会报错，错误会推迟到 `execute` 调用 `latest_id()` / `get_range()` / `get()` 时以 `AttributeError` 形式暴露；`UpdateLogRepository()` 的构造失败（例如底层数据库不可用）会在此处直接向上抛。对 `None` 的处理就是走默认构造。
- **同文件关系**：被 `create_tool` 间接调用（`ReadUpdateLogsTool()`）；它为 `execute` 提供 `self.repository`。

### `ReadUpdateLogsTool.execute(self, arguments: ReadUpdateLogsInput) -> ReadUpdateLogsOutput` （第 90 行）

- **作用**：这是工具的真正执行体，把「读取 ID 区间 `[start_id, end_id]` 的全部更新日志」这件事一次做完。它按顺序完成四件事：校验参数类型、确定实际结束 ID（不传则取当前最新 ID）、拿到这一段的原始记录并确认没有缺行、把每条记录包装成输出模型并附上 `latest_update_id` 后整体返回。它是「循环内移」这个设计思想的落点：无论要读 1 条还是 100 条，模型都只花一次工具调用。文件开头的 docstring 描述的正是这个行为。
- **参数**：`self`，工具实例（提供 `self.repository`）；`arguments: ReadUpdateLogsInput`，已经过 Pydantic 校验的入参对象，其中 `start_id >= 1`、`end_id` 要么是 `>= start_id` 的整数要么是 `None`、且长度不超过 `UPDATE_LOG_READ_RANGE_MAX`。
- **返回**：返回 `ReadUpdateLogsOutput`，包含按 ID 升序的 `records` 列表（每个元素是 `ReadUpdateLogOutput`，且每条都被塞入了当前的 `latest_update_id`）、实际使用的 `start_id`、解析后的 `end_id`、以及查询时的 `latest_update_id`。
- **内部流程**：
  1. 类型守卫：`if not isinstance(arguments, ReadUpdateLogsInput): raise TypeError(...)`，防止框架或调用方直接传 dict。
  2. 调用 `self.repository.latest_id()` 得到 `latest_update_id`（当前库里最大的更新 ID）。
  3. 如果 `latest_update_id < arguments.start_id`，说明起点就已经超出库里已有的范围，抛 `LookupError`，消息里同时给出请求的 ID 和最新 ID。
  4. 解析结束 ID：`end_id = arguments.end_id if arguments.end_id is not None else latest_update_id`；即「不传就一直读到最新」。
  5. 如果 `end_id > latest_update_id`，说明请求的终点超过库里已有范围，抛 `LookupError`，消息里给出请求终点与最新 ID。
  6. 取数据：优先走批量接口——`if hasattr(self.repository, "get_range")` 为真则调用 `self.repository.get_range(arguments.start_id, end_id)`；否则进入兼容分支（注释说明是给轻量级仓库替身用的），用列表推导对 `range(start_id, end_id + 1)` 里的每个 ID 调 `self.repository.get(update_id)`，再用列表推导把 `None` 结果过滤掉。两条路径的结果都赋给 `raw_records`。
  7. 完整性检查：算出 `expected_count = end_id - arguments.start_id + 1`；若 `len(raw_records) != expected_count`，说明中间缺行。此时用集合推导 `present = {record["update_id"] for record in raw_records}` 得到已存在的 ID 集合，再用生成器 + `next(...)` 找出区间内第一个不在集合里的 `missing` ID，抛 `LookupError(f"update log entry {missing} was not found")`。这里假设原始记录是支持 `["update_id"]` 下标访问的映射（dict 或类似结构）。
  8. 组装输出：新建空列表 `records`，遍历 `raw_records`，对每条记录先就地写入 `record["latest_update_id"] = latest_update_id`（这样每个单条输出模型都能带上当前最新 ID），再 `ReadUpdateLogOutput(**record)` 构造元素并追加。因为 `raw_records` 是按升序 ID 取得的，列表天然升序。
  9. 返回 `ReadUpdateLogsOutput(records=..., start_id=arguments.start_id, end_id=end_id, latest_update_id=latest_update_id)`。
- **异常/边界**：
  - `arguments` 类型不对 → `TypeError`。
  - `start_id` 超过最新 ID → `LookupError`（起点越界）。
  - `end_id` 超过最新 ID → `LookupError`（终点越界）。
  - 区间内缺行（例如被删除或写入不连续）→ `LookupError`，消息里指出第一个缺失的 ID。
  - 兼容分支下，`get()` 返回 `None` 会被过滤，因此缺行会走到上面那条缺行检查；但如果在 `get_range` 分支下 `record` 不是映射类型，`record["update_id"]` 会抛 `KeyError`/`TypeError`。
  - `record` 里若含有 `ReadUpdateLogOutput` 未声明的字段，`ReadUpdateLogOutput(**record)` 会抛 `ValidationError`（该模型同样用 `extra="forbid"`）。
  - 没有显式的超时处理：10 秒超时由 `ToolSpec.timeout_seconds` 在框架层控制，本函数自身不做计时或重试。
  - 空区间不可能出现：`validate_range` 已保证 `end_id >= start_id`，`expected_count` 至少为 1，`ReadUpdateLogsOutput.records` 的 `min_length=1` 也与此一致。
- **同文件关系**：接收本文件的 `ReadUpdateLogsInput`，构造并返回本文件的 `ReadUpdateLogsOutput`；使用 `__init__` 注入的 `self.repository`；引用了本文件导入的 `ReadUpdateLogOutput`（来自 `tool.read_update_log`）作为元素模型。本文件内没有其它函数调用它，它是工具被框架调用的入口点。

### `create_tool() -> BaseTool` （第 133 行）

- **作用**：这是工具工厂函数，供框架的工具发现/注册流程调用。它把「如何构造这个工具」这件事集中到一个无参函数里，框架不需要知道 `ReadUpdateLogsTool` 的类名、也不需要知道它构造时需要什么依赖，直接调用 `create_tool()` 就能拿到一个可用的工具实例。返回类型标注为基类 `BaseTool`，是刻意的抽象：调用方只依赖 `BaseTool` 契约（能读 `spec`、能 `execute`），不依赖具体实现类。
- **参数**：无参数。
- **返回**：返回一个新构造的 `ReadUpdateLogsTool` 实例，静态类型标注为 `BaseTool`。每次调用都会新建实例（并且因为 `__init__` 里 `repository=None`，每个实例会各自创建一个默认的 `UpdateLogRepository`）。
- **内部流程**：函数体只有一行 `return ReadUpdateLogsTool()`，即用默认参数构造工具。
- **异常/边界**：无特殊处理；若 `UpdateLogRepository()` 构造失败（如底层存储不可用），异常会原样向上抛出。
- **同文件关系**：调用本文件的 `ReadUpdateLogsTool`（进而触发 `ReadUpdateLogsTool.__init__`）；不被本文件内其它代码调用，只对外暴露；名字列在模块 `__all__` 中。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `ReadUpdateLogsInput` | 批量读取工具的入参模型，声明起始 ID 与可选结束 ID，并强制禁止额外字段和宽松类型转换。 |
| `ReadUpdateLogsInput.validate_range` | 跨字段校验器，保证 `end_id >= start_id` 且一次请求的行数不超过 `UPDATE_LOG_READ_RANGE_MAX`。 |
| `ReadUpdateLogsOutput` | 批量读取工具的出参模型，返回升序记录列表、实际起止 ID 与当前最新更新 ID。 |
| `ReadUpdateLogsTool` | 工具本体，用 `ToolSpec` 声明 `system.read_update_logs` 的元数据与使用指引，并实现范围读取逻辑。 |
| `ReadUpdateLogsTool.__init__` | 构造工具实例，接受可选的 `UpdateLogRepository`，未传时自建默认仓库。 |
| `ReadUpdateLogsTool.execute` | 一次调用内完成区间校验、最新 ID 查询、逐条取记录、缺行检测与输出模型组装。 |
| `create_tool` | 无参工厂函数，返回一个新的 `ReadUpdateLogsTool` 实例供框架注册。 |
