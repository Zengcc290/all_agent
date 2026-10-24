# core/runtime.py

## 一、这个文件是干什么的

这个文件是整个 Agent 运行时的「工具调用执行引擎」，负责把模型（或调用方）产出的一批 `ToolCall` 真正落地执行，并把结果汇总成 `BatchToolResult` 交回上层。它内部定义了三样东西：一个内部超时标记异常 `_ExecutionTimedOut`、一个按事件循环隔离异步原语的私有数据类 `_LoopExecutionState`，以及核心类 `ToolExecutionManager`。`ToolExecutionManager` 承担了完整的一条流水线：先对每个调用做静态校验（工具是否注册、schema 版本与哈希是否过期、参数能否通过 Pydantic 严格校验、带副作用的工具是否已被确认），再根据 `depends_on` 建立调用之间的依赖图并做环检测，最后按「依赖就绪 + 并发安全」的规则分批调度，通过全局信号量和按工具粒度的信号量双重限流去执行，并对超时、输出校验失败、执行异常做统一兜底。它同时处理并发语义上的细节：`asyncio` 的同步原语绑定事件循环，因此每个事件循环单独持有一套信号量；线程型工具超时后无法真正杀死，于是槽位会延迟到后台线程结束才归还，避免出现隐藏的并发重叠。上层（例如 Web 应用或 Agent 主循环）只需要构造一个 manager，把一批 `ToolCall` 和 `ExecutionContext` 丢给 `execute_batch`，就能拿到逐条的成功数据或结构化 `ToolError`，而不用关心调度、限流、超时和错误码映射。

## 二、函数与类逐条详解

### `class _ExecutionTimedOut(Exception)` （第 18 行）
- **作用**：这是一个纯粹的内部控制信号，用来表示「运行时自己判定的执行截止时间已到」。它存在的意义是把「超时」这件事从普通异常里区分出来，使得 `_run_one` 里可以单独用一个 `except _ExecutionTimedOut` 分支把超时翻译成带 `TIMEOUT` 错误码的 `ToolError`，而不是被下面宽泛的 `except Exception` 吞掉变成含糊的 `EXECUTION_ERROR`。它不带任何字段，也不做任何逻辑，只是被抛出、被捕获。之所以要自建一个而不是直接用 `asyncio.TimeoutError`，是因为这个超时是运行时主动强制的（由 `spec.timeout_seconds` 换算出的 deadline 决定），而不是工具内部自己抛的，语义上需要分开。
- **参数**：继承自 `Exception`，未定义 `__init__`，因此接受标准异常的可变位置参数（消息文本），但本文件里抛出时一律不传消息，只做类型标记。
- **返回**：无返回（异常类）。
- **内部流程**：没有任何方法体，只有一行文档字符串；实例化即 `_ExecutionTimedOut()`，在 `_await_until` 中 `raise`，在 `_run_one` 中被捕获。
- **异常/边界**：它本身就是要抛出的异常；没有任何额外校验或状态。
- **同文件关系**：被 `_await_until` 抛出；被 `_run_one` 捕获并转换为 `TIMEOUT` 的 `ToolError`。

### `class _LoopExecutionState` （第 22 行）
- **作用**：这是一个 `@dataclass`，用来承载「属于且仅属于某一个事件循环」的异步同步原语。之所以需要它，是因为 `asyncio.Semaphore` 在争用时会被绑定到当时运行的事件循环上，同一个 manager 如果被多次 `asyncio.run` 顺序复用（每次都是新的事件循环），把信号量放在 manager 实例上就会在第二个循环里出错。把每个循环需要的原语打包成一个状态对象，再用字典按循环做键，就能让 manager 跨循环安全复用。字段里既有全局并发上限对应的信号量，也有按工具维度缓存的一批信号量。
- **参数**：数据类字段 —— `global_semaphore: asyncio.Semaphore`（必填，代表整批调用的全局并发额度）；`tool_semaphores: dict[tuple[object, ...], asyncio.Semaphore]`（可选，默认由 `field(default_factory=dict)` 生成空字典，键是工具标识元组，值是该工具自己的信号量）。
- **返回**：无（数据类）。
- **内部流程**：没有自定义方法；实例化时 `global_semaphore` 由调用方传入，`tool_semaphores` 默认初始化为空字典，后续由 `_tool_semaphore` 惰性填充。
- **异常/边界**：无特殊处理；不做任何校验，字段类型仅作注解提示。
- **同文件关系**：由 `_loop_state` 构造并缓存，被 `_run_one`、`_await_until`（间接经 `_run_one`）、`_tool_semaphore`、`_release_after_background_work` 使用。

### `class ToolExecutionManager` （第 32 行）
- **作用**：这是本文件唯一的公开类，职责是「校验、调度、执行并聚合一批工具调用」。它把一个批次的 `ToolCall` 列表变成一一对应的 `ToolResult` 列表：先做静态前置校验（注册状态、schema 一致性、参数合法性、副作用确认、依赖合法性），再把有依赖关系的调用排序成波次执行，无依赖且声明并行安全的调用一起并发跑，不安全的调用一次只跑一个，最后按原始顺序返回结果。它还负责全局与单工具两级并发限制、每个工具的超时控制、同步工具放到线程池执行、输出模型严格校验，以及把所有失败统一翻译成结构化 `ToolError`。类内部还维护了一个按事件循环索引的状态表，以便实例能在多个顺序运行的 `asyncio.run` 之间复用。
- **参数**：无（类的构造参数见 `__init__`）。
- **返回**：无（类）。
- **内部流程**：类的实例方法组织成三层 —— 入口层 `execute_batch` 负责全流程编排；辅助层 `_abort_remaining`、`_error_result`、`_limited_message` 负责构造错误结果与压缩错误消息；执行层 `_loop_state`、`_tool_semaphore`、`_run_one`、`_await_until`、`_release_after_background_work` 负责信号量管理与单个调用的实际执行。类中所有异常都尽量在 `_run_one` 内被捕获并转成 `ToolResult(ok=False)`，因此 `execute_batch` 正常路径不会因为某个工具失败而抛出。
- **异常/边界**：类本身不抛异常；其方法会抛 `ValueError`、`TypeError`、`RuntimeError`（详见各方法）。
- **同文件关系**：它使用 `_LoopExecutionState` 与 `_ExecutionTimedOut`；类内各方法之间的调用关系在下面逐条说明。

### `ToolExecutionManager.__init__(self, registry: ToolRegistry, *, max_concurrency: int = 8) -> None` （第 35 行）
- **作用**：构造一个批次执行管理器，注入工具注册表并设定全局并发上限。它在构造时就把 `max_concurrency` 严格校验一遍，避免非法值（负数、0、布尔值、非整数）在真正调度时才引发难以定位的信号量错误。构造过程还会创建两个跨事件循环共享的结构：一个是「事件循环 → 该循环的异步状态」的字典，另一个是保护这个字典的线程锁 —— 因为状态字典可能被不同线程上的多个事件循环同时访问。注意这里并不会立刻创建信号量，信号量是惰性到第一次执行时才按循环建立的。
- **参数**：`registry: ToolRegistry` —— 必填，工具注册表，后续用于 `maybe_resolve`、`confirmation_key`、`call_confirmation_key`；`max_concurrency: int = 8` —— 仅限关键字传参，正整数，代表单个事件循环内同时在跑的工具调用数量上限，同时也会作为「工具自身未声明 `max_concurrency` 时」的默认单工具并发额度。
- **返回**：`None`。
- **内部流程**：第一步用 `isinstance(max_concurrency, bool)` 排除布尔值（因为 `True` 也是 `int`），再检查不是 `int` 或 `<= 0`，任一命中即 `raise ValueError("max_concurrency must be a positive integer")`。第二步把 `registry` 与 `max_concurrency` 存到实例属性上。第三步初始化 `self._loop_states`（空字典）与 `self._loop_states_lock = threading.Lock()`，并附上注释说明为什么要按循环隔离信号量。
- **异常/边界**：`max_concurrency` 为布尔、非整数或非正数时抛 `ValueError`；`registry` 不做类型校验，传入 `None` 会在后续 `maybe_resolve` 时以属性错误的形式暴露。
- **同文件关系**：被外部调用；它创建的状态字典被 `_loop_state` 读写，锁被 `_loop_state` 使用。

### `async def execute_batch(self, calls: list[ToolCall], context: ExecutionContext | None = None, *, failure_policy: Literal["continue", "fail_fast"] = "continue") -> BatchToolResult` （第 50 行）
- **作用**：这是整个管理器的总入口，也是本文件最核心的函数。它接收一批工具调用，先把每个调用「体检」一遍：`call_id` 是否在批内重复、工具是否已注册、调用记录的 schema 版本/哈希/注册代次是否已经过期、参数能否通过工具输入模型的严格校验、以及带副作用的工具是否已经在上下文里被确认过。随后它把 `depends_on` 声明解析成调用下标之间的依赖边，并顺手检测出「依赖不存在」「依赖在批内不唯一」「依赖自己」这三类非法情况。最后它进入调度循环：每一轮挑出所有依赖已经完成的下标作为就绪集，如果就绪集为空则说明依赖图存在环，整批剩余调用全部标记为 `DEPENDENCY_CYCLE`；否则在就绪集里优先只放一个「非并行安全」的调用，若没有则整批就绪调用一起跑，用 `asyncio.gather` 并发执行并把结果写回。`failure_policy` 决定失败后是继续跑还是把剩余未开始的调用统统标记为 `ABORTED`。这个函数保证返回的 `results` 与传入的 `calls` 顺序严格一一对应，长度一致。
- **参数**：`calls: list[ToolCall]` —— 位置参数，待执行的调用列表，必须是 `list` 或 `tuple`（代码里校验 `(list, tuple)`），且每个元素都必须是 `ToolCall` 实例；允许为空列表/空元组。`context: ExecutionContext | None = None` —— 可选，执行上下文，`None` 时自动新建一个空的 `ExecutionContext()`，用于读取 `confirmed_side_effects`；传入非 `ExecutionContext` 实例会抛 `TypeError`。`failure_policy: Literal["continue", "fail_fast"] = "continue"` —— 仅限关键字传参，`"continue"` 表示某条失败不影响其它调用继续执行，`"fail_fast"` 表示一旦出现任何已完成结果（包括前置校验失败的结果）或某批执行出现失败，就把剩余未开始的调用全部中止；其它取值抛 `ValueError`。
- **返回**：`BatchToolResult`。空输入直接返回 `BatchToolResult(results=[])`；正常结束时返回的 `results` 与 `calls` 同序同长，每一项要么是成功结果（含 `data`），要么是失败结果（含 `error`）。若内部调度出现未填满的 `None` 槽位（理论上不应发生），会抛 `RuntimeError("tool scheduler finished with incomplete results")` 而不是返回不完整结果。
- **内部流程**：
  1. 参数校验：`failure_policy` 必须属于 `{"continue", "fail_fast"}`；`calls` 必须是 `list`/`tuple`；逐项 `isinstance(call, ToolCall)`；`context` 为 `None` 则替换成新的 `ExecutionContext()`，否则必须是 `ExecutionContext` 实例。
  2. 空批次短路：`if not calls: return BatchToolResult(results=[])`。
  3. 取得当前事件循环的状态：`loop_state = self._loop_state()`。
  4. 建立 `indices_by_call_id`：遍历 `calls`，把每个 `call_id` 映射到它出现的所有下标；凡出现次数大于 1 的下标收集进 `duplicate_indices`。
  5. 初始化三个与 `calls` 等长的平行数组：`tools`（解析出的 `BaseTool` 或 `None`）、`normalized_arguments`（校验后的 Pydantic 模型或 `None`）、`errors`（`ToolError` 或 `None`）。
  6. 逐条前置校验：先看是否在 `duplicate_indices` 里，是则记 `DUPLICATE_CALL_ID` 并跳过；再 `self.registry.maybe_resolve(call.tool_name)`，返回 `None` 记 `UNKNOWN_TOOL`，否则解包成 `tool, generation` 并写入 `tools[index]`；接着比对 `call.schema_version != spec.version`、`call.schema_hash != spec.schema_hash`，以及 `call.registry_generation` 非空且不等于 `generation` 的情况，任一命中记 `SCHEMA_MISMATCH` 并跳过；随后用 `spec.input_model.model_validate(call.arguments, strict=True)` 做严格参数校验，`ValidationError` 记 `INVALID_ARGUMENTS` 且消息经 `_limited_message` 压缩，其它异常打 `LOGGER.error` 并记通用的 `INVALID_ARGUMENTS`。
  7. 副作用确认：若该条已出错或 `spec.side_effect == "read"` 就跳过；否则先看工具级确认键 `self.registry.confirmation_key(spec.name)` 是否在 `context.confirmed_side_effects` 中；若 `spec.side_effect == "destructive"`，则改用调用级确认键 `self.registry.call_confirmation_key(spec.name, normalized.model_dump(mode="json"))` 覆盖判断（此时断言校验后的参数非空）。最终未确认则记 `CONFIRMATION_REQUIRED`。代码注释明确说明工具权限目前只是元数据，运行时不做限制。
  8. 依赖解析：`dependencies` 是与 `calls` 等长的邻接表。对每条调用的每个 `dependency_id`，用 `indices_by_call_id` 找匹配下标：找不到且该条尚无错误则记 `UNKNOWN_DEPENDENCY`；匹配到多个且尚无错误则记 `AMBIGUOUS_DEPENDENCY`；匹配到自己且尚无错误则记 `DEPENDENCY_CYCLE`；否则把依赖下标追加进 `dependencies[index]`。注意这里用 `if errors[index] is None` 保证不会覆盖前面的校验错误。
  9. 结果初始化：把所有已记录的 `errors` 通过 `self._error_result(calls[index], error)` 转成失败 `ToolResult` 填入 `results`；据此切分出 `remaining`（结果仍为 `None` 的下标集合）与 `completed`（已有结果的下标集合）。
  10. `failure_policy == "fail_fast"` 且 `completed` 非空时，说明前置阶段就已有失败，直接 `self._abort_remaining(...)` 把 `remaining` 全部标记为 `ABORTED`。
  11. 主调度循环 `while remaining:`：计算 `ready`（所有依赖下标都在 `completed` 中的下标）；`ready` 为空时说明依赖图成环，把 `remaining` 中每一项都写成 `DEPENDENCY_CYCLE`（消息为 "tool call dependency graph contains a cycle"），清空 `remaining` 并 `break`；否则计算 `unsafe_ready`（`tools[index].spec.parallel_safe` 为假的就绪项），`batch` 取 `unsafe_ready[:1]`（一次只跑一个不安全工具）或整个 `ready`；用 `asyncio.gather` 并发调用 `self._run_one(...)`，传入下标、调用、工具、校验后的参数、依赖下标、`results`、`context`、`loop_state`；把返回的 `(index, result)` 逐个写回 `results`，从 `remaining` 移除并加入 `completed`。
  12. 每批执行后若 `failure_policy == "fail_fast"` 且批内有 `not result.ok`，再次调用 `self._abort_remaining(...)`。
  13. 循环结束后若 `results` 中仍有 `None`，抛 `RuntimeError`；否则返回 `BatchToolResult(results=[...])`（列表推导里再次过滤 `None`，实际此时不应有 `None`）。
- **异常/边界**：`failure_policy` 非法 → `ValueError`；`calls` 非列表/元组、含非 `ToolCall` 元素、`context` 类型不对 → `TypeError`；调度收尾不完整 → `RuntimeError`。空批次安全返回空结果；重复 `call_id`、未知工具、schema 过期、参数非法、未确认副作用、未知/歧义/自依赖、依赖成环都被转成 `ToolError` 而不是抛出；单个工具的执行异常由 `_run_one` 内部消化。
- **同文件关系**：调用 `_loop_state`、`_error_result`、`_abort_remaining`、`_run_one`、`_limited_message`（间接经参数校验分支）；被外部调用者调用。

### `ToolExecutionManager._abort_remaining(self, calls: list[ToolCall], results: list[ToolResult | None], remaining: set[int]) -> None` （第 249 行）
- **作用**：把「还没开始执行」的调用统一宣告为中止。它在 `fail_fast` 策略下被用到两次：一次是前置校验阶段就已经出现失败时，把其余调用全部中止；一次是某一批并发执行中出现失败时，把剩余调用全部中止。这样调用方拿到的结果集是完整且自洽的 —— 每条调用都有明确结论（成功、具体错误，或 `ABORTED`），不会出现结果缺失导致上层无法对齐的问题。它通过原地修改 `results` 列表和清空 `remaining` 集合来通知调用方「调度已经结束」。
- **参数**：`calls: list[ToolCall]` —— 原始调用列表，用于取出每条调用的 `call_id` 与 `tool_name` 来构造结果；`results: list[ToolResult | None]` —— 与 `calls` 等长的结果槽位列表，本函数会原地写入未完成位置；`remaining: set[int]` —— 尚未完成的下标集合，函数结束时会被清空。
- **返回**：`None`（结果通过 `results` 与 `remaining` 原地生效）。
- **内部流程**：遍历 `remaining` 中的每个下标，用 `self._error_result(calls[index], ToolError(code="ABORTED", message="batch aborted after a tool failure"))` 生成失败结果写入 `results[index]`；循环结束后执行 `remaining.clear()`，使 `execute_batch` 的主调度循环条件 `while remaining:` 变为假而退出。
- **异常/边界**：无特殊处理；不做下标越界检查，依赖调用方传入合法的 `remaining` 与等长 `results`。若 `remaining` 为空集合，循环不执行，仅清空（无副作用）。
- **同文件关系**：调用 `_error_result`；被 `execute_batch` 调用。

### `@staticmethod ToolExecutionManager._error_result(call: ToolCall, error: ToolError) -> ToolResult` （第 262 行）
- **作用**：这是一个极小的构造工具，把「一条调用 + 一个错误」组装成统一的失败结果对象。它被多处复用：前置校验失败、`fail_fast` 中止、依赖失败、超时、输出校验失败、执行异常，最终都通过它产出形状一致的 `ToolResult`，保证上层只需要看 `ok` 与 `error` 两个字段就能判断和处理失败。声明为静态方法是因为它不依赖任何实例状态，只是纯粹的字段搬运。
- **参数**：`call: ToolCall` —— 出错的那条调用，提供 `call_id` 与 `tool_name`；`error: ToolError` —— 已经构造好的结构化错误，包含 `code`、`message`，可选 `retryable`。
- **返回**：`ToolResult`，字段为 `call_id=call.call_id`、`tool_name=call.tool_name`、`ok=False`、`error=error`；`data` 保持默认（无数据）。
- **内部流程**：单条 `return ToolResult(...)` 语句，直接透传两个标识字段并把 `ok` 固定为 `False`。
- **异常/边界**：无特殊处理；不做 `call`/`error` 的类型或非空校验。
- **同文件关系**：被 `execute_batch`、`_abort_remaining`、`_run_one` 调用。

### `ToolExecutionManager._loop_state(self) -> _LoopExecutionState` （第 268 行）
- **作用**：取得「当前正在运行的事件循环」对应的异步状态对象，没有就新建一个。这是为了让 `asyncio.Semaphore` 永远只属于一个事件循环，从而支持同一个 manager 实例被多次顺序 `asyncio.run` 复用。它同时承担了清理职责：顺手把已经关闭的旧循环对应的状态从字典里删掉，避免长期运行的服务在反复创建/销毁事件循环时把字典撑大。整个读写都在线程锁保护下进行，因为同一个 manager 可能被不同线程上各自的事件循环使用。
- **参数**：无（`self` 之外没有参数）。
- **返回**：`_LoopExecutionState`，其中 `global_semaphore` 的初始容量是构造时保存的 `self.max_concurrency`；同一循环重复调用返回同一个对象（引用相等），不同循环返回各自独立的对象。
- **内部流程**：先 `loop = asyncio.get_running_loop()`（必须在协程内调用）；进入 `with self._loop_states_lock:` 临界区；用列表推导找出字典中「不是当前循环且 `is_closed()` 为真」的键作为 `closed_loops`，逐个 `del` 掉；再 `self._loop_states.get(loop)` 查当前循环的状态；为 `None` 时构造 `_LoopExecutionState(global_semaphore=asyncio.Semaphore(self.max_concurrency))` 并写入字典；最后返回该状态。
- **异常/边界**：不在事件循环内调用会由 `asyncio.get_running_loop()` 抛 `RuntimeError`；其它无特殊处理。清理逻辑不会误删当前循环的状态（条件里显式排除了 `known_loop is not loop`）。
- **同文件关系**：构造并使用 `_LoopExecutionState`；被 `execute_batch` 调用，其结果被传递给 `_run_one`。

### `ToolExecutionManager._tool_semaphore(self, loop_state: _LoopExecutionState, tool: BaseTool) -> asyncio.Semaphore` （第 286 行）
- **作用**：为某个具体工具取得（或惰性创建）它自己的并发信号量，用来限制「同一个工具」的并发度，与全局信号量配合形成两级限流。容量规则是：如果工具声明 `parallel_safe` 为假，容量固定为 1（即该工具串行执行）；否则取工具自带的 `spec.max_concurrency`，未设置时回落到管理器的 `max_concurrency`。缓存键把工具的 `name`、`version`、`schema_hash`、`parallel_safe` 与算出的 `capacity` 一起打包，这样同一个工具在重新注册/升级导致 schema 变化后不会被错误复用旧信号量。
- **参数**：`loop_state: _LoopExecutionState` —— 当前事件循环的状态对象，其 `tool_semaphores` 字典充当缓存；`tool: BaseTool` —— 需要限流的工具实例，通过 `tool.spec` 读取 `name`、`version`、`schema_hash`、`parallel_safe`、`max_concurrency`。
- **返回**：`asyncio.Semaphore`，容量为上面算出的 `capacity`；同一 `loop_state` 下相同键重复调用返回同一个实例。
- **内部流程**：读取 `spec = tool.spec`；用三元表达式算 `capacity`（`not spec.parallel_safe` → 1，否则 `spec.max_concurrency or self.max_concurrency`，这里用 `or` 意味着 `max_concurrency` 为 `0` 或 `None` 时都会回落到管理器默认值）；把五个元素组成 `key` 元组；在 `loop_state.tool_semaphores` 中查 `semaphore`；为 `None` 时新建 `asyncio.Semaphore(capacity)` 并写回字典；返回 `semaphore`。
- **异常/边界**：无特殊处理；没有加锁（因为同一循环内是单线程协作式调度，字典读写不会被真正并发打断）。若 `spec.max_concurrency` 为负数则由 `asyncio.Semaphore` 自行处理（本文件不做校验）。
- **同文件关系**：使用 `_LoopExecutionState`；被 `_run_one` 调用。

### `async def ToolExecutionManager._run_one(self, index: int, call: ToolCall, tool: BaseTool | None, arguments: BaseModel | None, dependency_indices: list[int], results: list[ToolResult | None], context: ExecutionContext, loop_state: _LoopExecutionState) -> tuple[int, ToolResult]` （第 302 行）
- **作用**：这是真正「跑一个工具调用」的函数，是调度器交给 `asyncio.gather` 的执行单元。它负责获取两级信号量、算出本次执行的绝对截止时间、判断工具是协程还是同步函数并选择对应执行方式（同步函数丢进 `asyncio.to_thread` 以免阻塞事件循环）、在超时前等待结果、把返回值交给工具的 `output_model` 做严格校验并转成字典，最后把成功或失败都封装成 `(index, ToolResult)` 返回，而不是向外抛异常。它还处理了一个棘手的边界：线程型工具无法被取消，超时后线程仍在跑，因此它不会立刻归还信号量，而是注册一个完成回调，等后台线程真正结束再释放两个槽位，防止超时导致隐藏的并发重叠。返回下标是为了让 `asyncio.gather` 的结果能对回原始位置。
- **参数**：`index: int` —— 该调用在批次中的下标，原样回传用于回填结果；`call: ToolCall` —— 原始调用，提供 `call_id` 与 `tool_name`；`tool: BaseTool | None` —— 已解析的工具，`None` 或 `arguments` 为 `None` 时视为调度器错误；`arguments: BaseModel | None` —— 已通过严格校验的参数模型；`dependency_indices: list[int]` —— 该调用依赖的下标列表，用于检查依赖是否失败；`results: list[ToolResult | None]` —— 全局结果槽位，只读地用来查依赖结果；`context: ExecutionContext` —— 执行上下文，按工具 `execute` 签名需要时注入；`loop_state: _LoopExecutionState` —— 当前事件循环的全局信号量所在状态。
- **返回**：`tuple[int, ToolResult]`，第一个元素是传入的 `index`，第二个是成功结果（`ok=True`，`data` 为 `spec.output_model` 校验后 `model_dump()` 的字典）或失败结果（`ok=False`，`error` 为 `DEPENDENCY_FAILED`、`TIMEOUT`、`INVALID_OUTPUT`、`EXECUTION_ERROR` 之一）。
- **内部流程**：
  1. 前置断言式检查：`tool is None or arguments is None` 时抛 `RuntimeError("scheduler received an unvalidated tool call")`。
  2. 依赖失败传播：若 `dependency_indices` 中存在某个下标的 `results[...]` 非空且 `ok` 为假，立即返回 `DEPENDENCY_FAILED`，不再占用任何信号量。
  3. 取 `spec = tool.spec` 与 `tool_semaphore = self._tool_semaphore(loop_state, tool)`；初始化 `global_acquired`、`tool_acquired` 为 `False`，`task = None`，`defer_release = False`。
  4. `try` 块内先 `await tool_semaphore.acquire()` 并置 `tool_acquired = True`，再 `await loop_state.global_semaphore.acquire()` 并置 `global_acquired = True`。注释说明：等待工具槽位期间不占用全局槽位，这样不同工具的排队不会互相饿死。
  5. `loop = asyncio.get_running_loop()`；`deadline = loop.time() + spec.timeout_seconds`（单调时钟，避免系统时间跳变）。
  6. 参数注入：`execute_arguments = [validated_arguments]`、`execute_keywords = {}`；用 `inspect.signature(tool.execute).parameters.get("context")` 探测 `execute` 是否声明了 `context` 形参 —— 若为 `KEYWORD_ONLY` 则放入 `execute_keywords["context"]`；若为 `POSITIONAL_ONLY` 或 `POSITIONAL_OR_KEYWORD` 则追加到位置参数；没有该形参则不注入（说明该工具不需要上下文）。
  7. 分派执行：`inspect.iscoroutinefunction(tool.execute)` 为真时，`task = asyncio.create_task(tool.execute(...))`，然后 `await self._await_until(task, deadline, cancel_on_timeout=True)`（协程可以安全取消）；否则用 `asyncio.create_task(asyncio.to_thread(tool.execute, ...))` 把同步函数放到线程池，再 `await self._await_until(task, deadline, cancel_on_timeout=False)`（线程不可取消，所以超时不取消）。
  8. 二次等待：若返回的 `value` 仍是可等待对象（`inspect.isawaitable(value)`），用 `asyncio.ensure_future(value)` 再包一层并再次 `await self._await_until(task, deadline, cancel_on_timeout=True)` —— 这覆盖了「同步函数返回了一个协程/可等待对象」的情况。
  9. 输出校验：`spec.output_model.model_validate(value, strict=True).model_dump()` 得到 `data`，返回 `ToolResult(call_id=..., tool_name=..., ok=True, data=data)`。
  10. `except _ExecutionTimedOut` → 返回 `TIMEOUT`，其 `retryable` 取 `spec.idempotent`（幂等工具才建议重试）。
  11. `except ValidationError` → 返回 `INVALID_OUTPUT`，消息经 `self._limited_message(exc)` 压缩。
  12. `except Exception` → `LOGGER.error("Tool execution failed for %s (%s)", spec.name, type(exc).__name__)`，返回 `EXECUTION_ERROR`，`retryable` 同样取 `spec.idempotent`；错误消息故意做成笼统的 "tool execution failed"，不把内部细节回传给模型。
  13. `finally`：若 `task` 未完成、且两个信号量都已获取，则置 `defer_release = True` 并通过 `task.add_done_callback(self._release_after_background_work(loop_state, tool_semaphore))` 注册延迟释放；若 `task` 已完成且未被取消，则调用 `task.exception()` 主动取出异常，避免 asyncio 报「任务异常未被检索」；最后若 `defer_release` 仍为假，则按需 `loop_state.global_semaphore.release()` 与 `tool_semaphore.release()` 归还槽位。
- **异常/边界**：`tool`/`arguments` 为 `None` → `RuntimeError`（属内部一致性错误，不被自身捕获，会冒泡到 `asyncio.gather`）。工具超时 → `TIMEOUT`；输出不符合 `output_model` → `INVALID_OUTPUT`；其它任何异常 → `EXECUTION_ERROR`；依赖失败 → `DEPENDENCY_FAILED`。信号量保证在 `finally` 中归还，或在后台线程结束时由回调归还，不会泄漏。
- **同文件关系**：调用 `_error_result`、`_tool_semaphore`、`_await_until`、`_release_after_background_work`、`_limited_message`；读取 `_LoopExecutionState` 的 `global_semaphore`；被 `execute_batch` 通过 `asyncio.gather` 调用。

### `async def ToolExecutionManager._await_until(self, task: asyncio.Task[Any], deadline: float, *, cancel_on_timeout: bool) -> Any` （第 420 行）
- **作用**：这是一个「带截止时间的等待」小工具，把 `asyncio.wait` 的用法收敛到一处。它根据单调时钟算出还剩多少时间，最多等这么久；如果在期限内任务完成就返回其结果（若任务本身失败，`task.result()` 会把异常重新抛出，交由 `_run_one` 的异常分支统一翻译），否则按调用方要求决定是否取消任务，并抛出 `_ExecutionTimedOut` 作为超时信号。`cancel_on_timeout` 这个开关存在的原因是：协程任务取消是干净可控的，而线程任务取消只是「取消等待」并不会真的停下线程，所以对线程型任务传 `False`，把线程是否仍在运行交给 `_run_one` 的延迟释放逻辑处理。
- **参数**：`task: asyncio.Task[Any]` —— 被等待的任务（可能是协程任务，也可能是 `asyncio.to_thread` 包装的线程任务）；`deadline: float` —— 绝对截止时间，取自 `loop.time()` 的单调时钟；`cancel_on_timeout: bool` —— 仅限关键字传参，超时时是否调用 `task.cancel()`。
- **返回**：`Any`，任务成功完成时的返回值（可能是任意类型，之后由 `output_model` 校验）。
- **内部流程**：`timeout = max(0.0, deadline - asyncio.get_running_loop().time())`（用 `max` 兜住已经过期的情况，保证不传负数给 `asyncio.wait`）；`done, _ = await asyncio.wait({task}, timeout=timeout)`；若 `done` 为空集，则在 `cancel_on_timeout` 为真时 `task.cancel()`，随后 `raise _ExecutionTimedOut`；否则 `return task.result()`。
- **异常/边界**：超时抛 `_ExecutionTimedOut`（不传消息）；任务自身抛出异常时由 `task.result()` 原样重抛；任务被取消时 `task.result()` 会抛 `asyncio.CancelledError`，会被 `_run_one` 的宽泛 `except Exception` 捕获（注：`CancelledError` 在新版 Python 中继承自 `BaseException`，实际是否会进入该分支取决于运行时版本）。`deadline` 已过时 `timeout` 为 `0.0`，`asyncio.wait` 会立即返回，等价于一次不阻塞的检查。
- **同文件关系**：抛出 `_ExecutionTimedOut`；被 `_run_one` 调用（两到三次：协程/线程执行一次，可等待返回值再一次）。

### `ToolExecutionManager._release_after_background_work(self, loop_state: _LoopExecutionState, tool_semaphore: asyncio.Semaphore)` （第 435 行）
- **作用**：这是一个「延迟释放信号量」的工厂函数。当同步工具在超时后仍在后台线程里跑时，`_run_one` 不能立刻归还信号量，否则会出现「已经超时返回、但线程还在跑」的隐藏并发重叠。这个函数把需要释放的两个信号量闭包捕获起来，返回一个可以挂到任务上的完成回调；等任务真正结束时，回调再归还槽位。它还顺手调用 `task.exception()` 把迟到的异常取出来，避免 asyncio 事后打印「Task exception was never retrieved」的告警。
- **参数**：`loop_state: _LoopExecutionState` —— 提供 `global_semaphore` 供回调释放；`tool_semaphore: asyncio.Semaphore` —— 该工具自己的信号量，供回调释放。
- **返回**：一个可调用对象（内部函数 `release`），签名接受一个 `asyncio.Task[Any]`，返回 `None`；该对象被用作 `task.add_done_callback(...)` 的回调。
- **内部流程**：定义内部函数 `release(task)`：若 `task.cancelled()` 为假则调用 `task.exception()` 以消费可能的异常；然后依次 `loop_state.global_semaphore.release()` 与 `tool_semaphore.release()`；函数本身只是 `return release`，不执行任何释放动作。
- **异常/边界**：无特殊处理；`release` 内部对已取消任务跳过 `task.exception()`（对已取消任务调用它会抛 `CancelledError`）。
- **同文件关系**：使用 `_LoopExecutionState`；被 `_run_one` 在需要延迟释放时调用；它返回的内嵌函数 `release` 在任务完成时由事件循环调用。

### `release(task: asyncio.Task[Any]) -> None` （第 440 行，嵌套在 `_release_after_background_work` 内部）
- **作用**：这是真正执行「事后归还并发槽位」的嵌套回调。它只在 `_run_one` 判定需要延迟释放（即超时后线程仍在运行）时才被注册，作用是在后台工作彻底结束后把全局信号量和工具信号量各归还一个，使后续排队的调用能够继续。同时它负责「消费」迟到任务的异常，防止未检索的任务异常在事件循环关闭时被打印成噪音日志。
- **参数**：`task: asyncio.Task[Any]` —— 已结束的后台任务，由 `add_done_callback` 自动传入。
- **返回**：`None`。
- **内部流程**：`if not task.cancelled(): task.exception()` 取出并丢弃异常（若任务以异常结束，这一步是必需的）；随后 `loop_state.global_semaphore.release()`；再 `tool_semaphore.release()`。两个信号量都通过闭包从外层函数捕获。
- **异常/边界**：对已取消的任务跳过 `task.exception()`，避免再次抛 `CancelledError`；无其它特殊处理。若该回调被重复触发（正常不会），信号量会被多释放，属于使用方责任。
- **同文件关系**：由 `_release_after_background_work` 定义并返回；被 `_run_one` 通过 `task.add_done_callback` 注册。

### `@staticmethod ToolExecutionManager._limited_message(error: Exception, limit: int = 2000) -> str` （第 449 行）
- **作用**：把异常转换成可以安全回传给模型或调用方的短消息。它做两件重要的事：一是对 `ValidationError` 做特殊处理，只保留字段路径与错误描述，刻意丢弃 Pydantic 默认字符串里的 `input_value`（因为那里面可能包含不受信任的用户参数甚至密钥），也丢弃 URL 与 context 信息；二是无论什么异常都做长度截断，避免超长错误文本把上下文撑爆。它被参数校验失败与输出校验失败两处复用，保证错误消息风格一致。
- **参数**：`error: Exception` —— 待转换的异常对象，通常是 `pydantic.ValidationError`，也可能是其它异常；`limit: int = 2000` —— 结果字符串的最大长度，超出即截断，默认 2000 个字符。
- **返回**：`str`，长度不超过 `limit` 的错误描述；当格式化结果为空时回落到异常类名，保证永不返回空串。
- **内部流程**：先判断 `isinstance(error, ValidationError)`：是则调用 `error.errors(include_url=False, include_context=False)` 逐项取出 `loc` 与 `msg`，用 `".".join(str(part) for part in item.get("loc", ()))` 拼出字段路径，`msg` 缺省为 `"validation failed"`，有路径则格式化为 `"路径: 消息"`，无路径则只留消息，最后用 `"; "` 连接成 `message`；不是 `ValidationError` 则 `message = str(error)`。随后 `message = message or type(error).__name__` 处理空字符串的情况，最后 `return message[:limit]` 按字符截断。
- **异常/边界**：无特殊处理；`error` 不是异常对象时 `str()` 与 `type().__name__` 仍可工作，但调用方约定传入异常。`limit` 传 `0` 会返回空字符串（此时不再走兜底，因为兜底发生在截断之前）。
- **同文件关系**：被 `execute_batch`（参数校验失败分支）与 `_run_one`（输出校验失败分支）调用；自身不调用本文件其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_ExecutionTimedOut` | 运行时强制超时的内部标记异常，用于把超时从普通异常中区分出来。 |
| `_LoopExecutionState` | 按事件循环隔离的异步状态容器，持有全局信号量与按工具缓存的信号量字典。 |
| `ToolExecutionManager` | 工具调用批次执行引擎，负责校验、依赖排序、并发限流、执行与结果聚合。 |
| `ToolExecutionManager.__init__` | 校验并保存注册表与全局并发上限，初始化按循环索引的状态表及其线程锁。 |
| `ToolExecutionManager.execute_batch` | 总入口：前置校验全部调用、解析依赖图、按波次并发调度并返回与输入等长同序的结果。 |
| `ToolExecutionManager._abort_remaining` | 把尚未开始的调用统一标记为 `ABORTED` 并清空待执行集合。 |
| `ToolExecutionManager._error_result` | 把一条调用与一个 `ToolError` 组装成 `ok=False` 的 `ToolResult`。 |
| `ToolExecutionManager._loop_state` | 取得当前事件循环的状态对象，惰性创建并清理已关闭循环的残留状态。 |
| `ToolExecutionManager._tool_semaphore` | 按工具标识与容量惰性创建并缓存该工具的并发信号量。 |
| `ToolExecutionManager._run_one` | 执行单个工具调用：获取两级信号量、按截止时间等待、校验输出、统一翻译异常并延迟释放槽位。 |
| `ToolExecutionManager._await_until` | 在给定截止时间前等待任务完成，超时则按需取消任务并抛 `_ExecutionTimedOut`。 |
| `ToolExecutionManager._release_after_background_work` | 返回一个完成回调工厂，用于后台线程真正结束后再归还信号量并消费迟到异常。 |
| `release`（嵌套函数） | 任务结束时释放全局与工具信号量，并取出被忽略的任务异常。 |
| `ToolExecutionManager._limited_message` | 把异常压成安全且限长的错误消息，对 `ValidationError` 剔除输入值等敏感信息。 |
