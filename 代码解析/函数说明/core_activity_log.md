# core/activity_log.py

## 一、这个文件是干什么的

这个文件是整个 Agent 运行时的「控制台活动日志输出层」，它把 ReAct 循环执行过程中的关键节点以人类可读的中文短句打印到标准输出，让交互式使用者在终端里能实时看到 Agent 正在做什么。

它做的事情非常收敛：对外暴露一组 `log_xxx` 形式的模块级函数（工具注册、工具发现汇总、ReAct 轮次开始、模型思考耗时、模型首字返回、模型思考内容、工具调用开始、工具调用耗时、最终回答、协议解析问题），每个函数只负责把一条格式化好的消息交给同一个 logger 输出。

它内部自带一个私有日志处理器 `_ConsoleLogHandler`，重写了 `logging.Handler.emit`，用 `print(..., flush=True)` 而不是标准 `StreamHandler` 来输出，这样即使在输出被重定向、缓冲策略异常的交互环境里，进度信息也能立刻刷出来，不会被憋在缓冲区里。

日志器名字固定为 `all_agent.activity`，级别固定 INFO，并且显式关闭了 `propagate`，避免同一条活动消息又被 root logger 的其它 handler 重复打印一遍。

文件还定义了两个私有清洗/规范化辅助函数：`_tool_names` 把任意可迭代的工具名去重、排序、过滤非字符串与空串；`_clean_text` 把任意文本压成单行空格分隔并截断到 2000 字符，防止把模型的长篇思考或多行堆栈直接灌进终端。

设计上有意「该静音的静音」：工具注册、工具发现汇总、轮次开始、最终回答这四个函数是空实现（只有 docstring、没有函数体语句），因为它们是内部流程细节，已经通过别的渠道（返回值或别的界面）呈现，不需要再占用面向用户的活动流。

它在项目里被 ReAct 执行器和工具注册流程当作「纯副作用」的展示工具调用：调用方只关心日志是否打出来，不关心返回值（全部返回 `None`），因此它可以被安全地插在任意执行路径上而不改变控制流。

## 二、函数与类逐条详解

### 类 `_ConsoleLogHandler(logging.Handler)` （第 9 行）
- **作用**：这是一个自定义的 `logging` 处理器子类，存在的唯一目的是把日志记录直接写进标准输出，而不是走 `logging.StreamHandler` 默认的 `sys.stderr` 通道。它用 `print` 代替流写入，是为了保证在任何交互式调用场景（比如被其它库接管了 stdout、或者 stdout 被包装过）下，进度信息都能被用户看见。文件名以 `_` 开头表明它是本模块私有实现细节，不打算被外部导入使用。它只被模块底部的初始化代码实例化一次，之后长期挂载在 `LOGGER` 上。
- **参数**：类本身没有自定义 `__init__`，继承 `logging.Handler` 的构造函数，因此实例化时不传参数（`_ConsoleLogHandler()`），内部使用父类的锁、级别、过滤器等基础设施。类的 docstring 中提到的行为由子类方法 `emit` 承担。
- **返回**：这是一个类，构造时返回一个 `logging.Handler` 实例，该实例可被 `LOGGER.addHandler` 接受。
- **内部流程**：类体里只定义了一个方法 `emit`；实例化时继承 `logging.Handler.__init__`（创建 `RLock`、设置 `_name`、`level`、`filters`、`lock` 等）；随后模块级代码对它调用 `setFormatter` 装配 `"[%(asctime)s] %(message)s"` 格式器，再 `addHandler` 挂到 `LOGGER` 上。
- **异常/边界**：类定义本身不抛异常；运行期异常全部由 `emit` 内部捕获处理（见下一条）。没有任何自定义校验或断言。
- **同文件关系**：它被本文件模块级初始化代码（第 20-25 行）实例化并注册到 `LOGGER`；它的 `emit` 方法被 `logging` 框架在 `LOGGER.info(...)` / `LOGGER.warning(...)` 调用时回调，从而服务于本文件所有 `log_*` 函数。

### `_ConsoleLogHandler.emit(self, record: logging.LogRecord) -> None` （第 12 行）
- **作用**：这是 `logging.Handler` 的抽象方法实现，日志框架每产生一条通过级别检查的记录就会调用它一次。它把 `LogRecord` 交给 `self.format(record)` 渲染成最终文本，然后用 `print(..., flush=True)` 写到标准输出并强制刷新缓冲区。使用 `flush=True` 是关键：交互式 Agent 的进度提示必须即时出现，否则用户会在长时间等待模型响应时误以为程序卡死。整个打印动作被 `try/except` 包住，保证日志输出永远不会反过来把业务流程炸掉。
- **参数**：
  - `self`：处理器实例本身，提供 `format`（来自 `logging.Handler`）与 `handleError`（来自 `logging.Handler`）。
  - `record`（`logging.LogRecord`）：由 `logging` 框架构造的日志记录对象，携带 `msg`、`args`、`levelno`、`created` 等字段；本方法只把它整体交给 `format`，不直接读取任何字段。
- **返回**：始终返回 `None`；副作用是向标准输出写入一行文本（通常带 `[HH:MM:SS]` 时间前缀）。
- **内部流程**：第一步进入 `try` 块；第二步调用 `self.format(record)` 得到字符串（该调用会应用模块初始化时设置的 `logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")`）；第三步调用内建 `print(该字符串, flush=True)` 输出；如果上述任意一步抛出异常（例如 `format` 因格式化参数不匹配失败、stdout 被关闭导致写入报错），则进入 `except Exception` 分支，调用 `self.handleError(record)`，由 `logging` 框架按标准方式处理（默认是把错误信息打到 stderr，并在存在 `raiseExceptions` 配置时输出 traceback）。
- **异常/边界**：显式捕获所有 `Exception`（代码里带 `# noqa: BLE001` 抑制 lint 警告），因此本方法对外不抛异常；`BaseException`（如 `KeyboardInterrupt`、`SystemExit`）不被捕获，仍会向上传播，这是符合预期的。对 `record` 为 `None` 之类的非法输入没有额外判断，会走 `except` 分支交给 `handleError`。
- **同文件关系**：它被 `logging` 框架回调，间接被本文件全部 `log_*` 函数（`log_model_completed`、`log_model_first_chunk`、`log_react_thought`、`log_tool_call_started`、`log_tool_call_completed`、`log_react_parse_issue`）触发；它自身不调用本文件其它函数。

### 模块级初始化：`LOGGER` 及其处理器装配（第 19-27 行）
- **作用**：这不是函数，但它是本文件所有日志输出的前提，因此单独说明。它取出（或创建）名为 `all_agent.activity` 的 logger，检查其 `handlers` 里是否已经存在 `_ConsoleLogHandler` 实例，只有在不存在时才新建并挂载一个带时间格式的处理器。这个「先判断再添加」的写法是为了幂等：如果模块被重复导入、或热重载导致代码二次执行，也不会出现同一条消息被打印两遍的情况。随后把级别固定为 `INFO`（低于 INFO 的 DEBUG 消息被丢弃），并把 `propagate` 设为 `False`（阻止消息冒泡到 root logger，避免上层 handler 重复输出）。
- **参数**：不涉及函数参数。
- **返回**：不涉及返回值，副作用是修改全局 logger 的处理器列表、级别与传播标志。
- **内部流程**：`logging.getLogger("all_agent.activity")` 取 logger（`logging` 内部有名字到实例的注册表，同名多次调用返回同一对象）；`any(isinstance(handler, _ConsoleLogHandler) for handler in LOGGER.handlers)` 做存在性判定；未命中则构造 `_ConsoleLogHandler()`、`setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))`、`addHandler(...)`；最后 `setLevel(logging.INFO)` 与 `propagate = False`。
- **异常/边界**：无特殊处理；`LOGGER.handlers` 在标准 `logging` 实现下始终是列表，不会为空引用。
- **同文件关系**：它实例化并使用类 `_ConsoleLogHandler`；它产出的 `LOGGER` 被本文件所有非空实现的 `log_*` 函数引用，并被 `__all__` 导出。

### `log_tool_registration(tool_name: str, generation: int | None, registered_names: Iterable[str]) -> None` （第 30 行）
- **作用**：这是工具注册流程的日志钩子，语义上表示「某个工具被注册进当前这一代工具集合」。它的实现被刻意留空——函数体只有 docstring，没有任何语句——因为工具注册属于框架内部装配细节，对使用者没有可操作的信息价值，打出来只会淹没真正有用的活动流。保留这个函数（而不是删掉让调用方不写）是为了让调用点保持统一的日志调用风格，将来若要恢复输出只需在此处补一行 `LOGGER.info`。调用方通常在工具注册/热更新完成后调用它，调用后除了「函数被调用过」之外不产生任何可观察效果。
- **参数**：
  - `tool_name`（`str`）：被注册的工具名称，理论上是工具的唯一标识；当前实现完全不读取它。
  - `generation`（`int | None`）：工具集合的「代号/代际」编号，可能为 `None`（表示未编号或首次注册）；当前实现完全不读取它。
  - `registered_names`（`Iterable[str]`）：注册后当前生效的全部工具名集合，可以是任意可迭代对象（列表、集合、生成器等）；当前实现完全不读取它，因此也不会消耗生成器。
- **返回**：始终返回 `None`。
- **内部流程**：函数体为空，调用后立即返回，不访问 `LOGGER`，不做任何清洗或格式化。
- **异常/边界**：因为不执行任何代码，所以对任何输入（包括 `None`、空迭代器、类型不符的对象）都不会抛异常；无特殊处理。
- **同文件关系**：不调用本文件任何其它函数；被工具注册相关的外部代码调用，与本文件的 `_tool_names`、`_clean_text` 无关联。

### `log_discovery_summary(package: str, registered_names: Iterable[str]) -> None` （第 38 行）
- **作用**：这是工具发现（自动扫描某个包并加载其中工具）完成后的汇总日志钩子。它的实现同样被刻意留空，原因是发现过程的明细（扫到了哪些包、注册了哪些名字）对最终用户属于噪音，只对框架开发者有意义，而框架开发者可以通过别的方式排查。函数名与参数签名保留了完整的语义信息，使得调用点读起来仍然自解释；若将来需要在调试模式下输出发现明细，只需在此实现里加条件判断与 `LOGGER.info`。
- **参数**：
  - `package`（`str`）：被扫描/发现的 Python 包名（例如某个工具包路径）；当前实现完全不读取它。
  - `registered_names`（`Iterable[str]`）：本次发现过程中成功注册的工具名集合；当前实现完全不读取它，因此不会遍历或消耗它。
- **返回**：始终返回 `None`。
- **内部流程**：函数体为空，直接返回，无任何分支、循环或库调用。
- **异常/边界**：无任何代码执行，故不会因传入非法值而抛异常；无特殊处理。
- **同文件关系**：不调用本文件任何其它函数；被外部工具发现流程调用，未被本文件内其它函数调用。

### `log_react_round_started(round_number: int, max_rounds: int | None) -> None` （第 42 行）
- **作用**：这是 ReAct 循环每一轮开始时的日志钩子，语义上表示「第 N 轮开始，总轮数上限为 M」。它同样被留空实现，因为轮次推进属于内部控制流：用户关心的是模型在想什么、调用了什么工具、花了多久，而不是「第几轮开始」这种框架内部节奏。留空可以让活动流更干净，避免在快速多轮推理时刷出大量无信息量的行。它在每一轮 ReAct 迭代的入口被调用，调用后不产生输出。
- **参数**：
  - `round_number`（`int`）：当前轮次序号，通常是 1 开始的整数；当前实现完全不读取它。
  - `max_rounds`（`int | None`）：允许的最大轮数上限，可能为 `None`（表示不限制或未配置）；当前实现完全不读取它。
- **返回**：始终返回 `None`。
- **内部流程**：函数体为空，无任何语句执行。
- **异常/边界**：无代码执行，不会抛异常；无特殊处理。
- **同文件关系**：不调用本文件任何其它函数；被 ReAct 执行器在每轮入口调用，与本文件其它 `log_react_*` 函数（`log_react_thought`、`log_react_parse_issue`、`log_react_final_answer`）构成同一组钩子但彼此无调用关系。

### `log_model_completed(round_number: int, elapsed_seconds: float) -> None` （第 46 行）
- **作用**：这是本文件里真正会输出的函数之一，用来报告「某一轮模型调用总共花了多少时间」。它把耗时以「模型思考耗时：第N轮 X.XXX秒」的紧凑中文格式写进活动流，让用户对模型响应速度有直观感受，也方便排查「是不是模型侧变慢了」。格式固定为三位小数（`%.3f`），保证多行之间宽度一致、便于肉眼纵向比对。它在模型（含流式聚合）完整结束后被调用，是性能观测的主要手段之一。之所以只报耗时而不报内容，是因为内容已经由 `log_react_thought` 单独负责，职责分离。
- **参数**：
  - `round_number`（`int`）：当前 ReAct 轮次序号，作为 `%d` 填充进消息，用于把耗时归属到具体轮次。
  - `elapsed_seconds`（`float`）：该轮从发起到完成的耗时秒数；调用方可能传入浮点数甚至极小值或负值（如计时器精度问题、时钟回拨）。
- **返回**：始终返回 `None`；副作用是向 `LOGGER` 提交一条 INFO 级日志。
- **内部流程**：第一步调用 `max(0.0, elapsed_seconds)` 对耗时做下限钳制，把任何负数统一抬到 `0.0`，避免打印出「-0.003秒」这种会让用户困惑的值；第二步以 `"模型思考耗时：第%d轮 %.3f秒"` 为模板，把 `round_number` 与钳制后的耗时作为位置参数交给 `LOGGER.info`；真正的字符串格式化由 `logging` 在渲染阶段惰性完成，最终经 `_ConsoleLogHandler.emit` 打印。
- **异常/边界**：如果 `elapsed_seconds` 不是数值类型（例如字符串），`max(0.0, elapsed_seconds)` 会抛 `TypeError`，本函数不做捕获；`round_number` 传入非整数时会在日志格式化阶段由 `%d` 触发错误，该错误发生在 handler 的 `format` 内部并被 `emit` 的 `except` 吞掉。对 `None` 没有兜底，会抛 `TypeError`；对负值、`0.0`、`NaN` 的处理：负值被钳为 `0.0`，`0.0` 原样输出，`NaN` 会原样打印为 `nan`（`max` 与 `NaN` 比较返回 `0.0` 视实现而定，标准行为下 `max(0.0, nan)` 返回 `0.0`）。
- **同文件关系**：调用本文件外部的 `LOGGER`（模块级对象）；不调用本文件的辅助函数（未使用 `_clean_text`，因为模板本身是受控文本）；被外部 ReAct 执行器在模型轮结束后调用。

### `log_model_first_chunk(round_number: int, elapsed_seconds: float) -> None` （第 52 行）
- **作用**：这是流式模型调用的「首字延迟」（time to first token）上报函数，输出「模型首字返回：第N轮 X.XXX秒（流式连接已建立）」。它解决的是流式场景下用户体感问题：整体耗时可能很长，但首字延迟短就意味着「已经开始吐字了」，用户不会觉得卡死。括号里的「（流式连接已建立）」是给使用者的提示语，说明这条日志代表连接与首个分片已经就绪。它通常在流式响应收到第一个 chunk 时调用，与 `log_model_completed` 形成「首字」与「全程」两个观测点。需要注意的是，这个函数虽然在本文件里被定义并实现，却没有被列入文件末尾的 `__all__` 导出列表。
- **参数**：
  - `round_number`（`int`）：当前轮次序号，以 `%d` 填入消息。
  - `elapsed_seconds`（`float`）：从发起请求到收到第一个分片的秒数；同样可能是负值或异常小值。
- **返回**：始终返回 `None`；副作用是提交一条 INFO 级日志。
- **内部流程**：先用 `max(0.0, elapsed_seconds)` 把耗时下限钳到 `0.0`；再以多行拼接的模板 `"模型首字返回：第%d轮 %.3f秒（流式连接已建立）"` 调用 `LOGGER.info`，把轮次与耗时按位置传入；格式化由 `logging` 惰性完成，最终由 `_ConsoleLogHandler.emit` 打印。
- **异常/边界**：与 `log_model_completed` 相同——非数值型 `elapsed_seconds` 会在 `max` 处抛 `TypeError`；`None` 无兜底；负值被钳制为 `0.0`；没有超时判断，函数本身不知道也不关心请求是否超时。
- **同文件关系**：使用模块级 `LOGGER`；不调用本文件的辅助函数；与 `log_model_completed` 是并列的耗时上报函数，彼此无调用；未被 `__all__` 收录。

### `log_react_thought(round_number: int, thought: str) -> None` （第 62 行）
- **作用**：这是 ReAct 循环中「展示模型当前思考内容」的函数，是本文件最核心的用户可见输出之一。它把模型这一轮产出的思考文本（reasoning / thought）经过清洗后打出来，让用户能看到 Agent 的推理过程，而不是只看到最终答案。它只打印思考，不打印工具结果或答案，从而保证活动流简短可读。文档字符串特意强调「Display only the model's current thought」，说明这是有意的职责收窄。它在每一轮拿到模型输出的思考部分后调用。
- **参数**：
  - `round_number`（`int`）：当前轮次序号。注意实现里并没有把它用进日志消息——`LOGGER.info("模型思考：%s", _clean_text(thought))` 只传了一个格式化参数——因此该参数目前是保留形参，接收但未使用。
  - `thought`（`str`）：模型产出的思考文本，可能包含换行、多余空白、超长内容甚至非字符串对象（会被 `_clean_text` 内部 `str()` 强制转换）。
- **返回**：始终返回 `None`；副作用是提交一条 INFO 级日志。
- **内部流程**：调用本文件的私有辅助函数 `_clean_text(thought)`，把文本压成单行、折叠连续空白、并在超过 2000 字符时截断为「前 1997 字符 + ...」；然后把清洗结果作为 `%s` 参数交给 `LOGGER.info("模型思考：%s", ...)`；`logging` 惰性格式化后由 `_ConsoleLogHandler.emit` 打印。
- **异常/边界**：`thought` 为 `None` 时不会抛异常，`_clean_text` 内部 `str(None)` 得到 `"None"` 并正常输出；超长文本被截断到 2000 字符以内，不会把终端刷爆；包含换行的文本被折叠成空格，保证一条日志占一行；`round_number` 未被使用，因此传任何值（包括 `None`）都不影响输出。
- **同文件关系**：调用 `_clean_text`；通过 `LOGGER` 间接依赖 `_ConsoleLogHandler.emit`；被外部 ReAct 执行器在每轮思考产出后调用。

### `log_tool_call_started(round_number: int, tool_names: Iterable[str]) -> None` （第 68 行）
- **作用**：这是工具调用开始时的日志函数，用来告诉用户「这一轮准备调用哪些工具」。它只输出工具名字列表，不输出参数，因为参数可能很大或含敏感内容，而名字足以让用户理解 Agent 的意图。文档字符串明确写着「Display only the tool names about to be executed」。当模型决定调用工具、执行器即将派发之前调用它。若工具名集合为空或全部非法，它会退化为打印「无」，避免出现「调用工具：」后面空荡荡的尴尬输出。
- **参数**：
  - `round_number`（`int`）：当前轮次序号；实现里同样没有被使用，只是保留形参（消息模板 `"调用工具：%s"` 只消费一个参数）。
  - `tool_names`（`Iterable[str]`）：本轮将要调用的工具名集合，可以是列表、集合、元组或任何可迭代对象；元素可能混入非字符串、空串、重复项。
- **返回**：始终返回 `None`；副作用是提交一条 INFO 级日志。
- **内部流程**：先调用 `_tool_names(tool_names)` 得到「过滤掉非字符串与空串、去重、升序排序」的 `list[str]`；再用 `", ".join(names) or "无"` 拼成逗号加空格的字符串，若结果为空串（即没有合法工具名）则由 `or` 兜底为中文「无」；最后把该字符串作为 `%s` 交给 `LOGGER.info("调用工具：%s", ...)`，经 `_ConsoleLogHandler.emit` 打印。
- **异常/边界**：`tool_names` 为 `None` 时，`_tool_names` 里的集合推导会抛 `TypeError`（`None` 不可迭代），本函数不做捕获；传入一次性生成器会被 `_tool_names` 完整消费一次，之后原生成器耗尽；空集合、空列表、全非法元素都安全地输出「无」；`round_number` 未使用，不影响行为。
- **同文件关系**：调用 `_tool_names`；通过 `LOGGER` 间接依赖 `_ConsoleLogHandler.emit`；与 `log_tool_call_completed` 是同一对「开始/结束」上报函数，两者共享 `_tool_names` 但彼此不互相调用。

### `log_tool_call_completed(round_number: int, tool_names: Iterable[str], elapsed_seconds: float) -> None` （第 75 行）
- **作用**：这是工具调用结束时的日志函数，报告「哪些工具、花了多久」。文档字符串强调「Display tool latency without dumping potentially large tool results」，即只报耗时、绝不把工具返回的大块内容打出来——这是防止终端被搜索结果、文件内容等海量文本淹没的关键设计。它让用户能判断是模型慢还是工具慢。通常在工具执行完毕（无论成功与否）后调用。输出形如「调用工具耗时：第2轮 read_file, glob 0.412秒」。
- **参数**：
  - `round_number`（`int`）：当前轮次序号，以 `%d` 填入消息，用于把耗时归属到具体轮次（与 `log_tool_call_started` 不同，这里确实用到了它）。
  - `tool_names`（`Iterable[str]`）：本轮执行的工具名集合，同样可能含重复、非字符串、空串元素。
  - `elapsed_seconds`（`float`）：工具执行耗时秒数，可能是负值或异常值。
- **返回**：始终返回 `None`；副作用是提交一条 INFO 级日志。
- **内部流程**：第一步 `names = _tool_names(tool_names)` 规范化工具名列表；第二步 `max(0.0, elapsed_seconds)` 钳制耗时下限；第三步把四个位置参数（轮次、`", ".join(names) or "无"`、钳制后的耗时）按模板 `"调用工具耗时：第%d轮 %s %.3f秒"` 交给 `LOGGER.info`；由 `logging` 惰性格式化、`_ConsoleLogHandler.emit` 输出。
- **异常/边界**：`tool_names` 为 `None` 会在 `_tool_names` 中抛 `TypeError`（未捕获）；空工具名集合输出「无」；`elapsed_seconds` 为负值被钳为 `0.0`，为 `None` 或非数值会在 `max` 处抛 `TypeError`（未捕获）；工具执行失败本身不影响本函数——它只负责记录，不判断成功与否。
- **同文件关系**：调用 `_tool_names`；通过 `LOGGER` 间接依赖 `_ConsoleLogHandler.emit`；与 `log_tool_call_started` 配对使用，二者无相互调用。

### `log_react_final_answer(round_number: int, answer: str) -> None` （第 91 行）
- **作用**：这是 ReAct 循环产出最终回答时的日志钩子。它的实现被刻意留空，原因是最终答案本身已经作为返回值交给调用方（也就是真正展示给用户的那条通道），如果日志层再打一遍，用户就会在终端看到同一段回答出现两次，既冗余又可能因为答案很长而刷屏。文档字符串「The final answer is already returned to the caller; keep it silent」直接点明了这个取舍。它在 ReAct 判定收敛、拿到最终答案时被调用，调用后不产生任何输出。保留空函数体是为了让调用点语义完整、未来需要时可无痛补实现。
- **参数**：
  - `round_number`（`int`）：产出最终答案的轮次序号；当前实现完全不读取它。
  - `answer`（`str`）：最终答案文本；当前实现完全不读取它，因此不会触发 `_clean_text` 的清洗或截断。
- **返回**：始终返回 `None`。
- **内部流程**：函数体为空，无任何语句、分支或库调用。
- **异常/边界**：不执行代码，任何输入（包括 `None`、超长字符串）都不会引发异常；无特殊处理。
- **同文件关系**：不调用本文件任何函数（尤其没有调用 `_clean_text`）；被外部 ReAct 执行器在收敛时调用；与 `log_react_parse_issue` 同属 ReAct 事件钩子但实现策略相反（一个静音、一个告警）。

### `log_react_parse_issue(round_number: int, issue: str) -> None` （第 95 行）
- **作用**：这是本文件里唯一使用 `warning` 级别的函数，用来暴露「模型输出的工具调用格式不符合协议」这类解析问题。它存在的意义是诊断：当 Agent 卡住、工具迟迟不被调用时，用户或开发者能立刻从终端看到是模型把工具调用写坏了（例如 JSON 缺括号、标签不闭合），而不是只能看到「什么都不发生」。文档字符串「Show protocol errors so a stuck tool call is diagnosable」正是这个意图。它在解析模型输出失败、但流程仍要继续（重试或提示模型纠正）时被调用。因为它用 `LOGGER.warning`，输出同样经 `_ConsoleLogHandler` 打印，格式与 INFO 消息一致（时间前缀 + 消息），级别差异仅体现在记录对象的 `levelno` 上。
- **参数**：
  - `round_number`（`int`）：出问题的轮次序号，以 `%d` 填入消息，便于定位是哪一轮的解析失败。
  - `issue`（`str`）：问题描述文本，可能是解析器抛出的异常信息、期望格式说明或原始片段；会被清洗与截断，因此超长的原始输出不会刷爆终端。
- **返回**：始终返回 `None`；副作用是提交一条 WARNING 级日志。
- **内部流程**：调用 `_clean_text(issue)` 做单行化、空白折叠与 2000 字符截断；再把轮次与清洗后的文本作为位置参数交给 `LOGGER.warning("工具调用格式问题：第%d轮 %s", ...)`；`logging` 渲染后由 `_ConsoleLogHandler.emit` 打印。
- **异常/边界**：`issue` 为 `None` 时被 `str(None)` 转成 `"None"` 输出，不抛异常；超长问题文本被截断为 1997 字符加省略号；多行堆栈被压成一行；`round_number` 非整数会在 handler 的 `format` 阶段报错并被 `emit` 的 `except` 吞掉，本函数不做校验。
- **同文件关系**：调用 `_clean_text`；通过 `LOGGER` 间接依赖 `_ConsoleLogHandler.emit`；被外部 ReAct 解析流程在检测到协议错误时调用。

### `_tool_names(names: Iterable[str]) -> list[str]` （第 101 行）
- **作用**：这是一个私有规范化辅助函数，把调用方传来的「工具名集合」整理成干净、确定顺序的列表。它同时完成三件事：过滤掉非字符串元素（防止 `None`、数字等混进来导致后续 `join` 报错）、过滤掉空字符串（防止出现连续逗号或前导分隔符）、去重并升序排序（让同一批工具无论以什么顺序传入，输出都一致，便于日志比对和人工阅读）。它被两个工具调用日志函数共用，是避免重复逻辑的收敛点。之所以返回 `list` 而不是保留集合，是因为调用方需要 `", ".join(...)` 的有序可拼接结果。文档字符串没有（函数无 docstring），但函数名以 `_` 开头标明私有。
- **参数**：
  - `names`（`Iterable[str]`）：任意可迭代的工具名容器（列表、集合、元组、生成器、字典的键视图等）。元素类型在运行时不做类型标注强制，靠内部 `isinstance` 检查兜底；不允许传 `None`（不可迭代）。
- **返回**：返回 `list[str]`——一个升序排序、无重复、全部为非空字符串的新列表；若输入为空或全部元素被过滤，则返回空列表 `[]`（注意：不是 `None`，空值兜底由调用方用 `or "无"` 处理）。
- **内部流程**：用集合推导式 `{name for name in names if isinstance(name, str) and name}` 一步完成遍历、过滤与去重（`and name` 利用字符串真值性同时排除空串）；然后 `sorted(...)` 把集合转成按字典序升序排列的列表并返回。整个过程中不修改输入对象。
- **异常/边界**：`names` 为 `None` 时，`for name in names` 会抛 `TypeError: 'NoneType' object is not iterable`，本函数不捕获也不兜底；输入元素为不可哈希对象时集合推导会抛 `TypeError`（但 `isinstance(name, str)` 前置判断使得非字符串元素在进入集合前就被过滤，因此实践中不会触发）；输入为一次性生成器时会被完整消费。
- **同文件关系**：它不调用本文件其它函数；被 `log_tool_call_started` 与 `log_tool_call_completed` 调用；未被 `__all__` 导出。

### `_clean_text(value: str, limit: int = 2000) -> str` （第 105 行）
- **作用**：这是一个私有文本清洗辅助函数，负责把任意输入转成「单行、空白折叠、长度受限」的安全日志文本。它要解决两个现实问题：一是模型的思考或解析错误信息里常含换行与大量缩进，直接打印会把一条日志撑成几十行、破坏活动流的可读性；二是原始输出可能极长（例如把整个文件内容塞进了 `issue`），必须截断以免刷屏或拖慢终端。它被 `log_react_thought` 与 `log_react_parse_issue` 共用。截断时保留末尾的 `"..."` 作为视觉提示，让读者知道内容被省略过，而不是以为模型就说了这么点。函数无 docstring，靠 `_` 前缀表明私有。
- **参数**：
  - `value`（`str`）：待清洗的文本；虽然标注为 `str`，但实现里先做 `str(value)`，因此实际可接受任意对象（`None`、数字、异常对象等）。
  - `limit`（`int`，默认 `2000`）：允许的最大字符数上限，含省略号在内。调用方目前都用默认值；若传入小于 3 的值，截断分支会走 `cleaned[: limit - 3] + "..."`，可能得到负索引切片（例如 `limit=2` 时 `cleaned[:-1] + "..."`）导致结果短于预期，属于未做校验的边界。
- **返回**：返回 `str`。若清洗后长度不超过 `limit`，原样返回该单行文本；否则返回「前 `limit - 3` 个字符 + `"..."`」，总长度约为 `limit`。输入为空串或全空白时返回空串 `""`。
- **内部流程**：第一步 `str(value)` 强制转字符串；第二步 `.split()` 以任意空白（空格、制表符、换行）切分并丢弃空片段；第三步 `" ".join(...)` 用单个空格重新拼接，得到折叠后的单行文本并存入 `cleaned`；第四步比较 `len(cleaned) <= limit`，成立则直接 `return cleaned`；否则走第五步 `return cleaned[: limit - 3] + "..."`。
- **异常/边界**：`str(value)` 对任何对象几乎都不会失败（除非对象的 `__str__` 自己抛异常，此时异常向上传播、本函数不捕获）；`limit` 为非整数会在切片处抛 `TypeError`；`limit` 小于 3 时切片行为退化（见上）；`value` 为 `None` 返回 `"None"` 而非空串，这是 `str()` 的语义决定的；不处理 Unicode 宽度或 emoji 截断问题。
- **同文件关系**：它不调用本文件其它函数；被 `log_react_thought` 与 `log_react_parse_issue` 调用；未被 `__all__` 导出。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_ConsoleLogHandler`（类） | 自定义 logging 处理器，用 `print(..., flush=True)` 把活动日志实时写到标准输出。 |
| `_ConsoleLogHandler.emit` | 把一条 `LogRecord` 格式化后打印并强制刷新，异常时交给 `handleError`，保证日志永不炸主流程。 |
| 模块级 `LOGGER` 装配 | 创建 `all_agent.activity` logger，幂等地挂载控制台处理器、设定 INFO 级别并关闭向 root 传播。 |
| `log_tool_registration` | 工具注册日志钩子，刻意留空以保持面向用户的活动流干净。 |
| `log_discovery_summary` | 工具发现汇总日志钩子，刻意留空以避免输出发现明细噪音。 |
| `log_react_round_started` | ReAct 轮次开始钩子，刻意留空，因为轮次属于内部控制流。 |
| `log_model_completed` | 打印「模型思考耗时：第N轮 X.XXX秒」，并对负耗时钳制为 0。 |
| `log_model_first_chunk` | 打印「模型首字返回：第N轮 X.XXX秒（流式连接已建立）」，上报流式首字延迟。 |
| `log_react_thought` | 清洗后打印模型本轮思考内容（单行、截断 2000 字符）。 |
| `log_tool_call_started` | 规范化后打印本轮即将调用的工具名列表，空则输出「无」。 |
| `log_tool_call_completed` | 打印本轮工具名与执行耗时，刻意不输出工具返回的大块结果。 |
| `log_react_final_answer` | 最终回答日志钩子，刻意留空，因为答案已通过返回值交给调用方。 |
| `log_react_parse_issue` | 以 WARNING 级别打印工具调用格式问题，便于诊断卡住的工具调用。 |
| `_tool_names` | 把工具名可迭代对象过滤非字符串与空串、去重并升序排序成列表。 |
| `_clean_text` | 把任意文本压成单行、折叠空白，并截断到默认 2000 字符（末尾加 `...`）。 |
