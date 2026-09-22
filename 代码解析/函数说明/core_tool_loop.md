# core/tool_loop.py

## 一、这个文件是干什么的

这个文件是整个项目里「模型 ↔ 工具」多轮对话的公共轮次控制器，只提供一套与具体协议无关的循环骨架。文件顶部的文档字符串明确说明了它的设计立场：这个循环刻意不去理解 ReAct 文本格式，也不理解各家模型原生的 function call 结构，那些解析工作全部交给各适配器（adapter）去完成。适配器负责解析一次模型响应，然后复用这里的 `ToolLoop` 类来处理所有适配器都共有的生命周期问题：轮次上限、`None` 响应（模型什么都没返回）时的安全兜底上限、以及同一工具调用被反复重复时的死循环检测。

从代码内容上看，这个文件非常小，只包含一个模块级类 `ToolLoop`，它下面有 `__init__`、`rounds`、`record_call` 三个方法，外加一个类属性 `DEFAULT_SAFETY_LIMIT` 和模块级的 `__all__` 导出声明。它不依赖项目里的任何重型模块，只导入标准库的 `json`、`collections.abc` 的 `Iterator`/`Mapping`、`typing.Any`，以及项目常量模块 `constants` 中的 `TOOL_LOOP_SAFETY_LIMIT`。

运行时的典型用法是：适配器创建一个 `ToolLoop` 实例（可能带业务侧配置的 `max_rounds`），用 `for round_number in loop.rounds():` 驱动多轮对话；每轮模型返回工具调用时，先把调用交给 `record_call` 记账并接受死循环检查，再真正执行工具。这样做的价值在于把「防跑飞」的策略集中到一处：轮次硬上限由 `max_rounds` 或 `safety_limit` 决定，而「同一个调用重复超过三次」这种模型卡死的典型症状会被立即抛错终止，而不是无限消耗 token 和时间。

`rounds()` 之所以写成生成器而不是简单的 `range()`，是因为 `max_rounds` 允许为 `None`（表示业务上没有显式轮次限制），此时要退化到 `safety_limit` 作为安全上限，生成器能自然地把这个二选一的判断封装在循环条件里。`record_call` 用「名字 + 参数」序列化成一个字符串签名来计数，因此对参数顺序不敏感（`sort_keys=True`），也能容纳不可 JSON 序列化的参数对象（`default=str` 兜底）。

另外要注意，这个文件的类属性 `DEFAULT_SAFETY_LIMIT` 只是 `constants.TOOL_LOOP_SAFETY_LIMIT` 的兼容别名，代码注释写明常量的单一来源是 `constants`，类属性保留仅为兼容旧的引用写法。

## 二、函数与类逐条详解

### `class ToolLoop` （第 18 行）
- **作用**：这是本文件唯一的模块级类，扮演「有界对话轮次生成器 + 工具调用死循环探测器」两个角色。它存在的理由是让所有模型适配器共享同一套循环安全策略，而不必各自重复实现轮次计数和重复调用检测。当适配器需要驱动一次「模型输出 → 解析工具调用 → 执行工具 → 把结果回填 → 再问模型」的多轮流程时，就会实例化它。它本身不持有任何 provider 客户端、不发起网络请求、不解析响应文本，纯粹是一个状态容器加两个操作接口。它通过 `rounds()` 向外提供轮次序列，通过 `record_call()` 接收每轮的调用记录，一旦发现同一个调用重复过多就抛出异常来强制终止外层循环。类内部用实例字典 `_calls` 保存「调用签名 → 出现次数」的映射，因此同一个实例的计数在整个生命周期内是累积的、不会自动清零。
- **参数**：类本身不接受参数，实例化参数由 `__init__` 定义（`max_rounds` 与仅限关键字传入的 `safety_limit`）。
- **返回**：类是构造器，返回 `ToolLoop` 实例。
- **内部流程**：类定义体内先声明类属性 `DEFAULT_SAFETY_LIMIT = TOOL_LOOP_SAFETY_LIMIT`，把 `constants` 里的常量绑定为类级默认值，供 `__init__` 的默认参数引用；随后依次定义 `__init__`、`rounds`、`record_call` 三个方法。类没有任何继承（隐式继承 `object`），也没有定义 `__slots__`、类方法或静态方法。
- **异常/边界**：类定义阶段若 `constants` 模块导入失败会直接 `ImportError`；若 `constants.TOOL_LOOP_SAFETY_LIMIT` 不存在则 `ImportError`；类体本身不做数值校验，校验全部推迟到实例化时。
- **同文件关系**：类内部的方法之间不互相调用，只有 `__init__` 会引用类属性 `DEFAULT_SAFETY_LIMIT` 作为默认值；`rounds` 和 `record_call` 分别读取 `__init__` 建立的实例状态。类通过模块末尾的 `__all__ = ["ToolLoop"]` 声明为对外导出的公开接口。

### `ToolLoop.DEFAULT_SAFETY_LIMIT` （类属性，第 22 行）
- **作用**：这是类级别的安全上限默认值，语义是「当调用方没有给出显式轮次上限时，最多允许跑多少轮」。它被声明为 `constants.TOOL_LOOP_SAFETY_LIMIT` 的直接别名，代码注释说明常量真正的单一来源是 `constants` 模块，这里保留一份类属性只是为了兼容仍然按 `ToolLoop.DEFAULT_SAFETY_LIMIT` 方式引用旧写法的调用方。这样即便常量的数值在 `constants` 里被调整，类属性也会自动跟着变，不会出现两处数值不一致。
- **参数**：无（它是属性而非函数）。
- **返回**：一个整数常量（具体数值由 `constants.TOOL_LOOP_SAFETY_LIMIT` 决定，本文件不做数值假设）。
- **内部流程**：在类体执行阶段完成一次名字绑定，把 `constants` 模块里读到的值赋给类命名空间中的 `DEFAULT_SAFETY_LIMIT`；随后 `__init__` 的默认参数表达式在函数定义时求值并引用它。
- **异常/边界**：若 `constants` 中没有该名字，类定义时就会 `ImportError`，整个模块无法导入；若该常量的值不满足「正整数」约束，错误会在 `__init__` 的校验里以 `ValueError` 形式暴露，而不是在这里。
- **同文件关系**：被 `__init__` 的默认参数 `safety_limit: int = DEFAULT_SAFETY_LIMIT` 引用；不被 `rounds` 或 `record_call` 直接读取（它们读的是实例属性 `self.safety_limit`）。

### `ToolLoop.__init__(self, max_rounds: int | None = None, *, safety_limit: int = DEFAULT_SAFETY_LIMIT) -> None` （第 24 行）
- **作用**：构造一个 `ToolLoop` 实例，并在这里完成全部参数合法性校验，把「非法配置」挡在对象创建阶段而不是等到循环跑到一半才炸。它把显式轮次上限 `max_rounds`（允许为 `None`，表示业务不限制）和安全兜底上限 `safety_limit` 保存为实例属性，同时初始化用于死循环检测的计数字典 `_calls`。之所以把 `safety_limit` 设计成仅限关键字参数（`*` 之后），是为了防止调用方误把两个整型上限按位置顺序传反而得不到预期行为，让代码在调用点更可读。校验里特意排除了 `bool`，因为 Python 中 `bool` 是 `int` 的子类，`True`/`False` 会被 `isinstance(x, int)` 判定为真，若不排除就可能出现 `max_rounds=True` 被当成 1 轮这种隐蔽的误用。
- **参数**：
  - `self`：实例本身，Python 自动传入。
  - `max_rounds`（`int | None`，默认 `None`）：显式轮次上限。取值必须是 `None`，或者大于等于 1 的整数；`None` 表示不设业务上限，此时实际生效的上限是 `safety_limit`。禁止传入 `bool`、浮点数、字符串、0 或负数。
  - `safety_limit`（`int`，默认 `DEFAULT_SAFETY_LIMIT`，仅限关键字传入）：安全兜底上限，必须是不小于 1 的整数，禁止 `bool`。它同时充当「`max_rounds` 为 `None` 时的实际轮次上限」和「防止模型陷入 `None` 响应空转的保险丝」。
- **返回**：`None`；构造函数只做副作用（写入实例属性）。
- **内部流程**：第一步用 `if max_rounds is not None and (...)` 组合条件校验 `max_rounds`：命中 `isinstance(max_rounds, bool)`、或 `not isinstance(max_rounds, int)`、或 `max_rounds < 1` 中任意一条就抛 `ValueError("max_rounds must be None or a positive integer")`。第二步以同样思路校验 `safety_limit`，不允许 `None`，命中 `bool`、非 `int`、或 `< 1` 就抛 `ValueError("safety_limit must be a positive integer")`。第三步依次赋值 `self.max_rounds = max_rounds`、`self.safety_limit = safety_limit`。第四步初始化 `self._calls: dict[str, int] = {}`，建立空的「调用签名 → 次数」字典，保证每个实例的重复调用统计互相独立。
- **异常/边界**：`max_rounds` 为 `True`/`False`、`0`、负数、浮点数、字符串等非法值时抛 `ValueError`；`safety_limit` 为 `True`/`False`、`0`、负数、非整数时抛 `ValueError`；`max_rounds` 为 `None` 是合法值，不会被拒。两个参数都合法时不做任何额外动作，也不会触发任何 I/O。
- **同文件关系**：引用类属性 `DEFAULT_SAFETY_LIMIT` 作为 `safety_limit` 的默认值；不调用本文件的其他方法。它写入的 `self.max_rounds`、`self.safety_limit` 被 `rounds` 读取，`self._calls` 被 `record_call` 读写。

### `ToolLoop.rounds(self) -> Iterator[int]` （第 35 行）
- **作用**：这是一个生成器方法，按顺序产出从 1 开始的轮次编号，直到配置的上限为止，供适配器用 `for` 循环驱动多轮对话。它把「到底以哪个数字为上限」的判断完全封装起来：如果构造时传了 `max_rounds`，就以 `max_rounds` 为界；如果传的是 `None`，就退化到 `safety_limit` 为界。这样调用方无需自己写 `max_rounds or safety_limit` 这类容易出错的表达式，也保证了「即使没有业务轮次限制，也绝不会无限循环」这一安全属性。它是一次性迭代器：每次调用都返回一个全新的生成器对象，从头开始重新计数，不会记忆上一次迭代到哪里；但生成器本身不修改实例状态，因此同一个实例可以被多次、甚至嵌套地调用 `rounds()`。
- **参数**：
  - `self`：实例本身，用于读取 `self.max_rounds` 与 `self.safety_limit`。
- **返回**：返回一个 `Iterator[int]`（生成器对象），依次产出 `1, 2, 3, ...` 直到上限。上限为 `max_rounds`（当它不是 `None` 时）或 `safety_limit`（当 `max_rounds` 为 `None` 时）；由于两者在 `__init__` 中都被校验为不小于 1，生成器至少会产出一个轮次编号。
- **内部流程**：先把局部计数器 `round_number` 初始化为 `0`；然后进入 `while` 循环，循环条件是一个或运算的两分支：当 `self.max_rounds is None` 时判断 `round_number < self.safety_limit`，当 `self.max_rounds is not None` 时判断 `round_number < self.max_rounds`；条件成立则先把 `round_number` 自增 1，再用 `yield` 把该编号交给调用方；调用方请求下一个值时循环重新求条件，直到达到上限后 `while` 结束，生成器自然 `StopIteration`，迭代终止。整个过程没有对外部状态做任何写入。
- **异常/边界**：正常情况下不抛异常。若实例属性在外部被改成非法值（例如被赋为字符串），比较运算会抛 `TypeError`；若 `safety_limit` 被外部改成 0 或负数，则生成器不产出任何值、直接结束（在 `__init__` 正常校验过的前提下不会发生）。生成器没有超时或取消机制，终止完全由调用方停止迭代或上限耗尽决定。
- **同文件关系**：只读取 `__init__` 写入的 `self.max_rounds` 和 `self.safety_limit`；不调用本文件任何其他函数或方法；也不被本文件内的其他函数调用（它的调用方是外部的模型适配器）。

### `ToolLoop.record_call(self, name: str, arguments: Mapping[str, Any] | Any) -> None` （第 45 行）
- **作用**：记录一次工具调用并做重复检测，目的是在 provider（模型侧）无休止重复同一个调用时快速失败，避免整个 Agent 循环卡死、空耗 token。它把「工具名 + 参数」规范化序列化成一个唯一签名，然后在实例的 `_calls` 字典里把该签名的计数加一；一旦同一签名的累计次数超过 3，就抛出 `RuntimeError` 强制中断。之所以要序列化而不是直接比较参数对象，是因为字典等可变容器不可哈希、也不支持可靠的相等比较顺序，序列化成排序后的 JSON 字符串可以得到稳定、可哈希的键。`ensure_ascii=False` 让中文等非 ASCII 参数保持可读，`separators=(",", ":")` 去掉多余空白使签名紧凑，`default=str` 则保证即使参数里含有不可 JSON 序列化的对象（例如自定义类实例、`datetime`），也能退化成字符串表示而不是直接抛序列化异常。需要注意的是计数是累积的、不会因为中间夹了别的成功调用而清零，即「同一个调用在整段会话里出现第 4 次」就会触发。
- **参数**：
  - `self`：实例本身，用于读写 `self._calls`。
  - `name`（`str`）：工具名称，通常是函数名或 provider 给出的工具标识；它作为签名数组的第一个元素参与序列化。
  - `arguments`（`Mapping[str, Any] | Any`）：该次调用的参数。声明上既接受字符串键的映射（典型是 `dict`），也接受任意对象（`| Any`），因为不同 provider 可能给出解析好的字典或原始参数对象；只要能被 `json.dumps` 处理（或经 `default=str` 兜底）即可。
- **返回**：`None`。它的可观察效果是更新 `self._calls` 中的计数，以及在超限时抛出异常。
- **内部流程**：第一步用 `json.dumps([name, arguments], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)` 生成签名字符串，其中把 `name` 和 `arguments` 放进同一个列表，保证「同名不同参」和「不同名同参」都能区分开。第二步用 `self._calls.get(signature, 0) + 1` 取出旧计数并加一，把结果写回 `self._calls[signature]`。第三步判断 `if count > 3`，成立则抛 `RuntimeError("tool loop stopped after the same tool call was repeated more than three times")`，把终止权交给外层调用者；不成立则方法正常返回。
- **异常/边界**：当同一签名累计出现第 4 次时抛 `RuntimeError`，此时该签名在字典里的计数已被更新为 4（异常抛出发生在写入之后），所以调用方若捕获异常后继续复用同一实例，再次调用该签名会立刻再次抛错。若 `arguments` 内含循环引用，`json.dumps` 会抛 `ValueError`；若含 `default=str` 也无法处理的对象（例如 `str()` 本身抛异常的极端对象），异常会向上传播。`name` 为空字符串不视为非法，会正常参与签名；`arguments` 为 `None` 也会被正常序列化为 `null` 并计数。方法自身不做任何去重清理，字典只增不减。
- **同文件关系**：读写 `__init__` 建立的 `self._calls` 字典；不调用本文件的其他方法，也不被本文件内的其他函数调用（调用方是外部的模型适配器，通常在每轮解析出工具调用后调用它）。它与 `rounds` 是互补关系：`rounds` 管轮次总量上限，`record_call` 管同一调用重复次数的上限。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `ToolLoop` | 模型与工具多轮对话的公共循环控制器，负责轮次上限与重复调用死循环检测。 |
| `ToolLoop.DEFAULT_SAFETY_LIMIT` | 类级安全上限默认值，是 `constants.TOOL_LOOP_SAFETY_LIMIT` 的兼容别名。 |
| `ToolLoop.__init__` | 校验并保存 `max_rounds` 与 `safety_limit`，初始化重复调用计数字典。 |
| `ToolLoop.rounds` | 生成器，按上限依次产出从 1 开始的轮次编号。 |
| `ToolLoop.record_call` | 序列化工具调用签名并计数，同一调用重复超过三次即抛 `RuntimeError`。 |
