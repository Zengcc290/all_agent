# agents/message_utils.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时里的一组「形状转换（shape translation）」工具函数集合，专门负责把各家大模型 SDK 返回的、类型五花八门的对象，统一规范化成 JSON 兼容的普通 Python 字典或字符串。不同的 Provider SDK 返回值风格差异很大：有的返回 Pydantic 模型（带 `model_dump` 方法），有的是 dataclass，有的干脆就是普通 dict，还有的只是带属性的普通对象；如果没有一层统一转换，上层的对话循环、流式适配器、工具调用执行器就得各自写一份兼容代码。文件开头的模块文档明确说明了它的历史动机：在它出现之前，`_field` 以及消息 / 工具调用转换逻辑在 `agent.py`（函数调用循环）、`react.py`（文本协议）和 `llm.py`（流式适配器）里被各复制了一份，结果就是「在某个模块里修好的 Provider 怪癖，在别的模块里依然是坏的」，所以这些逻辑被抽取到这里集中维护。

文件本身刻意保持「零依赖」：它只导入了标准库的 `json`、`collections.abc.Mapping` 和 `typing.Any`，不导入任何 Agent 运行时代码（模块文档最后一句专门强调 `Nothing here imports the agent runtime: these helpers only translate shapes.`），因此它可以被任意模块安全导入而不会产生循环导入。

文件内容按顺序由三部分组成：模块级常量（`DEFAULT_TOOL_NAME`、`MAX_TOOL_NAME_CHARS`、`MAX_TOOL_ERROR_CHARS`）、六个模块级函数（`field`、`safe_tool_name`、`safe_tool_call_error`、`tool_call_dict`、`message_dict`、`result_json`）、以及显式声明导出清单的 `__all__` 列表。文件里没有任何类，也没有嵌套函数，全部是扁平的模块级函数。

运行时的典型调用路径大致是：LLM 适配层拿到 SDK 的 assistant message 后调用 `message_dict` 把消息（含其中的 `tool_calls` 列表，逐个走 `tool_call_dict`）落成普通字典，写进对话 transcript；工具执行出错时用 `safe_tool_call_error` 生成有长度上限的错误文本；工具名在打日志或做占位符时先过 `safe_tool_name` 做截断保护；工具执行结果需要回填给 Provider 时用 `result_json` 序列化成字符串。`field` 则是所有转换函数底层的公共取值原语，被 `tool_call_dict`、`message_dict` 间接依赖。整体设计目标可以概括为：**输入宽容（任何形状都能吃）、输出确定（永远是 JSON 兼容结构）、输出有界（名字和错误文本都截断，避免日志与请求体被撑爆）**。

---

## 二、函数与类逐条详解

### `field(value, key, default=None) -> Any` （第 23 行）
- **作用**：这是一个统一的「取值原语」，用来从两种完全不同的数据载体里按同一个语义读取某个字段：如果 `value` 是映射类型（字典及其同类），就按 key 取值；否则就当成对象，按属性名取值。它存在的意义是消灭调用方到处写 `isinstance(x, dict)` 分支的重复代码——Provider SDK 一会儿返回 dict、一会儿返回对象，上层只想知道「这个字段是什么」，不该关心载体形态。它是本文件里 `tool_call_dict`、`message_dict` 能够同时兼容 dict 与对象两种输入的基础设施。当字段缺失时它不会抛异常，而是安静地退回 `default`，这让它特别适合处理 SDK 版本之间字段增删的情况。
- **参数**：
  - `value: Any`：待取值的载体，可以是 `Mapping`（`dict`、以及其它实现了 `collections.abc.Mapping` 的只读映射类型），也可以是任意普通对象；传 `None` 也不会崩，会走到 `getattr(None, key, default)` 从而返回 `default`。
  - `key: str`：要读取的字段名或属性名，映射走 key 查找、对象走属性查找，两者共用同一个字符串。
  - `default: Any = None`：字段不存在时的兜底返回值，默认 `None`；调用方可以传任意对象（例如 `tool_call_dict` 里给 `type` 传了 `"function"`、给 `arguments` 传了 `"{}"`）。
- **返回**：返回读到的字段值，类型为 `Any`（取决于载体里存的是什么）。当 `value` 是映射且 `key` 不在其中时返回 `default`；当 `value` 不是映射且对象上没有该属性时返回 `default`；其余情况返回真实值。注意：如果字段存在但其值本身就是 `None`，返回的也是 `None`，此时与「缺失」在返回值上不可区分。
- **内部流程**：第一步用 `isinstance(value, Mapping)` 判断载体是否为映射类型；第二步，若为映射，直接 `value.get(key, default)` 返回（`.get` 天然带默认值，不抛 `KeyError`）；第三步，若不是映射，走 `getattr(value, key, default)` 返回（`getattr` 的三参数形式在属性缺失时返回默认值，不抛 `AttributeError`）。整个函数没有循环、没有分支嵌套，是一条两分支的直路。
- **异常/边界**：正常的缺失场景都被 `Mapping.get` 与 `getattr` 的三参数形式吞掉，不会抛 `KeyError` / `AttributeError`。但在极端边界下仍可能抛异常：如果传进来的对象定义了会抛异常的 `__getattr__` 或属性 property（例如延迟加载属性内部报错），异常会原样向上冒泡；如果 `key` 不是字符串（比如传了 `int`），映射分支下 `.get` 通常仍能工作（若字典用非字符串键），但对象分支的 `getattr` 会抛 `TypeError`。`value` 为 `None` 时返回 `default`，无特殊处理。
- **同文件关系**：它不调用本文件里的任何其它函数，是纯叶子函数。被本文件里的 `tool_call_dict`（多次调用，取 `id`、`type`、`function` 以及 function 下的 `name`、`arguments`）和 `message_dict`（通过 `tool_call_dict` 间接使用）调用。

### `safe_tool_name(value) -> str` （第 31 行）
- **作用**：把任意可能是「工具名」的输入，规整成一个安全、有长度上限的字符串，用于日志输出和错误信息里的占位符。工具名可能来自模型生成的 JSON（不可信）、可能缺失、可能是 `None`、也可能是非字符串类型，直接拼进日志或错误消息里会带来两类风险：一是 `None` 之类会污染日志可读性，二是超长字符串（模型偶尔会吐出一个巨型名字）会把日志行或前端展示撑爆。这个函数通过「非字符串一律替换为默认名 + 去空白 + 空值兜底 + 硬截断」四重处理，保证输出永远是一个 1 到 200 字符之间的可用名字，绝不返回空串。它在工具调用参数解析失败、工具不存在、工具执行报错等路径上被用来标注「到底是哪个工具出的问题」。
- **参数**：
  - `value: Any`：候选工具名。只有 `str` 类型才会被真正使用；`None`、数字、字典、对象等任何非字符串类型都会被整体替换为常量 `DEFAULT_TOOL_NAME`（`"unknown.tool"`）。类型注解写的是 `Any`，正是因为调用方拿到的常常是未经验证的原始值。
- **返回**：返回 `str`。分支结果如下：`value` 是字符串时先做 `.strip()` 去首尾空白，若去空白后为空串（例如 `"   "`）则用 `DEFAULT_TOOL_NAME` 顶替；`value` 不是字符串时直接用 `DEFAULT_TOOL_NAME`。最终结果统一再切 `[:MAX_TOOL_NAME_CHARS]`，即最多保留前 200 个字符。因此返回值长度范围是 1~200 个字符（最短情况即 `"unknown.tool"` 本身）。
- **内部流程**：第一步，`isinstance(value, str)` 判断类型；是字符串就取 `value.strip()`，否则取 `DEFAULT_TOOL_NAME`。第二步，`(name or DEFAULT_TOOL_NAME)` 做空值兜底——利用空串的假值特性，把 `strip()` 之后变成 `""` 的情况换成默认名。第三步，对结果做切片 `[:MAX_TOOL_NAME_CHARS]` 截断并返回。没有循环，没有异常捕获。
- **异常/边界**：字符串截断是按字符（码点）切，不感知 Unicode 组合字符或代理对，理论上可能把某个 emoji 的代理对切断（Python 的 `str` 按码点切，通常安全，但会切掉半个组合序列）。`value` 为 `None`、空串、纯空白串、非字符串类型都已被显式兜底，返回 `DEFAULT_TOOL_NAME` 或截断后的结果，不会抛异常。无特殊异常处理（没有 try/except）。
- **同文件关系**：它不调用本文件里的任何函数（只读取模块级常量 `DEFAULT_TOOL_NAME` 与 `MAX_TOOL_NAME_CHARS`），本文件内也没有函数调用它；它的使用者是同项目的 `agent.py`、`react.py`、`llm.py` 等模块（从模块文档可知这些模块共享本文件的辅助函数）。

### `safe_tool_call_error(error) -> str` （第 38 行）
- **作用**：把工具执行过程中捕获到的异常对象，转换成一段可以安全放进日志、错误消息或回传给模型的文本，并且强制限制长度。工具执行失败时，异常信息可能来自第三方库、可能包含巨大的堆栈描述、甚至可能是模型生成的超长字符串，如果原样透传，轻则日志刷屏，重则让回传给 Provider 的消息体膨胀到超出上下文或请求体积限制。这个函数同时解决了另一个隐蔽问题：某些异常（如 `TimeoutError()`、自定义异常未传消息）的 `str()` 结果为空串，直接拼进错误提示会得到「工具 X 失败：」这样没有信息量的句子，因此它在消息为空时退化使用异常类名作为替代，保证输出永远有内容。
- **参数**：
  - `error: Exception`：捕获到的异常实例。注解要求是 `Exception` 子类，但函数内部只依赖 `str(error)` 和 `type(error).__name__`，所以任何对象（哪怕不是异常）在运行时也能工作。
- **返回**：返回 `str`，长度不超过 `MAX_TOOL_ERROR_CHARS`（1000）个字符。具体规则：先取 `str(error)`，若结果为空串（假值）则改用 `type(error).__name__`（例如 `"TimeoutError"`），最后统一切前 1000 个字符返回。由于异常类名不可能为空，返回值保证非空。
- **内部流程**：第一步 `str(error)` 求异常消息；第二步用 `or` 做短路兜底——`str(error)` 为空时取 `type(error).__name__`；第三步 `[:MAX_TOOL_ERROR_CHARS]` 截断并返回。没有 try/except 包裹 `str(error)` 本身，也没有循环。
- **异常/边界**：如果某个异常类重写了 `__str__` 且该方法自身抛异常，这里的 `str(error)` 会把该异常抛出去（没有捕获），这是一个已知的边界缺口。截断发生在字符维度，超长错误只保留前 1000 字符，后面的堆栈细节会被丢弃（这是刻意的有界化设计）。`error` 为 `None` 时不满足注解，但运行时会得到 `str(None) == "None"`，不会崩。
- **同文件关系**：不调用本文件里的任何函数（只使用模块级常量 `MAX_TOOL_ERROR_CHARS`），本文件内也没有函数调用它；由 `agent.py`、`react.py` 等执行工具调用的模块在异常处理分支中使用。

### `tool_call_dict(item) -> dict[str, Any]` （第 45 行）
- **作用**：把单个 SDK 返回的工具调用对象（tool call）转换成一个 JSON 兼容的普通字典，也就是 OpenAI 消息格式里 `tool_calls` 数组的一个元素。不同 Provider SDK 的工具调用对象字段名大体一致（`id`、`type`、`function.name`、`function.arguments`），但有的嵌套是对象、有的嵌套是字典，字段也可能缺失（比如某些实现不返回 `id`）。这个函数把嵌套结构拍平重建，并做两件关键的清洗：一是给缺失字段填上合理默认值（`type` 默认 `"function"`、`arguments` 默认 `"{}"`），二是把值为 `None` 的顶层键整个删掉，避免生成 `{"id": null}` 这种在严格 Provider 侧可能被拒或在 JSON 往返中产生歧义的字段。它是 `message_dict` 处理 `tool_calls` 列表时的逐元素处理器。
- **参数**：
  - `item: Any`：一个工具调用对象，可以是带 `id` / `type` / `function` 属性的 SDK 对象，也可以是普通 `dict`。两种形态都由 `field` 统一兼容。
- **返回**：返回 `dict[str, Any]`，即 JSON 兼容的工具调用结构。典型形态为 `{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}`。过滤规则：顶层键中值为 `None` 的会被剔除，因此当 `id` 缺失时结果里干脆没有 `id` 键；`type` 缺失时因为默认值 `"function"` 不为 `None` 而被保留；`function` 子字典内部的键**不做** `None` 过滤，因此 `name` 缺失时会出现 `"name": None`，`arguments` 缺失时会得到字符串 `"{}"`。
- **内部流程**：第一步，用 `field(item, "function")` 取出内层 function 对象（无默认值，缺失即 `None`）。第二步，构造 `result` 字典，其中 `id` 用 `field(item, "id")`（缺失为 `None`）、`type` 用 `field(item, "type", "function")`（缺失兜底为字符串 `"function"`）、`function` 子字典里 `name` 用 `field(function, "name")`、`arguments` 用 `field(function, "arguments", "{}")`。第三步，用字典推导式 `{key: value for key, value in result.items() if value is not None}` 过滤掉值为 `None` 的顶层键并返回。注意 `function` 子字典本身一定不是 `None`（因为它是新构造的字面量），所以它永远不会被过滤掉，即使内层 `function` 参数取不到值也会保留一个 `{"name": None, "arguments": "{}"}` 结构。
- **异常/边界**：`item` 为 `None` 时不抛异常，`field` 会把所有取值都退化为默认值，最终返回 `{"type": "function", "function": {"name": None, "arguments": "{}"}}`。`arguments` 字段如果是对象而不是字符串，会被原样放进字典（本函数不做 JSON 字符串化，序列化交给 `result_json` 或上层）。`field` 本身可能因对象的 `__getattr__` 抛异常而冒泡。无 try/except。
- **同文件关系**：调用本文件里的 `field`（共 5 次：`function`、`id`、`type`、`name`、`arguments`）。被本文件里的 `message_dict` 在规范化 `tool_calls` 列表时逐元素调用。

### `message_dict(message) -> dict[str, Any]` （第 60 行）
- **作用**：把 Provider SDK 返回的 assistant 消息对象规范化为一个纯字典，作为写入对话 transcript（历史消息列表）的标准形态。这是本文件里最核心、兼容分支最多的函数：它要同时应付三种输入形态——带 `model_dump` 的 Pydantic 模型、已经是 dict 的普通映射、以及只暴露属性的普通对象。转换后还需要保证两件事：一是消息一定有 `role` 字段（缺失时补 `"assistant"`，因为该函数语义上就是处理助手消息的），二是 `tool_calls` 里的每个元素都必须是 JSON 兼容的普通字典（模型对象不能直接塞进 transcript，否则后续序列化或比较会出问题）。它在函数调用循环每收到一轮模型回复时被调用，是「SDK 对象世界」与「项目内部纯数据世界」之间的边界闸门。
- **参数**：
  - `message: Any`：待规范化的消息。三种受支持的形态：实现了 `model_dump` 的 Pydantic / SDK 模型对象；`dict`（或其子类，注意这里判断的是 `isinstance(message, dict)` 而非 `Mapping`）；以及任意带 `role` / `content` / `tool_calls` 属性的普通对象。`None` 会走第三个分支，被当成「什么字段都没有的对象」处理。
- **返回**：返回 `dict[str, Any]`，一个 JSON 兼容的消息字典。至少包含 `role`（缺失时被补成 `"assistant"`）。若原消息带 `content`，则包含 `content`；若带非 `None` 的 `tool_calls`，则包含已被逐元素转换为普通字典的 `tool_calls` 列表；若带其它字段（仅 Pydantic 分支可能带来，如 `refusal`、`audio`、`name` 等），也会一并保留。Pydantic 分支因为用了 `exclude_none=True`，其输出中不会出现值为 `None` 的字段。
- **内部流程**：第一步，按优先级选择提取方式——若 `hasattr(message, "model_dump")` 为真，调用 `message.model_dump(exclude_none=True)` 得到字典（`exclude_none=True` 让所有 `None` 字段被剔除，输出更干净）；否则若 `isinstance(message, dict)`，用 `dict(message)` 做一次浅拷贝（拷贝的意义是后续 `setdefault` 和改写 `tool_calls` 不会污染调用方传进来的原字典）；否则进入兜底分支，用字典推导式配合海象运算符 `:=` 遍历 `("role", "content", "tool_calls")` 三个键名，逐个 `getattr(message, key, None)`，只保留值不为 `None` 的键。第二步，`data.setdefault("role", "assistant")` 保证 `role` 一定存在；注意它只在键**完全不存在**时才写入，因此如果消息里显式带了 `"role": None`，这里不会覆盖成 `"assistant"`。第三步，判断 `"tool_calls" in data and data["tool_calls"] is not None`，成立时把该字段重写为 `[tool_call_dict(item) for item in data["tool_calls"]]`，逐个转换；若 `tool_calls` 为空列表，条件仍成立（空列表不是 `None`），会得到一个空列表 `[]`，语义正确。最后返回 `data`。
- **异常/边界**：如果 `message.model_dump` 存在但不是可调用对象（例如同名属性是字符串），`hasattr` 判断通过但调用时会抛 `TypeError`，本函数不做防御。`model_dump(exclude_none=True)` 若 SDK 的该方法不接受 `exclude_none` 关键字（老版本 Pydantic v1 风格），会抛 `TypeError`，同样未捕获——这与 `result_json` 的两级兜底形成对比。`message` 为 `None` 时走兜底分支，得到 `{"role": "assistant"}`。`tool_calls` 为 `None` 时保持 `None` 不动（不会被转成列表）。`tool_calls` 若不是可迭代对象（比如是字符串），列表推导会逐字符迭代并传入 `tool_call_dict`，产生一串畸形但不会崩的字典。无 try/except。
- **同文件关系**：调用本文件里的 `tool_call_dict`（在列表推导中对每个工具调用元素调用）。本文件内没有其它函数调用它；由 `agent.py`（函数调用循环）、`react.py`（文本协议）、`llm.py`（流式适配器）等模块调用。

### `result_json(result) -> str` （第 79 行）
- **作用**：把工具函数的执行结果对象序列化成 JSON 字符串，用于回填给 Provider（例如作为 role 为 `tool` 的消息的 `content`）。它面对的现实是：工具返回值通常是 Pydantic 模型，而模型里常带有 `Any` 类型的宽松字段，里面可能塞着 `datetime`、`Decimal`、自定义对象等 `json.dumps` 默认不认识的东西。因此它设计了两级降级：先尝试 `model_dump(mode="json")`（Pydantic v2 的 JSON 模式，会把日期时间等转成字符串，最理想）；如果这个方法不存在或抛错（例如 Pydantic v1 模型、或 `mode` 参数不被接受），退回普通 `model_dump()`；最后 `json.dumps` 时再挂上 `default=str` 作为终极保险，让任何仍不可序列化的对象通过 `str()` 退化表示，保证函数**永远**能返回一个字符串而不是抛 `TypeError`。
- **参数**：
  - `result: Any`：工具的执行结果。注解是 `Any`，因为工具可以返回任何东西。设计上假设它至少拥有 `model_dump` 方法；如果完全没有（例如返回了一个普通 dict 或一个数字），两级 `model_dump` 调用都会失败，异常会被捕获，但随后的 `json.dumps(payload, ...)` 里 `payload` 会处于「可能未赋值」的状态——这是本函数一个真实的边界风险（见下）。
- **返回**：返回 `str`，一个 JSON 文本，中文不会被转义（`ensure_ascii=False`）。典型输出形如 `{"field": "值", "when": "2024-01-01T00:00:00"}`。返回内容不含额外的换行或缩进（未传 `indent`），便于直接作为消息内容传输。
- **内部流程**：第一步，`try` 块执行 `result.model_dump(mode="json")`，成功则 `payload` 是 JSON 模式下的字典（日期、UUID 等已被转成基本类型）。第二步，若上一步抛任何异常（注释标明 `# noqa: BLE001`，说明是刻意宽泛捕获），`except` 分支执行 `payload = result.model_dump()`，即退回默认模式的字典（此时内部可能仍有不可直接序列化的值）。第三步，`json.dumps(payload, ensure_ascii=False, default=str)`：`ensure_ascii=False` 保留原始中文与 Unicode 字符，`default=str` 让 `json` 编码器在遇到不认识的对象时调用 `str()` 生成兜底表示，而不是抛 `TypeError`。第四步，返回该字符串。
- **异常/边界**：两级 `model_dump` 的失败被 `except Exception` 吞掉，但吞掉之后 `payload` 可能从未被赋值：如果 `result` 根本没有 `model_dump` 方法，第一次调用抛 `AttributeError` 被捕获，`except` 里的 `result.model_dump()` 会再次抛 `AttributeError` 并**逃出函数**（该 except 块自身没有保护）。因此本函数的「永远返回字符串」保证只对「拥有 `model_dump` 的对象」成立；传入普通 dict 或标量时会抛 `AttributeError`。此外，`json.dumps` 遇到循环引用仍会抛 `ValueError: Circular reference detected`（`default=str` 不解决循环结构），遇到不可哈希/无法 `str()` 的极端对象理论上也可能失败。空对象、空模型都会正常序列化为 `{}`。
- **同文件关系**：不调用本文件里的任何其它函数，是独立的叶子函数。本文件内也没有函数调用它；由执行工具并把结果回填给 Provider 的模块（如 `agent.py` 的工具结果处理路径）调用。

### 模块级常量与导出清单（第 18–20 行、第 89–98 行）
- **作用**：文件顶部定义了三个常量，被上面的函数共同使用，集中放置使得「默认工具名」和「长度上限」这类策略值只在一个地方维护，便于统一调整。`DEFAULT_TOOL_NAME = "unknown.tool"` 是工具名无法确定时使用的占位名，其点号命名风格与项目中「命名空间.动作」式的工具命名保持一致，能让日志里一眼看出这是兜底值而不是真实工具。`MAX_TOOL_NAME_CHARS = 200` 限制工具名的最大字符数。`MAX_TOOL_ERROR_CHARS = 1000` 限制错误文本的最大字符数。文件末尾的 `__all__` 是一个字符串列表，按字母序列出了三个常量名和六个函数名，用于声明模块的公开接口，使得 `from agents.message_utils import *` 只会导入这些名字，同时向静态检查工具表明这些符号是刻意对外暴露的。
- **参数**：不适用（常量与导出清单没有参数）。
- **返回**：不适用。
- **内部流程**：常量在模块导入时直接绑定；`__all__` 是普通列表字面量，不执行任何逻辑。
- **异常/边界**：无特殊处理。若有人修改 `__all__` 但漏加新函数，`import *` 将不会导出该函数（不影响按名导入）。
- **同文件关系**：`DEFAULT_TOOL_NAME` 被 `safe_tool_name` 使用；`MAX_TOOL_NAME_CHARS` 被 `safe_tool_name` 使用；`MAX_TOOL_ERROR_CHARS` 被 `safe_tool_call_error` 使用；`__all__` 引用全部六个函数名与三个常量名，与函数定义相互对应，属于本文件内部的声明性关联。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `DEFAULT_TOOL_NAME`（常量） | 工具名无法确定时使用的占位字符串 `"unknown.tool"`。 |
| `MAX_TOOL_NAME_CHARS`（常量） | 工具名截断长度上限，值为 200 个字符。 |
| `MAX_TOOL_ERROR_CHARS`（常量） | 工具错误文本截断长度上限，值为 1000 个字符。 |
| `field(value, key, default=None)` | 从映射或对象中按同一语义读取字段，缺失时返回默认值。 |
| `safe_tool_name(value)` | 把任意输入规整为有长度上限、绝不为空的工具名，用于日志与占位。 |
| `safe_tool_call_error(error)` | 把异常转成有长度上限且保证非空的错误文本。 |
| `tool_call_dict(item)` | 把单个 SDK 工具调用对象转成 JSON 兼容的字典，并剔除值为 `None` 的顶层键。 |
| `message_dict(message)` | 把 Pydantic 模型 / dict / 普通对象形态的助手消息统一规范化为纯字典。 |
| `result_json(result)` | 把工具执行结果按两级 `model_dump` 降级后序列化为中文不转义的 JSON 字符串。 |
| `__all__`（导出清单） | 声明本模块对外公开的三个常量与六个函数名。 |
