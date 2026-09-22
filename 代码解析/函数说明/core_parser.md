# core/parser.py

## 一、这个文件是干什么的

`core/parser.py` 是整个 Agent 运行时里专门负责「把模型吐出来的工具调用内容变成内部结构化对象」的解析层。它处在大模型输出与工具执行引擎之间的边界上：上游是 OpenAI 兼容网关返回的原生 `tool_calls` 数组，或者 ReAct 文本协议里模型手写的 JSON 参数；下游是 `core/models.py` 里定义的 `ToolCall` 版本化信封，以及 `core/registry.py` 里的工具注册表。文件里没有任何执行逻辑，所有函数都是纯函数（除读取注册表状态外），因此可以安全地在「只解析、不落地」的探测场景里反复调用。它主要包含四类东西：一是两个公开入口 `parse_tool_calls`（解析裸 JSON 负载）与 `parse_openai_tool_calls`（解析原生调用并补齐版本号、schema 哈希、注册表代数）；二是一个容错的 JSON 解码器 `loads_model_json`；三是三个私有辅助函数 `_get`、`_balanced_json_substring`、`_repair_single_quoted_object`；四是一个常量拒绝钩子 `reject_json_constant`。之所以需要它，是因为真实模型输出极不稳定：会裹着解释性散文、会用单引号写 Python 风格字典、会漏掉调用 id、会写出 `NaN`。这个文件把这些噪声全部吸收掉，并统一抛错类型，让上层调用者只需要处理一种异常。它的核心设计原则是「绝不猜测、绝不执行」：参数只做结构与类型校验，不调用任何工具，不做副作用。文件顶部用 `from __future__ import annotations` 让所有注解延迟求值，因此 `str | dict[str, Any]` 这种联合类型写法在较老的 Python 上也能正常导入。模块级还引入了 `ast`、`json`、`Mapping`、`Any`、`ToolCall`、`BaseTool`、`ToolRegistry`，其中 `BaseTool` 只用于类型注解，说明注册表条目的形状是「工具对象 + 整数代数」的二元组。

## 二、函数与类逐条详解

### `parse_tool_calls(payload: str | dict[str, Any] | list[dict[str, Any]]) -> list[ToolCall]` （第 12 行）
- **作用**：这是本文件对外暴露的第一个公开入口，负责把一份「可能来自模型、也可能来自测试夹具或落盘记录」的工具调用负载，统一转换成 `ToolCall` 对象列表。它要解决的核心问题是输入形状的不确定性：调用方可能直接拿到模型返回的字符串（里面混着散文），也可能拿到已经 `json.loads` 过的字典（可能是单个调用对象，也可能是 `{"tool_calls": [...]}` 这种 OpenAI 风格的包装），还可能拿到一个列表。这个函数把这些形态全部归一化到「列表」这一种中间形态，再逐个做严格校验。它被设计成纯粹的解析动作，注释里明确写了 "without executing anything"，所以它可以在不产生任何副作用的前提下被反复调用，适合做预检、审计、回放。它不做工具名与注册表的比对，也不填充 `schema_version` 之类的字段，因此只能用于内部已知格式的负载；需要绑定注册表元信息时必须改用 `parse_openai_tool_calls`。当负载是字符串时，它会把解码工作完全委托给 `loads_model_json`，从而自动获得「包裹噪声剥离 + 单引号修复」的能力。
- **参数**：
  - `payload`：必填，类型为 `str | dict[str, Any] | list[dict[str, Any]]`，没有默认值。当它是 `str` 时，会先经过 `loads_model_json` 解码成 Python 对象；当它是 `dict` 时，会优先尝试取出 `"tool_calls"` 键的值，取不到就当作单个调用对象；当它是 `list` 时，直接逐项校验。传入其它类型（如 `int`、`None`、`tuple`）会在最后一步触发 `TypeError`。
- **返回**：返回 `list[ToolCall]`，顺序与输入列表一致，长度等于归一化后列表的长度。如果输入是空列表，则返回空列表；如果输入是 `{"tool_calls": []}` 这样的字典，同样返回空列表，不会因为「没有调用」而报错。
- **内部流程**：第一步判断 `isinstance(payload, str)`，成立则调用 `loads_model_json(payload)` 覆盖原变量。第二步再次判断是否为 `dict`，若是则执行 `payload = payload.get("tool_calls", payload)`，即优先解包 OpenAI 风格的 `tool_calls` 字段，若该键不存在则退回整个字典本身。第三步还是判断 `dict`（此时说明拿到的是单个调用对象），把它包成单元素列表 `[payload]`。第四步判断 `not isinstance(payload, list)`，若成立抛出 `TypeError("tool call payload must be an object or list")`。最后用列表推导式对每一项调用 `ToolCall.model_validate(item, strict=True)`，其中 `strict=True` 表示禁止 Pydantic 的隐式类型强转（例如不会把字符串 `"1"` 自动变成整数），保证进入系统的字段类型与声明完全一致。
- **异常/边界**：负载是字符串且无法解码时，`loads_model_json` 会抛出 `json.JSONDecodeError`（该函数已把 `ValueError` 一并收敛到这个类型），本函数不捕获，直接向上冒泡。负载类型不是 `str`/`dict`/`list` 时抛 `TypeError`。列表中的某一项不符合 `ToolCall` 的字段约束时，`model_validate` 会抛 Pydantic 的 `ValidationError`，整批解析失败，本函数不做部分成功或跳过坏项的处理。空字符串输入会走 `loads_model_json`，由于所有候选都为空而抛出 `json.JSONDecodeError`。空列表与 `{"tool_calls": []}` 属于正常边界，返回空列表。该函数不做长度上限校验，调用 id 的长度约束由 `ToolCall` 模型自身负责。
- **同文件关系**：调用了 `loads_model_json`。被本文件里的函数调用的情况为「无」，它是面向外部调用方的顶层入口，没有内部调用者。

### `parse_openai_tool_calls(tool_calls: list[Any], registry: ToolRegistry, name_map: dict[str, str] | None = None, registrations: Mapping[str, tuple[BaseTool, int]] | None = None) -> list[ToolCall]` （第 27 行）
- **作用**：这是第二个公开入口，专门处理「OpenAI 兼容网关返回的原生工具调用数组」，并把它们升级成内部使用的版本化信封。与 `parse_tool_calls` 的关键区别在于，它必须绑定工具注册表：解析出来的每一个调用都要带上该工具当前规范名、`schema_version`、`schema_hash` 和 `registry_generation`，这样后续执行时就能检测「模型是基于旧版 schema 生成的参数」这种漂移风险。它还承担了网关兼容性的脏活：注释明确指出部分 OpenAI 兼容网关会省略调用 id，此时不能整批失败，而是按位置生成稳定的兜底 id（`native-call-1`、`native-call-2`……），并说明 ReAct 循环对合成调用用的是同一套策略，因此两侧 id 形态保持一致。`name_map` 参数提供了别名到规范名的映射能力，用来吸收模型爱用旧名、缩写名的问题。`registrations` 参数允许调用方传入一份预先取好的注册表快照，避免在批量解析过程中注册表被并发修改导致版本号不一致，这对需要「同一批次内代数必须统一」的场景很重要。
- **参数**：
  - `tool_calls`：必填，类型标注为 `list[Any]`，实际接受 `list` 或 `tuple`。每一项可以是 `Mapping`（通常是字典）或任意带属性的对象，两种形态都由 `_get` 统一访问。传入其它类型抛 `TypeError`。
  - `registry`：必填，`ToolRegistry` 实例。当 `registrations` 为 `None` 时，用它来解析工具；它必须提供 `resolve(name) -> (tool, generation)` 方法。即使传了 `registrations`，这个参数仍然是必填位置参数，不会被忽略为空。
  - `name_map`：可选，`dict[str, str] | None`，默认 `None`。键是模型可能输出的原始名，值是注册表里的规范名。默认 `None` 时通过 `(name_map or {})` 退化成空字典，即不做任何重命名。
  - `registrations`：可选，`Mapping[str, tuple[BaseTool, int]] | None`，默认 `None`。键是规范工具名，值是「工具对象 + 注册表代数」的二元组。传入后本函数不再调用 `registry.resolve`，而是直接从这份映射里查表；查不到时同样按未知工具处理。它必须是支持 `[]` 下标访问的映射（若传入普通字典以外的 Mapping，需保证 `KeyError` 语义一致）。
- **返回**：返回 `list[ToolCall]`，长度与输入 `tool_calls` 一致，顺序保持原样。每个元素的 `call_id` 来自输入（或按位置生成的兜底值）、`tool_name` 是重命名后的规范名、`schema_version` 与 `schema_hash` 取自工具的 `spec`、`registry_generation` 取自解析时得到的代数、`arguments` 是解析后的字典。
- **内部流程**：第一步用 `isinstance(tool_calls, (list, tuple))` 做形态校验，不通过就抛 `TypeError("native tool calls must be a list")`。第二步初始化空结果列表 `parsed`，用 `enumerate` 带位置下标遍历输入。第三步对每一项先用 `_get` 依次取出 `id`、`function`、`name`、`arguments`，其中 `arguments` 的默认值是字符串 `"{}"`，保证缺失时退化为空参数对象。第四步做 id 校验：若 `call_id` 不是字符串或去空白后为空，则按 `f"native-call-{position + 1}"` 生成兜底 id（位置从 1 开始计数，与 ReAct 合成调用一致）；随后若 id 长度超过 128 抛 `ValueError`。第五步校验 `name`：不是非空字符串就抛 `ValueError("native tool call is missing id or function name")`。第六步做名称规范化：`canonical_name = (name_map or {}).get(name, name)`。第七步查工具与代数：`registrations is None` 时调 `registry.resolve(canonical_name)`，否则执行 `registrations[canonical_name]`，两者都解包成 `(tool, generation)`；`KeyError` 会被捕获并转换成 `ValueError(f"unknown tool '{name}'")`，注意错误消息里用的是模型给的原始名 `name` 而不是规范名，便于定位问题。第八步解析参数：若 `raw_arguments` 是字符串，则调用 `loads_model_json(raw_arguments or "{}")`，并把 `json.JSONDecodeError` 转成 `ValueError(f"invalid JSON arguments for tool '{name}'")`；若不是字符串（例如网关已经给了字典），则原样使用。第九步校验参数必须是 `dict`，否则抛 `TypeError(f"arguments for tool '{name}' must be a JSON object")`。最后构造 `ToolCall` 并追加到 `parsed`，循环结束后整体返回。
- **异常/边界**：`tool_calls` 不是列表/元组抛 `TypeError`；id 超过 128 字符抛 `ValueError`；缺少有效 id（被兜底覆盖，不报错）或缺少有效工具名抛 `ValueError`；工具名不在注册表（含别名映射后仍查不到）抛 `ValueError`，且用 `from exc` 保留了原始 `KeyError` 链；参数字符串不是合法 JSON 抛 `ValueError`；参数解析结果不是对象抛 `TypeError`。空列表输入返回空列表，不报错。`raw_arguments` 为空字符串时通过 `or "{}"` 退化成空字典，视为无参数。传入的 `registrations` 若缺少某个工具键，会走与 `registry.resolve` 失败相同的 `KeyError` 分支，报「未知工具」。本函数不校验参数是否符合工具 schema，只保证它是 JSON 对象，业务级校验留给执行阶段。
- **同文件关系**：调用了 `_get`（逐字段取属性）和 `loads_model_json`（解析参数字符串）。被本文件里的函数调用的情况为「无」，它与 `parse_tool_calls` 并列，是面向外部调用方的顶层入口。

### `_get(value: Any, key: str, default: Any = None) -> Any` （第 81 行）
- **作用**：这是一个极小的私有访问器，存在的唯一理由是抹平「字典」与「对象」两种数据形态的差异。OpenAI 兼容生态里，同样的工具调用有时以字典形式返回（原始 HTTP JSON），有时以 SDK 的对象形式返回（如带 `.id`、`.function` 属性的包装类），如果调用方在业务代码里到处写 `if isinstance(x, dict)` 分支会非常啰嗦。这个函数把这个分支收敛到一处：是 `Mapping` 就按字典取值，否则按属性取值。它被设计成永不抛错的安全访问器，因为默认值兜底后缺失字段会退化成 `None`，由上层决定这是不是错误。它只做一层访问，不做递归、不做类型转换，因此行为完全可预测。由于它足够小，被 `parse_openai_tool_calls` 在每次循环里调用四次（`id`、`function`、`name`、`arguments`），调用开销可以忽略。
- **参数**：
  - `value`：必填，`Any`。待访问的容器，既可以是 `Mapping`（如 `dict`），也可以是任意对象。传入 `None` 时 `isinstance` 判定为假，会走 `getattr(None, key, default)` 并安全返回 `default`。
  - `key`：必填，`str`。要读取的键名或属性名。
  - `default`：可选，`Any`，默认 `None`。字典分支下作为 `Mapping.get` 的兜底值，对象分支下作为 `getattr` 的兜底值。调用方在取 `arguments` 时显式传入了 `"{}"`。
- **返回**：返回 `Any`。容器是 `Mapping` 且存在该键时返回对应值；键不存在时返回 `default`。容器不是 `Mapping` 时返回同名属性的值；属性不存在时返回 `default`。任何情况下都不会因为缺键或缺属性而抛错。
- **内部流程**：第一步 `isinstance(value, Mapping)` 判断；为真则执行 `return value.get(key, default)`。为假则执行 `return getattr(value, key, default)`。整个函数只有这一个二分叉，没有循环、没有异常捕获、没有副作用。
- **异常/边界**：本函数自身不抛异常。需要注意 `getattr` 的三参形式只会吞掉 `AttributeError`：如果对象的同名属性是一个会抛错的 `property`，异常仍会向外冒泡。另外若 `value` 是 `Mapping` 的子类但重写了 `get` 并抛错，异常也会冒泡。传入 `default` 为可变对象时不会复制，多次调用共享同一引用（本文件调用点都传不可变的 `None` 或字符串，因此无隐患）。
- **同文件关系**：不调用本文件里的其它函数。被本文件里的 `parse_openai_tool_calls` 调用，共四处（`id`、`function`、`name`、`arguments`）。

### `loads_model_json(value: str, *, object_only: bool = False) -> Any` （第 87 行）
- **作用**：这是本文件的容错核心，负责把「模型产出的、几乎肯定不规范的 JSON 文本」解码成 Python 对象。它针对真实观测到的三类噪声依次尝试三种策略：原样解析、抽取第一个括号平衡的子串、把 Python 风格单引号字面量修复成 JSON。之所以要按这个顺序，是因为策略越靠后越「激进」，只有前面的严格策略失败时才应该动用。文档字符串还说明了两个关键设计决定：一是通过 `parse_constant=reject_json_constant` 主动拒绝 `NaN` 与 `Infinity`，因为模型偶尔会输出这些非标准常量，而下游做数值计算或序列化时会出问题，必须在入口处拦住；二是把所有尝试的失败都收敛成 `json.JSONDecodeError`，这样调用者只需处理一种异常类型，不必同时捕获 `ValueError`。`object_only` 参数是为 ReAct 动作解析准备的：当模型在一段散文里先提到一个 `[...]` 数组、后面才给出真正的 `Action Input` 对象时，如果允许扫描方括号，就会错误地截取到前面的数组；限制只扫描 `{...}` 就能避免这种遮蔽。该函数是纯函数，不修改入参、不做 IO。
- **参数**：
  - `value`：必填，`str`。待解码的文本，通常是模型的原始输出片段或工具参数原文。函数内部第一步就做 `value.strip()`，因此首尾空白（含换行、缩进）都会被忽略。传入空字符串或纯空白字符串时，所有候选都会因空值被跳过，最终抛 `json.JSONDecodeError`。
  - `object_only`：可选，关键字限定参数（`*` 之后），`bool`，默认 `False`。为 `True` 时只把 `{` 视为候选起点，忽略 `[`；为 `False` 时 `{` 与 `[` 都参与候选，取文本中最靠前的那个作为起点。
- **返回**：返回 `Any`，即 `json.loads` 成功时的结果，可能是 `dict`、`list`、`str`、`int`、`float`、`bool` 或 `None`，取决于文本内容。注意它并不保证返回字典——除非调用方传 `object_only=True`，否则一个纯数字字符串 `"1"` 或数组字符串 `"[1,2]"` 也会被成功解析并返回非字典结果，字典约束由上层（如 `parse_openai_tool_calls` 的 `isinstance(arguments, dict)`）负责。
- **内部流程**：第一步 `text = value.strip()` 去空白。第二步调用 `_balanced_json_substring(text, object_only=object_only)` 得到 `balanced`，可能为空字符串。第三步构造候选列表 `[text, balanced, _repair_single_quoted_object(balanced or text)]`，注意第三个候选是「优先对平衡子串做单引号修复，平衡子串为空时退回对全文做修复」。第四步初始化空字符串 `reason` 用于累积最后一次失败原因。第五步遍历候选：空候选直接 `continue` 跳过；非空候选调用 `json.loads(candidate, parse_constant=reject_json_constant)` 并直接 `return` 结果，一旦抛出 `json.JSONDecodeError` 或 `ValueError`（后者覆盖 `reject_json_constant` 抛出的 `ValueError`）就把 `str(exc)` 记入 `reason` 并继续下一个候选。第六步所有候选都失败时，用 `detail = f": {reason}" if reason else ""` 拼接原因，抛出 `json.JSONDecodeError(f"invalid JSON arguments{detail}", value, 0)`，其中 `value` 是未去空白的原始输入，位置参数固定为 `0`。
- **异常/边界**：全部候选失败时抛 `json.JSONDecodeError`，消息里带上最后一次失败原因以便排查。`NaN`/`Infinity` 被 `reject_json_constant` 拒绝，表现为 `ValueError`，随后被本函数捕获，最终同样以 `json.JSONDecodeError` 形式对外暴露。空字符串、纯空白、纯散文且不含任何括号的输入都会走到最终的 `JSONDecodeError`。含不平衡括号（例如 `{"a": 1`）时 `_balanced_json_substring` 返回空串，会退回对全文和全文修复版尝试，通常仍然失败并抛错。含单引号但语义不是字典（例如 `'abc'`）时，`_repair_single_quoted_object` 因 `isinstance(value, dict)` 不成立而返回空串，候选被跳过。注意 `json.loads` 在极深嵌套时可能抛 `RecursionError`，本函数未捕获该类型。候选去重没有做，最坏情况下同一个文本会被解析两次，属于可接受的性能代价。
- **同文件关系**：调用了 `_balanced_json_substring`、`_repair_single_quoted_object`、`reject_json_constant`。被本文件里的 `parse_tool_calls`（解码字符串负载）与 `parse_openai_tool_calls`（解码参数字符串）调用。

### `_balanced_json_substring(text: str, *, object_only: bool = False) -> str` （第 117 行）
- **作用**：这个私有函数负责从一段混杂文本中「切出第一个括号平衡的 JSON 块」。模型经常不老实：它会在 JSON 前后加上「好的，我来调用工具：」这样的说明文字，或者用 Markdown 代码围栏包住，直接 `json.loads` 整段必然失败。本函数通过手写的括号深度扫描，从第一个 `{` 或 `[` 开始，一直走到深度重新归零的位置，把这段子串返回，从而把说明文字剥掉。它同时处理了字符串字面量与转义，避免把字符串里的 `}` 或 `{` 误当成结构括号——这是简单计数法最常见的错误来源。`object_only` 参数把起点候选限制为 `{`，用于 ReAct 场景下防止前面出现的数组块遮蔽后面真正的对象。扫描是单趟线性的，遇到第一个完整块立即返回，不做嵌套块收集。它不校验括号内容是否是合法 JSON，只保证括号配平，因此返回值仍可能被 `json.loads` 拒绝，这是刻意的职责分离。
- **参数**：
  - `text`：必填，`str`。已由调用方 `strip` 过的文本（本函数自身不做去空白）。空字符串时两个 `find` 都返回 `-1`，直接返回空串。
  - `object_only`：可选，关键字限定参数，`bool`，默认 `False`。为 `True` 时 `starts` 元组只包含 `text.find("{")`；为 `False` 时包含 `{` 与 `[` 两个查找结果。
- **返回**：返回 `str`。成功时返回从起点字符到配平结束字符（含两端）的切片 `text[start : index + 1]`。找不到任何起点，或扫描到文本结束仍未配平时，返回空字符串 `""`（而不是抛错），把「没找到」与「解析失败」的区分交给调用方。
- **内部流程**：第一步构造 `starts` 元组（依 `object_only` 决定含一项还是两项）。第二步用生成器 `(index for index in starts if index != -1)` 过滤掉未找到的 `-1`，取 `min(...)`，`default=-1`；这一步保证在 `{` 与 `[` 都出现时选位置最靠前的那个作为起点。第三步若 `start == -1` 直接返回空串。第四步初始化三个状态变量：`depth = 0`、`in_string = False`、`escaped = False`。第五步用 `for index in range(start, len(text))` 逐字符扫描。第六步在每个字符上先判断 `in_string`：若在字符串内，遇到 `escaped` 为真则把 `escaped` 复位；否则遇到反斜杠把 `escaped` 置真；否则遇到双引号把 `in_string` 置假；随后 `continue` 跳过结构判断——这一段保证字符串里的括号和转义序列不参与计数。第七步不在字符串内时：遇到 `"` 置 `in_string = True`；遇到 `{` 或 `[` 让 `depth += 1`；遇到 `}` 或 `]` 让 `depth -= 1` 并检查是否归零，归零则立即返回切片。第八步循环自然结束（说明括号未配平）时返回空串。
- **异常/边界**：本函数不抛异常，所有失败路径都返回空串。`text` 为空或没有任何括号时返回空串。括号不平衡（如 `{"a": 1`）返回空串。字符串未闭合（如 `{"a": "abc`）会让 `in_string` 一直为真直到文本结束，循环走完返回空串。注意它只识别双引号字符串，单引号在 JSON 里非法但在 Python 风格字面量里常见，此时单引号内的括号会被误判为结构括号并参与计数——这个缺陷由后续的 `_repair_single_quoted_object` 兜底（它会退回对全文做修复）。另外混用括号（如 `{]`）也会让 `depth` 归零并返回一个语法非法的切片，本函数不做类型配对校验。
- **同文件关系**：不调用本文件里的其它函数。被本文件里的 `loads_model_json` 调用（用于生成第二个候选子串）。

### `_repair_single_quoted_object(candidate: str) -> str` （第 148 行）
- **作用**：这个私有函数处理模型最常见的一类格式错误：把 JSON 对象写成 Python 风格的 `{'key': 1}`，即用单引号做字符串定界符、并可能带 `True`/`False`/`None` 这类 Python 字面量。JSON 解析器会直接拒绝这种文本，而模型（尤其是被 Python 代码污染的提示词）经常这么写，所以需要一个修复层。它的做法不是自己写正则替换（那很容易在嵌套引号上出错），而是直接用 Python 标准库的 `ast.literal_eval` 把这个字面量求值成真正的 Python 对象，再用 `json.dumps` 序列化成合法 JSON 文本。这个「先求值再重编码」的策略既正确又简单，天然支持嵌套字典、列表、元组（会被转成 JSON 数组）与 Python 常量。它被刻意限制成只处理字典：非字典结果一律返回空串，因为工具参数必须是对象，返回数组或标量对上层没有意义。它是纯函数，不做 IO，也不会执行任意代码——`ast.literal_eval` 只允许字面量节点。
- **参数**：
  - `candidate`：必填，`str`。待修复的文本，通常是 `loads_model_json` 里的平衡子串，或该子串为空时的原始全文。空字符串或不含任何单引号的文本会被快速短路返回空串，避免无谓的求值开销。
- **返回**：返回 `str`。成功时返回 `json.dumps(value, ensure_ascii=False)` 生成的 JSON 文本（`ensure_ascii=False` 保证中文等非 ASCII 字符原样保留，不会被转成 `\uXXXX` 转义）。输入为空、不含单引号、`literal_eval` 失败、求值结果不是 `dict`、或 `json.dumps` 失败时，统一返回空字符串 `""`。
- **内部流程**：第一步检查 `not candidate or "'" not in candidate`，任一成立即返回空串（这是廉价的前置过滤，因为不含单引号时 JSON 解析器本就能处理，无需修复）。第二步在 `try` 中执行 `value = ast.literal_eval(candidate)`，捕获 `ValueError`、`SyntaxError`、`MemoryError`、`RecursionError` 四类异常并返回空串——前两类对应语法非法，后两类对应超长或超深输入。第三步判断 `isinstance(value, dict)`，为假返回空串。第四步在 `try` 中执行 `json.dumps(value, ensure_ascii=False)` 并返回结果，捕获 `TypeError` 与 `ValueError`（例如字典值含不可序列化对象，或存在循环引用）后返回空串。
- **异常/边界**：本函数不向外抛异常，所有失败都降级为空串，由 `loads_model_json` 把它当作「候选无效」跳过。特别注意：`ast.literal_eval` 会拒绝函数调用、名称引用、运算表达式等非字面量节点，因此 `{'a': foo()}` 这类文本会因 `ValueError` 返回空串。元组会被 `json.dumps` 转成数组，`True`/`False`/`None` 会被转成 `true`/`false`/`null`，属于预期内的语义变化。字典键若是非字符串（如 `{1: 'a'}`），`json.dumps` 会把整数键转成字符串 `"1"`，这是 Python 标准行为。含单引号的合法 JSON（例如 `{"a": "it's"}`）也会进入求值流程，但 `ast.literal_eval` 通常会因双引号包裹而成功，结果等价，不会造成破坏。
- **同文件关系**：不调用本文件里的其它函数。被本文件里的 `loads_model_json` 调用（用于生成第三个候选文本）。

### `reject_json_constant(value: str) -> None` （第 165 行）
- **作用**：这是一个回调钩子，专门交给 `json.loads` 的 `parse_constant` 参数使用。Python 的 `json` 模块默认接受 `NaN`、`Infinity`、`-Infinity` 这三个非标准常量（它们不属于 JSON 规范，只是 Python 的扩展），而模型有时会输出它们。如果放任进入系统，后续的 JSON 序列化、比较、持久化都可能产生不可预期行为（例如 `NaN != NaN` 导致缓存失效判断出错）。这个函数的策略是「一律拒绝」：无论传入哪个常量都抛 `ValueError`，从而让整次解析失败。它的存在让 `loads_model_json` 能够统一错误类型：`ValueError` 被那里的 `except` 一并捕获，最终统一转成 `json.JSONDecodeError`。函数本身只有一行 `raise`，没有返回值路径，是典型的「哨兵」函数。由于它被 `parse_constant` 调用，只有当输入文本中确实出现这些常量时才会被执行，正常 JSON 完全不受影响。
- **参数**：
  - `value`：必填，`str`。由 `json` 模块传入的被识别出的常量文本，取值只可能是 `"NaN"`、`"Infinity"` 或 `"-Infinity"`。本函数不检查其具体内容，因为三个值都要拒绝。
- **返回**：无正常返回值（注解为 `None`）。它永远以抛异常结束，因此调用方永远拿不到返回结果。
- **内部流程**：唯一一步是 `raise ValueError(f"invalid JSON constant: {value}")`，把触发拒绝的常量名拼进消息里，便于日志定位。没有条件分支、没有循环、没有状态。
- **异常/边界**：无条件抛出 `ValueError`，消息格式为 `invalid JSON constant: <常量名>`。该异常在 `loads_model_json` 内部被 `except (json.JSONDecodeError, ValueError)` 捕获，因此不会直接泄漏给最终调用者；但如果有人把这个函数当作独立的校验工具直接调用，就会立刻收到 `ValueError`。它不处理空字符串或其它输入，因为 `json` 模块不会用其它值调用它。
- **同文件关系**：不调用本文件里的其它函数。被本文件里的 `loads_model_json` 以 `parse_constant=reject_json_constant` 的形式间接调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `parse_tool_calls` | 把字符串、字典或列表形态的工具调用负载归一化并用严格模式校验成 `ToolCall` 列表，全程不执行任何工具。 |
| `parse_openai_tool_calls` | 解析 OpenAI 兼容的原生 `tool_calls` 数组，补齐别名映射、兜底调用 id 与注册表版本元信息，输出内部版本化信封。 |
| `_get` | 统一以「是 Mapping 就取键、否则取属性」的方式安全读取字段，缺键缺属性时返回默认值而不报错。 |
| `loads_model_json` | 依次尝试原文、括号平衡子串、单引号字面量修复三种策略解码模型 JSON，拒绝 `NaN`/`Infinity` 并统一抛 `JSONDecodeError`。 |
| `_balanced_json_substring` | 从混杂散文中扫描出第一个括号平衡的 `{...}` 或 `[...]` 子串，并用字符串/转义状态机避免误判括号。 |
| `_repair_single_quoted_object` | 用 `ast.literal_eval` 把 Python 风格单引号字典字面量求值后重新 `json.dumps` 成合法 JSON，非字典一律返回空串。 |
| `reject_json_constant` | 作为 `json.loads` 的 `parse_constant` 钩子，无条件抛出 `ValueError` 以拒绝 `NaN`/`Infinity` 等非标准常量。 |
