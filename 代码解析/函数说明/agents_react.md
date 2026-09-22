# agents/react.py

## 一、这个文件是干什么的

本文件实现了项目里的「文本协议版 Agent」，也就是 `ReActAgent`：它继承自 `agents/agent.py` 里的 `Agent`，但把「工具调用」的沟通方式从 OpenAI 原生的 function-calling 换成了纯文本的 ReAct 协议（`Thought` / `Action` / `Action Input` / `Observation` / `Final Answer`，并兼容中文的 `思考` / `行动` / `行动输入` / `观察` / `最终答案`）。

它存在的意义是：当接入的模型或本地模型只会返回文本、不支持 OpenAI 的 `tool_calls` 字段时，主运行时仍然能驱动工具。文件本身**不重复实现任何工具**，工具注册表、schema 校验、副作用确认、超时、输出校验全部委托给父类的 `execute_tool_calls`，本文件只负责「把模型文本解析成一次调用」以及「把执行结果拼回文本 Observation」。

文件内容大致分三层：

1. **解析层（模块级函数）**：一组正则表达式常量加 `parse_react_response`、`_extract_thought`、`_decode_action_input`、`_strip_code_fence` 等，把一段模型输出拆成 thought / action / arguments / final_answer / error，并且永不执行任何工具。
2. **参数修补层**：`_coerce_string_scalars` 与 `_declared_schema_types` 会在调用前把「被模型写成字符串的数字/布尔」按工具声明的 schema 转回正确类型，避免严格校验直接拒绝整次调用。
3. **主循环与运行时适配层**：`ReActAgent` 类持有惰性工具加载（`lazy_tools`）、目录工具优先（catalog-first）、热加载工具尾块（hot zone）、提示词前缀缓存（prompt cache）、待确认调用记录等能力，核心方法是 `run_with_react`（异步主循环）与 `run`（同步外壳）。

运行中被用到的路径是：Web 应用或命令行调用 `ReActAgent.run(...)` 或 `await agent.run_with_react(...)`；主循环每轮调用一次模型，解析文本，要么返回最终答案并写入历史，要么执行一个工具（`_execute_action`）或一组原生 tool_calls（`_execute_native_calls`），把 `Observation: {...}` 追加进对话后继续下一轮，直到达到 `max_rounds` 抛错。

另外，本文件通过 `_run_sync` 解决了一个工程问题：FastAPI 请求线程里已经有事件循环，不能直接 `asyncio.run`，于是把协程丢到独立工作线程里跑完。

---

## 二、函数与类逐条详解

### `ParsedReActResponse` （第 101 行）

- **作用**：这是一个 `@dataclass(frozen=True)` 冻结数据类，用来承载「一次模型响应里所有可以安全处理的片段」的解析结果。它把解析和执意拆开：解析器只负责填字段，绝不执行工具，调用方拿到对象后再决定是继续问模型、执行工具还是直接收尾。字段 `arguments` 为 `None` 表示 Action Input 缺失或格式错误；字段 `error` 保存一句可以直接回喂给模型当 Observation 的纠错提示。冻结（`frozen=True`）保证解析结果在循环里传递时不会被意外改写，也让它可以安全地放进缓存或比较。整个 `run_with_react` 的每一轮都以一个 `ParsedReActResponse` 为分支依据。
- **参数**：数据类字段依次为 `raw: str`（原始模型文本，必填，用于日志与调试）；`thought: str | None = None`（思考内容，可为空）；`final_answer: str | None = None`（最终答案文本）；`action: str | None = None`（工具名，可能是点号形式或双下划线形式）；`arguments: dict[str, Any] | None = None`（已解析成字典的调用参数）；`error: str | None = None`（解析层面的错误说明）。
- **返回**：类本身，构造后即为不可变实例；无返回值概念。
- **内部流程**：由 `@dataclass` 在类创建时自动生成 `__init__`、`__repr__`、`__eq__` 等方法；`frozen=True` 额外让属性赋值抛 `FrozenInstanceError`。类体内定义了两个派生属性 `is_final` 与 `has_action`，用字段是否非 `None` 做判断。
- **异常/边界**：构造时不做任何校验，`raw` 传非字符串也不会报错；字段全为默认值时对象仍然合法，只是 `is_final` 与 `has_action` 都为 `False`，主循环会把这种状态当成「没有动作也不是答案」的异常分支处理。
- **同文件关系**：被 `parse_react_response` 反复构造并返回；被 `run_with_react` 消费（读取 `thought`、`is_final`、`has_action`、`error`、`action`、`arguments`）；被 `_execute_action` 作为入参使用（读取 `action`、`arguments`、`error`）。

### `ParsedReActResponse.is_final` （第 117 行）

- **作用**：只读派生属性，回答「这次响应是不是一个最终答案」。判断依据非常直接：`final_answer is not None`。主循环用它来决定是否结束 ReAct 循环并把答案返回给调用方，也用它决定在流式回显模式下是否跳过重复打印。把判断收进属性而不是散落的 `if parsed.final_answer is not None`，能让调用点读起来更像协议语义。
- **参数**：无（`self` 除外）。
- **返回**：`bool`，`final_answer` 非 `None` 时为 `True`，否则 `False`。
- **内部流程**：单表达式 `return self.final_answer is not None`，不做 strip、不做空串判断。
- **异常/边界**：无特殊处理；注意空字符串 `""` 也会让本属性返回 `True`（因为不是 `None`），此时主循环会用 `parsed.final_answer or ""` 兜底成空串返回。
- **同文件关系**：被 `run_with_react` 调用；依赖同一数据类的 `final_answer` 字段。

### `ParsedReActResponse.has_action` （第 121 行）

- **作用**：只读派生属性，回答「这次响应里是否解析出了一个工具名」。主循环用它区分三种情况：有动作就执行工具、没有动作但有 error 就走畸形响应重试、既没有动作也没有 error 就走纯文本兜底当答案。它让「工具名解析失败但动作标记存在」和「压根没有动作标记」这两种语义能被分开处理。
- **参数**：无（`self` 除外）。
- **返回**：`bool`，`action` 非 `None` 时为 `True`。
- **内部流程**：单表达式 `return self.action is not None`。
- **异常/边界**：无特殊处理；`action` 为空字符串的情况在解析阶段就已被拦截并转成 `error`，因此这里不会拿到空串。
- **同文件关系**：被 `run_with_react` 调用（两处分支判断）；依赖同一数据类的 `action` 字段。

### `parse_react_response(text: str) -> ParsedReActResponse` （第 126 行）

- **作用**：整个 ReAct 协议的解析入口，把模型的一段自由文本翻译成结构化的 `ParsedReActResponse`。它按优先级识别多种格式：显式的 `Final Answer:`（含中文别名、加粗写法、全角冒号）、`<answer>...</answer>` XML 包裹、占位符形式的 `<answer>`、`Action:` + `Action Input:` 组合、以及最后的纯文本兜底。它刻意兼容了大量「供应商漂移」：中文标记名、全角冒号、被散文包裹的 JSON、Python 风格的单引号字面量。关键设计是**解析绝不执行工具**，任何不确定的地方都产出 `error` 让模型自我纠正。主循环每一轮都会调用它。
- **参数**：`text: str` —— 模型返回的正文文本。约束：必须是字符串，否则抛 `TypeError`；内容可以为空串（空串会走纯文本兜底，被当成空答案）。
- **返回**：`ParsedReActResponse`。分支对应关系：命中 `Final Answer` → 填 `final_answer`（若冒号后为空则取冒号之后整段文本）；命中 `<answer>` 且内容非空 → 填 `final_answer`；`<answer>` 内容为空 → 只填 `error`；整段只是 `<answer>` / `<final_answer>` / `[answer]` 占位符 → 只填 `error`；没有 `Action` 标记 → 把整段文本当作 `final_answer`；有 `Action` 但工具名为空 → 填 `error`；有 `Action` 但没有 `Action Input` 或输入为空 → 填 `action` 加 `error`；输入 JSON 解析失败 → 填 `action` 加 `error`；全部成功 → 填 `action` 与 `arguments`。
- **内部流程**：第一步做类型检查并保留 `raw = text`；第二步用 `_FINAL_RE` 搜索最终答案、用 `_extract_thought` 抽取思考、用 `_ACTION_RE` 搜索动作名；第三步若 `final_match` 命中就直接返回（冒号后为空时用 `final_match.end()` 截取后续文本）；第四步尝试 `_XML_ANSWER_RE.fullmatch`，命中且非空即返回答案，为空则返回错误；第五步判断是否只是答案占位符；第六步 `action_match is None` 时把整段 strip 后当最终答案返回；第七步对动作名做 `.strip().strip("`").strip()` 去掉反引号包裹，为空则报错；第八步用 `_ACTION_INPUT_RE.search(text, action_match.end())` 只在动作名之后搜索输入段，避免把 Thought 里的内容误当输入；第九步用 `_strip_code_fence` 去掉 ``` 代码围栏，为空则报错；第十步交给 `_decode_action_input` 解码，失败返回错误、成功返回 `arguments`。
- **异常/边界**：`text` 非字符串抛 `TypeError("ReAct response must be a string")`；其余情况一律不抛异常，全部通过 `error` 字段表达，保证主循环可以把错误当成 Observation 回喂。空文本被解释为「空最终答案」而不是错误；`<answer>` 标签内容全空白会被判为非法；动作名只有反引号或空白会被判为空。
- **同文件关系**：调用 `_extract_thought`、`_strip_code_fence`、`_decode_action_input`；使用模块级正则 `_FINAL_RE`、`_XML_ANSWER_RE`、`_ACTION_RE`、`_ACTION_INPUT_RE`；被 `run_with_react` 每轮调用。

### `_extract_thought(text: str) -> str | None` （第 234 行）

- **作用**：从模型文本里抽取 `Thought:` 段落的私有辅助函数。它接受英文 `thought` 以及中文 `思考`、`想法` 三种标记，允许加粗（`**Thought:**`）、允许全角冒号，并用前瞻断言把内容截断在下一个协议标记之前（也就是说，Thought 里可以换行，但不能吞掉后面的 Action）。抽取出的思考文本只用于日志与可观测性，不参与控制流。这是解析器的子步骤，被 `parse_react_response` 调用。
- **参数**：`text: str` —— 要搜索的完整模型文本，不做类型检查（调用方已经保证是字符串）。
- **返回**：`str | None`。找到标记且 strip 后非空 → 返回该文本；找到标记但内容全为空白 → 返回 `None`；完全没找到标记 → 返回 `None`。
- **内部流程**：用 `re.search` 配合内联标志 `(?im)` 与 `re.DOTALL`，模式由「行首 + 可选 `**` + 标记名 + 可选 `**` + 冒号 + 惰性捕获」加前瞻 `(?=^\s*(?:\*\*)?<标记前瞻>(?:\*\*)?\s*[：:]|\Z)` 组成，其中 `<标记前瞻>` 来自模块级常量 `_REACT_MARKER_LOOKAHEAD`。命中后取 `group(1).strip()`，用 `value or None` 把空串归一成 `None`。
- **异常/边界**：无特殊处理，不抛异常；正则整体失败只返回 `None`。多行 Thought 能正确保留，因为用了 `DOTALL` 加前瞻而不是单行匹配。
- **同文件关系**：被 `parse_react_response` 调用；依赖模块级常量 `_REACT_MARKER_LOOKAHEAD`；不调用本文件其它函数。

### `_decode_action_input(payload: str) -> tuple[dict[str, Any] | None, str | None]` （第 249 行）

- **作用**：把 Action Input 的原始文本解码成一个参数字典，是「宽容解析」的集中点。它本身不做复杂的字符串修补，而是委托给项目共享的 `core.loads_model_json(payload, object_only=True)`，后者会依次尝试：原样解析、截取第一个配平的 `{...}` 块、以及把 Python 风格单引号字面量修复成 JSON。之所以必须存在，是因为文本协议模型经常在 JSON 外面裹一层说明文字、或者用单引号，直接 `json.loads` 会失败并浪费一整轮。失败时它返回一句可直接读懂的纠错提示，让模型下一轮改正。
- **参数**：`payload: str` —— 已去掉代码围栏并 strip 过的 Action Input 文本，理论上非空（调用方已过滤空值）。
- **返回**：`tuple[dict[str, Any] | None, str | None]`。成功时返回 `(参数字典, None)`；失败时返回 `(None, 中文/英文纠错提示字符串)`。注意第二个元素是英文长句，明确要求「恰好一个 JSON 对象、双引号键、不要在对象后追加散文」。
- **内部流程**：先用 `try/except (TypeError, ValueError)` 包住 `loads_model_json` 调用，把任何解码异常降级成 `arguments = None`；随后用 `isinstance(arguments, dict)` 判断结果必须是对象（数组、标量、`None` 都算失败）；成功直接返回二元组，失败返回预置的提示文案。
- **异常/边界**：内部吞掉 `TypeError` 与 `ValueError`，对外不抛异常。JSON 数组、数字、字符串、`null` 等「合法 JSON 但不是对象」的情况统一按失败处理并给出纠错提示，这正是 `object_only=True` 的语义延伸。
- **同文件关系**：被 `parse_react_response` 调用；不调用本文件其它函数；依赖外部 `core.parser.loads_model_json`。

### `_strip_code_fence(value: str) -> str` （第 272 行）

- **作用**：去掉模型给 Action Input 套的 Markdown 代码围栏。模型经常写成 ```json ... ``` 或 ``` ... ```，这些反引号会污染 JSON 解码，导致 `loads_model_json` 走修复路径甚至直接失败。函数只处理「首行以 ``` 开头」这一种最常见形态，去掉首行，并且当末行恰好是纯 ``` 时一并去掉末行，其余内容原样保留换行结构。它是解析链里非常小但很关键的一环。
- **参数**：`value: str` —— 待清理文本，通常是 `_ACTION_INPUT_RE` 捕获组 strip 之后的内容。
- **返回**：`str` —— 清理后的文本，始终返回字符串（可能为空串），绝不返回 `None`。
- **内部流程**：先 `value.strip().splitlines()` 切行；若首行 strip 后以 ``` 开头，则丢弃首行；再判断（丢弃后的）末行 strip 后是否恰为 ```，是则丢弃末行；最后用 `"\n".join(lines).strip()` 重新拼接并去掉首尾空白。
- **异常/边界**：无特殊处理，空串输入返回空串（`splitlines()` 得到空列表，`join` 得到空串）；首行不是围栏时完全不动内容，只做首尾 strip。
- **同文件关系**：被 `parse_react_response` 调用；不调用本文件其它函数。

### `_coerce_string_scalars(arguments: dict[str, Any], schema: Mapping[str, Any] | None) -> dict[str, Any]` （第 281 行）

- **作用**：在调用工具之前做一次「schema 驱动的类型纠偏」。文本协议模型最常见的一类参数错误是把数字写成字符串（例如 `{"count": "3"}`），而项目用严格的 Pydantic 校验，这会让整次调用以 `INVALID_ARGUMENTS` 被拒；模型看不到真正原因，往往原样重发，循环白烧。本函数只在工具**自己声明**的类型范围内转换：声明 `integer` 就尝试 `int`，声明 `number` 就尝试 `float`，声明 `boolean` 且字符串是 `true`/`false`（大小写不敏感）就转成布尔。转换失败的值原样保留，交给正常的校验错误路径去报错。它直接修改并返回传入的字典。
- **参数**：`arguments: dict[str, Any]` —— 从解析结果拿到的参数字典（调用方 `_execute_action` 会先 `dict(...)` 拷贝一份）；`schema: Mapping[str, Any] | None` —— 工具声明的输入 JSON Schema，可能为 `None` 或非 Mapping。
- **返回**：`dict[str, Any]` —— 就地修改后的同一个字典对象；若 schema 不可用则原样返回入参。
- **内部流程**：先判断 `schema` 是否为 `Mapping`，再取 `schema["properties"]` 并再次确认是 `Mapping`，任一不满足直接返回原字典；随后遍历 `arguments.items()`，跳过非字符串值，取出该键的 `subschema`（非 Mapping 则跳过），调用 `_declared_schema_types` 得到声明类型集合；把值 strip 后按 `integer` → `int`、`number` → `float`、`boolean` + `true/false` → `bool` 的顺序尝试转换，转换异常用 `continue` 跳过（保持原值），成功则写回 `arguments[key]`。
- **异常/边界**：内部捕获 `ValueError`（`int("abc")` 等）后放弃该键的转换，不抛异常；`integer` 优先于 `number`，所以同时声明两者时按整数处理；空字符串转 `int`/`float` 会失败并保留原值；`boolean` 只认 `true`/`false` 两个字面量，`1`/`0`/`yes` 不转换。
- **同文件关系**：调用 `_declared_schema_types`；被 `_execute_action` 调用（在构造 `ToolCall` 之前）。

### `_declared_schema_types(subschema: Mapping[str, Any]) -> frozenset[str]` （第 329 行）

- **作用**：从单个属性的 JSON Schema 片段里提取出所有声明过的类型名，供 `_coerce_string_scalars` 判断该往哪个方向转换。它同时支持三种常见写法：`"type": "integer"` 单字符串、`"type": ["string", "null"]` 数组/元组、以及 `"anyOf": [{"type": "number"}, ...]` 联合类型。返回 `frozenset` 是为了让成员判断 O(1) 且顺序无关。只识别 `type` 字段，不递归处理嵌套 schema。
- **参数**：`subschema: Mapping[str, Any]` —— 单个属性的 schema 片段（调用方已确认是 Mapping）。
- **返回**：`frozenset[str]` —— 声明到的类型名集合；无法识别时返回空 `frozenset()`。
- **内部流程**：先读 `subschema.get("type")`：是 `str` 就包成单元素 frozenset 返回；是 `list`/`tuple` 就过滤出其中所有 `str` 元素返回；否则再看 `subschema.get("anyOf")`，是 `list` 就遍历其中每个 Mapping 并收集其字符串型 `type`，最后返回收集到的集合；三者都不满足返回空集合。
- **异常/边界**：无特殊处理，不抛异常；`anyOf` 中非 Mapping 元素被静默忽略；`type` 是数字等非法值会被忽略并落到空集合。
- **同文件关系**：被 `_coerce_string_scalars` 调用；不调用本文件其它函数。

### `_run_sync(coro: Any) -> Any` （第 345 行）

- **作用**：把协程在同步上下文里跑到完成。存在的理由是运行环境冲突：FastAPI 的请求线程里通常已经有一个正在运行的 event loop，此时直接调用 `asyncio.run` 会抛 `RuntimeError: asyncio.run() cannot be called from a running event loop`。解决方案是开一个只有 1 个工作线程的 `ThreadPoolExecutor`，把 `asyncio.run` 本身提交进去，在新线程里新建事件循环执行协程，主线程用 `.result()` 阻塞等待结果。工作线程在 `with` 块退出时被回收。这样 `ReActAgent.run` 就能在同步代码里安全地驱动异步主循环。
- **参数**：`coro: Any` —— 任意可被 `asyncio.run` 接受的协程对象；类型标注故意写成 `Any`，实际必须是 coroutine。
- **返回**：`Any` —— 协程的返回值原样透传（本文件里通常是 `str` 形式的最终答案）。
- **内部流程**：`with ThreadPoolExecutor(max_workers=1) as executor:` 创建线程池 → `executor.submit(asyncio.run, coro)` 提交 → `.result()` 阻塞取结果 → 退出 `with` 时 `shutdown` 等待线程结束。
- **异常/边界**：协程内部抛出的异常会被 `.result()` 重新抛出（不吞异常）；传非协程对象会在工作线程里由 `asyncio.run` 抛 `ValueError` 并同样透传；没有超时参数，协程若永不结束则本函数永久阻塞（超时控制由主循环里传下去的 `timeout` 参数在模型调用层负责）。
- **同文件关系**：只被 `ReActAgent.run` 调用；不调用本文件其它函数（它接收的协程来自 `run_with_react`）。

### `ReActAgent` （第 358 行）

- **作用**：本文件的核心类，继承 `agents.agent.Agent`。它在保留父类全部 provider 管理与工具管理能力的同时，把「对话推进」换成文本 ReAct 协议：`run_with_react` 是异步主循环，`run` 是同步外壳。它额外承担三件父类不做或做得不同的事：一是**惰性工具加载**（`lazy_tools=True` 时只把工具契约存进 SQLite 仓库，实现类在首次真正用到时才导入构造）；二是**catalog-first 模式**（先让模型调用目录工具 resolve 出 schema，再调用真实工具）；三是**提示词前缀缓存友好**的提示词组装（冻结清单 `_frozen_manifest` 保持字节稳定，热加载工具单独放在最末尾的 hot-zone 块里）。类上还定义了两段类常量提示词：`REACT_INSTRUCTIONS`（完整协议说明、中英双语示例、JSON 格式要求、以及「工具使用纪律」四条硬性规则）和 `CATALOG_FIRST_REACT_INSTRUCTIONS`（说明清单只是索引、需要先 resolve 契约、resolve 只取契约不等于注册或授权、可用 `limit: 20` 批量解析）。实例属性 `lazy_tools` 与 `pending_confirmations` 由本类引入，后者记录本轮里因缺少确认钥匙而失败的写操作。
- **参数**：无（类定义本身不接收参数）。
- **返回**：类对象；实例化后得到可 `run` / `await run_with_react` 的 Agent。
- **内部流程**：类体先定义两段提示词常量，再定义 `__init__`、工具管理覆写方法、主循环与全部私有辅助方法。主循环内部按「惰性加载 → 快照 → 组装提示词 → 调模型 → 解析 → 执行工具或收尾」的顺序推进。
- **异常/边界**：类本身无异常；实例化时 `__init__` 会校验 `lazy_tools` 类型并在缺仓库时抛 `RuntimeError`。
- **同文件关系**：调用本文件几乎所有函数与方法；被 `_run_sync`（通过 `run`）与本文件外的 Web/CLI 层使用。

### `ReActAgent.__init__(self, name: str, *, llm: Any | None = None, provider_config: str | None = None, provider_registry: Any | None = None, repository: ToolSpecRepository | None = None, auto_discover_tools: bool = True, tool_package: str | Any = "tool", discovery_strict: bool = False, lazy_tools: bool = False) -> None` （第 410 行）

- **作用**：构造 ReAct Agent 实例，默认「急切注册」所有工具；把 `lazy_tools=True` 显式打开后，改为只持久化元数据、把实现类的构造推迟到目录查询或直接调用时。它还要处理一个父类差异：父类 `Agent.__init__` 的 `auto_discover_tools` 会真的把工具实例塞进注册表，而本类必须传 `auto_discover_tools=False` 给父类，然后自己决定是走急切路径（调用 `self.discover_tools`）还是惰性路径（`discover_tools` 里只存元数据）。此外它把 `self.catalog_tool.repository_only = lazy_tools` 设上，保证惰性模式下目录工具始终从 SQLite 仓库搜索，而不是只在「已加载工具」这个小集合里搜。
- **参数**：`name: str`（Agent 名字，透传父类，用于身份与历史键等）；`llm: Any | None = None`（可注入的 LLM 客户端，透传父类）；`provider_config: str | None = None`（provider 配置名/路径，透传父类）；`provider_registry: Any | None = None`（provider 注册表对象，透传父类）；`repository: ToolSpecRepository | None = None`（工具契约仓库；为 `None` 且惰性模式加自动发现时会被自动新建一个）；`auto_discover_tools: bool = True`（是否在构造时就发现工具）；`tool_package: str | Any = "tool"`（工具包名或包对象，默认字符串 `"tool"`）；`discovery_strict: bool = False`（发现阶段的严格模式，透传父类并传给 `discover_tools`）；`lazy_tools: bool = False`（是否惰性加载，必须是布尔）。
- **返回**：`None`（构造函数）。
- **内部流程**：先用 `isinstance(lazy_tools, bool)` 校验，非布尔抛 `TypeError`；若 `repository is None and lazy_tools and auto_discover_tools` 则 `repository = ToolSpecRepository()`；设置 `self.lazy_tools`；初始化 `self.pending_confirmations = []`；调用 `super().__init__(...)`，其中固定传 `auto_discover_tools=False`，其余参数按用户值透传；随后设置 `self.catalog_tool.repository_only = lazy_tools`；最后若 `auto_discover_tools` 为真，调用 `self.discover_tools(strict=discovery_strict)`——注意这里调用的是本类覆写后的版本，所以惰性模式下实际只落库元数据。
- **异常/边界**：`lazy_tools` 非布尔抛 `TypeError("lazy_tools must be a boolean")`；惰性 + 自动发现但既没给仓库又无法新建（理论不会）会在后续 `discover_tools` 里抛 `RuntimeError("lazy tool discovery requires a ToolSpecRepository")`；父类 `__init__` 自身可能抛的异常（如 provider 配置无效）原样上抛。
- **同文件关系**：调用 `self.discover_tools`（本类覆写版）；被外部代码实例化；与 `_ensure_tool_loaded`、`_with_tool_instructions`、`_execute_action` 等共享 `lazy_tools`、`catalog_tool`、`repository` 状态。

### `ReActAgent.discover_tools(self, *, package: str | Any | None = None, replace: bool = False, strict: bool = False, reload_modules: bool = False)` （第 452 行）

- **作用**：发现工具模块并把它们的契约登记下来，但**在惰性模式下不保留任何工具实例**。这是「惰性加载」得以成立的关键：如果发现阶段就把实例留在注册表里，内存与导入副作用都白省了。急切模式下它直接转调父类实现，行为与普通 Agent 完全一致。惰性模式下它建一个临时的 `ToolRegistry` 作为脚手架，让 `discover_tool_modules` 把元数据写进 `self.repository`（`metadata_only=True`），随后把脚手架置 `None` 释放掉，让运行时注册表里只剩目录工具。
- **参数**：`package: str | Any | None = None`（要扫描的包；为 `None` 时用实例上的 `self.tool_package`）；`replace: bool = False`（是否替换已存在的同版本登记）；`strict: bool = False`（严格模式，发现出错时是否直接失败）；`reload_modules: bool = False`（是否强制重新导入模块，用于热加载场景）。全部为关键字参数。
- **返回**：发现报告对象（`report`）——急切模式下来自父类 `discover_tools` 的返回值，惰性模式下是 `discover_tool_modules` 的返回值；同时被写入 `self.tool_discovery_report`（仅惰性路径）。
- **内部流程**：若 `not self.lazy_tools`，调用 `super().discover_tools(...)` 并直接返回其 report；否则先检查 `self.repository is None`，是则抛 `RuntimeError`；计算 `selected_package`；`scratch_registry = ToolRegistry()`；调用 `discover_tool_modules(scratch_registry, package=selected_package, repository=self.repository, replace=..., strict=..., reload_modules=..., metadata_only=True)`；把 `scratch_registry` 置 `None` 释放；写 `self.tool_discovery_report = report`；返回 report。
- **异常/边界**：惰性且无仓库抛 `RuntimeError("lazy tool discovery requires a ToolSpecRepository")`；包不存在、模块导入失败等异常由 `discover_tool_modules` 决定是否吞掉，`strict=True` 时通常会向上抛；`package` 传非法类型由底层处理。
- **同文件关系**：被 `__init__` 调用；急切分支调用父类同名方法；惰性分支调用外部 `discover_tool_modules`（在导入时被别名为 `discover_tool_modules`）；与 `register_tool`、`_ensure_tool_loaded` 共享 `self.repository` 语义。

### `ReActAgent.register_tool(self, tool: BaseTool, *, replace: bool = False) -> None` （第 489 行）

- **作用**：把一个工具登记进 Agent，惰性模式下只持久化它的契约（spec + 实现引用），把实现类的构造推迟到第一次真正被调用或 resolve 到的时候。它通过 `f"{type(tool).__module__}:{type(tool).__qualname__}"` 生成 `implementation_ref`，这正是 `_ensure_tool_loaded` 之后用来重新导入并构造该工具的凭据。急切模式直接转调父类，保持与普通 Agent 相同的即时注册行为。登记后还会写一条工具注册活动日志。
- **参数**：`tool: BaseTool`（工具实例，惰性模式下必须是 `BaseTool` 实例，否则抛 `TypeError`）；`replace: bool = False`（同版本已存在时是否覆盖）。
- **返回**：`None`。
- **内部流程**：若 `not self.lazy_tools`，`return super().register_tool(tool, replace=replace)`；否则先 `isinstance(tool, BaseTool)` 校验；检查 `self.repository is None` 则抛 `RuntimeError`；计算 `implementation_ref`；调用 `self.repository.save(tool.spec, implementation_ref=implementation_ref, replace=replace)`；最后调用 `log_tool_registration(tool.spec.name, None, self.repository.active_tool_names())` 记日志。
- **异常/边界**：`tool` 不是 `BaseTool` 抛 `TypeError("tool must be a BaseTool instance")`；无仓库抛 `RuntimeError("lazy tool registration requires a ToolSpecRepository")`；仓库里已存在同版本且 `replace=False` 时由 `repository.save` 决定行为（通常报冲突）。
- **同文件关系**：急切分支调用父类；惰性分支与 `_ensure_tool_loaded`（读取 `implementation_ref`）配对使用；被外部注册代码调用。

### `ReActAgent.is_tool_registered(self, name: str, *, version: str | None = None, schema_hash: str | None = None) -> bool` （第 510 行）

- **作用**：查询某个工具是否已登记，惰性模式下要同时照顾两个来源：运行时注册表（已加载的实现）与 SQLite 仓库（只有契约）。因为目录工具本身永远活在注册表里，所以它的名字直接走父类判断；其它名字先去仓库查，再按可选 `version`/`schema_hash` 做精确匹配。这让上层「注册状态检查」在惰性模式下不会误报未注册。
- **参数**：`name: str`（工具名，点号形式）；`version: str | None = None`（限定版本，`None` 表示不限）；`schema_hash: str | None = None`（限定 schema 哈希，`None` 表示不校验）。
- **返回**：`bool` —— 急切模式返回父类结果；惰性模式下目录工具返回父类结果；仓库无记录或仓库不可用时返回 `False`；仓库有记录且（未指定 `schema_hash` 或哈希一致）时返回 `True`。
- **内部流程**：`not self.lazy_tools` → 直接 `super().is_tool_registered(...)`；`name == self.catalog_tool.spec.name` → 同样走父类；`self.repository is None` → `False`；否则 `stored = self.repository.get(name, version)`，返回 `stored is not None and (schema_hash is None or stored["schema_hash"] == schema_hash)`。
- **异常/边界**：无特殊处理，不抛异常；`stored` 记录里缺 `schema_hash` 键会抛 `KeyError`（依赖仓库返回结构固定）；`name` 传非字符串由仓库层决定行为。
- **同文件关系**：调用父类同名方法；与 `tool_registration_status`、`register_tool`、`_ensure_tool_loaded` 共享仓库视图；被外部或父类逻辑调用。

### `ReActAgent.tool_registration_status(self, name: str) -> dict[str, Any]` （第 532 行）

- **作用**：返回某个工具的详细登记状态字典，用于自检、审计与前端展示。它比 `is_tool_registered` 提供更多信息：版本、schema 哈希、注册表 generation（已加载才有，未加载为 `None`）以及实现引用字符串。惰性模式下若仓库里没有该名字，会退回父类实现（通常给出「未注册」的结论），保证行为在两种模式下语义一致。
- **参数**：`name: str`（工具名）。
- **返回**：`dict[str, Any]`。惰性模式且仓库命中时返回形如 `{"name", "registered": True, "version", "schema_hash", "generation", "implementation"}` 的字典，其中 `generation` 仅当 `name in self.tools` 时取注册表的真实值，否则为 `None`，`implementation` 为仓库里的 `implementation_ref`。其它情况返回父类的结果。
- **内部流程**：`not self.lazy_tools or name == self.catalog_tool.spec.name` → 父类；`self.repository is None` → 父类；`stored = self.repository.get(name)` 为 `None` → 父类；否则构造并返回上述字典，其中 generation 用条件表达式 `self.tools.registration_status(name)["generation"] if name in self.tools else None` 计算。
- **异常/边界**：无特殊处理，不抛异常；若 `name in self.tools` 但注册表返回结构缺 `generation` 键会抛 `KeyError`（依赖内部结构稳定）。
- **同文件关系**：调用父类同名方法；与 `is_tool_registered`、`register_tool`、`_ensure_tool_loaded` 共享仓库与注册表状态。

### `ReActAgent.tool_confirmation_key(self, name: str) -> str` （第 553 行）

- **作用**：在不加载工具实现的前提下，算出一个工具的「确认钥匙」字符串。写操作类工具在执行前需要人工确认，确认钥匙把工具名、版本、schema 哈希和注册表 generation 绑在一起，避免用户确认了旧版本契约后被换成新实现。惰性模式下工具可能还没加载进注册表，此时用固定的 generation `1` 占位（因为未加载的工具首次加载时 generation 就是 1），拼出 `name@version#schema_hash:1`；已加载的工具则交给注册表给出精确的 generation 绑定钥匙。这样前端可以在工具尚未实例化时就展示并校验确认钥匙。
- **参数**：`name: str`（工具名）。
- **返回**：`str` —— 确认钥匙。已加载时来自 `self.tools.confirmation_key(name)`；未加载时形如 `f"{name}@{stored['version']}#{stored['schema_hash']}:1"`。
- **内部流程**：先判断 `name in self.tools`，命中直接返回注册表钥匙；否则检查 `self.repository is None` 抛 `KeyError`；再 `stored = self.repository.get(name)`，为 `None` 抛 `KeyError`；最后按模板拼接返回。
- **异常/边界**：工具既不在注册表也不在仓库时抛 `KeyError(f"tool '{name}' is not registered")`；仓库无 `version`/`schema_hash` 键会抛 `KeyError`。
- **同文件关系**：与 `_record_pending_confirmation`（记录确认失败的调用）在语义上配套；被外部确认流程调用。

### `ReActAgent.run(self, query: str, **kwargs: Any) -> str` （第 570 行）

- **作用**：同步便捷入口。给调用方一个「像普通函数一样调用 Agent」的接口，内部把异步主循环 `run_with_react` 交给 `_run_sync` 在独立线程的事件循环里跑完。这样同步代码（或已经在事件循环里的 FastAPI 请求线程）也能安全使用 ReAct Agent，而不必自己处理 event loop 冲突。所有额外关键字参数（`max_rounds`、`model`、`tool_names`、`stream_echo` 等）原样透传给 `run_with_react`。
- **参数**：`query: str`（用户问题，必须是非空字符串，否则抛 `ValueError`）；`**kwargs: Any`（透传给 `run_with_react` 的全部关键字参数）。
- **返回**：`str` —— 模型给出的最终答案文本。
- **内部流程**：先做 `isinstance(query, str)` 与 `query.strip()` 非空校验，失败抛 `ValueError("query must be a non-empty string")`；然后 `return _run_sync(self.run_with_react(query, **kwargs))`。
- **异常/边界**：`query` 非字符串或全空白抛 `ValueError`；`run_with_react` 内部的异常（`TypeError`、`ValueError`、`RuntimeError`）经 `_run_sync` 原样上抛；kwargs 里出现 `run_with_react` 不认识的键会抛 `TypeError`。
- **同文件关系**：调用 `_run_sync` 与 `run_with_react`；被外部同步调用方使用。

### `ReActAgent.run_with_react(self, messages: str | Sequence[Mapping[str, Any]], context: ExecutionContext | None = None, *, max_rounds: int | None = None, model: str | None = None, temperature: float = DEFAULT_TEMPERATURE, timeout: float = DEFAULT_TIMEOUT, tool_names: list[str] | None = None, profile_name: str | None = None, provider_name: str | None = None, use_history: bool = True, defer_tool_loading: bool = True, prompt_cache_key: str | None = None, prompt_cache_retention: str | None = None, enable_prompt_cache: bool = True, stream_echo: bool = False) -> str` （第 577 行）

- **作用**：ReAct 的异步主循环，也是本文件最长、最核心的方法。它把「用户输入 → 组装系统提示词 → 调模型 → 解析文本 → 执行工具 → 追加 Observation → 再调模型」这条链路跑起来，直到模型给出最终答案或轮次耗尽。它同时负责：把输入归一成消息列表、校验所有布尔/数值参数、初始化历史对话、在惰性模式下按需加载工具实现、把热加载工具拼成末尾 hot-zone 块、按需附加提示词缓存键、兼容原生 `tool_calls` 回退路径、对「未标记答案」和「畸形答案」做有限次重试、以及把最终对话写回历史。
- **参数**：`messages: str | Sequence[Mapping[str, Any]]`（用户问题字符串，或与 `Agent.run_with_tools` 相同的 role/content 映射序列）；`context: ExecutionContext | None = None`（执行上下文，`None` 时新建一个；非 `ExecutionContext` 抛 `TypeError`）；`max_rounds: int | None = None`（最大轮次，`None` 或正整数，布尔/非整数/小于 1 抛 `ValueError`）；`model: str | None = None`（模型名覆盖）；`temperature: float = DEFAULT_TEMPERATURE`（采样温度，来自 `constants`）；`timeout: float = DEFAULT_TIMEOUT`（单次模型调用超时，来自 `constants`）；`tool_names: list[str] | None = None`（本次请求允许的工具白名单，必须是非空字符串列表，含未知名抛 `ValueError`）；`profile_name: str | None = None`（provider profile 名）；`provider_name: str | None = None`（provider 名）；`use_history: bool = True`（是否把该 profile 的历史对话作为前缀）；`defer_tool_loading: bool = True`（是否推迟工具加载，默认只先暴露目录工具）；`prompt_cache_key: str | None = None`（显式缓存键，经 `_validate_prompt_cache_key` 校验）；`prompt_cache_retention: str | None = None`（缓存保留策略，经 `_validate_prompt_cache_retention` 校验）；`enable_prompt_cache: bool = True`（是否启用前缀缓存）；`stream_echo: bool = False`（是否把最终答案流式回显到终端）。
- **返回**：`str` —— 最终答案文本（`parsed.final_answer or ""`，或纯文本兜底的 `content.strip()`）。
- **内部流程**：① 输入归一：字符串 → `[{"role": "user", "content": messages}]`，序列 → 逐个 `dict(...)` 拷贝（`bytes`/`bytearray` 被排除），其它类型抛 `TypeError`；非 Mapping 元素抛 `TypeError`。② 校验 `max_rounds`；清空 `self.pending_confirmations`；准备/校验 `context`；校验 `use_history`、`defer_tool_loading`、`enable_prompt_cache`、`stream_echo` 均为布尔；校验两个缓存参数；`echo_mode = "react_final" if stream_echo else None`。③ 通过 `_completion_target` 取 `completion_llm`/`selected_model`/`history_key`，通过 `_initial_conversation` 得到 `conversation`。④ `initial_snapshot = self.tools.snapshot()`，调用 `self._sync_frozen_manifest()` 冻结前缀清单，初始化 `loaded_tool_schemas = {}` 与 `visible_order`。⑤ 决定 `loaded_order`/`requested_names`：给了 `tool_names` 就去重、剔除已存在名、校验未知名字后按用户顺序；否则若 `defer_tool_loading or self.lazy_tools` 则只放目录工具；否则放全部可见工具。⑥ 按需计算 `cache_key`。⑦ 创建 `ToolLoop(max_rounds, safety_limit=ToolLoop.DEFAULT_SAFETY_LIMIT)`，初始化两个重试计数器。⑧ 进入 `for round_number in round_loop.rounds()`：记轮次日志；惰性模式下对 `loaded_order` 里未加载的名字逐个 `_ensure_tool_loaded`；重新快照并把符合白名单的 `_hot_tools` 追加进 `loaded_order`；构造 `registrations`；按「目录工具排最前」的稳定顺序把 schema 收进 `loaded_tool_schemas`；调用 `_with_tool_instructions` 组装本轮对话；若有热加载工具则把 `_hot_zone_lines` 生成的系统块**追加到最后一条消息之后**；组装 `options`（model/temperature/timeout/stream=False，可选缓存键与保留策略）；`await asyncio.to_thread(self._dispatch_model_call, ...)` 调模型。⑨ 归一响应：字符串响应视为纯文本；否则用 `_response_message` 取 message，读 `content` 与 `tool_calls`，构造 `assistant_message`。⑩ 若有原生 `tool_calls`：逐个 `round_loop.record_call`，追加 assistant 消息，`await self._execute_native_calls(...)` 拿 observations 并 extend，惰性模式下再 `_load_catalog_observation` 并补热工具，然后 `continue`。⑪ 文本路径：把 `content` 归一成字符串，`parsed = parse_react_response(content)`，追加 assistant 消息；非最终答案或非流式时记 `log_react_thought`。⑫ 若 `parsed.is_final`：先判断是否为「未标记答案」且本请求需要工具证据（`_should_require_tool_action`），需要则按 `REACT_UNMARKED_ANSWER_RETRY_LIMIT` 重试（追加 `Observation: <protocol_error>` 后 continue），超限则记日志并接受答案；随后记最终答案日志、必要时补打印（`stream_echo` 且响应未流式回显过）、`self._save_history(...)`、返回答案。⑬ 若 `parsed.error` 且无 action：`malformed_answer_retries` 加一，未超 `REACT_MALFORMED_ANSWER_RETRY_LIMIT` 则追加 Observation 后 continue，超限抛 `RuntimeError`（附错误与 `content[:200]`）。⑭ 若既无 action 又无 error（防御分支）：记日志、必要时补打印、存历史、返回 `content.strip()`。⑮ 否则执行动作：`round_loop.record_call(parsed.action, parsed.arguments or {})`，`await self._execute_action(...)`，把 `Observation: {_result_json(result)}` 作为 user 消息追加；惰性模式下 `_load_catalog_result` 并补热工具。⑯ 循环耗尽后 `raise RuntimeError("maximum ReAct rounds exceeded")`。
- **异常/边界**：`TypeError` 覆盖输入类型、`context` 类型、布尔参数类型；`ValueError` 覆盖空 query、非法 `max_rounds`、非法 `tool_names`、未知工具名；`RuntimeError` 覆盖畸形答案重试超限与轮次耗尽；模型调用层的超时由传入的 `timeout` 参数控制，异常经 `asyncio.to_thread` 上抛；`parse_react_response` 对非字符串会抛 `TypeError`，但此处已先把 `content` 强制成字符串，因此不会触发。
- **同文件关系**：调用 `parse_react_response`、`_should_require_tool_action`、`_initial_conversation`、`_with_tool_instructions`、`_hot_zone_lines`、`_ensure_tool_loaded`、`_load_catalog_observation`、`_load_catalog_result`、`_execute_action`、`_execute_native_calls`、`_response_message`、`_record_pending_confirmation`（间接）以及父类的 `_completion_target`、`_sync_frozen_manifest`、`_dispatch_model_call`、`_save_history`、`_hot_zone_lines` 之外的 `_openai_tool_name` 等；被 `run` 与本文件外的异步调用方调用。

### `ReActAgent._should_require_tool_action(request_messages: Sequence[Mapping[str, Any]], registrations: Mapping[str, tuple[Any, int]]) -> bool` （第 918 行）

- **作用**：`@staticmethod`，决定「一个没有 `Final Answer:` 标记的纯文本回答是否应该被打回重试」。它刻意保守：只有当请求文本里出现常见的实时/检索/工具意图词时才要求模型必须先用工具；普通的闲聊仍然允许直接走纯文本兜底回答，避免把简单对话也逼成工具调用。这个判断是「未标记答案重试」分支的闸门，也是防止模型绕过工具编造事实的一道软约束。
- **参数**：`request_messages: Sequence[Mapping[str, Any]]`（本次请求的原始消息序列，函数只挑 `role == "user"` 的 `content` 拼成一段文本）；`registrations: Mapping[str, tuple[Any, int]]`（本轮可用工具映射，为空则直接判定不需要工具）。
- **返回**：`bool` —— `registrations` 为空返回 `False`；否则把 user 消息内容拼接、`casefold()` 小写化后，只要包含任一标记词就返回 `True`，都不包含返回 `False`。
- **内部流程**：先 `if not registrations: return False`；再用生成式把 `role == "user"` 的 `message.get("content", "")` 转成字符串并以空格连接，整体 `casefold()`；最后对标记元组做 `any(marker in text for marker in (...))`，标记包括中文的 `现在`、`当前时间`、`今天`、`实时`、`查询`、`搜索`、`查找`、`检索`、`运势`、`天气`，以及英文的 `search`、`look up`、`current time`、`latest`、`today`。
- **异常/边界**：无特殊处理，不抛异常；非 Mapping 的 message 会在 `.get` 处抛 `AttributeError`（依赖调用方保证类型）；`content` 为 `None` 时 `str(None)` 得到 `"none"`，通常不会误命中标记词。
- **同文件关系**：被 `run_with_react` 调用（未标记答案分支）；不调用本文件其它函数。

### `ReActAgent._initial_conversation(self, request_messages: list[dict[str, Any]], *, use_history: bool, history_key: str) -> list[dict[str, Any]]` （第 958 行）

- **作用**：把「历史/系统前缀」和「本次请求消息」拼成本轮对话的初始列表。启用历史且该 profile 已有历史时，用历史消息的浅拷贝作为前缀（保证不把外部列表对象直接接进会被 append 的对话里）；否则退回 `_configured_prompt_messages()` 提供的配置化系统提示词。这个区分让多轮对话能延续上下文，又让首次对话有正确的系统人设。
- **参数**：`request_messages: list[dict[str, Any]]`（本轮请求消息，已由 `run_with_react` 归一化）；`use_history: bool`（关键字参数，是否使用历史）；`history_key: str`（关键字参数，用于在 `self._profile_histories` 里查历史的键）。
- **返回**：`list[dict[str, Any]]` —— `prefix + request_messages` 的新列表；前缀元素都是新字典（历史路径）或配置函数返回的字典。
- **内部流程**：判断 `use_history and self._profile_histories.get(history_key)` 是否为真；为真则 `prefix = [dict(item) for item in self._profile_histories[history_key]]`；否则 `prefix = self._configured_prompt_messages()`；最后返回拼接结果。
- **异常/边界**：无特殊处理；历史里若有非 Mapping 元素，`dict(item)` 会抛 `TypeError`/`ValueError`；`history_key` 不存在只是拿不到历史，不会报错。
- **同文件关系**：被 `run_with_react` 调用；依赖父类属性 `_profile_histories` 与方法 `_configured_prompt_messages`；其结果随后被 `_with_tool_instructions` 加工。

### `ReActAgent._with_tool_instructions(self, conversation: list[dict[str, Any]], registrations: Mapping[str, tuple[Any, int]], loaded_tool_schemas: Mapping[str, Mapping[str, Any]] | None = None, *, catalog_first: bool = False, visible_names: set[str] | None = None) -> list[dict[str, Any]]` （第 971 行）

- **作用**：组装本轮发给模型的系统指令，并在最前面插入一条 `system` 消息。这条系统消息包含：ReAct 协议说明（`REACT_INSTRUCTIONS`）、冻结的注册清单（`All registered tool names`）、可选的 catalog-first 说明、一句「只有下面 Available tools 里的工具现在可调用」的范围声明、以及每个可用工具的**完整契约**（用途/使用规范/逐变量类型必填默认约束说明/输出字段/副作用与确认）。设计上极其讲究前缀缓存：注册清单来自 `_frozen_manifest` 且排序稳定，工具条目按名字排序渲染，热加载工具被排除在这条缓存前缀之外，只把「已 resolve 但未注册」的 schema 放在后面。这样同一 epoch 内这段文本逐字节不变，provider 的前缀缓存才能命中。
- **参数**：`conversation: list[dict[str, Any]]`（已有对话，被放在 system 消息之后）；`registrations: Mapping[str, tuple[Any, int]]`（工具名 → (工具实例, generation)）；`loaded_tool_schemas: Mapping[str, Mapping[str, Any]] | None = None`（已加载工具的 schema 缓存，用于渲染那些不在 registrations 里但已 resolve 的工具）；`catalog_first: bool = False`（关键字参数，是否附加 catalog-first 提示）；`visible_names: set[str] | None = None`（关键字参数，白名单，用于过滤冻结清单的展示范围）。
- **返回**：`list[dict[str, Any]]` —— `[{"role": "system", "content": 拼接文本}, *conversation]`，即一条新的 system 消息加原有对话。
- **内部流程**：① 惰性模式下先把 `registrations` 过滤成只留目录工具（防止把尚未真正可用的工具写进提示词）。② 调用 `self._sync_frozen_manifest()` 保证冻结清单就绪。③ `frozen_names` 取冻结清单中符合 `visible_names` 的名字并排序，拼成逗号分隔的 `inventory`（空则 `"(none)"`）。④ `hot_names = set(self._hot_tools)`。⑤ `lines` 依次放入 `REACT_INSTRUCTIONS`、空行、`"All registered tool names: " + inventory`、空行。⑥ `catalog_first` 为真时追加 `CATALOG_FIRST_REACT_INSTRUCTIONS` 与空行。⑦ 追加「只有 Available tools 里的工具现在可调用」的范围声明与空行。⑧ 追加 `"Available tools:"`；把 `registrations` 中非热加载的项放进 `prefix_registrations`；若为空则追加 `"(No tools are currently available; answer directly.)"`，否则对 `sorted(prefix_registrations)` 逐个 `render_tool_entry(name, tool.spec)` 展开成完整契约行。⑨ 计算 `resolved_schemas`（`loaded_tool_schemas` 里不在 `registrations` 中的项），非空则追加 `"Loaded tool input schemas:"` 并按名排序用 `render_schema_block` 渲染。⑩ 把 `lines` 用 `"\n"` 连接成 `instruction` 字典并返回 `[instruction, *conversation]`。
- **异常/边界**：无特殊处理，不抛异常（依赖 `render_tool_entry`/`render_schema_block` 对合法 spec 的处理）；`registrations` 里若混入非二元组会解包报错（依赖调用方结构正确）；`loaded_tool_schemas` 为 `None` 时按空处理。
- **同文件关系**：被 `run_with_react` 每轮调用；调用父类 `_sync_frozen_manifest`；与 `_hot_zone_lines` 分工（本方法只渲染缓存前缀，热工具交给它）；使用模块导入的 `render_tool_entry`、`render_schema_block` 与类常量 `REACT_INSTRUCTIONS`、`CATALOG_FIRST_REACT_INSTRUCTIONS`。

### `ReActAgent._hot_zone_lines(self, loaded_tool_schemas: Mapping[str, Mapping[str, Any]], *, allowed_names: set[str] | None = None) -> list[str]` （第 1052 行）

- **作用**：渲染「热加载工具区」的行文本。热加载工具（对话进行中被新注册进来的工具）不能被写进上面那条被缓存的前缀系统消息，否则前缀字节一变，provider 的缓存全废；所以它们单独生成一个系统块，由 `run_with_react` 追加在整个对话的**最后**——那里既是模型注意力最强的地方，也是缓存里天然可变的位置。为了控制长度，最近重载的最多 4 个工具保留完整契约，其余热工具降级成一行名单提示。实现已被替换掉（从注册表消失）的工具仍然按名字列出，只是标注实现暂不可用。
- **参数**：`loaded_tool_schemas: Mapping[str, Mapping[str, Any]]`（已加载 schema 缓存，本方法签名保留该参数以统一调用形态，函数体不直接读取它）；`allowed_names: set[str] | None = None`（关键字参数，白名单过滤；`None` 表示不限制）。
- **返回**：`list[str]` —— 行列表。`self._hot_tools` 为空返回 `[]`；过滤后名单为空返回 `[]`；否则返回以说明行开头的多行文本（完整契约行、roster 行、以及实现不可用行）。
- **内部流程**：① `if not self._hot_tools: return []`。② `max_full = 4`。③ `roster` 取 `_hot_tools` 中符合 `allowed_names` 的名字并排序，空则返回 `[]`。④ `hot_snapshot = self.tools.snapshot()`。⑤ `renderable = [name for name in roster if name in hot_snapshot]`。⑥ `lines` 首行是 `"Hot-loaded tools (newly registered during this conversation, usable immediately):"`。⑦ `full_names = set(renderable[-max_full:])` 取列表最后 4 个（即最近重载的）。⑧ 遍历 `renderable`：在 `full_names` 里的用 `render_tool_entry(name, tool.spec)` 展开完整契约，否则追加 `f"- {name}: (see schema above or via catalog)"`。⑨ 对 `set(roster) - set(renderable)`（实现已消失的）排序后追加 `f"- {name}: (implementation temporarily unavailable)"`。⑩ 返回 `lines`。
- **异常/边界**：无特殊处理，不抛异常；`hot_snapshot[name]` 解包依赖快照结构为二元组；`loaded_tool_schemas` 参数虽被传入但未使用，属于兼容保留。
- **同文件关系**：被 `run_with_react` 调用（构造 hot_block）；调用父类属性 `_hot_tools` 与 `self.tools.snapshot()`；使用模块导入的 `render_tool_entry`；与 `_with_tool_instructions` 形成「缓存前缀 + 可变尾块」的分工。

### `ReActAgent._ensure_tool_loaded(self, name: str) -> None` （第 1100 行）

- **作用**：惰性加载的执行者——按名字把仓库里的契约还原成一个真正可执行的工具实例并注册进运行时注册表。它用登记时保存的 `implementation_ref`（`模块名:限定名`）动态导入模块、逐级 `getattr` 定位到类，并且优先使用模块里可选的 `create_tool()` 工厂函数（存在且可调用就用它），否则直接实例化目标类。加载后做一次严格的**契约一致性校验**：新实例的 `spec.name`、`version`、`schema_hash` 必须与仓库记录完全一致，否则抛错——这是防止磁盘上的实现与已缓存的 schema 脱节，从而避免模型按旧契约传参。校验通过才 `self.tools.register(tool)`。
- **参数**：`name: str`（工具名，点号形式；调用方已保证是仓库里的键）。
- **返回**：`None`。
- **内部流程**：① `if name in self.tools: return`（已加载直接短路）。② `self.repository is None` → 抛 `RuntimeError("lazy tool loading requires a ToolSpecRepository")`。③ `stored = self.repository.get(name)`，为 `None` 直接 `return`（静默忽略，不报错）。④ 取 `implementation_ref`，非字符串或不含 `":"` → 抛 `RuntimeError`。⑤ `module_name, qualname = implementation_ref.split(":", 1)`，任一为空 → 抛 `RuntimeError`。⑥ `importlib.import_module(module_name)`；⑦ 对 `qualname.split(".")` 逐级 `getattr` 得到 `target`。⑧ `factory = getattr(module, "create_tool", None)`；`tool = factory() if callable(factory) else target()`。⑨ `isinstance(tool, BaseTool)` 不成立 → 抛 `TypeError("create_tool() must return a BaseTool instance")`。⑩ 比对 `tool.spec.name != stored["tool_name"]`、`version`、`schema_hash` 任一项不符 → 抛 `RuntimeError`。⑪ `self.tools.register(tool)`。
- **异常/边界**：无仓库抛 `RuntimeError`；`implementation_ref` 缺失/畸形抛 `RuntimeError`；模块导入失败、属性不存在抛 `ImportError`/`AttributeError`；构造器抛异常原样上抛；返回类型不符抛 `TypeError`；契约不匹配抛 `RuntimeError`；仓库无记录时**静默返回**（不抛错），把「未注册」的判定留给调用方。
- **同文件关系**：被 `run_with_react`（惰性模式下预加载 `loaded_order`）、`_execute_action`（模型直接点名一个惰性工具时按需加载）、`_execute_native_calls`（原生调用路径同理）调用；与 `register_tool`（写入 `implementation_ref`）、`discover_tools`（写入元数据）配套；使用模块导入的 `importlib` 与 `BaseTool`。

### `ReActAgent._load_catalog_result(self, result: ToolResult, loaded_order: list[str], context: ExecutionContext, loaded_tool_schemas: dict[str, dict[str, Any]] | None = None) -> None` （第 1136 行）

- **作用**：解析目录工具（catalog）成功返回的结果，把其中携带的工具 schema 排进「后续要展示/加载」的队列，并把这些 schema 记进 `loaded_tool_schemas` 以便下一轮提示词里的 `Loaded tool input schemas:` 段落能渲染出来。它是 catalog-first 模式的关键一环：模型先 resolve 到契约，本方法把契约接进上下文，模型下一步才能真正调用工具。为了兼容旧版目录响应，它同时支持 `specs` 列表和单个 `spec` 字段两种形态。
- **参数**：`result: ToolResult`（一次工具执行结果）；`loaded_order: list[str]`（本轮的加载/展示顺序列表，本方法会就地 append）；`context: ExecutionContext`（执行上下文，签名保留用于统一调用形态，函数体未直接使用）；`loaded_tool_schemas: dict[str, dict[str, Any]] | None = None`（schema 缓存，命中时写入）。
- **返回**：`None`（就地修改 `loaded_order` 与 `loaded_tool_schemas`）。
- **内部流程**：① `if not result.ok or result.tool_name != self.catalog_tool.spec.name: return`（只处理成功的目录调用）。② `data = result.data if isinstance(result.data, Mapping) else {}`。③ `raw_specs = data.get("specs")`，非 list 则置 `[]`。④ `raw_spec = data.get("spec")`，若是 Mapping 且不在 `raw_specs` 中则 `insert(0, raw_spec)`（旧格式兼容）。⑤ 遍历 `raw_specs`：跳过非 Mapping；取 `tool_name`，非非空字符串则跳过；若 `self.repository is not None` 则 `stored = self.repository.get(tool_name, candidate.get("version"))`，为 `None` 则跳过（仓库里没有的契约不接进来）；若给了 `loaded_tool_schemas` 且 `candidate["input_schema"]` 是 Mapping，则 `loaded_tool_schemas[tool_name] = dict(input_schema)`；最后若 `tool_name not in loaded_order` 则 append。
- **异常/边界**：无特殊处理，不抛异常；`result.data` 不是 Mapping、`specs` 不是列表、`input_schema` 不是 Mapping 等情况全部静默跳过；`context` 参数未使用；重复的 tool_name 不会重复 append。
- **同文件关系**：被 `run_with_react` 与 `_load_catalog_observation` 调用；依赖 `self.catalog_tool` 与 `self.repository`；与 `_load_catalog_observation`、`_execute_action` 共同支撑 catalog-first 流程。

### `ReActAgent._load_catalog_observation(self, observation: Mapping[str, Any], loaded_order: list[str], context: ExecutionContext, loaded_tool_schemas: dict[str, dict[str, Any]] | None = None) -> None` （第 1173 行）

- **作用**：原生 `tool_calls` 回退路径上的适配器。当模型走的是 OpenAI function-calling（而不是文本协议）并调用了目录工具时，执行结果已经被序列化成一条 `{"role": "user", "content": "Observation: {...}"}` 消息；本方法把这条消息里的 JSON 反序列化回 `ToolResult`，再交给 `_load_catalog_result` 处理，从而让两条路径共享同一套「契约入上下文」逻辑。它只处理以 `"Observation: "` 开头的字符串内容，其它形状一律忽略。
- **参数**：`observation: Mapping[str, Any]`（一条 Observation 消息字典）；`loaded_order: list[str]`（加载顺序列表，透传）；`context: ExecutionContext`（执行上下文，透传）；`loaded_tool_schemas: dict[str, dict[str, Any]] | None = None`（schema 缓存，透传）。
- **返回**：`None`。
- **内部流程**：① `content = observation.get("content")`；非字符串或不以 `"Observation: "` 开头直接 `return`。② `payload = json.loads(content[len("Observation: "):])`。③ `result = ToolResult.model_validate(payload, strict=True)`。④ 调 `self._load_catalog_result(result, loaded_order, context, loaded_tool_schemas)`。②③ 被 `try/except (TypeError, ValueError, json.JSONDecodeError)` 包住，任何异常都直接 `return`。
- **异常/边界**：JSON 非法、结构不符合 `ToolResult`、类型错误全部被捕获并静默忽略（不抛异常）；`observation` 非 Mapping 会在 `.get` 处抛 `AttributeError`（依赖调用方保证类型）。
- **同文件关系**：调用 `_load_catalog_result`；被 `run_with_react` 在原生 tool_calls 分支调用；使用模块导入的 `json` 与 `ToolResult`。

### `ReActAgent._unavailable_tool_error(self, action_name: str, call_id: str, *, requested_names: set[str] | None = None) -> ToolResult` （第 1194 行）

- **作用**：为「工具已注册但本次请求不能调用」构造一个语义准确的错误结果。它区分两种成因并给出完全不同的纠错话术：一种是调用方传了显式 `tool_names` 白名单把该工具排除（例如非联网模式摘掉 `web.search`），此时返回 `TOOL_NOT_ENABLED` 并明确要求「不要再重试、也不要通过目录去 resolve」；另一种是延迟/惰性加载下 schema 还没加载，此时返回 `TOOL_SCHEMA_REQUIRED` 并指导模型先调用目录工具 resolve。这个区分非常重要——如果把白名单外的情况也说成「去 resolve」，模型会陷入无解的重复循环。
- **参数**：`action_name: str`（动作名，点号形式）；`call_id: str`（本次调用的 id，用于回填结果）；`requested_names: set[str] | None = None`（关键字参数，白名单；`None` 表示没有白名单约束）。
- **返回**：`ToolResult` —— `ok=False`，`error.code` 为 `TOOL_NOT_ENABLED` 或 `TOOL_SCHEMA_REQUIRED`，`tool_name` 经过 `_safe_tool_name` 脱敏，`message` 为对应的指导文案（`TOOL_SCHEMA_REQUIRED` 分支里会插入 `self.catalog_tool.spec.name` 作为要调用的目录工具名）。
- **内部流程**：① `safe_name = _safe_tool_name(action_name)`。② 若 `requested_names is not None and action_name not in requested_names` → 返回 `TOOL_NOT_ENABLED` 结果，文案说明该工具本次被禁用、不要重试也不要 resolve、只用 Available tools 里的工具。③ 否则返回 `TOOL_SCHEMA_REQUIRED` 结果，文案指导先调用目录工具并传 `action 'resolve'` 来加载 schema。
- **异常/边界**：无特殊处理，不抛异常；`self.catalog_tool.spec.name` 若缺失会抛 `AttributeError`（依赖实例状态正常）。
- **同文件关系**：被 `_execute_action`（白名单拦截、schema 未加载两种情况）与 `_execute_native_calls`（白名单拦截、解析失败但工具未加载两种情况）调用；使用父类导入的 `_safe_tool_name`。

### `ReActAgent._response_message(response: Any) -> Any` （第 1241 行）

- **作用**：`@staticmethod`，从原始 LLM 响应对象里取出第一条 choice 的 message。它把「响应格式不符合 OpenAI 约定」这类问题集中成两个明确的 `RuntimeError`，避免后续代码在 `None` 上做属性访问时报出难以理解的错误。主循环用它把 provider 返回值归一成 message，再读 `content` 与 `tool_calls`。
- **参数**：`response: Any`（provider 返回的响应对象，期望具备 `choices[0].message` 结构）。
- **返回**：`Any` —— 第一条 choice 的 message 对象。
- **内部流程**：① `choices = _field(response, "choices")`。② `if not choices: raise RuntimeError("LLM response contained no choices")`。③ `message = _field(choices[0], "message")`。④ `if message is None: raise RuntimeError("LLM response contained no message")`。⑤ 返回 `message`。
- **异常/边界**：`choices` 为空/`None` 抛 `RuntimeError`；`message` 为 `None` 抛 `RuntimeError`；`_field` 的取值方式决定了对象属性和字典键两种形态都能被读到。
- **同文件关系**：被 `run_with_react` 调用（仅在响应不是字符串时）；使用父类导入的 `_field`。

### `ReActAgent._execute_action(self, parsed: ParsedReActResponse, current_snapshot: Mapping[str, tuple[Any, int]], registrations: Mapping[str, tuple[Any, int]], context: ExecutionContext, *, call_number: int, loaded_tool_schemas: dict[str, dict[str, Any]] | None = None, requested_names: set[str] | None = None) -> ToolResult` （第 1251 行）

- **作用**：把一次文本协议解析出来的动作真正执行掉，是「解析层」与「执行层」的接缝。它要依次穿过好几道关卡：把模型写的工具名规范化（点号 ↔ 双下划线）、检查白名单、检查是否在快照里（不在就尝试惰性加载）、检查是否在可用 registrations 里、检查解析阶段是否带 error，全部通过后才构造 `ToolCall` 并调用父类的 `execute_tool_calls`。参数在构造调用前会经过 `_coerce_string_scalars` 纠偏。执行前后会记工具调用开始/结束日志并统计耗时，结果里若是 `CONFIRMATION_REQUIRED` 会被记入 `pending_confirmations`。
- **参数**：`parsed: ParsedReActResponse`（解析结果，必须带 `action`，函数开头用 `assert` 断言）；`current_snapshot: Mapping[str, tuple[Any, int]]`（本轮注册表快照，惰性加载成功后会被替换成刷新后的快照）；`registrations: Mapping[str, tuple[Any, int]]`（本轮允许调用的工具映射，惰性加载成功后会补入新项）；`context: ExecutionContext`（执行上下文，透传给 `execute_tool_calls`）；`call_number: int`（关键字参数，轮次号，用于生成 `call_id` 与日志）；`loaded_tool_schemas: dict[str, dict[str, Any]] | None = None`（关键字参数，加载成功后写入新工具的 schema）；`requested_names: set[str] | None = None`（关键字参数，白名单）。
- **返回**：`ToolResult` —— 各种前置失败会返回 `ok=False` 的结果（`TOOL_NOT_ENABLED`、`UNKNOWN_TOOL`、`TOOL_SCHEMA_REQUIRED`、`INVALID_TOOL_CALL`）；正常执行时返回 `execute_tool_calls` 批次的第一个结果。
- **内部流程**：① `assert parsed.action is not None`。② `action_name = self._canonical_action_name(parsed.action, current_snapshot)`；`call_id = f"react-call-{call_number}"`；`request_name = action_name.replace("__", ".")`。③ 若 `requested_names` 非空且 `request_name` 不在其中 → 返回 `_unavailable_tool_error(...)`。④ 若 `action_name not in current_snapshot`：把双下划线还原成点号得到 `lazy_name`；在 `self.lazy_tools` 且有仓库时 `repository.get(lazy_name)`，命中则 `_ensure_tool_loaded(lazy_name)`，再 `refreshed = self.tools.snapshot()`，若加载成功就替换 `current_snapshot`、把新工具塞进 `registrations`、把 `action_name` 改成 `lazy_name`、并写入 `loaded_tool_schemas`。⑤ 若仍不在快照 → 返回 `UNKNOWN_TOOL`（消息用原始 `parsed.action`）。⑥ 若 `action_name not in registrations` → 返回 `_unavailable_tool_error(...)`（区分禁用与未加载）。⑦ 若 `parsed.error is not None` → 返回 `INVALID_TOOL_CALL`，消息就是解析错误。⑧ 取 `tool, generation = registrations[action_name]`；`arguments = _coerce_string_scalars(dict(parsed.arguments or {}), tool.spec.input_schema)`。⑨ 构造 `ToolCall(call_id, tool_name, schema_version=tool.spec.version, schema_hash=tool.spec.schema_hash, registry_generation=generation, arguments)`。⑩ `log_tool_call_started(call_number, (action_name,))`，`tool_started_at = time.perf_counter()`。⑪ `try: batch = await self.execute_tool_calls([call], context)` / `finally: log_tool_call_completed(call_number, (action_name,), time.perf_counter() - tool_started_at)`。⑫ `result = batch.results[0]`；`self._record_pending_confirmation(call, result)`；返回 result。
- **异常/边界**：`parsed.action` 为 `None` 触发 `AssertionError`（正常调用链不会发生）；惰性加载过程中的 `RuntimeError`/`ImportError`/`TypeError` 会向上抛；`execute_tool_calls` 内部异常经 `finally` 记完日志后上抛；`batch.results` 为空会抛 `IndexError`（依赖执行管理器保证一一对应）。
- **同文件关系**：调用 `_canonical_action_name`、`_unavailable_tool_error`、`_ensure_tool_loaded`、`_coerce_string_scalars`、`_record_pending_confirmation`；被 `run_with_react` 调用；使用父类 `execute_tool_calls`；使用模块导入的 `ToolCall`、`time`、`log_tool_call_started`、`log_tool_call_completed`。

### `ReActAgent._record_pending_confirmation(self, call: ToolCall, result: ToolResult) -> None` （第 1346 行）

- **作用**：当一次工具执行因为「需要人工确认」而失败时，把这次调用的工具名和参数记进实例上的 `pending_confirmations` 列表，供上层（Web 界面/CLI）在循环结束后展示「有哪些写操作在等用户点头」。之所以要在 Agent 里留这个痕迹，是因为 ReAct 主循环只会把错误文本回喂给模型，模型可能会放弃这个动作，但用户仍然需要知道有这么一步被卡住了。去重逻辑保证同一个工具 + 同一份参数只记一次，避免模型重试时列表膨胀。
- **参数**：`call: ToolCall`（实际发出的调用，提供 `tool_name` 与 `arguments`）；`result: ToolResult`（该调用的执行结果，用于判断错误码）。
- **返回**：`None`（就地修改 `self.pending_confirmations`）。
- **内部流程**：① `if result.error is None or result.error.code != "CONFIRMATION_REQUIRED": return`。② `pending = {"tool_name": call.tool_name, "arguments": dict(call.arguments)}`（参数做浅拷贝，避免外部后续修改影响记录）。③ `if pending not in self.pending_confirmations: self.pending_confirmations.append(pending)`。
- **异常/边界**：无特殊处理，不抛异常；`call.arguments` 若为 `None` 会让 `dict(None)` 抛 `TypeError`（依赖 `ToolCall` 保证参数字典存在）；只有错误码恰为 `CONFIRMATION_REQUIRED` 才记录，其它错误（含权限、超时）一律忽略。
- **同文件关系**：被 `_execute_action` 与 `_execute_native_calls` 调用；读取由 `run_with_react` 在每轮开始时清空的 `self.pending_confirmations`（清空逻辑在 `run_with_react` 内，不在本方法）。

### `ReActAgent._execute_native_calls(self, native_calls: Sequence[Any], current_snapshot: Mapping[str, tuple[Any, int]], registrations: Mapping[str, tuple[Any, int]], context: ExecutionContext, *, round_number: int, loaded_tool_schemas: dict[str, dict[str, Any]] | None = None, requested_names: set[str] | None = None) -> list[dict[str, Any]]` （第 1353 行）

- **作用**：原生 function-calling 的兼容回退路径。虽然本文件主打文本 ReAct，但有些 provider 在被要求文本协议时仍会返回 `tool_calls`；与其报错，不如按父类 `Agent.run_with_tools` 的完全相同解析与执行路径把它跑掉。它先把 provider 用的双下划线名字规范化、按需惰性加载仓库里的实现，然后用 `parse_openai_tool_calls` 把原生调用转成 `ToolCall`，一次性批量执行，最后为每个原生调用生成一条 `Observation: {...}` 消息（保持与文本路径一致的上下文形态）。执行失败的调用会保留错误结果而不是抛异常，让模型下一轮可以自我纠正。
- **参数**：`native_calls: Sequence[Any]`（原生 tool_calls 列表）；`current_snapshot: Mapping[str, tuple[Any, int]]`（注册表快照，函数内先 `dict(...)` 拷贝一份可变副本）；`registrations: Mapping[str, tuple[Any, int]]`（可用工具映射，同样拷贝）；`context: ExecutionContext`（执行上下文）；`round_number: int`（关键字参数，轮次号，用于日志与兜底 call_id）；`loaded_tool_schemas: dict[str, dict[str, Any]] | None = None`（关键字参数，惰性加载后写入 schema）；`requested_names: set[str] | None = None`（关键字参数，白名单）。
- **返回**：`list[dict[str, Any]]` —— 与 `native_calls` 等长、顺序一致的 observation 消息列表，每条形如 `{"role": "user", "content": "Observation: <结果 JSON>"}`。
- **内部流程**：① 把 `current_snapshot` 与 `registrations` 转成可变字典。② 预加载循环：对每个 `native_call` 取 `function.name`，空名跳过；`_canonical_action_name` 规范化；已在快照里则跳过；`lazy_name = canonical.replace("__", ".")`；若白名单存在且不含 `lazy_name` 则跳过；非惰性或无仓库跳过；仓库无该名跳过；否则 `_ensure_tool_loaded(lazy_name)`，刷新 `current_snapshot`，命中则补进 `registrations` 与 `loaded_tool_schemas`。③ 构造 `name_map`（OpenAI 名字 → 注册名）与 `aliases`（OpenAI 名字 → 快照名）。④ 初始化 `calls`、`positions`、`results`。⑤ 逐位置解析：取 `call_id`（空则用 `f"react-native-call-{position+1}"`）；取 `provider_name`（默认 `"unknown.tool"`）；`canonical = aliases.get(provider_name, provider_name)`；`request_name` 为点号化结果；白名单不含则记 `_unavailable_tool_error` 并 continue；`parse_openai_tool_calls([native_call], self.tools, name_map, registrations)[0]` 解析；解析异常时若该工具在快照里但不在 registrations 里则记 `_unavailable_tool_error`，否则记 `INVALID_TOOL_CALL`（消息经 `_safe_tool_call_error(exc)` 脱敏）；解析成功则 append 到 `calls`/`positions`。⑥ 若 `calls` 非空：取 `tool_names`，记开始日志与起始时间，`await self.execute_tool_calls(calls, context)`，`finally` 里记完成日志与耗时；随后用 `zip(positions, batch.results, calls, strict=True)` 把结果写回 `results` 并逐个 `_record_pending_confirmation`。⑦ 最后按 `native_calls` 原顺序为每个位置生成 observation（结果取自 `results[position]`，经 `_result_json` 序列化）并返回。
- **异常/边界**：解析失败与白名单拦截都被转成 `ToolResult` 错误，不抛异常；`execute_tool_calls` 的异常经 `finally` 记日志后上抛；`zip(..., strict=True)` 在长度不匹配时抛 `ValueError`；`results[position]` 若因某分支遗漏未写入会抛 `KeyError`（代码已保证每条路径都写）。
- **同文件关系**：调用 `_canonical_action_name`、`_ensure_tool_loaded`、`_unavailable_tool_error`、`_record_pending_confirmation`；被 `run_with_react` 在原生 tool_calls 分支调用；使用父类 `_openai_tool_name`、`execute_tool_calls` 与模块导入的 `parse_openai_tool_calls`、`_field`、`_safe_tool_call_error`、`_safe_tool_name`、`_result_json`、`log_tool_call_started`、`log_tool_call_completed`、`time`。

### `ReActAgent._canonical_action_name(self, action: str, snapshot: Mapping[str, tuple[Any, int]]) -> str` （第 1479 行）

- **作用**：把模型给出的工具名映射回注册表里的规范名字。模型可能写点号形式（`system.current_time`），而注册表里存的是双下划线形式（`system__current_time`），或者反过来；本方法先直接匹配快照键，匹配不上就遍历快照，用父类的 `_openai_tool_name` 把每个注册名转成 provider 形态再比对。全都匹配不上时原样返回，让调用方走「未知工具」的错误路径而不是在这里静默改名。
- **参数**：`action: str`（模型给出的工具名）；`snapshot: Mapping[str, tuple[Any, int]]`（当前注册表快照，键是规范名）。
- **返回**：`str` —— 命中则返回快照里的规范名；否则返回传入的 `action` 原值。
- **内部流程**：① `if action in snapshot: return action`。② 遍历 `snapshot` 的键，若 `self._openai_tool_name(name) == action` 则返回该 `name`。③ 循环结束返回 `action`。
- **异常/边界**：无特殊处理，不抛异常；遍历顺序即字典插入顺序，理论上若有两个注册名映射到同一 provider 名会返回先遇到的那个（依赖注册表保证名字唯一）。
- **同文件关系**：被 `_execute_action`（规范化文本动作名）与 `_execute_native_calls`（预加载与解析阶段各用一次）调用；使用父类 `_openai_tool_name`。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `ParsedReActResponse` | 冻结数据类，承载一次模型响应解析出的 thought / final_answer / action / arguments / error。 |
| `ParsedReActResponse.is_final` | 只读属性，判断 `final_answer` 是否非 `None`，决定循环是否收尾。 |
| `ParsedReActResponse.has_action` | 只读属性，判断 `action` 是否非 `None`，区分「有动作」与「纯文本/畸形」。 |
| `parse_react_response` | ReAct 文本协议的总解析入口，把模型自由文本拆成结构化结果且绝不执行工具。 |
| `_extract_thought` | 抽取 `Thought:`（含中文别名、加粗、全角冒号）内容，仅用于日志。 |
| `_decode_action_input` | 用共享的宽容 JSON 解码器把 Action Input 变成参数字典，失败给出纠错提示。 |
| `_strip_code_fence` | 去掉 Action Input 外层可能包裹的 Markdown ``` 代码围栏。 |
| `_coerce_string_scalars` | 按工具 schema 把被写成字符串的数字/布尔转回声明类型，避免严格校验误拒。 |
| `_declared_schema_types` | 从单个属性 schema 中提取声明的类型名集合，支持 type 字符串、数组与 anyOf。 |
| `_run_sync` | 在独立工作线程里跑 `asyncio.run`，让同步代码可以安全驱动异步主循环。 |
| `ReActAgent` | 基于文本 ReAct 协议的 Agent 子类，支持惰性工具加载、catalog-first 与热加载尾块。 |
| `ReActAgent.__init__` | 构造实例，校验 `lazy_tools`，准备仓库与待确认列表，并按模式触发工具发现。 |
| `ReActAgent.discover_tools` | 发现工具；惰性模式下只把元数据落库、丢弃临时注册表，不留工具实例。 |
| `ReActAgent.register_tool` | 登记工具；惰性模式下只持久化契约与 `implementation_ref`，推迟实现构造。 |
| `ReActAgent.is_tool_registered` | 同时查运行时注册表与 SQLite 仓库，判断工具（可含版本/哈希约束）是否已登记。 |
| `ReActAgent.tool_registration_status` | 返回工具登记详情字典（版本、哈希、generation、实现引用），未加载时 generation 为 `None`。 |
| `ReActAgent.tool_confirmation_key` | 在不加载工具的前提下算出确认钥匙，未加载时用固定 generation `1` 拼接。 |
| `ReActAgent.run` | 同步便捷入口，校验 query 后经 `_run_sync` 执行 `run_with_react`。 |
| `ReActAgent.run_with_react` | ReAct 异步主循环：组装提示词、调模型、解析、执行工具、回喂 Observation 直到最终答案。 |
| `ReActAgent._should_require_tool_action` | 保守判断纯文本回答是否该被打回重试（仅当请求含实时/检索类意图词且有可用工具）。 |
| `ReActAgent._initial_conversation` | 把历史或配置化系统提示词与本次请求消息拼成本轮初始对话。 |
| `ReActAgent._with_tool_instructions` | 组装缓存友好的系统指令，渲染冻结清单与每个可用工具的完整契约。 |
| `ReActAgent._hot_zone_lines` | 渲染追加在对话末尾的热加载工具区，最近 4 个保留完整契约、其余降级为名单行。 |
| `ReActAgent._ensure_tool_loaded` | 用 `implementation_ref` 动态导入并构造工具，校验契约一致后注册进运行时注册表。 |
| `ReActAgent._load_catalog_result` | 解析目录工具成功结果，把携带的 schema 写入缓存并排进加载顺序。 |
| `ReActAgent._load_catalog_observation` | 从原生路径的 Observation 消息里还原 `ToolResult` 并转交 `_load_catalog_result`。 |
| `ReActAgent._unavailable_tool_error` | 为「已注册但本次不可调用」生成区分白名单禁用与 schema 未加载的精确错误。 |
| `ReActAgent._response_message` | 从原始 LLM 响应里取第一条 choice 的 message，缺失时抛 `RuntimeError`。 |
| `ReActAgent._execute_action` | 把一次解析出的文本动作过白名单、加载、纠偏后构造 `ToolCall` 并真正执行。 |
| `ReActAgent._record_pending_confirmation` | 把因缺确认钥匙而失败的写操作去重记入 `pending_confirmations`。 |
| `ReActAgent._execute_native_calls` | 兼容原生 tool_calls：规范化名字、按需加载、批量执行并生成 Observation 列表。 |
| `ReActAgent._canonical_action_name` | 把模型写的点号/双下划线工具名映射回注册表里的规范名，匹配不上则原样返回。 |
