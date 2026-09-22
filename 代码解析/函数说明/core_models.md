# core/models.py

## 一、这个文件是干什么的

这个文件是整个 Agent 运行时的「数据契约中心」，它不包含任何业务逻辑、网络调用或数据库操作，只负责用 Pydantic 与 dataclass 把系统里最核心的几种数据形状固定下来，让模型、工具注册表、执行器、记忆层之间传递的数据有唯一且强校验的定义。文件里主要包含三类东西：第一类是模型侧的工具调用协议对象 `ToolCall`、`ToolError`、`ToolResult`、`BatchToolResult`，它们描述「模型请求调用工具」与「工具返回结果」这两个方向的报文；第二类是工具注册表使用的静态元数据 `ToolSpec`，它把工具的名字、版本、输入输出模型、副作用、超时、并发、标签、使用规范等打包成一个不可变对象，并据此算出 schema 哈希；第三类是单次工具请求的执行上下文 `ExecutionContext`，携带主体标识、权限集合与已确认的副作用集合。所有 Pydantic 报文都继承自 `StrictModel`，统一开启了 `extra="forbid"`、`strict=True`、`validate_assignment=True`，也就是说未知字段会被直接拒绝、类型不做隐式宽松转换、赋值时也会重新校验，这是整个项目「不静默丢弃字段、不猜测类型」风格的基础。`ToolSpec` 与 `ExecutionContext` 是 frozen dataclass，实例化之后不可修改，因此可以安全地被缓存、共享和作为字典键使用。`ToolSpec` 在构造时会把输入/输出 JSON Schema 规范化序列化后做 SHA-256，得到 `schema_hash`，再用它拼出 `confirmation_key`，从而把「用户对某个副作用的确认」绑定到工具的确切可执行契约上；只要 schema 变了，哈希就变，旧的确认自然失效。`guidance` 字段刻意被排除在哈希之外，因为它只影响提示词措辞、不影响可执行契约，改它不应该让已存储的确认或工具缓存失效。整体上，这个文件是工具系统、确认机制、批量结果校验和 prompt 渲染共同依赖的底层地基，被运行时的多个层次反复导入使用。

## 二、函数与类逐条详解

### `class StrictModel(BaseModel)` （第 14 行）
- **作用**：它是本文件所有 Pydantic 报文模型的公共基类，存在的唯一目的是把项目要求的三种严格行为一次性配置好，避免每个模型各自重复声明配置、也避免有人漏配导致校验被放松。它统一开启 `extra="forbid"`，意味着任何未在模型中声明的字段都会让校验失败，而不是被悄悄丢掉，这对工具调用协议尤其关键，因为参数一旦被静默丢弃，工具就会在缺少输入的情况下执行。它开启 `strict=True`，意味着 Pydantic 不会做宽松的类型强制转换，例如字符串 `"1"` 不会被当作整数 `1` 接受，从而保证跨模型、跨工具的报文类型是确定的。它开启 `validate_assignment=True`，意味着对象创建之后即使给字段重新赋值，也会重新走一遍校验，防止运行时被改成非法状态。它本身不定义任何字段，因此不能直接实例化出有意义的对象，只作为继承基类使用。当本文件定义 `ToolCall`、`ToolError`、`ToolResult`、`BatchToolResult` 时，都直接继承它来获得这套严格语义。
- **参数**：无自定义参数。作为 Pydantic 模型，实例化时接收的关键字参数由各子类自己的字段决定；由于配置了 `extra="forbid"`，传入任何子类未声明的字段都会触发校验错误。
- **返回**：它不是函数，没有返回值；实例化子类时返回对应子类的实例。
- **内部流程**：类体只做一件事，就是把 `model_config` 设置为一个 `ConfigDict`，其中包含 `extra="forbid"`、`strict=True`、`validate_assignment=True` 三个键值对。Pydantic 在构建类时会读取这个 `model_config`，生成对应的核心校验 schema，之后所有子类在继承时自动带上这份配置。
- **异常/边界**：类定义本身不抛异常。它间接决定了子类的行为边界：未知字段触发校验错误、类型不匹配触发校验错误、赋值非法值触发校验错误。
- **同文件关系**：被 `ToolCall`、`ToolError`、`ToolResult`、`BatchToolResult` 四个类继承；不调用本文件中的其它函数。

### `class ToolCall(StrictModel)` （第 18 行）
- **作用**：它描述「模型产出的、请求执行某一个已注册工具版本」的调用意图，是模型输出到运行时执行器之间的标准报文。它把一次调用需要被审计和校验的信息全部带齐：调用的唯一标识、工具名、工具 schema 版本、schema 哈希、注册表代号、实参以及依赖关系。之所以要同时带 `schema_version` 和 `schema_hash`，是因为运行时要能判断模型看到的工具契约是否与当前注册表一致，防止模型拿着过期契约去调用已经改签名的工具。`registry_generation` 允许为 `None`，用于兼容那些不追踪注册表代号的调用来源，但一旦给出就必须大于等于 1。`depends_on` 用来表达同批次内多个调用之间的先后依赖，默认空列表表示该调用不依赖任何其它调用，可以独立执行。它被运行时的解析层用来把模型输出转成结构化调用，被校验层用来做契约一致性检查，也被执行器用来决定调度顺序。
- **参数**：字段即参数，全部通过关键字或位置传入构造：`type`（`Literal["tool_call"]`，默认 `"tool_call"`，固定字面量，用于消息类型的判别）；`call_id`（`str`，必填，长度 1 到 128，是该次调用的唯一标识，结果通过它回关联）；`tool_name`（`str`，必填，长度 1 到 200，要调用的工具名）；`schema_version`（`str`，必填，长度 1 到 32，模型所依据的工具 schema 版本）；`schema_hash`（`str`，必填，长度 1 到 128，模型所依据的工具 schema 哈希）；`registry_generation`（`int | None`，默认 `None`，给定时必须 `>= 1`，表示注册表代号）；`arguments`（`dict[str, Any]`，必填，工具实参，键为字符串，值类型不限，具体合法性由对应工具的输入模型负责）；`depends_on`（`list[str]`，默认空列表，列表元素被约束为长度 1 到 128 的非空字符串，表示本调用依赖的其它 `call_id`）。
- **返回**：构造时返回 `ToolCall` 实例；字段可通过属性访问，因 `validate_assignment=True`，后续赋值同样受校验约束。
- **内部流程**：继承 `StrictModel` 获得严格配置；Pydantic 依据字段注解和 `Field(...)` 约束生成校验器；实例化时逐个字段校验长度、类型与下界；`type` 字段的默认值让调用方无需显式书写即可满足判别需求；`depends_on` 通过 `default_factory=list` 在每次实例化时新建列表，避免多个实例共享同一个可变默认值。
- **异常/边界**：字段缺失、类型不符、字符串长度越界（如 `call_id` 为空串或超过 128 字符）、`registry_generation` 小于 1、`depends_on` 中出现空串或超长串、传入任何未声明字段，都会抛出 Pydantic 的 `ValidationError`。`arguments` 内部值的语义合法性不在本类检查范围内，留给工具自己的输入模型。无特殊超时或空值兜底逻辑。
- **同文件关系**：继承本文件的 `StrictModel`；不调用本文件中的任何函数，也不被本文件中的其它函数直接调用（由运行时其它模块消费）。

### `class ToolError(StrictModel)` （第 33 行）
- **作用**：它描述工具执行失败时的结构化错误信息，是失败结果里 `error` 字段的类型。它把错误拆成机器可判定的错误码、给人和模型看的错误消息、以及是否可重试三个维度，从而让上层可以按错误码做分支处理、按 `retryable` 决定是否重试，而不是去解析自由文本。错误码被限制在 64 个字符以内，是为了保证它是短标识（例如 `DUPLICATE_CALL_ID`、`TIMEOUT`）而不是把整段描述塞进去；消息上限 2000 字符，是为了让错误能完整传达又不至于污染上下文。`retryable` 默认 `False`，意味着除非工具显式声明可重试，否则运行时应当把它当作终态失败处理，这是一种保守的默认。它主要出现在 `ToolResult.error` 中，也被 `BatchToolResult` 的重复 `call_id` 校验逻辑读取错误码。
- **参数**：`code`（`str`，必填，长度 1 到 64，机器可读的错误标识）；`message`（`str`，必填，长度 1 到 2000，人类可读的错误描述）；`retryable`（`bool`，默认 `False`，表示该失败是否值得重试）。
- **返回**：构造时返回 `ToolError` 实例。
- **内部流程**：继承 `StrictModel` 得到严格配置；实例化时 Pydantic 校验 `code` 与 `message` 的长度区间、`retryable` 的布尔类型，并拒绝任何额外字段；无其它自定义逻辑。
- **异常/边界**：`code` 或 `message` 缺失、为空串、超长，`retryable` 非布尔值，或出现未声明字段时抛出 `ValidationError`。类本身不对错误码取值做枚举限制，任何非空短字符串都合法。无特殊处理。
- **同文件关系**：继承本文件的 `StrictModel`；被 `ToolResult` 用作 `error` 字段的类型，被 `BatchToolResult.validate_unique_call_ids` 通过 `result.error.code` 读取错误码。

### `class ToolResult(StrictModel)` （第 39 行）
- **作用**：它描述单个工具调用的执行结果，是执行器回传给模型与记忆层的标准报文。它通过 `call_id` 与发起调用的 `ToolCall` 一一对应，通过 `tool_name` 保留工具身份便于审计，通过布尔字段 `ok` 表示成功或失败，并分别用 `data` 承载成功时的返回数据、用 `error` 承载失败时的结构化错误。这种「成功必有数据、失败必有错误、且两者互斥」的约束由类内的校验器强制，防止出现既有数据又有错误、或既不成功也没有错误原因的模糊状态。它既可能单独出现（单次调用），也可能被包在 `BatchToolResult.results` 里（批量调用）。因为设置了 `validate_assignment=True`，即便先构造再修改 `ok`、`data`、`error`，也会重新触发一致性校验，非法组合无法落地。
- **参数**：`type`（`Literal["tool_result"]`，默认 `"tool_result"`，固定判别字面量）；`call_id`（`str`，必填，长度 1 到 128，对应调用的标识）；`tool_name`（`str`，必填，长度 1 到 200）；`ok`（`bool`，必填，成功为 `True`、失败为 `False`）；`data`（`Any | None`，默认 `None`，成功时的返回数据，类型不限）；`error`（`ToolError | None`，默认 `None`，失败时的错误对象）。
- **返回**：构造时返回 `ToolResult` 实例；模型校验器返回自身以便链式继续校验。
- **内部流程**：继承 `StrictModel`；实例化时 Pydantic 先做字段级校验（长度、类型、嵌套 `ToolError` 的校验），随后执行 `mode="after"` 的模型校验器 `validate_consistency`，由它检查 `ok` 与 `error`、`data` 的组合是否自洽；任一条件不满足即抛出 `ValueError`，被 Pydantic 包装为 `ValidationError`。
- **异常/边界**：字段缺失、类型不符、字符串长度越界、`error` 不是合法 `ToolError`、出现未声明字段都会抛 `ValidationError`；此外 `ok=True` 却带 `error`、`ok=False` 却不带 `error`、`ok=False` 却带 `data` 这三种语义矛盾也会抛错。类本身不对 `data` 的形状做进一步限制。无特殊超时处理。
- **同文件关系**：继承本文件的 `StrictModel`；使用本文件的 `ToolError` 作为字段类型；调用本文件的模型校验器 `validate_consistency`；被本文件的 `BatchToolResult` 作为 `results` 的元素类型使用。

### `validate_consistency(self) -> ToolResult` （第 48 行）
- **作用**：它是 `ToolResult` 的模型级校验器，负责强制「成功」与「失败」两种状态在字段组合上互斥且完整。没有它的话，构造出的结果对象可能出现 `ok=True` 同时又带着错误对象、或者 `ok=False` 却没有给出任何失败原因的情况，上层代码就必须到处写防御性判断，而且模型看到的失败结果会缺少可操作的错误信息。它还会拒绝失败结果携带 `data`，避免「失败但其实有数据」这种让调用方难以判断该不该采信数据的模糊状态。由于注册为 `mode="after"`，它运行在字段级校验全部通过之后，此时 `ok`、`data`、`error` 都已是可信类型，校验逻辑只需要关注三者之间的语义关系。它同时也会在赋值场景下被触发，因为基类开启了 `validate_assignment`。
- **参数**：`self`（`ToolResult`，隐式传入，正在被校验的模型实例本身），无其它参数。
- **返回**：校验通过时返回 `self`，即原实例；这是 Pydantic `mode="after"` 校验器的约定，返回的对象会成为最终模型实例。
- **内部流程**：先判断 `self.ok` 为真且 `self.error is not None`，若是则抛出 `ValueError("successful tool results cannot contain an error")`；再判断 `self.ok` 为假且 `self.error is None`，若是则抛出 `ValueError("failed tool results must contain an error")`；最后判断 `self.ok` 为假且 `self.data is not None`，若是则抛出 `ValueError("failed tool results cannot contain data")`；三个分支都未命中时返回 `self`。
- **异常/边界**：上述三种语义矛盾会抛 `ValueError`（被 Pydantic 汇总为 `ValidationError`）。对于 `ok=True` 且 `data=None` 的情况它不做限制，即成功但无返回值是被允许的；也不检查 `data` 的具体结构。无特殊处理。
- **同文件关系**：只读取 `ToolResult` 自身的字段；被 Pydantic 在 `ToolResult` 实例化与赋值时自动调用；不调用本文件中的其它函数，也不被本文件中的其它函数显式调用。

### `class BatchToolResult(StrictModel)` （第 58 行）
- **作用**：它把同一批次的多个 `ToolResult` 打包成一个报文，供运行时一次性回传多个工具调用的结果。批量结果最容易出的问题是 `call_id` 重复导致调用与结果无法一一对应，因此这个类在校验器里强制要求所有 `call_id` 唯一。唯一的例外是「重复 ID 诊断结果」：当运行时检测到输入里存在重复 ID 时，会为每个输入各返回一条诊断结果，这些诊断结果本身必然共享同一个 `call_id`，所以代码允许这种「同 ID 的多条结果全部是 `DUPLICATE_CALL_ID` 错误」的情况作为唯一合法的重复表示。这样设计既保证了正常情况下映射无歧义，又给运行时的诊断路径留了合法出口。它通过 `type` 字段与单个 `ToolResult` 区分开，便于消息流解析时判别。
- **参数**：`type`（`Literal["batch_tool_result"]`，默认 `"batch_tool_result"`，固定判别字面量）；`results`（`list[ToolResult]`，必填，本批次的结果列表，元素必须是合法的 `ToolResult`，允许为空列表）。
- **返回**：构造时返回 `BatchToolResult` 实例。
- **内部流程**：继承 `StrictModel`；实例化时 Pydantic 先逐个校验 `results` 中每个元素是否为合法 `ToolResult`（包含嵌套的一致性校验），随后执行 `mode="after"` 的 `validate_unique_call_ids` 做跨元素唯一性检查。
- **异常/边界**：`results` 缺失、类型不是列表、元素不是合法 `ToolResult`、出现未声明字段都会抛 `ValidationError`；`call_id` 重复且不满足「全部为 `DUPLICATE_CALL_ID` 错误」这一例外时也会抛错。空列表合法，不做非空要求。无特殊超时处理。
- **同文件关系**：继承本文件的 `StrictModel`；使用本文件的 `ToolResult` 作为元素类型；调用本文件的模型校验器 `validate_unique_call_ids`。

### `validate_unique_call_ids(self) -> BatchToolResult` （第 63 行）
- **作用**：它是 `BatchToolResult` 的模型级校验器，用来保证批量结果里的 `call_id` 在正常情况下互不重复，从而让「调用 → 结果」的映射保持唯一可解析。同时它实现了一个有意的例外：运行时在检测到重复 `call_id` 时会为每个输入各产生一条诊断结果，这些诊断结果天然共享同一个 `call_id`，因此代码只在「同一个 ID 下的所有结果都是 `DUPLICATE_CALL_ID` 错误」时才放行，其余任何重复都视为数据错误。之所以用「分组后逐组判断」而不是用集合长度比较，是因为需要区分「重复」与「合法的诊断式重复」，集合比较无法表达这一例外。它在字段级校验之后运行，此时每条 `ToolResult` 已经确保自身状态自洽，因此可以直接安全地读取 `result.error` 与 `result.error.code`。
- **参数**：`self`（`BatchToolResult`，隐式传入，正在被校验的实例），无其它参数。
- **返回**：校验通过时返回 `self`，作为最终模型实例；一旦发现非法重复则抛出 `ValueError`。
- **内部流程**：先创建一个 `defaultdict(list)` 作为分组容器 `grouped`；遍历 `self.results`，按 `result.call_id` 把每条结果追加进对应分组；然后遍历分组的所有值 `grouped.values()`，对元素数量小于 2 的分组直接 `continue` 跳过；对数量大于等于 2 的分组，检查是否所有元素都满足 `result.error is not None` 且 `result.error.code == "DUPLICATE_CALL_ID"`，只要有一个不满足就抛出 `ValueError("batch tool results must have unique call_id values")`；全部通过后返回 `self`。
- **异常/边界**：非法重复抛 `ValueError`（被包装为 `ValidationError`）；`results` 为空列表时循环不执行、直接返回，属于合法情况；分组键 `call_id` 一定存在且非空，因为 `ToolResult` 的字段约束已保证。无特殊处理。
- **同文件关系**：读取本文件 `ToolResult` 的 `call_id`、`error` 与 `ToolError.code` 字段；被 Pydantic 在 `BatchToolResult` 实例化与赋值时自动调用；不调用本文件中的其它函数。

### `class ToolSpec` （第 81 行，`@dataclass(frozen=True)`）
- **作用**：它是工具注册表使用的静态元数据载体，把一个工具「叫什么、干什么、哪个版本、输入输出长什么样、有没有副作用、需要什么权限、超时多久、能不能重试、能不能并行、并发上限多少、属于什么标签、建议先调用哪些工具、使用规范是什么」全部收拢到一个不可变对象里。它被标记为 `frozen=True`，因此实例化后不可修改，可以安全地缓存、跨请求共享，并因为 dataclass 默认生成 `__hash__` 而能放进集合或作为字典键（注意字段中的 `input_model`、`output_model` 是类对象，哈希稳定）。它显式声明 Pydantic 的输入输出模型类才是契约的唯一真相来源，而 `recommended_before_tools` 只是给模型看的建议性元数据，刻意不构成可执行的依赖图、运行时也从不强制，这样避免了「建议」被误当成「调度约束」。`guidance` 是工具的提示词级使用规范（什么时候该用、什么时候不该用、必须遵守什么硬约束），会被渲染进提示词，因此被限制在 1200 字符以内以控制上下文开销；它被刻意排除在 `schema_hash` 之外，因为改它只改变对模型的行为指示、不改变可执行契约，所以不应让已存储的确认或工具缓存失效。它在构造时通过 `__post_init__` 做全面校验并计算 `_schema_hash`，把「契约指纹」固化在实例上。
- **参数**：`name`（`str`，必填，必须匹配 `[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+`，即必须带命名空间，例如 `web.search`）；`description`（`str`，必填，去空白后非空且不超过 2000 字符）；`version`（`str`，必填，去空白后非空且不超过 32 字符）；`input_model`（`type[BaseModel]`，必填，必须是 Pydantic 模型类且配置 `extra="forbid"`）；`output_model`（`type[BaseModel]`，必填，同样必须是配置了 `extra="forbid"` 的 Pydantic 模型类）；`side_effect`（`str`，默认 `"read"`，去空白后非空且不超过 32 字符）；`permissions`（`tuple[str, ...]`，默认空元组，元素必须是非空字符串）；`timeout_seconds`（`float`，默认 `30.0`，必须是有限正数，布尔值不被接受）；`idempotent`（`bool`，默认 `True`）；`parallel_safe`（`bool`，默认 `True`）；`max_concurrency`（`int | None`，默认 `None`，给定时必须为正整数，布尔值不被接受）；`tags`（`tuple[str, ...]`，默认空元组，元素必须是非空字符串）；`recommended_before_tools`（`tuple[str, ...]`，默认空元组，元素必须匹配与 `name` 相同的命名空间正则，且不能包含自身）；`guidance`（`str`，默认 `""`，必须是不超过 1200 字符的字符串）；`_schema_hash`（`str`，`init=False` 且 `repr=False`，不参与构造，由 `__post_init__` 计算写入）。
- **返回**：构造时返回 `ToolSpec` 实例；因为 frozen，任何字段赋值都会抛异常。
- **内部流程**：dataclass 生成 `__init__` 接收上述字段（`_schema_hash` 因 `init=False` 不在参数列表），构造末尾自动调用 `__post_init__` 完成全部校验与哈希计算；之后实例的字段只读。
- **异常/边界**：所有非法输入都在 `__post_init__` 中被拦截，抛出 `ValueError` 或 `TypeError`（详见 `__post_init__` 条目）。实例化后赋值会因 `frozen=True` 抛 `FrozenInstanceError`。无特殊超时处理。
- **同文件关系**：它的 `__post_init__` 会调用 `self.input_model.model_json_schema()`、`self.output_model.model_json_schema()`、`json.dumps`、`hashlib.sha256` 与 `object.__setattr__`；`schema_hash`、`confirmation_key`、`input_schema`、`output_schema`、`summary`、`model_description` 都是它的成员，供注册表与提示词渲染层使用；本文件内没有其它函数调用它。

### `__post_init__(self) -> None` （第 113 行）
- **作用**：它是 `ToolSpec` 的构造后钩子，承担两件事：一是对这个 frozen dataclass 的所有字段做一次彻底的合法性校验，二是在校验全部通过后计算并写入 `_schema_hash`。之所以需要它，是因为 dataclass 本身只提供字段声明，不会像 Pydantic 那样自动校验，而工具注册表是系统的关键入口，元数据一旦非法（比如工具名没有命名空间、输入模型允许额外字段、超时是负数或 NaN）就会在运行时造成难以定位的问题，所以必须在构造阶段就把错误暴露出来。它对输入输出模型额外要求 `extra="forbid"`，理由是工具契约不能静默丢弃未知字段，否则模型多传的参数会被无声吞掉、行为不可预期。它把 `guidance` 限制在 1200 字符，因为该文本会进入提示词，过长会挤占上下文。它最后把输入输出两个 JSON Schema 放进一个字典，用 `sort_keys=True`、紧凑分隔符、`ensure_ascii=True` 做规范化序列化，再取 SHA-256，从而得到一个与字典键顺序无关、与 Python 版本无关的稳定指纹；由于实例是 frozen 的，普通 `self._schema_hash = ...` 会失败，所以使用 `object.__setattr__` 绕过冻结写入。
- **参数**：`self`（`ToolSpec`，隐式传入，正在初始化的实例），无其它参数。
- **返回**：无返回值（`None`）；其副作用是校验实例字段并在成功时设置 `self._schema_hash`。
- **内部流程**：第一步校验 `name` 是 `str` 且能 `re.fullmatch` 匹配 `[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+`，不满足则抛 `ValueError`，提示工具名必须使用命名空间。第二步校验 `description` 是 `str`、`strip()` 后非空、长度不超过 2000。第三步校验 `version` 是 `str`、去空白非空、长度不超过 32。第四步与第五步分别校验 `input_model`、`output_model` 是类（`isinstance(x, type)`）且是 `BaseModel` 的子类，否则抛 `TypeError`。第六步遍历 `("input_model", self.input_model)` 与 `("output_model", self.output_model)` 两个二元组，检查 `model.model_config.get("extra") != "forbid"` 时抛 `ValueError`。第七步校验 `side_effect` 是 `str`、去空白非空、长度不超过 32。第八步校验 `timeout_seconds`：先排除 `bool`（因为 `bool` 是 `int` 子类），再要求是 `int` 或 `float`，再要求 `math.isfinite` 为真（排除 `inf` 与 `nan`），再要求大于 0。第九步校验 `idempotent` 与 `parallel_safe` 都是 `bool`，否则抛 `TypeError`。第十步校验 `permissions` 是 `tuple` 且每个元素都是非空字符串，否则抛 `TypeError`。第十一步对 `tags` 做同样的元组与非空字符串检查。第十二步校验 `recommended_before_tools` 是 `tuple` 且每个元素都能匹配命名空间正则，否则抛 `TypeError`。第十三步检查 `self.name` 是否出现在 `self.recommended_before_tools` 中，是则抛 `ValueError`，禁止工具推荐自己作为前置工具。第十四步校验 `guidance` 是 `str` 且长度不超过 1200，否则抛 `TypeError`。第十五步校验 `max_concurrency`：为 `None` 时跳过，否则排除 `bool`、要求是 `int`、要求大于 0，不满足抛 `ValueError`。最后构造 `schema` 字典，键 `"input"` 与 `"output"` 分别对应两个模型的 `model_json_schema()`；用 `json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)` 得到规范化字符串 `encoded`；再用 `object.__setattr__(self, "_schema_hash", hashlib.sha256(encoded.encode("utf-8")).hexdigest())` 写入 64 位十六进制摘要。
- **异常/边界**：名称不合命名空间规则、描述为空或超长、版本为空或超长、输入输出模型不是 Pydantic 类、模型未配置 `extra="forbid"`、`side_effect` 非法、超时为布尔/非数值/非有限/非正、`max_concurrency` 非法、自我推荐，均抛 `ValueError`；`idempotent`/`parallel_safe` 非布尔、`permissions`/`tags`/`recommended_before_tools` 不是元组或含空/非法元素、`guidance` 非字符串或超长，均抛 `TypeError`。注意 `guidance` 长度为 0 是合法的（默认值就是空串）。字段校验顺序固定，因此多个问题同时存在时只会先报最早命中的那个。无特殊超时处理。
- **同文件关系**：由 dataclass 在 `ToolSpec` 实例化时自动调用；内部调用 `self.input_model.model_json_schema()` 与 `self.output_model.model_json_schema()`（即 `input_schema`、`output_schema` 两个属性所暴露的同一方法），并使用标准库 `re`、`math`、`json`、`hashlib`；它为 `schema_hash`、`confirmation_key`、`summary`、`model_description` 等成员准备好数据基础。

### `schema_hash` （property，第 211 行）
- **作用**：它是对外暴露工具契约指纹的只读属性，让注册表、缓存层、确认机制能够拿到 `__post_init__` 计算好的 SHA-256 摘要，而不需要接触私有的 `_schema_hash` 字段。之所以要把它包装成属性而不是直接暴露 `_schema_hash`，一是语义上对外只读、避免被误改，二是让内部存储细节（字段名、计算时机）可以自由调整而不影响调用方。它返回的哈希覆盖了输入与输出两个 JSON Schema 的规范化序列化结果，因此只要任一模型的字段、类型或约束发生变化，哈希就会变化；运行时可以据此判断模型看到的工具契约是否过期、已存储的确认是否还有效。它是 `confirmation_key` 的组成部分，也是 `summary()` 输出的关键字段之一。
- **参数**：`self`（`ToolSpec`，隐式传入），无其它参数。
- **返回**：返回 `str`，即 `__post_init__` 中写入的 64 位十六进制 SHA-256 摘要。
- **内部流程**：直接读取并返回 `self._schema_hash`，不做任何计算或缓存判断。
- **异常/边界**：如果实例不是经由正常构造产生（例如用 `object.__new__` 绕过 `__init__`），`_schema_hash` 未设置会抛 `AttributeError`；正常构造路径下不会出现该情况。无特殊处理。
- **同文件关系**：读取 `__post_init__` 写入的 `_schema_hash`；被 `confirmation_key` 与 `summary` 调用；被本文件外的注册表与确认逻辑使用。

### `confirmation_key` （property，第 215 行）
- **作用**：它把「对某个工具副作用的确认」绑定到工具的确切可执行契约上，返回形如 `名称@版本#schema哈希` 的字符串。设计意图是：只有当工具名、版本、以及输入输出 schema 全都一致时，确认键才一致；一旦工具的 schema 变了（哈希变）或版本变了，旧的确认键就不再匹配，之前记录的确认自动失效，用户必须重新确认，从而避免「用户确认的是旧契约、实际执行的是新契约」这种危险情况。注释里明确说它绑定的是可执行契约，这也解释了为什么 `guidance` 被排除在哈希之外——使用规范变化不该让确认失效。它通常被用作确认集合（例如 `ExecutionContext.confirmed_side_effects`）中的元素，运行时用相等比较来判断某个副作用是否已被确认。
- **参数**：`self`（`ToolSpec`，隐式传入），无其它参数。
- **返回**：返回 `str`，格式固定为 `f"{self.name}@{self.version}#{self.schema_hash}"`。
- **内部流程**：读取 `self.name`、`self.version` 与属性 `self.schema_hash`，用 `@` 与 `#` 两个分隔符拼接成一个字符串并返回；分隔符的选择让名称、版本、哈希三段在人工排查时也能一眼分开。
- **异常/边界**：不做任何校验，也不处理名称或版本中出现 `@`、`#` 的歧义情况（`__post_init__` 的命名空间正则已排除了名称中的这些字符，版本则未做字符限制）。无特殊处理。
- **同文件关系**：调用本文件的 `schema_hash` 属性；被本文件外的确认存储与检查逻辑使用，与 `ExecutionContext.confirmed_side_effects` 的语义配套。

### `input_schema` （property，第 220 行）
- **作用**：它以属性形式返回工具的输入模型 JSON Schema，供提示词渲染、文档生成、模型侧契约展示以及外部工具发现接口使用。之所以不直接暴露 `input_model` 让调用方自己调用，是因为 schema 是运行时对外表达契约的稳定形式，包一层属性可以统一出口、避免调用方各自处理模型细节。它每次访问都会重新调用 `model_json_schema()`，因此拿到的是当前模型类定义对应的最新 schema；对于 schema 随代码更新而变化的场景，这保证了展示与校验依据一致。它不参与 `schema_hash` 的计算路径（哈希是在 `__post_init__` 里单独调用模型方法得到的），但两者结果相同。
- **参数**：`self`（`ToolSpec`，隐式传入），无其它参数。
- **返回**：返回 `dict[str, Any]`，即 Pydantic 生成的输入 JSON Schema 字典。
- **内部流程**：直接调用 `self.input_model.model_json_schema()` 并把结果返回，没有缓存。
- **异常/边界**：若 `input_model` 不是合法的 Pydantic 模型，调用会失败；但 `__post_init__` 已保证它必须是配置了 `extra="forbid"` 的 `BaseModel` 子类，因此正常路径不会出错。无特殊处理。
- **同文件关系**：使用 `self.input_model`；与 `__post_init__` 中构造 `schema["input"]` 的调用等价；被本文件外的提示词渲染与文档层使用。

### `output_schema` （property，第 224 行）
- **作用**：它以属性形式返回工具输出模型的 JSON Schema，与 `input_schema` 对称，供模型了解工具会返回什么结构、供文档与外部发现接口展示返回契约。它让调用方不必知道内部字段是 `output_model` 还是别的名字，也避免外部直接操作模型类。由于每次访问都重新生成，它能反映模型类的当前定义。它在运行时契约检查与提示词渲染中与 `input_schema` 成对使用，共同构成工具对外的完整形状描述。
- **参数**：`self`（`ToolSpec`，隐式传入），无其它参数。
- **返回**：返回 `dict[str, Any]`，即 Pydantic 生成的输出 JSON Schema 字典。
- **内部流程**：直接调用 `self.output_model.model_json_schema()` 并返回结果，无缓存。
- **异常/边界**：与 `input_schema` 相同，`__post_init__` 的前置校验保证了模型类合法，正常路径不会抛错。无特殊处理。
- **同文件关系**：使用 `self.output_model`；与 `__post_init__` 中构造 `schema["output"]` 的调用等价；被本文件外的提示词渲染与文档层使用。

### `summary(self) -> dict[str, Any]` （第 227 行）
- **作用**：它把 `ToolSpec` 中适合对外展示与传输的字段整理成一个普通字典，用于工具清单接口、注册表快照、日志记录或提示词中的工具摘要。它刻意只挑选轻量、可 JSON 序列化的字段，因此把 `tags` 与 `recommended_before_tools` 两个元组转成了列表，而把 `input_model`、`output_model` 这两个类对象、以及 `permissions`、`timeout_seconds`、`idempotent`、`parallel_safe` 等字段排除在外，避免字典里出现不可序列化或不适合暴露给模型的内容。它包含 `schema_hash`，因此使用方可以据此判断工具契约是否发生变化。它包含 `guidance`，所以使用规范会随摘要一起呈现。它是把不可变 dataclass 转成可自由传递的普通数据结构的主要出口。
- **参数**：`self`（`ToolSpec`，隐式传入），无其它参数。
- **返回**：返回 `dict[str, Any]`，包含以下键：`tool_name`（工具名）、`description`（描述）、`version`（版本）、`schema_hash`（契约哈希）、`side_effect`（副作用类型）、`max_concurrency`（并发上限，可能为 `None`）、`tags`（标签列表）、`recommended_before_tools`（建议前置工具列表）、`guidance`（使用规范）。
- **内部流程**：构造一个字典字面量，逐个键写入对应字段：`self.name` 写入键 `"tool_name"`；`self.description`、`self.version`、属性 `self.schema_hash`、`self.side_effect`、`self.max_concurrency` 依次写入；`list(self.tags)` 与 `list(self.recommended_before_tools)` 把元组转成列表；`self.guidance` 写入。返回该字典。
- **异常/边界**：正常构造的实例不会抛异常；字段均为已校验值，因此无需再做防御。返回的字典是新对象，修改它不会影响 frozen 的 `ToolSpec`；但字典中的列表同样是新列表，修改也不影响原元组。无特殊处理。
- **同文件关系**：调用本文件的 `schema_hash` 属性；读取 `ToolSpec` 的多个字段；被本文件外的注册表、工具清单接口与提示词渲染层使用。

### `model_description` （property，第 241 行）
- **作用**：它返回给模型看的工具描述文本，也就是在基础 `description` 之外，按需追加一句「建议先调用的工具（仅建议、不强制）」的说明。之所以需要它，是因为 `recommended_before_tools` 是给模型的行为提示，但直接把裸列表塞给模型容易让它误以为这是硬性调度约束，所以这里显式写上「advisory only; not enforced」来消除歧义。当没有任何建议前置工具时，它直接返回原始描述，避免输出多余的空行或空句子，保持提示词紧凑。它体现了本文件「建议性元数据与可执行契约严格分离」的设计：真正强制的依赖要靠 `depends_on`，而这里只是措辞上的引导。
- **参数**：`self`（`ToolSpec`，隐式传入），无其它参数。
- **返回**：返回 `str`。当 `recommended_before_tools` 为空元组时，返回 `self.description` 本身；否则返回 `self.description` 加上换行，再接上固定前缀 `"Recommended preceding tools (advisory only; not enforced): "` 与用 `", "` 连接的工具名列表再加一个句点。
- **内部流程**：先判断 `if not self.recommended_before_tools`，为空则直接返回 `self.description`；否则用 `", ".join(self.recommended_before_tools)` 得到 `tools`，再用 f-string 把描述、换行、固定提示语和工具列表拼成最终字符串返回。
- **异常/边界**：`recommended_before_tools` 的元素已在 `__post_init__` 中保证是合法字符串，因此 `join` 不会因类型问题失败；空元组走早返回分支。无特殊处理。
- **同文件关系**：读取 `ToolSpec` 的 `description` 与 `recommended_before_tools` 字段；不调用本文件的其它函数；被本文件外的提示词渲染层使用（与 `core/tool_docs.py` 的渲染流程配套）。

### `class ExecutionContext` （第 254 行，`@dataclass(frozen=True)`）
- **作用**：它承载「一次工具请求」的执行上下文，说明这次执行是以谁的身份发起、拥有哪些权限、以及哪些副作用已经被确认。它被设计成 frozen dataclass，因此不可变、可安全共享与缓存，也便于作为参数在调用链中传递而不必担心被中途篡改，这对安全相关的语义尤其重要。`subject` 标识执行主体（默认 `"default"`），用于审计与归属；`permissions` 是权限集合，文档注释明确指出它目前仅作为工具兼容性与审计元数据保留，运行时与 Agent 目前都不据此做强制拦截，所以它是「记录性」而非「执行性」字段；`confirmed_side_effects` 存放注册表按代（generation）绑定的确认键（即 `ToolSpec.confirmation_key` 形式的值），运行时用它来判断某个副作用是否已被用户确认。注释还说明：被标记为 `destructive` 的工具不依赖这个集合，而是要求传入 `call_confirmation_key`，因为那种确认还需要绑定精确归一化后的实参。它使用 `frozenset` 而不是 `set` 或 `list`，既满足不可变要求，也让成员判断是 O(1)、且因为 dataclass 可哈希而让整个上下文可哈希。
- **参数**：`subject`（`str`，默认 `"default"`，去空白后必须非空）；`permissions`（`frozenset[str]`，默认空 frozenset，元素必须是非空且去空白后非空的字符串）；`confirmed_side_effects`（`frozenset[str]`，默认空 frozenset，元素同样必须是非空且去空白后非空的字符串）。
- **返回**：构造时返回 `ExecutionContext` 实例；因 frozen，构造后不可赋值修改。
- **内部流程**：dataclass 生成 `__init__` 接收三个字段（默认值在未提供时生效），构造末尾自动调用 `__post_init__` 做类型与内容校验；校验通过后实例字段只读，并可用于哈希与相等比较。
- **异常/边界**：非法值由 `__post_init__` 拦截，抛 `ValueError` 或 `TypeError`；实例化后赋值会因 frozen 抛 `FrozenInstanceError`。无特殊超时处理。
- **同文件关系**：调用自身的 `__post_init__`；语义上与 `ToolSpec.confirmation_key` 配套（确认键由后者生成、由前者的 `confirmed_side_effects` 承载）；本文件内没有其它函数调用它。

### `__post_init__(self) -> None` （第 270 行）
- **作用**：它是 `ExecutionContext` 的构造后校验钩子，用来保证这个安全相关的上下文对象在创建时就处于合法状态：主体必须是有意义的非空字符串（否则审计日志里会出现空白身份）、两个集合必须是 `frozenset` 且其中每个元素都是非空字符串（否则确认键或权限字符串里可能出现空串，导致成员判断行为不可预期）。它特别要求集合类型是 `frozenset` 而不是任意可迭代对象或 `set`，因为只有不可变集合才与 frozen dataclass 的「不可变、可哈希、可安全共享」定位一致。它对 `subject` 的检查同时排除了非字符串类型与纯空白字符串，对集合元素的检查则使用 `strip()` 判断，因此 `"  "` 这类只含空白的元素也会被拒绝。它不做任何权限语义层面的校验（比如权限名是否在白名单里），因为文档已说明权限当前不参与强制拦截，只作记录。
- **参数**：`self`（`ExecutionContext`，隐式传入，正在初始化的实例），无其它参数。
- **返回**：无返回值（`None`）；作用是通过校验或抛出异常来阻止非法实例存在。
- **内部流程**：第一步判断 `not isinstance(self.subject, str) or not self.subject.strip()`，成立则抛 `ValueError("subject must be a non-empty string")`。第二步判断 `not isinstance(self.permissions, frozenset) or not all(isinstance(item, str) and item.strip() for item in self.permissions)`，成立则抛 `TypeError("permissions must be a frozenset of non-empty strings")`。第三步对 `self.confirmed_side_effects` 做与第二步完全相同的结构检查，不满足则抛 `TypeError("confirmed_side_effects must be a frozenset of non-empty strings")`。三步全部通过即静默返回。
- **异常/边界**：`subject` 非字符串或为空白串抛 `ValueError`；两个集合不是 `frozenset`、或含非字符串元素、或含空白字符串元素抛 `TypeError`。空 frozenset 是合法默认值，会被接受。集合为空时 `all(...)` 对空迭代返回 `True`，因此不会误报。无特殊处理。
- **同文件关系**：由 dataclass 在 `ExecutionContext` 实例化时自动调用；只读取自身字段，不调用本文件中的其它函数，也不被本文件中的其它函数调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `StrictModel` | 所有 Pydantic 报文的公共基类，统一开启禁止额外字段、严格类型、赋值校验三项配置。 |
| `ToolCall` | 描述模型请求执行某个已注册工具版本的调用报文，携带调用标识、工具契约版本与哈希、实参与依赖。 |
| `ToolError` | 描述工具失败时的结构化错误，包含错误码、错误消息与是否可重试。 |
| `ToolResult` | 描述单次工具调用的结果，用 `ok` 区分成功失败并分别承载 `data` 或 `error`。 |
| `ToolResult.validate_consistency` | 强制成功结果不得带错误、失败结果必须带错误且不得带数据。 |
| `BatchToolResult` | 打包同批次多个工具结果的报文，并保证 `call_id` 一一对应。 |
| `BatchToolResult.validate_unique_call_ids` | 校验批量结果中 `call_id` 唯一，仅放行全部为 `DUPLICATE_CALL_ID` 的重复诊断结果。 |
| `ToolSpec` | 工具的不可变静态元数据载体，含名称、版本、输入输出模型、副作用、超时、并发、标签、使用规范与契约哈希。 |
| `ToolSpec.__post_init__` | 全面校验 `ToolSpec` 的所有字段，并计算输入输出 schema 的 SHA-256 契约指纹。 |
| `ToolSpec.schema_hash` | 只读属性，返回工具输入输出 schema 的规范化 SHA-256 摘要。 |
| `ToolSpec.confirmation_key` | 只读属性，返回 `名称@版本#schema哈希`，把副作用确认绑定到确切可执行契约。 |
| `ToolSpec.input_schema` | 只读属性，返回工具输入模型的 JSON Schema 字典。 |
| `ToolSpec.output_schema` | 只读属性，返回工具输出模型的 JSON Schema 字典。 |
| `ToolSpec.summary` | 把工具元数据整理成可序列化的普通字典供清单展示与日志使用。 |
| `ToolSpec.model_description` | 返回给模型看的描述文本，必要时追加「仅建议、不强制」的前置工具提示。 |
| `ExecutionContext` | 不可变的单次工具请求执行上下文，含主体标识、权限集合与已确认副作用集合。 |
| `ExecutionContext.__post_init__` | 校验主体非空、权限与已确认副作用必须是仅含非空字符串的 `frozenset`。 |
