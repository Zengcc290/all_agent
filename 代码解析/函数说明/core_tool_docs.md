# core/tool_docs.py

## 一、这个文件是干什么的

这个文件是一个**纯渲染（render）模块**，职责非常单一：把运行时注册表里每一个工具的 `ToolSpec`（工具契约对象），渲染成一段「给大模型看的、逐变量的中文契约文本」。它的存在是为了解决一个具体痛点——过去 ReAct 提示词只给模型「工具名 + 一句描述 + 原始 JSON Schema」，而原始 JSON Schema 对模型极不友好：必填项藏在 `required` 数组里、数值/长度约束散落在 `minLength`/`maximum`/`enum` 里、字段说明埋在 `properties.*.description` 里，而**输出字段、副作用、是否需要人工确认**这些关键信息根本没有进入提示词，导致模型猜参数名、漏必填项、对写操作反复试探。

文件内部只有函数，**没有任何 class 定义**，也没有全局可变状态；对外暴露的是四个 `render_*` 公共函数和两个深度常量。核心工作由一组私有辅助函数完成：`_resolve` 负责跟随 `$ref` 到 `$defs`，`_type_label` 把 schema 节点翻译成中文类型标签，`_constraints` 把约束关键字翻译成中文短语，`_format_default` 格式化默认值，`_field_lines` 把「属性表」渲染成一行一个变量的列表，`_confirmation_note` 根据副作用生成确认说明。

模块顶部定义了两个上限常量 `MAX_REF_DEPTH = 2`（嵌套模型展开深度上限，防止自引用模型把提示词撑爆）和 `MAX_FIELD_LINES = 60`（单工具渲染行数上限），以及一张中文类型名映射表 `_TYPE_LABELS`（把 `string`/`integer`/`array` 等翻译成「字符串」「整数」「数组」）。整个模块强调**确定性**：渲染顺序固定为「工具名排序 + 字段定义顺序」，文本中不含时间戳或随机值，因此同一批注册表每次渲染出的文本逐字节相同——这是 OpenAI 前缀缓存（KV cache）能复用的前提。

同一份渲染结果供两条链路共用，避免两处漂移：一条是 ReAct 文本协议（`agents/react.py` 的 `_with_tool_instructions`），另一条是原生 function-calling（`agents/agent.py` 的 `_definitions_for_registrations`，那里已经把 schema 放进 `tools` 字段，本模块再把同一份契约文本塞进 system 消息，让「prompt 里有全部工具信息」对两种协议都成立）。文件通过 `__all__` 显式声明了对外可见的名字。

## 二、函数与类逐条详解

本文件**不包含任何类**，因此下面按出现顺序逐条讲解全部 11 个模块级函数（含 7 个 `_` 开头的私有函数）。文件中也没有任何嵌套 `def`，`_type_label` 的递归是函数自调用，`_field_lines` 内部的推导式只是表达式，不构成嵌套函数。

### `_type_label(schema: Mapping[str, Any], defs: Mapping[str, Any], depth: int) -> str` （第 63 行）

- **作用**：把一个 JSON Schema 节点渲染成一句短小、人类/模型都易读的**中文类型标签**，是整个模块最核心的翻译器。它需要处理的情况远不止 `type` 字段：`enum` 要展开成取值列表、`const` 要写成固定值、`anyOf`/`oneOf` 要拆成多个分支用 `|` 连接、`type` 是列表时也要用 `|` 连接、数组要递归渲染 `items`、对象要渲染 `additionalProperties` 或 `properties` 的键名集合。没有它，模型看到的就只是 `{"type":"object"}` 这种毫无信息量的关键字；有了它，模型能看到「对象{query, top_k}」或「枚举["read", "write"]」这种直接可用的描述。它同时负责 `$ref` 的解引用（通过 `_resolve`）以及自引用深度的截断（返回「…」）。
- **参数**：
  - `schema`：`Mapping[str, Any]`，当前要渲染的 schema 节点（可能带 `$ref`、`enum`、`anyOf`、`type` 等任意组合）。函数不修改它，只读取；若传入的不是 Mapping，`_resolve` 的 `isinstance` 判断会让循环直接不执行，后续 `.get` 调用仍假定是 Mapping，因此调用方需保证是 Mapping（`_field_lines` 已用 `node if isinstance(node, Mapping) else {}` 兜底）。
  - `defs`：`Mapping[str, Any]`，即 schema 顶层 `$defs` 字典，用于把 `$ref` 指向的名字解析成真实子 schema；空字典 `{}` 表示没有可解析的定义（此时 `$ref` 会退化成 `{"type": 名字}`）。
  - `depth`：`int`，当前递归/展开深度，从 0 开始；每进入一层数组元素、对象附加值或 anyOf 分支就 `+1`。当 `depth > MAX_REF_DEPTH`（即大于 2）时，`_resolve` 返回占位类型「…」，从而硬性截断递归，防止自引用模型无限展开。无默认值，调用方必须显式传入。
- **返回**：总是返回 `str`，且保证非空。可能的具体形态：`枚举[...]`、`固定值 ...`、`A | B`（多分支或联合类型）、`任意`（无法判断类型时）、`数组[元素类型]`、`对象{键1, 键2}`、`对象{任意键: 值类型}`、纯「对象」、以及从 `_TYPE_LABELS` 查表得到的中文名（查不到时回退成原始英文类型名）。
- **内部流程**：第一步调用 `_resolve(schema, defs, depth)` 得到解引用后的 `resolved`。第二步按优先级做判断：① `resolved` 里有 `enum` → 用 `json.dumps(item, ensure_ascii=False)` 逐个序列化后用 `", "` 拼接，返回 `枚举[值1, 值2]`；② 有 `const` → 返回 `固定值 <json>`；③ 有 `anyOf` 或 `oneOf` → 取出分支列表（`resolved.get("anyOf") or resolved.get("oneOf") or []`），对**非 null** 的每个分支递归调用自身（`depth + 1`）得到标签列表；同时用 `_is_null` 扫描一遍，只要存在 null 分支就额外追加「空」；再用 `dict.fromkeys` 去重且保持顺序，非空则用 `" | "` 连接，全空则返回「任意」；④ `type` 是列表 → 逐个查 `_TYPE_LABELS`（查不到用原始字符串），`dict.fromkeys` 去重后 `" | "` 连接；⑤ `type == "array"` → 取 `items`（缺省 `{}`）并递归渲染，返回 `数组[...]`；⑥ `type == "object"` → 若 `additionalProperties` 是 Mapping 则返回 `对象{任意键: 类型}`，否则若 `properties` 是非空 Mapping 则把键名用 `", "` 拼成 `对象{a, b}`，都没有则返回「对象」；⑦ `type` 是普通字符串 → 查 `_TYPE_LABELS`，查不到就用原字符串；⑧ 走到最后还有 `properties` 键 → 返回「对象」；⑨ 全部不匹配 → 返回「任意」。
- **异常/边界**：本函数自身不主动抛异常。空 schema `{}` 会一路落到最后返回「任意」；`enum` 为空列表时会渲染成 `枚举[]`（不特殊处理）；`anyOf` 里混入 null 分支会被单独识别为「空」而不是一个类型分支；`depth` 超过 `MAX_REF_DEPTH` 时由 `_resolve` 返回 `{"type": "…"}`，最终输出「…」；`type` 为未知字符串（如自定义类型名）时原样透出英文，不会报错；`json.dumps` 对不可序列化的 `enum`/`const` 值（理论上不会出现在合法 schema 中）会抛 `TypeError`。
- **同文件关系**：调用了 `_resolve`、`_is_null`，并**递归调用自身**。被 `_field_lines`（第 183 行生成字段类型标签）以及它自己（数组元素、对象附加值、anyOf/oneOf 分支）调用。

### `_is_null(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> bool` （第 106 行）

- **作用**：判断一个 schema 节点解引用之后是不是「null 类型」。它专门服务于 `_type_label` 里的可空联合类型处理：JSON Schema 表达「可选字符串」通常写成 `anyOf: [{type: string}, {type: null}]`，如果不把 null 分支单独挑出来，渲染结果就会变成莫名其妙的「字符串 | 空 | ...」顺序混乱或重复。抽成独立小函数的好处是让 `_type_label` 里那段 `any(_is_null(branch, defs) for branch in branches)` 与逐分支过滤两处逻辑共用同一判断，不会出现两处口径不一致。它本身逻辑极短，但被调用的频率很高（每个 anyOf/oneOf 分支都要过一遍）。
- **参数**：
  - `schema`：`Mapping[str, Any]`，待判断的 schema 节点，可能是 `{"type": "null"}`、也可能是带 `$ref` 的引用节点、甚至空字典。
  - `defs`：`Mapping[str, Any]`，`$defs` 定义表，用于先解引用再判断。
- **返回**：`bool`。解引用后的节点 `type` 恰好等于字符串 `"null"` 时返回 `True`；其它一切情况（包括 `type` 是列表如 `["string","null"]`、没有 `type` 键、`$ref` 解析失败）都返回 `False`。
- **内部流程**：直接调用 `_resolve(schema, defs, 0)` 得到解引用结果（注意这里**硬编码 depth=0**，不受外层递归深度影响），然后取 `.get("type")` 与字面量 `"null"` 比较并返回布尔值。没有循环、没有分支嵌套。
- **异常/边界**：不抛异常。对空字典、非法 `$ref`、`type` 写成列表等边界情况一律返回 `False`（也就是说 `["null"]` 这种写法不会被识别为空类型，会走普通联合类型分支）；`_resolve` 若因深度超限返回 `{"type": "…"}`，这里也返回 `False`。
- **同文件关系**：调用 `_resolve`；只被 `_type_label` 调用（在 `anyOf`/`oneOf` 分支处理中出现两次：一次用于过滤 null 分支，一次用于判断是否需要追加「空」标签）。

### `_resolve(schema: Mapping[str, Any], defs: Mapping[str, Any], depth: int) -> Mapping[str, Any]` （第 111 行）

- **作用**：跟随 `$ref` 指针，把引用节点替换成 `$defs` 里的真实定义节点，是「嵌套模型展开」的基础设施。因为工具的输入/输出 schema 经常把复杂对象抽成 `$defs` 里的命名定义（例如 `{"$ref": "#/$defs/SearchFilter"}`），不解析的话模型只能看到一个引用字符串，完全不知道里面有哪些字段。这个函数还承担**安全阀**职责：用 `MAX_REF_DEPTH` 限制跳转次数，并用 `depth` 限制递归深度，双保险防止自引用模型（A 引用 B、B 又引用 A）把提示词无限撑大。解析失败时它不会崩，而是返回一个「把引用名当类型名」的降级结果，保证渲染链路永远能产出文本。
- **参数**：
  - `schema`：`Mapping[str, Any]`，起始节点，通常含 `$ref`，也可能已经是普通节点（此时循环不执行，原样返回）。
  - `defs`：`Mapping[str, Any]`，`$defs` 名字 → 子 schema 的映射。若传入的不是 Mapping（例如 `None`），`defs.get(name)` 会抛 `AttributeError`；调用方 `_field_lines` 已保证它至少是 `{}`。
  - `depth`：`int`，来自上层递归的当前深度。`depth > MAX_REF_DEPTH` 时函数直接放弃解析、返回 `{"type": "…"}`，让上层输出「…」而不是继续展开。
- **返回**：`Mapping[str, Any]`。三种情况：① 没有任何 `$ref` → 原样返回传入的 `schema` 对象（**同一个对象引用，不是副本**）；② 成功解析到定义 → 返回 `$defs` 里那个目标节点（同样不是副本）；③ `$ref` 指向的名字在 `defs` 里找不到或不是 Mapping → 返回新建的 `{"type": 名字}`，把引用名当成类型名兜底；④ `depth` 超限 → 返回新建的 `{"type": "…"}`。
- **内部流程**：初始化 `node = schema`、`hops = 0`。进入 `while` 循环，条件为「`node` 是 Mapping」且「`node` 里有 `$ref`」且「`hops <= MAX_REF_DEPTH`」。循环体内：把 `node["$ref"]` 转成字符串，用 `ref.rsplit("/", 1)[-1]` 取出引用路径的最后一段作为定义名（这样 `#/$defs/Foo` 和 `#/definitions/Foo` 都能取到 `Foo`）；从 `defs` 取目标，若目标不是 Mapping 就立刻返回 `{"type": name}`；否则把 `node` 换成目标节点、`hops += 1`，继续下一轮。循环结束后再判断一次 `depth > MAX_REF_DEPTH`，是则返回 `{"type": "…"}`，否则返回 `node`。
- **异常/边界**：`defs` 不是 Mapping 时 `defs.get` 会抛 `AttributeError`（调用方保证不会发生）；`$ref` 值是 None 等异常类型时 `str()` 兜底不会崩。链式引用（A→B→C→D）最多跟随 `MAX_REF_DEPTH + 1 = 3` 跳，第 4 跳起因为 `hops` 已超过 2 而退出循环，此时返回的是**仍带 `$ref` 的中间节点**（不是占位符），上层 `_type_label` 会把它当成普通节点处理并最终落到「任意」；找不到定义时降级成 `{"type": "引用名"}`，因此渲染文本里可能出现英文定义名当类型，属于有意为之的降级而非 bug。深度超限时返回 `{"type": "…"}`，`_TYPE_LABELS` 查不到 `…`，`_type_label` 会原样输出「…」。
- **同文件关系**：不调用本文件其它函数（只用内建 `str`/`isinstance`）。被 `_type_label`、`_is_null`、`_constraints`、`_field_lines` 四处调用，是本模块被复用最广的底层函数。

### `_constraints(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> str` （第 131 行）

- **作用**：把 JSON Schema 里散落的**约束关键字**翻译成一句中文短语，让模型明确知道「这个字段最短 3 个字」「这个数值必须 ≤ 100」「这个数组最多 5 项」「这个字符串必须匹配某正则」。原始 schema 里这些约束只是关键字名，模型经常视而不见，于是写出越界参数导致工具调用失败；把它们集中渲染成一行可读文字，能显著减少无效调用和重试。函数只负责「翻译」，不负责判断是否满足约束（校验由别处负责），也不负责截断或默认值展示。
- **参数**：
  - `schema`：`Mapping[str, Any]`，字段的 schema 节点，函数会先解引用再读取约束关键字。
  - `defs`：`Mapping[str, Any]`，`$defs` 定义表，用于解引用。
- **返回**：`str`。若一个约束都没有，返回空字符串 `""`（调用方据此决定不加「约束:」前缀）；否则返回用中文逗号 `，` 连接的多段描述，例如 `最短 3 字，最长 20 字，须匹配 ^[a-z]+$`。各段格式固定为：`最短 N 字`、`最长 N 字`、`≥ N`、`> N`、`≤ N`、`< N`、`至少 N 项`、`最多 N 项`、`须匹配 <pattern>`。
- **内部流程**：先 `_resolve(schema, defs, 0)`（depth 固定为 0）拿到 `resolved`；初始化空列表 `parts`；然后按固定顺序依次检查九个关键字：`minLength`、`maxLength`、`minimum`、`exclusiveMinimum`、`maximum`、`exclusiveMaximum`、`minItems`、`maxItems`、`pattern`，每命中一个就往 `parts` 追加一条格式化字符串（值直接以 `f-string` 内插，不做类型校验）；最后 `"，".join(parts)` 返回。顺序固定保证了渲染文本的确定性。
- **异常/边界**：不主动抛异常。约束值为 `0` 或 `False` 时因为用 `in` 判断仍然会被渲染出来（不会被误判为缺失）。约束值类型非法（如 `minLength: "abc"`）会被原样内插成 `最短 abc 字`，不做校验。只识别上述九个关键字，`format`、`multipleOf`、`uniqueItems` 等其它约束不会被渲染（既不报错也不提示）。深度超限返回的 `{"type": "…"}` 里没有任何约束键，结果就是空字符串。
- **同文件关系**：调用 `_resolve`；只被 `_field_lines` 调用（第 190 行），用于拼装每个字段行的「约束:」部分。

### `_format_default(value: Any) -> str` （第 157 行）

- **作用**：把字段的 `default` 值格式化成一段短小、无歧义、适合放进提示词的文本。它存在的意义是：模型看到 `默认=3` 就知道可以不传 `top_k`，看到 `默认="all"` 就知道字符串要带引号（避免把值当字段名猜），看到 `默认={"mode": "fast"}` 就知道对象结构长什么样。同时它对长默认值做截断，防止某个工具把一大坨配置对象当默认值写进 schema，把单行提示词撑成几百字符。它是纯函数，不读全局状态，输入相同输出必相同。
- **参数**：
  - `value`：`Any`，任意默认值，可能来自 JSON Schema 的 `default` 关键字，实际类型常见为 `None`、`str`、`bool`、`int`、`float`、`list`、`dict`，理论上也可以是任意可被 `json.dumps` 序列化的值。
- **返回**：`str`，永不为空字符串。规则：`None` → `"null"`；字符串 → `json.dumps(value, ensure_ascii=False)`（因此带引号、中文不转义），空字符串特判为 `""`（即两个引号）；布尔 → `"true"` / `"false"`；`list`/`dict` → 用 `json.dumps(..., ensure_ascii=False, sort_keys=True)` 序列化（`sort_keys` 保证键顺序稳定，维持渲染确定性），长度 ≤ 60 时原样返回，超过 60 则取前 57 个字符加省略号 `…`；其它类型（int、float 等）→ `str(value)`。
- **内部流程**：一串 `if` 顺序判断：先判 `None`，再判 `str`，再判 `bool`，再判 `(list, dict)`，最后 `str(value)` 兜底。注意 `bool` 判断排在 `str` 之后但在 `list/dict` 之前，由于 `bool` 不是 `str` 也不是 `list/dict`，实际判断互不干扰；`int`/`float` 会落到最后的 `str(value)`（所以 `3.0` 渲染成 `3.0`，`True` 不会落到这里因为前面已拦截）。截断逻辑以**字符数**为准而非字节数。
- **异常/边界**：`value` 若含 `json.dumps` 无法序列化的对象（如自定义类实例、`set`），会抛 `TypeError`；循环引用会抛 `ValueError`。浮点 `nan`/`inf` 会被 `json.dumps` 渲染成 `NaN`/`Infinity`（非标准 JSON 但 Python 允许）。截断只发生在 `list`/`dict` 分支，超长字符串不会被截断（`json.dumps` 结果原样返回），这是本函数一个不对称的边界行为。
- **同文件关系**：不调用本文件其它函数（只用 `json.dumps` 与内建 `str`/`isinstance`）。只被 `_field_lines` 调用（第 187 行，在生成「可选, 默认=…」状态时使用）。

### `_field_lines(schema: Mapping[str, Any], *, limit: int = MAX_FIELD_LINES) -> list[str]` （第 170 行）

- **作用**：把一份 schema 的「属性表」渲染成「一行一个变量」的文本列表，是输入变量区和输出字段区共用的渲染引擎。每一行都包含五类信息：字段名、中文类型标签、必填/可选状态（可选时带默认值）、约束短语、字段说明——这正是模块开头承诺的「逐变量契约文本」的核心。它同时负责两个重要的一致性工作：一是无论字段是直接写在内联 `properties` 里还是通过 `$ref` 引用 `$defs`，都先统一解引用再渲染；二是强制行数上限，超出时追加一行说明并停止，避免某个超大 schema（例如动态生成的宽表参数）把提示词预算吃光。由于输入 schema 和输出 schema 都走它，两条链路的字段格式天然一致，不会漂移。
- **参数**：
  - `schema`：`Mapping[str, Any]`，要渲染的 schema 根节点。函数会从中读取 `$defs`（用于解引用）、`properties`（字段表）、`required`（必填名单）。若 `properties` 不是 Mapping 或为空，直接返回占位行。
  - `limit`：`int`，**仅限关键字传入**（`*` 之后的 keyword-only 参数），默认 `MAX_FIELD_LINES`（60）。表示本区块最多渲染多少个字段行（占位行与截断说明行不计入这个计数语义之外，见下）。
- **返回**：`list[str]`，每个元素是一行已经带四空格缩进的文本。字段行格式为 `    - 名称: 类型 (必填)` / `    - 名称: 类型 (可选, 默认=…)` / `    - 名称: 类型 (可选)`，随后按需拼接 ` 约束: <短语>` 和 ` — <说明>`。没有可渲染字段时返回单元素列表 `["    （无输入变量）"]`。触达 `limit` 时最后追加一行 `    …（其余 N 个变量从略）`。
- **内部流程**：① 取 `schema.get("$defs")`，若不是 Mapping 就替换成 `{}`（保证后续 `_resolve` 安全）；② 取 `schema.get("properties")`，若不是 Mapping 或为空 → 立即返回占位行；③ 用 `set(schema.get("required") or ())` 构造必填集合（`or ()` 同时兼容 `None` 和空列表）；④ 遍历 `properties.items()`：先把 `node` 规范化为 Mapping（不是则用 `{}`），调用 `_resolve(node, defs, 0)` 得到 `resolved`，调用 `_type_label(node, defs, 0)` 得到类型标签；⑤ 判定状态：字段名在 `required` 里 → `必填`，否则若 `resolved` 里有 `default` → `可选, 默认=<_format_default(...)>`，否则 → `可选`（注意默认值取自 `resolved`，即解引用之后的节点）；⑥ 调用 `_constraints(node, defs)` 得到约束短语；⑦ 取说明：优先 `node.get("description")`，为空则退到 `resolved.get("description")`，再 `str(...).strip()` 去掉首尾空白；⑧ 拼出 `    - 名称: 类型 (状态)` 基础行，按需追加约束与说明，`append` 进 `lines`；⑨ 每追加一行就检查 `len(lines) >= limit`，一旦触达就再追加一行截断说明并 `break`。
- **异常/边界**：不主动抛异常；`properties` 为 `None`、空字典、非 Mapping 一律走占位行分支。`required` 为 `None` 时用 `or ()` 兜底。字段 `node` 不是 Mapping 时被替换为 `{}`，该字段会渲染成「任意 (可选)」。说明文字里的换行不会被清理，可能让一行文本在终端里视觉上断行。截断说明中的剩余数量用 `len(properties) - len(lines) + 1` 计算，由于 `len(lines)` 已包含刚追加的当前字段，该数字比真正未渲染的字段数**多 1**（例如 100 个字段、limit=60 时会显示「其余 41 个变量从略」，实际未渲染 40 个），属于提示文案上的细微偏差，不影响渲染主体内容。`limit` 传 0 或负数时会在第一个字段后立刻截断。
- **同文件关系**：调用 `_resolve`、`_type_label`、`_constraints`、`_format_default`；被 `render_schema_block`（第 213 行，渲染懒加载工具的裸 schema）与 `render_tool_entry`（第 234、236 行，分别渲染输入变量与输出字段）调用。

### `_confirmation_note(spec: ToolSpec) -> str` （第 204 行）

- **作用**：根据工具的副作用类型，生成一句明确的**副作用与确认策略**说明，直接写进工具契约文本。这是本模块被设计出来要补上的三块缺失信息之一（输出字段、副作用、是否需要人工确认）——过去提示词完全没提这些，导致模型可能在没有人工确认钥匙的情况下「假装已授权」就去调用写工具。有了这句话，模型能清楚知道：只读工具可以放心直接调，写工具则必须先拿到该工具的人工确认钥匙，且不得自行假定已授权。文案是硬编码的两句，保证同一工具每次渲染完全一致。
- **参数**：
  - `spec`：`ToolSpec`，工具的契约对象，本函数只读取它的 `side_effect` 字段。
- **返回**：`str`。当 `spec.side_effect == "write"` 时返回 `"写操作：执行前必须持有该工具的人工确认钥匙，模型不得自行假定已授权"`；其它任何取值（`"read"`、`None`、未知字符串等）都返回 `"只读操作：免确认，可直接调用"`。
- **内部流程**：单条 `if` 判断 `spec.side_effect == "write"`，命中返回写操作文案，否则返回只读文案。没有循环、没有外部调用、没有状态。
- **异常/边界**：不抛异常；但依赖 `spec` 有 `side_effect` 属性，若传入缺少该属性的对象会抛 `AttributeError`（调用方 `render_tool_entry` 只会在已确认 `isinstance(spec, ToolSpec)` 的路径上传入）。任何非 `"write"` 的值都被**默认当成只读**，这是「失败开放」方向的选择：如果将来出现第三种副作用类型（如 `"delete"`），本函数会把它误报成只读，需要同步修改。
- **同文件关系**：不调用本文件其它函数；只被 `render_tool_entry` 调用（第 237 行，用于拼装 `  副作用与确认:` 行）。

### `render_schema_block(name: str, schema: Mapping[str, Any]) -> list[str]` （第 210 行）

- **作用**：渲染一个「没有活的 `ToolSpec`」的工具——也就是**懒加载（lazily resolved）工具**——的裸输入 schema。有些工具在注册阶段只登记了名字和 schema，尚未构造出完整的 `ToolSpec` 对象（或注册表里拿到的是个没有 `spec` 属性的壳），此时 `render_tool_entry` 走不通，就需要这条更轻量的路径：只输出工具名、`输入变量:` 标题和字段列表。它的输出比完整契约短，但至少保证模型能看到参数名和必填状态，不至于完全瞎猜。因为不涉及版本、超时、幂等、输出字段等信息，它也不做那些渲染。
- **参数**：
  - `name`：`str`，工具名，会原样渲染成 `- <name>` 这一行（不带 `@版本`，因为懒加载阶段没有版本信息）。
  - `schema`：`Mapping[str, Any]`，该工具的输入 schema。函数会先 `dict(schema)` 复制一份再交给 `_field_lines`，这是为了把任意 Mapping 实现（例如只读代理、`MappingProxyType`）转成标准 `dict`，保证下游 `.get` 行为可预期。
- **返回**：`list[str]`，至少三行：第一行 `- <工具名>`，第二行 `  输入变量:`，其后是 `_field_lines` 产出的字段行（无字段时为 `    （无输入变量）`）。
- **内部流程**：用列表字面量一次性构造结果：先放工具名行，再放「输入变量:」标题，然后用 `*_field_lines(dict(schema))` 把字段行**解包展开**拼进同一个列表。没有条件分支。
- **异常/边界**：不主动抛异常。`schema` 为 `None` 或非 Mapping 时，`dict(schema)` 会抛 `TypeError`（对 `None`）或 `ValueError`（对不可迭代对象），函数不做兜底。schema 里没有 `properties` 时正常返回占位行。**本函数不渲染输出字段、不渲染副作用/确认说明、不渲染推荐前置工具**，因此对懒加载工具的提示信息是不完整的，这是设计上的取舍。
- **同文件关系**：调用 `_field_lines`；本文件内部没有任何函数调用它，它是通过 `__all__` 导出的对外公共 API，供提示词组装链路在缺少 `ToolSpec` 时使用。

### `render_tool_entry(name: str, spec: ToolSpec) -> list[str]` （第 216 行）

- **作用**：把**一个完整工具**渲染成一段提示词行块，是模块开头那段契约模板的直接实现者。它输出的第一行是信息密度极高的头部：`- 工具名@版本 [写操作 · 需确认 · 超时 30s · 幂等 · 可并行 · 权限 a/b]`，让模型一眼看到工具的读写性质、是否需人工确认、超时预算、是否幂等、能否并行、以及所需权限；随后依次给出用途、使用规范（`guidance`，逐工具专门性约束）、输入变量、输出字段、副作用与确认、推荐前置工具。之所以要一次给全，是因为模型在原生 function-calling 下只能看到 `tools` 里的 schema，看不到这些运行语义；把它们放进 system 消息能显著减少「试探性调用」和「漏参数」。
- **参数**：
  - `name`：`str`，工具名，直接用于头部行（不参与排序，排序由上层 `render_tool_catalog` 负责）。
  - `spec`：`ToolSpec`，完整工具契约。函数读取的字段包括：`side_effect`（决定读写与确认标志）、`timeout_seconds`（用 `:g` 格式化成紧凑数字）、`idempotent`、`parallel_safe`、`permissions`（可迭代的权限名集合，用 `/` 连接）、`version`、`description`（用途）、`guidance`（用 `getattr` 带默认值读取）、`input_schema`、`output_schema`、`recommended_before_tools`。
- **返回**：`list[str]`，每个元素是一行已按约定缩进的文本。顺序固定为：头部行 → `  用途: ...` → （有 guidance 时）`  使用规范: ...` → `  输入变量:` → 输入字段行 → `  输出字段:` → 输出字段行 → `  副作用与确认: ...` → （有推荐前置时）`  推荐前置: a, b`。行数随工具复杂度变化，没有 guidance 和推荐前置时会少两行。
- **内部流程**：① 构造 `flags` 列表：第一项按 `side_effect == "write"` 在 `写操作`/`只读` 间二选一，第二项对应地在 `需确认`/`免确认` 间二选一，第三项是 `超时 {spec.timeout_seconds:g}s`（`:g` 让 `30.0` 显示为 `30`），第四项 `幂等`/`非幂等`，第五项 `可并行`/`不可并行`；② 若 `spec.permissions` 非空，追加 `权限 ` 加 `/` 连接的权限名；③ 用 `' · '.join(flags)` 生成头部行 `- {name}@{version} [...]`；④ 追加 `  用途: {spec.description}`；⑤ 用 `getattr(spec, "guidance", "") or ""` 兼容性地取 guidance（老版本 `ToolSpec` 没有这个字段时返回空串），非空才追加 `  使用规范:` 行；⑥ 追加 `  输入变量:` 标题，再用 `lines.extend(_field_lines(spec.input_schema))` 展开输入字段；⑦ 追加 `  输出字段:` 标题，再用 `lines.extend(_field_lines(spec.output_schema))` 展开输出字段；⑧ 追加 `  副作用与确认: {_confirmation_note(spec)}`；⑨ 若 `spec.recommended_before_tools` 非空，追加 `  推荐前置: ` 加 `", "` 连接的工具名列表。全程无时间戳、无随机数、无集合遍历，保证确定性。
- **异常/边界**：`spec` 不是 `ToolSpec` 或缺少 `description`/`input_schema` 等属性时会抛 `AttributeError`；`timeout_seconds` 不是数字时 `:g` 格式化会抛 `ValueError`/`TypeError`。`permissions` 为空列表/空元组时不会追加权限标志（用真值判断）。`guidance` 用 `getattr` + `or ""` 双重兜底，因此该属性为 `None` 或不存在都安全。**输出字段区同样使用 `_field_lines`**，而 `_field_lines` 在找不到 `properties` 时会输出 `    （无输入变量）`，所以当某工具的输出 schema 是 `$ref` 形式或只有 `additionalProperties` 时，输出区可能显示「（无输入变量）」这句略显错位的占位文案——这是复用同一渲染器的副作用，不影响其余信息。`recommended_before_tools` 只做展示，不校验这些工具是否真的注册过。
- **同文件关系**：调用 `_field_lines`（两次，输入与输出）与 `_confirmation_note`；只被 `render_tool_catalog` 调用（第 256 行）。

### `render_tool_catalog(registrations: Mapping[str, tuple[Any, int]]) -> list[str]` （第 243 行）

- **作用**：把整个**注册表**（而不只是单个工具）渲染成提示词行列表，是「全量工具目录」的组装器。它做三件关键事：一是**按工具名排序**遍历，从而让同一批注册表无论插入顺序如何都渲染出逐字节相同的文本（这是前缀缓存复用的前提）；二是**过滤掉没有有效 `ToolSpec` 的条目**，避免注册表里的占位壳或懒加载项产生残缺块；三是在每个工具块之间插入一个空字符串作为分隔行，让模型（和人）能清楚分辨工具边界。它是 `render_tool_entry` 的批量驱动者，也是 `render_tool_catalog_text` 的唯一数据来源。
- **参数**：
  - `registrations`：`Mapping[str, tuple[Any, int]]`，工具注册表，键是工具名，值是二元组 `(tool, 计数/版本序号)`。元组的第二个元素在本函数中**被完全忽略**（解包成 `_`），只用来保持与注册表既有结构兼容。`tool` 可以是任何对象，函数用 `getattr(tool, "spec", None)` 探测它有没有 `spec` 属性。
- **返回**：`list[str]`，全部工具块的行按顺序拼接而成，块与块之间夹一个 `""` 空行；注册表为空或所有条目都无效时返回空列表 `[]`（**不会**返回占位文案，占位由 `render_tool_catalog_text` 负责）。
- **内部流程**：① 初始化 `lines = []`；② `for name in sorted(registrations)` 按工具名字典序遍历；③ 从 `registrations[name]` 解包出 `tool, _`；④ `spec = getattr(tool, "spec", None)`；⑤ 若 `spec` 不是 `ToolSpec` 实例（包括 `None`、其它类型）→ `continue` 跳过该工具；⑥ 若 `lines` 非空（说明前面已有内容）→ `append("")` 插入空行分隔；⑦ `lines.extend(render_tool_entry(name, spec))` 追加该工具的完整块。循环结束返回 `lines`。
- **异常/边界**：`registrations` 不是 Mapping 时 `sorted()` 或下标取值会抛 `TypeError`/`KeyError`；元组元素个数不是 2 时解包会抛 `ValueError`。工具缺 `spec` 属性或 `spec` 类型不对会被**静默跳过**，不报错也不提示，因此如果某工具没出现在提示词里，排查方向应从这里开始。`sorted()` 对工具名做的是普通字符串排序（区分大小写、按码位），不做自然序或本地化排序。空注册表返回 `[]`。
- **同文件关系**：调用 `render_tool_entry`；被 `render_tool_catalog_text` 调用（第 265 行）。

### `render_tool_catalog_text(registrations: Mapping[str, tuple[Any, int]]) -> str` （第 260 行）

- **作用**：`render_tool_catalog` 的**字符串化包装**，产出可以直接塞进 system 消息（或 ReAct 指令块）的一整段文本。它存在的意义是给调用方一个「一行调用拿到成品字符串」的便捷入口，免去调用方自己处理 `"\n".join` 和空注册表的情况；同时它定义了空目录时的固定占位文案 `(no tools are registered)`，让提示词里始终有明确内容，而不是出现一段空白（空白容易让模型误以为提示词被截断）。由于它只是拼装，本身不引入任何顺序或格式变化，确定性由下游 `render_tool_catalog` 保证。
- **参数**：
  - `registrations`：`Mapping[str, tuple[Any, int]]`，与 `render_tool_catalog` 完全相同的工具注册表（键为工具名，值为 `(tool, 序号)` 二元组），原样透传。
- **返回**：`str`。当 `render_tool_catalog` 返回非空列表时，返回用换行符 `"\n"` 连接后的整段文本（**末尾没有额外换行**）；当返回空列表时，返回固定字符串 `"(no tools are registered)"`。
- **内部流程**：先调用 `render_tool_catalog(registrations)` 得到 `lines`，再用条件表达式 `"\n".join(lines) if lines else "(no tools are registered)"` 返回。没有循环、没有其它分支。
- **异常/边界**：异常全部来自下游 `render_tool_catalog`（如注册表结构非法导致的 `TypeError`/`ValueError`），本函数不做捕获也不做额外校验。空注册表与「所有工具都被过滤掉」这两种情况**返回相同占位文案**，调用方无法从返回值区分二者。返回文本不包含结尾换行，拼接时由调用方自行决定分隔。
- **同文件关系**：调用 `render_tool_catalog`；本文件内部没有函数调用它，它通过 `__all__` 导出，是提示词组装链路最外层使用的公共入口（ReAct 文本协议与原生 function-calling 两条链路共用同一份输出）。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_type_label` | 把 JSON Schema 节点（含 enum/const/anyOf/oneOf/数组/对象/$ref）翻译成一句中文类型标签，超深时返回「…」。 |
| `_is_null` | 判断解引用后的 schema 节点是否为 null 类型，供联合类型分支识别可空性。 |
| `_resolve` | 跟随 `$ref` 到 `$defs` 取出真实定义节点，解析失败降级为 `{"type": 引用名}`，深度超限降级为 `{"type": "…"}`。 |
| `_constraints` | 把 minLength/maxLength/minimum/maxItems/pattern 等约束关键字翻译成一句中文短语，无约束返回空串。 |
| `_format_default` | 把字段默认值格式化成短文本（None→null、字符串带引号、布尔 true/false、长对象截断到 60 字符）。 |
| `_field_lines` | 把 schema 的 properties 渲染成「一行一个变量」的列表，含类型/必填/默认/约束/说明，并强制行数上限。 |
| `_confirmation_note` | 按 side_effect 是否为 write，生成写操作需人工确认或只读免确认的说明文案。 |
| `render_schema_block` | 为没有完整 ToolSpec 的懒加载工具渲染「工具名 + 输入变量 + 字段行」的轻量块。 |
| `render_tool_entry` | 把单个完整 ToolSpec 渲染成含标志头、用途、规范、输入、输出、副作用、推荐前置的完整契约块。 |
| `render_tool_catalog` | 按工具名排序遍历注册表，过滤无有效 spec 的条目，用空行分隔拼出全量工具目录行列表。 |
| `render_tool_catalog_text` | 把工具目录行列表连接成一整段字符串，空目录时返回固定占位文案。 |
