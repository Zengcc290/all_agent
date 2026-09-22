# tool/multi_recall.py

## 一、这个文件是干什么的

本文件实现的是「多路混合召回」这一整套能力：它把一个用户的检索问句先用问句分解器拆成若干条更短的探针查询（原句永远排在第一条），然后让每一条子查询各自去跑「向量检索 + FTS5 关键词检索」两路，把所有子查询产出的名次列表集中交给 RRF（Reciprocal Rank Fusion）做融合；图路同理，每条子查询各做一次 seed + expand，得到的路径按 `effective` 权重合并去重。这样做的原因是：型号、编号这类精确词在向量空间里区分度很差，而转述表达又只有向量路能召回；「小红的亲戚是谁」这类关系型问句单靠原句往往一条都命中不了，必须先拆出实体名和关系词才召回得到。文件里主要包含：两个模块级私有辅助函数（懒加载默认 pipeline、清洗查询文本）、一个 `QueryDecomposer` 协议及其两个实现（LLM 版与空降级版）、两个核心召回函数 `hybrid_recall_multi` 与 `graph_recall_multi`、一个工厂函数 `build_decomposer`、三个 Pydantic 模型（输入、输出、以及继承 `BaseTool` 的工具类）、以及 `create_tool` 工厂。它在项目运行中的入口是 `MultiRecallTool.execute`：Agent 通过工具注册表以 `knowledge.multi_recall` 这个名字调用它，`mode` 参数决定跑 hybrid、graph 还是两者都跑，最终把命中片段、实体列表、关系路径以及可直接喂给模型的路径上下文一次性返回。设计上有一条硬约束贯穿全文件：任一子查询或分解步骤失败都不能让整体失败，图后端抖动时只丢路径、保留向量证据，分解失败时原句照跑。

## 二、函数与类逐条详解

### `_default_pipeline() -> Any` （第 55 行）
- **作用**：构造并返回一个默认的记忆检索 pipeline 对象，作为 `MultiRecallTool` 在没有被外部注入 pipeline 时的兜底数据源。它存在的唯一理由是「延迟导入」：模块顶部的注释明确写了 `_memory` 会拉起 `memory.rag` 这一整条重依赖链，如果在模块顶层直接 `from ._memory import ...`，那么任何只想 import 本模块拿到 `MAX_SUB_QUERIES` 或模型类的地方都会被强制加载 rag 子系统，既拖慢启动也可能引入循环导入。因此它把 import 放进函数体内部，只有真正第一次需要 pipeline 时才付这个代价。它会被 `MultiRecallTool.pipeline` 这个 property 在 `self._pipeline is None` 时调用，也就是工具被实例化但没有传 pipeline、并且第一次执行检索的时候。
- **参数**：无参数。
- **返回**：返回 `build_default_pipeline()` 的返回值，类型标注为 `Any`（真实类型是项目里默认的 RAG/Memory pipeline 对象，本文件只用它调用 `document_repo()`、`retrieve()`、`graph.retrieve()`，所以不需要知道具体类型）。无论成功与否都返回同一个调用结果，没有其他分支。
- **内部流程**：第一步在函数体内执行 `from ._memory import build_default_pipeline`，把导入推迟到调用时刻；第二步直接 `return build_default_pipeline()`，把构造工作完全交给被导入的工厂函数。函数体内没有缓存、没有 try/except、没有任何判断分支，重复调用会重复构造。
- **异常/边界**：函数自身不做任何异常处理。如果 `_memory` 模块不存在、`build_default_pipeline` 符号缺失，会抛出 `ImportError`；如果工厂函数内部因为配置缺失（比如没有可用的数据库、没有 embedding 配置）而失败，异常会原样向上抛给 `MultiRecallTool.pipeline`，最终冒泡到 `execute`，导致本次工具调用失败。这是有意的：pipeline 是召回的根依赖，它坏了就没有任何降级路径可言。
- **同文件关系**：被 `MultiRecallTool.pipeline`（property，第 329-333 行）调用。它本身不调用本文件里的任何其他函数。

### `_clean_query_text(value: Any, *, max_length: int = 500) -> str` （第 63 行）
- **作用**：把任意输入规范成一条干净的查询字符串：先判断是不是字符串，然后把内部所有连续空白（空格、制表符、换行）压缩成单个空格，再按 `max_length` 截断，最后去掉首尾空白。文档字符串说明它的行为是对齐 `memory.ids._clean_text`，也就是让本文件生成的子查询与记忆子系统内部存文本时的清洗规则保持一致，避免出现「子查询里带换行导致 FTS5 分词异常」这类问题。它最主要的调用者是 `LLMQueryDecomposer._normalize`，用来清洗 LLM 返回的每一条 `sub_queries` 元素——因为 LLM 输出里混入换行、多余空格、前后空白的概率非常高，不清洗就会污染后续的向量检索和关键词检索。
- **参数**：`value`（`Any`）：待清洗的原始值，实际调用中通常是 LLM 返回的列表元素，可能是字符串也可能不是；非字符串一律视为无效输入。`max_length`（`int`，关键字参数，默认 `500`）：截断长度上限，注释与调用点都使用默认值，约束是正整数；由于清洗逻辑里没有对负数或零做保护，若传入 `0` 会得到空串，传入负数会得到除末尾若干字符外的奇怪切片结果，因此调用方不应传非正数。
- **返回**：返回 `str`。`value` 不是 `str` 时返回空字符串 `""`；是字符串时返回压缩空白并截断后的结果（可能为空串，例如输入全是空白）。永远不会返回 `None`。
- **内部流程**：第一步 `isinstance(value, str)` 判断类型，不是字符串直接 `return ""` 提前退出；第二步 `value.split()` 按任意空白切分成词列表（这一步同时丢弃了所有空片段）；第三步 `" ".join(...)` 用单个空格重新拼接，从而完成「空白压缩」；第四步 `[:max_length]` 做长度截断；第五步 `.strip()` 去掉截断后可能残留在首尾的空白并返回。
- **异常/边界**：无特殊处理，函数内部不会主动抛异常。空字符串、纯空白字符串、`None`、数字、列表等输入都会安全地得到 `""`（`None` 和数字走类型判断分支，空白串走 split/join 后得到空串）。不校验 `max_length` 的合法性，见上面的参数说明。
- **同文件关系**：被 `LLMQueryDecomposer._normalize`（第 142-146 行的列表推导式里调用了两次：一次作为过滤条件、一次作为结果值）调用。它不调用本文件里的任何其他函数。

### `class QueryDecomposer(Protocol)` （第 71 行）
- **作用**：定义「问句分解器」的统一接口契约，任何想把一句话拆成多条待测查询的对象只要实现 `decompose` 方法就满足这个协议。它用的是 `typing.Protocol`，因此是结构化子类型（鸭子类型）而非名义继承：`NullQueryDecomposer` 和 `LLMQueryDecomposer` 都没有显式继承它，但在类型检查层面都被视为合法的 `QueryDecomposer`。文档字符串说明它与项目里的 `KnowledgeExtractor` 同构，也就是沿用了同一个「协议 + 真实实现 + 空实现」的既有模式。它被用作 `MultiRecallTool.__init__` 的 `decomposer` 参数类型、`MultiRecallTool.decomposer` property 的返回类型、以及 `build_decomposer()` 的返回类型，从而让工具层不必知道到底用的是 LLM 分解器还是降级分解器。类本身在运行时没有任何可实例化的行为（`Protocol` 不可直接实例化），只承担类型与文档职责。
- **参数**：类无构造参数（不定义 `__init__`）。
- **返回**：不适用（类定义本身不返回值）。
- **内部流程**：类体只包含一段文档字符串和下面那个 `decompose` 方法签名（方法体是 `...`，即省略号占位，没有实现）。作为 `Protocol`，类型检查器会据此做结构化匹配。
- **异常/边界**：无特殊处理。注意它在运行时无法被实例化（`TypeError`），也不能用 `isinstance` 检查普通实现类（除非该协议被标记为 `runtime_checkable`，本文件没有这样做）。
- **同文件关系**：它的方法签名被 `NullQueryDecomposer.decompose` 与 `LLMQueryDecomposer.decompose` 实现；`MultiRecallTool.__init__`、`MultiRecallTool.decomposer`、`build_decomposer` 在类型标注中引用它。它不调用本文件里的任何函数。

### `QueryDecomposer.decompose(self, query: str) -> list[str]` （第 74 行）
- **作用**：协议声明的方法，规定「输入一条问句、输出若干条待测子查询」这一行为。它没有任何实现代码（函数体是 `...`），存在的意义是给所有实现类一个统一的签名约束，让 `MultiRecallTool.execute` 可以无条件地写 `self.decomposer.decompose(arguments.query)` 而不管背后是 LLM 还是降级实现。因为 `Protocol` 方法不参与运行时调用，这个方法实际上永远不会被真正执行。
- **参数**：`self`：协议实例（占位）。`query`（`str`）：待分解的原始问句。协议本身不对空串、超长串等做任何约定，具体约束由实现类各自决定。
- **返回**：声明返回 `list[str]`，即子查询字符串列表；协议不规定列表是否可空、顺序如何、原句是否必须在首位，这些语义约定写在实现类的文档与代码里。
- **内部流程**：无实现，只有 `...`。类型检查器读取该签名做结构化匹配，运行时若真的调用协议上的该方法会直接返回 `None`（因为函数体是省略号表达式）。
- **异常/边界**：无特殊处理（协议不定义异常语义）。
- **同文件关系**：由 `NullQueryDecomposer.decompose`（第 80 行）与 `LLMQueryDecomposer.decompose`（第 115 行）实现；被 `MultiRecallTool.execute` 通过 `self.decomposer.decompose(...)` 间接调用。

### `class NullQueryDecomposer` （第 77 行）
- **作用**：LLM 不可用时的安全降级实现，文档字符串把它定位为「永不因分解失败而答不出来」。它的行为极其简单：把原句原样当成唯一的子查询返回，于是多路召回退化成语义等价于单路召回，功能上不会报错、不会丢信息。它有两个使用场景：一是 `build_decomposer()` 在拿不到可用 API Key 或初始化 LLM 客户端失败时返回它；二是 `MultiRecallTool.execute` 在调用方显式传 `decompose=False` 时临时 `NullQueryDecomposer().decompose(arguments.query)` 来统一走「不分解」的路径。这个类没有状态，可以被随意反复实例化。
- **参数**：类无构造参数（不定义 `__init__`，因此实例化不需要任何实参）。
- **返回**：不适用（类定义本身不返回值）。
- **内部流程**：类体只有文档字符串和下面一个 `decompose` 方法，没有任何类属性、没有缓存、没有配置读取。
- **异常/边界**：无特殊处理，实例化永远不会失败。
- **同文件关系**：实现 `QueryDecomposer` 协议；被 `build_decomposer` 与 `MultiRecallTool.execute` 实例化并调用其 `decompose`。

### `NullQueryDecomposer.decompose(self, query: str) -> list[str]` （第 80 行）
- **作用**：把输入问句包装成单元素列表返回，是「不做任何分解」的正式实现。当调用方关闭了分解功能、或者环境里没有可用的 LLM 时，多路召回仍然需要一个合法的子查询列表，这个方法就负责提供它。它同时承担输入净化职责：只有非空字符串才会被返回，`None`、数字、纯空白串都会得到空列表，从而让上层能用「空列表」这一信号判断输入无效。
- **参数**：`self`：`NullQueryDecomposer` 实例。`query`（`str`）：原始问句；实际实现里做了 `isinstance(query, str)` 检查，因此即使调用方传进来非字符串（类型标注之外的输入）也不会崩溃。
- **返回**：返回 `list[str]`。当 `query` 是字符串且 `query.strip()` 非空时返回 `[query.strip()]`（注意返回的是去首尾空白后的版本，而不是原始 `query`）；否则返回空列表 `[]`。
- **内部流程**：单行条件表达式：`return [query.strip()] if isinstance(query, str) and query.strip() else []`。先判断类型，再判断去空白后是否为空，两个条件都满足才构造单元素列表。注意这里只做 `strip()`，不做内部空白压缩、不做长度截断，所以与 `_clean_query_text` 的清洗强度不同。
- **异常/边界**：无特殊处理，不抛异常。空串与纯空白串返回 `[]`；非字符串返回 `[]`。`MultiRecallTool.execute` 在拿到空列表时会把 `queries` 兜底重置为 `[arguments.query]`，因此这里返回空列表不会导致后续检索崩溃。
- **同文件关系**：实现 `QueryDecomposer.decompose`；被 `MultiRecallTool.execute` 调用。它调用了 Python 内置的 `strip`，不调用本文件里的任何其他函数。

### `class LLMQueryDecomposer` （第 84 行）
- **作用**：真正干活的问句分解器，通过 chat 客户端（`complete` 回调）让 LLM 把一条问句拆成若干条更短的探针查询。它是整个多路召回「多路」的来源：没有它，多路融合就退化成单查询两路融合。它的类属性 `SYSTEM_PROMPT` 承载了完整的中文提示词，明确要求模型只输出一个合法 JSON 对象（不要 Markdown、不要解释）、`sub_queries` 第一条必须是原句、最多 6 条、要去重、不能添加原文没有的实体或数字，并给出关系词/别称/上位词的变体示例（「小红的亲戚是谁」→「小红」「小红 亲戚」「小红 亲属 关系」）。类属性 `MAX_SUB_QUERIES` 直接绑定模块级常量 `MAX_SUB_QUERIES`（值为 6），使实例方法可以通过 `self.MAX_SUB_QUERIES` 访问这个上限。它由 `build_decomposer()` 在 API Key 可用时构造，并被 `MultiRecallTool.decomposer` 属性持有。
- **参数**：类本身不接收参数；构造参数见下面的 `__init__`。
- **返回**：不适用（类定义本身不返回值）。
- **内部流程**：类体依次定义 `SYSTEM_PROMPT` 字符串常量、`MAX_SUB_QUERIES = MAX_SUB_QUERIES` 类属性（右侧引用的是模块级同名常量，因此是 6），然后定义 `__init__`、`decompose`、`_normalize` 三个方法。类体自身没有执行副作用。
- **异常/边界**：无特殊处理。提示词里明确禁止额外文字，但实现层并不依赖模型守规矩——`_normalize` 会做容错。
- **同文件关系**：实现 `QueryDecomposer` 协议；被 `build_decomposer` 构造；其 `decompose` 调用本文件的 `_clean_query_text`（经由 `_normalize`）以及外部导入的 `response_content`、`parse_json_object`。

### `LLMQueryDecomposer.__init__(self, complete: Callable[..., Any], *, model: str | None = None, timeout: float = 60.0) -> None` （第 102 行）
- **作用**：初始化分解器，把「怎么调模型」这件事以回调形式注入进来，从而让本类不依赖任何具体的 LLM 客户端实现（便于测试时塞入假回调）。同时它做了一次前置校验：`complete` 必须可调用，否则立刻抛 `TypeError`，把配置错误暴露在构造阶段而不是等到检索时才莫名其妙地失败。`model` 与 `timeout` 会原样保存，之后在 `decompose` 里透传给 `complete`。
- **参数**：`self`：实例本身。`complete`（`Callable[..., Any]`，位置参数）：调用 chat 客户端的可调用对象，签名需接受 `messages` 位置参数以及 `model`、`temperature`、`timeout`、`stream` 关键字参数；`build_decomposer` 传入的是 `LLM(client).complete`。约束是必须 callable，否则抛 `TypeError("complete must be callable")`。`model`（`str | None`，关键字参数，默认 `None`）：模型名，`None` 表示由底层客户端决定默认模型；`build_decomposer` 会显式传入 `profile.default_model`。`timeout`（`float`，关键字参数，默认 `60.0`）：单次 LLM 调用超时秒数，直接透传给 `complete`，本类不做额外的时间控制；没有做正数校验。
- **返回**：返回 `None`（构造函数）。
- **内部流程**：第一步用 `callable(complete)` 做检查，不通过就 `raise TypeError("complete must be callable")`；第二步把 `complete` 赋给 `self.complete`；第三步把 `model` 赋给 `self.model`；第四步把 `timeout` 赋给 `self.timeout`。没有其他副作用，也没有注册任何资源。
- **异常/边界**：`complete` 不可调用时抛 `TypeError`，消息固定为 `"complete must be callable"`。对 `model` 与 `timeout` 不做任何合法性校验（`timeout` 传 `0` 或负数会原样传给底层客户端，由底层决定行为）。
- **同文件关系**：被 `build_decomposer`（第 263 行）调用，传入 `client.complete` 与 `model=profile.default_model`；它的属性随后被 `LLMQueryDecomposer.decompose` 使用。它不调用本文件里的任何函数。

### `LLMQueryDecomposer.decompose(self, query: str) -> list[str]` （第 115 行）
- **作用**：执行一次完整的「LLM 分解」流程：校验输入、组装 system + user 两条消息、以 `temperature=0.0`、`stream=False`、指定 `timeout` 调用 chat 客户端、从响应里抽文本、解析 JSON、最后交给 `_normalize` 做规范化。它最重要的设计点是那一段宽泛的 `except Exception`：任何环节（网络、超时、返回格式错误、JSON 解析失败、字段缺失）出问题都只记录为「返回原句」，绝不向上抛异常。这与模块顶部的核心承诺一致——分解失败不能导致答不出来，只能导致多路退化成单路。
- **参数**：`self`：`LLMQueryDecomposer` 实例，提供 `complete`、`model`、`timeout`、`SYSTEM_PROMPT`。`query`（`str`）：原始问句；实现里会先判断类型和 `strip()` 后是否为空。
- **返回**：返回 `list[str]`。三种情况：`query` 不是字符串或去空白后为空时返回 `[]`；LLM 调用或解析过程中抛出任何异常时返回 `[query]`（注意是未经 strip 的原始 `query`）；正常解析时返回 `self._normalize(query, value)` 的结果，其首元素永远是原句 `query`，长度被限制在 `MAX_SUB_QUERIES`（6）以内。
- **内部流程**：第一步 `if not isinstance(query, str) or not query.strip(): return []` 做输入守卫。第二步构造 `messages` 列表：第一条是 `{"role": "system", "content": self.SYSTEM_PROMPT}`，第二条是 `{"role": "user", "content": json.dumps({"query": query}, ensure_ascii=False)}`——用 JSON 包一层而不是裸拼字符串，`ensure_ascii=False` 保证中文不被转义成 `\uXXXX`。第三步进入 `try`：调用 `self.complete(messages, model=self.model, temperature=0.0, timeout=self.timeout, stream=False)` 拿到 `response`；用 `response_content(response)` 抽出文本 `raw`；用 `parse_json_object(raw)` 解析成 Python 字典 `value`。第四步 `except Exception` 分支直接 `return [query]`，注释明确说明这是有意的降级。第五步 `return self._normalize(query, value)`。`temperature=0.0` 是为了让分解结果稳定可复现。
- **异常/边界**：函数对外不抛异常。`try` 块内的一切异常（超时、连接失败、鉴权失败、响应体不是预期结构、`raw` 不是合法 JSON 对象等）都被 `except Exception` 吞掉并降级为返回原句，代价是分解能力静默丢失且这里不写日志（日志由 `build_decomposer` 在构造阶段负责）。空输入返回 `[]`，由 `MultiRecallTool.execute` 再兜底成 `[arguments.query]`。
- **同文件关系**：调用 `self._normalize`（第 136 行，本文件内的方法）；`_normalize` 内部又调用模块级 `_clean_query_text`（第 63 行）。它被 `MultiRecallTool.execute` 通过 `self.decomposer.decompose(...)` 调用。它使用的 `response_content` 与 `parse_json_object` 来自 `memory.rag.knowledge` 的导入。

### `LLMQueryDecomposer._normalize(self, query: str, value: dict[str, Any]) -> list[str]` （第 136 行）
- **作用**：把 LLM 解析出来的 JSON 对象规范化成最终的子查询列表。它承担三件事：把原句强行放在第一位（文档字符串写明「原句永远第一条」）、对候选子查询做清洗与过滤、去重并截断到 `MAX_SUB_QUERIES`。之所以需要它，是因为模型输出不可信：可能缺字段、字段类型不对、含空串或纯空白项、含与原句重复的项、条数超过 6 条，这些都必须在这里被抹平，否则会污染后续的向量与关键词检索（例如把空串送给 FTS5 查询）。
- **参数**：`self`：提供类属性 `MAX_SUB_QUERIES`（值 6）。`query`（`str`）：原始问句，会被无条件放在返回列表首位。`value`（`dict[str, Any]`）：`decompose` 里 `parse_json_object` 的解析结果，期望是形如 `{"sub_queries": [...]}` 的字典；实现里仍然额外做了 `isinstance(value, dict)` 判断，说明调用方不保证类型绝对正确。
- **返回**：返回 `list[str]`。若 `value` 不是字典、或 `value["sub_queries"]` 不是列表，返回 `[query]`（只保原句）。否则返回 `[query, *dict.fromkeys(cleaned)][: self.MAX_SUB_QUERIES]`，即原句在首位、后面跟清洗去重后的子查询、整体最多 6 条。注意即使 `cleaned` 为空，返回的也是 `[query]` 而不是空列表，因此调用方永远能拿到至少一条可执行的查询。
- **内部流程**：第一步 `items = value.get("sub_queries") if isinstance(value, dict) else None` 安全取值；第二步 `if not isinstance(items, list): return [query]` 做类型守卫。第三步列表推导式生成 `cleaned`：遍历 `items`，对每一项先判断 `isinstance(item, str) and _clean_query_text(item)`（即必须是字符串且清洗后非空）作为过滤条件，满足时取 `_clean_query_text(item)` 作为值——注意 `_clean_query_text` 在这里被调用了两次，属于为了写法紧凑而付出的重复计算。第四步 `dict.fromkeys(cleaned)` 利用字典键唯一性做保序去重（保留首次出现的顺序，这是 Python 3.7+ 字典有序性带来的技巧），再与原句一起展开成列表。第五步用 `[: self.MAX_SUB_QUERIES]` 截断到最多 6 条后返回。
- **异常/边界**：无特殊处理，不抛异常。`value` 为 `None`、字符串、列表、缺 `sub_queries` 键、`sub_queries` 是字符串而非列表等情况统统返回 `[query]`；列表里混入非字符串或空白项会被静默丢弃；重复项会被 `dict.fromkeys` 去重。一个细节：原句 `query` 本身没有参与去重，所以如果模型把原句也放进 `sub_queries`，返回列表里会出现两次原句，只是被截断逻辑限制在 6 条之内。截断发生在原句并入之后，理论上极端情况下若原句与大量子查询混排，仍不会挤掉原句，因为原句恒定在索引 0。
- **同文件关系**：被 `LLMQueryDecomposer.decompose`（第 134 行）调用；它自己调用模块级 `_clean_query_text`（第 63 行）。它不调用本文件里的其他函数。

### `hybrid_recall_multi(pipeline: Any, queries: list[str], *, limit: int = RAG_RETRIEVE_LIMIT, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> HybridRecallResult` （第 150 行）
- **作用**：多路混合召回的文本侧主体，文档字符串概括为「F3：每条子查询各跑向量+关键词路，N 路 rank 列表一起丢给 RRF 融合」。它接收已经分解好的子查询列表，为每条子查询分别取向量命中与关键词命中，把两者的 chunk_id 名次列表依次追加进一个大列表，最后统一交给 `fuse_hits` 做 RRF 融合。它的关键设计有两点：一是「单查询短路」——只有一条查询时直接委托给单查询版本的 `hybrid_recall`，避免为等价场景重复实现逻辑；二是「双降级」——当混合检索总开关关闭或拿不到文档仓库时，退化成按顺序对每条子查询调用 `pipeline.retrieve` 并用 `seen` 集合按 `memory_id` 去重，保证至少能返回去重后的向量结果而不是空。
- **参数**：`pipeline`（`Any`）：记忆检索 pipeline，需要提供 `document_repo()` 与 `retrieve(query, limit=, threshold=, metadata=)` 两个能力。`queries`（`list[str]`）：候选子查询列表，函数内部会先过滤掉非字符串与纯空白项；预期原句在首位但不强制。`limit`（`int`，关键字参数，默认 `RAG_RETRIEVE_LIMIT`）：最终返回条数上限；内部两路召回时用的是 `limit * 2`，先多召回再融合截断。`threshold`（`float | None`，关键字参数，默认 `None`）：相似度阈值，原样透传给向量路与降级路径，`None` 表示不设阈值。`metadata`（`Mapping[str, Any] | None`，关键字参数，默认 `None`）：检索过滤条件（例如按文档、按标签过滤），原样透传给向量路与降级路径；注意关键词路 `repository.search_keywords` 没有收到这个参数。
- **返回**：返回 `HybridRecallResult`。四种情况：过滤后 `queries` 为空时返回空的 `HybridRecallResult()`；只剩一条查询时返回 `hybrid_recall(...)` 的结果（单查询版本对象）；混合检索不可用时返回 `HybridRecallResult(chunks=chunks[:limit])`，其中 `chunks` 是各子查询 `pipeline.retrieve` 结果按 `memory_id` 去重后的拼接列表，`note` 与 `vector_available` 保持默认；正常多路融合时返回 `HybridRecallResult(chunks=chunks, note=note, vector_available=not note)`，`chunks` 由 `fuse_hits` 产出，`note` 是第一条非空的向量路降级说明。
- **内部流程**：第一步列表推导式过滤 `queries`，只保留非空字符串。第二步空列表守卫，直接返回空结果。第三步单查询短路：在函数体内 `from .hybrid_recall import hybrid_recall` 做延迟导入（避免模块顶层循环依赖），然后带全部参数转发。第四步取 `repository = pipeline.document_repo()`。第五步判断 `if not hybrid_enabled() or repository is None`，进入降级分支：初始化 `chunks` 与 `seen`，双重循环遍历每条查询的 `pipeline.retrieve(...)` 结果，用 `chunk.memory_id in seen` 跳过重复项、否则加入 `seen` 并追加到 `chunks`，最后返回 `chunks[:limit]`。第六步正常分支：初始化 `rank_lists`（名次列表的列表）、`vector_scores`（chunk_id→向量分）、`keyword_scores`（chunk_id→关键词分）、`note` 空串。第七步对每条查询循环：调用 `vector_hits(pipeline, query, limit=limit * 2, threshold=threshold, metadata=metadata)` 得到 `hits` 与 `current_note`，用 `note = note or current_note` 只保留第一条非空说明；再调用 `repository.search_keywords(query, limit=limit * 2)` 得到关键词命中并把 score 转成 `float`；然后把向量路的 chunk_id 列表与关键词路的 chunk_id 列表各自 `append` 进 `rank_lists`（因此 N 条子查询会产生 2N 个名次列表），并用 `vector_scores.update(hits)` / `keyword_scores.update(keyword_hits)` 合并分数（后出现的同名 chunk 会覆盖先前的分数）。第八步调用 `fuse_hits(repository, rank_lists, (vector_scores, keyword_scores), limit=limit)` 完成 RRF 融合。第九步返回 `HybridRecallResult(chunks=chunks, note=note, vector_available=not note)`——`vector_available` 用 `not note` 推断，即只要向量路报了降级说明就认为向量不可用。
- **异常/边界**：函数自身不捕获异常。`pipeline.document_repo()`、`pipeline.retrieve`、`vector_hits`、`repository.search_keywords`、`fuse_hits` 抛出的任何异常都会向上冒泡（这一点与图路不同：图路在 `graph_recall_multi` 里也没有捕获，但注释中「任一子查询失败不能让整体失败」的承诺主要由 `vector_hits` 内部与分解器的容错承担）。空 `queries` 有明确守卫；`limit` 为 0 或负数时切片会得到空列表，但函数不做校验。`metadata` 不影响关键词路是已知的不对称点。
- **同文件关系**：调用模块内无（它调用的是从 `.hybrid_recall` 导入的 `hybrid_enabled`、`vector_hits`、`fuse_hits`，以及延迟导入的 `hybrid_recall`）；被 `MultiRecallTool.execute`（第 356 行）调用，且只传 `limit`，因此 `threshold` 与 `metadata` 使用默认值。

### `graph_recall_multi(pipeline: Any, queries: list[str], *, limit: int = RAG_RETRIEVE_LIMIT, hops: int = RAG_GRAPH_HOPS, threshold: float | None = None, path_limit: int = RAG_GRAPH_PATH_LIMIT) -> GraphRAGResult` （第 205 行）
- **作用**：多路图召回主体，文档字符串概括为「F3：每条子查询各做 seed+expand，路径按 effective 合并去重」。它对每条子查询各调用一次 `pipeline.graph.retrieve`，然后把结果按三类分别合并：证据（evidence）按 `item.item.id` 去重保留首次出现、种子实体（entities）按顺序拼接后整体去重、关系路径（paths）以 `(path.entities, path.relations)` 为键去重并保留 `effective` 更高的那条。最后把路径按 `effective` 降序（并列时按关系数升序、再按 `target` 字符串升序）排序并截断到 `path_limit`。它是「拆句之后关系类问句才召回得到」这一论点的落地：不同子查询会给出不同种子，合并后能覆盖单句覆盖不到的关系链。
- **参数**：`pipeline`（`Any`）：检索 pipeline，必须提供 `graph.retrieve(query, limit=, hops=, threshold=, path_limit=)`。`queries`（`list[str]`）：候选子查询列表，内部先过滤非字符串与空白项。`limit`（`int`，关键字参数，默认 `RAG_RETRIEVE_LIMIT`）：合并后证据条数上限，同时也透传给每次单路 retrieve。`hops`（`int`，关键字参数，默认 `RAG_GRAPH_HOPS`）：图扩展最大跳数，透传给单路 retrieve。`threshold`（`float | None`，关键字参数，默认 `None`）：相似度/相关度阈值，透传给单路 retrieve。`path_limit`（`int`，关键字参数，默认 `RAG_GRAPH_PATH_LIMIT`）：最终保留的路径条数上限，同时透传给单路 retrieve 限制每条子查询的产出。
- **返回**：返回 `GraphRAGResult`。过滤后 `queries` 为空时抛 `ValueError`（不返回）。只剩一条查询时直接返回 `pipeline.graph.retrieve(...)` 的原始结果。多路时返回一个新的 `GraphRAGResult`，其中 `query` 是用 `" | "` 连接所有子查询拼出的合成问句，`evidence` 是去重后的证据列表切片到 `limit`，`paths` 是排序截断后的合并路径，`entities` 是按 `dict.fromkeys` 保序去重后的种子实体列表。
- **内部流程**：第一步过滤 `queries`；第二步空列表守卫 `raise ValueError("queries must be a non-empty list of strings")`；第三步单查询短路直接返回 `pipeline.graph.retrieve(queries[0], ...)`。第四步初始化三个累加容器：`evidence`（id→MemorySearchResult 的字典）、`paths`（`(entities 元组, relations 元组)`→`GraphPath` 的字典）、`seeds`（字符串列表）。第五步对每条查询循环：调用 `pipeline.graph.retrieve(...)` 得到 `result`；遍历 `result.evidence` 用 `evidence.setdefault(item.item.id, item)` 保留首个同 id 证据（`setdefault` 使后来者不覆盖）；用 `seeds.extend(result.entities)` 累加实体；遍历 `result.paths`，以 `key = (path.entities, path.relations)` 查 `paths.get(key)`，若不存在或 `path.effective > existing.effective` 则用新路径覆盖，从而实现「同一路径保留有效权重更高的版本」。第六步排序：`sorted(paths.values(), key=lambda path: (-path.effective, len(path.relations), path.target))[:path_limit]`，即先按 effective 从大到小，再按关系数从小到大，再按目标名升序，最后截断。第七步构造并返回 `GraphRAGResult`，其中 `query=" | ".join(queries)`、`evidence=list(evidence.values())[:limit]`、`paths=merged`、`entities=list(dict.fromkeys(seeds))`。
- **异常/边界**：`queries` 过滤后为空时主动抛 `ValueError`，消息固定。除此之外不捕获异常，`pipeline.graph.retrieve` 或路径对象属性访问（`path.effective`、`path.target` 等）失败都会向上冒泡；调用方 `MultiRecallTool.execute` 也没有包 try/except，因此图后端抖动会直接让整个工具调用失败——模块注释里「图后端抖动时只丢路径、保留向量证据」的容错在 `execute` 里并未实现为异常捕获（向量结果在异常前已算好但不会返回）。去重与排序对空输入是安全的：`result.evidence`、`result.entities`、`result.paths` 为空时不会写入任何内容；`path_limit` 为 0 时切片得到空列表。
- **同文件关系**：不调用本文件里的其他函数；被 `MultiRecallTool.execute`（第 374 行）调用，只传 `limit` 与 `hops`，因此 `threshold`、`path_limit` 使用默认值。

### `build_decomposer() -> QueryDecomposer` （第 249 行）
- **作用**：按当前配置装配一个问句分解器，是「有 LLM 就用 LLM，没有就用空降级」这一策略的唯一落点。它做三件事：读取 Provider 注册表拿到当前激活的 profile、解析出该 profile 的 API Key、在 Key 看起来可用时构造 `LLM` 客户端并包成 `LLMQueryDecomposer`。它对 Key 做了一条启发式校验 `not key.startswith("replace-with")`，用来识别配置文件里未替换的占位符（例如 `replace-with-your-key`），避免拿着假 Key 去反复请求。整个探测过程被 try/except 包住，任何失败都会记一条 warning 日志并退回 `NullQueryDecomposer`，保证工具在没配置 LLM 的环境里依然可用。
- **参数**：无参数。
- **返回**：返回 `QueryDecomposer`。成功时返回 `LLMQueryDecomposer(client.complete, model=profile.default_model)`；Key 为空、Key 以 `"replace-with"` 开头、或 try 块内任何一步抛异常时返回 `NullQueryDecomposer()`。函数保证不返回 `None`。
- **内部流程**：第一步进入 `try`，在函数体内执行 `from agents.llm import LLM` 与 `from agents.providers import ProviderRegistry` 两个延迟导入。第二步 `registry = ProviderRegistry()` 实例化注册表。第三步 `profile = registry.get(registry.active_profile)` 取当前激活的 profile。第四步 `key = registry.resolve_api_key(profile.name)` 解析 API Key。第五步 `if key and not key.startswith("replace-with")` 判断可用性；通过则 `client = LLM(api_key=key, base_url=profile.base_url, model=profile.default_model)`，然后 `return LLMQueryDecomposer(client.complete, model=profile.default_model)`。第六步 `except Exception` 分支调用 `LOGGER.warning("问句分解器不可用，多路召回退化为原句单路", exc_info=True)`，`exc_info=True` 会把完整堆栈写进日志便于排查。第七步（无论是否走进 except 的尾部路径）`return NullQueryDecomposer()`。注意如果 `key` 判断为假（空字符串、`None`）或命中占位符前缀，try 块会自然走完而不 return，同样落到末尾返回降级实现，此时不会打 warning。
- **异常/边界**：函数对外不抛异常，所有异常都被 `except Exception` 捕获并降级（包括导入失败、注册表构造失败、profile 缺失、`profile.name` 访问失败等），唯一的可观测副作用是 warning 日志。Key 为空或为占位符时静默降级，不打日志，因此「没配 Key」不会在日志里刷警告。
- **同文件关系**：构造并返回 `LLMQueryDecomposer`（调用其 `__init__`）或 `NullQueryDecomposer`；被 `MultiRecallTool.decomposer` property（第 336-339 行）调用。它不调用本文件里的其他函数。

### `class MultiRecallInput(BaseModel)` （第 272 行）
- **作用**：工具入参的 Pydantic 模型，定义了 Agent 调用 `knowledge.multi_recall` 时能传什么、默认值是什么、以及取值范围。`model_config = ConfigDict(extra="forbid", strict=True)` 是两条硬约束：`extra="forbid"` 意味着多传任何未知字段都会直接校验失败（防止 LLM 幻觉出参数名被静默忽略），`strict=True` 意味着不做宽松类型转换（例如字符串 `"8"` 不会自动变成整数 `8`）。它被挂在 `MultiRecallTool.spec` 的 `input_model` 上，由工具框架在校验通过后实例化并传给 `execute`。
- **参数**：类不定义 `__init__`，字段由 Pydantic 生成。字段共 5 个：`query`（`str`，必填，`min_length=1`、`max_length=2000`，描述为「原始检索问句」）；`mode`（`Literal["hybrid", "graph", "both"]`，默认 `"both"`，描述说明 hybrid 是向量×关键词多路融合、graph 是关系路径多路、both 是两者都跑）；`decompose`（`bool`，默认 `True`，描述为是否先用 LLM 把问句拆成多条探针查询，关闭则只跑原句）；`limit`（`int`，默认 `8`，约束 `ge=1, le=50`，描述为「每路返回条数上限」）；`hops`（`int`，默认 `2`，约束 `ge=0, le=3`，描述为「图路最大跳数」）。
- **返回**：不适用（类定义本身不返回值）；其实例是传给 `MultiRecallTool.execute` 的参数对象。
- **内部流程**：类体只包含 `model_config` 与 5 个字段声明。字段声明使用 `Field(...)` 携带默认值、约束与描述，Pydantic 在类创建时据此生成 `__init__`、`__repr__`、校验器等，这些生成方法在本文件中没有显式代码。
- **异常/边界**：由 Pydantic 在实例化时抛 `ValidationError`：`query` 缺失、为空串、超 2000 字符会失败；`mode` 不在三个字面量里会失败；`limit` 不在 1..50、`hops` 不在 0..3 会失败；传入未声明字段会因 `extra="forbid"` 失败；`strict=True` 使 `"8"` 这类字符串不会被强制转成 `int`。本文件不捕获这些异常，由工具框架统一处理。
- **同文件关系**：作为 `MultiRecallTool.spec` 的 `input_model`；被 `MultiRecallTool.execute` 的 `arguments` 形参标注引用。类内不调用本文件任何函数。

### `class MultiRecallOutput(BaseModel)` （第 288 行）
- **作用**：工具出参的 Pydantic 模型，规定 `knowledge.multi_recall` 返回给 Agent 的 JSON 结构，使调用方（以及模型本身）能稳定预期字段含义。它同样是 `extra="forbid", strict=True`，保证不会悄悄多出字段。它把三类信息并列放在一个对象里：融合后的文本命中（`hits`）、图召回结果（`entities`、`paths`、`context`）、以及过程元信息（`sub_queries` 记录实际执行的子查询、`note` 记录向量路降级原因）。`context` 字段尤其重要，它是由 `GraphRAGResult.build_context()` 生成的、可以直接拼进提示词喂给模型的关系路径上下文。
- **参数**：类不定义 `__init__`，字段由 Pydantic 生成。字段共 7 个：`query`（`str`，必填，原始问句）；`sub_queries`（`list[str]`，必填，描述为「实际执行的子查询；原句永远第一条」）；`note`（`str`，必填，描述为「向量路降级说明；空串表示正常」）；`hits`（`list[HybridHit]`，默认空列表，来自 `tool.hybrid_recall` 的命中模型）；`entities`（`list[str]`，默认空列表）；`paths`（`list[dict[str, Any]]`，默认空列表，元素是 `GraphPath.to_dict()` 的产物）；`context`（`str`，默认 `""`，描述为可直接喂给模型的关系路径上下文）。
- **返回**：不适用（类定义本身不返回值）；实例是 `execute` 的返回值，随后由工具框架序列化。
- **内部流程**：类体只有 `model_config` 与 7 个字段声明。`query`、`sub_queries`、`note` 没有默认值因此必填；`hits`、`entities`、`paths` 用 `default_factory=list` 避免可变默认值共享问题；`context` 用不可变字符串默认 `""`。
- **异常/边界**：由 Pydantic 在实例化时抛 `ValidationError`：缺少三个必填字段、`hits` 元素不是合法 `HybridHit`、`paths` 元素不是字典等都会失败。本文件不捕获。注意 `query` 字段没有长度约束（与 `MultiRecallInput.query` 不同）。
- **同文件关系**：作为 `MultiRecallTool.spec` 的 `output_model` 与 `MultiRecallTool.execute` 的返回类型标注；`execute` 末尾显式构造它的实例。类内不调用本文件任何函数。

### `class MultiRecallTool(BaseTool)` （第 300 行）
- **作用**：把整条多路召回能力封装成一个可被 Agent 调用的工具，注册名是 `knowledge.multi_recall`。类属性 `spec` 是 `ToolSpec` 实例，向框架声明了这个工具的全部元信息：英文 description 说明它会跨多条分解后的子查询召回、每条探针跑向量+FTS5 混合路和/或图路、然后把所有名次列表融合（chunk 用 RRF、图路径用 effective 权重），并建议用在「一种说法不够」的关系型问题上、强调是只读操作；`version="1.0.0"`；`input_model`/`output_model` 指向本文件的两个 Pydantic 模型；`side_effect="read"` 与 `permissions=()` 表明无写副作用、不需要额外权限；`timeout_seconds=90.0` 给整次调用 90 秒上限；`idempotent=True`、`parallel_safe=True` 说明重复执行与并发执行都安全；`tags` 为 `("memory", "recall", "multi", "graph", "rrf", "read")` 便于检索与分类；`guidance` 是给模型看的中文使用建议，明确指出单一明确问句应改用 `knowledge.hybrid_recall`、不要为省事把所有问题都走多路（更慢），需要图证据时 `mode` 传 graph 或 both、纯文本召回用 hybrid。
- **参数**：类本身不接收参数；构造参数见 `__init__`。
- **返回**：不适用（类定义本身不返回值）。
- **内部流程**：类体定义 `spec` 类属性（在类创建时即构造 `ToolSpec` 对象），然后定义 `__init__`、两个 property（`pipeline`、`decomposer`）与 `execute` 方法。`spec` 中的 `timeout_seconds=90.0` 与分解器默认 60 秒超时形成配合：单次分解最多 60 秒，整次工具调用最多 90 秒。
- **异常/边界**：无特殊处理（类体本身不执行可能失败的操作，`ToolSpec` 构造在导入时完成）。注意 `spec` 是类属性，所有实例共享同一个 `ToolSpec` 对象。
- **同文件关系**：继承 `core.BaseTool`；引用 `MultiRecallInput`、`MultiRecallOutput` 作为 `input_model`/`output_model`；其实例由 `create_tool` 创建；其方法调用 `_default_pipeline`、`build_decomposer`、`hybrid_recall_multi`、`graph_recall_multi`、`NullQueryDecomposer`。

### `MultiRecallTool.__init__(self, pipeline: Any = None, decomposer: QueryDecomposer | None = None) -> None` （第 325 行）
- **作用**：构造函数，只做一件事——把外部注入的 pipeline 与 decomposer 存到私有属性里，不立即构造任何重对象。这是刻意的「延迟初始化」设计：pipeline 与 decomposer 都通过 property 在首次访问时才真正构建（分别调用 `_default_pipeline()` 与 `build_decomposer()`），因此如果这个工具只是被注册进注册表却从未被调用，就不会拉起 rag 依赖链、也不会去读 Provider 配置。同时它也让测试可以传入替身对象，把 LLM 与数据库完全隔离。
- **参数**：`self`：实例本身。`pipeline`（`Any`，默认 `None`）：外部注入的检索 pipeline；`None` 表示「还没给，等 property 里懒加载默认实现」。`decomposer`（`QueryDecomposer | None`，默认 `None`）：外部注入的问句分解器；`None` 表示「等 property 里调用 `build_decomposer()`」。两者都只做赋值，不做类型校验，传错类型会在真正使用时才暴露。
- **返回**：返回 `None`。
- **内部流程**：两行赋值：`self._pipeline = pipeline`、`self._decomposer = decomposer`。没有日志、没有资源申请、没有对 `BaseTool` 父类构造的显式调用（父类无需参数）。
- **异常/边界**：无特殊处理，不抛异常。传入非 `None` 但类型不符的对象不会被立刻发现。
- **同文件关系**：由 `create_tool()`（第 392 行）以无参形式调用，也会被工具注册表在发现工具时调用；它设置的 `_pipeline`、`_decomposer` 分别被 `pipeline`、`decomposer` 两个 property 读取。

### `MultiRecallTool.pipeline` (property) -> `Any` （第 329-333 行）
- **作用**：惰性获取检索 pipeline 的属性访问器。它的存在使 pipeline 的构造被推迟到真正执行召回的那一刻，并且在构造成功后写回 `self._pipeline` 形成缓存，因此同一个工具实例只会构造一次 pipeline（注意：没有加锁，并发首次访问理论上可能重复构造，但 `parallel_safe=True` 的语义下这属于可接受的冗余）。它把「默认 pipeline 从哪来」这个知识集中在 `_default_pipeline()` 一个地方。
- **参数**：`self`：`MultiRecallTool` 实例，读取其 `_pipeline` 属性。
- **返回**：返回 `Any`。如果 `self._pipeline` 不是 `None`（构造时注入或之前已懒加载过），直接返回它；如果是 `None`，则调用 `_default_pipeline()` 赋值给 `self._pipeline` 后返回同一个对象。永远不会返回 `None`（除非 `_default_pipeline()` 本身返回 `None`）。
- **内部流程**：`if self._pipeline is None:` 判断是否已初始化；成立时执行 `self._pipeline = _default_pipeline()`；最后 `return self._pipeline`。用 `@property` 装饰，因此访问形式是 `tool.pipeline` 而不是 `tool.pipeline()`。
- **异常/边界**：不捕获异常。`_default_pipeline()` 抛出的 `ImportError` 或配置错误会原样冒泡给调用者（即 `execute`）。因为赋值发生在调用成功之后，构造失败时 `self._pipeline` 仍为 `None`，下次访问会再试一次，不会缓存失败状态。
- **同文件关系**：调用模块级 `_default_pipeline`（第 55 行）；被 `MultiRecallTool.execute`（第 349 行 `pipeline = self.pipeline`）调用。

### `MultiRecallTool.decomposer` (property) -> `QueryDecomposer` （第 335-339 行）
- **作用**：惰性获取问句分解器的属性访问器，与 `pipeline` 属性同构。它把「LLM 探测」这件有网络/配置副作用的事推迟到第一次真正需要分解时，并且同样把结果缓存回 `self._decomposer`，避免每次工具调用都重新读 Provider 配置、重新构造 LLM 客户端。如果调用方在构造时注入了自定义分解器（例如测试替身），这个属性会直接返回它，完全不触碰 Provider 注册表。
- **参数**：`self`：`MultiRecallTool` 实例，读取其 `_decomposer` 属性。
- **返回**：返回 `QueryDecomposer`。`self._decomposer` 非 `None` 时直接返回；为 `None` 时调用 `build_decomposer()` 赋值后返回。由于 `build_decomposer` 保证返回 `LLMQueryDecomposer` 或 `NullQueryDecomposer`，这里永远不会返回 `None`。
- **内部流程**：`if self._decomposer is None:` 判断；成立时 `self._decomposer = build_decomposer()`；最后 `return self._decomposer`。以 `@property` 装饰，访问形式为 `tool.decomposer`。
- **异常/边界**：不捕获异常（`build_decomposer` 内部已经吞掉了所有异常，因此实际很难抛出）。构造失败时不会缓存失败状态，下次访问会重试。
- **同文件关系**：调用模块级 `build_decomposer`（第 249 行）；被 `MultiRecallTool.execute`（第 343 行）调用。

### `MultiRecallTool.execute(self, arguments: MultiRecallInput) -> MultiRecallOutput` （第 341 行）
- **作用**：工具的运行时主体，把「分解 → 按 mode 跑混合路和图路 → 组装输出」串成一条流水线。它先根据 `arguments.decompose` 决定用真正的分解器还是临时用 `NullQueryDecomposer` 只取原句，再拿到懒加载的 pipeline，然后按 `mode` 分别调用 `hybrid_recall_multi` 与 `graph_recall_multi`，把混合结果的 chunk 逐个翻译成 `HybridHit`、把图结果的路径转成字典并生成上下文，最后一次性打包成 `MultiRecallOutput`。它是本文件所有其他组件的汇合点：分解器、两个召回函数、两个模型都在这里被真正使用。
- **参数**：`self`：`MultiRecallTool` 实例，提供 `decomposer` 与 `pipeline` 两个属性。`arguments`（`MultiRecallInput`）：已经过 Pydantic 校验的入参对象，含 `query`、`mode`、`decompose`、`limit`、`hops` 五个字段。注意 `execute` 只把 `limit` 传给 `hybrid_recall_multi`、把 `limit` 与 `hops` 传给 `graph_recall_multi`，因此两条路的 `threshold`、`metadata`、`path_limit` 都走各自的默认值，`arguments.limit`（1..50）会同时充当「每路返回条数上限」。
- **返回**：返回 `MultiRecallOutput`，字段赋值如下：`query` 是 `arguments.query` 原样；`sub_queries` 是实际执行过的 `queries` 列表（原句永远在首位，除非 `decompose=False` 时由 `NullQueryDecomposer` 返回单个原句）；`note` 只在跑了 hybrid 时被赋值为 `hybrid.note`，否则保持空串；`hits` 只在跑了 hybrid 时被填充，否则为空列表；`entities`、`paths`、`context` 只在跑了 graph 时被填充，否则分别为空列表、空列表、空串。
- **内部流程**：第一步决定查询列表：`arguments.decompose` 为真时调用 `self.decomposer.decompose(arguments.query)`，为假时调用 `NullQueryDecomposer().decompose(arguments.query)`（临时实例化，不复用 `self.decomposer`）。第二步兜底 `if not queries: queries = [arguments.query]`，保证即使分解器返回空列表也至少有一条原句可跑。第三步 `pipeline = self.pipeline` 触发懒加载。第四步初始化四个累加变量：`hits`（`list[HybridHit]` 空表）、`note`（空串）、`entities`（空表）、`paths`（空表）、`context`（空串）。第五步 `if arguments.mode in {"hybrid", "both"}`：调用 `hybrid_recall_multi(pipeline, queries, limit=arguments.limit)`，取 `note = hybrid.note`，然后用列表推导式把每个 `chunk` 映射成 `HybridHit`——`chunk_id` 取 `chunk.memory_id`；`document_id` 与 `chunk_index` 从 `chunk.metadata` 里取（缺失则为 `None`）；`snippet` 取 `(chunk.content or "")[:200]`，即内容前 200 字符并防止 `None`；`score` 强制 `float(chunk.score)`；`rrf_score` 优先取 `chunk.detail.get("rrf_score")`，取不到则回退成 `chunk.score` 再转 `float`（用 `or` 判断因此 `0.0` 也会走回退）；`vector_score` 与 `keyword_score` 直接从 `chunk.detail` 取，可能为 `None`。第六步 `if arguments.mode in {"graph", "both"}`：调用 `graph_recall_multi(pipeline, queries, limit=arguments.limit, hops=arguments.hops)`，然后 `entities = list(graph.entities)`、`paths = [path.to_dict() for path in graph.paths]`、`context = graph.build_context()`。第七步构造并返回 `MultiRecallOutput(query=arguments.query, sub_queries=queries, note=note, hits=hits, entities=entities, paths=paths, context=context)`。
- **异常/边界**：函数内没有任何 try/except。`arguments.query` 已被输入模型约束为非空且不超过 2000 字符，`mode` 已被约束为三个字面量之一，`limit`、`hops` 也在范围内，因此这些分支不需要再校验。分解器层面已由 `LLMQueryDecomposer` 与 `build_decomposer` 内部容错，不会因 LLM 失败而抛异常；但 pipeline 构造失败、`hybrid_recall_multi`/`graph_recall_multi` 内部（数据库、向量索引、图后端）抛出的异常都会冒泡到框架层，导致本次工具调用失败——这是本文件在「图后端抖动只丢路径」这一承诺上的实际边界。当 `mode="graph"` 时 `note` 恒为空串、`hits` 恒为空列表，输出里不会体现向量路的状态。
- **同文件关系**：调用 `MultiRecallTool.decomposer` 与 `MultiRecallTool.pipeline` 两个 property、模块级 `NullQueryDecomposer`（构造并调用其 `decompose`）、`hybrid_recall_multi`（第 356 行）、`graph_recall_multi`（第 374 行），并构造 `MultiRecallOutput`（以及其中的 `HybridHit`，后者从 `.hybrid_recall` 导入）。它被工具框架在 Agent 发起 `knowledge.multi_recall` 调用时执行。

### `create_tool() -> BaseTool` （第 391 行）
- **作用**：工具工厂函数，为框架提供统一的「创建工具实例」入口，让注册表不需要知道 `MultiRecallTool` 的构造细节。它每次调用都返回一个全新的 `MultiRecallTool`，由于 `__init__` 不接收任何参数、pipeline 与 decomposer 都是懒加载，因此这个函数非常轻，不会在注册阶段触发任何重资源构造。
- **参数**：无参数。
- **返回**：返回 `BaseTool`（实际运行时类型是 `MultiRecallTool`），无参构造，pipeline 与 decomposer 均为 `None` 等待懒加载。
- **内部流程**：单行 `return MultiRecallTool()`。没有缓存、没有条件判断、没有副作用。
- **异常/边界**：无特殊处理，`MultiRecallTool()` 的构造不会失败，因此本函数实际上不抛异常。
- **同文件关系**：构造 `MultiRecallTool`（调用其 `__init__`）；本身不被本文件中的其他函数调用，供外部工具注册/发现机制使用。它与 `__all__` 中的 `"create_tool"` 一起构成模块对外暴露面的一部分。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_default_pipeline` | 延迟导入 `_memory` 并构造默认检索 pipeline，避免模块顶层拉起 rag 重依赖。 |
| `_clean_query_text` | 把任意值清洗成压缩空白并截断的查询字符串，非字符串返回空串。 |
| `QueryDecomposer` | 问句分解器的结构化协议，只声明 `decompose(query) -> list[str]`。 |
| `QueryDecomposer.decompose` | 协议方法签名（无实现），约束所有分解器统一的入参与返回形态。 |
| `NullQueryDecomposer` | LLM 不可用时的安全降级分解器类，只返回原句。 |
| `NullQueryDecomposer.decompose` | 非空字符串返回 `[query.strip()]`，否则返回空列表。 |
| `LLMQueryDecomposer` | 通过 chat 客户端让 LLM 把问句拆成最多 6 条探针查询的分解器类。 |
| `LLMQueryDecomposer.__init__` | 校验并保存 `complete` 回调、模型名与超时时间。 |
| `LLMQueryDecomposer.decompose` | 组装提示词调用 LLM、解析 JSON，任何异常都降级为返回原句。 |
| `LLMQueryDecomposer._normalize` | 原句置于首位，清洗、去重并截断 LLM 返回的子查询列表。 |
| `hybrid_recall_multi` | 每条子查询各跑向量与关键词路，N 路名次列表交给 RRF 融合。 |
| `graph_recall_multi` | 每条子查询各做 seed+expand，路径按 `effective` 合并去重并排序截断。 |
| `build_decomposer` | 有可用 API Key 时构造 LLM 分解器，否则记警告并退回空降级实现。 |
| `MultiRecallInput` | 工具入参模型：`query`、`mode`、`decompose`、`limit`、`hops`，禁止额外字段与宽松转换。 |
| `MultiRecallOutput` | 工具出参模型：子查询、降级说明、融合命中、实体、路径与关系路径上下文。 |
| `MultiRecallTool` | 注册为 `knowledge.multi_recall` 的只读、幂等、可并行的多路召回工具类。 |
| `MultiRecallTool.__init__` | 仅保存注入的 pipeline 与 decomposer，其余全部延迟到首次使用时构造。 |
| `MultiRecallTool.pipeline` | 惰性构造并缓存默认 pipeline 的 property。 |
| `MultiRecallTool.decomposer` | 惰性构造并缓存问句分解器的 property。 |
| `MultiRecallTool.execute` | 分解问句、按 `mode` 跑混合路与图路，组装成 `MultiRecallOutput` 返回。 |
| `create_tool` | 无参构造并返回一个 `MultiRecallTool` 实例。 |
