# memory/rag/knowledge.py

## 一、这个文件是干什么的

这个文件是四层记忆系统中「RAG 知识抽取与知识图谱落库」这一环的核心实现，位于 `memory/rag/` 包内。它的职责是把一段原始文本（或一张图片）交给一个 OpenAI 兼容的聊天模型，让模型输出结构化的实体与关系补丁（patch），再把这些补丁经过严格校验后确定性地写进语义记忆（semantic memory）里，最终形成可查询、可追溯、可失效的历史知识图。

文件整体分成四块内容。第一块是「名字匹配基础设施」：模块级常量 `ENTITY_PREFIX_BOUNDARIES` 以及 `_normalize_for_match`、`_starts_at_boundary`、`is_prefix_match` 三个函数，负责判断两个实体名是否构成「词边界前缀」关系，避免把 `web` 和 `webfoo`、`hub` 和 `hubby` 这种形近但无关的实体错误合并。

第二块是「数据契约」：`EntityCandidate`、`RelationRole`、`RelationCandidate`、`ExtractionResult` 四个 Pydantic 模型，以及 `KnowledgeExtractor` 协议（Protocol）和 `NullKnowledgeExtractor` 空实现。它们定义了「LLM 可以提议什么」以及「什么形状的数据才被允许进入记忆库」，所有字段都带长度上限与取值约束，字符串统一走 `_clean_text` 清洗。

第三块是「抽取器实现」：`LLMKnowledgeExtractor` 类，内含一段非常长的中文 `SYSTEM_PROMPT`（规定实体、关系动作 assert/supersede/retract、取值基数 single/multi/temporal、多元角色、领域复用、证据要求等规则），并通过 `response_content`、`parse_json_object` 两个辅助函数把模型返回的聊天响应解析成合法的 JSON 对象，最后校验成 `ExtractionResult`。

第四块是「图谱物化与上下文渲染」：`EntityResolver` 类负责把抽取出的名字解析到稳定的实体记录（精确名 → 精确别名 → 前缀 → 模糊相似度四段式匹配），`_entity_similarity` 提供模糊相似度打分；`materialize_extraction` 把一次抽取结果确定性地写库（含 supersede/retract 的失效逻辑、证据累积、多元关系与时间字段透传）；`build_graph_context` 反向工作，从已有图里捞出与当前文本相关的实体和一跳邻居，渲染成提示词上下文喂给抽取器，`_known_domains` 与 `_fit_lines` 为它提供领域清单和按整行截断的能力。

在运行时的调用关系上，典型链路是：摄入流程先调用 `build_graph_context` 生成「已知图」上下文，再调用 `LLMKnowledgeExtractor.extract` 拿到 `ExtractionResult`，然后调用 `materialize_extraction` 落库；`materialize_extraction` 与 `build_graph_context` 共享同一个 `EntityResolver` 实例以避免每块文本都全量重扫实体。文件末尾的 `__all__` 显式导出了这些公开名字，同时把 `entity_id_for`、`normalize_entity_name`、`predicate_key_for`、`relation_id_for` 从 `memory.ids` 再导出，方便上层从本模块统一取用。文件本身刻意把「抽取」与「持久化」分离，使得在没有配置任何模型时也能用 `NullKnowledgeExtractor` 走通整条摄入路径，便于测试。

## 二、函数与类逐条详解

### `_normalize_for_match(value: str) -> str` （第 48 行）
- **作用**：把任意字符串规范成「用于实体名匹配」的比较键。它的关键特点是保留词分隔符（空格、连字符、下划线、点、斜杠等），只做大小写折叠和空白压缩，最后剥掉首尾的分隔符。之所以需要它，是因为另一套 `normalize_entity_name` 会把标点全部抹掉以便生成稳定的实体 id，但抹掉标点后就再也无法区分 `web` / `web 中转站` 与无关的 `webfoo` 了。因此凡是「要不要算命中」的判断都走这个函数，而「要不要算同一个 id」的判断走 `normalize_entity_name`。它在实体解析、前缀匹配、图上下文渲染等场景都会被反复调用。
- **参数**：
  - `value: str`：待规范化的原始字符串。虽然标注为 `str`，但内部直接交给 `_clean_text` 处理，因此传入 `None` 等非字符串值由 `_clean_text` 的容错行为决定；无默认值，必传。
- **返回**：返回规范化后的 `str`。如果输入为空、全是空白或全是分隔符，会返回空字符串 `""`（调用方通常把空串视为「无法匹配」）。
- **内部流程**：先调用 `_clean_text(value, max_length=ENTITY_NAME_MAX_LENGTH)` 做基础清洗与截断；随后 `.casefold()` 做大小写折叠（比 `lower()` 更彻底，能处理某些非 ASCII 字母）；接着用正则 `re.sub(r"\s+", " ", value)` 把连续空白压缩成单个空格；再用 `value.strip(...)` 剥掉首尾属于 `ENTITY_PREFIX_BOUNDARIES` 但不含空格的字符集合（即 `"".join(sorted(ENTITY_PREFIX_BOUNDARIES - {" "}))`，先排序保证字符集顺序稳定）；最后再 `.strip()` 去掉可能残留的空白。
- **异常/边界**：自身不主动抛异常；空值/非法值由 `_clean_text` 兜底（可能得到空串）；没有超时概念。空串是合法返回值，调用方必须自行判断。
- **同文件关系**：被 `EntityResolver._remember`、`EntityResolver._store_alias`、`EntityResolver.match`、`EntityResolver.resolve`、`materialize_extraction`（内部 `canonical_by_key` 与 `resolve_endpoint`）、`build_graph_context`（整句前缀匹配那一段）调用；它自己只调用外部导入的 `_clean_text`。

### `_starts_at_boundary(shorter: str, longer: str) -> bool` （第 61 行）
- **作用**：判断「短名字是否正好是长名字的词边界前缀」。这是防止实体误合并的最后一道闸门：它要求 `longer` 必须以 `shorter` 开头，并且紧跟在 `shorter` 之后的那个字符必须落在 `ENTITY_PREFIX_BOUNDARIES` 里（或两者完全等长）。这样 `web` 能命中 `web 中转站`（后面是空格），`deepseek` 能命中 `deepseek-v4.1-flash`（后面是连字符），而 `web` 不会命中 `webfoo`（后面是字母 `f`）。它被单独拆出来是为了让 `is_prefix_match` 只负责长度与阈值判断，逻辑更清晰。
- **参数**：
  - `shorter: str`：候选的短名字（已规范化）。空串会直接返回 `False`。
  - `longer: str`：候选的长名字（已规范化）。必须以 `shorter` 为前缀才可能返回 `True`。
- **返回**：返回 `bool`。当 `shorter` 为空、或 `longer` 不以 `shorter` 开头时返回 `False`；当两者长度完全相等时返回 `True`；否则返回 `longer[len(shorter)] in ENTITY_PREFIX_BOUNDARIES` 的结果。
- **内部流程**：第一步用 `if not shorter or not longer.startswith(shorter)` 短路排除不可能的情况；第二步判断 `len(shorter) == len(longer)`，相等即视为完全匹配返回 `True`；第三步取 `longer` 在 `len(shorter)` 位置上的字符，检查它是否属于模块级常量 `ENTITY_PREFIX_BOUNDARIES`（一个 `frozenset`，成员包括空格、`-`、`_`、`.`、`/`、`|`、`:`、`·`、`（`、`(`）。
- **异常/边界**：不做长度上限校验，也不做规范化（假定调用方已规范化）；若 `longer` 比 `shorter` 短则 `startswith` 自然为 `False`；无特殊异常处理。
- **同文件关系**：只被 `is_prefix_match` 调用；它自身不调用本文件任何函数。

### `is_prefix_match(left_key: str, right_key: str) -> bool` （第 71 行）
- **作用**：对外暴露的「词边界前缀匹配」判定函数。给定两个匹配键，它先把短的那个当 `shorter`、长的当 `longer`，再要求短键长度不低于 `ENTITY_PREFIX_MIN_LENGTH`（来自 constants 的最小长度阈值），最后委托 `_starts_at_boundary` 判定。它存在的意义是让裸的实体名（如 `hub`、`deepseek`）在后续文本里带上修饰语时依然能指回同一个实体，同时又不会因为共用了几个字符就把两个真正不同的实体并成一个。`EntityResolver` 的前缀回退匹配和 `build_graph_context` 的整句匹配都依赖它。
- **参数**：
  - `left_key: str`：第一个匹配键（通常已由 `_normalize_for_match` 规范化），无默认值。
  - `right_key: str`：第二个匹配键（同上），无默认值。
- **返回**：返回 `bool`。当较短的一方长度小于 `ENTITY_PREFIX_MIN_LENGTH` 时无条件返回 `False`；否则返回 `_starts_at_boundary` 的结果。两个参数顺序不影响结果（函数内部会按长度排序）。
- **内部流程**：用条件表达式比较 `len(left_key)` 与 `len(right_key)`，把短的赋给 `shorter`、长的赋给 `longer`；随后检查 `len(shorter) < ENTITY_PREFIX_MIN_LENGTH` 并提前返回 `False`；最后 `return _starts_at_boundary(shorter, longer)`。
- **异常/边界**：无特殊处理，不做清洗也不做长度截断；传入空串时因长度小于最小阈值而返回 `False`。
- **同文件关系**：调用 `_starts_at_boundary`；被 `EntityResolver._prefix_candidate` 与 `build_graph_context` 调用；同时出现在 `__all__` 中，属于对外公开的工具函数。

### `class EntityCandidate(BaseModel)` （第 89 行）
- **作用**：这是「实体候选」的数据契约，代表 LLM 在一次抽取里提议的一个实体节点。它把模型可能给出的杂乱字段收敛成五个受控字段：名字、类型、描述、置信度、别名列表，并用 Pydantic 的 `Field` 施加长度与数值范围约束。之所以要有这个类，是因为「LLM 只是提议者」，只有通过校验的数据才允许进入记忆库，`materialize_extraction` 遍历 `extraction.entities` 时读的就是这个类的实例。配置 `extra="ignore"` 意味着模型多返回的字段会被静默丢弃而不会报错，`strict=True` 则要求类型严格匹配，避免 `"0.8"` 这种字符串被强行转成浮点数。
- **参数**：类本身无构造参数，但字段即等价于关键字参数：
  - `name: str`：必填，`min_length=1`、`max_length=ENTITY_NAME_MAX_LENGTH`，实体规范名。
  - `entity_type: str`：默认 `ENTITY_DEFAULT_TYPE`，最大长度 80。
  - `description: str`：默认空串，最大长度 1000。
  - `confidence: float`：默认 `ENTITY_DEFAULT_CONFIDENCE`，取值区间 `[0, 1]`。
  - `aliases: list[str]`：默认空列表（`default_factory=list`），最多 20 项。
- **返回**：构造与校验成功后返回 `EntityCandidate` 实例；`name` 缺失、为空或超长，`confidence` 越界等会抛 Pydantic 校验错误。
- **内部流程**：Pydantic 在实例化时依次执行字段类型校验与 `Field` 约束校验，其中 `name`/`entity_type`/`description` 三个字段会先经过 `normalize_strings` 校验器做前置清洗，再进入长度与类型检查；`aliases` 列表长度由 `max_length=20` 限制。
- **异常/边界**：越界或类型不符时抛 `pydantic.ValidationError`；多余字段因 `extra="ignore"` 被忽略；空 `name` 被 `min_length=1` 拒绝。
- **同文件关系**：被 `ExtractionResult` 作为 `entities` 字段的元素类型引用；被 `materialize_extraction` 遍历读取；其校验器 `normalize_strings` 调用本文件的 `_clean_text`（外部导入）；出现在 `__all__` 中。

#### `EntityCandidate.normalize_strings(cls, value: Any) -> str` （第 100 行）
- **作用**：这是 Pydantic 的 `field_validator`，绑定在 `name`、`entity_type`、`description` 三个字段上，`mode="before"` 表示它在标准类型校验之前运行。它把模型可能返回的 `None`、带首尾空白的字符串、超长文本统一清洗成规范的字符串，从而让后续的长度约束在一个干净的值上判断，也避免数据库里存入带换行或控制字符的脏数据。
- **参数**：
  - `cls`：类方法隐式参数，由 `@classmethod` 提供。
  - `value: Any`：该字段的原始输入值，可能是 `str`、`None` 或其它任意类型。
- **返回**：返回 `str`，即 `_clean_text(value, max_length=1000)` 的结果；传入 `None` 或无法解析的内容时由 `_clean_text` 决定（通常返回空串）。
- **内部流程**：只有一行 `return _clean_text(value, max_length=1000)`，把清洗逻辑完全委托给 `memory.ids._clean_text`，统一截断上限为 1000 字符。
- **异常/边界**：自身不抛异常，异常行为取决于 `_clean_text`；注意清洗上限（1000）与字段自身上限（`name` 为 `ENTITY_NAME_MAX_LENGTH`）不同，超长名字会先被截到 1000，再被字段约束判定是否越界。
- **同文件关系**：只调用外部导入的 `_clean_text`；被 Pydantic 在校验 `EntityCandidate` 时自动调用。

### `class RelationRole(BaseModel)` （第 104 行）
- **作用**：描述一个 n 元观察中的「额外参与者」。当一个事实除了主语、谓语、宾语之外还涉及地点、活动、工具、人员等角色时，就用这个模型表达，例如「张三 --在--> 公司」还带有 `role=时间`、`role=陪同人` 这样的补充。它把角色的名称、取值、以及该取值对应的实体类型三者绑定在一起，保证物化时能把 `value` 也当作一个实体去解析（`materialize_extraction` 中的 `roles` 列表推导就是这么做的）。存在这个类的原因是把「多元关系」从自由文本升级为结构化字段，使检索侧可以按角色过滤。
- **参数**：字段即关键字参数：
  - `role: str`：必填，`min_length=1`、`max_length=100`，角色名（如「地点」「活动」）。
  - `value: str`：必填，`min_length=1`、`max_length=200`，角色取值（通常是一个实体名）。
  - `entity_type: str`：默认 `ENTITY_DEFAULT_TYPE`，最大长度 80，该取值的实体类型。
- **返回**：校验通过返回 `RelationRole` 实例；字段为空或超长时抛 `pydantic.ValidationError`。
- **内部流程**：实例化时三个字段都先经过 `normalize_strings` 前置清洗，再做 `min_length`/`max_length` 与类型校验；`extra="ignore"` 丢弃模型多给的键。
- **异常/边界**：空 `role` 或空 `value` 会被拒绝（`min_length=1`）；类型不符抛校验错误；无其它特殊处理。
- **同文件关系**：被 `RelationCandidate.roles` 作为列表元素类型引用；其校验器 `normalize_strings` 调用 `_clean_text`；在 `materialize_extraction` 中通过 `candidate.roles` 被读取并转换成普通字典；出现在 `__all__` 中。

#### `RelationRole.normalize_strings(cls, value: Any) -> str` （第 115 行）
- **作用**：与 `EntityCandidate` 中的同名校验器作用一致，但绑定的是 `role`、`value`、`entity_type` 三个字段，且截断上限更小（200 字符），因为角色名和角色取值都是短标识而非长描述。它保证这三个字段在进入长度校验前已被去空白、去控制字符并限长。
- **参数**：
  - `cls`：`@classmethod` 提供的类参数。
  - `value: Any`：字段原始输入，可能是任意类型。
- **返回**：返回 `str`，即 `_clean_text(value, max_length=200)` 的结果。
- **内部流程**：单行委托 `_clean_text`，无其它逻辑分支。
- **异常/边界**：自身不抛异常；清洗结果为空串时会进一步被 `min_length=1` 拒绝（针对 `role` 与 `value`）。
- **同文件关系**：只调用外部 `_clean_text`；由 Pydantic 在构造 `RelationRole` 时自动触发。

### `class RelationCandidate(BaseModel)` （第 119 行）
- **作用**：这是本文件最复杂的数据契约，代表 LLM 提议的一条「关系补丁」。它不仅承载主语、谓语、宾语，还承载了本系统实现时序知识图谱所需的全部语义：`action` 决定这条补丁是新增（assert）、更新（supersede）还是撤回（retract）；`cardinality` 决定同一 (subject, predicate) 槽位是单值、多值累积还是按时间追加；`roles` 承载多元参与者；`valid_from`/`valid_to`/`event_at` 承载时间信息；`status` 承载事实/计划/失效/不确定的状态分类；`confidence` 与 `evidence` 分别承载置信度和原文证据。`materialize_extraction` 正是逐条读取这些字段来决定写库动作与元数据内容的，因此这个类的字段设计直接决定了图谱的时序能力。它的注释里也说明了空时间值表示「无界」。
- **参数**：字段即关键字参数，除 `subject`/`predicate`/`object` 外均有默认值：
  - `subject: str`：必填，`min_length=1`、`max_length=200`，关系主语实体名。
  - `predicate: str`：必填，`min_length=1`、`max_length=100`，关系谓词。
  - `object: str`：必填，`min_length=1`、`max_length=200`，关系宾语实体名。
  - `action: Literal["assert","supersede","retract"]`：默认 `"assert"`。
  - `cardinality: Literal["single","multi","temporal"]`：默认 `"multi"`。
  - `roles: list[RelationRole]`：默认空列表，最多 20 项。
  - `valid_from: str`：默认空串，最大 60，关系成立时间点。
  - `valid_to: str`：默认空串，最大 60，关系失效时间点。
  - `status: Literal["fact","plan","expired","uncertain",""]`：默认 `"fact"`。
  - `event_at: str`：默认空串，最大 60，事件发生时刻。
  - `confidence: float`：默认 `0.75`，区间 `[0, 1]`。
  - `evidence: str`：默认空串，最大 1200，必须是原文片段。
- **返回**：校验通过返回 `RelationCandidate` 实例；缺主语/谓语/宾语、字面量取值非法、置信度越界等都会抛 `pydantic.ValidationError`。
- **内部流程**：实例化时，`subject`/`predicate`/`object`/`evidence`/`valid_from`/`valid_to`/`event_at` 先经 `normalize_strings` 清洗（上限 1200），`status` 先经 `normalize_status` 清洗（上限 20，`None` 变空串），随后 Pydantic 执行字面量枚举校验、长度校验、数值区间校验以及 `roles` 的元素级校验。
- **异常/边界**：`action`/`cardinality`/`status` 出现枚举外的取值会直接报校验错误（这也是防止 LLM 乱造动作名的关键闸门）；`subject` 等为空串被 `min_length=1` 拒绝；多余字段被 `extra="ignore"` 忽略。
- **同文件关系**：被 `ExtractionResult.relations` 引用为元素类型；被 `materialize_extraction` 逐字段读取；其两个校验器分别调用外部 `_clean_text`；出现在 `__all__` 中。

#### `RelationCandidate.normalize_strings(cls, value: Any) -> str` （第 143 行）
- **作用**：绑定在 `subject`、`predicate`、`object`、`evidence`、`valid_from`、`valid_to`、`event_at` 七个文本字段上的前置清洗器。这七个字段共用同一套清洗规则（去空白、限长 1200），用一个校验器集中处理可以避免重复代码，也让 `evidence` 这类长文本字段与短标识字段共享同一截断上限。
- **参数**：
  - `cls`：`@classmethod` 提供的类参数。
  - `value: Any`：字段原始输入值，类型不限。
- **返回**：返回 `str`，即 `_clean_text(value, max_length=1200)` 的结果。
- **内部流程**：单行 `return _clean_text(value, max_length=1200)`，无分支。
- **异常/边界**：自身不抛异常；清洗后的空串会让 `subject`/`predicate`/`object` 被 `min_length=1` 拒绝，而时间字段与证据字段允许为空串。
- **同文件关系**：只调用外部 `_clean_text`；由 Pydantic 在构造 `RelationCandidate` 时自动触发。

#### `RelationCandidate.normalize_status(cls, value: Any) -> str` （第 148 行）
- **作用**：专门为 `status` 字段写的前置清洗器。它比通用清洗多了一个显式的 `None` 判断：当模型没有给出 `status`（值为 `None`）时，返回空串而不是让清洗函数处理 `None`。这样 `status` 的空值语义（「未声明」）就被统一成空字符串，再由 `materialize_extraction` 用 `candidate.status or "fact"` 兜底为 `fact`，避免 `None` 一路传到元数据里。
- **参数**：
  - `cls`：`@classmethod` 提供的类参数。
  - `value: Any`：`status` 字段的原始输入，可能是 `None`、字符串或其它类型。
- **返回**：若 `value is not None` 返回 `_clean_text(value, max_length=20)`，否则返回 `""`。
- **内部流程**：用条件表达式判断 `value is not None`，非空时走 `_clean_text` 截断到 20 字符，为空时直接给空串；之后 Pydantic 仍会用 `Literal` 校验最终取值是否在 `{"fact","plan","expired","uncertain",""}` 之内。
- **异常/边界**：`None` 被安全转换为空串；清洗后若得到枚举外的值（例如模型写了「未知」），会在随后的字面量校验中抛错。
- **同文件关系**：只调用外部 `_clean_text`；由 Pydantic 在构造 `RelationCandidate` 时自动触发。

### `class ExtractionResult(BaseModel)` （第 152 行）
- **作用**：这是一次完整抽取的顶层结果容器，也是抽取器与持久化层之间唯一的交付物。它把模型输出组织成五个字段：领域（domain）、主题列表（topics）、实体列表（entities）、关系列表（relations）、关键词列表（keywords），并为每类列表设了容量上限（topics 20、entities 50、relations 80、keywords 30），与提示词里「单次最多 50 个实体、80 条关系」的约束相呼应。它既是 `LLMKnowledgeExtractor.extract` 的返回类型，也是 `NullKnowledgeExtractor` 的返回值（直接返回全默认的实例，等价于「什么都没抽到」），还是 `materialize_extraction` 的输入参数类型。`domain` 字段默认取常量 `DEFAULT_DOMAIN`，保证永远不会出现空领域。
- **参数**：字段即关键字参数，全部可选：
  - `domain: str`：默认 `DEFAULT_DOMAIN`，`min_length=1`、`max_length=100`。
  - `topics: list[str]`：默认空列表，最多 20 项。
  - `entities: list[EntityCandidate]`：默认空列表，最多 50 项。
  - `relations: list[RelationCandidate]`：默认空列表，最多 80 项。
  - `keywords: list[str]`：默认空列表，最多 30 项。
- **返回**：校验通过返回 `ExtractionResult` 实例；列表超长、元素类型不符、`domain` 为空等都会抛 `pydantic.ValidationError`。
- **内部流程**：实例化时 `domain` 先经 `normalize_domain`（清洗并在空值时回落到 `DEFAULT_DOMAIN`），`topics`/`keywords` 先经 `normalize_lists`（`None` 变空列表、非列表抛 `TypeError`、逐项清洗并剔除空项），`entities`/`relations` 递归触发各自的元素模型校验；最后统一检查各列表的 `max_length`。
- **异常/边界**：列表超长或元素非法抛校验错误；`normalize_lists` 对非列表输入主动抛 `TypeError`；`extra="ignore"` 丢弃多余字段；`strict=True` 防止宽松类型转换。
- **同文件关系**：被 `KnowledgeExtractor.extract` 的返回类型标注引用；被 `NullKnowledgeExtractor.extract` 直接构造返回；被 `LLMKnowledgeExtractor.extract` 通过 `model_validate` 构造；被 `materialize_extraction` 作为入参读取；出现在 `__all__` 中。

#### `ExtractionResult.normalize_domain(cls, value: Any) -> str` （第 163 行）
- **作用**：`domain` 字段的前置校验器。它的关键行为是「空值兜底」：先清洗输入，如果清洗结果是空串（模型没给领域、给了空白或给了无法解析的值），就返回常量 `DEFAULT_DOMAIN`。这保证了领域字段永远非空，从而不会让 `min_length=1` 的约束把一条本来可用的抽取结果整体打掉，也让下游按领域归档的逻辑不必处理空领域。
- **参数**：
  - `cls`：`@classmethod` 提供的类参数。
  - `value: Any`：`domain` 字段的原始输入。
- **返回**：返回 `str`：`_clean_text(value, max_length=100)` 的结果，若该结果为假值则返回 `DEFAULT_DOMAIN`。
- **内部流程**：一行表达式 `return _clean_text(value, max_length=100) or DEFAULT_DOMAIN`，利用 Python 的 `or` 短路把空串替换成默认领域。
- **异常/边界**：自身不抛异常；`None` 由 `_clean_text` 处理成空串后回落到 `DEFAULT_DOMAIN`；超长领域名会被截到 100 字符。
- **同文件关系**：只调用外部 `_clean_text` 与常量 `DEFAULT_DOMAIN`；由 Pydantic 在构造 `ExtractionResult` 时自动触发。

#### `ExtractionResult.normalize_lists(cls, value: Any) -> list[str]` （第 168 行）
- **作用**：`topics` 与 `keywords` 两个字符串列表字段共用的前置校验器。它做三件事：把 `None` 变成空列表（模型经常省略这两个字段）、拒绝非列表类型（防止模型返回一个字符串或字典导致后续遍历出错）、对列表内每一项做清洗并剔除清洗后为空的项（避免出现 `["", "  "]` 这种噪音）。这样 `materialize_extraction` 直接把 `extraction.topics` 写进元数据时就不必再做二次过滤。
- **参数**：
  - `cls`：`@classmethod` 提供的类参数。
  - `value: Any`：字段原始输入，期望是 `list` 或 `None`。
- **返回**：返回 `list[str]`。输入为 `None` 时返回 `[]`；输入为列表时返回清洗后的字符串列表（可能比原列表短）；输入为其它类型时抛 `TypeError`。
- **内部流程**：先判断 `value is None` 直接返回空列表；再判断 `not isinstance(value, list)` 时抛 `TypeError("topics and keywords must be lists")`；最后用列表推导 `[_clean_text(item, max_length=100) for item in value if _clean_text(item, max_length=100)]` 对每项清洗两次（一次用于过滤、一次用于取值），截断上限 100 字符。
- **异常/边界**：非列表输入抛 `TypeError`（注意这是显式抛出的，不是 Pydantic 的校验错误）；列表内的非字符串项交由 `_clean_text` 容错；空项被过滤掉；无超时概念。
- **同文件关系**：只调用外部 `_clean_text`；由 Pydantic 在构造 `ExtractionResult` 时自动触发。

### `class KnowledgeExtractor(Protocol)` （第 176 行）
- **作用**：这是一个结构化类型协议（`typing.Protocol`），用来描述「知识抽取器」应该长什么样。它不提供任何实现，只声明一个 `extract` 方法签名，任何拥有兼容 `extract` 方法的对象（`LLMKnowledgeExtractor`、`NullKnowledgeExtractor`，或测试里手写的确定性抽取器）都被视为满足该协议。之所以采用协议而不是抽象基类，是因为 Python 的鸭子类型让实现方无需显式继承，测试时可以随手写一个假对象注入，从而在没有配置任何模型的情况下也能跑通摄入链路。项目里凡是标注 `KnowledgeExtractor` 类型的地方，实际收到的可能是本文件的任一实现。
- **参数**：类无构造参数。
- **返回**：不实例化，仅作为类型标注使用。
- **内部流程**：类体内只有一个 `extract` 方法声明，函数体是 `...`（省略号），不做任何事。
- **异常/边界**：无运行时行为，因此无异常；若被误实例化，调用 `extract` 会返回 `None`，但这属于误用。
- **同文件关系**：`LLMKnowledgeExtractor` 与 `NullKnowledgeExtractor` 在结构上实现它（未显式继承）；被 `__all__` 导出。

#### `KnowledgeExtractor.extract(self, text: str, *, metadata: Mapping[str, Any] | None = None, graph_context: str = "", image: Any = None, mime_type: str = "image/jpeg") -> ExtractionResult` （第 177 行）
- **作用**：协议中声明的唯一方法，定义抽取器的统一调用入口。它规定了调用方必须传一段文本，可选地传元数据、已有图上下文、图片与图片 MIME 类型，并返回一个 `ExtractionResult`。把参数设计成「文本 + 元数据 + 图上下文 + 可选图片」的形状，使得纯文本摄入与多模态（图片）摄入能共用同一条代码路径；`graph_context` 参数则让调用方能把 `build_graph_context` 的结果直接塞进来，让模型复用已有实体名而不是新造变体。因为它只是声明，所以这里没有任何实际逻辑，仅用于类型检查与文档化。
- **参数**：
  - `self`：实例自身。
  - `text: str`：待抽取的文本内容，无默认值。
  - `metadata: Mapping[str, Any] | None`：可选元数据，关键字参数，默认 `None`；通常包含 `filename`/`source`/`reference_time`/`captured_at`/`event_at`/`modality` 等键。
  - `graph_context: str`：关键字参数，默认空串，已有的图上下文文本。
  - `image: Any`：关键字参数，默认 `None`，可以是图片字节或图片 URL/data URI。
  - `mime_type: str`：关键字参数，默认 `"image/jpeg"`，图片的 MIME 类型。
- **返回**：按声明返回 `ExtractionResult`；协议本身不产生返回值（函数体为 `...`，实际执行返回 `None`）。
- **内部流程**：无实现流程，只有 `...` 占位。
- **异常/边界**：协议不定义异常契约，具体实现自行决定（例如 `LLMKnowledgeExtractor.extract` 会在响应异常时抛 `ValueError`）。
- **同文件关系**：被 `NullKnowledgeExtractor` 与 `LLMKnowledgeExtractor` 以相同签名实现；在 `__all__` 中导出。

### `class NullKnowledgeExtractor` （第 188 行）
- **作用**：一个「什么都不做」的空抽取器，作为没有配置聊天模型时的安全兜底实现。它的 `extract` 无论收到什么输入都返回一个全默认的 `ExtractionResult`（空实体、空关系、默认领域），从而让上层摄入流程不需要写「如果没模型就跳过抽取」的分支。文档字符串明确写着「Safe fallback used when no chat model is configured」，因此它存在的价值在于让整条管道在离线或测试环境下依然可运行、可断言（抽取结果为 0 个实体 0 条关系），也常被用作测试替身。
- **参数**：类无构造参数，也不保存任何状态。
- **返回**：类实例；其 `extract` 方法返回 `ExtractionResult()`。
- **内部流程**：类体只包含一个 `extract` 方法，没有 `__init__`，因此实例化不执行任何逻辑。
- **异常/边界**：无任何异常路径；对空文本、`None` 图片等输入一律静默返回空结果。
- **同文件关系**：实现 `KnowledgeExtractor` 协议；其 `extract` 调用本文件的 `ExtractionResult` 构造器；在 `__all__` 中导出。

#### `NullKnowledgeExtractor.extract(self, text: str, *, metadata: Mapping[str, Any] | None = None, graph_context: str = "", image: Any = None, mime_type: str = "image/jpeg") -> ExtractionResult` （第 191 行）
- **作用**：空抽取器的实现方法。它刻意接受与 `LLMKnowledgeExtractor.extract` 完全一致的参数（包括全部关键字参数），这样两者可以互换注入而不需要调用方做适配。它不读文本、不看元数据、不解析图片，直接构造并返回默认的 `ExtractionResult`，语义上等价于「本次抽取没有发现任何知识」。当项目未配置 provider 或希望以确定性方式测试摄入路径时，就会被用到。
- **参数**：
  - `self`：实例自身。
  - `text: str`：待抽取文本，无默认值；本方法不会读取它。
  - `metadata: Mapping[str, Any] | None`：关键字参数，默认 `None`；本方法忽略。
  - `graph_context: str`：关键字参数，默认空串；本方法忽略。
  - `image: Any`：关键字参数，默认 `None`；本方法忽略。
  - `mime_type: str`：关键字参数，默认 `"image/jpeg"`；本方法忽略。
- **返回**：返回 `ExtractionResult()`，即 `domain=DEFAULT_DOMAIN`、`topics=[]`、`entities=[]`、`relations=[]`、`keywords=[]` 的默认实例。
- **内部流程**：仅一行 `return ExtractionResult()`，触发 Pydantic 默认值填充；无分支、无循环、无外部调用。
- **异常/边界**：无特殊处理，不会因空文本或 `None` 参数抛错。
- **同文件关系**：调用本文件的 `ExtractionResult`；实现 `KnowledgeExtractor` 协议的 `extract` 签名。

### `response_content(response: Any) -> str` （第 203 行）
- **作用**：从一次聊天补全的响应中取出助手文本。它同时兼容三种常见形态：直接传入的原始字符串、OpenAI 风格的字典响应（`{"choices": [{"message": {"content": ...}}]}`）、以及对象风格响应（带 `choices` 属性、`message` 属性、`content` 属性）。文档字符串说明它被抽取器和 `tool/multi_recall` 的查询分解器共用，因此两种调用方都能接受同样的 provider 响应形状。它把「响应形状差异」集中在一个地方处理，避免每处调用都写一遍属性/键的双路探测。
- **参数**：
  - `response: Any`：聊天补全响应，可以是 `str`、`Mapping`（字典）或任意带同名属性的对象；无默认值。
- **返回**：返回助手文本 `str`。当输入是字符串时原样返回；当响应无 `choices`、`choices` 为空、或最终 `content` 不是非空字符串时抛 `ValueError`。
- **内部流程**：先判断 `isinstance(response, str)` 直接返回；否则用 `response.get("choices")`（当它是 `Mapping`）或 `getattr(response, "choices", None)` 取 choices；若 choices 为空则抛 `ValueError("knowledge extraction response contained no choices")`；再对 `choices[0]` 用同样的键/属性双路探测取 `message`，再从 `message` 取 `content`；最后校验 `content` 是否为非空字符串，否则抛 `ValueError("knowledge extraction response contained no text")`；通过则返回该字符串。
- **异常/边界**：两种失败情形都抛 `ValueError`，错误信息分别指明「没有 choices」与「没有文本」；`content` 为 `None`、空串或纯空白（用 `.strip()` 判断）都视为无文本；`choices[0]` 若是 `None`，属性探测会返回 `None` 从而走到同一个 `ValueError`。
- **同文件关系**：被 `LLMKnowledgeExtractor.extract` 调用；自身不调用本文件其它函数；在 `__all__` 中导出。

### `parse_json_object(raw: str) -> dict[str, Any]` （第 222 行）
- **作用**：把模型返回的文本解析成 JSON 对象，并对 LLM 常见的「不听话」格式做容错。模型经常把 JSON 包在 ```json 代码围栏里，或在 JSON 前后加一句解释性文字（例如「好的，以下是抽取结果：」）。这个函数先剥离围栏，若直接解析失败再用「取第一个 `{` 到最后一个 `}` 之间子串」的兜底策略重试，从而大幅提高解析成功率。它只接受对象（`dict`），不接受数组或标量，因为后续的 `ExtractionResult.model_validate` 需要一个映射。
- **参数**：
  - `raw: str`：模型返回的原始文本，无默认值；允许带 Markdown 围栏与前后散文。
- **返回**：返回 `dict[str, Any]`，即解析出的 JSON 对象。若文本不含任何可解析的对象结构、或解析结果不是对象，则抛异常。
- **内部流程**：先 `candidate = raw.strip()`；若以 ``` 开头，用正则 `^```(?:json)?\s*` （忽略大小写）去掉起始围栏，再用 `\s*```$` 去掉结尾围栏并再次 `strip()`；然后 `json.loads(candidate)` 尝试直接解析；若抛 `json.JSONDecodeError`，则用 `candidate.find("{")` 与 `candidate.rfind("}")` 定位最外层对象边界，若 `start < 0` 或 `end <= start` 则抛 `ValueError("knowledge extraction response was not valid JSON")`，否则对 `candidate[start:end+1]` 再解析一次（这一次若仍失败，`JSONDecodeError` 会向上传播）；最后用 `isinstance(value, dict)` 校验，不是字典则抛 `TypeError("knowledge extraction response must be a JSON object")`。
- **异常/边界**：可能抛 `ValueError`（无 JSON 结构）、`TypeError`（解析出的是数组或标量）、以及兜底解析失败时的 `json.JSONDecodeError`；空字符串会走到 `find` 返回 `-1` 的分支抛 `ValueError`；嵌套花括号的场景依赖 `rfind` 取到最外层闭合位置，若 JSON 后有其它含 `}` 的文字会解析失败。
- **同文件关系**：被 `LLMKnowledgeExtractor.extract` 调用；自身不调用本文件其它函数；在 `__all__` 中导出。

### `class LLMKnowledgeExtractor` （第 241 行）
- **作用**：本文件的核心抽取器实现，通过一个 OpenAI 兼容的聊天客户端把文本或图片变成结构化的 `ExtractionResult`。类上挂着一段很长的中文 `SYSTEM_PROMPT` 类属性，这是整套抽取行为的「规则手册」：它规定要抽取所有实体、已知实体必须原样复用规范名、别名要放进 aliases；规定三种关系动作 assert/supersede/retract 的语义；规定三种取值基数 single/multi/temporal 的语义；规定多元关系放进 roles 且时间写 event_at；规定领域要优先复用已知领域、禁止用「未分类」「其他」「默认」、新领域名 2~10 个汉字；还规定了证据必须来自原文、单次最多 50 实体 80 关系、谓词最多 10 个汉字等硬约束，并给出字段格式说明。类本身持有 `complete` 可调用对象（模型客户端）、`model`、`vision_model`、`timeout` 四个状态，被摄入流程在配置了 provider 时实例化。
- **参数**：见 `__init__`。类属性 `SYSTEM_PROMPT` 是固定字符串，不可通过构造参数覆盖。
- **返回**：实例化返回抽取器对象，其 `extract` 返回 `ExtractionResult`。
- **内部流程**：类定义时先构建 `SYSTEM_PROMPT` 常量，随后定义 `__init__`、`extract` 与静态方法 `_image_data_url`。
- **异常/边界**：`__init__` 会对非可调用的 `complete` 抛 `TypeError`；`extract` 的异常路径见该方法。
- **同文件关系**：实现 `KnowledgeExtractor` 协议；内部调用 `response_content`、`parse_json_object`、`_image_data_url` 与 `ExtractionResult`；使用外部 `utc_now` 与 `_clean_text`；在 `__all__` 中导出。

#### `LLMKnowledgeExtractor.__init__(self, complete: Callable[..., Any], *, model: str | None = None, vision_model: str | None = None, timeout: float = 60.0) -> None` （第 290 行）
- **作用**：构造抽取器并做最小化的防御性校验。它把模型客户端（`complete`）、文本模型名、视觉模型名和超时时间保存为实例属性，供后续每次 `extract` 调用使用。之所以把 `complete` 设计成注入的可调用对象而不是在内部创建 HTTP 客户端，是为了让抽取器与具体 provider 解耦，测试时可以注入一个返回固定 JSON 的假函数。校验 `callable(complete)` 是为了在装配阶段就发现配置错误，而不是等到第一次抽取时才在调用处炸掉。
- **参数**：
  - `self`：实例自身。
  - `complete: Callable[..., Any]`：必填，位置参数，一个接受 `messages` 等关键字参数并返回聊天补全响应的可调用对象；必须是可调用的，否则抛 `TypeError`。
  - `model: str | None`：关键字参数，默认 `None`，纯文本抽取时使用的模型名；`None` 表示交给客户端自行决定默认模型。
  - `vision_model: str | None`：关键字参数，默认 `None`，带图片抽取时优先使用的模型名；为空时回退到 `model`。
  - `timeout: float`：关键字参数，默认 `60.0` 秒，传给模型客户端的超时时间。
- **返回**：返回 `None`（构造器），实例被初始化并保存四个属性。
- **内部流程**：先 `if not callable(complete): raise TypeError("complete must be callable")`；随后依次赋值 `self.complete`、`self.model`、`self.vision_model`、`self.timeout`；不做任何网络请求或模型调用。
- **异常/边界**：`complete` 不可调用时抛 `TypeError`；`timeout` 未做正数校验，传负数或 `0` 会在实际调用时由客户端决定行为；`model`/`vision_model` 允许为 `None`。
- **同文件关系**：为 `extract` 提供实例状态；不调用本文件其它函数。

#### `LLMKnowledgeExtractor.extract(self, text: str, *, metadata: Mapping[str, Any] | None = None, graph_context: str = "", image: Any = None, mime_type: str = "image/jpeg") -> ExtractionResult` （第 305 行）
- **作用**：执行一次真正的 LLM 知识抽取。它把输入整理成一个 JSON 载荷（来源文件名、截断后的正文、参考时间，以及可选的拍摄时间/事件时间/模态），可选地把「已知图」上下文一并附上，然后以 `system` + `user` 两条消息的形式调用注入的 `complete`；若带图片，则把 user 内容改成多模态数组（文本块 + image_url 块），并改用视觉模型名。模型返回后用 `response_content` 取文本、`parse_json_object` 解析、`ExtractionResult.model_validate` 校验，从而保证只有合法数据流出本函数。它被摄入流程在每次处理一块文本/图片时调用。
- **参数**：
  - `self`：实例自身。
  - `text: str`：待抽取文本，位置参数；非字符串或纯空白且未提供图片时会直接返回空结果。
  - `metadata: Mapping[str, Any] | None`：关键字参数，默认 `None`；可含 `filename`、`source`、`reference_time`、`captured_at`、`event_at`、`modality` 等键。
  - `graph_context: str`：关键字参数，默认空串；非空白时以键名 `"已知图"` 放进载荷。
  - `image: Any`：关键字参数，默认 `None`；可以是 `bytes`/`bytearray`/`memoryview` 或非空字符串（URL 或 data URI）。
  - `mime_type: str`：关键字参数，默认 `"image/jpeg"`；仅当 `image` 是字节时用于拼接 data URI。
- **返回**：返回校验通过的 `ExtractionResult`。若文本为空且无图片，返回默认 `ExtractionResult()`（不调用模型）。
- **内部流程**：第一步做空输入短路：`(not isinstance(text, str) or not text.strip()) and image is None` 时直接返回 `ExtractionResult()`；第二步 `metadata = dict(metadata or {})` 复制一份避免修改调用方对象；第三步从 `filename` 或 `source` 中取来源名并用 `_clean_text(..., max_length=300)` 清洗；第四步构造 `payload`，含 `source`、`text`（`text[:12000]` 截断，非字符串则给空串）、`reference_time`（`metadata` 里的值或 `utc_now().isoformat()`）；第五步遍历 `("captured_at", "event_at", "modality")`，只把有值的键加入载荷；第六步 `graph_context.strip()` 非空时加入 `payload["已知图"]`；第七步 `json.dumps(payload, ensure_ascii=False)` 序列化成提示词；第八步按有无图片分支：无图片时 `user_content` 是纯字符串、`selected_model = self.model`，有图片时 `user_content` 是包含 `{"type":"text",...}` 与 `{"type":"image_url","image_url":{"url": self._image_data_url(image, mime_type)}}` 的列表、`selected_model = self.vision_model or self.model`；第九步组装 `messages`（system 为 `SYSTEM_PROMPT`，user 为 `user_content`）；第十步调用 `self.complete(messages, model=selected_model, temperature=0.0, timeout=self.timeout, stream=False)`，其中 `temperature=0.0` 保证抽取尽可能确定性、`stream=False` 要求一次性返回；第十一步 `raw = response_content(response)`；第十二步 `return ExtractionResult.model_validate(parse_json_object(raw))`。
- **异常/边界**：`response_content` 可能抛 `ValueError`（响应无 choices 或无文本）；`parse_json_object` 可能抛 `ValueError`/`TypeError`/`json.JSONDecodeError`；`model_validate` 可能抛 `pydantic.ValidationError`（模型返回的字段越界或类型不符）；`_image_data_url` 在图片类型非法时抛 `TypeError`；`self.complete` 自身可能抛网络/超时异常，本函数不捕获，直接向上传播。文本超长会被静默截断到 12000 字符；`metadata` 为 `None` 时安全处理为空字典。
- **同文件关系**：调用本文件的 `response_content`、`parse_json_object`、`_image_data_url`、`ExtractionResult`（通过 `model_validate`）；使用外部 `_clean_text`、`utc_now`；实现 `KnowledgeExtractor` 协议的 `extract` 签名。

#### `LLMKnowledgeExtractor._image_data_url(image: Any, mime_type: str) -> str` （第 358 行）
- **作用**：把图片输入统一转换成 OpenAI 多模态接口能接受的 `image_url` 字符串。它支持两类输入：原始字节（会被 base64 编码成 `data:<mime>;base64,<...>` 形式）和已经是 URL 或 data URI 的非空字符串（原样返回）。这样上层既可以传本地读到的图片二进制，也可以直接传一个已经托管好的图片链接，两种用法共用同一个方法。定义为 `@staticmethod` 是因为它不依赖实例状态，纯粹是格式转换。
- **参数**：
  - `image: Any`：图片数据。可以是 `bytes`、`bytearray`、`memoryview`，或非空字符串（URL/data URI）。
  - `mime_type: str`：图片 MIME 类型，用于拼接 data URI 前缀，无默认值（由调用方传入，`extract` 中默认是 `"image/jpeg"`）。
- **返回**：返回 `str`：字节输入返回 `data:{mime_type};base64,{encoded}`；字符串输入返回去除首尾空白后的原字符串。类型不支持时抛 `TypeError`。
- **内部流程**：先判断 `isinstance(image, (bytes, bytearray, memoryview))`，成立则 `base64.b64encode(bytes(image)).decode("ascii")` 编码，并格式化成 data URI 返回；否则判断 `isinstance(image, str) and image.strip()`，成立则返回 `image.strip()`；两者都不满足时抛 `TypeError("image must be bytes or a non-empty URL/data URI")`。
- **异常/边界**：空字符串、纯空白字符串、`None`、其它对象类型都会抛 `TypeError`；`mime_type` 不做合法性校验，传入非法值会生成一个前缀错误的 data URI；无超时或大小限制（大图会生成很长的 base64 字符串）。
- **同文件关系**：被 `LLMKnowledgeExtractor.extract` 在图片分支中调用；自身只依赖标准库 `base64`。

### `_entity_similarity(left: str, right: str, aliases: list[str] | None = None) -> float` （第 367 行）
- **作用**：计算一个名字与某个已知实体（含其别名）之间的相似度分数，用于 `EntityResolver.resolve` 的最后一段模糊兜底匹配。它把两种信号取最大值：一是用 `SequenceMatcher` 计算的字符序列相似度（对拼写差异、轻微变体敏感），二是按词元集合计算的 Jaccard 重叠率（对词序变化、多词名字的部分重合敏感，且通过正则 `[\w\u4e00-\u9fff]+` 专门覆盖中文）。之所以两者并用，是因为单一指标容易漏判：字符相似度对「同一实体的两种叫法」不够稳，而词元重叠对「一个词是另一个词的子串」又不够细。
- **参数**：
  - `left: str`：待匹配的名字（通常是抽取出的新名字），无默认值。
  - `right: str`：已知实体的规范名，无默认值。
  - `aliases: list[str] | None`：已知实体的别名列表，默认 `None`；会被展开加入比较候选，使别名也能贡献分数。
- **返回**：返回 `float`，取值在 `[0.0, 1.0]` 之间，是「所有别名/规范名的序列相似度最大值」与「词元 Jaccard 重叠率」两者中的较大者。两个名字都为空时，序列分数为 0，词元重叠因分母用 `max(..., 1)` 保护也返回 0.0。
- **内部流程**：先用 `normalize_entity_name(left)` 得到 `left_key`；把 `right` 与所有 `aliases` 合成 `right_values`；对每个候选值用 `SequenceMatcher(None, left_key, normalize_entity_name(value)).ratio()` 算分并收集到 `scores`；然后用正则分词得到 `left_tokens` 与 `right_tokens`（均先 `casefold()`），计算交集大小除以并集大小（分母用 `max(len(union), 1)` 防止除零）得到 `overlap`；最后 `return max(max(scores, default=0.0), overlap)`。
- **异常/边界**：`aliases` 为 `None` 或空列表时只比较 `right`；`scores` 为空时 `max(..., default=0.0)` 兜底为 0.0；不做异常抛出；`normalize_entity_name` 可能把名字归一成空串，此时序列相似度基于空串计算，结果通常很低。
- **同文件关系**：调用外部 `normalize_entity_name`；被 `EntityResolver.resolve` 在模糊匹配阶段调用。

### `class EntityResolver` （第 377 行）
- **作用**：把抽取出来的实体名字解析到语义记忆里稳定的实体记录上，是防止图谱「同一实体反复新建」的关键组件。它维护两个字典索引：`_entities` 以规范化规范名为键、`_index` 同时索引规范名与所有别名，两者都指向 `MemoryItem`。解析顺序被文档字符串明确为「精确规范名 → 精确别名 → 前缀 → 模糊相似度」，别名被显式索引，因此一个写得完整的端点名总能收敛到同一个「星球」（实体节点）。它被 `materialize_extraction` 与 `build_graph_context` 使用，且设计成可注入复用以避免每块文本都全量重扫实体表。类上还有注释说明：节点属性的合并与写回已收敛到 `tool/graph_node_update.py` 的 `update_entity_node`，本类只负责「名字 → 已有实体」的匹配。
- **参数**：见 `__init__`。
- **返回**：实例化返回解析器对象；`resolve` 返回规范名字符串，`match` 返回 `MemoryItem` 或 `None`。
- **内部流程**：类定义依次为 `__init__`、`_load`、`_remember`、`_exact`、`_prefix_candidate`、`_store_alias`、`match`、`resolve`。
- **异常/边界**：构造时若 `manager` 缺少 `semantic` 属性会在 `_load` 中抛 `AttributeError`；`resolve` 对空名字抛 `ValueError`。
- **同文件关系**：调用 `_normalize_for_match`、`is_prefix_match`、`_entity_similarity`；被 `materialize_extraction`（默认构造或注入）与 `build_graph_context` 使用；在 `__all__` 中导出。

#### `EntityResolver.__init__(self, manager: MemoryManager, *, similarity_threshold: float = ENTITY_SIMILARITY_THRESHOLD) -> None` （第 385 行）
- **作用**：构造解析器并立即把记忆库中已有的实体全部载入索引。它保存记忆管理器引用、相似度阈值，初始化两个空字典 `_entities` 与 `_index`，然后调用 `_load()` 完成首次全量索引。这样做的好处是后续每次 `match`/`resolve` 都在内存里完成，避免逐条查询数据库；代价是构造有一定开销，这也是 `materialize_extraction` 与 `build_graph_context` 支持注入同一个解析器实例的原因。相似度阈值可配置，使得不同场景可以放宽或收紧模糊匹配。
- **参数**：
  - `self`：实例自身。
  - `manager: MemoryManager`：必填，位置参数，记忆管理器，需提供 `semantic.list()` 与 `semantic.facts()` 等接口。
  - `similarity_threshold: float`：关键字参数，默认取常量 `ENTITY_SIMILARITY_THRESHOLD`；作为 `resolve` 模糊匹配的判定阈值。
- **返回**：返回 `None`，实例完成初始化并已载入实体索引。
- **内部流程**：赋值 `self.manager`、`self.similarity_threshold`；初始化 `self._entities = {}` 与 `self._index = {}`；最后调用 `self._load()` 遍历现有实体建立索引。
- **异常/边界**：未校验 `similarity_threshold` 的范围（传 0 会让几乎所有名字都模糊命中，传大于 1 会让模糊匹配永不生效）；`manager` 不合法时异常在 `_load` 中抛出。
- **同文件关系**：调用本类的 `_load`；被 `materialize_extraction` 与 `build_graph_context` 构造。

#### `EntityResolver._load(self) -> None` （第 392 行）
- **作用**：全量重建实体索引。它先清空两个字典，再遍历 `manager.semantic.list()` 返回的所有语义记忆条目，只挑出 `metadata["kind"] == "entity"` 的条目交给 `_remember` 建索引。之所以要区分 kind，是因为语义层里既有实体节点也有关系事实，只有实体才参与名字解析。清空再重建的写法保证重复调用 `_load` 不会残留过期条目，也让索引与数据库状态重新对齐。
- **参数**：
  - `self`：实例自身；无其它参数。
- **返回**：返回 `None`，副作用是填充 `self._entities` 与 `self._index`。
- **内部流程**：`self._entities.clear()` 与 `self._index.clear()`；然后 `for item in self.manager.semantic.list():`，用 `if item.metadata.get("kind") != "entity": continue` 过滤；对通过过滤的条目调用 `self._remember(item)`。
- **异常/边界**：若条目缺少 `metadata` 属性或 `semantic.list()` 抛错则异常上抛；没有任何实体时两个字典保持为空，后续 `match` 一律返回 `None`。
- **同文件关系**：调用本类的 `_remember`；只被 `__init__` 调用。

#### `EntityResolver._remember(self, item: MemoryItem) -> None` （第 400 行）
- **作用**：把一个实体条目登记进两个索引。它从元数据里取规范名（优先 `canonical_name`，其次 `title`，最后退回 `item.content`），用 `_normalize_for_match` 生成键，然后写入 `_entities`（规范名索引）与 `_index`（规范名 + 别名索引）；再遍历 `metadata["aliases"]`，把每个别名规范化后也指向同一个条目。文档字符串强调「Index one entity under its canonical name and every known alias」，这是「别名写法总能收敛到同一实体」的实现基础。`_index` 使用 `setdefault` 写入规范名，保证先登记的条目不会被后来的同键条目覆盖。
- **参数**：
  - `self`：实例自身。
  - `item: MemoryItem`：待索引的实体条目，需带 `metadata` 与 `content`。
- **返回**：返回 `None`，副作用是更新两个索引字典。
- **内部流程**：取 `metadata = item.metadata`；用 `str(metadata.get("canonical_name") or metadata.get("title") or item.content)` 得到 `canonical`；`key = _normalize_for_match(canonical)`；若 `key` 为空则直接 `return`（不索引）；写 `self._entities[key] = item` 与 `self._index.setdefault(key, item)`；最后遍历 `metadata.get("aliases") or []`，对每个别名 `alias_key = _normalize_for_match(str(alias))`，非空时写 `self._index[alias_key] = item`（别名键会被后来的条目覆盖）。
- **异常/边界**：规范名为空或全是分隔符时静默跳过，不报错；`aliases` 为 `None` 时用 `or []` 兜底；别名与规范名冲突时后写入者覆盖 `_index` 中该键。
- **同文件关系**：调用 `_normalize_for_match`；被 `_load` 与 `resolve`（写完库后刷新索引）调用。

#### `EntityResolver._exact(self, key: str) -> MemoryItem | None` （第 415 行）
- **作用**：做精确查找，是解析流程的第一段。它先在规范名索引 `_entities` 里找，找不到再到别名索引 `_index` 里找，用 `or` 短路实现「规范名优先于别名」。虽然 `_index` 实际上也包含规范名，但分开写可以明确表达「先规范名、后别名」的语义优先级，也让代码意图更清晰。
- **参数**：
  - `self`：实例自身。
  - `key: str`：已规范化的匹配键，无默认值。
- **返回**：命中时返回对应的 `MemoryItem`，两处都未命中时返回 `None`。
- **内部流程**：单行 `return self._entities.get(key) or self._index.get(key)`；若 `_entities` 命中则不会再查 `_index`。
- **异常/边界**：传入空串时会返回 `None`（因为索引里不会有空键，`_remember` 已过滤）；无异常抛出。
- **同文件关系**：被 `match` 与 `resolve` 调用；自身不调用本文件其它函数。

#### `EntityResolver._prefix_candidate(self, key: str) -> MemoryItem | None` （第 418 行）
- **作用**：做前缀回退查找，是解析流程的第三段。它遍历整个 `_index`，用 `is_prefix_match` 找出所有与 `key` 构成词边界前缀关系的已知键，并在其中选择「共享长度最长」的那个条目返回。文档字符串特别说明了两点：一是「最长完整前缀获胜，部分字符重叠永远不算」，二是这个方法对索引是只读的——因为前缀命中时值得记录的别名是调用方实际写下的名字而不是规范化键，所以记录别名的动作放在 `resolve` 里而不是这里。这样设计避免了 `_prefix_candidate` 产生写副作用，使它在只读查询场景也能安全调用。
- **参数**：
  - `self`：实例自身。
  - `key: str`：已规范化的待匹配键，无默认值。
- **返回**：返回最佳匹配的 `MemoryItem`；没有任何前缀命中时返回 `None`。
- **内部流程**：初始化 `best = None`、`best_length = 0`；对 `list(self._index.items())` 的快照遍历（`list()` 避免遍历时索引被修改）；对每个 `known_key` 先用 `is_prefix_match(key, known_key)` 过滤；再用 `shared = min(len(key), len(known_key))` 计算共享长度，若 `shared <= best_length` 则跳过（保证严格更优才替换）；否则更新 `best` 与 `best_length`；循环结束后返回 `best`。
- **异常/边界**：索引为空时返回 `None`；`key` 为空串时 `is_prefix_match` 因长度不足返回 `False`，同样返回 `None`；并列长度时保留先遇到的条目（因为要求严格大于才替换）；无异常抛出。
- **同文件关系**：调用 `is_prefix_match`；被 `match` 与 `resolve` 调用。

#### `EntityResolver._store_alias(self, item: MemoryItem, name: str) -> MemoryItem` （第 437 行）
- **作用**：把一个新写法登记为已有实体的别名，并同步刷新本解析器的内存索引，返回写库后的最新条目。它解决的问题是：前缀匹配命中了某个实体（例如已存 `web 中转站`，本次出现 `web`），需要把 `web` 记为别名，否则下次再遇到 `web` 又只能靠前缀回退，且模型也无法从图上下文里看到这个别名。文档字符串强调了两个细节：索引保留规范化键，所以记录拼写不会扩大前缀匹配范围；所有原先指向写库前快照的索引键都必须重新指向新对象，否则下一次 `resolve` 会读到旧快照从而「丢掉」刚写入的别名。函数内部采用函数内导入 `tool.graph_node_update.update_entity_node`，注释解释了原因：该模块属于上层能力模块，模块级导入会形成 `memory.rag -> tool -> memory` 的初始化环。
- **参数**：
  - `self`：实例自身。
  - `item: MemoryItem`：已有的实体条目（写库前的快照），无默认值。
  - `name: str`：要登记为新别名的名字（调用方实际写下的拼写），无默认值。
- **返回**：返回 `MemoryItem`，即 `update_entity_node` 写回后的最新实体对象。
- **内部流程**：先取 `canonical = str(item.metadata.get("canonical_name") or item.content)`；在函数内 `from tool.graph_node_update import update_entity_node`；调用 `update_entity_node(self.manager, name, existing=item, aliases=[name], importance=item.importance, item_id=item.id)` 得到 `stored`；然后遍历 `list(self._index.items())`，把所有 `indexed.id == item.id` 的键重新指向 `stored`；最后把 `self._entities[_normalize_for_match(canonical)] = stored` 并返回 `stored`。
- **异常/边界**：`update_entity_node` 的异常（导入失败、写库失败）直接向上传播；`item.metadata` 缺失规范名时退回 `item.content`；若索引里没有任何键指向该 id，则只更新 `_entities` 中的规范名键；无空值特殊处理。
- **同文件关系**：调用 `_normalize_for_match` 与外部 `tool.graph_node_update.update_entity_node`；只被 `resolve` 调用。

#### `EntityResolver.match(self, name: str) -> MemoryItem | None` （第 466 行）
- **作用**：只读地判断一个名字是否指向已有实体，返回该实体或 `None`。它只做「精确 + 前缀」两段匹配，不做模糊相似度，因此调用成本低、语义保守，适合大量调用的场景——`build_graph_context` 就是对文本里切出的每一个词调用它来收集种子实体。把它与 `resolve` 分开，是因为「查询是否认识这个名字」和「把这个名字落库并可能新建实体」是两种完全不同的意图，前者绝不应该产生写副作用。
- **参数**：
  - `self`：实例自身。
  - `name: str`：待查询的名字，无默认值；内部会先规范化。
- **返回**：返回命中的 `MemoryItem`；名字规范化为空、或精确与前缀都未命中时返回 `None`。
- **内部流程**：`key = _normalize_for_match(name)`；若 `not key` 直接返回 `None`；否则 `return self._exact(key) or self._prefix_candidate(key)`，即先精确后前缀。
- **异常/边界**：空名或纯分隔符名返回 `None`；不修改任何状态；不抛异常。
- **同文件关系**：调用 `_normalize_for_match`、`_exact`、`_prefix_candidate`；被 `build_graph_context` 调用。

#### `EntityResolver.resolve(self, name: str, *, domain: str, entity_type: str = ENTITY_DEFAULT_TYPE, description: str = "", confidence: float = ENTITY_DEFAULT_CONFIDENCE, aliases: list[str] | None = None, source_id: str | None = None) -> str` （第 474 行）
- **作用**：把抽取出的实体名字解析到稳定实体并写库，返回规范名字符串。它按四段顺序尝试匹配：精确规范名/别名 → 前缀 → 模糊相似度；若全都未命中则视为新实体。命中后统一调用 `tool.graph_node_update.update_entity_node` 做属性合并与写回（把 domain、entity_type、description、aliases、importance、source_id 并进已有节点），并刷新自己的索引。其中有一处精心的设计：只有「精确命中」才允许把本次写下的名字捐献为新别名（`add_written_name=name_is_known`），因为前缀命中或模糊命中的名字只是「长得像」，如果也登记成别名就会污染那个实体。文档字符串与注释都强调本类只负责匹配，属性更新的唯一实现在 `tool/graph_node_update.py`。
- **参数**：
  - `self`：实例自身。
  - `name: str`：实体名字，位置参数，必填；会被 `_clean_text` 截断到 `ENTITY_NAME_MAX_LENGTH`。
  - `domain: str`：关键字参数，必填，实体所属领域。
  - `entity_type: str`：关键字参数，默认 `ENTITY_DEFAULT_TYPE`，实体类型。
  - `description: str`：关键字参数，默认空串，实体描述。
  - `confidence: float`：关键字参数，默认 `ENTITY_DEFAULT_CONFIDENCE`；作为 `importance` 传给写回函数。
  - `aliases: list[str] | None`：关键字参数，默认 `None`，本次抽取到的别名列表。
  - `source_id: str | None`：关键字参数，默认 `None`，来源记忆条目 id，用于建立来源关联。
- **返回**：返回 `str`，即写库后实体的 `canonical_name`（缺失时退回 `stored.content`）。名字为空时抛 `ValueError`。
- **内部流程**：先 `name = _clean_text(name, max_length=ENTITY_NAME_MAX_LENGTH)` 与 `key = _normalize_for_match(name)`，`key` 为空则 `raise ValueError("entity name must not be empty")`；接着 `existing = self._exact(key)` 并记录 `name_is_known = existing is not None`；若 `existing is None` 则尝试 `_prefix_candidate(key)`，命中时取该实体的书写名 `written`，若本次名字与 `written` 不同且 `key` 不在 `_index` 中，就调用 `_store_alias(existing, name)` 把新写法登记为别名（并用返回的最新条目替换 `existing`）；若仍为 `None`，遍历 `self._entities.values()`，对每个实体的规范名与已知别名调用 `_entity_similarity(name, canonical, known_aliases)`，一旦 `>= self.similarity_threshold` 就选中该实体并 `break`；随后在函数内 `from tool.graph_node_update import update_entity_node`，调用 `update_entity_node(self.manager, name, existing=existing, domain=domain, entity_type=entity_type, description=description, aliases=aliases, importance=confidence, source_id=source_id, add_written_name=name_is_known, item_id=existing.id if existing is not None else entity_id_for(name))` 得到 `stored`；再 `self._remember(stored)` 刷新索引；最后 `return str(stored.metadata.get("canonical_name") or stored.content)`。
- **异常/边界**：空名字抛 `ValueError`；`update_entity_node` 的异常向上传播；模糊匹配采用「第一个达到阈值的实体」而不是最高分实体（`break` 在遍历顺序上先到先得）；`existing` 为 `None` 时用 `entity_id_for(name)` 生成新 id；无超时处理。
- **同文件关系**：调用 `_clean_text`（外部）、`_normalize_for_match`、`_exact`、`_prefix_candidate`、`_store_alias`、`_entity_similarity`、`_remember`、`entity_id_for`（外部）以及外部 `tool.graph_node_update.update_entity_node`；被 `materialize_extraction`（直接与内部 `resolve_endpoint`）和 `build_graph_context` 间接使用。

### `materialize_extraction(manager: MemoryManager, extraction: ExtractionResult, *, source_item: MemoryItem, source_metadata: Mapping[str, Any] | None = None, relation_threshold: float = 0.6, resolver: EntityResolver | None = None) -> dict[str, Any]` （第 528 行）
- **作用**：把一次抽取结果确定性地物化进语义记忆，是「抽取」与「持久化」分离架构中的持久化端。它做两件事：先把所有实体候选交给 `EntityResolver.resolve` 落库并记录「名字 → 规范名」映射；再逐条处理关系候选，按 `action` 分派三种图变更——`assert` 新增或强化一条边，`supersede` 先把同一单值槽位的其它当前值标记失效再写入新值，`retract` 只把指定边标记失效而不删除历史。失效的边会以 `active=false` 保留在 SQLite 中形成审计轨迹，使当前状态查询只看到有效边、而历史查询仍可回溯。文档字符串还说明每个图变更都使用确定性 id，从而保证同一事实重复抽取时命中同一条记录而不是不断新增。函数最后返回一个统计字典供上层记录日志或展示进度。
- **参数**：
  - `manager: MemoryManager`：必填，位置参数，记忆管理器，提供 `get`、`semantic.add_fact`、`semantic.facts` 等接口。
  - `extraction: ExtractionResult`：必填，位置参数，本次抽取结果。
  - `source_item: MemoryItem`：关键字参数，必填，触发本次抽取的来源记忆条目（通常是文本块），其 `id` 会写入关系的 `source_ids` 与 `chunk_id`，`content` 会在缺少 evidence 时作为兜底证据，`modality` 会透传到事实元数据。
  - `source_metadata: Mapping[str, Any] | None`：关键字参数，默认 `None`；可提供 `filename`/`source`/`captured_at`/`modality` 等，用于填充事实元数据的来源与拍摄时间。
  - `relation_threshold: float`：关键字参数，默认 `0.6`；关系候选置信度低于该值会被跳过并计入 `skipped_relations`。
  - `resolver: EntityResolver | None`：关键字参数，默认 `None`；为空时现场构造一个 `EntityResolver(manager)`，传入时复用调用方的解析器（同一摄入批次的多个块共享一个实例以避免重复全量扫描）。
- **返回**：返回 `dict[str, Any]`，键为 `domain`（领域）、`topics`（主题列表）、`entities`（成功解析的实体计数）、`relations`（成功写入的有效关系计数）、`superseded`（被新值顶替而失效的旧关系计数）、`retracted`（被显式撤回的关系计数）、`skipped_relations`（因置信度不足或撤回目标不存在而跳过的计数）、`relation_items`（本次写入的关系 `MemoryItem` 列表）。
- **内部流程**：
  1. `metadata = dict(source_metadata or {})` 复制元数据，`source` 取 `filename` 或 `source` 或空串。
  2. `resolver` 为空时构造 `EntityResolver(manager)`；初始化 `canonical_by_key` 映射与 `entities` 计数。
  3. 遍历 `extraction.entities`，逐个调用 `resolver.resolve(...)`（带 domain、entity_type、description、confidence、aliases、`source_id=source_item.id`），把结果按 `_normalize_for_match(candidate.name)` 记入 `canonical_by_key`，`entities` 自增。
  4. 建立事实索引缓存 `known_items`（槽位键 → 该槽位所有事实列表）与懒加载标志 `facts_indexed`。
  5. 定义嵌套函数 `slot_key`（用 `normalize_entity_name` 拼接 `subject|predicate`）、`index_all_facts`（首次调用时遍历 `manager.semantic.facts()`，按 subject+predicate 建索引）、`remember`（把刚写入的条目在缓存中就地替换或追加）、`facts_for`（返回某槽位中 `active` 不为 `False` 的所有事实）、`resolve_endpoint`（先查 `canonical_by_key`，未命中再调 `resolver.resolve` 并只传 domain 与 source_id）。
  6. 初始化计数器 `relations`/`superseded`/`retracted`/`skipped_relations`、结果列表 `relation_items` 与当前时间字符串 `now`。
  7. 遍历 `extraction.relations`：置信度低于 `relation_threshold` 的直接 `skipped_relations += 1` 并 `continue`；否则解析主语与宾语规范名，把每个 role 的 `value` 也解析成规范名组成 `roles` 列表。
  8. 判断 `temporal`：只要 `event_at`、`valid_from`、`valid_to` 任一非空，或 `cardinality == "temporal"`，或存在 `roles`，就视为时序观察。
  9. 计算 `relation_id`：时序观察用 `observation_id_for(subject, predicate, object_name, roles=..., event_at=..., valid_from=..., valid_to=..., source_id=source_item.id)`，非时序用 `relation_id_for(subject, predicate, object_name)`；这样时序事实按观察维度生成独立 id，可追加历史而不互相覆盖。
  10. 用 `manager.get(relation_id, memory_type=MemoryType.SEMANTIC)` 取已有条目，读出 `existing_metadata`、累积 `source_ids` 集合与 `evidence_items` 列表；把 `source_item.id` 加入 `source_ids`，构造含 `source`/`chunk_id`/`evidence`（无 evidence 时截取 `source_item.content[:600]`）的证据记录并去重后追加。
  11. 若 `action == "retract"`：目标不存在则 `skipped_relations += 1` 并跳过；存在则复制元数据并写入 `active=False`、`superseded_at=now`、排序后的 `source_ids`、最近 20 条 `evidence_items`，用 `manager.semantic.add_fact(subject, predicate, object_name, metadata=..., confidence=float(existing.importance), item_id=relation_id)` 覆盖写回，`remember` 刷新缓存，`retracted += 1`，然后 `continue`。
  12. 计算是否需要「退休旧值」：`retire = candidate.action == "supersede" or (candidate.cardinality == "single" and not temporal)`；需要时遍历 `facts_for(subject, candidate.predicate)`，跳过自己（`stale.id == relation_id`）与已失效条目，对其余每条复制元数据、写入 `active=False`、`superseded_at=now`、`superseded_by=relation_id`，用相同 `item_id=stale.id` 覆盖写回并 `remember`，把 id 收集进 `superseded_by` 并递增 `superseded`。
  13. 构造新事实的 `fact_metadata`：包含 domain、topics、source、source_document、chunk_id、evidence、source_ids、evidence_items（保留最近 20 条）、`created_by="llm"`、`extraction_confidence`、`predicate_key`（由 `predicate_key_for` 生成）、action、cardinality、roles、`observation_id`（仅时序非空）、modality（来源条目的模态或元数据或 `"text"`）、captured_at、valid_from、valid_to、status（空值回落 `"fact"`）、event_at、`active=True`、空 `superseded_by`、空 `superseded_at`、以及 `supersedes=superseded_by`。
  14. 调用 `manager.semantic.add_fact(...)` 写入（置信度用 `candidate.confidence`，`item_id=relation_id`），`remember(written)`、追加进 `relation_items`、`relations += 1`。
  15. 循环结束后返回统计字典。
- **异常/边界**：`relation_threshold` 与 `resolver` 的类型不做校验；`source_metadata` 为 `None` 安全处理；`source_item.id` 为空时 `if source_item.id:` 会跳过加入 `source_ids`（但 `chunk_id` 仍写入空值）；`retract` 指向不存在的关系时静默跳过并计数；`manager.semantic.add_fact`、`manager.get`、`resolver.resolve` 的异常向上传播；`evidence_items` 通过 `[-20:]` 限制长度防止元数据无限膨胀；`roles` 中每个 `value` 都会被解析成实体，因此非法角色值可能触发 `resolve` 的 `ValueError`。
- **同文件关系**：调用本文件的 `EntityResolver`（构造与 `resolve`）、`_normalize_for_match`、`ExtractionResult`（类型）、以及外部 `normalize_entity_name`、`observation_id_for`、`relation_id_for`、`predicate_key_for`、`utc_now`、`MemoryType`；被上层摄入流程调用；在 `__all__` 中导出。

#### `materialize_extraction.slot_key(subject: str, predicate: str) -> str` （第 577 行）
- **作用**：这是 `materialize_extraction` 内部的嵌套函数，把「主语 + 谓词」压缩成一个字符串键，用作事实缓存的槽位标识。之所以需要它，是因为 supersede 与 single 基数的语义都是「同一 (subject, predicate) 槽位只能有一个当前值」，而缓存字典需要一个可哈希的键。它用 `normalize_entity_name` 而不是 `_normalize_for_match` 做归一，因为槽位判定关心的是「是不是同一个逻辑槽位」，此时抹掉标点、追求稳定 id 的归一方式更合适。
- **参数**：
  - `subject: str`：主语（通常是已解析的规范名），无默认值。
  - `predicate: str`：谓词，无默认值。
- **返回**：返回 `str`，形如 `"{归一化主语}|{归一化谓词}"`；两个参数都可能被归一成空串但仍会返回含分隔符的字符串。
- **内部流程**：单行 f-string，调用两次 `normalize_entity_name` 并用竖线拼接。
- **异常/边界**：无特殊处理，不抛异常；空值由 `normalize_entity_name` 容错。
- **同文件关系**：调用外部 `normalize_entity_name`；被同层的 `index_all_facts`、`remember`、`facts_for` 调用。

#### `materialize_extraction.index_all_facts() -> None` （第 580 行）
- **作用**：惰性构建「槽位 → 该槽位全部事实」的索引。它只在第一次被调用时真正遍历 `manager.semantic.facts()`，之后靠 `facts_indexed` 标志直接返回，从而避免处理一条关系就全表扫描一次。索引里保存的是**全部**事实（包括已失效的），因为 single 值槽位在历史数据或并发抽取下可能同时存在多条 active 行，supersede 必须把它们全部退休，而不是只退休缓存里第一条。这正是文档注释强调的点。
- **参数**：无参数（闭包捕获 `manager`、`known_items`、`facts_indexed`）。
- **返回**：返回 `None`，副作用是填充 `known_items` 并把 `facts_indexed` 置为 `True`。
- **内部流程**：先 `nonlocal facts_indexed`；若 `facts_indexed` 为真直接 `return`；否则先把它置 `True`（防止重入或异常后重复扫描），再遍历 `manager.semantic.facts()`，从每条元数据里取 `subject` 与 `predicate`，两者都非空时用 `known_items.setdefault(slot_key(subject, predicate), []).append(item)` 归组。
- **异常/边界**：若 `manager.semantic.facts()` 抛错，`facts_indexed` 已被置为 `True`，后续调用不会重试（索引可能不完整）；缺少 subject 或 predicate 的条目被静默忽略。
- **同文件关系**：调用同层 `slot_key`；被 `remember` 与 `facts_for` 调用。

#### `materialize_extraction.remember(item: MemoryItem) -> None` （第 591 行）
- **作用**：在本次物化过程中每次写库后刷新缓存副本。它保证同一块文本里后续的关系判断（尤其是 supersede 要退休哪些旧值）能看到前面刚写入或刚失效的条目，而不会读到过期的快照对象。它先确保索引已建（调用 `index_all_facts`），然后在对应槽位列表里按 id 就地替换；如果没找到同 id 条目就追加，覆盖「这条事实是本次新写入的、索引建立时还不存在」的情况。
- **参数**：
  - `item: MemoryItem`：刚写入或刚更新的语义条目，无默认值。
- **返回**：返回 `None`，副作用是就地修改 `known_items` 中对应槽位的列表。
- **内部流程**：取 `subject` 与 `predicate`（缺失任一则直接 `return`）；调用 `index_all_facts()` 确保索引存在；用 `known_items.setdefault(slot_key(subject, predicate), [])` 拿到槽位列表；`for position, cached in enumerate(slot):` 比较 `cached.id == item.id`，命中则 `slot[position] = item` 并 `return`；循环结束仍未命中则 `slot.append(item)`。
- **异常/边界**：条目缺少 subject 或 predicate 时静默不缓存；同一 id 在列表中只替换第一条；无异常抛出。
- **同文件关系**：调用同层 `index_all_facts` 与 `slot_key`；被 `materialize_extraction` 主体在每次 `add_fact` 之后调用。

#### `materialize_extraction.facts_for(subject: str, predicate: str) -> list[MemoryItem]` （第 605 行）
- **作用**：返回某个 (subject, predicate) 槽位下所有**当前有效**的事实。它是 supersede 退休逻辑的输入：调用方拿到这份列表后逐条排除自己并标记失效。过滤条件写成 `item.metadata.get("active", True) is not False`，意味着「元数据里没有 active 键」的历史数据被视为有效，只有显式 `active=False` 才排除，从而兼容旧数据。
- **参数**：
  - `subject: str`：主语名，无默认值。
  - `predicate: str`：谓词名，无默认值。
- **返回**：返回 `list[MemoryItem]`，槽位不存在时返回空列表；元素顺序与索引建立时 `manager.semantic.facts()` 的遍历顺序一致。
- **内部流程**：先 `index_all_facts()` 保证索引可用；再用列表推导遍历 `known_items.get(slot_key(subject, predicate), [])`，只保留 `item.metadata.get("active", True) is not False` 的条目。
- **异常/边界**：无匹配槽位返回 `[]`；不抛异常；不修改任何状态。
- **同文件关系**：调用同层 `index_all_facts` 与 `slot_key`；被 `materialize_extraction` 的 retire 分支调用。

#### `materialize_extraction.resolve_endpoint(name: str) -> str` （第 614 行）
- **作用**：把关系里的一个端点名字解析成规范名，是 `resolver.resolve` 的带缓存包装。它先查本次抽取已经建立的 `canonical_by_key` 映射（实体阶段刚解析过的名字都在里面），命中就直接返回，避免对同一名字重复调用 `resolve` 触发多余的写库与属性合并；未命中时才回退到 `resolver.resolve`，并且只传 `domain` 与 `source_id` 两个参数（关系端点没有自己的类型和描述信息）。它被用于解析主语、宾语以及每个 role 的取值。
- **参数**：
  - `name: str`：端点名字（可能是模型新提到的、实体阶段未出现的名字），无默认值。
- **返回**：返回 `str`，即该端点对应的实体规范名。
- **内部流程**：`cached = canonical_by_key.get(_normalize_for_match(name))`；若 `cached` 为真值则直接返回；否则 `return resolver.resolve(name, domain=extraction.domain, source_id=source_item.id)`。注意它不把新解析结果写回 `canonical_by_key`，因此同一新端点在同一次物化里多次出现时会多次调用 `resolve`（由 `resolver` 自身的索引保证幂等）。
- **异常/边界**：`name` 为空或纯空白时，`canonical_by_key` 查不到，`resolver.resolve` 会抛 `ValueError("entity name must not be empty")`；`resolver.resolve` 的其它异常同样向上传播。
- **同文件关系**：调用本文件的 `EntityResolver.resolve` 与 `_normalize_for_match`；被 `materialize_extraction` 主体调用三次场景（主语、宾语、roles 取值）。

### `build_graph_context(manager: MemoryManager, text: str, *, resolver: EntityResolver | None = None, max_relations: int = GRAPH_CONTEXT_MAX_RELATIONS, max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str` （第 794 行）
- **作用**：为抽取提示词渲染「已有知识图的相关切片」。它的工作方式是：从待抽取文本里找出所有能匹配到已有实体的词与整句前缀，得到种子实体；再把种子的名字沿已有事实做一跳扩展，收集邻居名字；然后渲染出三段文本——已知领域清单（提醒模型复用现有领域）、已知实体清单（带别名，要求模型必须复用规范名）、已知关系清单（当前有效的关系按置信度降序排列，附上少量已失效的历史关系并标注「历史，已失效」）。这样模型在抽取时就能复用规范名、并且知道该退休哪个旧值，而不是凭空造出第二个同名实体。文档字符串强调匹配策略是刻意保守的，并且 `resolver` 可注入以避免每块文本重复全量扫描实体。
- **参数**：
  - `manager: MemoryManager`：必填，位置参数，记忆管理器，用于遍历事实与构造默认解析器。
  - `text: str`：必填，位置参数，待抽取的原始文本；为 `None` 或空白时返回空串。
  - `resolver: EntityResolver | None`：关键字参数，默认 `None`；为空时现场构造 `EntityResolver(manager)`。
  - `max_relations: int`：关键字参数，默认 `GRAPH_CONTEXT_MAX_RELATIONS`；必须是正整数，否则抛 `ValueError`。
  - `max_chars: int`：关键字参数，默认 `RAG_CONTEXT_MAX_CHARS`；必须是正整数，否则抛 `ValueError`。
- **返回**：返回 `str`：渲染好的多行上下文文本（最终经 `_fit_lines` 按整行截断）；文本为空、或没有任何种子实体命中时返回空串 `""`。
- **内部流程**：
  1. 参数校验：若 `max_relations` 或 `max_chars` 是 `bool`、不是 `int`、或小于 1，则 `raise ValueError("max_relations and max_chars must be positive integers")`（显式排除 `bool`，因为 `True` 在 Python 里也是 `int`）。
  2. `resolver` 为空时构造默认实例；初始化 `seeds` 字典与 `folded = (text or "").casefold()`；若 `not folded.strip()` 直接返回空串。
  3. 用 `re.findall(r"[\w\u4e00-\u9fff]+", folded)` 把文本切成词，对每个词调用 `resolver.match(word)`，命中的按 `item.id` 去重放入 `seeds`。
  4. 整句前缀补充：遍历 `resolver._index.items()`，若 `is_prefix_match(_normalize_for_match(folded), key)` 成立（覆盖名字本身含分隔符的情况），把对应条目加入 `seeds`。
  5. 若 `seeds` 为空返回空串。
  6. 用种子条目的 `canonical_name`（或 `content`）构造 `names` 集合。
  7. 遍历 `manager.semantic.facts()`，跳过 `active` 为 `False` 的条目，取 subject 与 object，只要其中任一名在 `names` 里就把两者都加入 `names`（实现一跳邻居扩展）。
  8. 调用 `_known_domains()` 得到已知领域列表，渲染第一行「已知领域（优先复用，不要随意新建）：…」。
  9. 渲染「已知实体（必须复用，不要新造）：」标题，并为每个种子输出 `- 规范名；别名：a、b` 形式（无别名时不带后缀）。
  10. 初始化 `relations` 与 `retired` 两个 `(置信度, 行文本)` 列表；再次遍历 `manager.semantic.facts()`，跳过缺少 subject/predicate/object 的条目，跳过两端都不在 `names` 里的条目；生成 `subject --predicate--> object` 行，置信度取 `metadata["confidence"]`（缺失时用 `item.importance`，再兜底 0.0）；`active=False` 的进入 `retired` 并加后缀「（历史，已失效）」，否则进入 `relations`。
  11. 两个列表都按置信度降序排序；输出「已知关系（当前有效）：」标题，先追加 `relations[:max_relations]`，再追加 `retired[: max(1, max_relations // 3)]`（失效关系最多取有效配额的三分之一，且至少 1 条）。
  12. 最后 `return _fit_lines(lines, max_chars)`。
- **异常/边界**：`max_relations`/`max_chars` 非法时抛 `ValueError`；`text` 为 `None` 时由 `(text or "")` 兜底并返回空串；无种子命中返回空串；`resolver._index` 是私有属性却被直接遍历，若传入的是自定义对象需具备该属性；事实条目缺字段时被跳过；`float(metadata.get("confidence", item.importance) or 0.0)` 对 `None` 或非数值字符串可能抛 `ValueError`/`TypeError`。
- **同文件关系**：调用本文件的 `EntityResolver`（构造与 `match`）、`is_prefix_match`、`_normalize_for_match`、`_known_domains`、`_fit_lines`；被上层摄入流程调用；在 `__all__` 中导出。

### `_known_domains() -> list[str]` （第 892 行）
- **作用**：返回项目当前认可的领域名清单，供 `build_graph_context` 渲染「已知领域」提示，让 LLM 优先复用现有领域而不是为同一主题新造名字。清单的真实来源是 `tool.domain_classify` 模块的 `KNOWN_DOMAINS` 常量，这里做了函数内导入，是为了避免在导入本模块时就拉起 `tool.domain_classify`（可能造成模块初始化顺序问题）。导入失败时返回单元素列表 `["未分类"]` 作为常规兜底集，保证调用方永远拿到一个可迭代的非空列表。
- **参数**：无参数。
- **返回**：返回 `list[str]`：正常情况下是 `KNOWN_DOMAINS` 的浅拷贝；导入失败时返回 `["未分类"]`。
- **内部流程**：`try` 块内 `from tool.domain_classify import KNOWN_DOMAINS`，成功后 `return list(KNOWN_DOMAINS)`（复制一份避免调用方修改原常量）；`except Exception` 捕获任何异常（含 `ImportError`）并 `return ["未分类"]`，注释说明这是「顺便导入失败时返回常规集」。
- **异常/边界**：所有异常都被宽泛捕获（带 `# noqa: BLE001`），因此本函数自身不会抛异常；若 `KNOWN_DOMAINS` 不是可迭代对象，`list()` 会抛 `TypeError`（未被捕获）。
- **同文件关系**：被 `build_graph_context` 调用；自身只做外部导入。

### `_fit_lines(lines: list[str], max_chars: int) -> str` （第 902 行）
- **作用**：把渲染好的多行文本按字符预算裁剪并拼接。它只保留完整的行，绝不做半行截断——文档字符串解释了原因：一条被砍掉一半的关系描述会误导抽取器，让它以为存在一条不完整的事实。因此它在遇到第一条放不下的行时就停止，返回前面所有放得下的行。每行的成本按「行长度 + 换行符占位」计算（除第一行外每行多算 1 个字符），从而精确反映最终 `"\n".join` 后的长度。
- **参数**：
  - `lines: list[str]`：待裁剪的行列表，按重要性从高到低排列（调用方已排好序），无默认值。
  - `max_chars: int`：字符预算上限，无默认值；由 `build_graph_context` 保证是正整数。
- **返回**：返回 `str`，即被保留行用换行符拼接的结果。若第一行本身就超出预算，返回空字符串。
- **内部流程**：初始化 `kept = []` 与 `used = 0`；遍历每一行，计算 `cost = len(line) + (1 if kept else 0)`（非首行加一个换行符开销）；若 `used + cost > max_chars` 则 `break` 停止；否则追加该行并累加 `used`；最后 `return "\n".join(kept)`。
- **异常/边界**：`max_chars` 为负数或 0 时循环在第一次判断就 `break`，返回空串；`lines` 为空返回空串；不抛异常；不做字符宽度或字节长度区分（按 Python 字符数计算）。
- **同文件关系**：只被 `build_graph_context` 调用；自身不调用本文件其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_normalize_for_match` | 把字符串规范化成保留词分隔符的匹配键，用于实体名命中判断。 |
| `_starts_at_boundary` | 判断短名字是否是长名字的词边界前缀（后一字符必须是分隔符或两者等长）。 |
| `is_prefix_match` | 对外的前缀匹配入口，先按长度排序并施加最小长度阈值，再委托 `_starts_at_boundary`。 |
| `EntityCandidate` | LLM 提议的实体节点数据契约（名字、类型、描述、置信度、别名）。 |
| `EntityCandidate.normalize_strings` | 对 `name`/`entity_type`/`description` 做前置清洗与 1000 字符截断。 |
| `RelationRole` | n 元观察中一个额外参与者的数据契约（角色名、取值、实体类型）。 |
| `RelationRole.normalize_strings` | 对 `role`/`value`/`entity_type` 做前置清洗与 200 字符截断。 |
| `RelationCandidate` | LLM 提议的关系补丁契约，含动作、基数、角色、时间、状态、置信度与证据。 |
| `RelationCandidate.normalize_strings` | 对主语/谓语/宾语/证据/三个时间字段做前置清洗与 1200 字符截断。 |
| `RelationCandidate.normalize_status` | 把 `status` 的 `None` 转成空串并截断到 20 字符，交给枚举校验。 |
| `ExtractionResult` | 一次抽取的顶层结果容器（领域、主题、实体、关系、关键词）。 |
| `ExtractionResult.normalize_domain` | 清洗领域名，空值时回落到 `DEFAULT_DOMAIN`。 |
| `ExtractionResult.normalize_lists` | 把 `topics`/`keywords` 的 `None` 变空列表、拒绝非列表、逐项清洗并剔除空项。 |
| `KnowledgeExtractor` | 抽取器的结构化类型协议，只声明 `extract` 签名。 |
| `KnowledgeExtractor.extract` | 协议方法声明：文本 + 元数据 + 图上下文 + 可选图片，返回 `ExtractionResult`。 |
| `NullKnowledgeExtractor` | 未配置聊天模型时的空实现，永远返回空抽取结果。 |
| `NullKnowledgeExtractor.extract` | 忽略全部输入，直接返回默认 `ExtractionResult()`。 |
| `response_content` | 从字符串/字典/对象三种形态的聊天响应里取出助手文本。 |
| `parse_json_object` | 容错解析模型返回的 JSON 对象（剥离 ```json 围栏、截取最外层花括号）。 |
| `LLMKnowledgeExtractor` | 通过 OpenAI 兼容客户端做知识抽取的实现，内含中文 `SYSTEM_PROMPT` 规则手册。 |
| `LLMKnowledgeExtractor.__init__` | 保存 `complete` 客户端、文本模型、视觉模型与超时，并校验可调用性。 |
| `LLMKnowledgeExtractor.extract` | 组装 JSON 载荷与多模态消息、调用模型、解析并校验成 `ExtractionResult`。 |
| `LLMKnowledgeExtractor._image_data_url` | 把图片字节编码成 base64 data URI，或原样返回图片 URL。 |
| `_entity_similarity` | 取「序列相似度最大值」与「词元 Jaccard 重叠率」的较大者作为实体相似度。 |
| `EntityResolver` | 把名字按精确名/别名/前缀/模糊四段解析到稳定实体记录。 |
| `EntityResolver.__init__` | 保存 manager 与阈值，初始化两个索引字典并立即全量载入实体。 |
| `EntityResolver._load` | 清空索引后遍历语义记忆，只对 `kind == "entity"` 的条目重建索引。 |
| `EntityResolver._remember` | 把一个实体按其规范名与全部别名登记进两个索引字典。 |
| `EntityResolver._exact` | 先查规范名索引再查别名索引的精确查找。 |
| `EntityResolver._prefix_candidate` | 在所有已知键中挑出共享长度最长的词边界前缀命中项（只读）。 |
| `EntityResolver._store_alias` | 把新写法登记为已有实体的别名，并把索引中指向旧快照的键改指新对象。 |
| `EntityResolver.match` | 只读地做精确 + 前缀匹配，返回实体或 `None`。 |
| `EntityResolver.resolve` | 四段匹配后调用 `update_entity_node` 落库，返回规范名。 |
| `materialize_extraction` | 把抽取结果确定性地写库：实体解析 + assert/supersede/retract 关系补丁。 |
| `materialize_extraction.slot_key` | 用归一化的主语与谓词拼出事实槽位键。 |
| `materialize_extraction.index_all_facts` | 惰性遍历全部事实，建立「槽位 → 该槽位所有事实」索引。 |
| `materialize_extraction.remember` | 每次写库后就地替换或追加缓存中的对应事实条目。 |
| `materialize_extraction.facts_for` | 返回某槽位下所有当前有效（`active` 不为 `False`）的事实。 |
| `materialize_extraction.resolve_endpoint` | 先查本次抽取的规范名缓存，未命中再调 `resolver.resolve` 解析关系端点。 |
| `build_graph_context` | 从已有图里捞出与文本相关的实体与一跳邻居，渲染成抽取提示词上下文。 |
| `_known_domains` | 返回已知领域清单（来自 `tool.domain_classify.KNOWN_DOMAINS`），失败时给 `["未分类"]`。 |
| `_fit_lines` | 按字符预算只保留完整行并拼接，避免截断出半条关系。 |
