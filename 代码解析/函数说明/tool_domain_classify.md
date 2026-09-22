# tool/domain_classify.py

## 一、这个文件是干什么的

这个文件实现了一个「本地主题领域分类」能力：给定一段正文（可选再给一个标题），用纯关键词规则把它归到八个预设恒星系领域之一（编程开发、数学、物理、化学、生物医学、历史人文、经济管理、文学艺术），如果一条关键词都没命中，就返回兜底领域名。

它是这段逻辑的**唯一实现**：文件头注释明确说明，原先这份逻辑藏在 Web 层的私有模块 `web/domain_classifier.py`（已删除），Agent 无法调用；现在它被抽成本工具，`web/app.py` 的图接口与 `memory/rag/knowledge.build_graph_context` 都从这里取用，于是星云图挂载知识块、文档按多数领域归属、图检索往提示词里注入已知领域清单这三处复用同一份确定性实现。

文件里包含四类东西：一是模块级常量（`TOOL_ENABLED` 开关、`DOMAIN_KEYWORDS` 领域关键词表、`KNOWN_DOMAINS` 领域清单、`__all__` 导出清单）；二是两个纯函数 `classify_domain`（单段文本打分归类）与 `majority_domain`（多领域投票取众数）；三是两个 Pydantic 输入/输出模型 `ClassifyDomainInput`、`ClassifyDomainOutput`；四是一个继承 `BaseTool` 的工具类 `ClassifyDomainTool`（内部方法 `execute` 调用纯函数）以及工厂函数 `create_tool`。

设计约束是零依赖、纯规则、确定性：不联网、不消耗 LLM key、未配置任何云端 key 也能用，同文本永远同结果，可跨进程稳定复现；词表与权重都放在 `constants.py` 里（本文件只导入 `DEFAULT_DOMAIN`、`DOMAIN_TITLE_WEIGHT`）。分类一次只遍历一遍关键词表，万级节点构图时开销可忽略。注意它只做分类、不写库，需要落库的分类结果要走 `memory.rag` 或 `knowledge.hybrid_index` 的元数据。

## 二、函数与类逐条详解

### `classify_domain(text: str, *, title: str = "", default: str = DEFAULT) -> str` （第 84 行）

- **作用**：这是整个文件的核心打分函数，负责把一段正文文本归类到最匹配的领域并返回领域名字符串。它遍历 `DOMAIN_KEYWORDS` 里的每一个领域，统计该领域的关键词在文本中出现了多少次，得到该领域的得分，最后把得分最高的领域作为答案返回。为了让标题（往往是文件名或文档标题，主题词密度更高）发挥更大作用，标题里命中关键词会额外加权，权重取 `DOMAIN_TITLE_WEIGHT`，而不是记 1 分。它是纯函数：不读写全局状态、不联网、不依赖随机数，所以同一段输入永远得到同一个结果，这正是星云图挂恒星、文档归属、提示词注入领域清单这三处能放心复用的原因。Agent 侧通过 `ClassifyDomainTool.execute` 间接调用它，Web 层与记忆层则可以直接 import 调用。
- **参数**：
  - `text`：`str`，要分类的正文。可以为空字符串；函数内部会做 `(text or "").lower()` 的兜底与大小写归一化，所以传 `None`（类型标注之外的情况）也不会在这里崩。
  - `title`：`str`，关键字参数（`*` 之后，必须用 `title=` 传），默认空字符串。表示标题或文件名，命中关键词时按 `DOMAIN_TITLE_WEIGHT` 加权。
  - `default`：`str`，关键字参数，默认值是从 `constants` 导入的 `DEFAULT_DOMAIN`（在本文件里被别名为 `DEFAULT`）。当没有任何领域得分大于 0，或 `text` 与 `title` 同时为空时，返回这个值。从 `ClassifyDomainOutput` 的字段说明可知，这个兜底领域名的字面值是「未分类」。
- **返回**：返回 `str`，即命中的领域名，取值来自 `DOMAIN_KEYWORDS` 的键（也就是 `KNOWN_DOMAINS` 里的某一项），或者是传入/默认的 `default`。三种情况下返回 `default`：`text` 和 `title` 都为空；所有领域关键词都没命中（全部得分为 0）；所有领域得分都是 0 之外的极端情形其实不会发生，因为只有 `score > best_score` 才会替换，而初始 `best_score` 为 0，所以得分必须严格大于 0 才可能被选中。
- **内部流程**：第一步做空输入短路——`if not text and not title: return default`，避免无谓遍历；第二步把 `text` 和 `title` 分别转小写存入 `text_lower`、`title_lower`，实现大小写不敏感匹配；第三步初始化 `best_domain = default`、`best_score = 0`；第四步外层 `for domain, keywords in DOMAIN_KEYWORDS.items()` 逐个领域遍历，每个领域内先把 `score` 归零，再内层 `for kw in keywords` 逐词判断——把关键词也转小写后，若 `kw_lower in title_lower` 就 `score += DOMAIN_TITLE_WEIGHT`（注意是 `if/elif` 结构：标题命中后不再对同一关键词重复计正文分），否则若 `kw_lower in text_lower` 就 `score += 1`；第五步若该领域 `score > best_score`，就用它更新 `best_domain` 与 `best_score`（严格大于，意味着并列时先出现的领域胜出，顺序即 `DOMAIN_KEYWORDS` 的字典插入顺序：编程开发 → 数学 → 物理 → 化学 → 生物医学 → 历史人文 → 经济管理 → 文学艺术）；第六步返回 `best_domain`。
- **异常/边界**：不做显式抛异常，也没有 `try/except`。空字符串、空标题走短路直接返回 `default`；非字符串输入（如 `None`）在 `(text or "")` 处被兜成空串，`title` 同理，因此不会因 `None` 抛 `AttributeError`；非字符串但真值的输入（如数字）会在 `.lower()` 处抛 `AttributeError`，本文件未做类型校验。超时、缺失数据在此没有概念（纯内存计算，无 IO）。得分并列时按词表顺序取第一个，不做二次比较，这是确定性的来源之一。
- **同文件关系**：被 `ClassifyDomainTool.execute`（第 158 行）调用；它自身只使用本文件的模块级常量 `DOMAIN_KEYWORDS` 与导入的 `DOMAIN_TITLE_WEIGHT`、默认参数 `DEFAULT`，不调用本文件里任何其它函数。

### `majority_domain(domains: list[str], *, default: str = DEFAULT) -> str` （第 109 行）

- **作用**：对一组已经分好类的领域名做「多数投票」，返回出现次数最多的那个领域。使用场景是：一个文档往往由多个知识块组成，每块各自有一个领域，要把整个文档实体挂到某个领域恒星上时，就取这些块领域的众数，让文档归属与大多数块保持一致。它是与 `classify_domain` 配套的聚合函数：前者解决「一段文本属于哪」，后者解决「一堆归属里以哪个为准」。同样零依赖、确定性。
- **参数**：
  - `domains`：`list[str]`，待投票的领域名列表，元素通常是 `classify_domain` 的返回值，也允许混入任意字符串（如兜底值「未分类」）。允许为空列表。
  - `default`：`str`，关键字参数，默认 `DEFAULT`（即 `constants.DEFAULT_DOMAIN`）。当 `domains` 为空时返回它。
- **返回**：返回 `str`。列表非空时返回出现次数最多的领域名；列表为空时返回 `default`。并列最多时返回 `max` 在遍历字典项时先遇到的那个，而字典 `counts` 的键顺序是各领域首次出现的顺序，所以并列时的胜者取决于输入列表中谁先出现，结果依旧确定。
- **内部流程**：第一步 `if not domains: return default` 处理空输入（空列表以及类型标注之外的空值如 `None` 都走这条）；第二步建空字典 `counts: dict[str, int]`；第三步 `for d in domains` 逐个累加 `counts[d] = counts.get(d, 0) + 1`，把列表压成「领域 → 出现次数」的计数字典；第四步 `max(counts.items(), key=lambda kv: kv[1])[0]` 取计数值最大的那一项的键并返回。
- **异常/边界**：不主动抛异常。空输入返回 `default`；列表元素若为不可哈希类型（例如列表或字典）会在 `counts[d]` 处抛 `TypeError`，本文件未做校验；`None` 作为 `domains` 会因 `not domains` 为真而直接返回 `default`，不会崩。并列情形不做报错也不做二次裁决，静默按首次出现顺序返回。无 IO，因此无超时问题。
- **同文件关系**：不调用本文件里的任何其它函数；在本文件内也未被其它函数调用（它是对外导出的公开函数，供 Web 层/记忆层的文档归属逻辑使用）。它使用模块级的默认参数 `DEFAULT`。

### `class ClassifyDomainInput(BaseModel)` （第 120 行）

- **作用**：这是 `knowledge.classify_domain` 工具的入参模型，用 Pydantic 定义并做校验，让 Agent 的调用参数有明确契约。它把工具输入限定为两个可选的字符串字段，并用 `strict=True` 强制类型严格匹配，用 `extra="forbid"` 拒绝任何未声明的多余字段，从而避免模型幻觉出无关参数、或在运行时因隐式类型转换产生难以排查的行为。类本身不写任何方法，字段定义就是它的全部内容。
- **参数**：无自定义 `__init__`，实例化由 Pydantic 的 `BaseModel` 生成，接受两个字段：
  - `text`：`str`，默认 `""`，说明是「要分类的正文（可与 title 一起给，也可只给一个）」。
  - `title`：`str`，默认 `""`，说明是「标题或文件名；命中关键词时加权（文件名常含主题词）」。
- **返回**：类本身不返回运行结果；它被实例化后作为 `ClassifyDomainTool.execute` 的 `arguments` 参数传入，并从中读取 `.text` 与 `.title`。
- **内部流程**：类体内只有 `model_config = ConfigDict(extra="forbid", strict=True)` 与两个 `Field(...)` 声明；Pydantic 在实例化时据此收集并校验字段、拒绝多余键、执行严格类型检查（因此传数字给 `text` 会被拒绝而不是被强转成字符串），并自动生成 `model_json_schema` 之类的元信息供工具框架使用。
- **异常/边界**：字段类型不符（例如 `text=123`）、出现未声明字段（例如 `text="x", foo=1`）时，Pydantic 在实例化阶段抛 `ValidationError`，本文件不捕获，由工具框架统一处理为参数错误。两个字段都有默认值，因此 `ClassifyDomainInput()` 是合法的空输入，此时 `execute` 会把空文本交给 `classify_domain`，最终得到兜底领域。
- **同文件关系**：被 `ClassifyDomainTool.spec`（第 144 行 `input_model=ClassifyDomainInput`）引用；`ClassifyDomainTool.execute` 的类型标注使用它。它不调用本文件里的任何函数。

### `class ClassifyDomainOutput(BaseModel)` （第 127 行）

- **作用**：这是工具的出参模型，规定 `knowledge.classify_domain` 的返回值结构，让调用方（Agent、Web 接口）拿到的不只是一个裸字符串，而是带上下文的确定性结构。除了领域名本身，它还显式告诉调用方「这次到底有没有命中关键词」，以及「本地一共支持哪些领域」，避免模型把兜底值「未分类」误当成一个真实主题领域去使用。与输入模型一样，它只声明字段，不写自定义方法。
- **参数**：无自定义 `__init__`，由 Pydantic 生成，三个字段：
  - `domain`：`str`，必填（无默认值），说明是「命中的领域名；无命中时为『未分类』」。
  - `matched`：`bool`，必填，说明是「false 表示所有领域关键词都没命中，domain 是兜底值」。
  - `known_domains`：`list[str]`，默认由 `default_factory=list` 生成空列表，说明是「全部可选领域」；实际调用中由 `execute` 填入 `list(KNOWN_DOMAINS)`。
- **返回**：类本身不返回值；实例被 `execute` 构造并作为工具调用的结果返回。
- **内部流程**：类体内只有 `model_config = ConfigDict(extra="forbid", strict=True)` 和三个 `Field(...)` 声明，其中 `known_domains` 用 `default_factory=list` 而不是可变默认值 `[]`，避免所有实例共享同一个列表对象。Pydantic 在构造时校验 `domain` 与 `matched` 必须给出，并做严格类型检查。
- **异常/边界**：缺少必填字段 `domain` 或 `matched`、字段类型不符（例如 `matched="yes"`）、或出现多余字段时，构造阶段抛 `ValidationError`；本文件不捕获。`known_domains` 可省略，省略时为空列表。无 IO、无超时概念。
- **同文件关系**：被 `ClassifyDomainTool.spec`（第 145 行 `output_model=ClassifyDomainOutput`）引用，并在 `ClassifyDomainTool.execute` 中被实例化作为返回值。它不调用本文件里的任何函数。

### `class ClassifyDomainTool(BaseTool)` （第 135 行）

- **作用**：这是把纯函数 `classify_domain` 包装成 Agent 可调用工具（tool）的适配类。它继承框架的 `BaseTool`，通过类属性 `spec` 向框架声明工具的元信息：名称 `knowledge.classify_domain`、自然语言描述、版本、输入/输出模型、副作用等级、权限、超时、幂等性、并行安全性与标签、给模型的使用指引。这样 Agent 在工具列表里能看到「可以离线、确定性地把文本归到本地领域体系」，并按统一契约调用它。类本身只承载声明与一个执行方法，真正的分类逻辑仍在模块级纯函数里。
- **参数**：类没有自定义 `__init__`，由 `BaseTool` 提供；实例化时不接收分类参数。分类参数通过 `execute` 的 `arguments: ClassifyDomainInput` 传入。类属性 `spec` 的关键取值：`name="knowledge.classify_domain"`；`description` 为英文说明，点明是本地主题领域分类、确定性离线关键词规则、无命中返回「未分类」、可用于决定笔记属于哪个恒星系；`version="1.0.0"`；`input_model=ClassifyDomainInput`；`output_model=ClassifyDomainOutput`；`side_effect="read"`（只读、不写状态）；`permissions=()`（不需要任何额外权限）；`timeout_seconds=30.0`；`idempotent=True`；`parallel_safe=True`；`tags=("knowledge", "domain", "classify", "read")`；`guidance` 为中文使用指引，强调纯规则离线可复现、只分类不写库（要落库请走 `memory.rag` 或 `knowledge.hybrid_index` 的元数据），以及「不属于任何已知领域时返回未分类是正确结果，不要强行套一个领域」。
- **返回**：类作为可实例化类型被 `create_tool` 返回；它的实例作为工具对象注册进框架。
- **内部流程**：类体先定义 `spec = ToolSpec(...)`，把上述元信息一次性声明好；随后定义 `execute` 方法承载调用逻辑。整个类没有任何可变实例状态，配合 `parallel_safe=True` 与 `idempotent=True` 的声明，保证并发调用与重复调用都安全。
- **异常/边界**：类本身不做异常处理；参数校验失败由 Pydantic 在 `ClassifyDomainInput` 构造阶段抛 `ValidationError`，执行超时由框架依据 `timeout_seconds=30.0` 约束（本实现是纯内存计算，实际远低于该上限）。`TOOL_ENABLED = True`（第 30 行）是本模块的模块级启用开关，供加载方判断是否注册该工具。
- **同文件关系**：`spec` 引用同文件的 `ClassifyDomainInput`、`ClassifyDomainOutput`；`execute` 调用同文件的 `classify_domain`，并读取同文件的 `KNOWN_DOMAINS` 与导入的 `DEFAULT`；它被同文件的 `create_tool` 实例化。

### `ClassifyDomainTool.execute(self, arguments: ClassifyDomainInput) -> ClassifyDomainOutput` （第 158 行）

- **作用**：这是工具的实际执行入口，Agent 触发 `knowledge.classify_domain` 时框架调用的就是它。它只做三件事：把入参里的 `text` 与 `title` 交给纯函数 `classify_domain` 得到领域名；用「领域名是否等于兜底值 `DEFAULT`」反推这次有没有命中关键词；把完整的本地领域清单一并放进返回值。这样调用方既能拿到分类结果，也能明确区分「真的命中了某个领域」和「什么都没命中、给的是兜底值」这两种语义上完全不同的情况。
- **参数**：
  - `self`：`ClassifyDomainTool` 实例，方法内部不使用任何实例状态，因此天然可并发。
  - `arguments`：`ClassifyDomainInput`，已经过 Pydantic 校验的入参对象，提供 `.text`（正文，默认空串）与 `.title`（标题或文件名，默认空串）。
- **返回**：返回 `ClassifyDomainOutput` 实例，三个字段分别是：`domain` 为 `classify_domain` 的返回值（命中领域名或兜底值）；`matched` 为布尔值，`domain != DEFAULT` 时为 `True`（表示命中了某个领域），否则 `False`（表示全部关键词都没命中，`domain` 是兜底值）；`known_domains` 为 `list(KNOWN_DOMAINS)`，即把模块级元组转成列表后给出的全部可选领域名。
- **内部流程**：第一行 `domain = classify_domain(arguments.text, title=arguments.title)`，注意 `default` 使用函数默认值 `DEFAULT`，工具层没有暴露修改兜底领域的能力；第二行构造并返回 `ClassifyDomainOutput`，其中 `matched` 通过 `domain != DEFAULT` 现算，`known_domains` 通过 `list(KNOWN_DOMAINS)` 每次生成一个新列表（避免与模块级元组或其它实例共享可变对象）。整个过程不写任何状态、不做 IO。
- **异常/边界**：方法自身不抛异常也不捕获异常。若 `arguments` 不是 `ClassifyDomainInput`（例如传了 `dict`），访问 `.text` 会抛 `AttributeError`；正常情况下框架已用 `input_model` 完成校验，所以这条路径不会走到。空 `text` 与空 `title` 时 `classify_domain` 返回 `DEFAULT`，于是 `matched` 为 `False`、`domain` 为兜底值「未分类」——这正是 guidance 里说明的「正确结果」。无网络、无文件、无超时风险。
- **同文件关系**：调用同文件的 `classify_domain`；读取同文件的 `KNOWN_DOMAINS` 与导入的 `DEFAULT`；构造同文件的 `ClassifyDomainOutput`；被框架（经由 `ClassifyDomainTool` 的 `spec`）调用，本文件内没有其它函数调用它。

### `create_tool() -> BaseTool` （第 167 行）

- **作用**：这是工具的工厂函数，供工具加载/注册机制调用，返回一个可直接使用的 `ClassifyDomainTool` 实例。项目里约定每个工具模块暴露这样一个零参数工厂，加载方按模块扫描并调用 `create_tool()` 就能拿到工具对象，而不需要知道具体的类名或构造细节。当前实现是无状态的一次性构造，每次调用都返回一个新实例（工具本身无实例状态，因此多实例并存也安全）。
- **参数**：无参数。
- **返回**：返回 `BaseTool` 类型标注下的 `ClassifyDomainTool()` 新实例；调用方按基类接口使用（读取 `spec`、调用 `execute`），必要时可自行向下转型。
- **内部流程**：函数体只有一行 `return ClassifyDomainTool()`，直接实例化类，不读配置、不查环境变量、不做条件判断，因此永不失败。
- **异常/边界**：不抛异常；无空值、超时、缺失数据的处理需求。类实例化过程只涉及类属性 `spec` 的赋值，纯内存操作。
- **同文件关系**：调用同文件的 `ClassifyDomainTool` 构造实例；在本文件内没有被其它函数调用（`__all__` 中导出，供外部加载器使用）。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `classify_domain(text, *, title="", default=DEFAULT) -> str` | 遍历领域关键词表按命中次数（标题命中额外加权）打分，返回得分最高的领域名，全不命中时返回兜底领域。 |
| `majority_domain(domains, *, default=DEFAULT) -> str` | 对一组领域名做计数投票，返回出现次数最多的领域，空输入返回兜底领域。 |
| `ClassifyDomainInput(BaseModel)` | 工具入参模型，声明可选的 `text` 与 `title` 两个字符串字段，并禁止多余字段与隐式类型转换。 |
| `ClassifyDomainOutput(BaseModel)` | 工具出参模型，返回 `domain`、`matched`（是否真命中）与 `known_domains`（全部可选领域）。 |
| `ClassifyDomainTool(BaseTool)` | 把纯函数包装成 Agent 可调用的 `knowledge.classify_domain` 工具，用 `spec` 声明元信息与使用指引。 |
| `ClassifyDomainTool.execute(self, arguments) -> ClassifyDomainOutput` | 执行入口：调用 `classify_domain` 分类，用是否等于兜底值判定 `matched`，并附上全部可选领域列表。 |
| `create_tool() -> BaseTool` | 零参数工厂，返回一个 `ClassifyDomainTool` 实例供工具加载器注册。 |
