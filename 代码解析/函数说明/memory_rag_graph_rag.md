# memory/rag/graph_rag.py

## 一、这个文件是干什么的

这个文件实现的是「向量检索 + 知识图谱多跳扩展」的混合检索管道（Hybrid RAG），是四层记忆系统中语义记忆（SEMANTIC）层对外提供上下文的主要入口之一。它先让 `MemoryManager` 做一次普通的向量/关键词相似度检索拿到「证据」，再从这些证据和用户问题里识别出「种子实体」，然后沿着实体之间的有向关系边做广度优先的多跳扩展，最后把「证据 + 关系路径 + 关系证据」拼成一段可读文本喂给上层 Agent 作为回答依据。

文件里一共包含三类东西：两个数据载体（冻结 dataclass `GraphPath` 表示一条图谱路径、可变 dataclass `GraphRAGResult` 表示一次检索的完整结果），以及一个核心业务类 `GraphRAGPipeline`，它封装了参数校验、种子识别、加权 BFS 扩展、时间窗/状态过滤、边权排序与「回忆即强化」的全部逻辑。

它被用到的典型时机是：Web 应用或 Agent 在回答用户提问前，调用 `GraphRAGPipeline.build_context(query)` 或 `retrieve(query)`，把返回的字符串当作提示词中的「记忆上下文」，或者拿到结构化的 `GraphRAGResult` 自己再做二次处理。整个过程对图谱后端故障是容错的——一旦图谱侧抛异常，会退化为「只有向量证据、没有多跳路径」的结果，保证聊天回答不会因为图谱抖动而整体失败。

## 二、函数与类逐条详解

### `GraphPath` （第 22 行）

- **作用**：这是一个 `@dataclass(frozen=True)` 冻结数据类，用来描述图谱里从某个源实体出发、经过若干条关系边抵达某个目标实体的一条「路径」。它承载三样核心信息：路径上依次经过的实体（`entities`）、相邻实体之间的关系名（`relations`），以及每条边的属性字典（`evidence`）。此外它还额外保存两个「计算用」的字段：`effective` 用于按权重感知的排序打分，`steps` 保存真实的边三元组以便回写强化计数。之所以把它做成冻结的，是因为路径一旦生成就是不可变的快照，避免 BFS 过程中多个分支共享对象时被意外改写。它主要被 `GraphRAGPipeline._expand` 构造，并被 `GraphRAGResult` 持有和渲染。
- **参数**：数据类字段依次为：`source: str`（路径起点实体名，无默认值）、`target: str`（路径终点实体名，无默认值）、`relations: tuple[str, ...]`（路径上每条边的 relation 名，元组，顺序与跳数一致，无默认值）、`entities: tuple[str, ...]`（路径上依次经过的实体名元组，第一个元素通常等于 `source`，无默认值）、`confidence: float = 0.0`（整条路径的置信度，取路径上各边置信度的最小值，取值一般在 0.0~1.0）、`evidence: tuple[dict[str, Any], ...] = ()`（路径上每条边的属性字典，按跳数顺序排列，默认空元组）、`effective: float = 0.0`（F1 权重感知排序分，等于 confidence × 边权重沿路径取最小，注释明确说明它不影响 `to_dict` 输出）、`steps: tuple[dict[str, Any], ...] = ()`（路径的真实边 `{source, relation, target}` 三元组集合，供回忆强化按边递增，同样不进 `to_dict`）。
- **返回**：这是类定义本身，构造时返回 `GraphPath` 实例；由于 `frozen=True`，实例创建后字段只读，尝试赋值会抛 `dataclasses.FrozenInstanceError`。
- **内部流程**：没有自定义 `__init__`，由 dataclass 自动生成字段初始化与 `__repr__`/`__eq__`；`frozen=True` 让生成的 `__setattr__` 直接拒绝写操作。字段声明顺序决定了位置参数顺序，因此 `_expand` 中全部使用关键字参数构造以避免顺序错误。
- **异常/边界**：字段本身不做类型校验（Python 注解不强制），传错类型不会立刻报错；对 `confidence`/`effective` 没有范围约束，调用方需自行保证；修改实例字段会抛 `FrozenInstanceError`。
- **同文件关系**：被 `GraphRAGPipeline._expand` 构造并写入结果列表，被 `GraphRAGPipeline._reinforce` 读取 `steps`，被 `GraphRAGResult.build_context` 读取 `relations`/`entities`/`confidence`/`evidence`，被 `GraphRAGResult.to_dict` 间接调用其 `to_dict`。

### `GraphPath.to_dict() -> dict[str, Any]` （第 35 行）

- **作用**：把 `GraphPath` 实例序列化成普通字典，供上层做 JSON 输出、写日志、通过 Web API 返回给前端，或者交给别的模块做无依赖处理。它刻意只导出六个「对外语义」字段，而不导出 `effective` 和 `steps` 这两个内部计算字段——因为 `effective` 是排序中间量、`steps` 是强化回写用的原始边，对使用者没有意义，导出反而会污染 API 契约。因此这个方法是 `GraphPath` 对外表达形式的唯一权威出口。
- **参数**：无（仅隐式 `self`）。
- **返回**：返回 `dict[str, Any]`，固定含六个键：`"source"`（字符串）、`"target"`（字符串）、`"relations"`（由元组转成的列表）、`"entities"`（由元组转成的列表）、`"confidence"`（浮点数）、`"evidence"`（由元组转成的列表，其中每个元素都是原属性字典的浅拷贝）。任何情况下都返回完整字典，不返回 `None`。
- **内部流程**：直接用字面量字典一次性构造返回；对三个元组字段调用 `list(...)` 转成列表，因为 JSON 序列化更友好；对 `evidence` 使用列表推导 `[dict(item) for item in self.evidence]`，对每个边属性字典做一层浅拷贝，防止调用方修改返回值时污染原始数据；`effective` 与 `steps` 未被引用，天然被排除。
- **异常/边界**：如果 `evidence` 中混入了非映射对象（例如字符串），`dict(item)` 会抛 `TypeError` 或 `ValueError`；浅拷贝意味着嵌套的字典/列表仍是共享引用，深层修改仍会互相影响。空元组会被正常转成空列表。
- **同文件关系**：被 `GraphRAGResult.to_dict` 在列表推导中调用；不调用本文件其他函数。

### `GraphRAGResult` （第 46 行）

- **作用**：这是 `@dataclass`（非冻结）的结果聚合类，代表一次完整的图谱混合检索产出。它把四样东西打包在一起：原始查询 `query`、向量检索得到的证据列表 `evidence`、图谱扩展得到的路径列表 `paths`、以及识别出的种子实体名列表 `entities`。上层的 Agent 拿到它之后，既可以调用 `build_context` 直接得到拼好的提示词文本，也可以调用 `to_dict` 拿到结构化数据自己渲染。它是 `GraphRAGPipeline.retrieve` 的唯一返回类型，也是「图谱失败降级」时的返回载体（此时 `paths` 与 `entities` 为空列表）。
- **参数**：字段为：`query: str`（本次检索的原始查询文本，无默认值）、`evidence: list[MemorySearchResult] = field(default_factory=list)`（向量检索结果列表，使用 `default_factory` 保证每个实例拿到独立的新列表，避免可变默认值共享陷阱）、`paths: list[GraphPath] = field(default_factory=list)`（图谱路径列表）、`entities: list[str] = field(default_factory=list)`（种子实体名列表）。
- **返回**：类定义，构造时返回 `GraphRAGResult` 实例；非冻结，字段可在构造后修改。
- **内部流程**：无自定义 `__init__`，dataclass 自动生成初始化；两个列表字段使用 `field(default_factory=list)` 而非直接 `= []`，这是为了避免所有实例共享同一个列表对象。
- **异常/边界**：不做类型校验；`query` 允许为空字符串（`GraphRAGPipeline.retrieve` 在更上层已做非空校验）；列表字段允许传入 `None`，但那会导致后续 `build_context`/`to_dict` 抛 `TypeError`。
- **同文件关系**：被 `GraphRAGPipeline.retrieve` 构造并返回；其 `to_dict` 调用 `GraphPath.to_dict`；其 `build_context` 读取 `GraphPath` 的字段。

### `GraphRAGResult.to_dict() -> dict[str, Any]` （第 53 行）

- **作用**：把一次检索结果整体序列化成嵌套字典，用于 API 响应或持久化。它对三个列表字段分别做转换：`evidence` 中每个 `MemorySearchResult` 调自己的 `to_dict`，`paths` 中每个 `GraphPath` 调本文件的 `GraphPath.to_dict`，`entities` 则用 `list(...)` 复制一份。这样上层就能拿到纯 JSON 兼容结构，而不必关心内部对象类型。它通常在 Web 层或日志层被调用。
- **参数**：无（仅隐式 `self`）。
- **返回**：返回 `dict[str, Any]`，固定含四个键：`"query"`（原样字符串）、`"evidence"`（`MemorySearchResult.to_dict` 结果的列表）、`"paths"`（`GraphPath.to_dict` 结果的列表）、`"entities"`（字符串列表，是原列表的浅拷贝）。始终返回完整字典。
- **内部流程**：用一个字典字面量直接构造；`evidence` 用列表推导调用元素自身的 `to_dict()`；`paths` 用列表推导调用 `path.to_dict()`；`entities` 用 `list(self.entities)` 生成浅拷贝，避免调用方 `append` 污染结果对象内部状态。
- **异常/边界**：若 `evidence` 元素缺少 `to_dict` 方法会抛 `AttributeError`；若列表字段被显式赋成 `None` 会抛 `TypeError`；空列表正常返回空列表。无其他特殊处理。
- **同文件关系**：调用了 `GraphPath.to_dict`（经由 `path.to_dict()`）；被上层（本文件之外）调用；本文件内没有其他函数调用它。

### `GraphRAGResult.build_context(*, max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str` （第 61 行）

- **作用**：这是本文件最贴近「业务价值」的渲染方法：把结构化的检索结果拼成一段人类与 LLM 都能读的自然语言上下文。它先逐条渲染向量证据，每条前面加一个方括号标记，写明来源文件名、相似度分数，并在记忆已被更新（`active` 为 `False`）时追加「历史记录，已被更新」的警示；然后再逐条渲染图谱关系路径，输出「实体A -> 实体B（关系）」以及紧随其后的关系证据文本。这样 LLM 既能看到直接相关的原文片段，也能看到实体之间的推理链路。它通常在构造提示词前被调用一次。
- **参数**：`max_chars: int = RAG_CONTEXT_MAX_CHARS`（关键字专用参数，来自 `constants` 模块的常量，默认值即该常量；表示最终返回字符串的最大字符数，超长会被硬截断）。该参数没有显式类型校验。
- **返回**：返回拼接好的字符串。若 `evidence` 与 `paths` 都为空，返回空字符串 `""`（因为 `"\n\n".join([])` 为空串）。任何情况下都返回 `str`，且长度不超过 `max_chars`（由切片保证）。
- **内部流程**：第一步，创建空列表 `parts`。第二步，遍历 `self.evidence`：取出 `result.item`，用 `metadata.get("filename") or metadata.get("source") or "记忆库"` 依次回退得到来源标签；用 `metadata.get("active", True) is not False` 判断是否仍为当前有效记忆（注意用的是 `is not False`，所以 `None`、缺失都算有效）；若失效则 `marker` 设为 `"|历史记录，已被更新"`，否则为空串；然后把 `f"[证据|来源={source}|相似度={result.score:.3f}{marker}]\n{item.content}"` 追加进 `parts`（相似度格式化为三位小数）。第三步，遍历 `self.paths`：把 `relations` 用 `" -"` 连接成 `relation` 字符串；追加一行 `[关系路径|置信度={confidence:.3f}] 实体1 -> 实体2（关系）`（实体用 `" -> "` 连接）；再遍历该路径的 `path.evidence`，对每个边属性字典取 `evidence.get("evidence") or evidence.get("source") or ""`，非空则追加一行 `[关系证据] {text}`。第四步，用两个换行 `"\n\n"` 把 `parts` 连接成一个字符串，并用 `[:max_chars]` 做尾部硬截断后返回。
- **异常/边界**：若 `max_chars` 为负数，Python 切片会得到除尾部外的几乎全部内容（`[:-n]` 语义），属于未防御的边界；若 `max_chars` 为 `0` 返回空串；截断可能把一条证据或一行关系路径切成半句，不做完整性保护；若 `result.item.metadata` 不是字典会抛 `AttributeError`；若 `path.relations` 为空元组，`" -".join(())` 得到空串，括号内为空。无超时相关处理。
- **同文件关系**：读取 `GraphPath` 的 `relations`、`entities`、`confidence`、`evidence` 字段；被 `GraphRAGPipeline.build_context` 调用；本文件内不调用其他函数。

### `GraphRAGPipeline` （第 86 行）

- **作用**：这是整个文件的核心业务类，封装「先语义检索、再多跳图扩展」的完整管道。它持有唯一的依赖 `MemoryManager`，通过它同时访问向量检索能力（`manager.search`）、语义记忆存储（`manager.semantic`）以及底层图存储（`manager.semantic.graph_store`）。对外只暴露两个公开方法：`retrieve` 返回结构化结果，`build_context` 返回拼好的上下文字符串；其余以 `_` 开头的都是内部实现（种子识别、加权 BFS 扩展、边过滤、强化回写）。类文档字符串明确说明它的职责是「Retrieve semantic evidence, then expand one or more graph hops.」
- **参数**：无类级参数；实例化参数见 `__init__`。
- **返回**：类定义。
- **内部流程**：无类体逻辑（只有文档字符串），所有行为都在方法中。
- **异常/边界**：无特殊处理。
- **同文件关系**：包含并组织本文件 `GraphRAGResult`、`GraphPath` 的构造与使用；其方法之间互相调用（详见各方法条目）。

### `GraphRAGPipeline.__init__(self, manager: MemoryManager) -> None` （第 89 行）

- **作用**：构造管道实例，注入记忆管理器。之所以需要它，是因为整个类的所有检索与图谱操作都必须经由 `MemoryManager` 统一入口，而不能自己去连数据库或 Neo4j，这样上层可以在启动时只装配一次管理器，再把它传给管道。它只是简单地保存引用，不做任何初始化开销（不建连接、不预加载），因此可以低成本地按请求创建。被上层（如 Web 应用或 Agent 运行时）在需要做图谱增强检索时调用。
- **参数**：`manager: MemoryManager`（位置参数，无默认值；类型注解为 `..manager` 模块里的 `MemoryManager`，实际使用时不强校验，鸭子类型即可，只要提供 `search`、`semantic.list`、`semantic.related`、`semantic.graph_store.add_relation`、`get` 等方法/属性）。
- **返回**：无返回值（`None`）。
- **内部流程**：只有一步——`self.manager = manager`，把传入对象挂到实例属性上，供后续所有方法通过 `self.manager` 访问。
- **异常/边界**：不校验 `manager` 是否为 `None`；传入 `None` 不会在此处报错，而是在第一次调用 `retrieve` 时抛 `AttributeError`。无其他特殊处理。
- **同文件关系**：被本文件所有实例方法间接依赖（它们都通过 `self.manager` 访问外部能力）；不被本文件其他函数调用。

### `GraphRAGPipeline.retrieve(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, hops: int = RAG_GRAPH_HOPS, threshold: float | None = None, path_limit: int = RAG_GRAPH_PATH_LIMIT, at: str | None = None) -> GraphRAGResult` （第 92 行）

- **作用**：这是管道的公开主入口，一次调用完成「参数校验 → 向量检索 → 种子实体识别 → 图谱多跳扩展 → 结果裁剪与降级」全流程。它是上层拿到图谱增强检索结果的唯一推荐方式，通常在一次用户提问到达时被调用一次。它的一大特点是对图谱侧故障做隔离：即便图谱后端（Aura 代理或本地 Neo4j）抖动抛异常，也不会把整个调用打断，而是降级返回「只有向量证据」的结果，保证聊天回答不中断。另一个特点是它把 `hops=0` 视为「只做向量检索、不做任何图扩展」的合法用法。
- **参数**：`query: str`（位置参数，必填，非空字符串，内部会 `strip()` 后判空）；`limit: int = RAG_RETRIEVE_LIMIT`（关键字专用，最终返回的证据条数上限，必须是正整数，且不允许是 `bool`）；`hops: int = RAG_GRAPH_HOPS`（关键字专用，图扩展最大跳数，必须是整数且落在 `0 <= hops <= RAG_GRAPH_MAX_HOPS` 闭区间，`0` 表示跳过图扩展）；`threshold: float | None = None`（关键字专用，相似度阈值，`None` 表示不设阈值，原样透传给 `manager.search`，本方法自身不校验其类型/范围）；`path_limit: int = RAG_GRAPH_PATH_LIMIT`（关键字专用，返回的图谱路径条数上限，必须是正整数且不能是 `bool`）；`at: str | None = None`（关键字专用，时间基准字符串，`None` 表示「现在」，透传给 `_expand` 用于时间窗过滤）。
- **返回**：返回 `GraphRAGResult` 实例。正常路径下 `evidence` 是向量检索结果的前 `limit` 条、`paths` 是排序裁剪后的图谱路径、`entities` 是识别出的种子实体名列表。图谱阶段抛异常时返回 `GraphRAGResult(query=query, evidence=evidence[:limit], paths=[], entities=[])`，即路径与实体都为空。
- **内部流程**：第一步，做四项参数校验：`query` 必须是非空 `str`（否则 `ValueError("query must be a non-empty string")`）；`limit` 不能是 `bool`、必须是 `int` 且 `>= 1`（否则 `ValueError("limit must be a positive integer")`）；`hops` 不能是 `bool`、必须是 `int` 且 `0 <= hops <= RAG_GRAPH_MAX_HOPS`（否则 `ValueError` 并带上上限值）；`path_limit` 不能是 `bool`、必须是 `int` 且 `>= 1`（否则 `ValueError("path_limit must be a positive integer")`）。第二步，调用 `self.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=max(limit * 3, limit), threshold=threshold)`——注意它按 `limit * 3` 过采样（`max` 保证至少等于 `limit`），以便后续能从中筛出种子实体，最后才裁到 `limit`。第三步，在 `try` 块内调用 `self._find_seed_entities(query, evidence)` 得到 `seeds`，再调用 `self._expand(seeds, hops=hops, path_limit=path_limit, at=at)` 得到 `paths`。第四步，若第三步抛任意异常（`except Exception`，注释说明图谱后端抖动不应拖垮向量证据），直接返回降级结果。第五步，正常返回 `GraphRAGResult(query=query, evidence=evidence[:limit], paths=paths, entities=seeds)`。
- **异常/边界**：参数非法抛 `ValueError`（这是本方法唯一主动抛出的异常，且在校验阶段抛出，早于任何 IO）。`manager.search` 自身抛出的异常不被捕获，会向上传播。图谱阶段（种子识别 + 扩展）的任何异常被宽泛捕获并降级为无路径结果，注释提到这是为 Aura/本地 Neo4j 抖动设计；注意这个 `except` 也会吞掉种子识别里的编程错误，属于有意的取舍。`threshold` 传非法值由 `manager.search` 决定后果。`evidence` 为空列表时后续种子识别仍能工作（会退化到只靠 `manager.semantic.list()` 全量实体做匹配）。`hops=0` 时 `_expand` 立即返回空列表，不会走图查询。
- **同文件关系**：调用 `_find_seed_entities` 与 `_expand`（两者在同一个 `try` 块内）；间接经由 `_expand` 调用 `_weighted_edges`、`_reinforce_enabled`、`_reinforce`、`_edge_is_active`、`_edge_in_window`；构造并返回 `GraphRAGResult`；被同文件的 `build_context` 调用。

### `GraphRAGPipeline.build_context(self, query: str, *, limit: int = 5, hops: int = 1, max_chars: int = 12000) -> str` （第 141 行）

- **作用**：这是面向调用方最省事的便捷方法：一行调用就拿到可直接塞进提示词的上下文文本。它本质上是 `retrieve` 加 `GraphRAGResult.build_context` 的组合语法糖，把两步（检索、渲染）合并成一步，避免上层每次都要先拿结果再手动渲染。它的默认参数取值偏保守（`limit=5`、`hops=1`、`max_chars=12000`），适合「快速、单跳、上下文预算有限」的常规问答场景。当上层只关心最终文本、不关心结构化路径时就应该用它。
- **参数**：`query: str`（位置参数，必填，最终会由 `retrieve` 做非空校验）；`limit: int = 5`（关键字专用，证据条数上限，注意这里的默认值与 `retrieve` 的 `RAG_RETRIEVE_LIMIT` 不同，是硬编码的 5）；`hops: int = 1`（关键字专用，图扩展跳数，默认只扩一跳）；`max_chars: int = 12000`（关键字专用，返回文本的字符上限，同样与 `GraphRAGResult.build_context` 的常量默认值不同，是硬编码的 12000）。
- **返回**：返回 `str`，即渲染并截断后的上下文文本；若检索不到任何证据和路径，返回空字符串。
- **内部流程**：只有一步实质操作——调用 `self.retrieve(query, limit=limit, hops=hops)`（注意它没有把 `threshold`、`path_limit`、`at` 透传出去，因此这三个参数在本方法下永远是 `retrieve` 的默认值），然后立刻在结果对象上调用 `.build_context(max_chars=max_chars)` 并返回其返回值。
- **异常/边界**：`query` 非法或 `limit`/`hops` 越界时，由 `retrieve` 抛 `ValueError`，本方法不额外处理；`max_chars` 非法（如负数）的行为由 `GraphRAGResult.build_context` 的切片语义决定。图谱故障时因 `retrieve` 已降级，本方法仍会返回只含向量证据的文本。
- **同文件关系**：调用 `retrieve`；并调用 `GraphRAGResult.build_context`（经结果对象）；不被本文件其他函数调用。

### `GraphRAGPipeline._find_seed_entities(self, query: str, evidence: list[MemorySearchResult]) -> list[str]` （第 148 行）

- **作用**：负责从「用户问题 + 向量检索到的证据」里推断出应该作为图谱遍历起点的实体名（种子）。这是多跳扩展的前置步骤：没有种子就没有路径，种子的质量直接决定图谱召回的相关性。它用两条互补线索来收集候选：一是证据条目本身如果标记为实体（`metadata["kind"] == "entity"`）就直接当种子；二是证据的 `subject`/`object` 字段如果出现在问题文本里，说明这个关系端点与提问强相关。第三轮扫描则是拿全量已知实体（`manager.semantic.list()`）逐个与问题做名字/别名匹配，这样即使向量检索没召回该实体，只要问题里明确提了名字也能起跳。最后去重保序返回。
- **参数**：`query: str`（用户原始问题文本，会先 `casefold()` 用于大小写不敏感匹配）；`evidence: list[MemorySearchResult]`（向量检索返回的证据列表，来自 `retrieve` 中的 `manager.search`，元素需具备 `item` 属性，`item` 需具备 `metadata` 字典与 `content` 字符串）。
- **返回**：返回 `list[str]`，是去重（保序，用 `dict.fromkeys`）后的种子实体名列表。没有任何命中时返回空列表 `[]`，此时 `_expand` 会直接返回空路径列表。
- **内部流程**：第一步，初始化 `names: list[str] = []`，取 `known = self.manager.semantic.list()`（全量语义记忆条目），并把查询 `query_folded = query.casefold()`。第二步，遍历 `evidence`：取出 `item = result.item` 与 `metadata = item.metadata`；若 `metadata.get("kind") == "entity"`，则按 `canonical_name` → `title` → `item.content` 的顺序回退取值并 `str()` 后追加到 `names`；接着遍历 `("subject", "object")` 两个键，取值后判断 `value and str(value).casefold() in query_folded`，命中就把该值追加到 `names`（这是子串包含判断，不是精确相等）。第三步，遍历 `known`：跳过 `kind != "entity"` 的条目；用 `canonical_name` 或 `content` 作为 `name`，把 `metadata.get("aliases", [])` 全部 `str()` 化成别名列表；再用 `any(value.casefold() in query_folded for value in [name, *aliases] if value)` 判断名字或任一别名是否是问题的子串，命中则追加 `name`。第四步，`return list(dict.fromkeys(names))` 去重并保持首次出现的顺序。
- **异常/边界**：若 `result.item.metadata` 不是字典（例如是 `None`）会抛 `AttributeError`；若 `metadata["aliases"]` 是字符串而非列表，`for value in ...` 会逐字符迭代，产生大量单字符候选（未做防御）；若 `aliases` 为 `None`，`[name, *aliases]` 会抛 `TypeError`。匹配是「实体名是否为问题子串」的单向包含，实体名比问题还长时不会命中；对中文无分词，纯靠子串。`known` 为空时该轮循环直接跳过，不影响前两轮结果。无超时处理。注意本方法在图谱侧故障降级时也被包在 `retrieve` 的 `try` 中，因此它抛出的异常会导致整个图谱阶段被跳过。
- **同文件关系**：被 `GraphRAGPipeline.retrieve` 调用；它调用外部 `self.manager.semantic.list()`；不调用本文件其他函数；其返回的种子列表被 `_expand` 作为 BFS 起点，并被 `retrieve` 直接放进 `GraphRAGResult.entities`。

### `GraphRAGPipeline._expand(self, seeds: list[str], *, hops: int, path_limit: int, at: str | None = None) -> list[GraphPath]` （第 180 行）

- **作用**：这是图谱多跳扩展的核心实现，用加权广度优先搜索（BFS）从种子实体出发，逐跳收集可用的关系边，构造出一条条 `GraphPath` 并返回按分数排序后的前 `path_limit` 条。它同时承担了四类过滤与排序职责：跳过已失效/被撤回的边（`active` 与记忆记录双重确认）、按时间窗与事件时间过滤（F4）、对 `uncertain` 状态边降权、对 `expired` 边直接排除；并按边权重（F1）计算 `effective` 分数让「被反复回忆过的边」排到前面。BFS 过程中还通过 `visited` 集合与「目标实体已在路径中」两个条件防止重复与环路。最后对真正被采纳的路径执行「回忆即强化」回写。
- **参数**：`seeds: list[str]`（BFS 起点实体名列表，通常来自 `_find_seed_entities`；为空或 `hops == 0` 时直接返回空列表）；`hops: int`（关键字专用，最大扩展深度，BFS 在 `depth >= hops` 时停止向下一层展开）；`path_limit: int`（关键字专用，返回路径条数上限，同时是 BFS 循环的提前终止条件）；`at: str | None = None`（关键字专用，时间基准，透传给 `_weighted_edges` 与 `_edge_in_window`，`None` 表示以当前 UTC 时间为准）。
- **返回**：返回 `list[GraphPath]`。当 `hops == 0` 或 `seeds` 为空时返回 `[]`；否则返回排序后截断到 `path_limit` 条的路径列表（可能少于 `path_limit` 条，取决于图中有多少可用边）；若所有边都被过滤掉也返回 `[]`。注意排序在截断之后又做了一次「先全排序再切片」，但循环里已按 `len(paths) < path_limit` 提前退出，所以通常 `paths` 长度不会大幅超过 `path_limit`。
- **内部流程**：第一步，若 `hops == 0 or not seeds` 立即 `return []`。第二步，取 `reinforce = self._reinforce_enabled()` 缓存开关。第三步，初始化 `paths` 列表，并用 `deque` 构造 BFS 队列，每个元素是八元组 `(current, entities, relations, evidence, depth, confidence, effective, steps)`；初始元素为 `(seed, (seed,), (), (), 0, 1.0, 1.0, ())`，即路径从种子自身开始、置信度与权重分初始为 `1.0`。第四步，`visited: set[tuple[str, tuple[str, ...]]]` 记录 `(邻居实体, 新关系元组)` 组合以防重复扩展。第五步，主循环 `while queue and len(paths) < path_limit`：`popleft` 取出当前节点；若 `depth >= hops` 则 `continue`（不再向下展开）；否则调用 `self._weighted_edges(current, at=at)` 拿到按权重降序排好的邻接边。第六步，对每条边：解析 `source`（缺省回退为 `current`）、`target`（缺省回退为 `current`）、`neighbor`（若 `source == current` 则取 `target`，否则取 `source`，即支持边的方向与遍历方向相反的情况）、`relation`（缺省为 `"关联"`）、`props = dict(edge.get("properties") or {})`。第七步，依次过滤：`props.get("active") is False` 或 `_edge_is_active(props)` 返回假 → `continue`（注释说明被取代/撤回的边保留在库里审计，但绝不能承载检索跳转，且因为进程内/Neo4j 的边副本可能还留着退役前的标记，所以必须回查 SQLite 里的记忆记录）；`_edge_in_window(props, at=at)` 为假 → `continue`。第八步，解析 `edge_confidence = float(props.get("confidence", 0.0) or 0.0)` 与 `edge_status = str(props.get("status") or "fact")`；`status == "expired"` 直接 `continue`；`status == "uncertain"` 则把 `edge_confidence *= 0.5`（降权而非排除）。第九步，计算 `edge_weight = float(props.get("weight", 1.0) or 1.0)` 与 `edge_effective = edge_confidence * edge_weight`（注释说明全部权重为 1.0 时 `effective` 退化为原来的 `confidence` 语义）。第十步，构造 `next_entities = (*entities, neighbor)`、`next_relations = (*relations, relation)`、`marker = (neighbor, next_relations)`；若 `marker in visited` 或 `neighbor in entities`（成环）则 `continue`；否则把 `marker` 加入 `visited`。第十一步，计算 `next_confidence = min(confidence, edge_confidence or confidence)`（沿路径取最小，`or` 保证边置信度为 0 时退回上一层值而非把整条路径归零）、`next_effective = min(effective, edge_effective or effective)`、`next_evidence = (*evidence, props)`、`next_steps = (*steps, {"source": source, "relation": relation, "target": target})`。第十二步，立即把这条新路径追加进 `paths`（注意：路径是在「走出一条边」时就登记，而不是等到叶子节点），其中 `source=entities[0]`（路径真正的起点）、`target=neighbor`。第十三步，把八元组新状态 `append` 进队列，`depth + 1`。第十四步，循环结束后 `paths.sort(key=lambda item: (-item.effective, len(item.relations), item.target))`，即先按 `effective` 降序、再按跳数升序、最后按目标实体名升序稳定排序。第十五步，`adopted = paths[:path_limit]` 截断。第十六步，若 `reinforce` 为真则调用 `self._reinforce(adopted)` 回写强化计数。第十七步，`return adopted`。
- **异常/边界**：`float(props.get(...))` 对无法转成浮点数的值（如 `"abc"`）会抛 `ValueError`，本方法不捕获——该异常会冒泡到 `retrieve` 的 `except Exception` 从而降级为无路径结果；`self._weighted_edges` 或 `self._reinforce` 抛异常同理被 `retrieve` 吞掉。`hops == 0`、`seeds` 为空、图中无边都返回空列表而不报错。`path_limit` 很小（如 1）时 BFS 会尽早终止，可能只探索到图中一小部分，因此结果对 `path_limit` 敏感。`visited` 的键是 `(neighbor, next_relations)`，同一实体经由不同关系序列仍可被多次访问，因此仍可能生成多条共享中间节点的路径。`edge.get("properties")` 为 `None` 时用 `or {}` 兜底；`edge` 本身不是字典会抛 `AttributeError`。无超时控制，极端稠密图上依赖 `path_limit` 终止。
- **同文件关系**：被 `GraphRAGPipeline.retrieve` 调用；它调用 `_reinforce_enabled`、`_weighted_edges`、`_edge_is_active`、`_edge_in_window`、`_reinforce`；构造 `GraphPath` 对象并返回给 `retrieve`。

### `GraphRAGPipeline._weighted_edges(self, entity: str, *, at: str | None = None) -> list[dict[str, Any]]` （第 279 行）

- **作用**：这是取「某个实体周围所有可用边」的封装方法，并在返回前按边权重从大到小排序，实现 F1 的「权重优先遍历」策略——先探索被反复回忆、权重更高的边，从而让高价值路径更早进入 `paths`，在 `path_limit` 受限时优先被采纳。它把外部图存储的查询细节（`manager.semantic.related`）与本类的排序策略隔离开，`_expand` 只需拿一个已排好序的列表。文档字符串写明「Edges around `entity` with the strongest first (F1 权重优先遍历)」。
- **参数**：`entity: str`（要查询邻接边的实体名，通常是 BFS 当前节点 `current`）；`at: str | None = None`（关键字专用，时间基准，原样透传给 `manager.semantic.related(entity, at=at)`，由底层决定如何按时间筛选边）。
- **返回**：返回 `list[dict[str, Any]]`，即底层返回的边字典列表，每个元素预期含 `source`、`target`、`relation`、`properties` 等键；返回的是**原地排序后的同一个列表对象**（`edges.sort` 是原地排序），并且已经按权重降序排列。实体不存在或没有边时返回空列表。
- **内部流程**：第一步，调用 `edges = self.manager.semantic.related(entity, at=at)` 拿到邻接边列表。第二步，用 `edges.sort(key=lambda edge: float((edge.get("properties") or {}).get("weight", 1.0) or 1.0), reverse=True)` 原地按权重降序排序——注意它对每个元素都用 `(edge.get("properties") or {})` 兜底空字典，权重缺失或为假值（`0`、`None`、空串）时回退为 `1.0`。第三步，`return edges`。
- **异常/边界**：若某条边的 `properties["weight"]` 是无法转浮点的字符串，`float()` 会抛 `ValueError`，本方法不捕获（最终由 `retrieve` 的宽泛 `except` 降级）；若 `edge` 不是字典会抛 `AttributeError`；`manager.semantic.related` 抛出的异常同样向上冒泡。排序是稳定排序，权重相同的边保持底层返回顺序。无超时与分页处理，一次性取全部邻边。
- **同文件关系**：被 `GraphRAGPipeline._expand` 在 BFS 每一层调用；它调用外部 `self.manager.semantic.related`；不调用本文件其他函数。

### `GraphRAGPipeline._reinforce_enabled(self) -> bool` （第 293 行）

- **作用**：这是一个极薄的开关读取方法，用来集中判断 F1「回忆即强化」功能是否启用。之所以单独抽一个方法而不是在 `_expand` 里直接读常量，是为了让开关判断有唯一入口，将来要改成读配置、读环境变量或按实例开关时只需改这一处。它在 `_expand` 开头被调用一次并缓存到局部变量 `reinforce`，避免在 BFS 循环里反复读取。文档字符串注明开关来自 `constants.MEMORY_EDGE_REINFORCE`，默认开。
- **参数**：无（仅隐式 `self`）。
- **返回**：返回 `bool`——直接返回模块级导入的常量 `MEMORY_EDGE_REINFORCE` 的值。若该常量为真值（如 `True`、非零、非空）则启用强化；若为假值（如 `False`、`0`、`None`、空容器）则视为关闭。
- **内部流程**：只有一步，`return MEMORY_EDGE_REINFORCE`，不做任何类型转换或异常处理。
- **异常/边界**：无特殊处理；常量在模块导入时已由 `constants` 提供，若导入失败则整个模块无法加载（属于导入期错误，与本方法无关）。
- **同文件关系**：被 `GraphRAGPipeline._expand` 调用；不调用本文件其他函数。

### `GraphRAGPipeline._reinforce(self, paths: list[GraphPath]) -> None` （第 298 行）

- **作用**：实现「回忆即强化」：把本次真正返回给调用方的路径上的每一条边，在底层图存储里把权重加一（`bump=True`），使得以后这些边因为权重更高而更早被遍历、更容易进入最终结果，形成「常被用到的知识越来越容易被检索到」的正反馈。它只对**被采纳**（`adopted`）的路径做强化，被过滤掉或未进入 `path_limit` 的路径不计数，避免噪音边被无意中提权。方法内用 `bumped` 集合保证同一条边在同一轮内只加一次，即使它出现在多条路径里。
- **参数**：`paths: list[GraphPath]`（本次实际返回的路径列表，来自 `_expand` 中的 `adopted`；每个元素的 `steps` 字段是 `{source, relation, target}` 字典元组）。
- **返回**：无返回值（`None`）。副作用是调用底层 `add_relation(*key, bump=True)` 修改边权重。
- **内部流程**：第一步，初始化 `bumped: set[tuple[str, str, str]] = set()`，并取出方法引用 `add_relation = self.manager.semantic.graph_store.add_relation`（提前绑定以避免循环内重复属性查找）。第二步，双层循环：外层遍历 `paths`，内层遍历 `path.steps`；对每个 `step` 组装键 `key = (step["source"], step["relation"], step["target"])`。第三步，若 `key` 已在 `bumped` 中则 `continue`，否则加入 `bumped` 并执行 `add_relation(*key, bump=True)`，即把 `source`、`relation`、`target` 作为三个位置参数传入，并用关键字参数 `bump=True` 表示「递增权重」而非「覆盖写入」。
- **异常/边界**：若 `step` 缺少 `source`/`relation`/`target` 任一键，`step[...]` 会抛 `KeyError`；若底层 `add_relation` 抛异常（如存储不可用），异常会向上冒泡到 `_expand`，进而被 `retrieve` 的宽泛 `except Exception` 捕获——这会导致**整轮图谱结果被丢弃并降级为空路径**，即便路径本来已经算好，这是一个需要留意的副作用耦合。`paths` 为空时循环不执行，什么也不做。无超时处理。
- **同文件关系**：被 `GraphRAGPipeline._expand` 在返回前条件调用（取决于 `_reinforce_enabled`）；读取 `GraphPath.steps`；调用外部 `self.manager.semantic.graph_store.add_relation`；不调用本文件其他函数。

### `GraphRAGPipeline._edge_is_active(self, properties: dict[str, Any]) -> bool` （第 311 行）

- **作用**：用来在一条边真正承载检索跳转之前，回到「记忆记录」这一权威数据源确认它仍然有效。设计动机写在代码注释里：被取代（superseded）或撤回（retracted）的边会保留在存储中供审计，但绝不能参与检索；而进程内缓存或 Neo4j 里的边副本可能还残留着退役前的 `active` 标记，所以不能只信边属性，必须用 `memory_id` 回查 SQLite 里的语义记忆条目再判一次。它由 `_expand` 在每条边上调用，是「双重确认」的第二道关卡。文档字符串写明「Confirm an edge against its memory record before it carries a hop.」
- **参数**：`properties: dict[str, Any]`（边属性字典，来自 `edge["properties"]`；本方法只关心其中的 `memory_id` 键）。
- **返回**：返回 `bool`。三种情况：`properties` 中没有 `memory_id`（或其为假值）→ 返回 `True`（无记录可查，默认放行）；有 `memory_id` 但 `manager.get(...)` 查不到该条目 → 返回 `False`（记录已不存在，视为失效）；查到条目且其 `metadata["active"]` 不是 `False`（注意用 `is not False`，所以缺失或 `None` 都算有效）→ 返回 `True`，否则返回 `False`。
- **内部流程**：第一步，`memory_id = properties.get("memory_id")`。第二步，`if not memory_id: return True`，即空值快速放行（这也意味着空字符串、`0`、`None` 都被当作「无 id」）。第三步，`item = self.manager.get(str(memory_id), memory_type=MemoryType.SEMANTIC)`，把 id 转成字符串并按语义记忆类型查询。第四步，`if item is None: return False`。第五步，`return item.metadata.get("active", True) is not False`。
- **异常/边界**：`manager.get` 抛异常会向上冒泡（最终可能触发 `retrieve` 的降级）；若 `item.metadata` 不是字典会抛 `AttributeError`；若 `properties` 本身不是字典会抛 `AttributeError`（调用方 `_expand` 已用 `dict(...)` 兜底成字典）。查询成本是每条边一次存储读取，在稠密图上可能成为性能瓶颈，代码未做缓存。无超时处理。
- **同文件关系**：被 `GraphRAGPipeline._expand` 在边过滤阶段调用（与 `props.get("active") is False` 组成 `or` 条件）；调用外部 `self.manager.get`；不调用本文件其他函数。

### `GraphRAGPipeline._edge_in_window(self, properties: dict[str, Any], *, at: str | None = None) -> bool` （第 322 行）

- **作用**：实现 F4 的时间/状态时间窗过滤：判断一条边在给定时刻 `at`（或当前时刻）是否「仍然成立」，从而决定它能不能承载这一跳。它处理三类时间信息：边的 `event_at`（事件实际发生时间，仅在调用方显式给了 `at` 时才用于「事件晚于查询时刻则排除」的未来事件过滤）、以及 `valid_from` / `valid_to` 构成的有效期区间。设计原则是「空值即无界」——缺少时间字段或字段为空字符串就当作没有约束；对畸形时间字符串采取宽容策略（返回 `True`，不静默丢边，交给 `status`/`active` 去判断）。文档字符串写明「Time-window filter at `at` (or now); empty bounds are unbounded.」
- **参数**：`properties: dict[str, Any]`（边属性字典，本方法读取 `event_at`、`valid_from`、`valid_to` 三个键）；`at: str | None = None`（关键字专用，时间基准字符串，`None` 表示使用 `utc_now()` 的当前时刻；同时它也决定是否启用 `event_at` 未来事件过滤）。
- **返回**：返回 `bool`。`True` 表示该边在时间维度上可用、允许承载跳转；`False` 表示应被排除。具体返回 `False` 的情形有：`at` 无法被解析成时间；`event_at` 可解析且晚于基准时刻；基准时刻早于 `valid_from`；基准时刻晚于 `valid_to`。其余情形（含字段缺失、字段为空、时间畸形、`ensure_datetime` 返回 `None`）均返回 `True`。
- **内部流程**：第一步，做函数内延迟导入 `from ..base import ensure_datetime, utc_now`（放在函数体内而非模块顶部，避免潜在的循环导入或加载开销）。第二步，在 `try` 中计算基准时刻：`moment = ensure_datetime(at) if at else utc_now()`；若抛 `TypeError` 或 `ValueError` 则直接 `return False`（时间基准非法时保守排除该边）。第三步，若 `moment is None`（解析函数对某些输入返回 `None`）则回退为 `utc_now()`。第四步，取 `event_raw = str(properties.get("event_at") or "").strip()`；仅当 `event_raw` 非空**且** `at` 为真时才尝试解析 `event_at`：解析失败则 `event_at = None`（不排除），解析成功且 `event_at > moment` 则 `return False`（未来事件不参与当前时点的检索）；注意如果 `at` 为 `None`（即用「现在」作基准），则完全跳过 `event_at` 判断。第五步，遍历 `("valid_from", "valid_to")`：取值并 `str(...).strip()`，空则 `continue`；在 `try` 中调用 `ensure_datetime(raw)` 解析，若抛 `TypeError`/`ValueError` 则 `return True`（注释写明：畸形时间不静默丢边，交给 status/active 判断）；若解析结果为 `None` 则 `continue`。第六步，做区间比较：`key == "valid_from"` 且 `moment < bound` → `return False`；`key == "valid_to"` 且 `moment > bound` → `return False`（边界取闭区间，等于边界值时放行）。第七步，全部通过则 `return True`。
- **异常/边界**：`properties` 不是字典时 `.get` 抛 `AttributeError`（调用方已保证是字典）。`ensure_datetime` 对 `at` 抛的 `TypeError`/`ValueError` 被捕获并返回 `False`（保守排除）；对 `valid_from`/`valid_to`/`event_at` 抛的同类异常则分别被宽容处理为放行或不比较。`at` 为 `None` 时 `event_at` 完全不参与判断，这是刻意的行为差异。`valid_from` 晚于 `valid_to` 这种自相矛盾的区间不会报错，只会按两次比较的逻辑分别判定。无超时处理。
- **同文件关系**：被 `GraphRAGPipeline._expand` 在边过滤阶段调用；它延迟导入并调用外部 `..base` 的 `ensure_datetime` 与 `utc_now`；不调用本文件其他函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `GraphPath` | 冻结数据类，描述一条从源实体到目标实体的图谱路径，携带实体、关系、边属性、权重排序分与真实边三元组。 |
| `GraphPath.to_dict()` | 把路径序列化成只含对外语义字段的字典，刻意排除 `effective` 与 `steps`。 |
| `GraphRAGResult` | 一次图谱混合检索的结果聚合类，打包查询、向量证据、图谱路径与种子实体。 |
| `GraphRAGResult.to_dict()` | 把整次检索结果递归序列化成嵌套字典，供 API 或日志使用。 |
| `GraphRAGResult.build_context()` | 把证据与关系路径渲染成带来源/相似度/置信度标记的上下文文本并按 `max_chars` 截断。 |
| `GraphRAGPipeline` | 核心管道类，先做语义检索再多跳扩展图谱，对外提供 `retrieve` 与 `build_context`。 |
| `GraphRAGPipeline.__init__()` | 注入并保存 `MemoryManager`，不做任何额外初始化。 |
| `GraphRAGPipeline.retrieve()` | 校验参数、向量过采样检索、识别种子、多跳扩展，并在图谱故障时降级为纯向量结果。 |
| `GraphRAGPipeline.build_context()` | `retrieve` 加结果渲染的组合便捷方法，一行拿到提示词上下文。 |
| `GraphRAGPipeline._find_seed_entities()` | 从证据的实体标记、`subject`/`object` 字段和全量实体的名字/别名中识别图谱遍历起点并去重。 |
| `GraphRAGPipeline._expand()` | 用加权 BFS 从种子出发扩展多跳路径，做失效/时间窗/状态过滤、环路去重、排序裁剪与强化回写。 |
| `GraphRAGPipeline._weighted_edges()` | 取某实体周围的所有边并按权重降序排序，实现权重优先遍历。 |
| `GraphRAGPipeline._reinforce_enabled()` | 读取 `constants.MEMORY_EDGE_REINFORCE` 判断「回忆即强化」是否启用。 |
| `GraphRAGPipeline._reinforce()` | 对被采纳路径上的每条边调用 `add_relation(..., bump=True)` 递增权重，同轮内每边只加一次。 |
| `GraphRAGPipeline._edge_is_active()` | 用边的 `memory_id` 回查记忆记录，双重确认该边未被取代或撤回。 |
| `GraphRAGPipeline._edge_in_window()` | 按 `event_at`、`valid_from`、`valid_to` 与基准时刻判断边是否落在有效时间窗内，空值视为无界。 |
