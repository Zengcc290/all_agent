# tool/reconcile.py

## 一、这个文件是干什么的

这个文件实现的是「三库对账」能力：把真值源（SQLite 的 `documents` / `chunks`）、向量投影（向量库，例如 Qdrant）和图投影（图存储，例如 Neo4j 或内存回退）放在一起比对，回答三个问题：每个库现在各有多少东西（计数）、真值源里已经标记为 `indexed` 的分块在向量库里是否真的都存在、有 fact 语义行却没有对应图关系的是哪些。之所以需要这个能力，是因为投影天生可能落后于真值源——云端嵌入当时不可达、进程被 kill、Neo4j 抖动，都会让投影缺数据，所以必须有一个随时能跑的「体检」动作来给出证据。

本模块的定位非常明确：**只读、不写、不改任何数据**，因此可以被随时调用、被 Agent 调用、被 UI 轮询；真正的修复动作在 `knowledge.repair_drift`（写工具）里，本文件只负责把漂移类别和 id 报出来。

文件内容大致分三层：第一层是四个模块级纯函数工具（`is_fact_item`、`fact_items`、`projected_vector_ids`、`projected_edge_ids`）加一个内部辅助函数 `_drift_entry`，负责从各个存储里「问出」事实和 id 集合；第二层是核心的 `reconcile_report`，把上面这些集合做差集运算，产出计数字典和漂移列表；第三层是一组 Pydantic 模型（`ReconcileInput`、`ReconcileCounts`、`DriftEntry`、`ReconcileOutput`）和一个 `BaseTool` 子类 `ReconcileTool`，把这个能力包装成名字为 `knowledge.reconcile` 的 Agent 工具，最后用 `create_tool()` 作为工厂函数导出。

模块里还有两个模块级常量：`TOOL_ENABLED = True`（工具启用开关）和 `REPAIRABLE_KINDS = ("missing_vector", "missing_edge")`（可自愈的漂移类型，只补投影、绝不删改真值源）。文件末尾的 `__all__` 声明了对外公开的名字，方便 `web/app.py` 的 `/api/reconcile`、`/api/stats` 以及工具注册表直接引用，避免再在 Web 层内联重复实现这套逻辑。

## 二、函数与类逐条详解

### `is_fact_item(item: Any) -> bool` （第 39 行）
- **作用**：判断一条记忆条目（`MemoryItem` 或任何带 `metadata` 的对象）是不是一条「事实」——也就是它是否同时具备 `subject`（主语）、`predicate`（谓语）、`object`（宾语）三个元数据字段。语义层里存的东西不止事实一种，还可能有概念、摘要之类的普通语义行，只有三元组齐全的行才应该被投影成图里的一条边。因此在对账「图投影是否缺边」之前，必须先用它把真正的 fact 行筛出来，否则会把本不该有边的行也算成漂移，产生误报。它被设计成接收 `Any` 而不是 `MemoryItem`，是为了在内存回退、不同存储实现、测试替身等场景下都能用 duck typing 判断，不强依赖具体类型。
- **参数**：`item`，类型 `Any`，无默认值。传入的是任意一个对象，正常情况下是 `memory.base.MemoryItem`；只要它有一个 `metadata` 属性且该属性是 dict-like 或 `None` 即可正常工作。
- **返回**：返回 `bool`。当 `item.metadata` 里 `subject`、`predicate`、`object` 三个键的取值全部为真值（非空字符串、非 `None`、非 `0`、非空容器）时返回 `True`；只要有任意一个缺失或为空/假值，就返回 `False`。
- **内部流程**：第一步用 `getattr(item, "metadata", {})` 读取元数据，并紧接一个 `or {}` 兜底——这样即使属性不存在、或者属性值是 `None`、或者是空 dict，都能安全地退化成一个空字典，避免后续 `.get` 抛 `AttributeError`。第二步用 `all(...)` 配合生成器表达式遍历固定的三元键名 `("subject", "predicate", "object")`，逐个调用 `metadata.get(key)`，只要有一个键的取值是假值，`all` 就短路返回 `False`，三个都通过才返回 `True`。整个函数没有任何副作用，不修改 `item`，也不访问数据库。
- **异常/边界**：正常路径不抛异常。边界情况都靠 `or {}` 和 `getattr` 的默认值吸收：`item` 没有 `metadata` 属性、`metadata` 为 `None`、`metadata` 是空字典，都会得到空字典进而返回 `False`。若传入的是 `None` 本身，`getattr(None, "metadata", {})` 同样返回 `{}`，结果也是 `False`。若 `metadata` 是一个不支持 `.get` 的对象（例如普通字符串），则会抛 `AttributeError`——代码未对此做额外防御。
- **同文件关系**：它被 `fact_items` 在列表推导里逐条调用，也被 `reconcile_report` 间接依赖（通过 `fact_items`）。它自己只依赖 `getattr` 和 `all` 等内建能力，不调用本文件里其它函数。

### `fact_items(manager: MemoryManager) -> list[MemoryItem]` （第 46 行）
- **作用**：从一个 `MemoryManager` 里取出「所有真正是事实的语义行」，也就是把语义层的全部 fact 行过一遍 `is_fact_item` 过滤后返回一个列表。对账逻辑需要知道「按真值源的说法，图投影里本应该有哪些边」，而这个答案的来源就是语义层里所有三元组齐全的行。把这一步单独抽成函数，是为了让「事实集合」的定义只有一个出处：`reconcile_report` 用它算缺失边，别的调用方（比如修复工具或统计接口）也能复用同一套筛选口径，避免各处各写一份过滤条件导致口径漂移。它本身不查数据库、不做差集，只是纯粹的读取加筛选。
- **参数**：`manager`，类型 `MemoryManager`，无默认值。必须是已经构造好的记忆管理器实例，并且它的 `semantic` 子对象上要有一个可调用的 `facts()` 方法，用来枚举语义层的事实行。
- **返回**：返回 `list[MemoryItem]`。列表里是按 `manager.semantic.facts()` 的产出顺序排列、且通过 `is_fact_item` 检查的所有条目；如果语义层为空、或者没有任何一行同时具备三个三元键，则返回空列表（不会返回 `None`）。
- **内部流程**：函数体只有一行列表推导：遍历 `manager.semantic.facts()` 产出的每一项 `item`，把 `is_fact_item(item)` 为真的留下，组成新列表返回。它先调用 `manager.semantic.facts()` 拿全量语义行，再逐条做元数据判断；不排序、不去重（去重由上层在 `reconcile_report` 里通过 set 完成）。
- **异常/边界**：如果 `manager` 为 `None` 或缺少 `semantic` 属性，会在访问 `manager.semantic` 时抛 `AttributeError`；如果 `semantic` 上没有 `facts` 方法则抛 `AttributeError`；`facts()` 内部若抛异常（例如存储不可达）会原样向上传播。空语义层会自然得到空列表，属于正常情况而非错误。
- **同文件关系**：它调用了本文件的 `is_fact_item`；它自己被 `reconcile_report` 调用（`facts = fact_items(manager)`）。

### `projected_vector_ids(manager: MemoryManager) -> set[str] | None` （第 52 行）
- **作用**：把向量库里现存的所有「应用级 id」枚举出来，转成一个字符串集合，供上层和真值源的 `indexed` 分块 id 做差集。它是判断「哪些分块被标记已索引但向量库里其实没有」以及「哪些向量点是孤儿」的前提。之所以返回 `None` 而不是空集合，是因为「向量库是空的」和「向量库根本没法枚举」是两件完全不同的事：前者意味着真值源里所有已索引分块都算缺失，后者意味着这次对账干脆不该报向量相关漂移，否则会把「存储不支持或不可达」误报成大规模数据丢失。函数用能力探测（看对象上有没有 `list_ids`）来决定走哪条路，从而兼容不同向量存储实现。
- **参数**：`manager`，类型 `MemoryManager`，无默认值。需要它的 `vector_store` 属性上可能带有一个可调用的 `list_ids()` 方法；这个方法存在与否直接决定返回值是集合还是 `None`。
- **返回**：返回 `set[str] | None`。当 `manager.vector_store` 上存在可调用的 `list_ids`，并且调用成功时，返回由 `list_ids()` 每个元素经 `str()` 转换后组成的集合（可能为空集，表示向量库确实为空）；当 `list_ids` 不存在或不可调用时返回 `None`；当调用过程中抛出任何异常时也返回 `None`。
- **内部流程**：先用 `getattr(manager.vector_store, "list_ids", None)` 取出候选方法；接着用 `callable(list_ids)` 判断它是否真的能调用，不能就立即 `return None`（这是「存储不支持枚举」的分支）。然后进入 `try` 块，用集合推导 `{str(value) for value in list_ids()}` 枚举并统一转成字符串——统一成字符串是为了和真值源那边的 id 集合类型对齐，避免 int 与 str 比较导致差集算错。`except Exception` 捕获一切异常并 `return None`，注释明确说明意图是「读不到就跳过向量对账，不误报漂移」。
- **异常/边界**：函数自身对外不抛异常：`AttributeError`（没有 `vector_store` 属性）实际上会发生在 `getattr` 那一行且不被 `try` 覆盖，属于唯一可能外泄的异常；`list_ids` 不存在、不可调用、调用超时或连接失败等一切内部错误都被宽泛的 `except Exception` 吞掉并转成 `None`。空向量库返回空集合（不是 `None`），这是有意区分。
- **同文件关系**：它不调用本文件里任何其它函数；被 `reconcile_report` 调用（`vectors = projected_vector_ids(manager)`）。

### `projected_edge_ids(manager: MemoryManager) -> set[str] | None` （第 64 行）
- **作用**：把图投影里现存的全部 `memory_id`（也就是每条关系边对应的记忆条目 id）枚举成字符串集合，用来和语义层的 fact 集合做差集，找出「有 fact 却没有图边」的缺失关系。它和 `projected_vector_ids` 是同一套设计思路的姊妹函数：内存回退实现和 Neo4j 实现都提供同一个方法名 `relation_memory_ids`，所以这里用能力探测加异常兜底的方式统一取数。同样地，返回 `None` 专门表示「无法枚举」，与「图里确实一条边都没有（空集合）」区分开来，避免在 Neo4j 抖动时误报全部事实缺边。
- **参数**：`manager`，类型 `MemoryManager`，无默认值。需要它的 `graph_store` 属性上可能带有一个可调用的 `relation_memory_ids()` 方法。
- **返回**：返回 `set[str] | None`。当 `manager.graph_store` 上存在可调用的 `relation_memory_ids` 且调用成功时，返回由该方法产出的、经过滤空值并 `str()` 转换后的 id 集合（可能为空集）；当方法缺失、不可调用或调用抛异常时返回 `None`。
- **内部流程**：第一步 `getattr(manager.graph_store, "relation_memory_ids", None)` 探测方法，第二步 `callable(...)` 判断，不可调用就 `return None`。第三步在 `try` 里做集合推导，注意这里比向量那版多了一个过滤条件 `if str(value)`：先对每个元素转字符串，只有转出来非空（不是 `""`）才收进集合——这能挡掉图存储里可能存在的空 id 或 `None` 之类的脏值，防止它们污染差集。异常同样被 `except Exception` 捕获并转成 `None`。
- **异常/边界**：与 `projected_vector_ids` 类似，函数内部不主动抛错；`manager` 没有 `graph_store` 属性时会在 `getattr` 行抛 `AttributeError` 且不被捕获。方法不存在、不可调用、Neo4j 不可达、查询超时等都被吞掉返回 `None`。空图返回空集合。空字符串 id 被显式丢弃。
- **同文件关系**：它不调用本文件里任何其它函数；被 `reconcile_report` 调用（`edges = projected_edge_ids(manager)`）。

### `_drift_entry(kind: str, ids: list[str], *, include_ids: bool) -> dict[str, Any]` （第 76 行）
- **作用**：把「某一类漂移」的原始 id 列表打包成统一的字典结构，也就是漂移条目的构造函数。对账结果里可能出现三类漂移（`missing_vector`、`orphan_vector`、`missing_edge`），它们的形状完全一致：一个类别名、一个数量、一份可选的 id 明细。把这段打包逻辑抽成一个带下划线前缀的内部辅助函数，是为了让 `reconcile_report` 里三处构造漂移条目的代码保持完全一致的字段名与语义，同时集中处理「只要数量不要明细」这种精简输出的需求——当调用方只关心数量时，id 明细会被换成空列表，避免返回一个巨大的 id 数组。
- **参数**：`kind`，类型 `str`，无默认值，表示漂移类别名，取值由调用方决定，本文件里只会传 `"missing_vector"`、`"orphan_vector"`、`"missing_edge"` 三种。`ids`，类型 `list[str]`，无默认值，是这一类漂移涉及的 id 列表，调用方传入的已经是排好序的列表；函数只读取它的长度和内容，不修改它。`include_ids`，类型 `bool`，是仅限关键字参数（签名里的 `*` 使其不能按位置传递），无默认值，表示是否把 `ids` 明细放进结果里。
- **返回**：返回 `dict[str, Any]`，固定含三个键：`kind`（原样回填类别名）、`count`（`len(ids)`，即这一类漂移的数量，无论是否输出明细都照实统计）、`ids`（`include_ids` 为真时是 `list(ids)` 的浅拷贝，为假时是空列表 `[]`）。
- **内部流程**：函数体只有一条 `return`，直接构造字典字面量。关键点有两个：一是 `count` 用 `len(ids)` 而不是 `len(... if include_ids ...)`，保证「只统计不列举」时数量依然准确；二是 `ids` 字段用 `list(ids)` 复制一份，避免把调用方传进来的列表对象直接暴露出去被后续修改。
- **异常/边界**：若 `ids` 为 `None` 会在 `len()` 处抛 `TypeError`，代码未做防御；空列表是合法输入，会得到 `count: 0` 的条目（不过调用方只在非空时才构造条目，所以实际不会产生 0 计数项）。不涉及 I/O，无超时概念。
- **同文件关系**：它不调用本文件其它函数；被 `reconcile_report` 在三个分支里调用，分别对应向量缺失、向量孤儿、图边缺失。

### `reconcile_report(manager: MemoryManager, repository: DocumentRepository | None, *, include_ids: bool = True) -> dict[str, Any]` （第 80 行）
- **作用**：这是整个模块的核心，负责把三个存储的现状汇总成一份「对账报告」：一份计数字典加一份漂移列表。它一次性采集真值源的分块 id 与「已标记 indexed」的分块 id、文档存储里的记忆 id、语义层的事实、向量库 id 集合、图投影 id 集合，然后做集合差运算：`indexed - vectors` 得到「真值源说已投影但向量库没有」的缺失向量，`vectors - chunk_ids - memory_ids` 得到「向量库里有、但真值源里既不是分块也不是文档」的孤儿向量，`facts - edges` 得到「有事实却没有图边」的缺失关系。它是只读的，不写任何存储，因此可以随时被 Agent 或 UI 调用；真正的修复交给别的写工具。它的另一条重要设计原则是「能对多少对多少」：某个库不可用时（例如非 SQLite 文档库、向量库无法枚举），只跳过对应的对账维度，绝不整体失败。
- **参数**：`manager`，类型 `MemoryManager`，无默认值，是记忆管理器，提供 `document_store`、`semantic`、`vector_store`、`graph_store` 等子对象。`repository`，类型 `DocumentRepository | None`，无默认值，是文档仓储（SQLite 真值源）；传 `None` 表示内存模式或非 SQLite 文档库，此时真值源的两个计数按 0 处理，但向量和图的对账照常进行。`include_ids`，类型 `bool`，仅限关键字参数，默认 `True`，控制漂移条目里是否带上具体 id 列表；只关心数量时传 `False` 可以避免返回超长列表。
- **返回**：返回 `dict[str, Any]`，含两个顶层键。`counts` 是一个字典，含五个计数：`chunks`（真值源分块去重后行数）、`chunks_indexed_sqlite`（真值源里标记为已投影的分块数）、`qdrant_points`（向量库点数，无法枚举时为 `-1`）、`facts`（语义层三元组事实条数）、`neo4j_edges`（图投影关系数，无法枚举时为 `-1`）。`drift` 是一个列表，元素是 `_drift_entry` 产出的字典；只有当对应集合能枚举且差集非空时才追加条目，因此三库一致时 `drift` 是空列表。
- **内部流程**：第一步采集真值源：若 `repository` 不为 `None`，用 `set(repository.chunk_ids())` 拿到全部分块 id，再用 `set(repository.chunk_ids(vector_status="indexed"))` 拿到其中状态为 `indexed` 的子集；若为 `None`，两者都退化为空集合。第二步用集合推导 `{item.id for item in manager.document_store.list(include_expired=True)}` 拿到文档存储里的全部记忆 id（`include_expired=True` 表示过期的也算，因为对账要看物理存在而非逻辑有效）。第三步分别调用 `fact_items(manager)`、`projected_vector_ids(manager)`、`projected_edge_ids(manager)` 拿到事实列表与两个可能为 `None` 的投影 id 集合。第四步初始化空列表 `drift`，进入向量分支：`vectors is not None` 时算 `missing = sorted(indexed - vectors)` 与 `orphan = sorted(vectors - chunk_ids - memory_ids)`，各自非空才 `drift.append(_drift_entry(...))`；注意孤儿向量的判定同时减掉了分块 id 和文档 id，因为记忆条目本身也可能被投影进向量库。第五步进入图分支：`edges is not None` 时算 `missing_edges = sorted({item.id for item in facts} - edges)`，非空才追加 `missing_edge` 条目。最后返回 `counts` 与 `drift` 组装成的字典，其中 `qdrant_points` 和 `neo4j_edges` 在对应集合为 `None` 时写 `-1`，用哨兵值表达「无法枚举」而不是伪装成 0。
- **异常/边界**：`repository.chunk_ids()` 或 `manager.document_store.list()` 抛出的异常会向上传播，本函数不吞这些错误；相对地，向量与图两路读不到数据时由 `projected_vector_ids` / `projected_edge_ids` 返回 `None`，本函数据此跳过该维度且不产生任何漂移条目（这是防误报的关键边界）。`repository=None` 是受支持的正常输入，两个真值源计数为 0，此时向量分支里 `missing` 必然为空，`orphan` 会等于 `vectors - memory_ids`，即文档 id 之外的向量点都会被视为孤儿。三个差集都做了 `sorted()`，保证输出顺序稳定、便于比对和测试。没有超时参数，超时控制由上层工具的 `timeout_seconds` 负责。
- **同文件关系**：它调用了本文件的 `fact_items`、`projected_vector_ids`、`projected_edge_ids`、`_drift_entry`；它自己被 `ReconcileTool.execute` 调用，也是 `web/app.py` 里 `/api/reconcile`、`/api/stats` 的底层实现。

### `class ReconcileInput(BaseModel)` （第 129 行）
- **作用**：这是 `knowledge.reconcile` 工具的输入参数模型，用 Pydantic 定义并校验 Agent 传进来的参数。它存在的意义是给工具一个强类型的、可自动生成 JSON Schema 的入参契约，让 `BaseTool` 框架能在执行前拦掉非法参数。目前它只承载一个开关 `include_ids`，用来让调用方在「要证据明细」和「只要数量」之间做选择。
- **类配置**：`model_config = ConfigDict(extra="forbid", strict=True)`。`extra="forbid"` 表示多传任何未声明的字段都会直接校验失败，防止模型幻觉出参数名却被静默忽略；`strict=True` 表示不做宽松类型转换，类型必须严格匹配（例如不能用 `"true"` 字符串冒充布尔值）。
- **字段**：`include_ids: bool = Field(default=True, description="是否返回漂移的具体 id 列表；只关心数量时传 false，避免长列表。")`——默认 `True`，即默认返回漂移的 id 明细。
- **方法**：本文件内没有为它定义任何方法，校验、序列化和 JSON Schema 生成全部由 Pydantic 的 `BaseModel` 基类提供。
- **同文件关系**：它被 `ReconcileTool.spec` 声明为 `input_model`，也被 `ReconcileTool.execute` 的签名用作参数类型。

### `class ReconcileCounts(BaseModel)` （第 138 行）
- **作用**：这是对账报告里「计数」部分的输出模型，把 `reconcile_report` 返回的 `counts` 字典固定成五个具名整数字段。它的价值在于让工具输出有稳定的结构：无论底层存储是哪种实现，Agent 和 Web 接口看到的字段名与含义都完全一致，并且每个字段都带描述，便于生成给模型看的 Schema。
- **类配置**：同样使用 `ConfigDict(extra="forbid", strict=True)`，禁止额外字段并要求严格类型。
- **字段**：`chunks: int`（真值源 chunks 行数）、`chunks_indexed_sqlite: int`（真值源里标记为已投影的分块数）、`qdrant_points: int`（向量库里的点数；`-1` 表示无法枚举）、`facts: int`（语义层里 (主语, 谓语, 宾语) 事实条数）、`neo4j_edges: int`（图投影里的关系数；`-1` 表示无法枚举）。五个字段全部必填，`Field` 只提供描述，没有默认值。
- **方法**：本文件内没有定义任何方法，全部继承自 `BaseModel`。
- **同文件关系**：它被 `ReconcileOutput` 用作 `counts` 字段的类型，并在 `ReconcileTool.execute` 里通过 `ReconcileCounts(**report["counts"])` 从报告字典实例化。

### `class DriftEntry(BaseModel)` （第 148 行）
- **作用**：这是漂移条目的输出模型，对应 `_drift_entry` 产出的字典结构，描述「一类漂移」的类别、数量和可选 id 明细。有了它，工具输出里的每条漂移都是一个结构化的对象，Agent 可以直接读 `kind` 决定该向 `knowledge.repair_drift` 传什么类别名，读 `ids` 定位具体数据。
- **类配置**：`ConfigDict(extra="forbid", strict=True)`，禁止额外字段、严格类型。
- **字段**：`kind: str`，描述写明取值为 `missing_vector` / `orphan_vector` / `missing_edge` 三类；`count: int`，该类别漂移的数量，没有描述也没有默认值；`ids: list[str]`，通过 `Field(default_factory=list, description="include_ids=false 时为空列表。")` 定义，默认工厂保证每个实例拿到独立的空列表而不是共享的可变默认值。
- **方法**：本文件内没有定义任何方法，全部继承自 `BaseModel`。
- **同文件关系**：它被 `ReconcileOutput` 用作 `drift` 列表的元素类型，并在 `ReconcileTool.execute` 里通过 `[DriftEntry(**entry) for entry in report["drift"]]` 从报告字典逐条实例化。

### `class ReconcileOutput(BaseModel)` （第 156 行）
- **作用**：这是 `knowledge.reconcile` 工具的整体输出模型，把 `reconcile_report` 的原始字典包装成强类型、可被框架校验和序列化的结果对象。除了直接搬运计数和漂移，它还额外提供一个派生字段 `consistent`，让调用方不必自己去数 `drift` 是否为空，一眼就能判断三库是否一致。
- **类配置**：`ConfigDict(extra="forbid", strict=True)`，禁止额外字段、严格类型。
- **字段**：`counts: ReconcileCounts`，嵌套的计数模型，必填；`drift: list[DriftEntry]`，通过 `Field(default_factory=list)` 定义，默认空列表，表示没有漂移；`consistent: bool`，描述为「true 表示三库计数无漂移」，必填，由 `execute` 里用 `not report["drift"]` 计算。
- **方法**：本文件内没有定义任何方法，全部继承自 `BaseModel`。
- **同文件关系**：它被 `ReconcileTool.spec` 声明为 `output_model`，也被 `ReconcileTool.execute` 作为构造并返回的结果类型；它内部组合了 `ReconcileCounts` 与 `DriftEntry`。

### `class ReconcileTool(BaseTool)` （第 164 行）
- **作用**：这是把对账能力暴露给 Agent 的工具类，继承自 `core.BaseTool`。它承担三件事：用类属性 `spec` 向工具注册表声明自己的名字（`knowledge.reconcile`）、描述、版本、输入输出模型、副作用等级、权限、超时、幂等性、并行安全性和标签；用构造函数允许外部注入一个 `MemoryManager`（也支持懒加载默认管理器）；用 `execute` 把参数转成一次 `reconcile_report` 调用再包装成输出模型。声明 `side_effect="read"`、`permissions=()`、`idempotent=True`、`parallel_safe=True`、`timeout_seconds=60.0`，明确告诉框架这是一个只读、可随时并发运行、不会改变任何数据的工具。
- **类属性 `spec`**：`ToolSpec` 实例。`name="knowledge.reconcile"`；`description` 用英文说明它检查 SQLite 分块（真值源）、向量投影、图投影三者是否一致，报告计数以及各类漂移（标记为 indexed 却不在向量库的分块、孤儿向量、没有图边的 fact），并强调只读、修复请用 `knowledge.repair_drift`；`version="1.0.0"`；`input_model=ReconcileInput`；`output_model=ReconcileOutput`；`side_effect="read"`；`permissions=()`；`timeout_seconds=60.0`；`idempotent=True`；`parallel_safe=True`；`tags=("knowledge", "reconcile", "storage", "drift", "read")`；`guidance` 用中文提示「怀疑三库不一致时（召回缺结果、图里少边、统计对不上）先跑它拿证据」，并叮嘱要修复时再调 `knowledge.repair_drift`，且只传它报告的漂移类别、不要自己猜类别名。
- **同文件关系**：它引用了本文件的 `ReconcileInput`、`ReconcileOutput`、`ReconcileCounts`、`DriftEntry`、`reconcile_report`；它自己被本文件的 `create_tool` 实例化。它内部还引用了同包内的 `._memory.build_default_manager` 与 `.hybrid_index.repository_for`（这两个属于其它文件）。

#### `__init__(self, manager: MemoryManager | None = None) -> None` （第 188 行）
- **作用**：构造工具实例，并支持依赖注入。把 `manager` 做成可选参数，使得生产环境可以不传、由 `manager` 属性在第一次使用时懒加载默认管理器，而测试或特殊场景可以传入一个现成的（甚至是被 mock 的）`MemoryManager`，从而不必真的去连数据库。它只保存引用、不做任何校验或连接，因此构造成本极低。
- **参数**：`self`，实例本身。`manager`，类型 `MemoryManager | None`，默认 `None`；传 `None` 表示「先不指定，用到时再建默认管理器」，传实例则固定使用该实例。
- **返回**：返回 `None`（构造函数不返回值），效果是把参数存进 `self._manager`。
- **内部流程**：唯一一步是 `self._manager = manager`，把入参原样存入私有属性 `_manager`。不触发任何 I/O，不调用 `build_default_manager`。
- **异常/边界**：无特殊处理，不会主动抛异常；即使传入的不是 `MemoryManager` 而是任意对象也不会在这里报错，问题会推迟到真正调用 `execute` 时才暴露。
- **同文件关系**：它设置的状态被本类的 `manager` 属性和 `execute` 方法读取；它被 `create_tool` 间接调用（`ReconcileTool()`）。

#### `manager` （property，第 191 行）
- **作用**：这是一个只读属性，用来获取当前工具使用的 `MemoryManager`，并实现「懒加载默认管理器」。它让工具在构造时不依赖运行时环境（例如不需要在导入阶段就初始化存储连接），而在真正需要执行对账时才按需构建，这既加快了工具注册的速度，也避免了在没有配置存储的环境里导入即失败。构建一次后会缓存到 `self._manager`，后续访问直接复用，不会重复创建。
- **参数**：只有 `self`，无其它参数（属性形式，调用方写 `tool.manager` 而不是 `tool.manager()`）。
- **返回**：返回 `MemoryManager`，保证非 `None`：要么是构造时注入的那个实例，要么是本次懒加载创建的默认管理器。
- **内部流程**：先判断 `self._manager is None`；若是，则执行函数内的延迟导入 `from ._memory import build_default_manager`（放在函数体内是为了避免模块级循环导入），调用 `build_default_manager()` 得到实例并赋给 `self._manager`；最后无条件 `return self._manager`。当 `_manager` 已有值时跳过创建分支直接返回。
- **异常/边界**：如果 `build_default_manager` 不存在或构建过程中失败（例如配置缺失、存储不可达），异常会向上传播，属性访问失败；函数本身不做 try/except，也没有重试。并发场景下若两个线程同时首次访问，可能各自构建一次默认管理器（代码未加锁），但由于 `parallel_safe=True` 的工具通常在同一进程内共享实例，这是可接受的权衡。
- **同文件关系**：它被本类的 `execute` 通过 `self.manager` 调用，并把结果同时作为 `reconcile_report` 的第一个实参和 `repository_for(...)` 的实参；它读取 `__init__` 写入的 `self._manager`。

#### `execute(self, arguments: ReconcileInput) -> ReconcileOutput` （第 199 行）
- **作用**：这是工具的实际执行入口，由 `BaseTool` 框架在校验完参数后调用。它把入参模型里的 `include_ids` 转交给核心函数 `reconcile_report`，再把手写的字典报告转成强类型的 `ReconcileOutput` 返回。它承担了「工具层」与「纯逻辑层」之间的适配职责：核心函数返回普通字典便于复用，工具层负责把它塞进 Pydantic 模型以获得框架校验与序列化能力。同时它负责解析出文档仓储（真值源），这样 `reconcile_report` 本身不必关心仓储从哪来。
- **参数**：`self`，实例本身。`arguments`，类型 `ReconcileInput`，无默认值，是框架已校验过的入参对象，其中 `include_ids` 控制是否输出漂移 id 明细。
- **返回**：返回 `ReconcileOutput` 实例，含 `counts`（由 `ReconcileCounts(**report["counts"])` 构造）、`drift`（由报告里每个条目构造 `DriftEntry` 组成的列表）、`consistent`（`not report["drift"]`，即漂移列表为空时为 `True`）。
- **内部流程**：第一步在函数体内延迟导入 `from .hybrid_index import repository_for`，避免模块级循环依赖。第二步调用 `reconcile_report(self.manager, repository_for(self.manager), include_ids=arguments.include_ids)`——注意 `self.manager` 会被访问两次，第一次触发懒加载并缓存，第二次直接命中缓存；`repository_for` 根据管理器决定返回 `DocumentRepository` 还是 `None`。第三步用报告字典构造输出：`counts` 走 `**` 展开直接喂给 `ReconcileCounts`，`drift` 用列表推导逐条 `DriftEntry(**entry)`，`consistent` 用 `not report["drift"]` 计算后一并返回。
- **异常/边界**：参数校验已由框架在调用前完成，所以这里假定 `arguments` 合法。若 `repository_for` 抛异常（例如仓储不可用）或 `reconcile_report` 内部抛异常，异常会向上传播给框架处理；若报告字典的键与模型字段不匹配，`ReconcileCounts` / `DriftEntry` 因 `extra="forbid"` 会抛 Pydantic 校验错误——这正是期望的失败方式，能在第一时间暴露接口漂移。`arguments.include_ids=False` 时漂移条目里的 `ids` 会是空列表，但 `count` 仍然准确。超时不由本方法处理，而是由 `spec.timeout_seconds=60.0` 交给框架。
- **同文件关系**：它调用了本文件的 `reconcile_report`，并构造 `ReconcileCounts`、`DriftEntry`、`ReconcileOutput`；它通过 `self.manager` 使用本类的属性；它被 `BaseTool` 框架在工具调用时执行，本文件内没有其它函数调用它。

### `create_tool() -> BaseTool` （第 214 行）
- **作用**：这是工具的工厂函数，用来按需创建一个 `ReconcileTool` 实例。工具注册表通常约定每个工具模块提供一个无参工厂，这样注册流程不需要知道各个工具类各自的构造签名，也能保证每次拿到的是干净的新实例而不是共享的全局单例。这里不传 `manager`，因此创建的实例处于懒加载状态，第一次执行时才构建默认管理器。
- **参数**：无参数。
- **返回**：返回 `BaseTool`，实际类型是 `ReconcileTool`（以基类类型标注，便于注册表统一处理）。
- **内部流程**：函数体只有一条 `return ReconcileTool()`，直接用默认参数构造并返回。
- **异常/边界**：无特殊处理；构造过程本身几乎不会失败（不触发 I/O），若 `ReconcileTool` 的定义或基类导入有问题才会在导入或调用时暴露。
- **同文件关系**：它调用本文件的 `ReconcileTool` 构造函数；它自己不被本文件内任何函数调用，是对外（工具注册表）暴露的入口之一，并被列在 `__all__` 中。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `is_fact_item` | 判断一条记忆条目的元数据里是否同时具备 subject/predicate/object，从而确认它是一条事实。 |
| `fact_items` | 从记忆管理器的语义层取出所有通过 `is_fact_item` 过滤的事实行列表。 |
| `projected_vector_ids` | 枚举向量库里的全部应用级 id 并转成字符串集合，无法枚举时返回 `None` 以跳过对账。 |
| `projected_edge_ids` | 枚举图投影里的全部关系 memory_id 并转成字符串集合，无法枚举时返回 `None` 以跳过对账。 |
| `_drift_entry` | 把一类漂移的类别、数量和可选 id 明细打包成统一字典。 |
| `reconcile_report` | 汇总三库计数并对真值源、向量投影、图投影做差集，产出计数与漂移报告。 |
| `ReconcileInput` | `knowledge.reconcile` 工具的入参模型，唯一字段是是否返回漂移 id 明细的 `include_ids`。 |
| `ReconcileCounts` | 报告里计数部分的输出模型，固定 chunks、chunks_indexed_sqlite、qdrant_points、facts、neo4j_edges 五个字段。 |
| `DriftEntry` | 单条漂移的输出模型，含类别 kind、数量 count 和可选 id 列表。 |
| `ReconcileOutput` | 工具整体输出模型，组合 counts 与 drift，并给出派生布尔字段 consistent。 |
| `ReconcileTool` | 把对账能力封装成名为 `knowledge.reconcile` 的只读 BaseTool 工具类。 |
| `ReconcileTool.__init__` | 保存可选注入的 MemoryManager 引用，支持懒加载默认管理器。 |
| `ReconcileTool.manager` | 只读属性，按需构建并缓存默认 MemoryManager 后返回。 |
| `ReconcileTool.execute` | 解析文档仓储、调用 `reconcile_report`，并把字典报告包装成 `ReconcileOutput` 返回。 |
| `create_tool` | 无参工厂函数，返回一个默认配置的 `ReconcileTool` 实例。 |
