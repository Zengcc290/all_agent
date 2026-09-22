# memory/types/semantic.py

## 一、这个文件是干什么的

这个文件是四层记忆体系中「语义记忆（Semantic Memory）」这一层的具体实现，文件顶部注释就写明了它的定位：*Semantic memory with graph-backed entity relations*，即「带图结构实体关系的语义记忆」。它对外只导出一个类 `SemanticMemory`，这个类继承自 `memory/base.py` 里的 `BaseMemory`，并把 `memory_type` 固定为 `MemoryType.SEMANTIC`，从而在统一的记忆框架里占据「语义层」这个槽位。

它的核心职责是把「事实三元组」这种知识形态同时写进两套存储：一套是基类提供的文档存储（`document_store`，负责真正的真值保存、按 id 读取、列出、删除、清空），另一套是可选的图数据库（`Neo4jGraphStore`，负责实体—关系—观察的图投影）。也就是说，语义记忆里的每一条事实都以 `MemoryItem` 的形式落在文档存储里，同时在图里被投影成边（`add_relation`）和观察节点（`add_observation`），从而既能按 id 精确取回事实文本，又能按实体做图遍历式的关联查询。

文件里实现的功能点包括：写入/合并一条事实（`add_fact`）、给图边组装属性字典（`_graph_properties`）、读取端点实体的附加属性（`_endpoint_attributes`）、把一条事实完整投影到图里（`_write_edge`）、以关系名调用写入（`add_relation`）、删除一条事实并同步删掉图上的边（`delete`）、清空整层记忆与整张图（`clear`）、按实体查询关联（`related`）、以及按实体过滤本地事实列表（`facts`）。

它在运行时被上层使用的方式很典型：知识抽取流水线抽到 `(subject, predicate, object)` 之后调用 `add_fact` 落库，GraphRAG 或问答侧通过 `related` 拿到某个实体在图上的邻居关系，通过 `facts` 拿到本地事实条目列表；当记忆需要重写或清理时走 `delete` / `clear`。文件里还专门处理了一个历史遗留问题：早期写入用的是 `fact:` 前缀的 id，统一后改用 `relation:` 前缀的 id，代码通过 legacy id 回退逻辑保证旧数据能被原地更新而不是被重复插入。

整体来看，这个文件是「语义层」的存储适配器：向上提供简洁的语义 API（写事实、查关系、删事实），向下屏蔽「文档存储 + 图存储」双写的复杂性，并且在双写之间维持 `active`、`superseded_by`、`confidence` 等状态字段的一致性。

## 二、函数与类逐条详解

### `class SemanticMemory(BaseMemory)` （第 15 行）
- **作用**：这是整个文件唯一的类，也是语义记忆层的公开门面。它通过继承 `BaseMemory` 复用通用记忆能力（文档存储、`add`、`list`、`delete`、`clear`、`get` 等），通过类属性 `memory_type = MemoryType.SEMANTIC` 把自己注册成「语义」这一类记忆，从而让上层可以按类型分派到它。它额外持有一个 `graph_store` 属性，把「事实真值」与「实体关系图」绑定在一起：真值放文档存储，关系拓扑放图存储。需要它的时候，通常是运行时在构建四层记忆时实例化它，或者抽取流水线/检索链路需要写入或查询实体关系时调用它的方法。
- **参数**：类本身不接收参数；实例化参数由 `__init__` 定义，且构造时是仅关键字（keyword-only）参数：`graph_store`（可选的 `Neo4jGraphStore`）以及任意透传给基类的 `**kwargs`。
- **返回**：类本身无返回值，它的实例就是语义记忆对象。
- **内部流程**：类体里只做了两件事——声明类属性 `memory_type`，以及定义全部实例方法（`__init__`、`add_fact`、`_graph_properties`、`_endpoint_attributes`、`_write_edge`、`add_relation`、`delete`、`clear`、`related`、`facts`）。真正的运行逻辑分布在这些方法里。
- **异常/边界**：类定义阶段无异常处理；继承自 `BaseMemory`，若基类初始化要求某些参数而构造时缺失，异常会在 `__init__` 里由基类抛出。
- **同文件关系**：它包含并组织了本文件所有的方法；方法之间通过 `self.` 相互调用（例如 `add_fact` 调用 `_write_edge`，`_write_edge` 调用 `_endpoint_attributes` 和 `_graph_properties`）。

### `__init__(self, *, graph_store: Neo4jGraphStore | None = None, **kwargs: Any) -> None` （第 18 行）
- **作用**：构造语义记忆实例，是整个类唯一的初始化入口。它先调用基类 `BaseMemory` 的初始化把通用记忆状态（记忆类型、文档存储等）准备好，然后确保一定存在一个可用的图存储对象：如果调用方传了 `graph_store` 就用传进来的，没传就延迟导入并新建一个 `Neo4jGraphStore`。这样设计的原因是这个文件被导入时不应立刻依赖图数据库的可用性（所以 `Neo4jGraphStore` 只在 `TYPE_CHECKING` 里做类型标注，真实导入放在函数体内），同时也让测试或离线场景可以注入一个假的图存储替换真实 Neo4j。
- **参数**：
  - `graph_store`：`Neo4jGraphStore | None`，默认 `None`，仅关键字参数。为 `None` 时会在内部延迟导入并实例化一个真实的 `Neo4jGraphStore`；传入实例时直接使用，不做类型校验。
  - `**kwargs`：`Any`，任意额外的关键字参数，原样透传给 `BaseMemory.__init__`，用于配置基类行为（例如记忆类型相关的配置）；本文件不解析这些参数。
- **返回**：`None`。构造函数没有返回值，副作用是设置 `self.graph_store`。
- **内部流程**：第一步调用 `super().__init__(memory_type=self.memory_type, **kwargs)`，把类属性上的 `MemoryType.SEMANTIC` 显式传给基类并透传其余关键字参数；第二步判断 `graph_store is None`，成立则执行函数内的 `from ..storage import Neo4jGraphStore` 延迟导入，再调用 `Neo4jGraphStore()` 无参构造；第三步把结果赋给 `self.graph_store`，供后续 `_write_edge`、`delete`、`clear`、`related` 使用。
- **异常/边界**：若 `graph_store` 为 `None` 且延迟导入 `..storage.Neo4jGraphStore` 失败（模块缺失或循环导入），会抛出 `ImportError`；若 `Neo4jGraphStore()` 构造时因为缺少连接配置而失败，异常会原样向上冒泡。传入 `graph_store=None` 之外的空值（例如 `False`）不会被判定为 `None`，会被直接当作图存储使用并在后续调用时报错。基类初始化抛出的异常不做捕获。
- **同文件关系**：被本文件其他所有方法间接受益（它们通过 `self.graph_store` 访问图存储）；直接调用基类 `BaseMemory.__init__`。本文件内没有其他函数调用它。

### `add_fact(self, subject: str, predicate: str, object: str, *, metadata: Mapping[str, Any] | None = None, confidence: float = 1.0, item_id: str | None = None) -> MemoryItem` （第 28 行）
- **作用**：这是语义层最重要的写入口，负责把一条 `(subject, predicate, object)` 事实三元组写入语义记忆，并同步投影到图里。它同时承担「新建」和「更新/合并」两种语义：如果同一个三元组已经存在，它不会插入第二条，而是把元数据合并、把 `importance` 抬到旧值与新置信度的较大者，再以同一个 id 覆盖写回，并刷新图上的边。它还必须处理一个历史兼容问题——旧版本写入的事实 id 前缀是 `fact:`，新版本统一为 `relation:`，所以它在找不到新 id 时会回退去查 legacy id，把旧条目原地更新，避免同一事实被存成两份。抽取流水线写入事实、上层手工补一条知识、以及撤回/重新激活一条事实（通过 `metadata` 里带 `active`）都会走到这里。
- **参数**：
  - `subject`：`str`，必填，事实的主语。必须是「非空字符串」，纯空白字符串会被判为非法。
  - `predicate`：`str`，必填，事实的谓词（关系名）。同样必须是非空、非纯空白字符串。
  - `object`：`str`，必填，事实的宾语。注意这个名字遮蔽了内置函数 `object`，在本方法体内 `object` 指的就是这个字符串参数。同样必须非空、非纯空白。
  - `metadata`：`Mapping[str, Any] | None`，默认 `None`，仅关键字参数。附加元数据，会与已有元数据做浅合并；常见键包括 `evidence`、`source`、`active`、`roles`、`domain`、时间字段等。为 `None` 时视为空字典。
  - `confidence`：`float`，默认 `1.0`，仅关键字参数。置信度，约束是「数值类型且落在 `[0, 1]` 闭区间内」；布尔值被显式拒绝（因为 `bool` 是 `int` 的子类，`True`/`False` 会绕过数值判断，所以代码单独用 `isinstance(confidence, bool)` 拦截）。新建条目时它同时作为 `importance`。
  - `item_id`：`str | None`，默认 `None`，仅关键字参数。显式指定条目 id；给了就用它，不给就按 `relation_id_for(subject, predicate, object)` 推导。给了 `item_id` 时会跳过 legacy id 回退查询。
- **返回**：返回 `MemoryItem`，即写入（或更新）后的那条语义记忆条目。无论是新建分支还是合并分支，最终返回的都是 `self.add(...)` 的返回值（合并分支返回的是被刷新过的同 id 条目）。
- **内部流程**：
  1. 合法性校验：用 `all(...)` 加生成器检查 `subject`/`predicate`/`object` 三者是否都是 `str` 且 `strip()` 后非空，任一不满足就抛 `ValueError`。
  2. 置信度校验：若 `confidence` 是 `bool`、或不是 `int`/`float`、或不在 `0 <= confidence <= 1` 范围内，抛 `ValueError`。
  3. 计算 id：`fact_id = item_id or relation_id_for(subject, predicate, object)`，与抽取流水线使用同一套规范化 id 方案（注释里说明这是为了避免同一三元组既存成 `fact:...` 又存成 `relation:...`）。
  4. 查重：用 `self.document_store.get(fact_id)` 取已有条目 `existing`。
  5. legacy 回退：若 `existing is None` 且调用方没显式给 `item_id`，用 `legacy_fact_id_for(subject, predicate, object)` 算出旧 id 并再查一次；若查到且其 `memory_type` 等于本类的 `memory_type`（语义层），则把 `fact_id` 与 `existing` 都切换到 legacy 那一份，实现旧数据原地升级。
  6. 合并分支（`existing is not None` 且类型匹配）：把 `existing.metadata` 复制成 `merged_metadata`，再用 `metadata` 覆盖上去；接着判断——只有当 `metadata is None` 或其中没有 `"active"` 键时，才把 `merged_metadata["active"]` 置为 `True`、`"superseded_by"` 置为 `[]`、`"superseded_at"` 置为 `""`（注释说明：显式传入 `active=False` 必须能在合并中存活，这样撤回操作才是幂等的）；然后调用 `self.add(文本, metadata=merged_metadata, importance=max(existing.importance, confidence), item_id=fact_id)` 覆盖写入；最后调用 `self._write_edge(...)` 把合并后的状态刷到图上，避免图上的边一直保留旧的 `active`/`memory_id` 而与文档存储不一致；返回 `updated`。
  7. 新建分支：把 `metadata` 复制成 `item_metadata`，再强制写入 `subject`、`predicate`、`object`、`confidence` 四个键（覆盖用户传入的同名键）；调用 `self.add(f"{subject} {predicate} {object}", metadata=item_metadata, importance=confidence, item_id=fact_id)`；然后 `self._write_edge(subject, predicate, object, item, item_metadata)`；返回 `item`。
  8. 两个分支的文本表示都是 `"{subject} {predicate} {object}"`，也就是三元组以空格拼接后的自然语句形式作为条目内容。
- **异常/边界**：`ValueError` 有两种触发场景——三元组元素非法，或 `confidence` 非法（含布尔值）。若 `existing` 存在但 `memory_type` 不是语义类型（例如同一个 id 被别的记忆层占用），代码不会走合并分支，而是继续往下走新建分支并尝试用同一 id 覆盖写入；legacy 回退同样只在类型匹配时才采用旧 id。`metadata=None` 被安全地当成空映射处理（`dict(metadata or {})`）。`importance=max(existing.importance, confidence)` 要求 `existing.importance` 可与 `confidence` 比较，若历史数据里该字段类型异常则会在比较时报 `TypeError`。图写入失败（`_write_edge` 内部异常）不会被捕获，会向上抛出，此时文档存储里的条目已经写入，存在双写不一致的可能。
- **同文件关系**：调用了本文件的 `_write_edge`（两个分支各一次）；间接依赖 `_graph_properties` 与 `_endpoint_attributes`（由 `_write_edge` 调用）。被本文件的 `add_relation` 直接调用（`add_relation` 只是它的别名式包装）。另外调用了基类的 `self.add` 与 `self.document_store.get`，以及 `memory/ids.py` 的 `relation_id_for`、`legacy_fact_id_for`。

### `_graph_properties(self, item: MemoryItem, metadata: Mapping[str, Any]) -> dict[str, Any]` （第 99 行）
- **作用**：把一条事实的元数据整理成「图边属性字典」。它存在的意义是保证无论从哪条写入路径进图，写到 Neo4j 边上的键集合都完全一致（文档字符串明确写了 *the same key set is written on every path*），否则不同路径写出的边属性参差不齐，查询和图算法就会拿到不一致的数据。它固定写入 `memory_id` 和 `confidence` 两个基础键，再从一份白名单里挑选实际存在的元数据键拷贝进去，白名单之外的键会被丢弃，从而避免把任意用户元数据（可能很大或类型不受 Neo4j 支持）灌进图数据库。
- **参数**：
  - `item`：`MemoryItem`，必填。用来取 `item.id` 作为边属性里的 `memory_id`，这是图边回指文档存储条目的关键字段。
  - `metadata`：`Mapping[str, Any]`，必填。事实的元数据映射，可能来自新建路径的 `item_metadata` 或合并路径的 `merged_metadata`。
- **返回**：`dict[str, Any]`，一个全新的属性字典。至少包含 `memory_id`（值为 `item.id`）与 `confidence`（取 `metadata["confidence"]`，缺失时用默认 `1.0`）；此外按白名单条件性包含：`evidence`、`source`、`source_document`、`chunk_id`、`predicate_key`、`action`、`cardinality`、`active`、`superseded_by`、`superseded_at`、`supersedes`、`valid_from`、`valid_to`、`status`、`event_at`、`captured_at`、`modality`、`observation_id`。
- **内部流程**：先构造基础字典 `{"memory_id": item.id, "confidence": metadata.get("confidence", 1.0)}`；然后用一个 `for key in (...)` 循环遍历写死在代码里的 18 个候选键名，每个键用 `if key in metadata` 判断是否存在，存在才 `properties[key] = metadata[key]` 原样拷入（不做类型转换）；最后返回这个字典。
- **异常/边界**：本身不抛异常，也不做类型校验——如果 `metadata` 里某个白名单键的值类型 Neo4j 不接受，异常会在真正写图时抛出。`metadata` 缺少 `confidence` 时用 `1.0` 兜底，所以不会 `KeyError`。注意 `confidence` 的兜底只在「键不存在」时生效，若键存在但值为 `None`，会原样写入 `None`。白名单外的键被静默忽略，不报错也不警告。
- **同文件关系**：被本文件的 `_write_edge` 调用（`_write_edge` 里以 `self._graph_properties(item, metadata)` 形式使用），`_write_edge` 又把结果同时用于主关系的 `add_relation` 与 `roles` 里的兼容投影关系，以及 `add_observation` 的属性。它自身不调用本文件任何其他函数。

### `_endpoint_attributes(self, name: str) -> dict[str, Any]` （第 133 行）
- **作用**：查询某个实体端点在文档存储里对应的「实体条目」，并从中提取图节点需要的附加属性。抽取流水线会先写实体再写关系，所以正常情况下能查到；查不到时这个方法选择「安静地返回空字典」，让图节点保持默认值，而不是报错中断整个事实写入——文档字符串明确写了 *a miss simply leaves the graph node at its defaults*。它返回的 `domain`、`aliases`、`importance`、`entity_type` 会被 `_write_edge` 用来给 Neo4j 的实体节点补充领域、别名、重要度和实体类型。
- **参数**：
  - `name`：`str`，必填。实体名称，方法内部用 `entity_id_for(name)` 把它规范化成实体条目的 id 去文档存储里查。
- **返回**：`dict[str, Any]`。命中且类型正确时返回四个键：`domain`（`str`，取 `metadata["domain"]`，缺失时为空串）、`aliases`（`list`，取 `metadata["aliases"]`，值为假时退化为空列表）、`importance`（`float`，由 `item.importance` 转换而来）、`entity_type`（`str`，取 `metadata["entity_type"]`，缺失时默认为中文「概念」）。未命中或类型不符时返回空字典 `{}`。
- **内部流程**：第一步 `self.document_store.get(entity_id_for(name))` 取条目 `item`；第二步判断 `item is None or item.metadata.get("kind") != "entity"`，任一成立就 `return {}`（也就是只有 `kind == "entity"` 的条目才被认作实体）；第三步构造并返回包含 `domain`、`aliases`、`importance`、`entity_type` 四个键的字典，其中 `aliases` 用 `list(... or [])` 保证是列表，`importance` 用 `float(...)` 强转。
- **异常/边界**：不主动抛异常。若条目存在但 `item.metadata` 不是映射、或 `item.importance` 无法转成 `float`（例如是 `None` 或非数值字符串），`float(item.importance)` 会抛 `TypeError`/`ValueError`。`domain`、`entity_type` 用 `str(...)` 强转，基本不会失败。条目存在但 `kind` 不是 `"entity"`（例如是事实条目或观察条目）时按未命中处理，返回空字典。
- **同文件关系**：被本文件的 `_write_edge` 调用三次——分别用于主语（`source_entity`）、宾语（`target_entity`），以及 `metadata["roles"]` 里每个额外参与者的值。它自身不调用本文件其他函数，只调用 `memory/ids.py` 的 `entity_id_for`。

### `_write_edge(self, subject: str, predicate: str, object: str, item: MemoryItem, metadata: Mapping[str, Any]) -> None` （第 150 行）
- **作用**：把一个已经落库的事实完整投影到图存储，是这个文件里最重的私有方法。它不只写一条边，而是写三类东西：一条主语到宾语的主关系边；对 `metadata["roles"]` 里声明的每个额外语义角色各写一条兼容性关系边（用角色名作为谓词，主语指向角色值）；以及一个「观察」节点（`add_observation`），把主语、宾语和所有额外角色作为参与者挂在同一次观察下。这样做的原因是文档注释里写明的：已有 GraphRAG 走的是 `RELATED` 边遍历，而权威的 n 元结构是 `MemoryObservation + HAS_PARTICIPANT` 拓扑，所以两种形态必须同时存在，前者服务旧查询路径，后者保证语义表达力。它在 `add_fact` 的新建与合并两个分支末尾都会被调用，确保文档存储和图存储的状态始终对齐。
- **参数**：
  - `subject`：`str`，必填，主语实体名。
  - `predicate`：`str`，必填，谓词名，同时作为主关系边的类型和观察的谓词。
  - `object`：`str`，必填，宾语实体名。
  - `item`：`MemoryItem`，必填，已写入文档存储的事实条目，用来取 `item.id`、`item.created_at`，并作为边属性里的 `memory_id`。
  - `metadata`：`Mapping[str, Any]`，必填，事实元数据，用于生成边属性、读取 `roles`、读取 `domain`。
- **返回**：`None`。所有效果都是副作用（写图数据库），没有返回值。
- **内部流程**：
  1. 分别调用 `self._endpoint_attributes(subject)` 与 `self._endpoint_attributes(object)` 得到 `source_entity`、`target_entity`。
  2. 调用 `self._graph_properties(item, metadata)` 得到边属性 `properties`。
  3. 调用 `self.graph_store.add_relation(subject, predicate, object, properties=properties, source_domain=..., target_domain=..., source_aliases=..., target_aliases=..., source_importance=..., target_importance=...)`：领域与别名从 `source_entity`/`target_entity` 取（缺失用空串与空列表兜底），重要度用 `float(...)` 且默认 `0.5`。
  4. 构造 `participants` 列表，先放两个固定参与者：主语（`role="subject"`、`ordinal=0`）与宾语（`role="object"`、`ordinal=1`），每个都用 `**source_entity` / `**target_entity` 展开，把 `domain`、`aliases`、`importance`、`entity_type` 一并带上。
  5. 用 `enumerate(metadata.get("roles") or [], start=2)` 遍历额外角色，序号从 2 开始递增。对每一项：不是 `Mapping` 就 `continue` 跳过；取 `role` 与 `value` 并 `strip()`，任一为空就 `continue` 跳过；用 `self._endpoint_attributes(value)` 取该角色值的实体属性；组装 `participant`（`name`、`role`、`ordinal`，加上展开的 `attributes`）；如果原始角色项里带 `entity_type`，则用它覆盖 `participant["entity_type"]`；追加到 `participants`；同时为该角色调用一次 `self.graph_store.add_relation(subject, role_name, value, properties={**properties, "observation_id": item.id}, ...)` 写兼容性边（属性上额外打上 `observation_id`，领域/别名/重要度取自 `attributes`）。
  6. 用 `getattr(self.graph_store, "add_observation", None)` 探测图存储是否实现了 `add_observation`，再用 `callable(...)` 判断可调用；满足时调用 `add_observation(item.id, predicate, participants, properties={**properties, "domain": str(metadata.get("domain") or ""), "created_at": item.created_at.isoformat()})`，把观察 id 设为条目 id，并补上 `domain` 与 ISO 格式的创建时间。
- **异常/边界**：不主动做参数校验（校验已在上游 `add_fact` 完成）。`metadata.get("roles")` 为 `None` 或空时用 `or []` 兜底，不会报错；`roles` 中非映射项、缺 `role`、缺 `value` 的项都被静默跳过而不报错，这是刻意的容错设计。`add_observation` 通过 `getattr` + `callable` 双重探测，图存储没实现该方法时直接跳过，不会 `AttributeError`。`item.created_at.isoformat()` 假定 `created_at` 是 `datetime`，若为 `None` 会抛 `AttributeError`。图存储调用本身抛出的异常（连接失败、Cypher 错误等）不捕获，会向上冒泡到 `add_fact` 的调用方；此时文档存储已写入而图可能只写了一半，存在部分写入的风险。
- **同文件关系**：调用了本文件的 `_endpoint_attributes`（至少两次，roles 非空时更多）与 `_graph_properties`（一次）；被本文件的 `add_fact` 在两个返回分支前各调用一次。它是本文件内唯一直接与 `self.graph_store` 交互完成写入的方法。

### `add_relation(self, source: str, relation: str, target: str, *, metadata: Mapping[str, Any] | None = None) -> MemoryItem` （第 234 行）
- **作用**：这是给「关系」语汇调用方准备的便捷别名。它把参数名从 `(subject, predicate, object)` 换成更符合图数据库直觉的 `(source, relation, target)`，然后原样转发给 `add_fact`，因此继承 `add_fact` 的全部行为——包括合法性校验、置信度默认 `1.0`、id 规范化、legacy id 回退、元数据合并、图投影。它存在的原因是不同调用方（图视角 vs 事实视角）用词不同，提供一个语义更贴合的入口能减少误用；同时它也是公开 API 的一部分，让外部代码不必知道内部统一叫 `add_fact`。
- **参数**：
  - `source`：`str`，必填，关系起点实体名，对应 `add_fact` 的 `subject`。
  - `relation`：`str`，必填，关系名，对应 `add_fact` 的 `predicate`。
  - `target`：`str`，必填，关系终点实体名，对应 `add_fact` 的 `object`。
  - `metadata`：`Mapping[str, Any] | None`，默认 `None`，仅关键字参数，直接透传。
- **返回**：返回 `MemoryItem`，就是 `add_fact` 的返回值。
- **内部流程**：只有一步——`return self.add_fact(source, relation, target, metadata=metadata)`。它不设置 `confidence`，也不设置 `item_id`，因此走的是 `add_fact` 的默认置信度 `1.0` 与自动推导 id 的路径。
- **异常/边界**：异常完全由 `add_fact` 决定：三元组非法或（在默认路径下不可能出现的）置信度非法会抛 `ValueError`。本方法自身没有额外的空值或类型处理，也没有任何兜底。
- **同文件关系**：直接调用本文件的 `add_fact`，自身不被本文件其他函数调用。

### `delete(self, item_id: str) -> bool` （第 244 行）
- **作用**：删除一条语义事实，并同步删除它在图上对应的关系投影。它先确认待删条目确实存在且属于语义记忆层（`memory_type` 匹配），只有满足条件才真正调用基类的 `delete`；删除成功后，再用 `getattr` + `callable` 探测图存储是否提供 `delete_memory_relation`，有就按 `item_id` 删掉图上的边。之所以要先做类型检查，是为了防止用一个恰好属于其他记忆层（情景层、程序层等）的 id 误删语义层数据，也避免把别的层的条目从图上误删。记忆重写、事实撤回后清理、以及上层主动纠错时都会调用它。
- **参数**：
  - `item_id`：`str`，必填。待删除条目的 id，通常是 `relation:...` 形式的规范化事实 id，也可能是 legacy `fact:...` id 或调用方显式传入的自定义 id。
- **返回**：`bool`。返回 `False` 表示条目不存在、或存在但 `memory_type` 不是语义类型、或基类 `delete` 返回了假值；返回 `True` 表示文档存储里确实删掉了该条目（图上的边若图存储支持删除，也一并删除）。
- **内部流程**：第一步 `self.document_store.get(item_id)` 取条目；第二步判断 `item is None or item.memory_type != self.memory_type`，成立直接 `return False`；第三步调用 `super().delete(item_id)` 得到 `removed`；第四步若 `removed` 为真，用 `getattr(self.graph_store, "delete_memory_relation", None)` 取方法并判断 `callable`，可调用则执行 `remove_relation(item_id)`；第五步返回 `removed`。
- **异常/边界**：本方法不抛自定义异常。条目不存在或类型不符时静默返回 `False`，不报错。图存储没有 `delete_memory_relation` 方法时跳过图删除，文档存储的删除结果仍然返回 `True`，此时图里可能残留边（静默不一致）。图删除本身抛出的异常不被捕获，会向上冒泡——注意此时文档存储里的条目已经被删掉了。
- **同文件关系**：调用基类 `BaseMemory.delete` 与 `self.document_store.get`；不调用本文件其他方法，也不被本文件其他方法调用。

### `clear(self) -> int` （第 255 行）
- **作用**：清空整个语义记忆层，并把图上的完整投影一起清掉。文档字符串写的是 *Clear semantic truth and its complete graph projection*，也就是「清掉语义真值以及它对应的整张图投影」。它先调用基类的 `clear` 把文档存储里的语义条目全部删除并拿到删除数量，然后探测图存储是否有 `clear` 方法，有就把整张图也清空。用于测试重置、用户主动清空记忆、或整体重建记忆库的场景。
- **参数**：无参数（除 `self` 外）。
- **返回**：`int`，即基类 `clear()` 返回的被删除条目数量。注意这个数字只反映文档存储里删了多少条，不包含图上被删除的节点/边数量。
- **内部流程**：第一步 `count = super().clear()` 清空基类管辖的文档存储并记录数量；第二步 `clear_graph = getattr(self.graph_store, "clear", None)` 取图存储的 `clear` 方法；第三步 `if callable(clear_graph): clear_graph()` 调用它清空整张图；第四步 `return count`。
- **异常/边界**：图存储没有 `clear` 方法时跳过，不报错，只清文档存储。图清空抛出的异常不被捕获，会向上冒泡，此时文档存储已经清空而图可能没清干净。本方法不接收参数，所以不存在参数非法的情况；返回值在没有任何条目时是 `0`。
- **同文件关系**：调用基类 `BaseMemory.clear` 与 `self.graph_store.clear`；不调用本文件其他方法，也不被本文件其他方法调用。

### `related(self, entity: str, *, relation: str | None = None, at: str | None = None) -> list[dict[str, Any]]` （第 264 行）
- **作用**：按实体查询图上的关联关系，是语义层对外的主要读接口之一。它不做任何本地计算，而是把查询直接下推给图存储的 `get_relations`，因为只有图数据库才知道实体之间有哪些边、边上有哪些属性。可选的 `relation` 用于把结果限制在某种关系类型上，可选的 `at` 用于做时间点过滤（读取某个时间有效的关系），这两个参数都由图存储解释。GraphRAG 检索、实体详情展示、以及问答链路里「这个实体和谁有什么关系」这类需求都会走它。
- **参数**：
  - `entity`：`str`，必填，要查询的实体名（图节点名）。
  - `relation`：`str | None`，默认 `None`，仅关键字参数。为 `None` 时不限制关系类型，返回该实体的全部关系；给了具体关系名则只返回该类型。
  - `at`：`str | None`，默认 `None`，仅关键字参数。时间点过滤条件（字符串形式），为 `None` 时不按时间过滤；具体格式与语义由图存储的 `get_relations` 决定。
- **返回**：`list[dict[str, Any]]`，图存储返回的关系记录列表，每条通常是以字典形式表达的一条边及其属性（具体键集合由 `get_relations` 决定）。实体没有任何关系时返回空列表。
- **内部流程**：唯一一步——`return self.graph_store.get_relations(entity, relation=relation, at=at)`，把两个可选参数都以关键字形式透传。
- **异常/边界**：本方法自身不做任何校验或兜底，空字符串 `entity`、非法的时间字符串都会原样传给图存储。图数据库不可用、查询语法错误、实体不存在等情况下的行为（返回空列表还是抛异常）完全取决于 `get_relations` 的实现，本方法不捕获异常。
- **同文件关系**：只调用 `self.graph_store.get_relations`；不调用本文件其他方法，也不被本文件其他方法调用。

### `facts(self, entity: str | None = None) -> list[MemoryItem]` （第 269 行）
- **作用**：从本地文档存储（而不是图数据库）列出语义事实条目，并支持按实体过滤。它的意义在于提供一条不依赖图数据库的读取路径：即使 Neo4j 不可用，也能从文档存储里把所有语义事实读出来；而按实体过滤时它看的是条目元数据里的 `subject` 与 `object` 字段，因此能精确回答「这条事实的主语或宾语是不是这个实体」。它常被用来做本地检索、事实列表展示、统计语义层规模，或在没有图查询能力的场景下做降级读取。
- **参数**：
  - `entity`：`str | None`，默认 `None`。为 `None` 时返回全部语义条目；给了实体名时只返回 `metadata["subject"]` 或 `metadata["object"]` 等于该实体名的条目。注意这里是精确相等比较，不做名称规范化（不会像 `_endpoint_attributes` 那样先转成 `entity_id_for`），也不做别名展开。
- **返回**：`list[MemoryItem]`。`entity is None` 时直接返回 `self.list()` 的结果；否则返回过滤后的新列表（原列表不被修改，过滤用列表推导生成）。没有任何匹配时返回空列表。
- **内部流程**：第一步 `items = self.list()` 从基类拿全部语义条目；第二步判断 `entity is None`，成立就直接返回 `items`；第三步否则执行列表推导，对每个 `item` 判断 `entity in (item.metadata.get("subject"), item.metadata.get("object"))`，用元组成员判断同时覆盖主语与宾语两种情况，命中则保留。
- **异常/边界**：不主动抛异常。若某个条目的 `metadata` 不是映射，`item.metadata.get` 会抛 `AttributeError`。`metadata` 里缺少 `subject`/`object` 键时 `.get` 返回 `None`，只是不匹配而不会报错。由于是精确匹配，传入实体别名或大小写不同的名称会查不到结果。返回的是新列表，调用方修改它不会影响内部存储，但列表里的 `MemoryItem` 仍是共享对象。
- **同文件关系**：调用基类 `BaseMemory.list`；不调用本文件其他方法，也不被本文件其他方法调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `SemanticMemory` | 语义记忆层的实现类，继承 `BaseMemory`，把事实三元组同时写入文档存储与图存储，对外提供写事实、查关系、删事实、清空的接口。 |
| `__init__` | 初始化语义记忆，调用基类构造并确保存在可用的图存储（未传时延迟导入并新建 `Neo4jGraphStore`）。 |
| `add_fact` | 写入或合并一条 `(subject, predicate, object)` 事实，做参数校验、id 规范化、legacy id 回退、元数据合并，并同步刷新图上的边。 |
| `_graph_properties` | 按固定白名单把条目 id 与元数据整理成图边属性字典，保证各写入路径写进 Neo4j 的键集合一致。 |
| `_endpoint_attributes` | 从文档存储里查实体的 `domain`/`aliases`/`importance`/`entity_type`，查不到就返回空字典让图节点保持默认值。 |
| `_write_edge` | 把一条事实完整投影到图里：写主关系边、写 `roles` 里的额外角色兼容边，并写一个带全部参与者的观察节点。 |
| `add_relation` | `add_fact` 的关系语汇别名，把 `(source, relation, target)` 原样转发给 `add_fact`。 |
| `delete` | 校验条目存在且属于语义层后删除该事实，并在图存储支持时同步删除对应的关系投影。 |
| `clear` | 清空语义层全部条目并返回删除数量，同时在图存储支持时清空整张图投影。 |
| `related` | 把按实体（可加关系类型与时间点过滤）的关联查询直接下推给图存储的 `get_relations`。 |
| `facts` | 从本地文档存储列出全部语义事实，或在给出实体名时按 `subject`/`object` 精确过滤。 |
