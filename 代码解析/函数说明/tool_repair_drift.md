# tool/repair_drift.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时「四层记忆系统」里的**漂移自愈工具**，模块级唯一职责就是：当 `knowledge.reconcile` 报告三库（真值源 / 向量库 / 图库）之间出现投影漂移时，执行**幂等**修复，把缺失的投影补回来。

它刻意只做「补」这一个方向的动作：`missing_vector`（真值源里标了 `indexed`、但向量库里查不到的分块）会被重新嵌入并 upsert 回向量库，同时把分块状态置回 `indexed`、把仍是 `parsed` 的文档推进到 `vectorized`；`missing_edge`（语义层有 fact、图里却没有对应关系）会用同一个 `item_id` 重放 `semantic.add_fact`，依赖图存储按 id 幂等。它明确**不**处理 `orphan_vector`（孤儿向量），因为删除不可逆、必须由人决定，所以遇到这个类型直接抛 `ValueError` 拒绝，而不是悄悄扩大副作用面。

文件内部结构分三层：一个纯函数 `repair_drift` 承载全部修复逻辑（这是整段逻辑的唯一实现，`web/app.py` 里原先内联的版本已删除，`POST /api/reconcile` 改为调用这里并把 `ValueError` 映射成 422）；三个 Pydantic 模型 `RepairDriftInput` / `RepairedCounts` / `RepairDriftOutput` 定义工具的入参与出参契约；一个 `BaseTool` 子类 `RepairDriftTool` 把函数包装成运行时可注册、可被 Agent 调用的标准工具，并配上 `ToolSpec` 元数据（写副作用、300 秒超时、幂等、非并行安全）。

外部通过 `create_tool()` 工厂函数拿到工具实例，或者直接调用模块级 `repair_drift()`；`__all__` 显式声明了对外暴露的 6 个名字。模块级常量 `TOOL_ENABLED = True` 是工具发现机制读取的开关标记。

## 二、函数与类逐条详解

### `repair_drift(manager: MemoryManager, repository: DocumentRepository | None, kinds: list[str]) -> dict[str, Any]` （第 37 行）

- **作用**：这是整个模块的核心实现，负责在已知漂移报告的前提下，把 `missing_vector` 和 `missing_edge` 两类缺失投影幂等地补回来。它不自己判断漂移，而是先调用 `reconcile_report` 重新取一份当前漂移快照，再按调用方传入的 `kinds` 白名单筛选出真正要修的部分，避免做多余写操作。之所以需要它，是因为向量库或图库可能因为进程崩溃、嵌入服务短暂不可用、或手工改库而与真值源失去同步，需要一个「只补不删」的安全修复入口。它被 `RepairDriftTool.execute` 调用（Web 的 `POST /api/reconcile` 最终走到那里），也可以被脚本或定时任务直接调用。函数严格遵守「绝不删除、绝不改写真值源内容」的边界，返回值只报告补了多少条，不报告删改。
- **参数**：
  - `manager`：`MemoryManager` 实例，必填。它提供 `manager.embedding`（批量嵌入）、`manager.vector_store`（向量 upsert）、`manager.semantic`（事实写入）三个能力，是修复动作的实际执行者。
  - `repository`：`DocumentRepository | None`，必填但允许传 `None`。它是 SQLite 文档库仓储，用来按 `chunk_id` 取分块、改分块向量状态、读文档、改文档状态。传 `None` 表示当前是内存模式或非 SQLite 文档库，此时真值源里不存在「已标记投影」的分块，`missing_vector` 恒为 0。
  - `kinds`：`list[str]`，必填但允许传 `None`（内部按空列表处理）。表示本次要修复的漂移类型白名单，合法元素只有 `REPAIRABLE_KINDS` 里的项（即 `missing_vector` 与 `missing_edge`）；传入其它字符串会直接抛 `ValueError`。重复元素会被去重，空列表表示「什么都不修」，直接返回全 0 结果。
- **返回**：返回 `dict[str, Any]`，结构固定为 `{"repaired": {"missing_vector": int, "missing_edge": int}}`。其中 `missing_vector` 是本次真正重新嵌入并 upsert 成功的分块数（等于被找到的有效分块条数，不是报告里的 id 总数）；`missing_edge` 是本次真正重放成功的事实条数。两个键在未请求或未命中时都为 0，绝不会缺失键。
- **内部流程**：第一步做参数规范化与校验——`requested = list(dict.fromkeys(kinds or []))` 用字典键去重且保持原有顺序，同时把 `None` 归一成空列表；接着遍历 `requested`，把不在 `REPAIRABLE_KINDS` 里的项收集到 `unknown`，只要非空就立刻 `raise ValueError("不支持的修复类型：...")`，这样保证校验发生在任何写操作之前，不会产生「修了一半才发现参数错」的半成品状态。第二步取漂移快照——调用 `reconcile_report(manager, repository)`，取其 `["drift"]` 列表，用字典推导压成 `{entry["kind"]: entry["ids"]}` 形式的 `entries`，便于按类型 O(1) 取 id 列表。第三步初始化计数器 `repaired = {"missing_vector": 0, "missing_edge": 0}`。第四步处理向量缺失：仅当 `"missing_vector"` 在 `requested` 中、`entries.get("missing_vector")` 非空、且 `repository is not None` 三个条件同时成立时进入；先用生成器表达式 `repository.get_chunk(chunk_id) for chunk_id in ids` 逐个取分块，过滤掉返回 `None` 的（分块可能已被删除），得到 `chunks`；若 `chunks` 非空，调用 `manager.embedding.embed_batch([chunk.text for chunk in chunks])` 一次性批量嵌入（比逐条嵌入省调用开销），再用 `zip(chunks, vectors, strict=True)` 严格配对；循环体内对每个分块调用 `manager.vector_store.upsert_chunk(...)`，写入 `chunk.chunk_id`、向量、`document_id`、`chunk_index`，并把 `source` 传空串、`memory_type` 传 `MemoryType.SEMANTIC.value`（与知识层写入约定一致），随后调用 `repository.set_chunk_vector_status(chunk.chunk_id, "indexed")` 把真值源里的分块状态改回已索引；循环结束后把 `repaired["missing_vector"]` 设为 `len(chunks)`，再用集合推导 `{chunk.document_id for chunk in chunks}` 对涉及的文档去重，逐个 `get_document`，若文档存在且 `document.status == "parsed"` 就 `set_status(document_id, "vectorized")`，把文档整体推进到下一阶段。第五步处理图边缺失：仅当 `"missing_edge"` 在 `requested` 中且 `entries.get("missing_edge")` 非空时进入；先把 id 列表转成 `wanted` 集合，然后遍历 `fact_items(manager)` 拿到的全部事实项，`item.id not in wanted` 的直接 `continue` 跳过，命中的调用 `manager.semantic.add_fact(subject, predicate, object, metadata=item.metadata, confidence=float(item.importance), item_id=item.id)`，其中主谓宾从 `item.metadata` 的 `subject` / `predicate` / `object` 三个键取字符串，置信度取 `item.importance` 并转 `float`，最关键的是把原来的 `item.id` 作为 `item_id` 传回去，使图存储能按 id 幂等去重，重放不会产生重复边；每成功一条 `repaired["missing_edge"] += 1`。第六步返回 `{"repaired": repaired}`。
- **异常/边界**：`kinds` 含未知类型时抛 `ValueError`，这是调用方（Web 层）映射成 HTTP 422 的依据；`zip(..., strict=True)` 在嵌入服务返回的向量条数与分块条数不一致时会抛 `ValueError`，属于「宁可失败也不写错位向量」的保护；`repository.get_chunk` 返回 `None`（分块已被删）时被过滤掉，不会计入修复数，也不会写空向量；`repository` 为 `None` 时整个 `missing_vector` 分支被短路，计数保持 0，与报告里 `chunks_indexed_sqlite` 为 0 一致，不存在「假装修好了」；`kinds` 传 `None` 或空列表时不做任何事直接返回全 0；`fact_items` 返回为空时循环体一次不执行，`missing_edge` 保持 0；`entries` 中某类型缺失用 `.get()` 取，不会 `KeyError`；`orphan_vector` 之类不在白名单的类型会在第一步就被 `ValueError` 拒绝，函数不会去删任何向量。
- **同文件关系**：它调用了本文件外的 `reconcile_report`、`fact_items`、`REPAIRABLE_KINDS`（来自 `.reconcile`）、`MemoryType`、`MemoryManager`、`DocumentRepository`。在本文件内部，它被 `RepairDriftTool.execute` 调用；它不调用本文件里的其它函数或类（三个 Pydantic 模型与工具类都是它外层的包装）。

### `class RepairDriftInput(BaseModel)` （第 102 行）

- **作用**：这是 `knowledge.repair_drift` 工具的输入契约模型，用来把 Agent 或 HTTP 调用方传来的 JSON 参数校验并归一成结构化对象。它存在的意义是把「要修复哪些漂移类型」这一唯一入参用 Pydantic 声明清楚，让运行时在真正执行修复前就能拦下多余字段、类型不对、条数过多的请求，避免脏参数进入写路径。它被 `RepairDriftTool.spec` 的 `input_model` 字段引用，由工具框架在调用 `execute` 之前完成实例化与校验。它本身不含任何业务逻辑，只是一个纯数据容器。字段只有 `repair` 一个，语义上对应 `repair_drift` 函数的 `kinds` 参数。
- **参数**：本类没有自定义 `__init__`，由 Pydantic 依据字段声明生成构造逻辑，唯一字段为 `repair: list[str]`，默认值由 `default_factory=list` 提供（即不传时为 `[]`），约束 `max_length=2`（最多两个元素，正好对应两个可修类型），描述为「要修复的漂移类型，可选项只有 missing_vector 与 missing_edge」。
- **返回**：类本身不返回值；实例化后得到一个携带 `repair` 列表的模型对象，供 `execute` 读取 `arguments.repair`。校验失败时 Pydantic 会抛 `ValidationError` 而不是返回错误值。
- **内部流程**：类体先设置 `model_config = ConfigDict(extra="forbid", strict=True)`，`extra="forbid"` 表示出现未声明字段（例如误写 `types`）直接报错，`strict=True` 表示不做宽松类型强转（传字符串给 `list[str]` 不会被偷偷拆成字符列表）；然后用 `Field(...)` 声明 `repair` 字段，把默认值、长度上限和人类可读描述绑定上去。没有校验器、没有计算属性，实例化即完成。
- **异常/边界**：字段缺失时使用默认空列表，不会报错；传入非列表（如字符串、数字）或列表元素非字符串时抛 `ValidationError`（strict 模式下不做隐式转换）；元素个数超过 2 抛 `ValidationError`；传入未声明字段抛 `ValidationError`。注意模型层**不校验**元素取值是否在 `REPAIRABLE_KINDS` 内，这一步留给 `repair_drift` 抛 `ValueError`。
- **同文件关系**：被 `RepairDriftTool.spec`（`input_model=RepairDriftInput`）引用，并被 `RepairDriftTool.execute` 的类型注解使用；它自己不调用本文件任何函数。名字出现在 `__all__` 中。

### `class RepairedCounts(BaseModel)` （第 112 行）

- **作用**：这是修复结果的计数模型，用来把 `repair_drift` 返回的字典里的两个数字变成有类型、有字段说明的结构化对象，保证工具输出契约稳定、可被框架序列化成 JSON Schema。它存在的原因是工具的输出必须是可校验的模型，而不是随意形状的 `dict`，这样上层（Agent 或 Web 响应）拿到的 `repaired` 字段一定有 `missing_vector` 和 `missing_edge` 两个整数字段。它被 `RepairDriftOutput` 组合引用，也被 `RepairDriftTool.execute` 在构造返回值时使用。类内只有两个必填整数字段，没有业务方法。
- **参数**：无自定义构造参数；Pydantic 依据字段声明生成 `__init__`，需要（且只接受）两个关键字参数 `missing_vector: int` 与 `missing_edge: int`，二者都没有默认值，因此都是必填。
- **返回**：实例化后得到一个携带两项计数的对象；校验失败时抛 `ValidationError`。
- **内部流程**：设置 `model_config = ConfigDict(extra="forbid", strict=True)`，禁止多余字段、禁止宽松强转；随后声明两个 `Field(description=...)` 字段——`missing_vector` 描述为「本次补回的向量条数」，`missing_edge` 描述为「本次补回的图关系条数」。没有校验器与派生逻辑，构造即完成。
- **异常/边界**：缺少任一字段抛 `ValidationError`；传入 `float` 或数字字符串在 `strict=True` 下不会被接受为 `int`，抛 `ValidationError`；传入额外字段抛 `ValidationError`。对负数没有做 `ge=0` 约束，即从模型层面看负数也是合法值，但上游 `repair_drift` 只会产出非负计数。
- **同文件关系**：被 `RepairDriftOutput` 作为字段类型引用，并在 `RepairDriftTool.execute` 中通过 `RepairedCounts(**result["repaired"])` 构造；它自己不调用本文件任何函数。名字出现在 `__all__` 中。

### `class RepairDriftOutput(BaseModel)` （第 119 行）

- **作用**：这是 `knowledge.repair_drift` 工具的输出契约模型，把修复结果包成一个顶层对象，使工具的返回 JSON 形状固定为 `{"repaired": {"missing_vector": N, "missing_edge": M}}`。它的存在是为了让工具输出有明确的 Schema（`RepairDriftTool.spec` 的 `output_model` 指向它），运行时可以据此校验、展示和序列化结果，也让 `execute` 的返回类型注解具备实际约束力。它被 `execute` 用于构造最终返回值，被工具框架用于描述输出结构。类内只组合了一个字段，没有业务逻辑。
- **参数**：无自定义构造参数；Pydantic 生成的构造需要一个关键字参数 `repaired: RepairedCounts`（或可被校验成该模型的数据），必填。
- **返回**：实例化后得到一个顶层输出对象，其 `repaired` 属性是 `RepairedCounts` 实例；校验失败时抛 `ValidationError`。
- **内部流程**：设置 `model_config = ConfigDict(extra="forbid", strict=True)`，然后声明唯一字段 `repaired: RepairedCounts`。由于 `RepairDriftTool.execute` 传入的已经是 `RepairedCounts` 实例，构造过程中只会做类型确认，不会重复解析。没有校验器与派生逻辑。
- **异常/边界**：`repaired` 缺失抛 `ValidationError`；传入无法被校验成 `RepairedCounts` 的值（例如裸字典缺少字段，或字符串）抛 `ValidationError`；传入额外顶层字段抛 `ValidationError`。
- **同文件关系**：被 `RepairDriftTool.spec`（`output_model=RepairDriftOutput`）引用，并在 `RepairDriftTool.execute` 中作为返回类型与构造目标；内部依赖同文件的 `RepairedCounts`。名字出现在 `__all__` 中。

### `class RepairDriftTool(BaseTool)` （第 125 行）

- **作用**：这是把 `repair_drift` 纯函数包装成运行时标准工具的适配类，继承 `core.BaseTool`，让 Agent 能通过工具调用机制（名称 `knowledge.repair_drift`）触发漂移修复。它承担三件事：声明工具元数据（名称、描述、版本、入参模型、出参模型、副作用类型、权限、超时、幂等性、并行安全性、标签、给模型看的使用指引），延迟构建并缓存 `MemoryManager`，以及在 `execute` 里把「取 manager → 取文档仓储 → 调修复函数 → 转成输出模型」这条链路串起来。它被 `create_tool()` 工厂函数实例化，也会被工具注册/发现机制按 `TOOL_ENABLED` 标记加载。类设计上刻意把 manager 做成可选注入，方便测试或复用已有运行时上下文。
- **参数**：类的构造参数见下面的 `__init__` 条目；类级属性 `spec` 是一个 `ToolSpec` 实例，其关键字段取值为：`name="knowledge.repair_drift"`、`description` 为一段英文说明（幂等修复存储漂移、只增不删、拒绝 `orphan_vector` 因为删除必须由人决定）、`version="1.0.0"`、`input_model=RepairDriftInput`、`output_model=RepairDriftOutput`、`side_effect="write"`（标明是写操作）、`permissions=()`（不额外要求权限）、`timeout_seconds=300.0`（5 分钟上限）、`idempotent=True`、`parallel_safe=False`（同一存储上并发修复不安全）、`tags=("knowledge", "reconcile", "repair", "storage", "write")`、`guidance` 为中文使用建议（只在 reconcile 报告漂移且用户同意修复时调用，只补投影绝不删改真值源，`orphan_vector` 会被拒绝，修完再跑一次 reconcile 验证归零）。
- **返回**：类本身不返回值；实例化得到可注册的工具对象。其 `execute` 方法返回 `RepairDriftOutput`。
- **内部流程**：类体只做两件事——定义类级 `spec`（在类创建时立即构造 `ToolSpec`，把工具的全部元信息固化下来，供框架内省），以及定义实例方法 `__init__`、属性 `manager`、方法 `execute`。方法级流程见下面三条。
- **异常/边界**：类定义阶段若 `ToolSpec` 字段名或取值不被基类接受，会在导入模块时直接报错；`spec` 中声明 `permissions=()` 表示不需要额外权限校验；`parallel_safe=False` 意味着框架应避免并发调度本工具。类本身不捕获异常，`execute` 中 `repair_drift` 抛出的 `ValueError` 会向上传播，由 Web 层映射为 422。
- **同文件关系**：它引用同文件的 `RepairDriftInput`、`RepairDriftOutput`、`RepairedCounts`、`repair_drift`；被同文件的 `create_tool` 实例化。名字出现在 `__all__` 中。

### `RepairDriftTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 150 行）

- **作用**：构造工具实例，并把可选的 `MemoryManager` 依赖记录下来。之所以把 manager 设计成可选，是为了让工具既能由运行时注入已存在的 manager（避免重复构建、保证与其他工具共享同一套存储连接），也能在没有任何上下文时延迟自建（见 `manager` 属性）。它只在对象创建时执行一次，不做任何 I/O、不连接数据库、不校验 manager 是否可用。它被 `create_tool()` 以无参形式调用，也可能被测试或上层以显式 manager 调用。
- **参数**：`self` 为实例本身；`manager`：`MemoryManager | None`，默认 `None`，表示要绑定的记忆管理器，传 `None` 时留待后续惰性构建。
- **返回**：无返回值（`None`），只产生副作用——把参数存入实例私有属性 `self._manager`。
- **内部流程**：单步操作 `self._manager = manager`，把参数原样保存，不做类型检查、不做包装。没有其它初始化逻辑。
- **异常/边界**：无特殊处理。传入任何对象（包括类型错误的值）都不会在这里报错，问题会推迟到 `execute` 真正使用 manager 时暴露。
- **同文件关系**：它保存的值被同文件的 `manager` 属性读取、被 `execute` 间接使用；它被同文件的 `create_tool` 调用（无参形式）。

### `RepairDriftTool.manager` （property，第 153 行） -> `MemoryManager`

- **作用**：这是一个只读属性，用惰性方式提供 `MemoryManager` 实例。第一次访问时，如果构造时没注入 manager（`self._manager is None`），它就调用 `build_default_manager()` 现场构建一个并缓存到 `self._manager`；之后再访问直接返回缓存对象，保证同一工具实例在整个生命周期里复用同一个 manager，不会每次执行都新建一套存储连接。它被 `execute` 的第一行使用。属性形式（而不是普通方法）让 `execute` 里写 `self.manager` 即可，语义更清晰。
- **参数**：只有 `self`，无其它参数。
- **返回**：返回 `MemoryManager` 实例——要么是构造时注入的那个，要么是首次访问时由 `build_default_manager()` 新建并缓存的那个。
- **内部流程**：先判断 `if self._manager is None:`；条件成立时执行**函数内局部导入** `from ._memory import build_default_manager`（放在函数体内而不是模块顶部，通常是为了避免模块导入期的循环依赖或推迟重依赖的加载开销），然后 `self._manager = build_default_manager()` 赋值缓存；最后 `return self._manager`。
- **异常/边界**：无特殊处理。若 `build_default_manager()` 内部因配置缺失或存储不可用而抛异常，该异常会原样向上传播，且此时 `self._manager` 仍为 `None`，下次访问会再次尝试构建（不会缓存失败状态）。并发访问同一实例时没有加锁，理论上可能重复构建，但工具已声明 `parallel_safe=False`。
- **同文件关系**：它读取 `__init__` 保存的 `self._manager`，并调用本文件之外的 `build_default_manager`（来自 `._memory`）；它被同文件的 `execute` 调用。

### `RepairDriftTool.execute(self, arguments: RepairDriftInput) -> RepairDriftOutput` （第 161 行）

- **作用**：这是工具的执行入口，框架校验完入参后调用它来完成一次漂移修复。它负责把工具层的输入模型翻译成核心函数需要的三个参数：从 `self.manager` 拿记忆管理器，用 `repository_for(manager)` 推导出对应的文档仓储（可能是 `None`），再把 `arguments.repair` 当作要修的类型白名单传进去；拿到字典结果后，把 `result["repaired"]` 展开构造成 `RepairedCounts`，再包成 `RepairDriftOutput` 返回。它存在的意义是让纯函数 `repair_drift` 与运行时工具协议解耦：函数只管逻辑，工具类只管协议适配。它由工具调度框架在 Agent 请求 `knowledge.repair_drift` 时调用，也可由 Web 的 `POST /api/reconcile` 路径间接触发。
- **参数**：`self` 为工具实例；`arguments`：`RepairDriftInput` 实例，必填，其 `repair` 字段是要修复的漂移类型列表（默认空列表，最多 2 项）。
- **返回**：返回 `RepairDriftOutput` 实例，其 `repaired` 属性是 `RepairedCounts`，含 `missing_vector` 与 `missing_edge` 两个整数计数。
- **内部流程**：第一行 `manager = self.manager`，触发惰性构建（若尚未构建）；第二行 `result = repair_drift(manager, repository_for(manager), arguments.repair)`，注意 `repository_for(manager)` 的返回值直接作为 `repository` 实参传入，可能为 `None`，正好对应核心函数文档里说的内存/非 SQLite 模式；第三行 `return RepairDriftOutput(repaired=RepairedCounts(**result["repaired"]))`，用字典解包把 `{"missing_vector": ..., "missing_edge": ...}` 变成 `RepairedCounts` 的关键字参数，完成从裸字典到强类型输出的转换。
- **异常/边界**：`arguments.repair` 含未知类型时，`repair_drift` 抛出的 `ValueError` 不被捕获，直接向上传播给框架/Web 层（映射为 422）；`repository_for` 或 `build_default_manager` 的异常同样透传；`result["repaired"]` 键缺失理论上不会发生（核心函数恒定返回两个键），若真缺失会抛 `KeyError`；嵌入条数与分块数不一致时 `zip(strict=True)` 抛的 `ValueError` 也向上传播。无超时自处理逻辑——超时由 `spec.timeout_seconds=300.0` 在框架层控制。
- **同文件关系**：它调用同文件的 `repair_drift`（核心函数）、`RepairedCounts` 与 `RepairDriftOutput`（输出模型），并通过 `self.manager` 间接使用同文件 `__init__` 保存的依赖；它调用本文件之外的 `repository_for`。它被工具框架调用，不被本文件内其它函数调用。

### `create_tool() -> BaseTool` （第 167 行）

- **作用**：这是工具工厂函数，是外部（工具注册表、插件加载器或测试代码）获取 `RepairDriftTool` 实例的标准入口。之所以用工厂函数而不是直接暴露类，是为了与项目里其它工具的发现约定保持一致——加载器通常只要求模块提供 `create_tool()` 这个零参可调用对象，从而不必关心具体类名和构造签名。它每次调用都返回一个**新的**工具实例，实例内部没有注入 manager，因此每个实例第一次执行时会各自惰性构建 manager（调用方若想共享 manager，应自行构造 `RepairDriftTool(manager=...)`）。它没有任何参数与配置读取逻辑。
- **参数**：无参数。
- **返回**：返回 `BaseTool` 类型的对象，实际类型是 `RepairDriftTool`，且其内部 `_manager` 初始为 `None`。
- **内部流程**：单步 `return RepairDriftTool()`——无参构造，触发 `__init__` 把 `_manager` 置为 `None`，不做任何 I/O 或配置解析。构造完成后工具即可被注册，`spec` 元数据随类定义已就绪。
- **异常/边界**：无特殊处理。只要模块导入成功、`ToolSpec` 构造无异常，本函数不会失败；它不校验环境配置，配置问题会推迟到首次执行访问 `manager` 时暴露。
- **同文件关系**：它实例化同文件的 `RepairDriftTool`；不被本文件内其它函数调用。名字出现在 `__all__` 中。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `repair_drift` | 核心修复函数：按白名单校验类型后重新嵌入缺失分块、重放缺失图边，幂等地只补投影不改真值源。 |
| `RepairDriftInput` | 工具入参模型，只含一个最多两项的 `repair` 类型列表，禁止多余字段与宽松类型转换。 |
| `RepairedCounts` | 修复计数模型，必填 `missing_vector` 与 `missing_edge` 两个整数字段。 |
| `RepairDriftOutput` | 工具出参模型，把 `repaired` 计数对象包成固定的顶层返回结构。 |
| `RepairDriftTool` | 把核心函数包装成运行时标准工具，声明 `knowledge.repair_drift` 的元数据、超时、幂等与写副作用等规格。 |
| `RepairDriftTool.__init__` | 记录可选的 `MemoryManager` 依赖到 `self._manager`，不注入时留待惰性构建。 |
| `RepairDriftTool.manager` | 只读属性：首次访问时惰性构建并缓存 `MemoryManager`，之后复用同一实例。 |
| `RepairDriftTool.execute` | 执行入口：取 manager 与文档仓储、调用 `repair_drift`、把结果字典转成 `RepairDriftOutput`。 |
| `create_tool` | 零参工厂函数，返回一个未注入 manager 的新 `RepairDriftTool` 实例供工具发现机制使用。 |
