# tool/orphan_entities.py

## 一、这个文件是干什么的

这个文件实现的是知识星云（记忆系统）里的「孤儿实体检测」能力，也就是一次只读的「图体检」：它负责找出那些**完全孤立**的实体节点，也就是用户口中所说的「白建的实体」。文件顶部的模块 docstring 明确了「完全孤立」的判定口径，四条必须全部满足才算孤儿：第一，没有任何活跃的关系边（既不是任何事实的 `subject`，也不是 `object`）；第二，没有被任何原句（chunk）通过「提及」边引用（`source_ids` 里不含 chunk id）；第三，没有 `kind=note` 的备注挂靠在它名下；第四，不是 seed 播种出来的实体（避免把种子星图当噪音清掉）。

文件里包含的东西可以分成三层：第一层是纯逻辑层，两个模块级函数 `find_orphan_entities`（扫描并返回孤儿实体列表）和 `orphan_summary`（把一条实体压成适合喂给模型的小字典）；第二层是数据契约层，三个 Pydantic 模型 `OrphanEntitiesInput`、`OrphanEntity`、`OrphanEntitiesOutput`，分别描述工具入参、单条孤儿实体记录、工具出参；第三层是工具装配层，类 `OrphanEntitiesTool` 把上面两层接到项目的 `BaseTool` / `ToolSpec` 框架上，并额外提供 `create_tool()` 工厂函数供注册表调用。

它在运行时的定位是「只读能力」：`ToolSpec` 里 `side_effect="read"`、`permissions=()`、`idempotent=True`、`parallel_safe=True`，因此可以随时被调用、也可以并发调用，不会产生副作用。模块 docstring 特别强调，删除必须由用户明确指定目标后调用 `memory.manage`，并经过工具运行时的写确认；本工具不会创建另一套未接线的确认协议。另外 docstring 还说明本模块是这段逻辑的**唯一实现**，`web/cleanup.py` 里曾经的 `find_orphan_entities` 已经删除，星云图的 `orphan_entities` 统计会复用这里的实现（通过把已取回的 semantic 列表传给 `items` 参数，避免二次全表扫描）。

---

## 二、函数与类逐条详解

### `find_orphan_entities(manager: MemoryManager, *, items: list[MemoryItem] | None = None) -> list[MemoryItem]` （第 30 行）

- **作用**：这是整个文件的算法核心，也是「孤儿实体」判定口径的唯一落地点。它接收一个记忆管理器，遍历记忆库里的 semantic（语义）类型记忆项，先挑出所有 `kind == "entity"` 的实体，再反过来统计整个记忆库里哪些名字被事实引用过、哪些记忆项是 chunk、哪些实体被 note 挂靠过，最后用这三个「被使用过」的集合去过滤实体列表，剩下的就是完全孤立的实体。之所以需要它，是因为知识星云在长期运行中会累积大量「建了但没人用」的实体节点，用户需要一个只读的体检入口来量化这个问题。它被 `OrphanEntitiesTool.execute` 在每次工具调用时调用；同时因为它支持传入 `items` 复用调用方已经取回的 semantic 列表，星云图构建时的统计也能直接复用它，从而避免对整个记忆库做第二次全表扫描。它只读不写、不删除任何东西，所以可以安全地随时调用。
- **参数**：
  - `manager: MemoryManager`：位置参数，必填。记忆管理器实例，提供 `list(memory_type=...)` 接口用于拉取记忆项。当 `items` 为 `None` 时，本函数会通过它取数据，因此这种情况下它必须是一个可用的、能正常返回列表的 manager。
  - `items: list[MemoryItem] | None`：关键字专用参数（前面有 `*`，只能以关键字形式传入），可选，默认 `None`。含义是「调用方已经取回的 semantic 记忆项列表」。传 `None` 表示本函数自己去找 manager 要数据；传入一个列表则完全跳过取数步骤，直接在这个列表上做分析——这就是复用、避免二次全表扫描的机制。传入的列表按语义应当只包含 semantic 类型的记忆项，但函数本身不会对此再做校验。
- **返回**：返回 `list[MemoryItem]`，即「完全孤立的实体记忆项」列表。列表里元素的顺序等于输入 `items` 中实体出现的顺序（也就是记忆库顺序），不是按重要性或时间排序。如果没有任何实体满足条件，返回空列表 `[]`；函数不会返回 `None`，也不会对数量做任何截断（截断由上层 `execute` 依据 `limit` 完成）。
- **内部流程**：第一步，`items = manager.list(memory_type="semantic") if items is None else items`，即只有调用方没给数据时才自己去 manager 取 semantic 列表。第二步，用列表推导 `entities = [item for item in items if item.metadata.get("kind") == "entity"]` 筛出所有实体项，作为候选池。第三步，初始化三个空集合：`used_in_fact`（被事实引用过的名字）、`chunk_ids`（chunk 记忆项的 id）、`note_entities`（被备注挂靠的实体名）。第四步，对 `items`（注意是全部 semantic 项，不只是实体）做一次遍历，逐项读取 `item.metadata`：如果 `subject` 字段有值，就把 `str(metadata["subject"])` 加进 `used_in_fact`；如果 `object` 字段有值，同样加进 `used_in_fact`（这两个分支合起来就覆盖了「作为事实的主语或宾语」）；如果 `document_id` 不为 `None` 且 metadata 里存在键 `"chunk_index"`，就认为这一项是一条 chunk，把 `item.id` 加进 `chunk_ids`；如果 `kind == "note"` 且 `entity` 字段有值，就把 `str(metadata["entity"])` 加进 `note_entities`。第五步，初始化 `orphans` 空列表，对第四步筛出的每个实体逐项判定，任何一条命中「已使用」就 `continue` 跳过：先看 `metadata.get("seed")`，为真值则跳过（种子实体豁免，这是第四条判据）；再用 `name = str(metadata.get("canonical_name") or metadata.get("title") or item.content)` 计算实体的规范名（优先 `canonical_name`，其次 `title`，最后退化为正文内容）；若 `name in used_in_fact` 跳过（有事实边）；若 `name in note_entities` 跳过（有备注挂靠）；最后用 `source_ids = [str(value) for value in metadata.get("source_ids") or []]` 取出该实体的来源 id 列表并全部转字符串，若其中任意一个 `source_id in chunk_ids` 则跳过（被 chunk 提及过）。四道关卡全部通过的实体才被 `orphans.append(item)` 收进结果。第六步，`return orphans`。
- **异常/边界**：函数本身没有 try/except，也没有显式 raise。边界处理主要靠「防御式默认值」：`items` 为 `None` 时回落到 `manager.list(...)`；`metadata.get(...)` 全部使用带默认值的 `get`，所以 metadata 里缺少 `kind`、`subject`、`object`、`document_id`、`chunk_index`、`entity`、`seed`、`canonical_name`、`title`、`source_ids` 这些键都不会报 `KeyError`；`source_ids` 为 `None` 或缺失时用 `or []` 兜底成空列表，因此 `any(...)` 直接为假、不会崩；名字计算里用 `or` 链，`canonical_name` 与 `title` 都是假值（空串、`None`）时会退化到 `item.content`。潜在风险点在于：如果传入的 `items` 里混入了非 semantic 项，它们仍然会参与第二阶段的统计（这其实是设计意图，事实和 note 可能本来就不是实体类型）；如果 `manager.list` 本身抛异常（例如数据库不可用），异常会原样向上抛给调用方，本函数不做包装；如果 `items` 传的是非列表的可迭代对象，代码里的列表推导仍可工作，但函数内部多次遍历同一对象，若传入的是一次性迭代器（generator）会在第二次遍历时得到空结果——按类型标注它应当是 list。
- **同文件关系**：它被本文件的 `OrphanEntitiesTool.execute` 调用（`find_orphan_entities(self.manager)`）。它自身不调用本文件里的任何其他函数；它调用的 `orphan_summary` 并不在它内部，而是在上层 `execute` 里对它的返回值逐条使用。`OrphanEntitiesTool.manager` 属性负责为它准备 `manager` 参数。

---

### `orphan_summary(item: MemoryItem) -> dict[str, object]` （第 72 行）

- **作用**：这是一个「序列化/瘦身」辅助函数，作用是把一条孤儿实体记忆项转换成一个小而扁平的字典，只保留模型和用户真正关心的五个字段，避免把整条 `MemoryItem`（可能包含很长的 metadata、向量、时间戳等）原样塞进工具返回值里。它被设计成「model-friendly」，也就是字段名直白、类型简单（字符串、浮点数），方便工具输出模型的校验与前端展示。它的使用时机很明确：只在已经确认某条实体是孤儿之后才会被调用，因此它内部完全不需要再做任何孤儿判定，只负责取字段。它让 `find_orphan_entities` 的返回值与 `OrphanEntity` 这个 Pydantic 输出模型之间有了一个解耦的中间层：将来输出模型增删字段时，只需要改这一个函数。
- **参数**：
  - `item: MemoryItem`：位置参数，必填。一条记忆项，按调用约定它应当是一条 `kind == "entity"` 的语义记忆项（函数本身不校验这一点，传别的类型也能跑，只是语义上没有意义）。函数会读取它的 `metadata`、`id`、`content`、`importance` 四个属性。
- **返回**：返回 `dict[str, object]`，恰好包含五个键：`"id"`（记忆项 id，原样取 `item.id`）、`"name"`（实体名，计算规则与 `find_orphan_entities` 中一致：`canonical_name` → `title` → `item.content` 依次回退，并统一 `str()` 化）、`"domain"`（域名字符串，取 `metadata.get("domain")`，缺失或为假值时回落为空字符串 `""`）、`"content"`（实体正文，直接取 `item.content`）、`"importance"`（重要性，`round(float(item.importance), 3)`，即转成浮点数后保留三位小数）。返回的字典顺序就是上面这个书写顺序，可直接用于 `OrphanEntity(**dict)` 解包构造。
- **异常/边界**：没有 try/except，也不主动 raise。边界情况：`metadata` 缺少 `canonical_name`、`title`、`domain` 时由 `or` 链和 `""` 默认值兜底，不会 `KeyError`；`item.content` 为空串时 `name` 会退化成空字符串（不做额外报错）；如果 `item.importance` 是 `None` 或不可转浮点的值，`float(item.importance)` 会抛 `TypeError` / `ValueError`，本函数不捕获，会向调用方传播——按 `MemoryItem` 的约定 importance 应当是数值，所以正常路径不会触发；`round(..., 3)` 对已经是三位的值无影响，对超长小数做四舍五入式的截断。函数不会修改传入的 `item`，是纯读取操作。
- **同文件关系**：它被本文件的 `OrphanEntitiesTool.execute` 调用（`[OrphanEntity(**orphan_summary(item)) for item in orphans[: arguments.limit]]`），用于把 `find_orphan_entities` 返回的每条 `MemoryItem` 转成构造 `OrphanEntity` 所需的字典。它自身不调用本文件里的任何其他函数，也不被 `find_orphan_entities` 调用。

---

### `class OrphanEntitiesInput(BaseModel)` （第 85 行）

- **作用**：这是 `knowledge.orphan_entities` 工具的**入参契约**，用 Pydantic 模型描述「调用这个工具时允许传什么」。它的存在让工具运行时可以在真正执行前就对模型给出的参数做严格校验，从而把「模型乱传参数」的问题挡在业务逻辑之外。它同时承担了给 LLM 看的作用：`limit` 字段上的 `description` 会被工具框架提取成参数说明，明确告诉模型「count 始终是完整数量」，避免模型误以为 `limit` 会影响统计总数。这个类本身不含任何业务方法，只是一个声明式的数据容器。
- **参数**：这个类没有 `__init__` 参数需要手写（继承自 `BaseModel`，由 Pydantic 依据字段声明生成）。它的类级配置与字段如下：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：Pydantic v2 的模型配置。`extra="forbid"` 表示禁止传入任何未声明的字段，多传一个键就直接校验失败，防止模型夹带无效参数；`strict=True` 表示严格类型模式，不做「字符串自动转整数」这类宽松强制转换，类型必须对得上。
  - `limit: int = Field(default=50, ge=1, le=1000, description=...)`：唯一的字段，整型，默认值 50，取值下界 `ge=1`（至少返回 1 条）、上界 `le=1000`（最多返回 1000 条），越界会触发校验错误。描述文字是「最多返回多少个孤儿实体（count 始终是完整数量）。」，用于向模型解释语义：`limit` 只控制 `entities` 列表的截断长度，不改变 `count` 这个总数。
- **返回**：类本身不返回东西；它被实例化后产生一个 `OrphanEntitiesInput` 实例，实例上有且仅有 `limit` 一个属性（整型）。实例是工具 `execute` 方法的入参。
- **内部流程**：没有自定义方法，全部行为来自 Pydantic：定义时收集类级注解生成字段 schema；实例化时（`OrphanEntitiesInput(**kwargs)`）按 `model_config` 的严格模式和 `extra="forbid"` 校验输入字典；校验失败时由 Pydantic 抛出 `ValidationError`（由工具运行时捕获并反馈给模型）；校验通过则把 `limit` 存成实例属性并完成默认值填充。工具运行时通常还会用 `OrphanEntitiesInput.model_json_schema()` 之类的机制生成给模型看的参数 JSON Schema。
- **异常/边界**：本类没有自定义异常处理。边界行为由 Pydantic 决定：`limit` 缺失时取默认值 50；`limit` 小于 1 或大于 1000 抛 `ValidationError`；`limit` 传入字符串（如 `"10"`）在 `strict=True` 下同样抛 `ValidationError`，不会被悄悄转成整数；传入未声明字段（如 `"foo": 1`）因 `extra="forbid"` 抛 `ValidationError`。也就是说，所有非法输入都以校验异常的形式在进入 `execute` 之前被拒绝。
- **同文件关系**：它被本文件的 `OrphanEntitiesTool` 通过 `spec = ToolSpec(..., input_model=OrphanEntitiesInput, ...)` 注册为输入模型，并被 `OrphanEntitiesTool.execute(self, arguments: OrphanEntitiesInput)` 的类型标注所使用（`execute` 内部读取 `arguments.limit`）。它不调用本文件里的任何函数，也不被 `find_orphan_entities`、`orphan_summary`、`create_tool` 直接使用。

---

### `class OrphanEntity(BaseModel)` （第 96 行）

- **作用**：这是**单条孤儿实体记录**的输出模型，用来描述工具返回列表里的每一个元素。它把 `orphan_summary` 产出的字典固化成有类型、有字段名的结构，让工具输出既可以被 Pydantic 校验，又能被序列化成稳定的 JSON 给模型和前端消费。它刻意保持字段精简（id、name、domain、content、importance），因为孤儿实体可能一次返回很多条，字段越少越省 token、越容易读。这个类同样不含任何业务方法，只是声明式数据容器；字段默认值的存在使得即使某些 metadata 缺失，记录也能被构造出来。
- **参数**：没有手写 `__init__`，实例化参数就是下面这些字段（Pydantic 由字段声明生成构造签名），按声明顺序为：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：与入参模型一致的严格配置——禁止多余字段、开启严格类型模式。
  - `id: str`：必填，记忆项的唯一标识。
  - `name: str`：必填，实体展示名（由 `canonical_name` / `title` / `content` 回退计算而来）。
  - `domain: str = ""`：可选，实体所属领域，默认空字符串。
  - `content: str = ""`：可选，实体正文/摘要，默认空字符串。
  - `importance: float = 0.5`：可选，重要性分值，默认 0.5（注意这里声明为 `float` 且开了 `strict=True`，因此传入 `int` 是否被接受取决于 Pydantic 严格模式下对 int→float 的处理规则；`orphan_summary` 已经先 `float(...)` 转换，正常路径传入的一定是浮点数）。
- **返回**：类本身不返回东西；实例化后得到一条 `OrphanEntity` 记录，带有上述五个属性，可被 `model_dump()` 序列化成字典、被 `model_dump_json()` 序列化成 JSON。`OrphanEntitiesOutput.entities` 就是由这些实例组成的列表。
- **内部流程**：无自定义方法，行为全部由 Pydantic 提供：类定义时收集字段注解与默认值生成 schema；实例化时执行校验、填充默认值（`domain`、`content`、`importance` 未给就用默认值）；校验失败抛 `ValidationError`。在本文件中它的典型用法是 `OrphanEntity(**orphan_summary(item))`，即把 `orphan_summary` 返回的五个键直接解包成关键字参数。
- **异常/边界**：无自定义异常处理。边界：缺少必填字段 `id` 或 `name` 抛 `ValidationError`；多传字段因 `extra="forbid"` 抛 `ValidationError`；`domain`、`content`、`importance` 缺失时分别取 `""`、`""`、`0.5`。由于 `orphan_summary` 总是显式产出全部五个键，正常流程下不会走到默认值分支，默认值只是为了让这个模型在别的调用场景下也能单独使用。
- **同文件关系**：它被 `OrphanEntitiesOutput` 用作 `entities: list[OrphanEntity]` 的元素类型，并被 `OrphanEntitiesTool.execute` 用来构造每条输出记录（`OrphanEntity(**orphan_summary(item))`），因此它间接依赖 `orphan_summary` 的返回键集合与自己的字段名完全对齐。它不调用本文件里的任何函数。

---

### `class OrphanEntitiesOutput(BaseModel)` （第 106 行）

- **作用**：这是 `knowledge.orphan_entities` 工具的**出参契约**，定义工具成功执行后返回给模型的结构：一个总数 `count` 加一个最多 `limit` 条的实体明细列表 `entities`。把「总数」和「明细」分开是有意的设计：`count` 告诉模型图里到底有多少完全孤立的实体（完整数量，不受截断影响），`entities` 只给出前若干条作为样本，这样即使孤儿成千上万，返回值也不会爆炸。它同样是声明式数据容器，不含业务方法。
- **参数**：没有手写 `__init__`，实例化参数为两个字段：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：严格模式、禁止多余字段，与前两个模型保持一致。
  - `count: int = Field(description="完全孤立实体的总数。")`：必填整型，语义是全部孤儿实体的完整数量（注意它带 `description` 但没有默认值，构造时必须显式传入）。
  - `entities: list[OrphanEntity] = Field(default_factory=list, description="前 limit 个（按记忆库顺序）。")`：可选列表，元素类型是 `OrphanEntity`，默认由 `default_factory=list` 生成空列表（用工厂而不是直接 `= []` 是为了避免可变默认值共享问题）。描述说明它只包含前 `limit` 个，且顺序遵循记忆库顺序。
- **返回**：类本身不返回东西；实例化后得到工具的输出对象，具有 `count` 与 `entities` 两个属性，可被工具运行时序列化返回给调用方/模型。
- **内部流程**：无自定义方法。Pydantic 在类定义时依据字段声明构建 schema；在 `OrphanEntitiesTool.execute` 中以 `OrphanEntitiesOutput(count=len(orphans), entities=[...])` 的形式实例化；实例化时校验 `count` 必须是整型、`entities` 必须是 `OrphanEntity` 列表（元素会逐个校验），并通过 `default_factory` 在未提供 `entities` 时补空列表。
- **异常/边界**：无自定义异常处理。边界：未传 `count` 抛 `ValidationError`；`entities` 未传时得到空列表（这正好对应「没有孤儿」的情形）；`entities` 里若混入非 `OrphanEntity` 的字典，Pydantic 会尝试按模型校验转换，失败则抛 `ValidationError`；多余字段因 `extra="forbid"` 被拒绝。本文件不在此模型上做任何数量上限校验，`limit` 的约束在 `OrphanEntitiesInput` 上。
- **同文件关系**：它被 `OrphanEntitiesTool` 通过 `spec = ToolSpec(..., output_model=OrphanEntitiesOutput, ...)` 注册为输出模型，并被 `OrphanEntitiesTool.execute` 的返回类型标注和实际返回语句使用。它依赖同文件的 `OrphanEntity` 作为列表元素类型。它不调用本文件里的任何函数。

---

### `class OrphanEntitiesTool(BaseTool)` （第 113 行）

- **作用**：这是把「孤儿实体检测」这套逻辑接入项目工具生态的适配器类。它继承项目的 `BaseTool`，通过类级属性 `spec`（一个 `ToolSpec`）向工具注册表声明自己的身份：工具名 `knowledge.orphan_entities`、英文描述、版本 `1.0.0`、输入/输出模型、副作用等级、权限、超时、幂等性、并发安全性、标签以及给模型的 `guidance` 提示。它的职责是「声明元数据 + 持有 manager + 实现 execute 三步」，本身不包含任何孤儿判定算法（算法在 `find_orphan_entities` 里）。当模型或上层系统需要做知识库健康检查时，会通过工具运行时按 `spec.name` 找到并调用它；由于 `side_effect="read"`、`permissions=()`、`idempotent=True`、`parallel_safe=True`，它可以在只读场景下自由、并发地被调用，不需要写确认。类里还通过 `guidance` 明确约束模型行为：先向用户展示候选，只有用户点名确认后才能调用 `memory.manage`，不要自行删除。
- **参数**：类级属性 `spec` 是 `ToolSpec` 实例，其字段取值为：`name="knowledge.orphan_entities"`；`description` 为一段英文说明，强调「找出完全孤立的实体节点：无关系边、无 chunk 提及、无备注挂靠、且非种子；只读健康检查，删除仍是独立的、需运行时确认的 `memory.manage` 动作」；`version="1.0.0"`；`input_model=OrphanEntitiesInput`；`output_model=OrphanEntitiesOutput`；`side_effect="read"`（只读，不产生写副作用）；`permissions=()`（空元组，不需要任何额外权限）；`timeout_seconds=60.0`（单次执行最长 60 秒）；`idempotent=True`（重复调用结果一致）；`parallel_safe=True`（可安全并发）；`tags=("knowledge", "graph", "orphan", "health", "read")`（用于分类检索）；`guidance` 为一段中文提示，说明四项判定口径（种子实体不会被误判成垃圾），并规定清理流程必须先展示候选、经用户点名确认后才能调用 `memory.manage`。
  该类没有手写 `__init__` 参数以外的构造参数，唯一构造参数见下方 `__init__` 条目。
- **返回**：类本身不是函数，不返回东西；它的实例是注册表里的一个工具对象，被调用时由 `execute` 返回 `OrphanEntitiesOutput`。
- **内部流程**：类体的执行顺序是：先定义 `spec` 类级属性（模块导入时即构造 `ToolSpec`，其中引用同文件的三个 Pydantic 模型），再定义 `__init__` 保存 manager，再定义 `manager` 属性做惰性构建，最后定义 `execute` 完成「扫描 → 截断 → 包装」的调用链。整个类不包含循环或分支逻辑（除 `manager` 属性里的惰性构建分支外），业务复杂度都被下推到模块级函数里。
- **异常/边界**：类定义阶段无异常处理；`spec` 构造失败（例如字段类型不匹配）会在模块导入时直接暴露。运行期异常主要由 `execute` 与 `manager` 属性产生：manager 无法构建、`manager.list` 失败、`orphan_summary` 里 `float(importance)` 失败等都会向上传播；本类不做捕获。超时由工具运行时依据 `timeout_seconds=60.0` 控制，本类自身不实现计时或中断逻辑。
- **同文件关系**：它通过 `spec` 引用同文件的 `OrphanEntitiesInput`、`OrphanEntitiesOutput`；`execute` 调用同文件的 `find_orphan_entities` 和 `orphan_summary`，并用 `OrphanEntity` 构造明细；`manager` 属性在自身未注入 manager 时从 `. _memory` 导入 `build_default_manager`（这是本文件唯一一处对外部模块的函数级延迟导入，目的是避免导入期循环依赖与不必要的初始化开销）。`create_tool()` 负责创建它的实例。

---

### `OrphanEntitiesTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 136 行）

- **作用**：构造函数，只做一件事：把外部传入的记忆管理器存到实例的私有属性 `self._manager` 上。它之所以把参数设计成可选的 `None`，是为了支持两种使用场景——测试或星云图统计等场景可以注入一个现成的、可能已经被复用或替换过的 manager；而普通运行时（例如 `create_tool()` 直接 `OrphanEntitiesTool()`）不传任何东西，让 `manager` 属性在第一次真正需要时再去惰性构建默认管理器。这样构造工具对象本身非常轻量，不会在注册阶段就触发记忆库的初始化。它不做任何校验、不打印日志、不建立连接。
- **参数**：
  - `self`：实例自身，由 Python 自动传入。
  - `manager: MemoryManager | None`：可选，默认 `None`。传入一个 `MemoryManager` 实例时会被原样保存，后续 `execute` 直接使用它；传入 `None`（或不传）时 `self._manager` 保持 `None`，把构建推迟到 `manager` 属性被访问时。
- **返回**：返回 `None`。构造函数不返回实例。
- **内部流程**：只有一条赋值语句 `self._manager = manager`，没有分支、循环、校验或延迟导入。真正的惰性初始化逻辑不在这里，而在紧随其后的 `manager` 属性里。
- **异常/边界**：无特殊处理。传 `None` 是合法的正常用法（表示「稍后惰性构建」）；传入类型不对的对象不会被这里拒绝，错误会在 `execute` 调用 `manager.list(...)` 时才暴露为 `AttributeError` 之类。
- **同文件关系**：它设置的状态被同类的 `manager` 属性读取（`if self._manager is None` 分支），并最终被 `execute` 通过 `self.manager` 使用。它不调用本文件里的任何函数；它被 `create_tool()` 间接触发（`create_tool()` 返回 `OrphanEntitiesTool()`，即以默认 `None` 调用它）。

---

### `OrphanEntitiesTool.manager` （property，第 139 行）

- **作用**：这是一个只读属性，充当记忆管理器的**惰性单例访问点**。任何需要 manager 的代码都通过 `self.manager` 拿，而不是直接碰 `self._manager`，这样就能保证：第一次访问时若构造时没有注入 manager，就自动去构建一个默认管理器并缓存下来；之后所有访问都直接返回同一个缓存对象，不会重复构建。这种「属性即依赖入口」的写法让工具在注册阶段零成本，同时又在运行阶段自动获得可用的依赖，避免了在每个调用点写 `if self._manager is None` 的重复判断。它也顺带让测试可以在构造时注入替身，覆盖默认构建路径。
- **参数**：无参数（属性访问形式 `tool.manager`，`self` 由 Python 自动传入）。
- **返回**：返回 `MemoryManager`。如果构造时注入了 manager，就返回那个实例；否则第一次访问时构建默认管理器、存入 `self._manager` 并返回。返回值保证非 `None`（除非 `build_default_manager()` 本身返回 `None`，按约定它应当返回一个 manager）。
- **内部流程**：第一步判断 `if self._manager is None:`；为真时执行函数内延迟导入 `from ._memory import build_default_manager`（放在函数体内是为了避免模块导入期就拉起记忆子系统，规避潜在的循环导入与初始化副作用），然后 `self._manager = build_default_manager()` 完成构建与缓存。第二步 `return self._manager`，无论走的是缓存命中还是刚构建完的分支，都返回同一个实例。整个过程没有加锁，属于「惰性初始化」而非严格线程安全单例。
- **异常/边界**：无 try/except。若 `._memory` 模块不存在或 `build_default_manager` 导入失败，会抛 `ImportError`；若默认构建过程本身失败（例如配置缺失、存储不可用），异常会从 `build_default_manager()` 原样传播，且此时 `self._manager` 仍为 `None`，下次访问会再试一次（不会缓存失败状态）。并发首次访问理论上可能构建出多个 manager 实例（无锁），但由于每次都会把结果写回 `self._manager`，后续访问收敛到最后一个。
- **同文件关系**：它读取 `__init__` 写入的 `self._manager`，被同类的 `execute` 通过 `self.manager` 调用。它对外部模块的依赖只有延迟导入的 `._memory.build_default_manager`；它不调用本文件里的其他函数。

---

### `OrphanEntitiesTool.execute(self, arguments: OrphanEntitiesInput) -> OrphanEntitiesOutput` （第 147 行）

- **作用**：这是工具的运行时入口，也是唯一把「算法」和「数据契约」缝在一起的地方。它接收已经过 Pydantic 校验的入参，调用 `find_orphan_entities(self.manager)` 拿到全部孤儿实体，然后用 `count=len(orphans)` 记录**完整数量**，同时按 `arguments.limit` 对列表做切片，把前 `limit` 条通过 `orphan_summary` 转成字典再构造成 `OrphanEntity`，最后打包成 `OrphanEntitiesOutput` 返回。它存在的意义是：让底层判定函数保持纯粹（只返回列表、不关心分页和序列化），而把「截断」与「模型友好化」这两个属于工具层的关注点集中在这一个方法里。工具运行时每次调用 `knowledge.orphan_entities` 时都会走到它。
- **参数**：
  - `self`：实例自身，自动传入；方法内部通过 `self.manager` 取记忆管理器。
  - `arguments: OrphanEntitiesInput`：必填，已经校验过的入参对象。方法只使用其中的 `arguments.limit`（整型，范围 1–1000，默认 50），它决定 `entities` 列表最多返回多少条。
- **返回**：返回 `OrphanEntitiesOutput` 实例，其中 `count` 是完全孤立实体的总数（整型，等于 `find_orphan_entities` 返回列表的长度，**不受** `limit` 影响），`entities` 是由 `OrphanEntity` 组成的列表，长度为 `min(count, arguments.limit)`，顺序与记忆库顺序一致。
- **内部流程**：第一步，`orphans = find_orphan_entities(self.manager)`——注意这里没有传 `items`，所以 `find_orphan_entities` 会自己调用 `manager.list(memory_type="semantic")` 做一次全量拉取；`self.manager` 的访问可能触发 `manager` 属性里的惰性构建。第二步，构造输出：`count=len(orphans)`；`entities=[OrphanEntity(**orphan_summary(item)) for item in orphans[: arguments.limit]]`，即先切片截断（`orphans[:limit]`，切片天然容忍 `limit` 大于列表长度），再对每条调用 `orphan_summary` 得到五键字典，用 `**` 解包成 `OrphanEntity` 的关键字参数完成校验与构造。第三步，返回该 `OrphanEntitiesOutput`。方法内没有循环之外的额外分支，也没有日志、缓存或去重。
- **异常/边界**：无 try/except，也不主动 raise。边界情况：`orphans` 为空列表时 `count=0`、`entities=[]`，返回正常；`arguments.limit` 大于实际孤儿数时切片返回全部，不报错；`limit` 已由 `OrphanEntitiesInput` 约束在 1–1000，因此这里不需要再校验，若绕过校验传入非法值（如 0 或负数），切片会得到空列表而 `count` 仍是完整数量，不会抛异常。可能向上传播的异常包括：`self.manager` 惰性构建失败（`ImportError` 或构建过程自身的异常）、`manager.list` 失败（存储层异常）、以及 `orphan_summary` 中 `float(item.importance)` 失败（`TypeError` / `ValueError`）。超时不由本方法控制，而由 `spec.timeout_seconds=60.0` 在工具运行时层面生效。
- **同文件关系**：它调用同文件的 `find_orphan_entities`（获取孤儿列表）、`orphan_summary`（把每条实体压成字典）和 `OrphanEntity`（构造输出元素），并通过 `self.manager` 间接依赖同类的 `manager` 属性与 `__init__`。它被工具运行时按 `spec.name` 调用，也被同文件的 `create_tool()` 间接触发（创建实例后由框架调用 execute）。它不调用 `OrphanEntitiesInput`、`OrphanEntitiesOutput` 的方法（仅实例化后者）。

---

### `create_tool() -> BaseTool` （第 155 行）

- **作用**：这是模块级的工厂函数，是工具注册体系约定的入口。它不接受任何参数，直接返回一个用默认配置构造的 `OrphanEntitiesTool` 实例，也就是 manager 为 `None`、依赖在首次访问时才惰性构建的那种「轻量实例」。它的存在是为了让注册表能够用统一的方式发现并实例化本模块提供的工具（按名字找到 `create_tool` 并调用即可），而不必知道 `OrphanEntitiesTool` 的构造细节，也不必在注册阶段就注入依赖。它被 `__all__` 导出，属于本模块的公开 API 之一。
- **参数**：无。
- **返回**：返回 `BaseTool` 类型（实际运行时的具体类型是 `OrphanEntitiesTool`）。返回的对象尚未持有 manager，`_manager` 为 `None`，要等到第一次调用 `execute`（进而访问 `manager` 属性）时才会构建默认管理器。
- **内部流程**：只有一行 `return OrphanEntitiesTool()`。没有分支、没有缓存、没有单例逻辑——每次调用都会创建一个新的工具实例（实例之间不共享 manager，各自惰性构建）。它不做任何参数读取、环境检查或异常包装。
- **异常/边界**：无特殊处理。构造函数本身几乎不会失败（只做一次属性赋值），因此正常路径下不会抛异常；如果类定义/导入阶段就有问题，那在导入模块时就已经暴露，与本函数无关。重复调用会产生多个独立实例，属于预期行为。
- **同文件关系**：它调用同文件的 `OrphanEntitiesTool` 构造函数（以默认 `manager=None`），因此间接依赖该类的 `__init__`、`manager` 属性与 `execute`。它自身不被本文件里的其他函数调用；它被外部工具注册机制调用，并被本文件末尾的 `__all__` 列为公开导出名。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `find_orphan_entities` | 遍历 semantic 记忆，按「无关系边、无 chunk 提及、无备注、非种子」四条判据筛出完全孤立的实体列表，可复用调用方传入的 items 避免二次全表扫描。 |
| `orphan_summary` | 把一条孤儿实体记忆项压成含 id、name、domain、content、importance 五个键的模型友好字典。 |
| `OrphanEntitiesInput` | 工具入参契约：严格模式、禁止多余字段，只有 1–1000、默认 50 的 `limit` 一个字段。 |
| `OrphanEntity` | 单条孤儿实体的输出模型：id、name 必填，domain、content、importance 带默认值。 |
| `OrphanEntitiesOutput` | 工具出参契约：完整总数 `count` 加最多 `limit` 条 `entities` 明细。 |
| `OrphanEntitiesTool` | 把孤儿检测逻辑接入工具框架的适配器类，用 `ToolSpec` 声明只读、无权限、幂等、可并发的元数据与给模型的 guidance。 |
| `OrphanEntitiesTool.__init__` | 构造函数，仅把可选的 manager 存入 `self._manager`，把依赖构建推迟到真正需要时。 |
| `OrphanEntitiesTool.manager` | 只读属性，惰性构建并缓存默认 `MemoryManager`，作为工具内部统一的依赖访问点。 |
| `OrphanEntitiesTool.execute` | 工具运行入口：扫描全部孤儿、用 `count` 记录总数、按 `limit` 切片并转成 `OrphanEntity` 输出。 |
| `create_tool` | 模块级工厂，无参返回一个默认构造（manager 为 None、惰性构建）的 `OrphanEntitiesTool` 实例供注册表使用。 |
