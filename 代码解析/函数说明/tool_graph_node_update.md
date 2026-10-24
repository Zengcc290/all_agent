# tool/graph_node_update.py

## 一、这个文件是干什么的

本文件是知识图谱「实体节点更新」这一能力的唯一实现处，把原先散落在 `EntityResolver.resolve` / `_store_alias` 里的**属性合并逻辑**收敛成一个可被外部单独调用的模块。文件对外暴露三层东西：两个普通函数（`push_graph_entity`、`update_entity_node`）负责真正的「合并 + 写回真值源 + 镜像到图投影」，`find_entity_node` 负责按 id 优先、规范化名兜底地定位一个节点；两个 Pydantic 模型（`GraphNodeUpdateInput` / `GraphNodeUpdateOutput`）定义工具调用的入参与出参契约；一个 `BaseTool` 子类（`GraphNodeUpdateTool`）把上面这些组装成一个名字为 `knowledge.graph_node_update` 的 Agent 工具，最后由 `create_tool()` 作为工厂函数交给工具注册表。

在运行期（`python -m web.app` 起服务后），Agent 在需要「修正或丰富一个已存在的实体节点」时会选中这个工具：`execute` 先 `find_entity_node` 找出匹配节点，再 `update_entity_node` 把新值合并进去，`manager.semantic.add` 写入真值源（`memories` 里的实体行），随后 `push_graph_entity` 尝试把同样的属性刷到图投影上。合并策略有两条硬规则：**别名只增不减**（取并集），**重要度只升不降**（`max(旧值, 新值)`），目的是避免一次低置信抽取把已经积累好的节点信息打回去。图投影侧只 `MATCH` 已存在的节点，不会凭空造节点，因此「更新节点」这个动作永远不会改变图的节点集合。模块顶部还有 `TOOL_ENABLED = True` 这个开关，供工具装载层判断是否注册本工具；底部 `__all__` 明确列出了对外可用符号。

## 二、函数与类逐条详解

### `_default_manager() -> MemoryManager` （第 45 行）
- **作用**：构造并返回一个「共享的、落盘的」记忆管理器实例。之所以需要它，是因为 `GraphNodeUpdateTool` 允许在不传 `manager` 的情况下使用，这时必须有人去创建默认管理器；把它单独抽成一个私有函数，是为了让「惰性导入」这件事只发生在一处。函数体里刻意把 `from ._memory import build_default_manager` 放在函数内部而不是模块顶层，注释写明了原因：`_memory` 会连带拉起 `memory.rag`，如果在模块导入期就引入，会造成较重的导入链甚至潜在循环导入。因此只有在真正需要默认管理器的时刻（第一次访问 `manager` 属性）才会付这份导入成本。
- **参数**：无参数。
- **返回**：`MemoryManager`，即 `_memory.build_default_manager()` 的返回值，具体是磁盘上共享的那一份管理器实例（由 `build_default_manager` 决定是新建还是复用）。
- **内部流程**：唯一一步就是函数内 `from ._memory import build_default_manager`，紧接着 `return build_default_manager()`，不做任何缓存、不加锁、不做异常包装。
- **异常/边界**：本身不抛异常，但把 `_memory` 导入失败（`ImportError`）和 `build_default_manager()` 内部可能抛出的任何异常原样向上传播；没有空值处理逻辑，因为无输入。
- **同文件关系**：被 `GraphNodeUpdateTool.manager` 属性（第 263 行）在 `self._manager is None` 时调用；它自身不调用本文件里的任何其他函数。

### `push_graph_entity(manager, name, *, domain, aliases, importance) -> bool` （第 53 行）
- **作用**：把实体的属性镜像到「图投影」这一侧，也就是让图存储里的节点展示属性与真值源保持一致。它存在的原因是：真值源（`manager.semantic.add` 写入的实体行）和图投影是两套存储，属性更新必须两边都刷，否则 Agent 从图里读到的 domain / 别名 / 重要度会过期。这个函数被设计成「尽力而为」的：如果图存储根本没提供 `update_entity`，或者图里并没有这个节点，它就安静地返回 `False`，而不是报错，因为「更新节点」不应该因为投影缺失而整体失败。注释明确它「只 MATCH 已存在的节点」，所以造节点的职责仍然归 `add_relation` / `add_observation`。
- **参数**：
  - `manager: MemoryManager`：记忆管理器，函数从中取 `graph_store`。位置参数，无默认值。
  - `name: str`：要更新的图节点名（调用方传的是 canonical 名）。位置参数，无默认值。
  - `domain: str`：关键字参数（`*` 之后强制关键字传参），新的领域字符串。
  - `aliases: Iterable[str]`：关键字参数，别名集合，任意可迭代对象（列表、集合、生成器等都可）；函数内部会逐个 `str()` 转换。
  - `importance: float`：关键字参数，重要度数值；函数内部会 `float()` 转换。
- **返回**：`bool`。成功调用底层 `update_entity` 时返回其结果的布尔化值（`bool(...)`）；若 `manager.graph_store` 为 `None` 或没有可调用的 `update_entity`，返回 `False`。因此 `False` 的语义是「没写进图投影」，可能是「不支持」也可能是「图里没有这个节点」，二者在本函数层面不区分。
- **内部流程**：先用 `getattr(getattr(manager, "graph_store", None), "update_entity", None)` 做两层防御性取值，得到 `updater`；然后 `if not callable(updater): return False`；最后调用 `updater(name, domain=domain, aliases=[str(value) for value in aliases], importance=float(importance))`，并把返回值用 `bool()` 包一层返回。注意 `aliases` 会被物化成一个新列表，所以传入生成器也是安全的。
- **异常/边界**：`manager` 上没有 `graph_store` 属性、或 `graph_store` 为 `None`、或它没有 `update_entity` 方法时，返回 `False` 而非抛异常。`aliases` 为 `None` 时会抛 `TypeError`（不可迭代），空的可迭代对象则得到空列表并照常传递。底层 `update_entity` 自身抛出的异常不被捕获，会向上传播。无超时控制。
- **同文件关系**：被 `update_entity_node`（第 146 行）调用；它自身不调用本文件里的任何其他函数（只用 `getattr` / `bool` / `str` / `float` 等内建）。

### `update_entity_node(manager, name, *, existing=None, domain="", entity_type=ENTITY_DEFAULT_TYPE, description="", aliases=None, importance=ENTITY_DEFAULT_CONFIDENCE, source_id=None, add_written_name=False, item_id=None, create_if_missing=True) -> MemoryItem` （第 76 行）
- **作用**：本模块的核心函数，负责把调用方给出的新属性「合并」到一个实体节点上，并落盘持久化。它把三件事串在一起：以已有实体记录为基准做属性合并（空值表示不改、别名取并集、重要度取 `max`）、通过 `manager.semantic.add` 写回真值源、再通过 `push_graph_entity` 镜像到图投影。它是唯一实现属性更新的地方，`EntityResolver` 只保留匹配职责，需要更新属性时回头调用它。当 `existing` 为 `None` 时它既可以在 `create_if_missing=True` 下新建一个实体节点（用 `entity_id_for(canonical)` 生成 id），也可以在 `create_if_missing=False` 下直接抛 `LookupError`，从而让上层能明确区分「改了一个节点」和「本来不存在」。
- **参数**：
  - `manager: MemoryManager`：记忆管理器，用于 `manager.semantic.add` 以及经 `push_graph_entity` 访问 `graph_store`。位置参数。
  - `name: str`：调用方写的实体名，会被 `_clean_text(name, max_length=ENTITY_NAME_MAX_LENGTH)` 清洗与截断；清洗后为空则报错。位置参数。
  - `existing: MemoryItem | None = None`：已经匹配到的实体条目；`None` 表示「没有匹配」。它决定了 canonical 名、既有别名、既有来源、既有重要度和目标 id 的来源。
  - `domain: str = ""`：新领域。空串（falsy）表示不修改，沿用 `metadata["domain"]`，再退到 `DEFAULT_DOMAIN`。
  - `entity_type: str = ENTITY_DEFAULT_TYPE`：新实体类型。falsy 时沿用旧值，再退到 `ENTITY_DEFAULT_TYPE`。
  - `description: str = ""`：新描述。falsy 时沿用旧值，旧值缺失时为空串。注意无法用它把已有描述清空。
  - `aliases: Iterable[str] | None = None`：要并入的别名，任意可迭代对象；`None` 表示不修改。并入前会做 `strip()`、剔除空串、剔除与 canonical 相同的值。
  - `importance: float = ENTITY_DEFAULT_CONFIDENCE`：新重要度；最终取它与旧值的较大者，因此只升不降。
  - `source_id: str | None = None`：要追加到 `source_ids` 的来源 id；falsy 时不追加。
  - `add_written_name: bool = False`：为 `True` 时把清洗后的 `name` 本身也记为一个别名（前提是它不等于 canonical）。文档字符串说明：精确命中查询时希望这样做，而「形近前缀命中」时绝不能这样做。
  - `item_id: str | None = None`：显式指定目标 id；为 `None` 时优先用 `existing.id`，再退到 `entity_id_for(canonical)`。
  - `create_if_missing: bool = True`：`existing` 为 `None` 且该值为 `False` 时抛 `LookupError`。
- **返回**：`MemoryItem`，即 `manager.semantic.add(...)` 返回的条目对象（写入后带 id、content=canonical、metadata、importance 的实体条目）。
- **内部流程**：① `_clean_text` 清洗并限长，空则 `raise ValueError`；② 若 `existing is None and not create_if_missing` 则 `raise LookupError`；③ 用 `dict(existing.metadata)` 复制出 `metadata`（`existing` 为 `None` 时是空字典），这样不会就地改动旧条目的 metadata；④ 计算 `canonical`：优先 `metadata["canonical_name"]`，其次 `existing.content`，最后退回清洗后的 `cleaned`；⑤ 从 `metadata["aliases"]` 收集 `known_aliases`（跳过 `strip()` 后为空的值）；⑥ 若 `add_written_name` 且 `cleaned != canonical`，把 `cleaned` 加入别名；⑦ 若传了 `aliases`，把每个 `strip()` 后非空且不等于 `canonical` 的值并入 `known_aliases`（用集合推导去重）；⑧ 从 `metadata["source_ids"]` 收集 `source_ids`，若给了 `source_id` 则加入；⑨ `metadata.update({...})` 一次性写回 `kind="entity"`、`title`/`canonical_name`=canonical、`entity_type`、`description`、`domain`、排序后的 `aliases`、排序后的 `source_ids`（其中 `entity_type` / `description` / `domain` 都用 `or` 链实现「新值优先、旧值兜底、常量兜底」）；⑩ 计算 `target_id`；⑪ 计算 `resolved_importance = max(float(importance), float(existing.importance) if existing else 0.0)`；⑫ 调 `manager.semantic.add(canonical, metadata=metadata, importance=resolved_importance, item_id=target_id)` 得到 `item`；⑬ 调 `push_graph_entity(manager, canonical, domain=metadata["domain"], aliases=metadata["aliases"], importance=resolved_importance)` 镜像到图；⑭ 返回 `item`。
- **异常/边界**：名称为空（清洗后为空串）抛 `ValueError("entity name must not be empty")`；`existing is None` 且 `create_if_missing=False` 抛 `LookupError(f"entity node not found: {cleaned}")`；`existing.importance` 不是可 `float()` 的值会抛 `TypeError`/`ValueError`；`aliases` 传了不可迭代对象会抛 `TypeError`；`metadata` 中 `aliases` / `source_ids` 若不是可迭代对象同样会抛异常。空值语义方面，`domain` / `entity_type` / `description` 传空串都表示「不修改」，因此没有清空字段的途径；`aliases=None` 表示不动别名，空列表则等于不动（因为没有可并入项）。没有超时与重试逻辑。
- **同文件关系**：调用本文件的 `push_graph_entity`（镜像图投影）以及从 `memory.ids` 导入的 `_clean_text`、`entity_id_for`；被 `GraphNodeUpdateTool.execute`（第 269 行）调用。它不调用 `find_entity_node`（匹配由调用方先做好再以 `existing` 传入）。

### `find_entity_node(manager, name) -> MemoryItem | None` （第 156 行）
- **作用**：在语义记忆里定位「一个」实体节点，是 `update_entity_node` 的前置匹配步骤。它采用两级策略：先按 `entity_id_for(cleaned)` 直接查文档存储（O(1) 的 id 直查，命中即可返回），不中才退化为遍历 `manager.semantic.list()` 按规范化名比对。之所以要第二级，是因为调用方给的名字可能是别名或大小写/标点不同的写法，而 id 直查只能命中由该名字确定性生成的 id。返回 `None` 表示「没有匹配」，上层据此决定是新建还是报错（见 `create_if_missing`）。
- **参数**：
  - `manager: MemoryManager`：记忆管理器，提供 `document_store.get` 与 `semantic.list`。位置参数。
  - `name: str`：待定位的实体名，可为 canonical 名或已知别名；会先经 `_clean_text` 清洗并限长到 `ENTITY_NAME_MAX_LENGTH`。
- **返回**：`MemoryItem | None`。id 直查命中且 `memory_type == MemoryType.SEMANTIC` 且 `metadata["kind"] == "entity"` 时返回该条目；否则遍历所有实体条目，canonical 名规范化后相等、或任一别名规范化后相等时返回该条目；都不匹配返回 `None`。清洗后为空也返回 `None`。
- **内部流程**：① `cleaned = _clean_text(name, max_length=ENTITY_NAME_MAX_LENGTH)`，若为假值立即 `return None`；② `direct = manager.document_store.get(entity_id_for(cleaned))`；③ 三重条件校验 `direct is not None and direct.memory_type == MemoryType.SEMANTIC and direct.metadata.get("kind") == "entity"` 通过则直接返回 `direct`（这一步把「不是语义记忆」「不是实体」的条目挡掉）；④ `key = normalize_entity_name(cleaned)`；⑤ `for item in manager.semantic.list():` 中先 `if item.metadata.get("kind") != "entity": continue` 跳过非实体；⑥ 取 `canonical = str(item.metadata.get("canonical_name") or item.content)`，若 `normalize_entity_name(canonical) == key` 则返回；⑦ 否则用 `any(...)` 在 `item.metadata.get("aliases") or []` 上逐个 `normalize_entity_name(str(alias)) == key` 判断，命中即返回该条目；⑧ 循环结束仍无命中则 `return None`。
- **异常/边界**：名称为空或清洗后为空时返回 `None`，不抛异常；`manager.document_store.get` 或 `semantic.list()` 抛出的异常不被捕获；`metadata` 里 `aliases` 为 `None` 时靠 `or []` 兜底为空列表，但如果它是非可迭代的非空值（例如数字）则会在迭代时抛 `TypeError`。没有「多个匹配」的处理逻辑，遍历中第一个命中的条目即被返回，因此结果依赖 `semantic.list()` 的顺序。无超时控制。
- **同文件关系**：不调用本文件里的任何其他函数（只用从 `memory.ids` 导入的 `_clean_text`、`entity_id_for`、`normalize_entity_name` 和 `MemoryType`）；被 `GraphNodeUpdateTool.execute`（第 268 行）调用。

### `class GraphNodeUpdateInput(BaseModel)` （第 184 行）
- **作用**：定义工具 `knowledge.graph_node_update` 的入参契约。它继承 Pydantic 的 `BaseModel`，用 `model_config = ConfigDict(extra="forbid", strict=True)` 明确两件事：多传未知字段直接报错（`extra="forbid"`，防止 Agent 幻觉出多余参数），类型必须严格匹配（`strict=True`，例如不会把字符串 `"0.5"` 悄悄转成浮点）。每个字段的 `description` 同时充当给模型看的参数说明，字段级 `min_length` / `max_length` / `ge` / `le` 约束把非法取值挡在 `execute` 之前。注意它只描述「想改什么」，不包含节点定位结果；真正的匹配结果由 `execute` 里调 `find_entity_node` 得到。
- **参数**：无（类的构造由 Pydantic 生成的 `__init__` 承担，字段如下）。
- **返回**：不适用（构造出的实例即 `GraphNodeUpdateInput`）。
- **字段清单**：
  - `name: str`：必填，`min_length=1`、`max_length=ENTITY_NAME_MAX_LENGTH`，说明为「要更新的实体名（canonical 名或已知别名都可定位到同一节点）」。
  - `domain: str | None = None`：`max_length=100`，null 或空串表示不修改。
  - `entity_type: str | None = None`：`max_length=80`，null 或空串表示不修改。
  - `description: str | None = None`：`max_length=1000`，null 或空串表示不修改。
  - `aliases: list[str] | None = None`：`max_length=20`（在 Pydantic 中作用于列表长度，即最多 20 个别名），取并集、不删除已有别名，null 表示不修改。
  - `importance: float | None = None`：`ge=0`、`le=1`，最终取 `max(旧值, 新值)`，null 表示不修改。
  - `create_if_missing: bool = True`：找不到节点时是否新建，`false` 时返回错误。
- **内部流程**：本类没有自定义方法，全部行为来自 Pydantic 的模型构建：解析输入 → 校验类型与约束 → 填充默认值 → 生成实例；校验失败抛 `pydantic.ValidationError`，由 `BaseTool` 调用层负责转换成工具错误。
- **异常/边界**：缺 `name`、`name` 为空串或超长、`aliases` 超过 20 项、`importance` 越界、传入未声明字段、类型不符，都会在实例化阶段抛 `ValidationError`，不会进入 `execute`。空串语义方面，`domain`/`entity_type`/`description` 传空串与传 null 等价（都是「不修改」）。
- **同文件关系**：被 `GraphNodeUpdateTool.spec` 通过 `input_model=GraphNodeUpdateInput`（第 243 行）引用，并被 `GraphNodeUpdateTool.execute` 的参数类型标注（第 266 行）使用；不调用本文件任何函数。

### `class GraphNodeUpdateOutput(BaseModel)` （第 218 行）
- **作用**：定义工具调用的返回契约，让 Agent 能直接看到「改完之后节点长什么样」以及「这次到底是新建还是更新」「图投影有没有同步上」。同样使用 `ConfigDict(extra="forbid", strict=True)`，保证输出结构固定、不会夹带意外字段。其中 `graph_projected` 是一个信息量较大的字段：它告诉调用方这次属性只写进了真值源、还是连图投影里已存在的节点也一并刷新了，从而让 Agent 知道图查询侧是否已经能看到新属性。
- **参数**：无（字段如下）。
- **返回**：不适用。
- **字段清单**：
  - `name: str`：实际写入的 canonical 实体名。
  - `node_id: str`：写入条目的 id（来自 `MemoryItem.id`）。
  - `created: bool`：本次是否属于新建（由 `execute` 中 `existing is None` 得出）。
  - `domain: str`：更新后生效的领域。
  - `entity_type: str`：更新后生效的实体类型。
  - `aliases: list[str]`：更新后的完整别名列表（已排序，来自 metadata）。
  - `importance: float`：更新后生效的重要度。
  - `graph_projected: bool`：图投影里是否已存在该节点并被同步刷新（不存在则只写真值源）。
- **内部流程**：无自定义方法，构造时由 Pydantic 完成类型校验与序列化准备；`strict=True` 意味着 `execute` 里必须传入精确类型，所以那里对每个字段都做了 `str(...)` / `float(...)` / `bool(...)` 显式转换。
- **异常/边界**：若 `execute` 传入的字段类型与标注不符或缺失，构造时抛 `ValidationError`；本类自身不做任何空值兜底，兜底责任在 `execute`（例如 `domain` 用 `or ""`）。
- **同文件关系**：被 `GraphNodeUpdateTool.spec` 通过 `output_model=GraphNodeUpdateOutput`（第 244 行）引用，并在 `GraphNodeUpdateTool.execute`（第 287 行）中被实例化返回；不调用本文件任何函数。

### `class GraphNodeUpdateTool(BaseTool)` （第 233 行）
- **作用**：把上述纯函数与两个数据模型包装成一个可被 Agent 调用的工具。它继承 `core.BaseTool`，通过类属性 `spec`（一个 `ToolSpec`）声明工具元数据：名字 `knowledge.graph_node_update`、英文描述（说明「更新一个已存在的知识图谱实体节点的 domain / 实体类型 / 描述 / 别名 / 重要度；别名只合并不删除、重要度只增；想创建新事实请用入库工具」）、版本 `1.0.0`、输入输出模型、`side_effect="write"`（声明这是写操作）、`permissions=()`（不需要额外权限）、`timeout_seconds=30.0`、`idempotent=True`（同样入参重复调用结果一致）、`parallel_safe=False`（不可与其他操作并行，因为会写共享存储）、`tags=("graph", "entity", "node", "update", "write")`，以及中文 `guidance`：只用于修正或丰富**已存在**的实体节点，绝不用来创建新实体或新关系；别名只增不减、重要度只升不降，所以不要指望用空值清掉已有字段；`create_if_missing` 默认 false、找不到节点会明确报错而不是新建。这里有一处代码与文案不一致值得留意：`guidance` 里写「`create_if_missing` 默认 false」，而 `GraphNodeUpdateInput.create_if_missing` 的实际默认值是 `True`（第 213 行），因此真实行为是「默认会新建」，`guidance` 的这句话与代码不符，实际以字段默认值为准。
- **参数**：无（类本身）；构造由下面的 `__init__` 负责。
- **返回**：不适用。
- **内部流程**：类体内只有类属性 `spec`（在类定义时即构造好 `ToolSpec`）与三个方法定义，没有类级别的副作用逻辑。
- **异常/边界**：`ToolSpec` 在类定义时构造，若参数非法会在导入模块时就失败；类本身不处理异常。
- **同文件关系**：`spec` 引用 `GraphNodeUpdateInput` / `GraphNodeUpdateOutput`；其方法调用本文件的 `_default_manager`、`find_entity_node`、`update_entity_node`；被 `create_tool`（第 300 行）实例化。

### `GraphNodeUpdateTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 257 行）
- **作用**：构造函数，只做一件事——把外部注入的管理器保存到实例上，供后续 `manager` 属性惰性取用。之所以允许注入，是为了在测试或上层已持有共享管理器的场景下复用同一个 `MemoryManager`（避免重复打开落盘存储）；不传则走默认惰性创建路径。这里刻意不在构造时就创建默认管理器，避免「只是注册了工具」就把整个记忆栈拉起来。
- **参数**：
  - `self`：实例本身。
  - `manager: MemoryManager | None = None`：可选的管理器；`None` 表示稍后由 `manager` 属性惰性构建。
- **返回**：`None`。
- **内部流程**：唯一语句 `self._manager = manager`，不做校验、不做类型检查、不触发任何 IO。
- **异常/边界**：无特殊处理；传入任何对象都会被原样保存，类型错误会推迟到实际使用时才暴露。
- **同文件关系**：调用本文件无其他函数；被 `create_tool` 间接调用（`GraphNodeUpdateTool()` 不传参数）。

### `GraphNodeUpdateTool.manager` （属性，第 260 行，`-> MemoryManager`）
- **作用**：这是一个只读的 `@property`，对外提供「一定能拿到管理器」的入口。它实现了惰性单例语义：第一次访问时如果 `self._manager` 还是 `None`，就调用 `_default_manager()` 创建并**写回** `self._manager`，之后再访问就直接复用，不会重复构建。这样做让工具在未被真正调用前保持轻量，同时保证同一次工具使用过程中始终用同一个管理器。
- **参数**：只有 `self`（属性访问不需要显式传参）。
- **返回**：`MemoryManager`，即注入的管理器或惰性构建出的默认管理器；保证不为 `None`。
- **内部流程**：`if self._manager is None:` → `self._manager = _default_manager()`；然后 `return self._manager`。
- **异常/边界**：若 `_default_manager()` 抛异常（例如 `_memory` 导入失败或磁盘初始化失败），异常向上传播，且因为赋值发生在调用之后，`self._manager` 仍保持 `None`，下次访问会再试一次。没有并发保护（多线程同时首次访问可能构建两次，但本工具 `parallel_safe=False`，实际不会被并行调用）。无超时控制。
- **同文件关系**：调用本文件的 `_default_manager`；被 `GraphNodeUpdateTool.execute`（第 267 行）读取。

### `GraphNodeUpdateTool.execute(self, arguments: GraphNodeUpdateInput) -> GraphNodeUpdateOutput` （第 266 行）
- **作用**：工具的实际执行体，把「查节点 → 合并更新 → 探测图投影 → 组装出参」四步串起来。它是整个模块面向 Agent 的唯一入口：Agent 传来的 JSON 先被校验成 `GraphNodeUpdateInput`，再进入这里。这里做了一件关键的语义映射——把入参里的 `None`（表示「不修改」）转换成下层纯函数期望的「falsy 空值」：`domain or ""`、`entity_type or ENTITY_DEFAULT_TYPE`、`description or ""`、`importance is None` 时用 `ENTITY_DEFAULT_CONFIDENCE` 兜底；`aliases` 直接透传（`None` 交给 `update_entity_node` 处理）。更新完成后它还要单独探测一次图投影是否真的存在该节点，把结果放进 `graph_projected`，让调用方知道这次是「真值源 + 图投影都刷了」还是「只刷了真值源」。
- **参数**：
  - `self`：工具实例。
  - `arguments: GraphNodeUpdateInput`：已通过 Pydantic 校验的入参对象，字段含义见 `GraphNodeUpdateInput` 一节。
- **返回**：`GraphNodeUpdateOutput`，其中 `name` 取 `item.metadata["canonical_name"]`，`node_id` 取 `item.id`，`created` 为 `existing is None`，`domain` 取 `item.metadata.get("domain") or ""`，`entity_type` 取 `item.metadata.get("entity_type") or ""`，`aliases` 为 metadata 里别名列表的字符串化副本，`importance` 为 `float(item.importance)`，`graph_projected` 为投影探测结果的布尔值。
- **内部流程**：① `manager = self.manager`（触发惰性构建，见 `manager` 属性）；② `existing = find_entity_node(manager, arguments.name)`；③ 调 `update_entity_node(manager, arguments.name, existing=existing, domain=..., entity_type=..., description=..., aliases=arguments.aliases, importance=..., create_if_missing=arguments.create_if_missing)` 得到 `item`；④ `projected = bool(getattr(manager.graph_store, "entity", lambda _name: {})(item.metadata["canonical_name"]))` —— 用 `getattr` 取图存储的 `entity` 查询方法，缺失时退化为一个「返回空字典」的 lambda（因此 `bool({})` 为 `False`），再用返回值的真假判断节点是否存在于投影中；⑤ 构造并返回 `GraphNodeUpdateOutput`，各字段都做了显式类型转换以适配 `strict=True`。
- **异常/边界**：`arguments.name` 清洗后为空时由 `update_entity_node` 抛 `ValueError`；`existing is None` 且 `arguments.create_if_missing` 为 `False` 时由 `update_entity_node` 抛 `LookupError`；`manager.graph_store` 属性缺失会在第 ④ 步的 `getattr(manager, "graph_store", ...)` 位置抛 `AttributeError`（此处只对 `graph_store` 上的 `entity` 做了兜底，没有对 `manager` 缺少 `graph_store` 做兜底）；`item.metadata` 缺少 `canonical_name` 键会抛 `KeyError`（因为用的是下标访问而非 `.get`）。空值处理集中在参数转换那几处 `or` 与 `is None` 判断。没有超时控制，超时由 `ToolSpec.timeout_seconds=30.0` 在外层生效；失败时不会回滚已经写入的真值源。
- **同文件关系**：调用本文件的 `find_entity_node`（第 268 行）、`update_entity_node`（第 269 行）以及通过 `manager` 属性间接调用 `_default_manager`；被工具运行时的调度层调用（本文件内无调用者）。

### `create_tool() -> BaseTool` （第 299 行）
- **作用**：工具工厂函数，供工具注册/发现机制调用。它把「如何构造这个工具」与「工具的实现细节」解耦：注册表只需要按约定调用无参的 `create_tool()`，就能拿到一个可用的 `BaseTool` 实例，而不必知道构造签名里还有个可选的 `manager` 参数。目前实现就是无参构造 `GraphNodeUpdateTool()`，因此默认走惰性管理器路径，第一次执行时才真正构建记忆管理器。
- **参数**：无参数。
- **返回**：`BaseTool`，实际类型是 `GraphNodeUpdateTool` 实例（返回类型标注为基类，是刻意的抽象）。
- **内部流程**：唯一语句 `return GraphNodeUpdateTool()`。
- **异常/边界**：无特殊处理；`GraphNodeUpdateTool()` 的构造本身不会抛异常（不做 IO），异常只会在此后使用工具时出现。
- **同文件关系**：调用本文件的 `GraphNodeUpdateTool` 构造函数；不被本文件内其他函数调用（由外部注册表调用）。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_default_manager` | 惰性导入 `_memory` 并构建共享的落盘 `MemoryManager`，避免模块导入期拉起重依赖。 |
| `push_graph_entity` | 把实体的 domain / 别名 / 重要度尽力镜像到图投影，图里没有该节点或不支持更新时返回 `False`。 |
| `update_entity_node` | 合并新旧属性（空值不改、别名取并集、重要度只升不降）并写回真值源、镜像图投影，是属性更新的唯一实现。 |
| `find_entity_node` | 先按 id 直查、再按规范化名与别名遍历语义记忆，定位一个实体节点或返回 `None`。 |
| `GraphNodeUpdateInput` | 工具入参模型，严格禁止多余字段，约束名称长度、别名数量与重要度范围，null/空串表示不修改。 |
| `GraphNodeUpdateOutput` | 工具出参模型，回传写入后的 canonical 名、节点 id、是否新建、领域、类型、别名、重要度与图投影是否已刷新。 |
| `GraphNodeUpdateTool` | 继承 `BaseTool` 的工具类，用 `ToolSpec` 声明名字 `knowledge.graph_node_update`、写副作用、30 秒超时与使用指引。 |
| `GraphNodeUpdateTool.__init__` | 仅保存可选注入的 `MemoryManager`，不触发任何 IO 或默认管理器构建。 |
| `GraphNodeUpdateTool.manager` | 只读属性，首次访问时惰性构建并缓存默认管理器，保证后续调用拿到同一个实例。 |
| `GraphNodeUpdateTool.execute` | 串起查节点、合并更新、探测图投影与组装出参，是 Agent 调用本工具的实际执行入口。 |
| `create_tool` | 无参工厂函数，返回一个默认惰性管理器的 `GraphNodeUpdateTool` 实例供注册表装载。 |
