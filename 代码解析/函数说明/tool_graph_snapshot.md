# tool/graph_snapshot.py

## 一、这个文件是干什么的

这个文件是「知识星云」可视化功能背后的**唯一投影实现**：它把四层记忆系统里散落在图存储（Neo4j 或内存回退）与 SQLite 里的数据，一次性「压扁」成前端星云图直接能吃的 `nodes + edges` 结构，并附带一整套统计数字。文件开头的模块 docstring 明确写下了映射规则：`domain` 映射为恒星（前端做星系定位）、`entity` 与 `chunk` 映射为行星（分别代表实体和文档原句）、`fact`/`note`/`event` 映射为卫星；而「关系」不再作为节点存在，二元关系直接变成一条有向边 `source→target`，谓词放在 `relation` 字段里。它对外暴露两条路径：一条是纯函数 `build_graph()`（给 Web 层的 `GET /api/graph` 用，也被 Agent 直接调用），另一条是封装成标准工具协议的 `GraphSnapshotTool`（工具名 `knowledge.graph_snapshot`，只读、幂等、可并行），供 Agent 在对话中主动询问「现在知识图谱长什么样」。文件顶部还有 `TOOL_ENABLED = True` 这个模块级开关，工具注册中心据此决定是否装载本工具。整个文件没有任何写操作：拓扑从图存储读，原文/预览/事件卫星从 SQLite 读，领域归类委托给同包的 `classify_domain`/`majority_domain`，孤儿实体统计委托给 `find_orphan_entities`，因此它可以安全缓存、可以被任意次重复调用。此外文件里还定义了三个 Pydantic 模型（输入、统计、输出）来约束工具协议的边界，以及一个 `create_tool()` 工厂函数作为注册入口。

## 二、函数与类逐条详解

### `domain_color(name: str) -> str` （第 56 行）

- **作用**：把一个领域名称稳定地映射成调色板 `NEBULA_PALETTE` 里的某一个颜色值。星云图里每个星系（领域）需要一种固定颜色，这样同一次渲染、不同次请求、甚至不同进程重启之后，同一个领域永远拿到同一个颜色，用户看到的星图不会「变色抖动」。它之所以用 `zlib.crc32` 而不是 Python 内建的 `hash()`，正是因为内建 `hash()` 对字符串开启了随机化种子（PYTHONHASHSEED），跨进程不稳定，而 crc32 是确定性算法。这个函数在每次创建节点时都会被间接调用（通过 `_node`），所以它是整张图颜色一致性的基础。
- **参数**：
  - `name`：`str`，领域名称，例如 `"技术"`、`"生活"`。允许传入空串或 `None`（类型标注是 `str`，但代码用 `(name or DEFAULT_DOMAIN)` 做了兜底），此时会被替换为常量 `DEFAULT_DOMAIN`。没有默认值，是必填位置参数。
- **返回**：`str`，`NEBULA_PALETTE` 中下标为 `crc32(utf-8 字节) % len(NEBULA_PALETTE)` 的那一项。因为取了模，返回值必然落在调色板的合法下标范围内，不会越界，也不会返回空。
- **内部流程**：第一步先做空值兜底 `name or DEFAULT_DOMAIN`，把空串、`None` 之类的假值统一成默认领域名；第二步用 `.encode("utf-8")` 把字符串编码成字节串；第三步调用 `zlib.crc32(...)` 得到 32 位无符号校验和；第四步对 `len(NEBULA_PALETTE)` 取模，得到稳定下标；第五步从调色板里取出该下标对应的颜色字符串直接返回。整个过程没有循环、没有分支（除了空值兜底）、没有外部 I/O。
- **异常/边界**：正常输入不会抛异常。传入非字符串（例如整数）时，`or` 判断会保留原值，随后 `.encode` 会抛 `AttributeError`，代码没有捕获；传入空串或 `None` 会静默退化为 `DEFAULT_DOMAIN` 的颜色。调色板为空列表时会触发 `ZeroDivisionError`（取模除零），但那是常量配置错误，本文件不做防御。
- **同文件关系**：它被本文件里的 `_node()` 调用（`_node` 在构造每个节点字典时计算 `"color"` 字段）。它自己只依赖外部常量 `NEBULA_PALETTE` 与 `DEFAULT_DOMAIN`，不调用本文件里的其它函数。

### `_date(item: MemoryItem) -> str` （第 62 行）

- **作用**：从一个记忆条目上取一个「日期字符串」用于星云图节点的 `date` 字段。前端星云图需要给节点标注日期（例如时序回放、悬停提示），但记忆条目的 `created_at` 字段在不同后端下类型并不统一——正常是 `datetime`，某些回退或序列化路径下可能是字符串。这个私有小函数的作用就是把这两种情况都收敛成字符串：能取到 `.date()` 就格式化成 ISO 日期（`YYYY-MM-DD`），取不到就退化成字符串截断。它是节点构造的辅助工具，保证 `date` 字段永远是字符串，前端不必做类型判断。
- **参数**：
  - `item`：`MemoryItem`，一条记忆条目对象。函数只读取它的 `created_at` 属性。没有默认值。
- **返回**：`str`。当 `item.created_at` 是带 `.date()` 方法的时间对象时，返回 `isoformat()` 的结果，形如 `"2024-05-17"`；当它没有 `.date()` 方法（例如本身就是字符串）时，返回 `str(created)` 的前 `NEBULA_DATE_CHARS` 个字符。两种路径都保证返回字符串。
- **内部流程**：先把 `item.created_at` 取到局部变量 `created`；然后进入 `try` 块调用 `created.date().isoformat()` 并直接返回；如果这一步抛出 `AttributeError`（说明 `created` 没有 `.date()`），就走 `except AttributeError` 分支，返回 `str(created)[:NEBULA_DATE_CHARS]`——即把任意对象转成字符串后按固定长度截断，避免超长字符串污染图数据。
- **异常/边界**：只捕获 `AttributeError`。如果 `item` 本身是 `None` 或没有 `created_at` 属性，会在 `created = item.created_at` 这一行抛 `AttributeError`，而该行位于 `try` 之外，因此异常会向外传播，不被本函数吞掉。如果 `created` 是 `None`，`str(None)` 得到 `"None"`，会作为日期字符串返回（属于脏数据的静默降级，不抛异常）。截断长度由常量 `NEBULA_DATE_CHARS` 决定。
- **同文件关系**：它被 `build_graph()` 内部的多个位置调用（实体节点、文档 `doc_meta` 的 `"date"`、备注节点、事件节点），用来填充 `_node()` 的 `date` 参数。它自己不调用本文件里的其它函数。

### `_node(node_id: str, kind: str, title: str, *, content: str = "", domain: str = "", date: str = "", importance: float = 0.5, parent: str | None = None, source: str | None = None) -> dict[str, Any]` （第 70 行）

- **作用**：这是全文件最核心的「节点工厂」。星云图前端要求每个节点是一个字段齐全的字典，如果各处手写字典，很容易漏字段或字段名不一致，导致前端某个节点画不出来。这个函数把节点的十一个字段一次性组装好：`id`、`kind`、`title`、`content`、`domain`、`date`、`importance`、`color`、`parent`、`source`、`meta`。其中 `color` 由 `domain_color(domain)` 自动推导，`importance` 统一四舍五入到三位小数，`meta` 统一初始化为空字典（留给调用方后续塞别名、`entity_type`、`document_id`、`related_entities` 等扩展信息），`domain` 为空时自动补 `DEFAULT_DOMAIN`。`build_graph` 里每一种节点（领域、实体、文档、备注、事件、事实）都是通过它生成的，因此它保证了输出结构的一致性。
- **参数**：
  - `node_id`：`str`，节点唯一标识，例如 `"dom:技术"`、`"ent:张三"`、`"doc:xxx"`、或直接复用 `MemoryItem.id`。必填。
  - `kind`：`str`，节点类型，取值在 `domain`/`entity`/`chunk`/`note`/`event`/`fact` 之中（前端按此决定恒星/行星/卫星的渲染与层级）。必填。
  - `title`：`str`，节点显示名。必填。
  - `content`：`str`，默认 `""`。节点的正文或预览文本，前端用于侧栏展示。
  - `domain`：`str`，默认 `""`。所属领域；空串会被 `domain or DEFAULT_DOMAIN` 兜底为默认领域。
  - `date`：`str`，默认 `""`。节点日期字符串，通常由 `_date()` 产出。
  - `importance`：`float`，默认 `0.5`。重要性/权重，用于前端大小或亮度；内部会 `round(float(importance), 3)`，因此传入 `int`、`str` 数字、`None` 之外的数值都会被强转。
  - `parent`：`str | None`，默认 `None`。父节点 id（行星挂恒星、卫星挂行星），前端据此做环绕布局。
  - `source`：`str | None`，默认 `None`。来源标识（文件名或 source），用于溯源。
  - 注意 `content` 之后的所有参数都是**关键字参数**（`*` 之后），调用时不能按位置传。
- **返回**：`dict[str, Any]`，一个包含上述十一个键的普通字典。键名固定，`meta` 恒为 `{}`（新建的空字典，每次调用都是新对象，不存在共享引用问题）。
- **内部流程**：函数体就是一个大的字典字面量构造。先放 `"id"`、`"kind"`、`"title"`、`"content"`；`"domain"` 用 `domain or DEFAULT_DOMAIN` 兜底；`"date"` 原样放入；`"importance"` 先 `float()` 再 `round(..., 3)`；`"color"` 调用 `domain_color(domain)`（注意这里传的是**原始** `domain` 参数而不是兜底后的值，但因为 `domain_color` 内部也会做同样的兜底，所以结果一致）；`"parent"`、`"source"` 原样放入；最后 `"meta": {}`。没有循环与条件分支之外的逻辑。
- **异常/边界**：`importance` 传 `None` 会在 `float(None)` 处抛 `TypeError`；传非数字字符串抛 `ValueError`；传 `NaN` 不会报错但会得到 `nan`。`domain` 传 `None` 会被兜底。其余参数不做校验，传什么类型就原样放进字典（Python 不做运行时类型检查）。不涉及超时。
- **同文件关系**：它调用 `domain_color()`。它被 `build_graph()` 里的 `domain_node()`、`entity_node()`、备注节点构造、事件节点构造、事实节点构造、文档节点构造这六处调用。

### `_entity_aliases(manager: MemoryManager) -> dict[str, list[str]]` （第 88 行）

- **作用**：一次性从图存储里把所有「实体名 → 别名列表」的映射取回来，供后续给实体节点填充 `meta["aliases"]` 使用（注释里标注为 U7 实体侧栏需求：P4 阶段已经把别名写进实体属性，这里只读不写）。之所以「一次取回」而不是每个实体查一次，是为了避免 N 次图库往返；之所以要包一层 `try`，是因为图库抖动（连不上、超时、协议错）时不能因为拿不到别名就让整张星云图构建失败，退化成「没有别名」是可接受的降级。
- **参数**：
  - `manager`：`MemoryManager`，记忆管理器实例。函数通过反射方式从它身上找 `graph_store`，再找 `graph_store.entity_aliases`，不直接依赖具体实现类型。
- **返回**：`dict[str, list[str]]`，键是实体名字符串，值是该实体的别名列表（列表元素都被 `str()` 转过）。当 `manager` 没有 `graph_store`、或 `graph_store` 没有可调用的 `entity_aliases`、或调用过程抛任何异常时，返回空字典 `{}`。
- **内部流程**：第一步用嵌套 `getattr` 加默认值 `None` 安全地取到 `getter = getattr(getattr(manager, "graph_store", None), "entity_aliases", None)`；第二步用 `callable(getter)` 判断，不可调用就直接 `return {}`；第三步在 `try` 里调用 `getter()`，对返回的字典做字典推导：键统一 `str(name)`，值用列表推导把每个别名 `str(alias)`，并用 `(aliases or [])` 兜住 `None`；第四步 `except Exception` 捕获一切异常返回 `{}`（带 `noqa: BLE001` 注释表明这是有意的宽泛捕获）。
- **异常/边界**：内部完全吞掉异常（包括 `AttributeError`、连接错误、超时错误等），对外不抛。别名值为 `None` 时按空列表处理。如果 `getter()` 返回的不是字典而是列表，`.items()` 会抛 `AttributeError`，同样被吞掉并返回 `{}`。没有超时控制。
- **同文件关系**：它被 `build_graph()` 调用一次，结果保存在局部变量 `aliases_by_entity`，随后由 `entity_node()` 读取。它不调用本文件里的其它函数。

### `_graph_snapshot(manager: MemoryManager, *, at: str | None = None) -> dict[str, Any] | None` （第 103 行）

- **作用**：从图存储后端读取「拓扑快照」。这个文件的设计原则是：**拓扑（实体与关系）来自图存储，原文与预览来自 SQLite**。所以构建图之前必须先问图存储要一份快照。但老版本后端可能根本没有 `graph_snapshot` 能力，或者图库临时不可用，因此本函数用 `None` 作为哨兵值来表达「图后端不可用，请调用方走兼容回退路径（只用 SQLite 信息）」。`TypeError`/`ValueError` 被特意重新抛出而不是吞掉，是因为那通常意味着调用方传参错了（例如 `at` 格式不被后端接受），属于编程错误，应该暴露出来而不是静默降级。
- **参数**：
  - `manager`：`MemoryManager`，记忆管理器，通过反射取其 `graph_store.graph_snapshot`。
  - `at`：`str | None`，关键字参数，默认 `None`。时间点（ISO 8601 字符串），用于向图存储请求「某个历史时刻的拓扑」；`None` 表示当前状态。`build_graph` 会把工具输入的 `at` 空串转成 `None` 后传进来。
- **返回**：`dict[str, Any] | None`。成功时返回 `dict(getter(at=at))`，即后端快照的浅拷贝字典（通常含 `mode`、`entities`、`observations`、`relations` 等键）；当 `graph_store` 不存在、或没有可调用的 `graph_snapshot`、或调用抛出非 `TypeError`/`ValueError` 的异常时返回 `None`。
- **内部流程**：第一步用嵌套 `getattr` 取 `getter`；第二步 `callable(getter)` 不成立则返回 `None`；第三步 `try` 内调用 `getter(at=at)` 并用 `dict(...)` 包一层（确保拿到的是可安全遍历的普通字典，同时避免把后端对象直接泄漏给上层）；第四步显式 `except (TypeError, ValueError): raise` 把参数类错误原样上抛；第五步 `except Exception` 兜底返回 `None`。
- **异常/边界**：会向上抛 `TypeError` 与 `ValueError`（这是刻意设计）。其它异常（网络、超时、后端内部错误、`KeyError` 等）一律吞掉返回 `None`。如果后端返回 `None`，`dict(None)` 会抛 `TypeError`，该异常会被重新抛出（因为 `TypeError` 在显式重抛之列）——这是一个需要注意的边界。
- **同文件关系**：它被 `build_graph()` 调用，返回值存在局部变量 `snapshot`，后续决定走「图存储拓扑」分支还是「SQLite 回退」分支，并影响返回值的 `graph_source` 字段。它不调用本文件里的其它函数。

### `build_graph(manager: MemoryManager, *, at: str | None = None) -> dict[str, Any]` （第 119 行）

- **作用**：整个文件的主干函数，也是「四层记忆 → 一张星云图」的完整实现。它把记忆条目、图存储拓扑、文档分块、实体别名、领域归类、孤儿统计全部汇总，输出一个包含 `graph_source`、`as_of`、`stats`、`nodes`、`edges` 五个键的字典。Web 层的 `GET /api/graph` 与 `GraphSnapshotTool.execute()` 都调用它。它内部把工作分成若干「遍」：先建显式实体节点，再收集 RAG 文档分块并归并成唯一文档行星，再把事实（三元组）与文档建立「提及」关联，然后建备注卫星与事件卫星，接着从图存储快照里读实体/观察/关系并生成边，最后用 SQLite 里的事实做补充或回退，收尾时去重文档、连「提及」边与「属于」边、统计各类节点数量与孤儿实体数。它是纯读取、无副作用的，所以可以被缓存或重复调用。
- **参数**：
  - `manager`：`MemoryManager`，记忆管理器，提供 `list()` 拿全部记忆条目，并提供 `graph_store` 供 `_entity_aliases`/`_graph_snapshot` 反射使用；同时被传给 `find_orphan_entities` 做孤儿统计。
  - `at`：`str | None`，关键字参数，默认 `None`。可选时间点，会原样透传给 `_graph_snapshot`，并原样（或空串）写入返回值的 `as_of`。只影响图存储的时序拓扑，不影响 SQLite 侧数据。
- **返回**：`dict[str, Any]`，结构固定为：
  - `graph_source`：`str`。若拿到了图快照，则为快照里的 `mode`（缺省 `"graph"`）；否则为 `"sqlite-fallback"`。
  - `as_of`：`str`。等于 `at or ""`。
  - `stats`：`dict`，含 `domains`、`entities`、`relations`、`facts`、`chunks`、`notes`、`events`、`edges`、`historical_facts`、`orphan_entities`、`total` 共 11 个计数。注意 `relations` 与 `edges` 都是 `len(edges)`（同一数值的两个别名）。
  - `nodes`：`list[dict]`，所有节点字典（顺序为插入顺序，即先领域/实体，再文档，再补入的图存储实体与事实节点等）。
  - `edges`：`list[dict]`，所有边字典。
- **内部流程**（按代码顺序）：
  1. `items = manager.list()` 一次性取出全部记忆条目；`snapshot = _graph_snapshot(manager, at=at)` 取图存储拓扑（可能为 `None`）。
  2. 初始化容器：`nodes`（节点字典，键为节点 id）、`edges`（边列表）、`domains`（领域名 → 领域节点 id）、`entity_ids`（实体名 → 实体节点 id）、`aliases_by_entity = _entity_aliases(manager)`、`historical_facts = 0`、`fact_keys`（已登记三元组集合，用于去重）、`chunk_to_doc`（记忆条目 id → 文档 id）、`doc_meta`（文档 id → 文档聚合信息）、`linked_pairs`（已连过的有向对集合，用于防重复边）。
  3. 定义四个闭包：`domain_node`（懒创建领域恒星节点，`dom:<name>`，`importance=0.9`，空名兜底默认领域并 `strip()`）、`entity_node`（懒创建实体行星节点，支持传入 `MemoryItem` 以便复用 `item.id` 与真实内容/日期/重要性/来源；否则用 `ent:<name>` 并给 `importance=0.6`；同时尝试从 `aliases_by_entity` 里按实体名或节点标题查别名塞进 `meta["aliases"]`）、`link_triple`（把 `(subject, predicate, object)` 三元组登记并生成有向边，先用 `fact_keys` 去重三元组、再用 `linked_pairs` 去重 `(source_id, predicate, target_id)`）、`attach_doc_entities`（把一个或多个实体名追加进某个文档的 `entity_set`/`entities`，文档不存在则直接返回）。
  4. **第一遍：显式实体**。筛出 `metadata["kind"] == "entity"` 的条目，逐个调用 `entity_node`，标题取 `metadata["title"]` 或 `content`，领域取 `metadata["domain"]` 或默认领域，并把 `item` 传进去以便节点携带真实内容。
  5. **RAG 分块收集**。遍历所有条目，只处理「`metadata["document_id"]` 不为 `None` 且含 `chunk_index`」的分块：记 `chunk_to_doc[item.id] = document_id`；用 `classify_domain(item.content, title=filename/source)` 给这一块判定领域；若文档首次出现，就在 `doc_meta` 里建立聚合项（`filename`、`preview`（内容前 `NEBULA_CONTENT_PREVIEW_CHARS` 字符）、`date`、`importance`、`domains` 列表、`entities` 列表、`entity_set` 集合、`source`）；否则只把本块领域追加进 `domains`。注意这一步**只收集不建节点**，节点要等实体都建好之后才建。
  6. **事实登记**。遍历所有条目，若 `subject`/`predicate`/`object` 三者齐全，则解析 `doc_id`（优先 `metadata["document_id"]`，否则用 `chunk_to_doc` 反查 `metadata["chunk_id"]`），调用 `attach_doc_entities(doc_id, subject, obj)` 把主语宾语挂到文档上，并把该条目收进 `sqlite_facts` 列表待后面处理边。
  7. **实体 → 文档回填**。再次遍历 `kind == "entity"` 的条目，解析它所属文档：先看 `metadata["document_id"]`，再看 `chunk_id` 反查，再退一步在 `metadata["source_ids"]` 里找第一个能命中 `chunk_to_doc` 的条目；实体名取 `canonical_name` 或 `title` 或 `content`；然后 `attach_doc_entities`。这一步的意义是：**即使某个实体还没有任何关系边，它也会出现在其文档的实体列表里**，后面文档行星会用「提及」边连上它，避免实体在图上变成孤儿。
  8. **备注卫星**。遍历 `kind == "note"` 的条目：从 `entity_ids` 里按 `metadata["entity"]`（先 `strip()`）找父实体；领域取 `metadata["domain"]`，若没有且父实体存在则复用父节点字典里的 `domain`，否则用默认领域；如果找不到父实体，就把父节点降级为领域恒星（`domain_node(domain)`）。最后用 `_node` 建 `kind="note"` 节点，标题取 `metadata["title"]` 或字面量 `"档案"`。
  9. **事件卫星**。遍历所有条目，跳过三类：① 三元组齐全的（已被事实处理）、② `kind` 属于 `{"entity", "note"}` 的、③ 含 `document_id` 且含 `chunk_index` 的分块。再额外跳过「一句话入库」类噪声：`metadata["ingest_job_id"]` 存在，或 `title`/`source` 等于 `"一句话入库"`，或内容以 `"添加了一条知识"` 开头。剩下的才建 `kind="event"` 节点，标题取 `metadata["title"]` 或内容前 `NEBULA_EVENT_TITLE_CHARS` 字符，父节点是所属领域恒星。
  10. **图存储拓扑分支**（`snapshot is not None` 时）：
      - 遍历 `snapshot["entities"]`：跳过空名；用 `entity_node(name, domain=properties["domain"] or DEFAULT_DOMAIN)` 建实体；把 `properties["aliases"]` 写进 `meta["aliases"]`；有 `properties["entity_type"]` 则写进 `meta["entity_type"]`。
      - 先把 `snapshot["observations"]` 的 id 收集成 `observation_ids` 集合（供后面 relations 去重用）。
      - 遍历 `snapshot["observations"]`（多元观察）：取 `observation_id`，跳过空 id；谓词默认 `"关联"`；把 `participants` 按 `ordinal` 升序排序；从中找出 `role == "subject"` 和 `role == "object"` 的参与者名字，缺任一就跳过；判断 `active = properties.get("active", True) is not False`；收集 `role` 不属于 subject/object 的 `extra` 参与者。若 `active` 为假，只把 `historical_facts` 加一然后 `continue`（历史事实不建边）。否则解析文档 id（`properties["document_id"]` 或 `chunk_to_doc[chunk_id]`；都没有时遍历 `doc_meta` 按 `source`/`filename` 匹配一次），`attach_doc_entities` 把主语宾语挂上；构造公共边属性字典 `common`（`confidence` 默认 0.75、`evidence`、`event_at`、`valid_from`、`valid_to`、`active`、`observation_id`）。若存在 `extra` 参与者，则走「多元观察」路径：建主语实体节点；若该观察 id 还没有节点，就用 `_node` 建一个 `kind="fact"` 的节点（标题形如 `"主语 -谓词-> 宾语"`，`content` 用 evidence，日期取 `event_at` 或 `created_at` 截断，`importance` 用 confidence，父节点为主语实体）；随后把 `subject`/`predicate`/`object`/`participants` 以及全部 `properties` 合并进该节点的 `meta`；登记 `fact_keys`；追加一条 `主语 → 观察节点` 的边（id 形如 `edge:<观察id>:subject`，relation 为谓词，带 `common`）；再对每个非主语参与者建实体节点并追加一条 `观察节点 → 参与者` 的边（id 用序号，relation 在 `role == "object"` 时统一写成 `"宾语"`，否则用原 role，并额外带 `role` 字段与 `common`）。若没有 `extra`，则退化为普通二元关系，调用 `link_triple(subject, predicate, object_name, domain=domain, extra=common)`。
      - 遍历 `snapshot["relations"]`：跳过 `properties["active"] is False` 的；跳过 `properties["memory_id"]` 命中 `observation_ids` 的（说明这条关系已由 observation 表达，避免重复）；取 `source`/`target`/`relation`（默认 `"关联"`），任一为空则跳过；最后 `link_triple(source, predicate, target, extra=dict(properties))`。
  11. **SQLite 事实补充/回退**。遍历前面收集的 `sqlite_facts`：若 `active` 为假，只有在**图快照不可用**（`snapshot is None`）且该三元组尚未登记时，才把 `historical_facts` 加一并登记三元组，然后 `continue`；否则调用 `link_triple`，`extra` 里带上 `confidence`（缺省用 `item.importance`）、`evidence`、`source_document`（或 `source`）、`chunk_id`、`active: True`、`cardinality`（默认 `"multi"`）。由于 `link_triple` 内部用 `fact_keys` 去重，图存储已经给出的关系不会被 SQLite 重复添加。
  12. **文档去重**。遍历 `doc_meta`，把 `preview` 用 `" ".join(str(...).split())` 归一化空白，`key` 取「归一化预览 → 文件名 → 文档 id」中第一个非空者；同一 `key` 的多个文档被合并：领域列表 `extend` 到已存在的那个，实体名去重后追加。这一步是为了处理「同一份原文被切成多块/多次入库」造成的重复行星。
  13. **建文档行星节点**。遍历去重后的文档：领域用 `majority_domain(info["domains"]) or DEFAULT_DOMAIN`（多数投票）；节点 id 为 `f"doc:{document_id}"`，若已存在则跳过；标题取文件名，若是 `"一句话入库"` 或 `"问答抽取"` 且有预览，则改取预览第一行的前 40 个字符；用 `_node` 建 `kind="chunk"` 节点，父节点为领域恒星；然后写入 `meta["document_id"]` 与 `meta["related_entities"]`；最后为该文档的每个实体名（能在 `entity_ids` 里找到的）追加一条 `文档 → 实体` 的「提及」边，并用 `linked_pairs` 去重。
  14. **恒星 ↔ 行星连线**。遍历所有节点，只处理 `kind` 为 `entity` 或 `chunk` 的：拼出 `star_id = f"dom:{node['domain']}"`，若该领域节点不存在则跳过；用 `(star_id, "属于", node_id)` 在 `linked_pairs` 里去重；追加一条 `领域 → 节点` 的 `"属于"` 边，带 `confidence: 1.0` 与 `structural: True`，让星图把恒星系和它辖下的行星用引力桥真正连起来。
  15. **统计与收尾**。`node_list = list(nodes.values())`；用 `kinds` 字典（初值含 `domain`/`entity`/`fact`/`chunk`/`note`/`event` 六个键，均为 0）遍历节点累加各类数量；调用 `find_orphan_entities(manager, items=items)` 取孤儿实体列表并取其长度作为 `orphan_entities`；最后组装并返回结果字典，`graph_source` 按快照是否存在取 `snapshot["mode"]`（默认 `"graph"`）或 `"sqlite-fallback"`，`as_of` 取 `at or ""`，`stats` 里的 `relations` 与 `edges` 都等于 `len(edges)`，`total` 等于节点总数。
- **异常/边界**：本函数自身没有 `try/except`，异常来自被调用方：`manager.list()` 失败会直接上抛；`_graph_snapshot` 可能上抛 `TypeError`/`ValueError`；`find_orphan_entities` 失败会上抛；`majority_domain`/`classify_domain` 的异常也未被捕获。空值处理相当密集：空领域名兜底 `DEFAULT_DOMAIN`，空实体名在 `entity_node` 里兜底默认领域，空 observation id 跳过，缺 subject/object 的观察跳过，空 source/target 的关系跳过，`properties.get(...)` 全部带默认值。`participants` 排序时用 `int(value.get("ordinal") or 0)`，若 `ordinal` 是非数字字符串会抛 `ValueError`（未捕获）。`int(participant.get('ordinal') or 0)` 在边 id 里同理。整体无超时控制，`at` 只透传不校验格式。
- **同文件关系**：它调用 `_graph_snapshot()`、`_entity_aliases()`、`_date()`、`_node()`、`domain_color()`（间接经 `_node`），以及内部的四个闭包 `domain_node()`、`entity_node()`、`link_triple()`、`attach_doc_entities()`；还调用外部同包函数 `classify_domain()`、`majority_domain()`、`find_orphan_entities()`。它被本文件的 `GraphSnapshotTool.execute()` 调用，也被 `__all__` 导出供 Web 层直接使用。

### `domain_node(name: str) -> str` （第 133 行，`build_graph` 内部嵌套函数）

- **作用**：领域恒星的「懒创建 + 记忆化」构造器。星云图里领域节点是星系中心，必须唯一：同一个领域名无论被多少个实体、文档、事件引用，都只能有一个恒星节点，否则前端会出现多个重叠星系。这个闭包用 `domains` 字典做缓存：第一次见到某个领域名才真正建节点，之后直接返回已有 id。它还负责把空名/纯空白名规范化成默认领域名，避免出现 `"dom:"` 这种畸形 id。
- **参数**：
  - `name`：`str`，领域名称。允许空串、`None`、纯空白字符串；内部会 `(name or DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN` 双重兜底。无默认值。
- **返回**：`str`，该领域的节点 id，形如 `"dom:<规范化领域名>"`。对同一领域名重复调用返回同一个字符串。
- **内部流程**：先规范化名称（假值换默认领域 → `strip()` 去空白 → 若结果为空串再换默认领域）；然后判断 `name not in domains`：不在就构造 `node_id = f"dom:{name}"`，调用 `_node(node_id, "domain", name, domain=name, importance=0.9)` 建节点并以 `node_id` 为键放进 `nodes`，同时把 `domains[name] = node_id` 登记进缓存；最后返回 `domains[name]`。恒星节点的 `importance` 固定 0.9，高于实体（0.6）和文档（继承原条目重要性），体现它在布局中的核心地位。
- **异常/边界**：无特殊处理（名称规范化已覆盖空值）。若外部传入非字符串（如整数），`or` 判断保留原值后 `.strip()` 会抛 `AttributeError`，未捕获。
- **同文件关系**：它调用 `_node()`。它被 `entity_node()`（作为实体默认父节点）、事件卫星构造、文档行星构造这三处调用。

### `entity_node(name: str, *, domain: str = "", item: MemoryItem | None = None, title: str | None = None) -> str` （第 141 行，`build_graph` 内部嵌套函数）

- **作用**：实体行星的「懒创建 + 记忆化 + 同名合并」构造器。全局约定是「全局同名同一个」——同一个实体名在图里只能有一颗行星，所以它用 `entity_ids` 字典缓存名字到节点 id 的映射。它有两种创建模式：如果调用方给了一个 `MemoryItem`（来自显式 `kind=entity` 的记忆条目），就用该条目的 `id` 作为节点 id，并把条目的正文、日期、重要性、来源（文件名或 source）都带上，让行星有真实内容可展示；如果没有条目（例如实体只是从图存储或关系里推导出来的名字），就用 `ent:<名字>` 作为合成 id，内容为空、日期为空、重要性 0.6。它还会顺带把别名写进 `meta["aliases"]`，供前端实体侧栏展示。
- **参数**：
  - `name`：`str`，实体名。内部 `(name or "").strip()`；若结果为空则用 `DEFAULT_DOMAIN` 当名字（避免空 id）。无默认值。
  - `domain`：`str`，关键字参数，默认 `""`。实体所属领域，用于设置 `parent` 与节点 `domain`；空值会被 `domain or DEFAULT_DOMAIN` 兜底。
  - `item`：`MemoryItem | None`，关键字参数，默认 `None`。可选的来源记忆条目；不为 `None` 时走「有内容模式」。
  - `title`：`str | None`，关键字参数，默认 `None`。显式指定节点标题，优先级高于 `item.metadata["title"]` 和实体名。
- **返回**：`str`，实体节点 id。若该名字已在 `entity_ids` 中，直接返回缓存 id（**不会**因为这次调用带了 `item` 而更新节点内容，属于「先到先得」）；否则返回新创建的节点 id。
- **内部流程**：① 规范化名字，空则用默认领域名；② `if key in entity_ids: return entity_ids[key]` 命中缓存直接返回；③ 分支 A（`item is not None`）：`node_id = item.id`，标题取 `title or item.metadata.get("title") or key`，`content`/`date`/`importance` 分别取 `item.content`、`_date(item)`、`item.importance`，`source` 取 `item.metadata` 里的 `filename` 或 `source`；④ 分支 B（无条目）：`node_id = f"ent:{key}"`，标题为名字，`content`/`date` 为空串，`importance = 0.6`，`source = None`；⑤ 调用 `_node(...)` 建 `kind="entity"` 节点，`parent=domain_node(domain or DEFAULT_DOMAIN)`，注意这里会**先创建/取到领域恒星**再建实体；⑥ 别名查找：`aliases_by_entity.get(key) or aliases_by_entity.get(node_title) or []`，非空则写入 `node["meta"]["aliases"]`（注释说明这是 U7 实体侧栏需求，只读不写）；⑦ 把节点放进 `nodes[node_id]`，登记 `entity_ids[key] = node_id`，返回 id。
- **异常/边界**：无 try/except。`item.metadata` 若是 `None`，`.get` 会抛 `AttributeError`（未捕获）。名字为空时静默使用默认领域名，可能导致多个空名实体被合并成同一个节点（这是刻意的兜底）。同名实体第二次出现时不会覆盖内容，也不会更新领域。`importance` 直接来自 `item.importance`，不做范围校验。
- **同文件关系**：它调用 `_node()`、`_date()`、`domain_node()`，并读取 `build_graph` 外层的 `entity_ids`、`nodes`、`aliases_by_entity` 三个变量。它被 `build_graph()` 主流程多处调用（第一遍显式实体、`link_triple()` 内部对主语宾语各调一次、图存储实体分支、观察的额外参与者、多元观察路径的主语），也被 `link_triple()` 调用。

### `link_triple(subject: str, predicate: str, obj: str, *, domain: str = "", extra: dict[str, Any] | None = None) -> None` （第 169 行，`build_graph` 内部嵌套函数）

- **作用**：把一条二元语义关系「落成图上的一条有向边」。星云图的设计决定关系不是节点，而是一条 `source→target` 的边、谓词放在 `relation` 字段。这个闭包承担三件事：确保主语和宾语两个实体节点存在（不存在就懒创建）、做两级去重（先按三元组 `fact_keys`，再按 `(source_id, predicate, target_id)` 的 `linked_pairs`）、追加边字典。二级去重的意义在于：不同来源（图存储的 relations 与 SQLite 的 facts）可能给出同一个关系，必须只保留一条边，否则前端会画出重叠边。
- **参数**：
  - `subject`：`str`，主语实体名（或任意会被 `str()` 化的值）。无默认值。
  - `predicate`：`str`，谓词/关系名，写入边的 `relation` 字段。无默认值。
  - `obj`：`str`，宾语实体名。无默认值。
  - `domain`：`str`，关键字参数，默认 `""`。传给 `entity_node` 决定新建实体挂到哪个领域。
  - `extra`：`dict[str, Any] | None`，关键字参数，默认 `None`。额外的边属性（如 `confidence`、`evidence`、`event_at`、`cardinality` 等），会用 `**dict(extra or {})` 展开合并进边字典。
- **返回**：`None`。它不返回边对象，边被直接 append 进外层的 `edges` 列表；调用方通过副作用观察结果。
- **内部流程**：① `triple = (str(subject), str(predicate), str(obj))`，先做字符串归一化（保证 `1` 和 `"1"` 不会当成两个不同三元组）；② 若 `triple in fact_keys` 直接 `return`（三元组级去重）；③ 否则 `fact_keys.add(triple)`；④ 调 `entity_node(subject, domain=domain)` 与 `entity_node(obj, domain=domain)` 拿到两端 id；⑤ 组 `pair = (source_id, predicate, target_id)`，若已在 `linked_pairs` 里则 `return`（端点级去重，能拦住「同一对端点同一谓词但名字写法不同」的重复）；⑥ `linked_pairs.add(pair)`；⑦ `edges.append({...})`，边字典含 `id`（`f"edge:{source_id}:{predicate}:{target_id}"`）、`source`、`target`、`relation`（等于 `predicate`），以及展开的 `extra`。注意 `extra` 若含 `id`/`source`/`target`/`relation` 同名键会覆盖前面的值（Python 字典字面量后者胜出）。
- **异常/边界**：无特殊处理，不做空值校验——传空字符串会正常生成 `ent:` 之类的节点（空名在 `entity_node` 里会被兜底成默认领域名）。`extra` 为 `None` 时按空字典处理。不涉及超时。
- **同文件关系**：它调用 `entity_node()`（两次），读写外层的 `fact_keys`、`linked_pairs`、`edges`。它被 `build_graph()` 里的多元观察回退路径、图存储 relations 循环、SQLite 事实循环三处调用。

### `attach_doc_entities(doc_id: str | None, *names: object) -> None` （第 197 行，`build_graph` 内部嵌套函数）

- **作用**：把一个或多个实体名登记到某个文档的「被提及实体」集合里。文档行星后面会用这些名字生成「提及」边，所以这个函数决定了「原句行星能连到哪些实体」。它同时维护两个结构：`entity_set`（集合，用于 O(1) 去重判断）和 `entities`（列表，用于保持稳定的输出顺序），这是典型的「集合判重 + 列表保序」双结构写法。它也被用来保证「即使实体没有任何关系边，只要在文档里被提到过，就不会变成图上的孤儿」。
- **参数**：
  - `doc_id`：`str | None`，文档 id。若为 `None`、空串、或不在 `doc_meta` 里，函数立即返回不做任何事。无默认值。
  - `*names`：`object`，可变位置参数，任意个实体名（可能来自 `metadata` 的各种字段，类型不保证是 `str`）。每个名字会先 `str(name or "").strip()`，空则跳过。
- **返回**：`None`。结果通过修改 `doc_meta[doc_id]` 里的 `entity_set` 与 `entities` 体现。
- **内部流程**：① `if not doc_id or doc_id not in doc_meta: return` 做前置守卫；② 取出 `info = doc_meta[doc_id]`；③ 遍历 `names`：把每个名字 `str(name or "").strip()`；④ 若名字非空且不在 `info["entity_set"]` 里，则先 `add` 进集合，再 `append` 进列表（保证列表无重复且顺序稳定）。
- **异常/边界**：无异常抛出（`doc_meta` 的存在性已用守卫检查）。`doc_id` 是 `None` 或空串时静默忽略——这正好覆盖了「某些事实条目没有文档归属」的常见情况，属于刻意的宽容处理。`names` 为空时函数只做守卫判断后返回。
- **同文件关系**：它读写外层的 `doc_meta`，不调用本文件里的其它函数。它被 `build_graph()` 主流程中的事实登记循环、实体→文档回填循环、图存储观察循环三处调用。

### `class GraphSnapshotInput(BaseModel)` （第 585 行）

- **作用**：`knowledge.graph_snapshot` 工具的**输入契约**。它继承 Pydantic 的 `BaseModel`，通过 `model_config = ConfigDict(extra="forbid", strict=True)` 声明了两条硬约束：多传任何未知字段直接报错（`extra="forbid"`，防止模型幻觉出无意义参数），且类型必须严格匹配、不做隐式转换（`strict=True`，例如字符串 `"200"` 不会被当成整数 `200`）。工具协议层用它来校验并反序列化 Agent 传进来的参数，是「工具边界上的守门员」。它本身没有方法，只有三个带描述与约束的字段，字段的 `description` 会进入模型的工具 schema，帮助 LLM 正确填参。
- **参数**（即三个字段，均有默认值，因此整个输入可以留空调用）：
  - `at`：`str`，默认 `""`。可选时间点（ISO 8601），只影响图存储的时序拓扑，留空表示当前状态。注意默认值是空串而不是 `None`，`execute()` 会把空串转成 `None` 再传给 `build_graph`。
  - `include_nodes`：`bool`，默认 `True`。为 `false` 时只返回统计，不返回节点与边——这是为「大图放进模型上下文之前先看规模」准备的省 token 开关。
  - `max_nodes`：`int`，默认 `200`，约束 `ge=1`、`le=5000`。限制返回的节点数与边数上限（两者各自独立按此值截断），`stats` 始终是完整数量。
- **返回**：作为模型类，它本身不「返回」；实例化得到的是一个校验后的输入对象，供 `GraphSnapshotTool.execute()` 读取三个属性。
- **内部流程**：没有显式方法体，逻辑由 Pydantic 在实例化时执行：解析传入字典 → 按 `extra="forbid"` 拒绝未知键 → 按 `strict=True` 校验类型 → 对 `max_nodes` 施加 `ge`/`le` 边界 → 缺失字段填入默认值 → 得到实例。`Field(description=...)` 提供元数据供工具 schema 生成。
- **异常/边界**：校验失败时抛 `pydantic.ValidationError`（例如 `max_nodes=0`、`max_nodes=6000`、`include_nodes="yes"` 在严格模式下、或多传字段），由工具框架负责转换为错误响应。没有超时概念。
- **同文件关系**：它被 `GraphSnapshotTool.spec` 通过 `input_model=GraphSnapshotInput` 引用，并被 `execute()` 的参数类型标注使用。它不调用本文件里的其它函数。

### `class GraphStats(BaseModel)` （第 604 行）

- **作用**：星云图统计数字的**输出契约**，是 `GraphSnapshotOutput.stats` 字段的类型。它把 `build_graph()` 返回的那个 `stats` 字典结构化，让工具输出有稳定、可校验、可被下游程序消费的形状。所有字段都是 `int` 且无默认值（即全部必填），另有 `extra="forbid"` + `strict=True` 保证不会出现多余字段、也不会把字符串数字混进来。`build_graph` 里 `relations` 与 `edges` 数值相同，这里也保留了两个字段，因为它们在语义上分别是「关系条数」和「边条数」，前端可能分别展示。
- **参数**（即十一个必填字段）：
  - `domains`：`int`，领域（恒星）节点数量。
  - `entities`：`int`，实体（行星）节点数量。
  - `relations`：`int`，关系条数（等于边数）。
  - `facts`：`int`，事实（卫星）节点数量。
  - `chunks`：`int`，文档/原句（行星）节点数量。
  - `notes`：`int`，备注（卫星）节点数量。
  - `events`：`int`，事件（卫星）节点数量。
  - `edges`：`int`，边总数。
  - `historical_facts`：`int`，描述为「已失效（active=false）的历史事实数」。
  - `orphan_entities`：`int`，描述为「完全孤立的实体数（复用 knowledge.orphan_entities）」。
  - `total`：`int`，描述为「节点总数」。
- **返回**：无（模型类）。实例化后提供上述十一个整数属性。
- **内部流程**：无方法体，完全由 Pydantic 在 `GraphStats(**payload["stats"])` 时执行字段校验与赋值。三个带 `Field(description=...)` 的字段会携带说明文本进入输出 schema。
- **异常/边界**：若 `payload["stats"]` 缺少任一键，实例化抛 `ValidationError`（`Missing`）；若含额外键，因 `extra="forbid"` 也抛 `ValidationError`。所以 `build_graph` 的 `stats` 字典键集必须与这里的字段集**完全一致**，这是两者之间的隐式契约。
- **同文件关系**：它被 `GraphSnapshotTool.execute()` 用 `GraphStats(**payload["stats"])` 实例化，并被 `GraphSnapshotOutput.stats` 的类型标注引用。它不调用本文件里的其它函数。

### `class GraphSnapshotOutput(BaseModel)` （第 620 行）

- **作用**：`knowledge.graph_snapshot` 工具的**输出契约**，也是 `execute()` 的返回类型。它把 `build_graph()` 产出的原始字典包装成结构化、可校验、可序列化的工具结果，供 Agent 框架回填给模型。它同时承担「截断语义」的表达：`truncated` 字段明确告诉调用方节点/边列表是否被裁剪过，而 `stats` 永远是全量，避免模型把被截断的列表误当成全图。`extra="forbid"` + `strict=True` 保证输出形状稳定。
- **参数**（即五个字段）：
  - `graph_source`：`str`，必填。描述为「拓扑来源：图存储的 mode，或 sqlite-fallback」。
  - `as_of`：`str`，必填。描述为「实际生效的时间点；空串表示当前状态」。
  - `stats`：`GraphStats`，必填。嵌套的统计模型。
  - `nodes`：`list[dict[str, Any]]`，默认空列表（`default_factory=list`）。节点字典列表，被截断时只含前 `max_nodes` 个。
  - `edges`：`list[dict[str, Any]]`，默认空列表。边字典列表，同样可能被截断。
  - `truncated`：`bool`，必填。描述为「true 表示节点/边被 max_nodes 截断，只有 stats 是全量」。
- **返回**：无（模型类）。实例化后提供上述五个属性，可被框架序列化成 JSON 交给模型。
- **内部流程**：无方法体。由 `execute()` 中 `GraphSnapshotOutput(graph_source=..., as_of=..., stats=..., nodes=..., edges=..., truncated=...)` 构造，Pydantic 负责类型校验、默认值填充（`nodes`/`edges` 未传时用空列表）与嵌套模型校验。
- **异常/边界**：缺 `graph_source`/`as_of`/`stats`/`truncated` 或类型不符会抛 `ValidationError`；`nodes`/`edges` 里的元素是 `dict[str, Any]`，内部值不做进一步校验（`Any` 放行），所以节点字典的字段正确性由 `build_graph` 负责而非此处。
- **同文件关系**：它引用 `GraphStats`，被 `GraphSnapshotTool.spec` 以 `output_model=GraphSnapshotOutput` 引用，并被 `execute()` 作为返回类型标注与实例化目标。

### `class GraphSnapshotTool(BaseTool)` （第 631 行）

- **作用**：把 `build_graph()` 这套投影能力封装成符合项目工具协议的标准工具，工具名 `knowledge.graph_snapshot`。类属性 `spec`（一个 `ToolSpec`）声明了工具的元信息：中英文混合的 `description` 说明它是「把四层记忆投影成星图（nodes + edges）并带统计，只读，关系来自图存储、原文与预览来自 SQLite」；`version="1.0.0"`；`input_model`/`output_model` 绑定上面两个 Pydantic 模型；`side_effect="read"` 与 `permissions=()` 表明无副作用、无需任何权限；`timeout_seconds=180.0` 给大图构建留出三分钟预算；`idempotent=True`、`parallel_safe=True` 允许缓存与并发调用；`tags` 便于检索分类；`guidance` 是给模型的自然语言使用建议——需要整张星图做展示或宏观分析时用它，但「回答具体事实请用召回或图检索工具，不要用整图快照代替检索（慢且噪声大）」，并提醒 `max_nodes` 会截断、截断时以统计与 `truncated` 标记为准。类本身只负责「协议包装」，真正的算法在 `build_graph` 里。
- **参数**：类无构造参数意义上的字段；实例化参数见 `__init__`。
- **返回**：无（类）。
- **内部流程**：类体主要是一个 `spec = ToolSpec(...)` 类属性赋值，加上三个方法（`__init__`、`manager` 属性、`execute`）。`ToolSpec` 在类定义时构造一次，作为工具注册中心读取的元数据来源。
- **异常/边界**：`ToolSpec` 的字段若不合法，会在导入模块（类定义）时立刻抛异常，属于「启动即失败」的显式设计，而不是运行时才发现。除此之外无特殊处理。
- **同文件关系**：它引用本文件的 `GraphSnapshotInput`、`GraphSnapshotOutput`、`build_graph`（在 `execute` 里调用），并在 `create_tool()` 里被实例化，在 `__all__` 里被导出。它继承外部的 `BaseTool`。

### `GraphSnapshotTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 655 行）

- **作用**：工具的构造函数。它接受一个可选的记忆管理器并保存到实例属性 `self._manager` 上，从而支持**依赖注入**：测试或 Web 层可以把已经建好的 `MemoryManager` 直接塞进来复用；不传时保持 `None`，等真正用到时再由 `manager` 属性懒加载一个默认实例。这种「构造时不建、用时才建」的写法避免了工具实例化阶段的副作用（连接数据库、连图库）拖慢工具注册。
- **参数**：
  - `manager`：`MemoryManager | None`，默认 `None`。外部注入的记忆管理器；传 `None` 表示「稍后自动构建默认管理器」。
- **返回**：`None`（构造函数）。
- **内部流程**：函数体只有一行 `self._manager = manager`，把参数原样存为私有属性，不做任何校验、不做任何初始化 I/O。
- **异常/边界**：无特殊处理，不会抛异常；传入任意对象（即使不是 `MemoryManager`）也会被原样保存，类型错误要到 `execute()` 使用它时才会暴露。
- **同文件关系**：它设置的状态被 `manager` 属性读取，也被 `execute()` 间接使用（通过 `self.manager`）。它被 `create_tool()` 以无参方式调用。不调用本文件里的其它函数。

### `GraphSnapshotTool.manager` （property，第 658 行）

- **作用**：一个只读属性，用来「惰性获取」记忆管理器。它是 `__init__` 注入机制的另一半：如果构造时没有注入，第一次访问这个属性时才在函数内部 `from ._memory import build_default_manager` 并调用它建一个默认管理器，然后缓存回 `self._manager`，后续访问直接复用同一个实例。把 `import` 放在函数体内（延迟导入）是为了避免模块导入期的循环依赖——`_memory` 很可能又会引用工具模块，顶层导入容易形成环。对调用方而言，`self.manager` 永远能拿到一个可用的 `MemoryManager`，不必关心它是注入的还是新建的。
- **参数**：无（只有隐式的 `self`）。
- **返回**：`MemoryManager`。返回的一定是非 `None` 的管理器实例（要么是注入的，要么是新建并缓存的）。
- **内部流程**：① 判断 `if self._manager is None`；② 成立则执行函数内导入 `from ._memory import build_default_manager`；③ 调用 `build_default_manager()` 并把结果赋给 `self._manager`；④ 返回 `self._manager`。不成立则直接返回已有的 `self._manager`。整个过程是「检查 → 建 → 缓存 → 返回」的经典懒初始化。
- **异常/边界**：若 `_memory.build_default_manager()` 内部失败（例如数据库打不开），异常会从这个属性访问点向上抛，不做捕获。非线程安全的懒初始化（并发首次访问可能各建一个实例，但最终只有一个被缓存），不过由于工具声明了 `parallel_safe=True`，实际并发首次调用时理论上存在极小概率重复构建——代码没有加锁，属于已知的宽松处理。
- **同文件关系**：它读取 `__init__` 设置的 `self._manager`，被 `execute()` 通过 `self.manager` 调用。不调用本文件里的其它函数（只调用外部的 `build_default_manager`）。

### `GraphSnapshotTool.execute(self, arguments: GraphSnapshotInput) -> GraphSnapshotOutput` （第 666 行）

- **作用**：工具的实际执行入口，是「工具协议」到「纯函数算法」的适配层。它做三件事：调用 `build_graph()` 拿到全量图数据；按 `arguments.include_nodes` 与 `arguments.max_nodes` 对节点和边做裁剪并设置 `truncated` 标记；把结果包装成 `GraphSnapshotOutput`（其中 `stats` 用 `GraphStats` 做结构化校验）。裁剪逻辑的存在是因为整图可能很大，直接塞进模型上下文既慢又噪声大，所以要么完全不要节点（只看统计），要么只取前 N 个。注意 `stats` 永远来自未被裁剪的全量结果，这是它和 `nodes`/`edges` 的关键区别。
- **参数**：
  - `arguments`：`GraphSnapshotInput`，已通过 Pydantic 校验的输入对象。其中 `at` 为空串时会被转成 `None` 传给 `build_graph`；`include_nodes` 决定是否返回节点与边；`max_nodes` 决定裁剪上限（1~5000）。
- **返回**：`GraphSnapshotOutput`。字段来源：`graph_source` 与 `as_of` 直接取 `payload` 对应值（都经 `str()` 转换）；`stats` 为 `GraphStats(**payload["stats"])`；`nodes`/`edges` 为裁剪后的列表；`truncated` 为布尔标记。
- **内部流程**：① `payload = build_graph(self.manager, at=arguments.at or None)`——`self.manager` 触发惰性管理器获取，`arguments.at or None` 把空串统一成 `None`；② `nodes = list(payload["nodes"])`、`edges = list(payload["edges"])` 复制一份，避免后续切片影响到原结构（切片本身也返回新列表，这里主要是显式表达「副本」语义）；③ `truncated = False` 初始化；④ 若 `not arguments.include_nodes`：`truncated = bool(nodes or edges)`（只要原来有内容就标记为已截断，因为调用方确实拿不到节点了），然后把 `nodes`、`edges` 都置为空列表；⑤ 否则若 `len(nodes) > arguments.max_nodes or len(edges) > arguments.max_nodes`：置 `truncated = True`，并把两个列表分别切到前 `max_nodes` 个（`nodes[:max_nodes]`、`edges[:max_nodes]`）；⑥ 构造并返回 `GraphSnapshotOutput`，`stats` 用 `GraphStats(**payload["stats"])` 展开校验。
- **异常/边界**：自身没有 try/except。`build_graph` 抛出的异常（例如 `manager.list()` 失败、`_graph_snapshot` 上抛的 `TypeError`/`ValueError`、`find_orphan_entities` 失败）会直接向上传播，由工具框架处理。若 `payload["stats"]` 键集与 `GraphStats` 字段不一致会抛 `ValidationError`。边界情况：`include_nodes=False` 且图完全为空时 `truncated` 为 `False`（因为没有东西可截断）；`max_nodes` 恰好等于列表长度时不置 `truncated`（判断用的是严格大于）；节点和边各自独立按 `max_nodes` 截断，可能出现「节点被截而边未截」的组合。`timeout_seconds=180.0` 的超时控制由框架层执行，本函数内部不实现。
- **同文件关系**：它调用 `build_graph()`，并实例化 `GraphStats` 与 `GraphSnapshotOutput`；通过 `self.manager` 属性获取管理器。它被工具框架（外部）在收到 `knowledge.graph_snapshot` 调用时执行，也被 `create_tool()` 产出的实例所承载。

### `create_tool() -> BaseTool` （第 688 行）

- **作用**：工具的工厂函数，是工具注册中心发现并装载本工具的标准入口。项目里的工具通常以「模块 + `create_tool()` 工厂」的形式被扫描，这样每个工具可以有自己独立的构造逻辑（例如需要注入依赖），而不必强制所有工具都提供无参构造函数。它被 `__all__` 导出，说明它是本模块的公开 API 之一。由于 `GraphSnapshotTool.__init__` 的 `manager` 参数有默认值 `None`，这里无参调用即可，管理器会在首次执行时惰性构建。
- **参数**：无。
- **返回**：`BaseTool`，实际返回的是 `GraphSnapshotTool()` 实例（子类实例，类型标注用父类以保持工厂的通用性）。
- **内部流程**：函数体只有一行 `return GraphSnapshotTool()`，不做参数解析、不做配置读取、不做缓存（每次调用都新建一个工具实例）。
- **异常/边界**：无特殊处理，不会抛异常（除非 `GraphSnapshotTool` 类定义本身有问题，那属于导入期错误）。
- **同文件关系**：它实例化本文件的 `GraphSnapshotTool`，自身被外部工具注册逻辑调用，并在 `__all__` 中导出。它不调用本文件里的其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `domain_color` | 用 crc32 把领域名稳定映射成调色板里的一个颜色，保证跨进程同一领域颜色不变。 |
| `_date` | 把记忆条目的 `created_at` 统一转成日期字符串（能取 `.date()` 就 ISO 化，否则字符串截断）。 |
| `_node` | 节点工厂：组装包含 id/kind/title/content/domain/date/importance/color/parent/source/meta 的节点字典。 |
| `_entity_aliases` | 一次性从图存储读回「实体名 → 别名列表」，读不到就退化为空表而不让整图失败。 |
| `_graph_snapshot` | 从图存储读取拓扑快照，不可用时返回 `None` 让调用方走 SQLite 回退。 |
| `build_graph` | 主干函数：把四层记忆完整投影成星云图的 nodes + edges + stats（纯读取、可缓存）。 |
| `domain_node`（嵌套） | 领域恒星的懒创建与记忆化，保证一个领域只有一个 `dom:` 节点。 |
| `entity_node`（嵌套） | 实体行星的懒创建、同名合并与别名填充，支持带 MemoryItem 的「有内容模式」。 |
| `link_triple`（嵌套） | 确保两端实体存在并追加一条去重后的有向关系边。 |
| `attach_doc_entities`（嵌套） | 把一个或多个实体名去重地登记到某文档的「被提及实体」集合与列表中。 |
| `GraphSnapshotInput` | 工具输入模型：`at`、`include_nodes`、`max_nodes` 三个严格校验的参数。 |
| `GraphStats` | 工具输出中的统计模型，含领域/实体/事实/文档/备注/事件/边/历史事实/孤儿/总数等 11 个计数。 |
| `GraphSnapshotOutput` | 工具输出模型：拓扑来源、时间点、统计、节点、边与截断标记。 |
| `GraphSnapshotTool` | 把 `build_graph` 包装成只读、幂等、可并行的标准工具 `knowledge.graph_snapshot`（含 spec 元信息）。 |
| `GraphSnapshotTool.__init__` | 保存可选注入的 `MemoryManager` 到 `self._manager`，支持依赖注入与惰性构建。 |
| `GraphSnapshotTool.manager` | 只读属性：注入的管理器为空时延迟导入并构建默认管理器，然后缓存复用。 |
| `GraphSnapshotTool.execute` | 调用 `build_graph` 并按 `include_nodes`/`max_nodes` 裁剪节点与边，包装成 `GraphSnapshotOutput` 返回。 |
| `create_tool` | 工具工厂：无参实例化并返回 `GraphSnapshotTool()`，供工具注册中心装载。 |
