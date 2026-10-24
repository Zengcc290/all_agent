# tool/_documents.py

## 一、这个文件是干什么的

`tool/_documents.py` 是整个项目里「文档真值源」（数据库中的 `documents` 表与 `chunks` 表）的**共享序列化助手模块**。它的职责非常单一而明确：把从存储层读出来的 ORM/数据记录对象（`DocumentRecord`、`ChunkRecord`）转换成 API 与工具层都能直接消费的**纯 Python 字典**（JSON 可序列化形状）。之所以要把这个转换动作单独抽成一个文件，是因为项目里有三个文档类工具——`knowledge.document_list`、`knowledge.document_get`、`knowledge.document_revectorize`——都需要输出同样形状的文档/分块数据。如果每个工具各写一份转换代码，字段一旦增删就会三处漂移，出现「同一个文档在列表接口和详情接口里字段不一致」这种难以排查的问题。把转换逻辑收敛到这里，就只有一个真值源。

这个文件是一个**不可发现模块**（undiscoverable module）：`core.discovery` 在扫描工具目录时会忽略所有以 `_` 开头的模块，因此本文件既没有 `TOOL_ENABLED` 常量，也没有 `create_tool()` 工厂函数，它永远不会被注册成一个可被 Agent 调用的工具，只能被其它模块 `import` 使用。这一点从命名上的下划线前缀就能看出来，是有意为之的设计约束。

文件内容极其轻量：模块级只依赖 `typing.Any` 与 `memory.storage.document_repo` 中的 `ChunkRecord`、`DocumentRecord` 两个记录类型，然后定义了三个模块级函数——`chunk_summary`（单个分块的 JSON 形状）、`document_summary`（列表视图的文档行 + 分块数量）、`document_detail`（详情视图的完整文档 + 全部子分块），最后用 `__all__` 显式声明对外导出面。它不新建仓储、不打开数据库、不做任何 I/O：文件顶部的文档字符串特意强调，仓储实例的创建（`repository_for(manager)`）已经在 `tool/hybrid_index.py` 里按「`:memory:` → `None`」的口径实现过了，本文件直接复用那一份，绝不写第二份，避免连接口径分叉。

在运行链路上，它处在「存储层 → 工具层/API 层」的中间夹层：三个文档工具先通过仓储拿到 `DocumentRecord` / `ChunkRecord` 列表，再调用这里的函数把记录拍平成字典，最后交给工具返回值序列化或 FastAPI 的 JSON 响应。因为它没有副作用、没有状态、输入输出一一对应，所以也是整个项目里最容易测试、最不可能出 bug 的一层。

## 二、函数与类逐条详解

### `chunk_summary(chunk: ChunkRecord) -> dict[str, Any]` （第 19 行）

- **作用**：把一条分块记录（`ChunkRecord`）转换成 API 与工具层统一暴露的 JSON 字典形状。分块是文档被切分后的最小检索单元，一个文档通常对应很多条分块，所以这个函数是整个模块里被调用次数最多的一个——它是 `document_detail` 内部按列表推导逐条调用的小工具。之所以需要它，是因为 `ChunkRecord` 是存储层的记录对象，字段名、字段集合都受数据库 schema 约束，而对外暴露的契约（API 响应、工具返回）需要一份稳定、可 JSON 序列化的字段清单；这个函数就是两者之间的唯一映射点。它的输出刻意只保留「定位 + 内容 + 向量状态」三类信息：`chunk_id` 与 `chunk_index` 用于标识和排序，`char_start` / `char_end` 用于在原文里定位切片区间，`text` 是分块正文，`vector_status` 表明这条分块是否已经完成向量化。需要注意的是它**不包含**所属文档的 id——调用方在 `document_detail` 里已经处于某个文档的上下文内，重复携带文档 id 是冗余的。
- **参数**：
  - `chunk`：必填，类型为 `ChunkRecord`（从 `memory.storage.document_repo` 导入）。这是存储层返回的单条分块记录对象，函数通过属性访问（`chunk.chunk_id`、`chunk.chunk_index`、`chunk.char_start`、`chunk.char_end`、`chunk.text`、`chunk.vector_status`）读取字段。没有默认值，不接受 `None`。约束上，它必须是一个具备上述六个属性的对象；由于函数只做属性读取，任何满足该属性协议的鸭子类型对象也能工作，但类型标注约定的就是 `ChunkRecord`。
- **返回**：返回一个 `dict[str, Any]`，固定包含六个键：`chunk_id`（分块标识）、`chunk_index`（分块在文档内的序号）、`char_start`（分块在原文中的起始字符偏移）、`char_end`（结束字符偏移）、`text`（分块文本内容）、`vector_status`（向量化状态）。所有值都直接取自传入记录的对应属性，函数不做任何类型转换或格式化，因此返回值的具体类型（如 `chunk_id` 是字符串还是整数）完全取决于 `ChunkRecord` 的定义。任何情况下都返回该字典，不存在条件分支导致的返回 `None`。
- **内部流程**：整个函数体只有一条 `return` 语句，直接构造并返回一个字典字面量。执行顺序就是字典键的书写顺序：先取 `chunk.chunk_id`，再 `chunk.chunk_index`，然后 `chunk.char_start`、`chunk.char_end`、`chunk.text`，最后 `chunk.vector_status`。没有循环、没有条件判断、没有调用任何库函数，也没有调用本文件里的其它函数。字段顺序在这个函数里保持稳定，便于阅读和 diff 对比。
- **异常/边界**：如果 `chunk` 为 `None`，在第一个属性访问处就会抛出 `AttributeError`；如果传入的对象缺少上述六个属性中的任意一个，同样会抛出 `AttributeError`。函数自身不做任何空值校验、不做默认值兜底、不做异常捕获，也不存在超时概念（纯内存计算）。`text` 字段即使为空字符串或很长的文本，也会原样透传，不做截断。
- **同文件关系**：它不调用本文件里的任何其它函数（是三个函数里唯一一个完全独立的叶子函数）；被本文件里的 `document_detail` 在列表推导 `[chunk_summary(chunk) for chunk in chunks]` 中逐条调用。

### `document_summary(document: DocumentRecord, *, chunk_count: int) -> dict[str, Any]` （第 32 行）

- **作用**：把一条文档记录（`DocumentRecord`）加上它的分块数量，转换成**列表视图**的 JSON 字典形状。所谓列表视图，就是「一次要返回很多篇文档、只需要概览信息」的场景，例如 `knowledge.document_list` 工具或文档列表接口。它故意**不包含** `raw_text`（原文全文）和 `chunks`（分块明细），因为一篇长文档的原文可能有几十万字，如果在列表里逐篇携带，响应体会爆炸式膨胀、传输和序列化都会变慢；同理也不包含 `permission`、`error`、`updated_at` 这些只有在查看单篇详情时才有意义的字段。`chunk_count` 被单独作为一个关键字参数传入，而不是从 `document` 上读取，是因为分块数量属于 `chunks` 表的聚合结果（通常是仓储层用 `COUNT(*)` 查出来的），文档记录本身并不携带这个信息——把它设计成显式的必填关键字参数，能让调用方无法「忘记传」或「传错位置」，也让这个函数保持纯函数、不依赖仓储。
- **参数**：
  - `document`：必填，类型为 `DocumentRecord`（从 `memory.storage.document_repo` 导入）。存储层返回的单条文档记录，函数读取它的 `document_id`、`title`、`source`、`tags`、`status`、`created_at` 六个属性。不接受 `None`，无默认值。
  - `chunk_count`：必填，类型为 `int`，且被声明为**仅关键字参数**（签名里的 `*` 之后，因此调用时必须写成 `chunk_count=...`，不能按位置传）。它表示该文档当前拥有的分块总数，典型来源是仓储层的聚合计数。约束上它应当是非负整数；函数本身不做校验，传负数或非整数也不会在这里报错，只会原样放进返回值。
- **返回**：返回一个 `dict[str, Any]`，固定包含七个键：`document_id`、`title`、`source`、`tags`、`status`、`chunk_count`、`created_at`。其中前六个中的五个直接取自 `document` 的对应属性，`chunk_count` 取自参数。`tags` 通常是一个标签列表，`created_at` 通常是时间戳或时间字符串，具体类型取决于 `DocumentRecord` 的定义。任何情况下都返回该字典，没有条件分支。
- **内部流程**：函数体同样只有一条 `return` 语句，构造字典字面量。取值顺序为：先 `document.document_id`，再 `document.title`、`document.source`、`document.tags`、`document.status`，接着把参数 `chunk_count` 直接写入，最后 `document.created_at`。没有循环、没有判断、没有调用外部库函数。因为 `chunk_count` 是关键字参数，调用方无法通过位置参数误传（例如把 `chunk_count` 和 `document` 顺序搞反会在调用点直接 `TypeError`）。
- **异常/边界**：`document` 为 `None` 或缺少上述六个属性时抛 `AttributeError`；以位置参数传第二个参数会抛 `TypeError`（因为它是 keyword-only）；`chunk_count` 传入 `None` 不会被拦截，会原样出现在返回字典里（下游 JSON 序列化成 `null`）。函数不做空值兜底、不做类型校验、不捕获异常，也没有超时或 I/O 相关边界。
- **同文件关系**：它不调用本文件里的任何其它函数；也没有被本文件里的其它函数调用（本文件内部是平铺的三个独立函数，没有互相依赖的调用链，只有 `document_detail` → `chunk_summary` 这一条调用关系）。

### `document_detail(document: DocumentRecord, chunks: list[ChunkRecord]) -> dict[str, Any]` （第 46 行）

- **作用**：把一条文档记录连同它的**全部分块**转换成**详情视图**的 JSON 字典形状，是三个函数里信息最完整、字段最多的一个。它对应「查看单篇文档」的场景：`knowledge.document_get` 工具以及文档详情接口需要拿到原文全文、权限、错误信息、创建与更新时间，还需要把该文档的所有分块按顺序一并给出（例如前端做高亮定位、或者 `knowledge.document_revectorize` 判断哪些分块还需要重新向量化）。与 `document_summary` 相比，它多出了 `raw_text`、`permission`、`error`、`updated_at` 四个字段，并且把 `chunks` 明细整个内嵌进来；也正因为它最重，所以绝不能被列表接口复用——这正是本项目把列表视图与详情视图拆成两个函数的原因。它在内部逐条调用 `chunk_summary`，从而保证「详情里嵌的分块形状」和「单独暴露分块时的形状」完全一致，不会出现两份字段清单。
- **参数**：
  - `document`：必填，类型为 `DocumentRecord`。存储层返回的单条文档记录，函数读取它的 `document_id`、`title`、`raw_text`、`source`、`tags`、`permission`、`status`、`error`、`created_at`、`updated_at` 十个属性。不接受 `None`，无默认值。注意 `raw_text` 可能是很长的字符串；`error` 在文档处理成功时通常为 `None` 或空字符串。
  - `chunks`：必填，类型为 `list[ChunkRecord]`，即该文档对应的分块记录列表。函数对它做遍历，顺序即输出顺序，因此调用方（仓储层）应当按 `chunk_index` 升序取出，才能让详情里的分块顺序与原文顺序一致；本函数自己**不做排序**。允许传入空列表（表示文档尚未切分或分块已被清空），此时 `chunks` 字段为 `[]`。不接受 `None`（传 `None` 会在列表推导处抛 `TypeError`）。
- **返回**：返回一个 `dict[str, Any]`，固定包含十个键：`document_id`、`title`、`raw_text`、`source`、`tags`、`permission`、`status`、`error`、`created_at`、`updated_at`，外加第十一个键 `chunks`，其值是 `chunk_summary` 对每个分块生成字典所组成的列表。也就是说返回的是一个「文档元数据 + 原文 + 分块数组」的嵌套结构。任何情况下都返回该字典，没有条件分支；`chunks` 为空列表时依旧返回空列表而不是 `None`。
- **内部流程**：函数体是一条 `return` 语句，按书写顺序逐项取值：先 `document.document_id`、`document.title`、`document.raw_text`、`document.source`、`document.tags`、`document.permission`、`document.status`、`document.error`、`document.created_at`、`document.updated_at`，最后一项 `chunks` 执行列表推导 `[chunk_summary(chunk) for chunk in chunks]`，对传入列表从左到右逐个调用本文件的 `chunk_summary` 并收集结果。因此实际执行顺序是：先完成十个标量字段的属性读取，再遍历分块列表完成嵌套序列化；如果某个分块对象属性缺失，异常会在列表推导进行到该条时抛出，此时整个函数调用失败、不会返回部分结果。没有过滤、没有排序、没有分页、没有去重。
- **异常/边界**：`document` 为 `None` 或缺少上述十个属性之一时抛 `AttributeError`；`chunks` 为 `None` 时抛 `TypeError`（不可迭代）；`chunks` 里某个元素为 `None` 或属性缺失时，由 `chunk_summary` 抛 `AttributeError`。传入空列表是合法的，返回 `"chunks": []`。函数不做空值兜底、不做长度限制、不做异常捕获；`raw_text` 很长时也不截断，序列化压力完全交给调用方。无超时、无 I/O。
- **同文件关系**：它调用了本文件里的 `chunk_summary`（对每个分块逐个调用，是本文件内部唯一的函数间调用）；它没有被本文件里的其它函数调用（它是详情视图的顶层出口，由文件外的三个文档工具与 API 层调用）。它不调用 `document_summary`，两者是并列的两种视图，互不复用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `chunk_summary(chunk: ChunkRecord) -> dict[str, Any]` | 把单条分块记录拍平成含分块 id、序号、字符区间、文本与向量状态的 JSON 字典。 |
| `document_summary(document: DocumentRecord, *, chunk_count: int) -> dict[str, Any]` | 把单条文档记录加上分块数量拍平成不含原文与分块明细的列表视图字典。 |
| `document_detail(document: DocumentRecord, chunks: list[ChunkRecord]) -> dict[str, Any]` | 把单条文档记录连同全部子分块拍平成含原文、权限、错误与分块数组的详情视图字典。 |
