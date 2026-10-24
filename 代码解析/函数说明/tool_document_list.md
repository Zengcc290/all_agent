# tool/document_list.py

## 一、这个文件是干什么的

这个文件实现了一个只读的「文档列表」工具，用来按标签、状态做分页浏览真值源（`documents` / `chunks` 两张表）里的文档。在项目里，向量索引和图索引都只是真值源的投影，所以「库里到底有哪几篇文档、各自处于什么状态、每篇被切成了多少块」是任何检索动作之前最该先问清楚的问题；过去 Agent 只能通过 `GET /api/documents` 这个 Web 端点才能看到，本文件把这段逻辑抽成了 Agent 可直接调用的独立能力。

文件里包含三层东西：第一层是一个纯函数 `list_documents`，它是整段分页列表逻辑的**唯一实现**（`web/app.py` 里原先内联的列表构造与分页校验已被删除，端点只保留 422 错误码映射），它只依赖仓储对象、完全不感知 HTTP；第二层是三个 Pydantic 模型 `DocumentListInput`、`DocumentSummary`、`DocumentListOutput`，分别描述工具的入参、单条文档摘要和整体出参结构；第三层是 `DocumentListTool` 工具类，它把 `spec` 元信息、内存管理器惰性获取和 `execute` 执行逻辑封装起来，并通过模块级 `create_tool()` 供上层注册。

它被用到的典型时机是：Agent 在检索（retrieve）或重新向量化（revectorize）之前，先调用 `knowledge.document_list` 看看库存与状态，再决定下一步调 `knowledge.document_get` 读正文。分页上限口径来自 `constants.WEB_DOCUMENTS_PAGE_SIZE_MAX`，与仓储层的下界校验互补——仓储只管 `page_size >= 1`，上限属于 API 契约，由本文件负责把关。

---

## 二、函数与类逐条详解

### `list_documents(repository: DocumentRepository, *, tag: str = "", status: str = "", page: int = 1, page_size: int = 20) -> dict[str, Any]` （第 32 行）

- **作用**：这是整个文件的核心业务函数，负责把「按条件分页取文档 + 附带每篇文档的分块数」这一件事一次性做完，返回一个可直接被 HTTP 端点或工具层复用的普通字典。它之所以被单独抽成模块级函数而不是写成工具类的方法，是为了让 Web 端点（`web/app.py`）和 Agent 工具（`DocumentListTool.execute`）共用同一份实现，避免两处逻辑漂移。函数对分页参数先做严格校验、再委托仓储查询，最后用 `document_summary` 把仓储返回的原始行对象统一转成摘要字典，并补上分块数。它完全不感知 HTTP，遇到非法入参只抛 `ValueError`，由调用方自行决定映射成什么错误码。当 Agent 需要「先看看库里有哪些文档」时，就是通过工具类间接触发它。
- **参数**：
  - `repository: DocumentRepository`：必填位置参数，文档仓储对象，提供 `list_documents(...)` 与 `chunk_counts()` 两个查询方法；它决定了实际读的是哪个真值源（磁盘 SQLite 还是其它实现）。
  - `tag: str = ""`：关键字参数（`*` 之后只能按关键字传），标签过滤条件，空串表示不按标签过滤；取值范围由仓储层决定，本函数不做长度校验。
  - `status: str = ""`：关键字参数，状态过滤条件（如 `parsed` / `vectorized` / `extracted` / `failed`），空串表示不按状态过滤。
  - `page: int = 1`：关键字参数，页码，期望从 1 开始；本函数**不校验**它（校验交给 Pydantic 模型 `DocumentListInput` 的 `ge=1` 或仓储层），只是原样透传给仓储并原样回填到返回值里。
  - `page_size: int = 20`：关键字参数，每页条数，默认 20；必须是不带布尔值的整数且落在 `1..WEB_DOCUMENTS_PAGE_SIZE_MAX` 闭区间内，否则抛 `ValueError`。
- **返回**：返回 `dict[str, Any]`，固定含四个键：`total`（符合过滤条件的文档总数，整数，注意不是本页条数）、`page`（回显请求页码）、`page_size`（回显请求页大小）、`items`（列表，每个元素是 `document_summary` 产出的摘要字典，且其中的分块数已被填成真实值）。即使某一页没有任何文档，也返回 `items: []` 而不是 `None`。
- **内部流程**：
  1. 先做入参守卫：用 `isinstance(page_size, bool)` 单独拦截布尔值（因为 Python 里 `True`/`False` 也是 `int` 的子类，不拦会把 `True` 当成 1），再判断 `not isinstance(page_size, int)` 与 `not 1 <= page_size <= WEB_DOCUMENTS_PAGE_SIZE_MAX`，三者任一成立就 `raise ValueError(f"page_size 必须是 1 到 {WEB_DOCUMENTS_PAGE_SIZE_MAX} 之间的整数")`。
  2. 调用 `repository.list_documents(tag=tag, status=status, page=page, page_size=page_size)`，用元组解包拿到 `items, total`。
  3. 调用 `repository.chunk_counts()` 拿到一张「文档 ID → 分块数」的计数表，赋给局部变量 `counts`。
  4. 构造返回字典：`total`、`page`、`page_size` 直接回填；`items` 用列表推导逐条调用 `document_summary(item, chunk_count=counts.get(item.document_id, 0))`，其中 `.get(..., 0)` 保证计数表里没有该文档时按 0 处理。
  5. 直接 `return` 这个字典，不做任何二次排序或裁剪。
- **异常/边界**：`page_size` 为布尔值、非整数、小于 1 或超过 `WEB_DOCUMENTS_PAGE_SIZE_MAX` 时抛 `ValueError`（消息里带上上限值，便于调用方直接展示）。`page` 不合法（如 0、负数）本函数不拦，由上游 Pydantic 模型或仓储层处理。`tag`/`status` 为空串时等价于「不过滤」。仓储方法自身抛出的异常（如连接失败、SQL 错误）不被捕获，原样向上传播。计数表缺失某个 `document_id` 时按 0 处理，不报错。不涉及超时处理（超时由工具层的 `timeout_seconds` 统一管理）。
- **同文件关系**：调用了本文件从 `. _documents` 导入的 `document_summary`（用于生成单条摘要）。被本文件的 `DocumentListTool.execute` 调用，是工具执行路径上的唯一业务实现。

---

### `class DocumentListInput(BaseModel)` （第 69 行）

- **作用**：这是工具入参的数据契约模型，继承 Pydantic 的 `BaseModel`，用来把 Agent（或上层调用方）传来的原始参数做一次强类型、强约束的校验与归一化。它存在的意义是把「参数合法性」和「业务查询逻辑」分开：`list_documents` 只关心 `page_size` 的上下限，而这里通过 `Field` 约束把标签长度、状态长度、页码下界、页大小上下界一次性声明清楚，校验失败时由框架统一报错，工具层无需手写重复检查。该模型被挂在 `DocumentListTool.spec` 的 `input_model` 字段上，是工具被框架调用时参数解析的目标类型。它本身不包含任何业务方法，只承载字段与配置。
- **参数**（即模型字段）：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：模型级配置。`extra="forbid"` 表示传入未声明的字段会直接报错，防止调用方拼错参数名被静默忽略；`strict=True` 表示不做类型宽松转换（例如不把字符串 `"20"` 自动转成整数 20）。
  - `tag: str = Field(default="", max_length=100, description="按标签过滤；空串表示不过滤。")`：标签过滤条件，默认空串（不过滤），最长 100 个字符，超出会被校验拒绝。
  - `status: str = Field(default="", max_length=40, description="按状态过滤（parsed/vectorized/extracted/failed 等）；空串表示不过滤。")`：状态过滤条件，默认空串，最长 40 个字符；描述里列举了常见状态取值，但模型本身不做枚举白名单限制。
  - `page: int = Field(default=1, ge=1, description="页码，从 1 开始。")`：页码，默认 1，最小 1（`ge=1`），没有上界；传 0 或负数会被校验拒绝。
  - `page_size: int = Field(default=20, ge=1, le=WEB_DOCUMENTS_PAGE_SIZE_MAX, description=f"每页条目数，上限 {WEB_DOCUMENTS_PAGE_SIZE_MAX}。")`：每页条数，默认 20，下界 1、上界为常量 `WEB_DOCUMENTS_PAGE_SIZE_MAX`，其描述文本通过 f-string 动态带上真实上限值。
- **返回**：类本身不「返回」；它被实例化后得到一个不可随意塞入额外字段的入参对象，供 `DocumentListTool.execute` 以 `arguments.tag`、`arguments.status`、`arguments.page`、`arguments.page_size` 的方式读取。
- **内部流程**：Python 导入本模块时，类体先执行 `model_config` 赋值，再由 Pydantic 的元类收集四个 `Field` 定义并生成校验器；此后每次 `DocumentListInput(...)` 调用都会按 `model_config` 的策略依次校验类型、范围与额外字段。模型内部没有自定义 `__init__`、`field_validator` 或 `model_validator`。
- **异常/边界**：传入多余字段抛 Pydantic 校验错误（因 `extra="forbid"`）；类型不符（因 `strict=True`，如用字符串传数字）抛校验错误；`page < 1`、`page_size` 越界、`tag`/`status` 超长同样抛校验错误。缺省字段会使用各自的默认值，空值 `None` 不会被当成默认值（会被判为类型错误）。没有自定义异常处理。
- **同文件关系**：作为 `DocumentListTool.spec` 的 `input_model` 被工具类引用；`DocumentListTool.execute` 的入参就是它的实例；它列在模块 `__all__` 中对外导出。不调用本文件任何函数。

---

### `class DocumentSummary(BaseModel)` （第 87 行）

- **作用**：这是「单篇文档摘要」的对外数据结构，用来把真值源里的一行文档记录裁剪成检索前最需要的少量字段，避免把正文等大字段带进列表结果。它同时是 `DocumentListOutput.items` 的元素类型，因此也承担着「列表里每条数据长什么样」的文档化职责。它与同文件的 `list_documents` 返回的字典结构一一对应：`DocumentListTool.execute` 会用 `DocumentSummary(**item)` 把每个摘要字典转成该模型实例。模型本身不含业务方法，纯数据载体。
- **参数**（即模型字段）：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：禁止额外字段、禁止宽松类型转换，保证从字典构造时字段名写错会立刻暴露。
  - `document_id: str`：文档唯一标识，必填且无默认值，是列表里唯一的强标识字段。
  - `title: str = ""`：文档标题，默认空串。
  - `source: str = ""`：文档来源（如文件路径或来源标识），默认空串。
  - `tags: list[str] = Field(default_factory=list)`：标签列表，默认空列表；用 `default_factory` 而不是可变默认值，避免多个实例共享同一个列表对象。
  - `status: str = ""`：文档当前处理状态，默认空串。
  - `chunk_count: int = 0`：该文档的分块数量，默认 0；由 `list_documents` 从仓储的计数表填充。
  - `created_at: str = ""`：创建时间，默认空串，以字符串形式保存（不做时间类型解析）。
- **返回**：类不返回值；实例化后提供上述七个只读语义字段，供上层序列化给 Agent 或 HTTP 响应。
- **内部流程**：模块导入时由 Pydantic 元类解析字段定义；实例化时（典型路径是 `DocumentSummary(**item)`，即把 `list_documents` 产出的字典按键展开传入）逐字段校验类型与额外键。类体内没有自定义方法、校验器或计算属性。
- **异常/边界**：缺少 `document_id` 会抛校验错误；传入字典里出现模型未声明的键（例如仓储新增了字段但模型没同步）会因 `extra="forbid"` 抛校验错误，这是刻意的严格策略；`chunk_count` 传 `None` 或字符串会因 `strict=True` 被判为类型错误。无自定义异常处理。
- **同文件关系**：被 `DocumentListOutput.items` 引用为元素类型；被 `DocumentListTool.execute` 实例化；列在模块 `__all__` 中导出。不调用本文件任何函数。

---

### `class DocumentListOutput(BaseModel)` （第 99 行）

- **作用**：这是工具的返回契约模型，描述「一页文档 + 总数」的整体出参形状，让工具框架能对 `execute` 的返回值做统一校验与序列化。它把分页元信息（`total`/`page`/`page_size`）与数据体（`items`）放在同一个对象里，使调用方不必再猜字段名。它也是 `DocumentListTool.spec` 的 `output_model`，因此对外暴露的字段集合由它唯一确定。模型本身不含业务逻辑。
- **参数**（即模型字段）：
  - `model_config = ConfigDict(extra="forbid", strict=True)`：禁止额外字段、禁止宽松转换。
  - `total: int = Field(description="符合过滤条件的文档总数（不是本页条数）。")`：必填，符合过滤条件的文档总数；描述特意强调它不等于本页条数，避免调用方误用。
  - `page: int`：必填，当前页码。
  - `page_size: int`：必填，当前页大小。
  - `items: list[DocumentSummary] = Field(default_factory=list)`：本页文档摘要列表，默认空列表，元素为同文件的 `DocumentSummary` 模型实例；使用 `default_factory` 避免共享可变默认值。
- **返回**：类不返回值；实例化后作为 `execute` 的最终产物被框架消费或序列化。
- **内部流程**：导入时由 Pydantic 解析四个字段（其中 `items` 会递归校验每个元素的 `DocumentSummary` 结构）；`DocumentListTool.execute` 里以关键字参数构造它，把 `list_documents` 返回字典中的四个键逐一搬进来。没有自定义方法与校验器。
- **异常/边界**：缺 `total`/`page`/`page_size` 会抛校验错误；`items` 中某个元素不符合 `DocumentSummary` 结构会抛校验错误；`total` 与 `items` 长度不一致**不会**被校验（两者是独立字段，不做交叉一致性检查）；`items` 省略时默认空列表。无自定义异常处理。
- **同文件关系**：引用同文件的 `DocumentSummary` 作为元素类型；被 `DocumentListTool.execute` 构造并返回，也是 `DocumentListTool.spec` 的 `output_model`；列在模块 `__all__` 中导出。不调用本文件任何函数。

---

### `class DocumentListTool(BaseTool)` （第 108 行）

- **作用**：这是本文件对外暴露的工具实现类，继承框架的 `BaseTool`，把「文档列表」这个能力注册进 Agent 的工具系统。它承载两部分内容：一是类属性 `spec`，用 `ToolSpec` 声明工具的元信息（名称 `knowledge.document_list`、中英文描述、版本 `1.0.0`、入参/出参模型、副作用等级 `read`、权限、超时、幂等与并发安全标记、标签和给模型看的 `guidance` 提示）；二是实例状态与执行逻辑，即内存管理器 `_manager`、惰性获取的 `manager` 属性和真正干活的 `execute` 方法。它被使用时，框架先按 `input_model` 校验参数，再调用 `execute` 拿到 `DocumentListOutput`。类本身只声明元信息与状态，不含额外业务代码。
- **参数**：类没有构造参数之外的参数；其行为由类属性 `spec`（`ToolSpec` 实例，字段包括 `name="knowledge.document_list"`、`description`、`version="1.0.0"`、`input_model=DocumentListInput`、`output_model=DocumentListOutput`、`side_effect="read"`、`permissions=()`、`timeout_seconds=60.0`、`idempotent=True`、`parallel_safe=True`、`tags=("knowledge", "document", "list", "read")`、`guidance`）定义。
- **返回**：类本身不返回；实例化后由框架调用其 `execute` 得到 `DocumentListOutput`。
- **内部流程**：模块导入时先构造类属性 `spec`（此时 `DocumentListInput`/`DocumentListOutput` 已在同模块中定义完毕），随后定义 `__init__`、`manager` 属性与 `execute` 方法。运行时框架读 `spec` 决定如何暴露与校验该工具，并调用 `execute` 完成实际查询。
- **异常/边界**：`spec` 中 `permissions=()` 表示不需要额外权限，`side_effect="read"` 表示只读，`timeout_seconds=60.0` 表示单次调用超时上限 60 秒，`idempotent=True` 与 `parallel_safe=True` 表示可安全重复调用与并发调用。类定义阶段本身不抛异常。
- **同文件关系**：引用同文件的 `DocumentListInput`、`DocumentSummary`、`DocumentListOutput`、`list_documents`；被同文件的 `create_tool()` 实例化；列在模块 `__all__` 中导出。

#### `__init__(self, manager: MemoryManager | None = None) -> None` （第 132 行）

- **作用**：构造函数，只做一件事——把外部可选传入的内存管理器保存到实例私有属性 `self._manager` 上。之所以允许外部注入，是为了测试或特殊场景下替换数据源；之所以把「真正去构建默认管理器」推迟到属性访问时才做，是为了避免导入本模块就触发记忆库初始化（那可能带来较重的 I/O 或依赖装配开销）。它不做任何校验，也不访问仓储，因此构造工具对象本身非常轻量。
- **参数**：
  - `self`：实例本身。
  - `manager: MemoryManager | None = None`：可选的记忆管理器实例；传 `None`（默认）表示稍后由 `manager` 属性惰性构建默认管理器，传具体实例则直接使用它作为数据源入口。
- **返回**：`None`，构造函数的返回值被忽略。
- **内部流程**：单条语句 `self._manager = manager`，把参数原样存到实例属性；没有其它初始化、没有副作用、没有校验分支。
- **异常/边界**：无特殊处理——即使传入类型不合法（比如一个字符串）也不会在此报错，问题会在后续 `manager` 属性或 `execute` 使用它时才暴露。
- **同文件关系**：为同文件的 `manager` 属性和 `DocumentListTool.execute` 准备状态；不调用本文件其它函数。

#### `manager` （property，`def manager(self) -> MemoryManager`，第 135 行）

- **作用**：这是一个只读属性，用来在需要时惰性地拿到可用的 `MemoryManager`。它实现了「默认管理器延迟构建」的策略：第一次访问且外部没有注入管理器时，才在函数内部导入并调用 `build_default_manager()` 并把结果缓存回 `self._manager`，之后每次访问都直接复用同一实例。把 `from ._memory import build_default_manager` 写在函数体内而不是模块顶部，是为了规避模块导入期的循环依赖，同时避免未使用该工具时也付出构建记忆库的代价。它被 `execute` 在解析仓储前调用。
- **参数**：仅 `self`（属性没有显式调用参数）。
- **返回**：`MemoryManager` 实例。若 `self._manager` 已有值（外部注入或此前已构建）则直接返回该对象；否则构建默认管理器、写回 `self._manager` 后返回，保证多次访问返回同一对象。
- **内部流程**：
  1. 判断 `if self._manager is None:`。
  2. 成立时执行函数内导入 `from ._memory import build_default_manager`。
  3. 调用 `build_default_manager()`，把返回值赋给 `self._manager`。
  4. `return self._manager`（无论走哪条分支都返回该属性）。
- **异常/边界**：若 `build_default_manager()` 失败（例如记忆库目录不可用、依赖缺失），异常会原样抛出，属性不会被赋值，下次访问会再次尝试构建。已注入的 `None` 与「未注入」无法区分，都会触发默认构建。无自定义异常处理与超时控制。
- **同文件关系**：被同文件的 `DocumentListTool.execute` 调用（作为 `repository_for` 的入参）；它依赖 `._memory.build_default_manager`（外部模块）与 `__init__` 设置的 `self._manager`。

#### `execute(self, arguments: DocumentListInput) -> DocumentListOutput` （第 143 行）

- **作用**：这是工具的实际执行入口，把经过校验的入参对象翻译成一次真值源查询，并把查询结果包装成输出模型。它先通过 `repository_for(self.manager)` 从记忆管理器里解析出文档仓储，若解析结果为 `None`（说明当前记忆库是 `:memory:` 内存模式，没有持久化的 `documents`/`chunks` 真值源），就抛出 `LookupError` 明确告诉调用方该能力不可用。随后它调用同文件的 `list_documents` 完成真正的分页查询，再逐条把字典转成 `DocumentSummary` 并组装 `DocumentListOutput` 返回。所有 HTTP 语义都被排除在外，它只处理领域层的数据搬运。
- **参数**：
  - `self`：工具实例。
  - `arguments: DocumentListInput`：已经过 Pydantic 校验的入参对象，其 `tag`、`status`、`page`、`page_size` 四个字段分别对应过滤与分页条件；调用方应保证类型正确（框架通常已完成校验）。
- **返回**：`DocumentListOutput` 实例，字段为 `total`（总数）、`page`（页码）、`page_size`（页大小）、`items`（`DocumentSummary` 列表）。正常路径下一定返回该模型，不会返回 `None`。
- **内部流程**：
  1. `repository = repository_for(self.manager)`：先触发 `manager` 属性（必要时惰性构建默认管理器），再解析仓储。
  2. `if repository is None:` 成立时 `raise LookupError("当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源")`。
  3. `page = list_documents(repository, tag=arguments.tag, status=arguments.status, page=arguments.page, page_size=arguments.page_size)`：以关键字方式传参调用核心函数，返回结果字典。
  4. 构造并返回 `DocumentListOutput`：`total`/`page`/`page_size` 从 `page` 字典同名键取值；`items` 用列表推导 `[DocumentSummary(**item) for item in page["items"]]`，把每个摘要字典按键展开成模型实例。
- **异常/边界**：仓储不可用时抛 `LookupError`（不是 `ValueError`，与参数错误的语义区分开）。若 `arguments.page_size` 越过 `WEB_DOCUMENTS_PAGE_SIZE_MAX` 或不是整数，会由 `list_documents` 抛 `ValueError`；但正常流程中该字段已被 `DocumentListInput` 的 `ge`/`le` 约束挡住，所以此异常主要出现在绕过模型直接调用时。`DocumentSummary(**item)` 遇到字典里出现模型未声明的键会抛 Pydantic 校验错误。`page["items"]` 为空时返回空 `items` 列表而不报错。无超时与重试逻辑（超时由 `spec.timeout_seconds=60.0` 在上层统一约束）。
- **同文件关系**：调用了同文件的 `manager` 属性（间接）、`list_documents` 函数、`DocumentSummary` 与 `DocumentListOutput` 模型；被框架在工具调用时执行，被同文件 `create_tool()` 返回的实例所承载。

---

### `create_tool() -> BaseTool` （第 164 行）

- **作用**：这是模块级的工厂函数，用来给上层（工具注册表或插件加载器）提供一个统一、无参的创建入口。它把「如何构造 `DocumentListTool`」这件事封装起来，使注册方不需要知道具体类名与构造细节，也让本模块与其它工具模块保持一致的装配约定（每个工具模块都暴露一个 `create_tool`）。它每次调用都新建一个独立的工具实例，不做缓存或单例复用。
- **参数**：无参数。
- **返回**：`BaseTool` 类型的实例——实际返回的是 `DocumentListTool()`，其 `_manager` 为 `None`（即尚未绑定记忆管理器，将在首次 `execute` 时惰性构建默认管理器）。
- **内部流程**：单条语句 `return DocumentListTool()`，走 `DocumentListTool.__init__` 的默认 `manager=None` 路径，因此函数本身没有任何分支、循环或 I/O。
- **异常/边界**：无特殊处理；除非 `DocumentListTool` 类定义本身出错，否则不会抛异常。反复调用会得到多个互不共享 `_manager` 的实例。
- **同文件关系**：调用同文件的 `DocumentListTool` 构造函数；不调用本文件其它函数，也不被本文件内其它函数调用（供外部模块使用）；列在模块 `__all__` 中导出。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `list_documents` | 校验 `page_size` 后按标签/状态分页查询真值源，并为每条文档补上分块数，返回总数、页码、页大小与摘要列表组成的字典。 |
| `DocumentListInput` | 工具入参的 Pydantic 契约模型，声明 `tag`/`status` 长度上限与 `page`/`page_size` 的取值范围，并禁止多余字段与宽松类型转换。 |
| `DocumentSummary` | 单篇文档摘要的数据模型，承载 `document_id`、标题、来源、标签、状态、分块数与创建时间七个字段。 |
| `DocumentListOutput` | 工具出参的数据模型，把总数、页码、页大小与本页 `DocumentSummary` 列表打包成一个返回对象。 |
| `DocumentListTool` | 文档列表工具的实现类，用 `ToolSpec` 声明只读、幂等、可并发的元信息，并对外提供 `execute` 能力。 |
| `DocumentListTool.__init__` | 构造函数，把可选注入的记忆管理器存到 `self._manager`，不做校验与初始化副作用。 |
| `DocumentListTool.manager` | 只读属性，在首次访问且未注入时惰性构建并缓存默认 `MemoryManager`。 |
| `DocumentListTool.execute` | 解析出文档仓储（内存模式下抛 `LookupError`），调用 `list_documents` 并把结果组装成 `DocumentListOutput` 返回。 |
| `create_tool` | 无参工厂函数，返回一个新的 `DocumentListTool` 实例供上层注册。 |
