# tool/memory_query.py

## 一、这个文件是干什么的

这个文件是 Agent 运行时里「读记忆」这一件小事的完整实现，它把一个内置工具 `memory.query` 定义、打包并注册成运行时可以调用的形态。文件顶部注释说明它是 `memory.add` / `memory.manage` 的只读伙伴：因为把工具声明成 `side_effect="read"`，运行时的「写操作需要用户确认」那道闸门就不会拦住它，于是 Agent 可以在不需要用户批准的情况下检索自己的记忆。

从结构上看，文件里只有四样东西：两个 Pydantic 数据模型（`MemoryQueryInput` 描述入参、`MemoryQueryOutput` 描述出参）、一个继承自 `BaseTool` 的工具类 `MemoryQueryTool`（内含 `spec` 声明、懒加载的 `manager` 属性、真正干活的 `execute`）、以及一个工厂函数 `create_tool`。它对外依赖 `core` 的 `BaseTool`/`ToolSpec` 和 `memory` 的 `MemoryManager`/`MemoryType`，并从同包的 `_memory` 模块复用元数据归一化、默认管理器构造等公共部件，避免和 `memory.add`、`memory.manage` 重复造轮子。

运行时通过工具发现机制拿到 `MemoryQueryTool`（或者调用 `create_tool()`），把它的 `spec` 暴露给模型；模型决定调用后，运行时用 `MemoryQueryInput` 校验参数、调用 `execute`、再用 `MemoryQueryOutput` 收敛返回值。数据源默认是按 `MEMORY_DB_PATH`（或项目旁的 `memory.sqlite3`）打开的 `MemoryManager`，应用如果需要换后端，可以在构造 `MemoryQueryTool` 时注入自己的管理器。工具支持三种动作：跨层搜索（`search`）、按 id 取单条（`get`）、按层列表（`list`），并且通过 `timeout_seconds=10.0`、`idempotent=True`、`parallel_safe=True` 向运行时承诺了自己是快速、幂等、可并发执行的只读操作。

---

## 二、函数与类逐条详解

### `class MemoryQueryInput(BaseModel)` （第 32 行）

- **作用**：这是 `memory.query` 工具的入参模型，用来在模型生成的 JSON 参数真正进入业务逻辑之前做一次严格的形状校验和类型收敛。它把三种动作共用的字段（`action`、`memory_type`、`item_id`、`query`、`metadata`、`limit`）都放在同一个模型里，让运行时只需要一个 schema 就能把工具签名暴露给 LLM。因为工具是只读的检索口，参数种类不多，用一个扁平模型比拆成三个按动作区分的子模型更省事，也更好让模型理解。它同时承担「默认值填充」的职责，例如 `limit` 不传时自动取 10、`memory_type` 不传时留给 `execute` 去决定按层还是全层。另外它挂了一个 `model_validator(mode="before")`，在字段解析之前先做一次元数据形态的归一化，使得调用方传进来的各种「元数据的写法」都能被统一成模型能吃的形状。
- **参数**：本类不是函数，它的字段即参数：
  - `action: Literal["search", "get", "list"]`：必填，无默认值。只允许这三个字面量之一，其它值会被 Pydantic 直接拒绝。它决定 `execute` 走哪条分支。
  - `memory_type: MemoryScope | None`：可选，默认 `None`。类型是 `MemoryScope`（从 `._memory` 导入的枚举/字面量集合，代表记忆层级），含义是「读哪一层记忆」。字段描述里写明：对 `search` 而言留空表示搜索全部四层（历史问答与经验存放在 `episodic` 层）；对其它动作留空则默认成 `working`。
  - `item_id: str | None`：可选，默认 `None`。只在 `action="get"` 时使用，表示要取的那条记忆的 id。
  - `query: str | None`：可选，默认 `None`。只在 `action="search"` 时使用，表示检索文本。
  - `metadata: list[MemoryMetadata] | None`：可选，默认 `None`。一个元数据过滤条件列表，元素类型是 `MemoryMetadata`（同样来自 `._memory`），用于在搜索时按标签/元数据收窄结果。
  - `limit: int`：默认 `10`，带约束 `ge=1, le=100`，即最小 1、最大 100，越界会被校验拒绝。
  - 类级配置 `model_config = ConfigDict(extra="forbid", strict=True)`：`extra="forbid"` 表示出现任何未声明字段就报错（防止模型幻觉出多余参数），`strict=True` 表示不做宽松的类型强转（例如字符串 `"10"` 不会被悄悄当成整数 10）。
- **返回**：本类是数据模型，构造成功时返回一个 `MemoryQueryInput` 实例；构造失败（缺字段、类型不对、越界、多字段）时由 Pydantic 抛出 `ValidationError`，不会返回对象。
- **内部流程**：实例化时 Pydantic 先执行 `mode="before"` 的 `normalize_metadata` 校验器，把原始输入（通常是 dict）先过一遍 `normalize_metadata_payload` 归一化；随后按字段声明逐个解析：`action` 做字面量匹配，`memory_type` 做 `MemoryScope` 类型解析（`None` 直接放行），`item_id`/`query` 允许 `None`，`metadata` 解析成 `MemoryMetadata` 列表，`limit` 走 `ge`/`le` 边界检查；最后应用 `extra="forbid"` 检查是否有未声明键。整个过程中 `strict=True` 生效，不做隐式转换。
- **异常/边界**：会抛 `pydantic.ValidationError`：`action` 不在三个字面量内、`limit` 小于 1 或大于 100、`memory_type` 不是合法 `MemoryScope`、`metadata` 元素形状不对、出现额外字段、类型不严格匹配，都会触发。对空值本身不做业务语义检查——例如 `action="search"` 却不给 `query`，模型层不会拦，留到 `execute` 里抛 `ValueError`；`metadata=None` 是合法输入，表示不带过滤条件。
- **同文件关系**：它被 `MemoryQueryTool.spec` 通过 `input_model=MemoryQueryInput` 引用，被 `MemoryQueryTool.execute` 作为参数类型注解并逐字段读取；它自己调用了同文件 `MemoryQueryInput.normalize_metadata`（由 Pydantic 在校验链中触发），而后者又调用了从 `._memory` 导入的 `normalize_metadata_payload`。它也被 `__all__` 导出。

---

### `MemoryQueryInput.normalize_metadata(cls, value: Any) -> Any` （第 49 行）

- **作用**：这是一个挂在 `MemoryQueryInput` 上的「前置校验器」，职责是在 Pydantic 正式解析字段之前，先把调用方给进来的整份原始输入做一次元数据形态归一化。之所以需要它，是因为 `metadata` 这个字段在实际调用中可能出现多种写法（例如由不同 Agent 或不同版本的提示词生成），如果不先归一化，Pydantic 会因为这些写法差异直接报错，工具就变得很脆。它把归一化的具体逻辑委托给同包 `._memory` 里的 `normalize_metadata_payload`，从而保证 `memory.query` 与 `memory.add`、`memory.manage` 对元数据的理解完全一致，不会出现「同一个 payload 在写入工具里合法、在读取工具里非法」的不一致。它只做搬运，不做业务判断。
- **参数**：
  - `cls`：类方法隐含参数，指向 `MemoryQueryInput` 类本身；本函数体没有使用它。
  - `value: Any`：Pydantic 在 `mode="before"` 阶段传进来的原始输入，通常是 `dict`（模型从 JSON 反序列化时的形态），也可能是任何其它被传入的值（例如已经是模型实例、`None`、或其它类型）。本函数不假设它一定是 dict，直接原样交给 `normalize_metadata_payload` 处理。
- **返回**：返回 `normalize_metadata_payload` 的处理结果，类型标注为 `Any`，实际应当是 Pydantic 后续能继续解析的形态（归一化后的 dict 或等价结构）。如果 `normalize_metadata_payload` 对输入不做改动，则等于原样返回。
- **内部流程**：只有一步——`return normalize_metadata_payload(value)`。装饰器顺序是 `@model_validator(mode="before")` 叠在 `@classmethod` 之上，意味着 Pydantic 在字段解析前把整份原始输入交给它，它返回的结果会替换原始输入继续走后续校验。函数体没有任何分支、循环或局部变量。
- **异常/边界**：自身不主动抛异常，也不会对 `None` 或非法形态做兜底判断；如果传入的值让 `normalize_metadata_payload` 抛错，异常会原样向上冒泡并被 Pydantic 包装成校验失败。边界行为完全由被调用的 `._memory.normalize_metadata_payload` 决定。
- **同文件关系**：它调用的是从 `._memory` 导入的 `normalize_metadata_payload`（不在本文件内定义）；它被 `MemoryQueryInput` 的校验链在实例化时自动调用，属于 `MemoryQueryInput` 类的一部分，本文件内没有其它函数显式调用它。

---

### `class MemoryQueryOutput(BaseModel)` （第 55 行）

- **作用**：这是 `memory.query` 工具的出参模型，用来把 `execute` 的结果收敛成一个形状固定、可被运行时和 LLM 稳定消费的结构。它只有三个字段：做了什么动作、返回了几条、以及条目本身，这样的设计让模型不需要理解 `MemoryItem` 这类内部对象的细节，只要读字典就行。之所以要一个显式的输出模型，是因为工具运行时会用 `output_model` 做结果校验与序列化，固定 schema 能防止内部对象泄漏到工具边界之外，也便于上层做统一的日志和展示。它把条目统一放成 `list[dict[str, Any]]`，是为了兼容不同记忆层返回的不同字段组合。
- **参数**：本类是数据模型，字段即参数：
  - `action: str`：必填，无默认值。回显这次调用执行的是哪个动作（`search`/`get`/`list`），方便调用方把结果和请求对上。
  - `count: int = 0`：默认 `0`。表示 `items` 里实际有多少条，由 `execute` 显式传入 `len(items)`。
  - `items: list[dict[str, Any]]`：默认由 `Field(default_factory=list)` 生成一个空列表（用工厂而不是可变默认值，避免多个实例共享同一个列表）。元素是每条记忆序列化后的字典。
  - 类级配置 `model_config = ConfigDict(extra="forbid", strict=True)`：与入参模型一致，禁止额外字段、禁止宽松类型强转，保证输出形状严格可控。
- **返回**：构造成功返回 `MemoryQueryOutput` 实例；字段不合法（例如 `count` 传了字符串、多传了未声明字段）时抛 `pydantic.ValidationError`。
- **内部流程**：实例化时 Pydantic 按声明顺序解析三个字段：`action` 必须给，`count` 缺省填 0，`items` 缺省调用 `list()` 生成新列表；然后执行 `extra="forbid"` 与 `strict=True` 检查。类本身没有定义任何方法或校验器，全部逻辑来自 Pydantic 的默认行为。
- **异常/边界**：会抛 `pydantic.ValidationError`（字段缺失、类型不严格匹配、出现额外字段）。对空结果没有特殊处理——`count=0`、`items=[]` 就是合法的「没查到」表达，`execute` 在 `get` 未命中时正是这样返回的。
- **同文件关系**：它被 `MemoryQueryTool.spec` 通过 `output_model=MemoryQueryOutput` 引用，并被 `MemoryQueryTool.execute` 在结尾构造并返回；它自己在本文件内不调用任何函数。它也被 `__all__` 导出。

---

### `class MemoryQueryTool(BaseTool)` （第 63 行）

- **作用**：这是整个文件的主体，把「读记忆」的能力封装成一个符合运行时契约的工具。它继承 `core.BaseTool`，通过类属性 `spec` 一次性声明工具的全部元信息，通过 `execute` 实现真正的业务分发。类里刻意让 `MemoryManager` 懒加载：`__init__` 只把注入的管理器存下来，真正的 SQLite 连接推迟到第一次访问 `manager` 属性时才建立，这样工具被发现、被列出、被导入的时候不会产生任何数据库副作用，也不会在没用到记忆的应用里白白打开一个文件。它提供三种读动作，覆盖了「跨层找」「按 id 看」「按层列」这三类最常见的检索需求，并且对外承诺只读、幂等、可并发、10 秒超时，让运行时可以放心地把它放进检索路径而不触发写确认。
- **参数**：本类不是函数。它的类级属性 `spec` 是一个 `ToolSpec`，字段含义为：`name="memory.query"`（工具名，模型调用时使用）、`description`（一句英文说明：搜索/检视/列出记忆里存了什么，只读、绝不修改记忆）、`version="1.0.0"`、`input_model=MemoryQueryInput`、`output_model=MemoryQueryOutput`、`side_effect="read"`（关键：让写确认闸门不拦截）、`permissions=("memory.read",)`（所需权限）、`timeout_seconds=10.0`、`idempotent=True`、`parallel_safe=True`、`tags=("memory", "search", "read")`、`guidance`（一段中文使用指引，明确划清与 `knowledge.hybrid_recall`、`memory.rag_search` 的分工：要找文档分块的语义或关键词命中用前者，要图事实或拼好的上下文块用后者）。
- **返回**：本类是类型定义，实例化后得到可被运行时调用的工具对象。
- **内部流程**：类定义阶段先构造 `ToolSpec` 常量并绑定为类属性；实例化阶段走 `__init__` 保存可选的管理器；被调用时运行时用 `input_model` 校验参数，再调用 `execute`，`execute` 内部按 `action` 分支去访问懒加载的 `manager`，最后用 `output_model` 包装结果。
- **异常/边界**：类本身不抛异常；实例化不做任何 IO，因此即使记忆后端不可用，构造也不会失败（失败会推迟到 `manager` 属性首次被访问时）。`execute` 内部的两处 `ValueError` 与底层 `MemoryManager` 的异常会向外传播，由运行时的工具调用层处理。
- **同文件关系**：它引用同文件的 `MemoryQueryInput`、`MemoryQueryOutput`，调用同文件的 `MemoryQueryTool.__init__`、`MemoryQueryTool.manager`、`MemoryQueryTool.execute`；它被同文件的 `create_tool` 实例化并返回，也被 `__all__` 导出。它使用的 `build_default_manager`、`metadata_dict` 来自 `._memory`，`MemoryType`、`MemoryManager` 来自 `memory`，`ToolSpec`、`BaseTool` 来自 `core`。

---

### `MemoryQueryTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 85 行）

- **作用**：构造函数，负责接收一个可选的记忆管理器并把它存到实例属性 `self._manager` 上，除此之外什么都不做。它刻意不在这里创建默认管理器，代码注释明确写了原因：懒加载可以让「导入工具 / 发现工具」这个动作永远不会打开 SQLite。这在实际运行中很重要，因为工具发现往往发生在应用启动早期，或者发生在根本不需要记忆功能的应用里，此时建立数据库连接既慢又可能因为路径/权限问题直接让启动失败。把创建推迟到真正要读数据的那一刻，就把这类风险限制在了单次调用内部，而不是拖垮整个进程启动。它同时保留了依赖注入的口子：应用可以在构造时传入自己的 `MemoryManager`（例如换成 Postgres 或内存实现），从而完全不碰默认的 SQLite 路径。
- **参数**：
  - `self`：实例本身，方法体只写它的 `_manager` 属性。
  - `manager: MemoryManager | None`：可选，默认 `None`。传入则直接使用该管理器实例（不做类型检查以外的验证）；传 `None` 表示「用默认的」，真正的构造被推迟到 `manager` 属性第一次被读取时，由 `build_default_manager()` 完成（它按 `MEMORY_DB_PATH` 或项目旁的 `memory.sqlite3` 打开）。
- **返回**：无返回值（`None`），只产生副作用——设置 `self._manager`。
- **内部流程**：只有一行有效逻辑：`self._manager = manager`。没有分支、没有循环、没有 IO、没有校验。
- **异常/边界**：无特殊处理。传入任何对象都不会在这里被拒绝（类型注解只是提示，运行时不强制）；`manager=None` 是预期中的正常输入，不是错误。
- **同文件关系**：它被同文件的 `create_tool` 间接触发（`create_tool` 调用 `MemoryQueryTool()`，等价于 `manager=None`）；它设置的 `self._manager` 被同文件的 `manager` 属性读取。它自身不调用本文件里的任何函数。

---

### `MemoryQueryTool.manager` （property，第 89 行）

- **作用**：这是一个只读属性（用 `@property` 装饰），是工具访问记忆后端的唯一入口，同时承担懒加载职责。第一次读取它时，如果 `self._manager` 还是 `None`（说明构造时没有注入管理器），就调用 `build_default_manager()` 现场创建一个默认管理器并缓存回 `self._manager`；之后再读就直接返回缓存，不会重复创建。这样设计的效果是：数据库连接在真正需要时才建立，而且每个工具实例最多只建一次，既避免了启动期的 IO，也避免了每次查询都重新打开 SQLite 的开销。把「取管理器」抽成属性而不是散落在 `execute` 里判断，也让三个分支（search/get/list）共用同一套懒加载逻辑，代码更干净。它返回类型标注为 `MemoryManager`（非 Optional），意味着调用方拿到的永远是一个可用对象。
- **参数**：无显式参数（属性访问形式，隐式 `self`）。
- **返回**：返回 `MemoryManager` 实例。若构造时注入过，则返回那个注入的实例；否则返回 `build_default_manager()` 新建的实例，并将其缓存到 `self._manager`，后续调用返回同一个对象。
- **内部流程**：进入属性 getter → 判断 `self._manager is None` → 为真则执行 `self._manager = build_default_manager()`（这一步会真正打开记忆后端，通常是 SQLite）→ 返回 `self._manager`。没有异常捕获，没有锁；判断与赋值不是原子操作，但工具本身被声明为 `parallel_safe`，且运行时不会在同一实例上并发首次访问以外的场景下产生数据竞争，因此实现选择了最简单的写法。
- **异常/边界**：自身不抛异常，但 `build_default_manager()` 在打开数据库失败时（路径不可写、文件损坏、依赖缺失等）会抛出的异常会原样向上传播，并最终冒泡到工具调用层——也就是说默认后端的初始化失败表现为「本次查询报错」，而不是「工具构造失败」。注入过管理器的实例永远不会走到这一行，因此也不受该失败影响。
- **同文件关系**：它调用的是从 `._memory` 导入的 `build_default_manager`（不在本文件内定义）；它被同文件的 `MemoryQueryTool.execute` 三次使用（`search`、`get`、`list` 三条分支各自通过 `self.manager` 取后端）。它不调用本文件内的其它函数。

---

### `MemoryQueryTool.execute(self, arguments: MemoryQueryInput) -> MemoryQueryOutput` （第 95 行）

- **作用**：这是工具真正干活的方法，也是整个文件唯一的业务逻辑所在。它接收已经被 Pydantic 校验过的 `MemoryQueryInput`，按 `action` 分成三条互斥分支，分别去记忆后端做「跨层搜索」「按 id 取单条」「按层列表」，再把结果统一序列化成字典列表，包进 `MemoryQueryOutput` 返回。它需要处理三个动作共用一个入参模型带来的差异：`search` 必须有 `query`、`get` 必须有 `item_id`，这两条约束 Pydantic 模型层管不了（因为字段是可选的），所以在这里用 `ValueError` 显式补上，让错误信息对模型友好（明确说「search 需要 query」）。它还负责把「层」这个概念的默认值落地：调用方不指定 `memory_type` 时，非搜索动作按 `working` 层处理；而搜索动作则把 `scope`（也就是 `None`）原样传给管理器，由管理器解释为「搜全部四层」。最后它用统一的 `to_dict()` 把内部 `MemoryItem` 对象转成纯字典，保证输出模型能稳定序列化。
- **参数**：
  - `self`：工具实例，用于通过 `self.manager` 取懒加载的管理器。
  - `arguments: MemoryQueryInput`：已经过校验的入参对象，类型即同文件的 `MemoryQueryInput`。运行时保证它是该类型的实例（而非原始 dict），因此方法内直接按属性访问（`arguments.action` 等），不做二次校验。
- **返回**：返回 `MemoryQueryOutput` 实例，字段为 `action`（原样回显输入的动作字符串）、`count`（`len(items)`，即本次实际返回的条数）、`items`（序列化后的字典列表）。三种分支下 `items` 的内容为：`search` 是管理器返回的全部命中项（数量受 `limit` 限制）；`get` 命中时是只含一条的列表，未命中时是空列表（因此 `count` 为 0，而不是报错）；`list` 是该层条目按 `limit` 截断后的列表。
- **内部流程**：
  1. `action = arguments.action`，`scope = arguments.memory_type`，把动作和层取到局部变量。
  2. `memory_type = MemoryType(scope or "working")`：用 `or` 做默认值——`scope` 为 `None` 时退化成字符串 `"working"`，再交给 `MemoryType(...)` 构造出枚举成员。注意这一步在三条分支之前无条件执行，所以即使 `action="search"`（此时 `scope` 可能为 `None`），也会先算出一个 `working` 的 `MemoryType`，只是搜索分支并不使用它。
  3. `items: list[dict[str, Any]] = []`，初始化结果容器。
  4. 若 `action == "search"`：先检查 `arguments.query is None`，是则 `raise ValueError("query is required for search")`；否则调用 `self.manager.search(arguments.query, memory_type=scope, limit=arguments.limit, metadata=metadata_dict(arguments.metadata))`，注意这里传的是原始的 `scope`（可能是 `None`，表示全层搜索）而不是上面算出的 `memory_type`，`metadata` 则通过同包 `._memory` 的 `metadata_dict` 把 `MemoryMetadata` 列表转成管理器要的字典形态；随后用列表推导 `[result.to_dict() for result in results]` 逐条序列化。
  5. 若 `action == "get"`：先检查 `arguments.item_id is None`，是则 `raise ValueError("item_id is required for get")`；否则调用 `self.manager.get(arguments.item_id, memory_type=memory_type)`，得到可能为 `None` 的单条结果；用三元表达式 `[item.to_dict()] if item is not None else []` 把「命中一条」和「未命中」统一成列表。
  6. 否则（`action == "list"`，由模型的 `Literal` 保证只剩这一种可能）：调用 `self.manager.list(memory_type=memory_type)` 拿到该层全部条目，用切片 `values[: arguments.limit]` 按 `limit` 截断后再逐条 `to_dict()`。截断放在 Python 侧而不是交给后端，是因为管理器的 `list` 接口在这个文件里只按 `memory_type` 调用，没有传 limit 参数。
  7. 最后 `return MemoryQueryOutput(action=action, count=len(items), items=items)`，把动作、条数、条目一次性打包。
- **异常/边界**：
  - `ValueError`：`action="search"` 且 `query is None`，或 `action="get"` 且 `item_id is None`，分别抛出带明确文案的 `ValueError`；这是本方法主动抛出的两类异常。
  - `None` 处理：`memory_type` 为 `None` 时按 `working` 兜底（搜索分支例外，`None` 被解释为全层）；`metadata` 为 `None` 时由 `metadata_dict` 负责转成合适的空形态；`get` 未命中时返回空列表而不是抛错。
  - 非法的 `action` 值理论上到不了这里（`MemoryQueryInput` 的 `Literal` 已限制），所以 `else` 分支被安全地当作 `list` 使用。
  - 底层异常：`self.manager` 首次访问时 `build_default_manager()` 可能因数据库不可用抛错；`search`/`get`/`list` 调用本身抛出的异常也都不被捕获，直接向上传播给运行时的工具调用层。
  - 超时：本方法内部没有超时控制，10 秒上限来自 `spec.timeout_seconds`，由运行时在外部施加。
  - 空结果：`search` 无命中或 `list` 层为空时，`items=[]`、`count=0`，属于正常返回。
- **同文件关系**：它通过 `self.manager` 调用同文件的 `MemoryQueryTool.manager` 属性（间接触发 `build_default_manager`），使用同文件的 `MemoryQueryInput` 作为参数类型、`MemoryQueryOutput` 作为返回类型；它调用的 `metadata_dict` 来自 `._memory`，`MemoryType` 来自 `memory`，这两个都不在本文件内定义。它被同文件的 `create_tool` 所创建的工具实例在运行时调用，本文件内没有其它函数直接调用它。

---

### `create_tool() -> BaseTool` （第 121 行）

- **作用**：这是模块级的工厂函数，也是这个文件对外暴露的「创建工具」的标准入口。运行时或插件装载器通常只需要一个无参可调用对象来拿到工具实例，而不关心具体类名，这个函数就提供了这样一层薄封装。它不接收任何配置，因此创建的永远是使用默认懒加载管理器的实例——数据库连接仍然推迟到第一次查询时才建立，所以调用它本身没有任何 IO 开销，可以安全地在模块导入期或工具注册期被调用。它把返回类型标注成基类 `BaseTool` 而不是 `MemoryQueryTool`，是一种有意的抽象：调用方只应依赖工具基类契约，从而在将来替换实现时不必改调用点。由于没有缓存，每次调用都会得到一个全新的工具实例，各自持有独立的 `_manager` 槽位。
- **参数**：无参数。
- **返回**：返回一个 `BaseTool`（实际运行时类型是 `MemoryQueryTool`）实例，其 `_manager` 为 `None`，等待首次访问 `manager` 时懒加载默认记忆管理器。
- **内部流程**：单步执行 `return MemoryQueryTool()`——即用默认参数 `manager=None` 调用同文件 `MemoryQueryTool.__init__`，构造出工具实例后直接返回。没有分支、循环、异常处理或缓存。
- **异常/边界**：无特殊处理。构造函数不做 IO，因此正常情况下不会抛异常；一旦底层默认管理器不可用，错误会推迟到实际查询时才暴露，而不是在这里。
- **同文件关系**：它调用同文件的 `MemoryQueryTool.__init__`（通过 `MemoryQueryTool()`），返回同文件的 `MemoryQueryTool` 实例；它被 `__all__` 导出，本文件内没有其它函数调用它。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `MemoryQueryInput` | `memory.query` 的严格入参模型，声明 action、记忆层、id、查询文本、元数据过滤与 limit，并禁止额外字段。 |
| `MemoryQueryInput.normalize_metadata` | 前置校验器，在字段解析前把原始输入交给 `normalize_metadata_payload` 做元数据形态归一化。 |
| `MemoryQueryOutput` | `memory.query` 的出参模型，固定返回 action、命中条数 count 和字典化的条目列表 items。 |
| `MemoryQueryTool` | 只读记忆查询工具类，通过 `spec` 声明元信息并实现三种读动作，管理器懒加载以避免启动期打开 SQLite。 |
| `MemoryQueryTool.__init__` | 构造函数，只保存可选注入的 `MemoryManager` 到 `self._manager`，不做任何 IO。 |
| `MemoryQueryTool.manager` | 只读属性，首次访问时用 `build_default_manager()` 创建并缓存默认管理器，实现懒加载。 |
| `MemoryQueryTool.execute` | 按 action 分发到 search/get/list，补上 query/item_id 的必填校验，统一序列化并返回 `MemoryQueryOutput`。 |
| `create_tool` | 无参工厂函数，返回一个使用默认懒加载管理器的 `MemoryQueryTool` 实例。 |
