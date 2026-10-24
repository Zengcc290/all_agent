# core/catalog.py

## 一、这个文件是干什么的

这个文件实现了 Agent 运行时里的「工具目录（tool catalog）」能力，也就是让模型在不确定当前有哪些工具可用、或者只知道某个工具的契约还没加载进来的时候，能够先去目录里查一查、把工具的完整契约（输入 schema、输出 schema、权限、超时、幂等性等）取回来，然后再决定怎么调用。文件本身不做任何真正的业务动作，它是一个受限的、只读的门面（facade）：代码注释里明确写了 "it never accepts raw SQL"，也就是说模型只能通过结构化的 action（search / get_spec / resolve）和 intent 文本去检索，而不能拼接任意查询语句。

文件里主要包含三样东西：一是两个 pydantic 数据模型 `CatalogInput` 与 `CatalogOutput`，分别定义这个工具调用的入参契约与出参契约；二是工具本体 `ToolCatalogTool`，它继承自 `core.registry.BaseTool`，类属性 `spec` 里声明了工具名 `system.tool_catalog`、版本 `1.0`、副作用等级 `read`、标签以及一段给模型看的中文使用指引（guidance）；三是这个类内部的若干辅助方法，负责代际（generation）校验、意图分词、评分排序以及把内部 `ToolSpec` 或数据库行转换成对外的字典结构。

它在运行时的定位是「惰性加载（lazy loading）的入口」：当本次请求没有把某个工具的完整契约直接塞进上下文时，模型应该先调 `resolve` 拿到契约；`resolve` 会一次性返回所有匹配的完整契约（放在 `specs` 里），`spec` 字段则只保留第一个，用来兼容那些期望只拿到一个工具的旧调用方。值得注意的是，`resolve` 仅仅是把契约取回来，并不代表工具已经注册或已经获得授权，这一点在 guidance 里被特意强调过。

另外一个重要设计是：目录既可以从内存里的 `ToolRegistry` 取数据，也可以从持久化的 `ToolSpecRepository` 取数据。当某个工具的实现还没有被加载进当前进程时，目录仍然可以返回它的元数据（此时 `registry_generation` 记为 `0`，含义就是 "catalog-only"），这样才支撑得起「先查目录、后加载实现」的懒执行工作流。当 `repository_only=True` 或者注册表里除了目录工具本身之外没有任何可执行工具时，检索会完全走仓库路径；否则走内存注册表路径，并用仓库做一次一致性过滤。

## 二、函数与类逐条详解

### `CatalogInput` （第 13 行）
- **作用**：这是 `system.tool_catalog` 工具的输入契约模型，继承自 pydantic 的 `BaseModel`。它定义了模型调用目录工具时唯一被允许传入的字段集合，并用 `action` 区分三种使用方式：`search`（只列出候选工具的摘要）、`get_spec`（按工具名取某个工具的完整契约）、`resolve`（按能力描述意图一次性取回所有匹配的完整契约）。它存在的意义是把「模型想干什么」收敛成有限枚举加受长度限制的字符串，从而杜绝把任意 SQL 或任意自由文本当成查询语言塞进目录。运行时框架会用这个模型去校验模型产生的工具调用参数，校验失败会在真正执行 `execute` 之前就被拦截。
- **参数**：类本身没有构造参数，它的字段就是它的「参数」。`action`：必填，类型是 `Literal["search", "get_spec", "resolve"]`，只能取这三个字符串之一，取值非法时 pydantic 直接报错。`intent`：可选，类型 `str | None`，默认 `None`，最大长度 500 个字符，用来描述「能力」而不是「用户的具体问题」。`tool_name`：可选，类型 `str | None`，默认 `None`，最大长度 200 个字符，只在 `get_spec` 时真正必需。`version`：可选，类型 `str | None`，默认 `None`，最大长度 32 个字符，用于在 `get_spec` 时指定期望的版本。`limit`：类型 `int`，默认值 `20`，约束是 `ge=1` 且 `le=20`，也就是最少 1 条、最多 20 条，调用方可以显式调小以缩小结果集，但不能超过 20。模型配置是 `extra="forbid"`（传未知字段直接报错）加 `strict=True`（不做宽松类型转换，例如字符串 `"20"` 不会自动转成整数 20）。
- **返回**：作为数据模型，它的「返回」就是它自身的实例；实例上带有上述五个字段，且构造/校验完成后会带上 `validate_action_arguments` 施加的跨字段一致性保证（即 `get_spec` 必有非空 `tool_name`，`resolve` 必有非空 `intent`）。校验不通过时不会返回实例，而是由 pydantic 抛出 `ValidationError`。
- **内部流程**：pydantic 在实例化时先按字段声明逐个做类型与约束校验（枚举取值、长度上限、`limit` 的上下界、禁止额外字段、严格模式），全部通过后进入 `mode="after"` 的模型级校验器 `validate_action_arguments`，由它检查 action 与参数之间的搭配是否成立；该校验器返回 `self`，pydantic 用它作为最终结果。
- **异常/边界**：`action` 不在三个枚举值内、`intent` 超过 500 字符、`tool_name` 超过 200 字符、`version` 超过 32 字符、`limit` 小于 1 或大于 20、传入未声明字段、类型不严格匹配，都会触发 pydantic 的 `ValidationError`。`intent`、`tool_name`、`version` 为 `None` 是合法的，是否「必填」由 action 决定，由模型级校验器负责把关。
- **同文件关系**：它被 `ToolCatalogTool.spec` 通过 `input_model=CatalogInput` 引用，从而与目录工具绑定；`ToolCatalogTool.execute` 用 `isinstance(arguments, CatalogInput)` 做类型检查，并在方法体里再次检查 `tool_name` 与 `intent`（与 `validate_action_arguments` 形成双重保险）。

### `CatalogInput.validate_action_arguments(self) -> CatalogInput` （第 26 行）
- **作用**：这是挂在 `CatalogInput` 上的 pydantic 模型级校验器，装饰器 `@model_validator(mode="after")` 表示它在所有单字段校验都通过之后运行。它解决的是「单字段都合法、但字段之间的组合不合法」的问题：比如只传 `action="get_spec"` 而不给 `tool_name`，单看每个字段都没错，可是这次调用毫无意义，必须在进入执行逻辑之前就拒绝。有了它，`execute` 里就能假定 action 与参数是自洽的，而不必把同样的错误提示散落在业务代码各处。它还会顺带处理「只传了空白字符」这种伪合法输入。
- **参数**：`self`：即当前正在被校验的 `CatalogInput` 实例，此时所有字段已经赋值完成，可以直接读取 `self.action`、`self.tool_name`、`self.intent`。没有其它参数。
- **返回**：返回 `self`，即校验通过后的同一个实例，pydantic 约定 `mode="after"` 的校验器必须返回模型实例（或等价值），否则会把校验结果替换掉。
- **内部流程**：第一步判断 `self.action == "get_spec"` 是否成立，如果成立就用 `(self.tool_name or "").strip()` 取出去掉首尾空白后的工具名，若结果为空字符串（包括 `tool_name` 为 `None` 的情况，因为 `None or ""` 得到空串），抛出 `ValueError("tool_name is required for get_spec")`。第二步判断 `self.action == "resolve"`，同理用 `(self.intent or "").strip()` 判断意图是否为空，为空则抛出 `ValueError("intent is required for resolve")`。第三步，两个分支都没触发异常时返回 `self`。注意 `action == "search"` 不做任何跨字段要求，`search` 允许 `intent` 为空。
- **异常/边界**：抛出 `ValueError`，但在 pydantic 的校验流程里这个异常会被捕获并包装成 `ValidationError`，作为字段级错误呈现给调用方，不会原样冒泡。空值处理是「视为缺失」：`None`、空串、纯空白串（空格、制表符等能被 `strip()` 去掉的字符）都判为不满足要求。`action` 为 `search` 时无任何额外约束。
- **同文件关系**：它只被 pydantic 在校验 `CatalogInput` 时自动调用，文件内部没有显式调用点；它与 `ToolCatalogTool.execute` 开头的重复检查互为补充（`execute` 里检查 `get_spec` 的 `tool_name`，并依赖 `intent or ""` 的写法来容忍 `intent` 为空）。

### `CatalogOutput` （第 34 行）
- **作用**：这是 `system.tool_catalog` 工具的输出契约模型，同样继承 `BaseModel`。它把三种 action 的结果统一到同一个结构里：`candidates` 承载 `search` 的候选工具摘要列表，`spec` 承载单个完整契约（兼容只期望一个结果的旧调用方），`specs` 承载 `resolve`/`get_spec` 返回的全部匹配完整契约。有了它，模型无论调哪种 action 都面对同一个稳定的出参形状，不必根据 action 去猜字段名。它也起到「出参白名单」的作用：`extra="forbid"` 保证目录不会意外把内部字段泄漏出去。
- **参数**：类本身没有构造参数，只有三个字段。`candidates`：`list[dict[str, Any]]`，默认通过 `default_factory=list` 生成空列表，元素是工具摘要字典（包含工具名、描述、版本、schema 哈希、副作用、最大并发、标签、推荐前置工具、注册表代际等）。`spec`：`dict[str, Any] | None`，默认 `None`，存放单个完整契约。`specs`：`list[dict[str, Any]]`，默认空列表，存放本次调用匹配到的所有完整契约。模型配置为 `extra="forbid"` 与 `strict=True`。
- **返回**：作为数据模型，它的返回值是实例本身；三种 action 下实际填充的字段不同——`search` 只填 `candidates`，`resolve` 同时填 `spec`（第一个）与 `specs`（全部），`get_spec` 也同时填 `spec` 与 `specs`（两者内容相同，都是那一个契约）。
- **内部流程**：pydantic 按字段声明做类型校验，`candidates`/`specs` 的元素必须是字典，`spec` 必须是字典或 `None`；严格模式下不做隐式类型转换；校验通过后实例即可被框架序列化回给模型。
- **异常/边界**：字段类型不符、传入未声明字段时抛 `ValidationError`。三个字段全部有默认值，因此 `CatalogOutput()` 这个「什么都不返回」的实例也是合法的，但本文件里的代码从不这样构造。字典内部的具体键由本文件的两个转换方法决定，pydantic 不校验字典内部的键集合。
- **同文件关系**：被 `ToolCatalogTool.spec` 通过 `output_model=CatalogOutput` 引用；`ToolCatalogTool.execute` 的全部返回语句都构造它（`search` 路径返回 `CatalogOutput(candidates=...)`，`resolve`/`get_spec` 路径返回 `CatalogOutput(spec=..., specs=...)`）。

### `ToolCatalogTool` （第 44 行）
- **作用**：这是「工具目录」这个工具本身的实现类，继承自 `core.registry.BaseTool`。它对外暴露的唯一业务入口是 `execute`，内部把 `search`/`get_spec`/`resolve` 三种语义都实现了一遍，并且同时支持「从内存注册表查」和「从持久化仓库查」两条数据来源。它存在的根本理由是支撑惰性加载：模型先通过它发现能力、取回完整契约，再决定是否真正调用那个能力；因为它的副作用等级被声明为 `read`，目录查询本身不会改动任何状态，可以被安全地反复调用。类上还带了一段中文 guidance，明确告诉模型：只有当本次请求没有给出某个工具的完整契约时才用它，`intent` 要写能力描述而不是用户问题内容，已经拿到完整契约的工具就直接调用、不要多此一举地 resolve 一次，并且 resolve 只代表取回契约、不代表已注册或已授权。
- **参数**：作为类，它没有「参数」，但有一个关键的类属性 `spec`，是一个 `ToolSpec` 实例，字段为：`name="system.tool_catalog"`（工具在注册表里的唯一名字）、`description="Find available tools or load complete versioned schemas for one or more matching tools."`、`version="1.0"`、`input_model=CatalogInput`、`output_model=CatalogOutput`、`side_effect="read"`、`tags=("catalog", "discovery")`，以及上面提到的 `guidance` 中文提示文本。实例化的参数见 `__init__`。
- **返回**：类本身不返回值；实例化后得到的是可以被 `ToolRegistry` 注册、被框架按 `spec` 调度、执行时返回 `CatalogOutput` 的工具对象。
- **内部流程**：框架读到 `spec` 后知道这个工具接受 `CatalogInput`、产出 `CatalogOutput`、副作用为只读；调用时框架把模型给出的参数校验成 `CatalogInput`，再调用实例的 `execute`，`execute` 内部根据 action 分流到仓库检索路径或注册表检索路径，并用若干辅助方法完成代际校验、打分排序和结构转换。
- **异常/边界**：类定义本身不会抛异常。需要留意的是，类属性 `spec` 在所有实例之间共享，本文件的代码从不修改它。`__init__` 中并没有调用父类的构造逻辑（文件里看不到 `super().__init__()`），因此对父类初始化有依赖的行为需由 `BaseTool` 自身保证。
- **同文件关系**：它是本文件的核心，包含 `__init__`、`execute`、`_active_generation`、`_has_loaded_tools`、`_stored_generation`、`_search_specs`、`_intent_terms`、`_full_stored_spec`、`_full_spec` 九个方法；这些方法之间互相调用形成完整的目录查询链路。

### `ToolCatalogTool.__init__(self, registry, repository=None, *, repository_only=False) -> None` （第 66 行）
- **作用**：构造目录工具实例，把外部依赖注入进来并做一次防御性类型检查。它需要两个协作对象：`registry` 是内存里的工具注册表，提供 `snapshot()`、`specs()`、`resolve()`、`maybe_resolve()` 等能力；`repository` 是可选的持久化契约仓库，提供 `search()` 与 `get()`。`repository_only` 用来强制检索只走仓库路径，适用于「本进程只做目录服务、不加载任何真实实现」的部署形态。这个方法是整条目录查询链路的前置条件：没有它注入的依赖，`execute` 里的所有分支都无法工作。
- **参数**：`registry`：类型 `ToolRegistry`，必填，位置参数，是内存注册表实例，文件内部不做类型检查，传错对象会在后续调用其方法时出错。`repository`：类型 `ToolSpecRepository | None`，默认 `None`，位置参数，为 `None` 时表示没有持久化仓库，此时 `get_spec` 无法返回未加载实现的元数据，检索也只会走注册表路径。`repository_only`：类型 `bool`，默认 `False`，仅限关键字参数（`*` 之后的参数必须用关键字传入），语义是「只从仓库检索，忽略内存注册表里的工具」。约束是必须严格是 `bool`，传入 `0`、`1`、`"yes"` 之类会被 `isinstance(repository_only, bool)` 判定为非布尔而拒绝。
- **返回**：返回 `None`，构造出的实例把三个值分别保存为同名属性 `self.registry`、`self.repository`、`self.repository_only`，供 `execute` 及其辅助方法读取。
- **内部流程**：第一步把 `registry` 赋给 `self.registry`；第二步把 `repository` 赋给 `self.repository`；第三步用 `isinstance(repository_only, bool)` 检查类型，若不是布尔就抛 `TypeError("repository_only must be a boolean")`；第四步把通过检查的 `repository_only` 赋给 `self.repository_only`。注意检查顺序是先赋值前两个属性再校验第三个参数，因此校验失败时对象已经带着 `registry`/`repository` 处于半初始化状态，只是这个对象不会被返回给调用方。
- **异常/边界**：`repository_only` 不是 `bool` 时抛 `TypeError`。`repository` 为 `None` 是正常合法情况，不是错误；`registry` 为 `None` 也不会在这里报错，但后续 `execute` 调用注册表方法时会抛 `AttributeError`。没有超时、空值或其它特殊处理。
- **同文件关系**：它写入的 `self.registry`、`self.repository`、`self.repository_only` 三个属性被 `execute`、`_active_generation`、`_has_loaded_tools`、`_stored_generation`、`_search_specs` 读取；它本身不调用本文件里的任何函数。

### `ToolCatalogTool.execute(self, arguments, context=None) -> CatalogOutput` （第 79 行）
- **作用**：这是目录工具的执行主体，也是本文件最核心的方法。它按 `arguments.action` 分流：`get_spec` 走「按名字取单个契约」的分支；其余两种 action 先决定检索数据源（仓库优先或注册表优先），再执行检索，`resolve` 返回全部匹配的完整契约，`search` 返回候选摘要列表。它承担了三件关键职责：一是参数与上下文的类型防御；二是把「契约来自仓库还是来自内存」这两条路径的结果统一成同一种输出结构；三是在仓库和注册表之间做一致性校验（schema 哈希比对、版本比对），避免模型拿到已经过期的契约。整个惰性加载工作流能否成立，全靠它把未加载实现的工具元数据也能返回出去。
- **参数**：`arguments`：类型 `CatalogInput`，必填，位置参数，承载 action、intent、tool_name、version、limit；必须是 `CatalogInput` 实例，否则抛 `TypeError`。`context`：类型 `ExecutionContext | None`，默认 `None`，位置参数，执行上下文；为 `None` 时方法内部会新建一个默认的 `ExecutionContext()`，传入非 `ExecutionContext` 且非 `None` 的值会抛 `TypeError`。需要说明的是，在本文件的实现里 `context` 最终只在 `_search_specs` 的 `_context` 形参处被接收，且该形参未被使用，所以它目前主要起接口占位与类型校验作用。
- **返回**：统一返回 `CatalogOutput`。`get_spec` 成功时返回 `CatalogOutput(spec=full_spec, specs=[full_spec])`，其中 `full_spec` 是那个工具的完整契约字典。`resolve` 在仓库路径下返回 `CatalogOutput(spec=specs[0], specs=specs)`，在注册表路径下返回 `CatalogOutput(spec=self._full_spec(spec, ...), specs=[...])`，两种情况下 `spec` 都是列表里的第一个元素。`search` 在仓库路径下返回带 `candidates` 的 `CatalogOutput`（每个候选含 `tool_name`、`description`、`version`、`schema_hash`、`side_effect`、`max_concurrency`、`tags`、`recommended_before_tools`、`registry_generation`），在注册表路径下返回 `{**spec.summary(), "registry_generation": generation}` 组成的候选列表。
- **内部流程**：第一步用 `isinstance(arguments, CatalogInput)` 校验入参类型，不通过抛 `TypeError("arguments must be a CatalogInput instance")`。第二步处理上下文：`context is None` 就新建 `ExecutionContext()`，否则用 `isinstance` 检查，不是 `ExecutionContext` 就抛 `TypeError("context must be an ExecutionContext instance")`。第三步进入 `get_spec` 分支：先确认 `arguments.tool_name` 非空，否则抛 `ValueError`；然后用 `self.registry.maybe_resolve(tool_name)` 尝试解析注册信息，若返回 `None`（实现未加载），则要求 `self.repository` 存在（否则抛 `ValueError`，提示工具未注册），再用 `self.repository.get(tool_name, arguments.version)` 取持久化行，仍取不到就抛 `ValueError`；取到行后用 `self._full_stored_spec(stored, 0)` 转成完整契约并直接返回（代际固定为 0，表示 catalog-only）。若 `maybe_resolve` 成功，解包出 `tool, generation`；若调用方指定了 `version` 且与当前活跃版本不一致，抛 `ValueError("requested tool version is not active")`；若存在仓库，则用 `self.repository.get(tool.spec.name, tool.spec.version)` 取行并比对 `schema_hash`，取不到或不相等就抛 `ValueError("active tool metadata is not synchronized")`；最后用 `self._full_spec(tool.spec, generation)` 生成完整契约返回。第四步处理非 `get_spec` 分支的数据源选择：条件是 `self.repository is not None` 且（`self.repository_only` 为真 或 `self._has_loaded_tools()` 为假），即「强制走仓库」或「注册表里除目录工具外没有别的可执行工具」；进入该分支后调用 `self.repository.search(arguments.intent or "", arguments.limit)` 得到 `selected_records`；若是 `resolve` 且结果非空，用列表推导对每条记录调用 `self._full_stored_spec(record, self._stored_generation(record))` 得到 `specs`，返回 `CatalogOutput(spec=specs[0], specs=specs)`；若是 `resolve` 但结果为空，抛 `ValueError("no matching tool found")`；否则（`search`）用列表推导把每条记录映射成候选字典返回，其中 `tags`、`recommended_before_tools` 用 `list(...)` 拷贝成新列表，`registry_generation` 由 `self._stored_generation(item)` 得出。第五步走注册表路径：调用 `self._search_specs(arguments.intent or "", arguments.limit, context)` 得到 `selected`；若是 `resolve` 且非空，对每个 spec 调用 `self._full_spec(spec, self._active_generation(spec))` 生成 `specs` 并返回 `spec=specs[0]`；若是 `resolve` 但为空，抛 `ValueError("no matching tool found")`；否则遍历 `selected`，对每个 spec 调用 `self._active_generation(spec)` 取代际，用 `{**spec.summary(), "registry_generation": generation}` 组装候选并返回 `CatalogOutput(candidates=candidates)`。
- **异常/边界**：`arguments` 不是 `CatalogInput` 抛 `TypeError`；`context` 非 `None` 且类型不对抛 `TypeError`；`get_spec` 缺少 `tool_name` 抛 `ValueError`；`get_spec` 在注册表和仓库里都找不到工具抛 `ValueError`（提示 `tool '<名字>' is not registered`）；`get_spec` 指定了与活跃版本不符的 `version` 抛 `ValueError`；仓库里的 `schema_hash` 与活跃工具不一致抛 `ValueError("active tool metadata is not synchronized")`；`resolve` 没有任何匹配时抛 `ValueError("no matching tool found")`。空值方面，`arguments.intent` 为 `None` 时统一用 `arguments.intent or ""` 兜成空串再传给检索方法，不会因此崩溃；`repository` 为 `None` 时 `get_spec` 只依赖注册表，检索则跳过仓库分支。本方法没有捕获任何异常，也没有超时控制，所有错误都以异常形式向上抛给框架。
- **同文件关系**：它调用了本文件的 `self._has_loaded_tools()`、`self._full_stored_spec()`、`self._stored_generation()`、`self._search_specs()`、`self._full_spec()`、`self._active_generation()`，并构造 `CatalogInput` 之外的 `ExecutionContext`、`CatalogOutput`；它不调用 `CatalogInput.validate_action_arguments`（那个由 pydantic 自动触发）。它自身是被框架/注册表在模型发起 `system.tool_catalog` 调用时调用的入口，文件内部没有其它函数调用它。

### `ToolCatalogTool._active_generation(self, spec) -> int` （第 170 行）
- **作用**：把「一个内存里的 `ToolSpec`」换算成它在注册表里的活跃代际号，并顺带做一次并发安全校验。代际号是注册表用来标识「工具集合第几次变更」的版本计数，模型拿到候选或契约时可以据此判断自己手上的信息是否还新鲜。这个方法的关键价值在于第二次解析：调用方可能是在遍历一个较早时刻取得的 spec 列表，如果在这期间注册表被替换过，同一个名字下的 spec 对象就会变，此时继续返回旧代际号会让模型误以为信息仍然有效，所以必须报错而不是静默返回。
- **参数**：`self`：当前工具实例，用于访问 `self.registry`。`spec`：类型 `ToolSpec`，必填，是要查询代际的目标工具契约对象；它必须来自本进程注册表（或者与注册表中的对象相等）。
- **返回**：返回 `int`，即 `self.registry.resolve(spec.name)` 解包出的第二个元素 `generation`，代表该工具当前所处的注册表代际。
- **内部流程**：第一步调用 `self.registry.resolve(spec.name)`，这一步会在注册表里按名字做一次强制解析，返回 `(tool, generation)` 元组；如果名字不存在，由 `resolve` 自己抛异常（本方法不做兜底）。第二步用 `tool.spec != spec` 比较解析出来的契约与传入的契约，若两者不相等（说明在这次请求执行期间目录内容发生了变化），抛 `ValueError("tool catalog changed while the request was running")`。第三步返回 `generation`。
- **异常/边界**：目录在请求期间发生变化时抛 `ValueError`；工具名在注册表中不存在时由 `self.registry.resolve` 抛异常（本文件看不到该实现，因此不对其异常类型做假设）；没有对 `spec` 为 `None` 或类型错误做检查，传错对象会在 `spec.name` 处抛 `AttributeError`。无超时处理。
- **同文件关系**：它调用外部的 `self.registry.resolve`；被 `ToolCatalogTool.execute` 在注册表检索路径下调用（对 `resolve` 的每个匹配项、以及 `search` 的每个候选各调用一次）。

### `ToolCatalogTool._has_loaded_tools(self) -> bool` （第 176 行）
- **作用**：判断当前注册表里除了目录工具自己之外，是否还存在至少一个可执行的工具。这个判断决定了 `execute` 在非 `get_spec` 分支里走哪条数据源：如果注册表是空的（只有目录工具），说明本进程并没有加载任何真实实现，此时应当完全依赖持久化仓库来提供目录信息；反之说明内存里有真实工具，可以用注册表作为主数据源。它让「纯目录服务」和「完整运行时」两种部署形态共用同一份代码而无需额外配置。
- **参数**：`self`：当前工具实例，用于访问 `self.registry` 和类属性 `self.spec.name`。无其它参数。
- **返回**：返回 `bool`。只要 `self.registry.snapshot()` 返回的名字序列里存在任何一个不等于 `"system.tool_catalog"` 的名字，就返回 `True`；否则（包括注册表为空、或注册表里只有目录工具自己）返回 `False`。
- **内部流程**：调用 `self.registry.snapshot()` 取出当前注册表里所有工具名的快照，用生成器表达式 `name != self.spec.name for name in ...` 逐项比较，交给内置函数 `any()` 短路求值：一旦遇到第一个非目录工具就立即返回 `True`，遍历完都没有则返回 `False`。
- **异常/边界**：`snapshot()` 本身抛出的异常不会被捕获，会向上传播；注册表为空时安全返回 `False`；无空值或超时特殊处理。注意它比较的是类属性 `self.spec.name`（也就是 `"system.tool_catalog"`），而不是实例属性，因此子类若覆写 `spec` 会以子类的名字为准。
- **同文件关系**：它调用外部的 `self.registry.snapshot()`；被 `ToolCatalogTool.execute` 在决定是否进入仓库检索分支时调用一次（与 `self.repository_only` 做「或」判断）。

### `ToolCatalogTool._stored_generation(self, stored) -> int` （第 183 行）
- **作用**：为一条来自持久化仓库的记录计算「当前真实的注册表代际号」，并在实现尚未加载时明确返回 `0` 表示 catalog-only。它解决的是「仓库里有元数据、内存里可能还没有实现」这个中间状态的表达问题：如果某个工具的实现已经被懒加载进本进程，而且仓库里记录的版本与 schema 哈希都和内存里的一致，就返回真实代际；否则返回 `0`，提示调用方「这条信息只是目录元数据，不代表已经注册可执行」。这样模型就不会把一条仅存在于数据库里的元数据误当成已就绪的工具。
- **参数**：`self`：当前工具实例，用于访问 `self.registry`。`stored`：类型 `dict[str, Any]`，必填，是一条仓库记录（或与仓库记录同构的字典），本方法会读取其中的 `"tool_name"`、`"version"`、`"schema_hash"` 三个键；若缺少这些键会抛 `KeyError`。
- **返回**：返回 `int`。当 `self.registry.maybe_resolve(stored["tool_name"])` 返回 `None` 时返回 `0`；当解析出的工具版本 `tool.spec.version` 与 `stored["version"]` 不一致、或 `tool.spec.schema_hash` 与 `stored["schema_hash"]` 不一致时也返回 `0`；只有名字、版本、schema 哈希三者全部吻合时才返回真实的 `generation`。
- **内部流程**：第一步调用 `self.registry.maybe_resolve(stored["tool_name"])` 做宽松解析，拿不到注册信息直接返回 `0`。第二步解包 `registration` 为 `tool, generation`。第三步用一个 `if` 判断两个不等条件（版本不等 或 schema 哈希不等），任一成立返回 `0`。第四步返回 `generation`。整体是一个「先宽松探测、再严格比对、不匹配就降级为 0」的模式，从不抛业务异常。
- **异常/边界**：`stored` 缺少 `"tool_name"`、`"version"` 或 `"schema_hash"` 键时抛 `KeyError`；`stored` 不是字典时抛 `TypeError`；`self.registry` 为 `None` 时抛 `AttributeError`。工具未加载、版本不符、哈希不符这三种情况都不抛异常，而是统一返回 `0` 作为降级信号。无超时处理。
- **同文件关系**：它调用外部的 `self.registry.maybe_resolve`；被 `ToolCatalogTool.execute` 在仓库检索路径下调用——`resolve` 时作为 `self._full_stored_spec(record, self._stored_generation(record))` 的第二个实参，`search` 时作为候选字典里 `registry_generation` 的取值来源。

### `ToolCatalogTool._search_specs(self, intent, limit, _context) -> list[ToolSpec]` （第 200 行）
- **作用**：在内存注册表的工具集合上做一次确定性的关键词检索与排序，返回最匹配的若干 `ToolSpec`。它是注册表检索路径的核心算法：先取当前所有工具契约，再（在存在仓库时）用仓库做一次一致性过滤，剔除那些元数据已经过期或根本不在仓库里的工具，避免模型看到与持久化状态脱节的契约；然后对每个契约在「工具名 + 描述 + 标签」组成的文本上统计命中词数作为分数，按分数降序、同分按工具名升序排序，最后截断到 `limit` 条。用确定性排序而不是模糊相似度，是为了让同样的 intent 始终得到同样的结果，便于复现与排查。
- **参数**：`self`：当前工具实例，用于访问 `self.registry` 与 `self.repository`。`intent`：类型 `str`，必填，能力描述文本；调用方（`execute`）已用 `arguments.intent or ""` 保证它是字符串，因此这里不会收到 `None`。`limit`：类型 `int`，必填，返回条数上限，来自 `CatalogInput.limit`，取值范围 1 到 20。`_context`：类型 `ExecutionContext`，必填，执行上下文；下划线前缀表示它在当前实现里被有意忽略（未被读取），保留它是为了将来可能按上下文做权限或可见性过滤。
- **返回**：返回 `list[ToolSpec]`。当 `intent` 分词后得到非空词集合时，只返回分数大于 0 的契约（按分数从高到低、同分按名字字典序）；当词集合为空（例如 intent 为空串、或只包含标点等无法切出词的内容）时，所有契约的分数都是 0，此时不做过筛，直接把全部契约按名字升序返回。无论哪种情况，最终都截断到最多 `limit` 个元素。如果注册表为空、或者仓库过滤后没有剩下任何契约，返回空列表。
- **内部流程**：第一步用字典推导 `{spec.name: spec for spec in self.registry.specs()}` 建立名字到契约的映射 `active`。第二步判断 `self.repository is not None`，若成立则用字典推导重建 `active`：对每个条目调用 `self.repository.get(name, spec.version)`（用海象运算符 `:=` 把结果绑定到 `stored`），只保留 `stored` 非 `None` 且 `stored["schema_hash"] == spec.schema_hash` 的条目，这一步把仓库与内存不一致的工具全部剔除。第三步调用 `self._intent_terms(intent)` 得到词集合 `terms`。第四步初始化空列表 `ranked`，遍历 `active.values()`：把 `spec.name`、`spec.description`、展开后的 `spec.tags` 用空格拼成 `haystack` 并整体 `casefold()` 统一大小写；计算 `score`，当 `terms` 非空时用生成器求和统计每个词是否作为子串出现在 `haystack` 中，当 `terms` 为空时直接把 `score` 记为 0；把 `(score, spec)` 追加进 `ranked`。第五步用 `ranked.sort(key=lambda item: (-item[0], item[1].name))` 排序，负数分数实现降序，名字实现同分升序。第六步用列表推导返回 `[spec for score, spec in ranked if not terms or score > 0][:limit]`，即词集合为空时不过滤，否则只留分数大于 0 的，最后切片取前 `limit` 个。
- **异常/边界**：注册表或仓库方法自身抛出的异常不被捕获；`self.registry.specs()` 返回空时安全返回空列表；`intent` 为空串时按「不过滤、全量返回（受 limit 限制）」处理；`limit` 由上层模型约束在 1 到 20 之间，本方法自身不校验，若传入 `0` 会因切片 `[:0]` 返回空列表，传入负数会得到除尾部外的几乎所有元素（本文件不会出现这种调用）。无超时处理。
- **同文件关系**：它调用本文件的 `self._intent_terms(intent)` 做分词，调用外部的 `self.registry.specs()` 与 `self.repository.get()`；被 `ToolCatalogTool.execute` 在注册表检索路径下调用一次。

### `ToolCatalogTool._intent_terms(intent) -> set[str]` （第 225 行）
- **作用**：把一个可能中英混杂的能力描述切成用于检索的词集合，并额外做一层中英别名扩展。它的存在有非常具体的工程理由，代码注释里写得很清楚：模型经常用用户的语言（中文）描述能力，而工具契约是用英文写的；如果只按 ASCII 空格切词，一整句中文会变成一个巨大的 token，导致每次检索都零命中，模型就会反复重试同一个 catalog 调用，陷入空转。因此这个方法一方面用正则抽出英文/数字/下划线/连字符构成的词，另一方面通过一张固定的中文到英文能力词别名表，把「日志」「读取」「查询」「工具」这类中文能力词映射成 `log`、`read`、`search`、`tool` 等英文词。作者特意强调它保持确定性且刻意做得很小：别名描述的是「能力词」，不是用户数据，也不是任意 SQL，所以不存在注入面。
- **参数**：`intent`：类型 `str`，必填，能力描述文本，例如 `"web search"` 或 `"读取文件"`。它是静态方法（`@staticmethod`），因此没有 `self` 参数，也不依赖任何实例状态。调用方 `_search_specs` 保证传入的是字符串（上游已用 `or ""` 兜底）。
- **返回**：返回 `set[str]`，即去重后的检索词集合。集合里既包含从原文抽出的英文/数字/下划线/连字符词，也包含因命中中文别名而补充进来的英文扩展词。如果 `intent` 是空串或只含无法匹配的字符，返回空集合。
- **内部流程**：第一步 `intent.casefold()` 做大小写归一化（比 `lower()` 更彻底），紧接着 `.replace(".", " ")` 把英文句点替换成空格，这样 `system.tool_catalog` 这类带点的名字会被拆成可检索的片段。第二步用 `re.findall(r"[a-z0-9_-]+", normalized)` 抽出所有由小写字母、数字、下划线、连字符组成的连续片段，装进集合 `terms`（集合天然去重）。第三步定义字典 `aliases`，键是中文能力词，值是英文扩展词元组，共十三组：`日志`→`("log", "update-log", "audit")`、`记录`→`("log", "update-log", "audit")`、`读取`→`("read", "retrieve", "get")`、`查看`→`("read", "retrieve", "get")`、`查询`→`("read", "retrieve", "search", "get")`、`参数`→`("schema", "spec", "catalog")`、`工具`→`("tool",)`、`列表`→`("list", "search", "catalog")`、`全部`→`("all", "bulk")`、`时间`→`("time", "current")`、`搜索`→`("search",)`、`网页`→`("web", "search")`、`记忆`→`("memory", "rag")`。第四步遍历 `aliases.items()`，用 `if chinese in normalized` 做子串判断（注意是在归一化后的整句里做包含判断，所以「查看日志」会同时命中 `查看` 和 `日志` 两组），命中就用 `terms.update(expansions)` 把扩展词并入集合。第五步返回 `terms`。
- **异常/边界**：`intent` 必须是字符串，传入 `None` 会在 `.casefold()` 处抛 `AttributeError`（上游已保证不会发生）；空串、纯标点、纯 emoji 等切不出词也不会报错，只是返回空集合或只含别名扩展词。正则里不包含中文字符类，因此中文本身不会作为词进入集合，中文只通过别名表起作用；别名表是硬编码的，不覆盖的中文词（例如「翻译」「发送邮件」）不会产生扩展词。无超时处理。
- **同文件关系**：它是静态方法，不调用本文件里任何其它函数（只使用标准库 `re`）；被 `ToolCatalogTool._search_specs` 调用一次，其返回值决定后者是否做分数过滤以及每个 spec 的得分。

### `ToolCatalogTool._full_stored_spec(stored, generation) -> dict[str, Any]` （第 259 行）
- **作用**：把一条持久化仓库记录（字典）转换成与 `_full_spec` 完全同构的完整契约字典。它存在的意义是「让两条数据源的输出长得一模一样」：无论契约是从内存注册表来的还是从数据库行来的，调用方（以及模型）都看到同一组键、同样的嵌套结构，不需要为来源不同写两套解析逻辑。它同时承担了「字段拷贝」职责，把仓库行里的 `tags`、`recommended_before_tools`、`permissions` 用 `list(...)` 复制成新列表，避免调用方意外改动仓库缓存里的原始容器。
- **参数**：`stored`：类型 `dict[str, Any]`，必填，一条仓库记录，本方法会读取它的 `tool_name`、`description`、`version`、`schema_hash`、`side_effect`、`max_concurrency`、`tags`、`recommended_before_tools`、`input_schema`、`output_schema`、`permissions`、`timeout_seconds`、`idempotent`、`parallel_safe` 共十四个键；缺任何一个都会抛 `KeyError`。`generation`：类型 `int`，必填，要写进结果的注册表代际号；调用方在 `get_spec` 的未加载分支传 `0`，在检索分支传 `self._stored_generation(record)` 的结果。它是静态方法（`@staticmethod`），没有 `self` 参数。
- **返回**：返回 `dict[str, Any]`，键依次为 `tool_name`、`description`、`version`、`schema_hash`、`side_effect`、`max_concurrency`、`tags`（新列表）、`recommended_before_tools`（新列表）、`registry_generation`（即传入的 `generation`）、`input_schema`、`output_schema`、`permissions`（新列表）、`timeout_seconds`、`idempotent`、`parallel_safe`。与 `_full_spec` 相比，键集合一致，只是 `max_concurrency` 在这里出现在列表的中间位置，而 `_full_spec` 用字典展开后把它放在末尾，字典本身不关心顺序，因此两者结构等价。
- **内部流程**：整个方法体就是一条 `return` 语句，直接构造字面量字典：前八个键中 `tool_name`、`description`、`version`、`schema_hash`、`side_effect`、`max_concurrency` 原样取值，`tags` 与 `recommended_before_tools` 用 `list(...)` 拷贝；第九个键 `registry_generation` 用传入的 `generation`；后面 `input_schema`、`output_schema` 原样取值，`permissions` 用 `list(...)` 拷贝，`timeout_seconds`、`idempotent`、`parallel_safe` 原样取值。没有任何条件分支、循环或异常捕获。
- **异常/边界**：`stored` 缺少上述任一键时抛 `KeyError`；`stored` 不是支持下标访问的对象时抛 `TypeError`；`tags`、`recommended_before_tools`、`permissions` 若为 `None`，`list(None)` 会抛 `TypeError`（本文件假定仓库行里这几个字段一定是可迭代的）。没有空值兜底，没有超时处理。
- **同文件关系**：它是静态方法，不调用本文件里任何其它函数；被 `ToolCatalogTool.execute` 调用两次——`get_spec` 的未加载分支里 `self._full_stored_spec(stored, 0)`，以及仓库检索分支里对 `resolve` 的每条记录 `self._full_stored_spec(record, self._stored_generation(record))`。它与 `_full_spec` 是一对镜像方法，输出结构必须保持同步。

### `ToolCatalogTool._full_spec(spec, generation) -> dict[str, Any]` （第 281 行）
- **作用**：把内存里的 `ToolSpec` 对象展开成完整契约字典，作为对外（给模型或上层调用方）的最终交付形态。`ToolSpec` 是内部对象，字段是属性；模型需要的是可 JSON 序列化的普通字典，所以必须有一个方法把属性摊平成键值对，并且补上 `registry_generation` 这个运行时信息（`ToolSpec` 自身不带代际概念，代际是注册表层面的状态）。它同时负责把 `permissions` 拷贝成新列表，并显式带上 `max_concurrency`，确保与 `_full_stored_spec` 的输出键集合一致，这样调用方不必关心契约来自哪条路径。
- **参数**：`spec`：类型 `ToolSpec`，必填，内存中的工具契约对象，方法会读取它的 `summary()` 方法以及 `input_schema`、`output_schema`、`permissions`、`timeout_seconds`、`idempotent`、`parallel_safe`、`max_concurrency` 等属性。`generation`：类型 `int`，必填，要写进 `registry_generation` 的代际号，由调用方通过 `self._active_generation(spec)`（注册表检索路径）或 `self.registry.maybe_resolve` 解包出的代际（`get_spec` 已加载分支）提供。它是静态方法（`@staticmethod`），没有 `self` 参数。
- **返回**：返回 `dict[str, Any]`。字典由两部分合并而成：先展开 `spec.summary()` 返回的所有键（按 `execute` 中 `search` 路径的用法可知，`summary()` 至少包含 `tool_name`、`description`、`version`、`schema_hash`、`side_effect`、`tags`、`recommended_before_tools` 等摘要字段），然后用字面量覆盖/追加 `registry_generation`、`input_schema`、`output_schema`、`permissions`（新列表）、`timeout_seconds`、`idempotent`、`parallel_safe`、`max_concurrency`。如果 `summary()` 里本来就有同名键，字面量里的值会覆盖它（Python 字典展开的后者优先规则）。
- **内部流程**：方法体是一条 `return` 语句，先写 `**spec.summary()` 把摘要字典整体展开，再依次列出八个显式键：`registry_generation` 取传入的 `generation`；`input_schema`、`output_schema` 取属性原值；`permissions` 用 `list(spec.permissions)` 拷贝；`timeout_seconds`、`idempotent`、`parallel_safe`、`max_concurrency` 取属性原值。没有条件分支、循环或异常捕获。
- **异常/边界**：`spec` 没有 `summary()` 方法或缺少上述属性时抛 `AttributeError`；`spec.summary()` 返回的不是映射类型时展开会抛 `TypeError`；`spec.permissions` 为 `None` 时 `list(None)` 抛 `TypeError`。无空值兜底，无超时处理。
- **同文件关系**：它是静态方法，只调用外部 `ToolSpec.summary()`，不调用本文件里任何其它函数；被 `ToolCatalogTool.execute` 调用——`get_spec` 的已加载分支 `self._full_spec(tool.spec, generation)`，以及注册表检索路径下对每个匹配 spec 调用 `self._full_spec(spec, self._active_generation(spec))`。它与 `_full_stored_spec` 互为镜像，两者的输出键集合必须保持一致。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `CatalogInput` | 目录工具的输入契约模型，用受限的 action 枚举加长度受限的 intent/tool_name/version/limit 描述一次目录查询请求。 |
| `CatalogInput.validate_action_arguments` | 模型级校验器，强制 `get_spec` 必须给出非空 `tool_name`、`resolve` 必须给出非空 `intent`，否则抛错。 |
| `CatalogOutput` | 目录工具的输出契约模型，用 `candidates` 装检索摘要、`spec` 装单个完整契约、`specs` 装全部匹配契约。 |
| `ToolCatalogTool` | 只读的工具目录门面，声明 `system.tool_catalog` 契约与中文使用指引，支撑工具契约的惰性发现与加载。 |
| `ToolCatalogTool.__init__` | 注入内存注册表与可选持久化仓库，并校验 `repository_only` 必须是布尔值。 |
| `ToolCatalogTool.execute` | 按 action 分流，从仓库或注册表取回候选摘要与完整契约，并做版本与 schema 哈希一致性校验。 |
| `ToolCatalogTool._active_generation` | 把内存契约换算成当前注册表代际，若请求期间目录发生变化则报错。 |
| `ToolCatalogTool._has_loaded_tools` | 判断注册表里除目录工具外是否还有可执行工具，用于选择检索数据源。 |
| `ToolCatalogTool._stored_generation` | 为仓库记录计算真实代际，实现未加载或版本/哈希不符时统一降级为 `0`（catalog-only）。 |
| `ToolCatalogTool._search_specs` | 在注册表契约集合上按意图词打分排序并截断，同时用仓库过滤掉不一致的契约。 |
| `ToolCatalogTool._intent_terms` | 把中英混杂的能力描述切成检索词，并用固定别名表把中文能力词扩展成英文词。 |
| `ToolCatalogTool._full_stored_spec` | 把一条仓库记录转换成与 `_full_spec` 同构的完整契约字典。 |
| `ToolCatalogTool._full_spec` | 把内存里的 `ToolSpec` 展开成带代际号的完整契约字典。 |
