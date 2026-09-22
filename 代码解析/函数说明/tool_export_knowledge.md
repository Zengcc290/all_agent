# tool/export_knowledge.py

## 一、这个文件是干什么的

这个文件是「知识库导出」这一能力的唯一实现，位于 `tool/` 工具目录下，职责是把四层记忆系统里的记忆条目序列化成一份**可移植、可自描述、只读**的 JSON 载荷。所谓可移植，是指载荷里只允许出现纯 JSON 类型（字符串、数字、布尔、列表、字典），不能混入 Python 特有的对象或自定义类型，这样才能交给外部工具消费或者写盘备份；所谓可自描述，是指载荷里必须带上 `format` 版本号和 `counts` 计数，让导入方在真正写库之前就能先校验兼容性和规模；所谓只读，是指整个过程绝不修改记忆库，因此 Agent 可以在任意时刻安全地调用它（例如用户说「把知识库导出来看看」）。

文件内部分成三层结构。最底层是两个纯函数 `export_payload` 与 `export_filename`，前者负责真正的序列化与计数，后者负责生成带时间戳的下载文件名。中间层是三个 Pydantic 模型 `ExportKnowledgeInput`、`ExportCounts`、`ExportKnowledgeOutput`，分别定义工具的入参、计数子结构、出参结构，并通过 `strict=True` 与 `extra="forbid"` 把接口收得很紧。最上层是工具类 `ExportKnowledgeTool`，它继承 `core.BaseTool`，用 `ToolSpec` 声明工具元信息（名字 `knowledge.export`、只读副作用、幂等、并行安全、超时 120 秒等），再实现 `execute` 把纯函数的结果包装成模型对象。文件末尾的 `create_tool()` 是工厂函数，供工具注册表按需构造实例。

在运行时，这个模块通过 `TOOL_ENABLED = True` 声明自己处于启用状态，并且被 `__all__` 显式导出了全部公开符号。它与 `knowledge.import` 是一对：导出的载荷形状刻意做成与导入工具的入参一致，因此导出之后可以直接原样导回，形成备份与迁移的闭环。Web 层的 `/api/export` 接口只是复用它并额外加上 `Content-Disposition` 文件名，本身不含任何导出逻辑——文件头注释也明确说明，`web/app.py` 里原先内联的导出构造已经被删除，本模块是这段逻辑的唯一出处。

## 二、函数与类逐条详解

### `export_payload(manager: MemoryManager, *, include_expired: bool = False, limit: int = 0, exported_at: str | None = None) -> dict[str, Any]` （第 32 行）

- **作用**：这是整个导出能力的核心实现，负责把记忆库里的一批记忆条目取出来、转成纯字典、按需截断，并组装成一份自描述的导出载荷。它需要同时满足三个诉求：载荷必须是可移植的纯 JSON 结构；载荷必须带版本号和分类计数，让导入方能「先验后写」；计数口径必须写死在载荷里，而不是留给导入方去猜哪些条目算语义层、哪些算情景层。它被 `ExportKnowledgeTool.execute` 调用，也是 Web 层 `/api/export` 真正依赖的函数，所以任何导出行为都收敛到这一处，避免出现两份实现。因为它是纯函数（只依赖传入的 `manager`，不碰模块级可变状态），所以既能被工具层调用，也能被 HTTP 层直接调用。
- **参数**：
  - `manager: MemoryManager`：记忆管理器实例，是唯一的必需参数。它只被用来调用 `list()` 取条目，因此调用方只要能提供一个具备 `list(include_expired=...)` 方法并返回带 `to_dict()` 的条目对象的管理器即可。
  - `include_expired: bool = False`：关键字限定参数，是否把已过期的记忆条目也算进来。默认 `False`，与导出文件（备份）的默认口径保持一致，避免把已经失效的知识当作有效知识迁移出去。
  - `limit: int = 0`：关键字限定参数，最多导出多少条，`0` 表示「不限制、全量」。必须是非负整数，且明确禁止 `bool`（因为 Python 里 `True`/`False` 是 `int` 的子类，若不加限制，`limit=True` 会被当成 1 静默通过）。工具层默认给 100，防止工具结果撑爆模型上下文；HTTP 导出传 0，导出整库。
  - `exported_at: str | None = None`：关键字限定参数，导出时间戳字符串。默认 `None`，此时由函数内部用当前 UTC 时间生成 ISO 格式字符串；如果调用方传入（例如为了测试可复现或为了保持某个固定时间），则原样写进载荷，函数不做任何格式校验。
- **返回**：返回一个 `dict[str, Any]`，固定包含四个键：`format`（值为模块级常量 `EXPORT_FORMAT`，即 `"knowledge-nebula-export/v1"`）、`exported_at`（时间戳字符串）、`counts`（一个字典，含 `total`、`semantic`、`episodic` 三个整数）、`items`（条目字典组成的列表）。注意 `counts` 统计的是**截断之后**的 `items`，也就是说 `limit` 生效时计数反映的是实际导出条数，而不是库里的总数。
- **内部流程**：第一步做入参防御性校验，用 `isinstance(limit, bool) or not isinstance(limit, int) or limit < 0` 一次性拦住布尔值、非整数（如浮点、字符串、`None`）和负数三种非法情况，命中就抛 `ValueError("limit must be a non-negative integer")`。第二步用一个列表推导把 `manager.list(include_expired=include_expired)` 返回的每个条目对象调用 `to_dict()` 转成字典，得到 `items`。第三步判断 `if limit:`——注意这里用的是真值判断而不是 `limit > 0`，但因为前面已经保证了 `limit` 是非负整数，所以 `0` 与 `False` 以外的正数才会进入分支，执行 `items = items[:limit]` 做头部截断（保留最先返回的前 limit 条，而不是随机抽样）。第四步组装返回字典：`exported_at` 用 `exported_at or datetime.now(UTC).isoformat()` 兜底生成带时区的 ISO 8601 时间串；`counts.total` 直接取 `len(items)`；`counts.semantic` 与 `counts.episodic` 用两个生成器表达式分别统计 `item["memory_type"]` 等于 `"semantic"` 和 `"episodic"` 的条目数。
- **异常/边界**：`limit` 为布尔、非整数或负数时抛 `ValueError`。`limit` 大于实际条目数时切片不会报错，只是返回全部条目。`limit=0` 表示全量，不做任何截断。`exported_at` 传入空字符串 `""` 时，因为用了 `or` 兜底，会被替换成当前时间（这是一个隐含的边界行为：空串不能作为显式时间戳）。若 `manager.list()` 抛异常（例如底层存储不可用），本函数不做捕获，异常直接向上冒泡。若某条目缺少 `memory_type` 键或 `to_dict()` 返回值不含该键，计数阶段会抛 `KeyError`；若条目对象没有 `to_dict()` 方法则抛 `AttributeError`。函数本身不处理空库情况，空库会正常返回 `items: []` 且三个计数全为 0。
- **同文件关系**：它调用了模块级常量 `EXPORT_FORMAT`，被同文件的 `ExportKnowledgeTool.execute` 调用，也被 `__all__` 导出。它不调用 `export_filename`，文件名生成是分开的一步。

### `export_filename(*, stamp: str | None = None) -> str` （第 65 行）

- **作用**：生成导出文件的建议文件名，形如 `knowledge_export_20240101-120000.json`。它被单独抽出来，是因为文件名的语义与载荷内的时间语义并不相同：这里的 `stamp` 只是给人看的本地时间戳，用于让多次导出在磁盘上不重名、便于人工辨认先后顺序，而载荷里的 `exported_at` 才是机器可校验的时间语义。HTTP 层会把这个文件名用作 `Content-Disposition` 的下载名，工具层则把它放进输出模型，让模型知道建议的落盘名字。默认情况下它取当前 UTC 时间并格式化成紧凑的 `%Y%m%d-%H%M%S`，不含时区后缀、不含冒号，因此可以直接作为 Windows 与类 Unix 系统上的合法文件名。
- **参数**：
  - `stamp: str | None = None`：关键字限定参数，时间戳字符串。默认 `None`，此时由函数用 `datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")` 生成；若调用方传入则原样嵌入文件名，函数不校验其格式，也不做非法字符过滤。
- **返回**：返回一个 `str`，格式固定为 `knowledge_export_{stamp}.json`，其中 `stamp` 是传入值或自动生成的时间戳。
- **内部流程**：第一步用 `stamp = stamp or datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")` 做兜底赋值——传入非空字符串时直接使用，传入 `None` 或空串时生成当前时间戳。第二步用 f-string 拼出 `knowledge_export_` 前缀、时间戳、`.json` 后缀三部分并返回。全程没有文件系统操作，它只是生成字符串，不创建、不写入任何文件。
- **异常/边界**：`stamp` 为空字符串时按「未提供」处理，改用当前时间。`stamp` 若含有 `/`、`\`、`:` 等非法字符，函数不会拦截，可能生成不可用的文件名，责任在调用方。由于格式串里只有年月日时分秒，同一秒内多次调用会得到相同文件名，存在覆盖风险，函数不做去重或加随机后缀。无特殊异常处理。
- **同文件关系**：它只依赖模块顶部的 `datetime` 与 `UTC` 导入，被同文件的 `ExportKnowledgeTool.execute` 调用，也被 `__all__` 导出；它不调用本文件里的任何其它函数。

### `class ExportKnowledgeInput(BaseModel)` （第 72 行）

- **作用**：定义 `knowledge.export` 工具的入参结构，是模型（Agent）在调用这个工具时必须填写的字段契约。它继承 Pydantic 的 `BaseModel`，因此自带类型校验、默认值填充和 JSON Schema 生成能力，工具运行时框架会用它来解析并校验模型给出的参数。之所以要有这个类，而不是让 `execute` 直接收裸字典，是为了让非法参数在进入业务逻辑之前就被拦住，同时让 `ToolSpec` 能通过 `input_model` 自动对外暴露一份可读的参数说明。类本身不包含任何方法，只有两个字段声明和一条配置，它的价值在于声明与校验。
- **参数**：本类不接收构造位置参数之外的额外数据，`model_config = ConfigDict(extra="forbid", strict=True)` 意味着**多传任何未声明的字段都会直接报校验错误**，且不做类型宽松转换（例如字符串 `"100"` 不会被自动转成整数 `100`）。两个字段如下：
  - `include_expired: bool = Field(default=False, description=...)`：是否包含已过期条目，默认 `False`（与导出文件的口径一致），描述文本明确写了默认不包含。
  - `limit: int = Field(default=100, ge=0, le=10_000, description=...)`：最多返回多少条，默认 `100`，约束为 `ge=0`（不小于 0）且 `le=10_000`（不超过一万条）。描述说明 `0` 表示不限制，HTTP 导出用 0，而工具默认 100 是为了防止上下文爆炸。
- **返回**：作为 Pydantic 模型，实例化成功时返回一个已校验的模型对象，字段可通过属性访问（如 `arguments.limit`）；实例化失败时抛 `pydantic.ValidationError`。类本身没有业务返回值。
- **内部流程**：Pydantic 在类定义阶段读取字段注解与 `Field` 元数据，构建校验器与 JSON Schema；在实例化阶段按字段逐个校验类型与取值范围，`extra="forbid"` 触发未声明字段检查，`strict=True` 触发严格类型检查，全部通过后填充默认值并生成不可变字段集（Pydantic v2 语义下赋值仍可用，但本文件未使用）。本类没有自定义 `__init__`、校验器或方法，流程全部由框架完成。
- **异常/边界**：`limit` 超出 `[0, 10000]` 范围、类型不是整数（包括 `True`/`False` 这类布尔值在严格模式下会被拒绝）、或传入未声明字段时，抛 `pydantic.ValidationError`。`include_expired` 传非布尔值同样报错。缺失字段不报错，走默认值。无特殊自定义处理。
- **同文件关系**：被 `ExportKnowledgeTool.spec` 通过 `input_model=ExportKnowledgeInput` 引用，并被 `ExportKnowledgeTool.execute` 的类型注解使用；也被 `__all__` 导出。它不调用本文件的任何函数。

### `class ExportCounts(BaseModel)` （第 86 行）

- **作用**：定义导出载荷里 `counts` 子结构的模型，用来承载「这次导出到底有多少条、其中语义层多少、情景层多少」这一组计数。它被单独拆成一个模型而不是塞进输出模型的扁平字段里，是为了让计数口径在类型层面成为一组结构化的、可复用的契约，导入方和前端都能按 `counts.total` 这种路径稳定读取。它也是导出「可核对」这一目标的载体：导入方拿到载荷后先看 `format` 再核对 `counts`，就能在写库之前判断规模是否合理。该类不含任何方法。
- **参数**：同样启用 `model_config = ConfigDict(extra="forbid", strict=True)`，禁止额外字段并严格校验类型。三个字段全部为 `int`：
  - `total: int`：条目总数，无默认值，必填。
  - `semantic: int = Field(description="其中语义层条目数。")`：其中属于语义层（semantic）的条目数，必填。
  - `episodic: int = Field(description="其中情景层条目数。")`：其中属于情景层（episodic）的条目数，必填。
- **返回**：实例化成功返回一个字段齐全的模型对象；失败抛 `pydantic.ValidationError`。无业务返回值。
- **内部流程**：类定义阶段由 Pydantic 生成校验器与 schema；实例化阶段按 `total`、`semantic`、`episodic` 三个必填整数校验并落值。本类没有自定义方法或校验器，也不做「semantic + episodic 是否等于 total」这类跨字段一致性检查——这个约束在本文件里并不存在，计数可能因其它记忆类型而小于 `total`。
- **异常/边界**：缺少任一必填字段、字段类型不是整数、或传入额外字段时抛 `pydantic.ValidationError`。三个计数为负数不会被本模型拦截（没有 `ge=0` 约束），但正常调用路径下由 `export_payload` 计算得出，不会出现负数。无特殊自定义处理。
- **同文件关系**：被 `ExportKnowledgeOutput.counts` 作为字段类型引用，并在 `ExportKnowledgeTool.execute` 里通过 `ExportCounts(**payload["counts"])` 构造；也被 `__all__` 导出。它不调用本文件的任何函数。

### `class ExportKnowledgeOutput(BaseModel)` （第 94 行）

- **作用**：定义 `knowledge.export` 工具的输出结构，是工具返回给 Agent 运行时的最终数据形状。它把 `export_payload` 产出的裸字典「收编」成强类型模型，一方面让框架能按 `output_model` 生成对外可见的输出 schema，另一方面保证返回给模型的数据一定带有格式版本、时间、计数和建议文件名这四类信息。它比裸字典多了一个 `filename` 字段，因为文件名属于「HTTP 层/人」的关切，`export_payload` 这个纯序列化函数不负责生成它。该类不含任何方法。
- **参数**：启用 `model_config = ConfigDict(extra="forbid", strict=True)`，禁止额外字段并严格校验。四个字段如下：
  - `format: str`：载荷格式版本，必填，实际取值来自 `EXPORT_FORMAT`。
  - `exported_at: str`：导出时间戳字符串，必填。
  - `counts: ExportCounts`：计数子模型，必填，必须是 `ExportCounts` 实例或能通过校验的等价字典。
  - `items: list[dict[str, Any]] = Field(default_factory=list)`：条目列表，默认为空列表。使用 `default_factory=list` 而不是 `default=[]`，避免了可变默认值被所有实例共享这一经典陷阱。
  - `filename: str = Field(description="建议的导出文件名（HTTP 层用作下载名）。")`：建议的导出文件名，必填，描述明确说明 HTTP 层会把它用作下载名。
- **返回**：实例化成功返回模型对象，`ExportKnowledgeTool.execute` 直接把它作为工具执行结果返回；失败抛 `pydantic.ValidationError`。
- **内部流程**：类定义阶段由 Pydantic 读取字段与默认值工厂生成校验器和输出 schema；实例化阶段逐个校验 `format`、`exported_at`、`counts`（嵌套模型校验）、`items`（列表内每个元素必须是字典）、`filename`。`items` 为空时合法，走默认空列表或显式传入的空列表。本类不含自定义方法、校验器或序列化钩子。
- **异常/边界**：`counts` 传入非法结构（如缺少 `total`）会抛 `ValidationError`；`items` 里混入非字典元素（严格模式下）会抛 `ValidationError`；传入未声明字段会抛 `ValidationError`。字段缺失（`format`、`exported_at`、`counts`、`filename`）同样报错。无特殊自定义处理。
- **同文件关系**：引用同文件的 `ExportCounts` 作为字段类型，被 `ExportKnowledgeTool.spec` 通过 `output_model=ExportKnowledgeOutput` 引用，并被 `ExportKnowledgeTool.execute` 实例化返回；也被 `__all__` 导出。它不调用本文件的任何函数。

### `class ExportKnowledgeTool(BaseTool)` （第 104 行）

- **作用**：这是导出能力对 Agent 运行时暴露的工具封装，继承 `core.BaseTool`，把纯函数逻辑接进统一的工具协议。它承担三件事：用类属性 `spec` 声明工具的元信息与调用契约；用 `__init__` 与 `manager` 属性管理记忆管理器实例（支持延迟构造）；用 `execute` 把入参模型翻译成对 `export_payload` 与 `export_filename` 的调用，再把结果包装成出参模型。它之所以要存在，是因为 Agent 需要一个可被注册、可被发现、可被安全调度（声明了只读、幂等、并行安全、超时）的工具对象，而纯函数本身不具备这些元信息。类上声明的 `spec` 内容为：工具名 `knowledge.export`；描述说明它把整个知识库导出为可移植自描述的 JSON 载荷（格式版本 + 计数 + 条目），只读，并提示用 `knowledge.import` 导回；版本 `1.0.0`；`side_effect="read"`；`permissions=()` 表示不需要额外权限；`timeout_seconds=120.0`；`idempotent=True`；`parallel_safe=True`；标签 `("knowledge", "export", "backup", "read")`；`guidance` 提示用户要备份、迁移或把整库交给外部工具消费时使用，默认不导出过期项，要全量传 `limit=0`，并强调它只读不改数据、载荷形状与 `knowledge.import` 入参一致、导出后可直接导回。
- **参数**：类本身不接收参数；`spec` 是类级属性，在类体执行时构造一次，被该类的所有实例共享（`BaseTool` 若在实例上改写 `spec` 需自行注意，本文件未改写）。
- **返回**：类对象本身；实例化由 `__init__` 负责，返回值见下条。
- **内部流程**：类体先构造 `ToolSpec` 实例并绑定到 `spec`，随后定义 `__init__`、`manager` 属性与 `execute` 方法。工具注册表通过 `create_tool()` 拿到实例后，框架读取 `spec` 完成注册与参数校验，调用时由框架先按 `input_model` 解析参数，再调 `execute`，最后按 `output_model` 校验返回值。
- **异常/边界**：类体构造 `ToolSpec` 时若字段不合法（例如 `side_effect` 取值不被支持）会在导入模块时直接抛错，属于启动期失败。除此之外类定义阶段无特殊处理。
- **同文件关系**：引用同文件的 `ExportKnowledgeInput`、`ExportKnowledgeOutput` 作为 `spec` 的输入输出模型，方法内部调用 `export_payload`、`export_filename` 与 `ExportCounts`；被同文件的 `create_tool()` 实例化，也被 `__all__` 导出。另外在 `manager` 属性内部会从同包的 `._memory` 导入 `build_default_manager`（跨文件引用，仅用于延迟构造默认管理器）。

### `__init__(self, manager: MemoryManager | None = None) -> None` （第 127 行）

- **作用**：工具类的构造方法，唯一职责是保存外部注入的记忆管理器引用，让同一个工具实例可以在测试或特定场景下复用调用方给定的管理器，而不是每次都去构造默认管理器。它把管理器存进私有属性 `self._manager` 而不是直接调用构造逻辑，是为了配合下面的 `manager` 属性实现**延迟构造**：只有在真正需要用到管理器时才去建默认实例，从而避免「只是注册工具、从不调用导出」的场景白白付出构造记忆库连接的开销。默认参数 `None` 表示「我不关心，用默认的就行」，此时由 `manager` 属性负责兜底。
- **参数**：
  - `manager: MemoryManager | None = None`：可选的位置/关键字参数，外部注入的记忆管理器。传 `None`（默认）表示不注入，后续首次访问 `manager` 属性时会调用 `build_default_manager()` 构造默认管理器；传实例则直接使用该实例，属性不会再触发默认构造。
- **返回**：无返回值（`-> None`），只产生副作用：设置实例属性 `self._manager`。
- **内部流程**：一行赋值 `self._manager = manager`。不调用父类 `BaseTool.__init__`（本文件未显式调用，依赖父类无需初始化或已由元类/框架处理），不做参数校验，不建立任何连接。
- **异常/边界**：传入的 `manager` 不会被校验类型，若传入不具备 `list()` 方法的对象，错误会推迟到 `execute` 调用 `export_payload` 时才以 `AttributeError` 形式暴露。传入 `None` 合法。无特殊异常处理。
- **同文件关系**：为同文件的 `manager` 属性与 `execute` 方法提供 `self._manager` 状态；被同文件的 `create_tool()` 间接调用（`ExportKnowledgeTool()` 不传参）。

### `manager(self) -> MemoryManager` （第 130 行，`@property`）

- **作用**：这是一个只读属性（由 `@property` 装饰），对外暴露「本工具实际使用的记忆管理器」，同时承担懒加载职责。当构造时没有注入管理器（`self._manager is None`）时，它会在首次访问时导入并调用 `build_default_manager()` 构造默认管理器，然后**把结果缓存回 `self._manager`**，因此后续每次访问都直接返回同一个实例，不会重复构造。这样设计让工具在注册阶段保持轻量，又在真正执行导出时自动获得一个可用的记忆库入口。它是 `execute` 获取管理器的唯一途径，保证了「注入优先、否则用默认」的单一决策点。
- **参数**：无参数（属性访问形式，`self` 由 Python 自动传入）。
- **返回**：返回一个 `MemoryManager` 实例。若构造时注入过，返回注入的那个；否则返回 `build_default_manager()` 构造并缓存的那个。任何情况下都返回非 `None` 值（除非 `build_default_manager()` 本身返回 `None`，本文件未做该防御）。
- **内部流程**：第一步判断 `if self._manager is None:`。为真时进入分支：用函数内导入 `from ._memory import build_default_manager`（放在函数内部而不是模块顶部，是为了避免模块导入期的循环依赖或提前初始化记忆子系统），调用它并赋值给 `self._manager`。第二步无条件 `return self._manager`，把（可能是刚构造的）实例交给调用方。
- **异常/边界**：若 `._memory` 模块不存在或 `build_default_manager` 导入失败，访问该属性会抛 `ImportError`；若默认构造过程抛异常（例如存储路径不可用），异常会从属性访问处向上冒泡，本属性不做捕获，也不会把半成品写回 `self._manager`（赋值发生在调用返回之后）。并发场景下，两个线程同时首次访问可能各自构造一次默认管理器，存在重复构造但没有加锁保护。无特殊异常处理。
- **同文件关系**：被同文件的 `ExportKnowledgeTool.execute` 通过 `self.manager` 调用；它依赖同包 `._memory` 模块的 `build_default_manager`（跨文件）；它读取并写入 `__init__` 设置的 `self._manager`。

### `execute(self, arguments: ExportKnowledgeInput) -> ExportKnowledgeOutput` （第 138 行）

- **作用**：这是工具的实际执行入口，由 Agent 运行时在参数校验通过后调用。它做两件事：把已经校验过的入参转发给纯函数 `export_payload` 完成真正的序列化，然后把返回的裸字典逐字段翻译成强类型的 `ExportKnowledgeOutput`，并额外补上建议文件名。之所以要在中间做这层「字典 → 模型」的搬运，是因为底层纯函数刻意只返回可移植的纯 JSON 结构（便于被 HTTP 层和外部工具直接复用），而工具协议要求返回经过 `output_model` 校验的模型对象，这一层就是把两者粘起来的适配器。它本身不含任何业务判断，所有逻辑都在被调用的纯函数里。
- **参数**：
  - `arguments: ExportKnowledgeInput`：已经通过 Pydantic 校验的入参模型实例。函数从中读取 `arguments.include_expired`（布尔，是否含过期项）和 `arguments.limit`（非负整数，0 表示全量）两个字段。传入非该模型的等价对象会导致属性访问失败，但正常调用路径下由框架保证类型正确。
- **返回**：返回一个 `ExportKnowledgeOutput` 实例，字段取自 `export_payload` 的结果：`format`、`exported_at` 原样透传；`counts` 用 `ExportCounts(**payload["counts"])` 由计数字典展开构造；`items` 原样透传；`filename` 由 `export_filename()` 现场生成（未传 `stamp`，因此是当前 UTC 时间戳）。
- **内部流程**：第一步调用 `export_payload(self.manager, include_expired=arguments.include_expired, limit=arguments.limit)`，注意这里通过 `self.manager` 属性取管理器，因此会触发上一节的懒加载逻辑；同时**没有**传 `exported_at`，所以时间戳由 `export_payload` 内部用当前 UTC 时间生成。第二步构造 `ExportKnowledgeOutput`：`format` 与 `exported_at` 按字典键取值，`counts` 用 `**` 展开字典构造嵌套模型，`items` 直接赋值，`filename` 调用 `export_filename()` 生成。第三步把构造好的模型作为返回值交给框架，由框架按 `output_model` 再校验一次并序列化给模型。
- **异常/边界**：`export_payload` 抛出的 `ValueError`（例如 `limit` 为负——虽然入参模型已有 `ge=0` 约束，理论上不会走到）会原样向上冒泡。`self.manager` 懒加载失败时抛 `ImportError` 或底层构造异常。`payload` 缺少预期键时抛 `KeyError`。`payload["counts"]` 结构不匹配 `ExportCounts` 时抛 `pydantic.ValidationError`。若记忆库为空，`items` 为空列表、计数全 0，函数正常返回，不报错。无特殊异常捕获。
- **同文件关系**：调用同文件的 `export_payload`、`export_filename`、`ExportCounts`，并构造同文件的 `ExportKnowledgeOutput`；通过 `self.manager` 间接使用 `__init__` 与 `manager` 属性；被工具框架（跨文件，`core.BaseTool` 协议）调用。

### `create_tool() -> BaseTool` （第 153 行）

- **作用**：这是一个极简的工厂函数，用来在工具注册流程中按需创建一个全新的 `ExportKnowledgeTool` 实例。之所以不直接暴露类而提供工厂，是因为工具注册表通常约定「每个工具模块提供一个 `create_tool()` 入口」——这样注册表不需要知道每个工具类的构造签名（有些工具可能需要额外依赖或配置），只要统一调用无参工厂即可，同时也便于将来在不改变调用方的前提下替换具体实现或做包装。它不传任何参数给构造函数，因此创建出来的实例使用懒加载的默认记忆管理器。
- **参数**：无参数。
- **返回**：返回 `BaseTool` 类型的实例（实际运行时是 `ExportKnowledgeTool` 对象），已带有类级 `spec` 元信息，可直接被注册与调度。
- **内部流程**：单行 `return ExportKnowledgeTool()`，不缓存实例、不做单例控制、不做任何校验或配置注入。每次调用都会产生一个新对象，但新对象之间不共享记忆管理器（各自懒加载各自的默认管理器，除非框架本身对默认管理器做了单例）。
- **异常/边界**：若 `ExportKnowledgeTool` 构造过程抛错（本文件里只有一行赋值，实际不会），异常会向上冒泡。无特殊异常处理，无参数校验需求。
- **同文件关系**：实例化同文件的 `ExportKnowledgeTool`；被外部工具注册表调用（跨文件）；被 `__all__` 导出。它不调用本文件的其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `export_payload` | 把记忆库条目取出来转成纯字典、按 limit 截断，并组装成带格式版本号与分类计数的自描述 JSON 载荷。 |
| `export_filename` | 生成形如 `knowledge_export_<时间戳>.json` 的建议下载文件名。 |
| `ExportKnowledgeInput` | 定义导出工具的入参契约：`include_expired` 与带范围约束的 `limit`。 |
| `ExportCounts` | 定义载荷中 `counts` 子结构：`total`、`semantic`、`episodic` 三个计数。 |
| `ExportKnowledgeOutput` | 定义导出工具的出参契约：格式版本、导出时间、计数、条目列表与建议文件名。 |
| `ExportKnowledgeTool` | 继承 `BaseTool` 的导出工具封装，用 `ToolSpec` 声明只读、幂等、并行安全的工具元信息。 |
| `ExportKnowledgeTool.__init__` | 保存外部注入的记忆管理器引用（可为 `None`），为懒加载做准备。 |
| `ExportKnowledgeTool.manager` | 只读属性，未注入时按需导入并构造默认记忆管理器并缓存复用。 |
| `ExportKnowledgeTool.execute` | 调用 `export_payload` 与 `export_filename`，把结果包装成 `ExportKnowledgeOutput` 返回。 |
| `create_tool` | 无参工厂函数，创建并返回一个 `ExportKnowledgeTool` 实例供工具注册表使用。 |
