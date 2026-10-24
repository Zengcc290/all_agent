# tool/seed_knowledge.py

## 一、这个文件是干什么的

这个文件是「种子播种工具」，它把 Aetheria 星图的一份演示种子数据（实体、关系、备注）批量导入到四层记忆系统的语义记忆里，让一个空库在几分钟内变成一份可以演示、可以画图、可以问答的星图。它被设计成一个独立的运维能力：既能被 Agent 当作工具调用（`knowledge.seed`），也能被 Web 应用的启动钩子或 `POST /api/seed` 端点直接调用，因为文件里既提供了纯函数 `seed()`，又提供了把它包装成工具的 `SeedKnowledgeTool`。整个模块最重要的性质是**幂等**：判定依据不是外部状态文件，而是记忆库自身的内容——每条种子项的 `metadata` 里都写入固定标记 `SEED_MARK`（值为 `"aetheria-seed-v1"`），只要语义记忆里已经存在任意一条带这个标记的条目，第二次调用就会直接跳过并返回跳过原因，不会重复灌数据。模块顶层定义了三个模块级常量：`TOOL_ENABLED = True` 用于让工具注册器发现这个工具，`SEED_MARK` 是幂等标记（跨工具契约，不能改），`SEED_FILE` 指向仓库内固定位置的种子数据文件 `web/seed_data.json`（由 `__file__` 上溯两级得到仓库根再拼 `web`）。文件里主要包含：一个执行导入的函数 `seed()`、两个 pydantic 模型 `SeedKnowledgeInput` / `SeedKnowledgeOutput`（分别描述工具入参与出参）、一个继承 `BaseTool` 的工具类 `SeedKnowledgeTool`（内含 `__init__`、`manager` 属性、`execute` 方法），以及一个工厂函数 `create_tool()`。文件末尾的 `__all__` 列出了对外公开的七个名字，说明这个模块同时被当作库和当作插件使用。注释里特别说明：这段逻辑以前藏在 `web/seed.py` 里，现在 `web/seed.py` 已删除，本模块是唯一实现，启动钩子与端点都改为调用这里；同时 `knowledge.orphan_entities` 的「完全孤立实体」判定会把 seed 实体排除在外，所以 `metadata.seed` 这个标记是跨模块契约。

## 二、函数与类逐条详解

### `seed(manager: MemoryManager, path: Path | None = None) -> dict[str, Any]` （第 41 行）

- **作用**：这是整个模块的「干活函数」，负责把种子 JSON 文件里的三类数据（entities / relations / notes）真正写进记忆库，并返回一份统计报告。它是模块里唯一直接操作记忆库的地方，`SeedKnowledgeTool.execute` 只是给它套了一层参数校验和结果建模的外壳，因此无论从工具调用还是从启动钩子调用，最终都会走到它。它把幂等判断、文件存在性判断、数据解析、逐条写入、计数统计全部收在一个函数里，调用者拿到的字典既包含「有没有真的写」也包含「写了多少」。它存在的原因是：早期这段逻辑被复制在 Web 层，Agent 既无法主动播种也无法解释图里为什么有这些节点，抽成纯函数后逻辑只有一份。它被调用的时机是「用户要求灌演示数据」「初始化星图」「应用首次启动且库为空」这类场景。它不关心自己是被谁调用的，也不做权限判断，那些都由工具层负责。

- **参数**：
  - `manager: MemoryManager`：必填，记忆管理器实例，所有写入都通过它完成；函数会调用它的 `list(memory_type=...)` 查已有条目、`add(...)` 写实体和备注、并通过它的 `semantic` 子对象调用 `add_fact(...)` 写关系。传 `None` 会在调用方法时直接抛 `AttributeError`，函数本身不做防御。
  - `path: Path | None = None`：可选，种子数据文件的路径。默认 `None` 表示使用模块级常量 `SEED_FILE`（仓库内 `web/seed_data.json`）。传入的值会被 `Path(path)` 再包一层，所以传字符串路径或 `Path` 对象都可以；传空串 `""` 时 `path is not None` 成立，会被当作 `Path("")` 即当前目录去判断，通常会落到「文件不存在」分支，所以工具层在 `execute` 里用 `arguments.path or None` 把空串转成了 `None`。

- **返回**：返回 `dict[str, Any]`，共有四种可能的形态。第一种是种子文件不存在：`{"seeded": False, "reason": "种子文件不存在：<路径>"}`。第二种是已播种过：`{"seeded": False, "reason": "已播种过（幂等跳过）", "existing": <库内语义条目总数>}`。第三种是正常播种完成：`{"seeded": True, "source": "<实际读取的文件路径>", "entities": <写入实体数>, "relations": <写入关系数>, "notes": <写入备注数>}`，其中三个计数只统计**真正通过校验并写入**的条目，被跳过的空名字/空内容项不计入。函数永远不返回 `None`，也不会在正常情况下抛异常给调用者（除非文件内容不是合法 JSON，或记忆库写入本身报错）。

- **内部流程**：
  1. 用 `Path(path) if path is not None else SEED_FILE` 决定 `seed_path`。
  2. 调用 `seed_path.is_file()` 判断文件是否存在，不存在就直接返回「文件不存在」字典，不读文件、不碰记忆库。
  3. 调用 `manager.list(memory_type=MemoryType.SEMANTIC)` 取出库里所有语义记忆条目，然后用 `any(item.metadata.get("seed") == SEED_MARK for item in semantic_items)` 判断是否已经播过种；只要有一条命中就返回跳过，并把 `len(semantic_items)` 作为 `existing` 一起返回，方便调用者了解库的规模。
  4. 用 `seed_path.read_text(encoding="utf-8")` 读全文再 `json.loads` 解析，得到字典 `data`；随后用 `data.get("entities") or []`、`data.get("relations") or []`、`data.get("notes") or []` 取三个列表，这样即使键缺失或值为 `None` 也会退化成空列表，不会因为缺键而崩。
  5. 遍历 `entities`：用 `str(entity.get("name") or "").strip()` 取名字，空名字 `continue` 跳过；否则调用 `manager.add(name, memory_type=MemoryType.SEMANTIC, metadata={...}, importance=float(entity.get("importance", 0.85)))`，metadata 里固定写 `kind="entity"`、`title=name`、`domain=entity.get("domain") or "未分类"`、`seed=SEED_MARK`，`importance` 默认 0.85。每成功一条 `entity_count += 1`。
  6. 遍历 `relations`：分别取出 `subject`、`predicate`、`object` 并 `strip()`，三者只要有一个为空就 `continue` 跳过；否则调用 `manager.semantic.add_fact(subject, predicate, obj, metadata={...}, confidence=float(relation.get("confidence", 1.0)))`，metadata 含 `domain`（默认 `"未分类"`）、`note`、`date`（都默认空串）、`seed=SEED_MARK`，置信度默认 1.0。每成功一条 `relation_count += 1`。
  7. 遍历 `notes`：用 `str(note.get("content") or "").strip()` 取正文，空正文 `continue`；否则调用 `manager.add(content, memory_type=MemoryType.SEMANTIC, metadata={...}, importance=0.4)`，metadata 含 `kind="note"`、`entity`、`domain`、`title`（默认 `"档案"`）、`date`、`seed=SEED_MARK`；注意备注的 importance 是硬编码的 0.4，不读 JSON 里的值。每成功一条 `note_count += 1`。
  8. 最后返回带 `seeded=True` 和三项计数的汇总字典。

- **异常/边界**：
  - 文件缺失：不抛异常，返回 `seeded=False` 加原因字符串。
  - 已播种：不抛异常，返回 `seeded=False` 加原因与 `existing`。
  - JSON 非法或文件编码不是 UTF-8：`json.loads` / `read_text` 会抛 `json.JSONDecodeError` 或 `UnicodeDecodeError`，函数不做捕获，异常向上冒泡给调用者（工具层也没有 try/except）。
  - 顶层不是字典（比如是列表）：`data.get(...)` 会抛 `AttributeError`，无特殊处理。
  - `entities` / `relations` / `notes` 值不是列表（比如是字符串或数字）：`or []` 不会兜住非空非列表值，`for` 迭代会抛 `TypeError`，无特殊处理。
  - 单条记录不是字典：`entity.get(...)` 会抛 `AttributeError`，无特殊处理。
  - `importance` / `confidence` 值不是数字：`float(...)` 会抛 `ValueError` 或 `TypeError`，无特殊处理。
  - 空名字、空三元组、空正文：静默跳过，不计入计数，也不会中断整个导入。
  - 部分写入后中途失败：函数没有事务/回滚，前面已经写进去的条目会保留；但因为没有写完全部条目就异常退出，库里不会出现「完整播种」的状态，而已经写入的那些条目**已经带了 `seed` 标记**，所以下一次调用会被幂等判断拦住——这是一个需要注意的边界（代码本身没有任何补偿逻辑）。
  - 并发调用：函数自身没有任何锁，两个进程同时调用可能都通过幂等检查并重复写入；并发安全由工具层的 `parallel_safe=False` 声明来规避。
  - 超时：函数内部没有超时控制，超时上限 120 秒写在工具 `spec` 上，由工具运行时负责。

- **同文件关系**：它被 `SeedKnowledgeTool.execute` 调用（`seed(self.manager, arguments.path or None)`）。它自身调用了模块级常量 `SEED_FILE` 与 `SEED_MARK`，但不调用本文件里任何其它函数或方法。

### `class SeedKnowledgeInput(BaseModel)` （第 126 行）

- **作用**：这是工具 `knowledge.seed` 的输入模型，用来在真正读写记忆库之前把 Agent（或 HTTP 调用方）传进来的参数校验、归一化成一个强类型对象。它只承载一个字段 `path`，因为播种工具的设计是「默认用仓库内的种子文件，只有想换数据源时才传路径」，把数据源选择权显式暴露出来而不是让工具去猜。它继承 pydantic 的 `BaseModel`，并在 `model_config` 里开启 `extra="forbid"` 与 `strict=True`：前者表示调用方多传任何未知字段都会被判为非法，后者表示不做宽松的类型强转（例如不会把数字 123 自动变成字符串 `"123"`）。它存在的意义是让「参数错误」在进入 `seed()` 之前就被挡住，避免脏参数一路传到记忆库写入环节。

- **参数**：类本身不接收参数，字段如下。
  - `path: str`：默认 `""`（空串），`max_length=1000`。含义是种子数据文件路径；空串表示用仓库内的 `web/seed_data.json`。约束是必须是字符串且长度不超过 1000 个字符，超出会触发 pydantic 校验错误；由于 `strict=True`，传 `None`、数字、列表等都会校验失败。字段描述里也明确写了「空串表示用仓库内的 web/seed_data.json」，这个约定与 `execute` 里 `arguments.path or None` 的写法互相配合。

- **返回**：它不是函数，实例化后返回一个 `SeedKnowledgeInput` 对象；该对象只有一个属性 `path`（字符串）。校验失败时 pydantic 抛 `ValidationError`。

- **内部流程**：pydantic 在类定义阶段收集 `model_config` 与字段声明；实例化时按 `strict=True` 的规则检查 `path` 的存在性与类型，按 `max_length=1000` 检查长度，按 `extra="forbid"` 拒绝任何未声明字段；全部通过后把值存到实例属性上。类体内没有自定义的 `__init__`、校验器或方法。

- **异常/边界**：字段有默认值 `""`，所以完全不传参数也能构造成功（等价于「用默认种子文件」）。类型不符、超长、传入未声明字段都会抛 pydantic 的 `ValidationError`，由工具运行时负责转成给 Agent 的错误信息。无自定义异常处理。

- **同文件关系**：它被 `SeedKnowledgeTool.spec` 通过 `input_model=SeedKnowledgeInput` 引用，也被 `SeedKnowledgeTool.execute` 的类型标注使用（`arguments: SeedKnowledgeInput`）。它不调用本文件里的任何函数。

### `class SeedKnowledgeOutput(BaseModel)` （第 136 行）

- **作用**：这是工具 `knowledge.seed` 的输出模型，把 `seed()` 返回的松散字典整理成一份字段固定、语义明确的强类型结果，让 Agent 能可靠地读出「这次到底写了没有、为什么没写、写了多少」。它同时覆盖成功与失败两种返回形态：成功时 `seeded=True` 且三个计数有值，失败时 `seeded=False` 且 `reason` 有值，可能还带 `existing`。因为 `seed()` 的字典里不同分支缺不同的键，这个模型给所有字段都配了默认值，从而保证无论哪条分支都能构造出完整对象，不会因为缺键而报错。它也承担着对外契约的作用：`__all__` 把它导出，说明它属于公开 API。

- **参数**：类本身不接收参数，字段如下。
  - `seeded: bool`：必填（无默认值），`true` 表示本次真的写入了；`false` 表示幂等跳过或文件缺失。
  - `reason: str`：默认 `""`，未播种时的原因（文件不存在 / 已播种过）。
  - `source: str`：默认 `""`，实际读取的种子文件路径，只有成功播种时才会被填。
  - `existing: int`：默认 `0`，已播种过时库内现有的语义条目数。
  - `entities: int`：默认 `0`，本次写入的实体数。
  - `relations: int`：默认 `0`，本次写入的关系（事实）数。
  - `notes: int`：默认 `0`，本次写入的备注数。
  - `model_config = ConfigDict(extra="forbid", strict=True)`：禁止额外字段，禁止宽松类型转换。

- **返回**：不是函数；实例化返回一个 `SeedKnowledgeOutput` 对象，携带上述七个属性。构造时若 `seeded` 缺失或类型不符会抛 `ValidationError`。

- **内部流程**：与 `SeedKnowledgeInput` 类似，pydantic 在类定义时读取 `model_config` 与字段默认值，实例化时校验七个字段的类型与额外键，然后保存属性。类体内没有自定义方法、校验器或 `__init__`。

- **异常/边界**：除 `seeded` 外所有字段都有默认值，所以只要给了 `seeded` 就能构造成功；缺 `seeded` 或类型不对（在 `strict=True` 下 `1`、`"true"` 都不算 `bool`）会抛 `ValidationError`。无自定义异常处理。

- **同文件关系**：被 `SeedKnowledgeTool.spec` 通过 `output_model=SeedKnowledgeOutput` 引用，并被 `SeedKnowledgeTool.execute` 实例化并作为其返回类型。它不调用本文件里的任何函数。

### `class SeedKnowledgeTool(BaseTool)` （第 148 行）

- **作用**：这是把 `seed()` 包装成 Agent 可调用工具的类，工具名为 `knowledge.seed`。它继承 `core.BaseTool`，通过类属性 `spec`（一个 `ToolSpec`）向运行时声明自己的元信息：名称、描述、版本、入参模型、出参模型、副作用类型 `"write"`、权限（空元组，表示不需要额外权限）、超时 120 秒、幂等标记 `True`、并发安全 `False`、标签 `("knowledge", "seed", "demo", "write")`，以及给 Agent 看的中文使用指引 `guidance`。这段 `guidance` 明确告诉模型「只有用户要求灌演示数据或初始化星图时才调用」「它是幂等的，重复调用会返回跳过原因，不要因为 seeded 为 false 就重试」「默认取仓库内 web/seed_data.json，要换数据源才传 path」，这几句实际上是防止 Agent 陷入「失败就重试」死循环的关键约束。类本身还负责惰性持有 `MemoryManager`：构造时可以不传，真正执行时才通过 `_memory.build_default_manager` 建一个默认管理器，这样工具对象可以很早被创建（注册阶段），而昂贵的记忆库初始化推迟到第一次调用。

- **参数**：类本身不接收参数（`spec` 是类属性，不是构造参数）。构造参数见 `__init__`。

- **返回**：类是模板，实例化返回一个可执行的工具对象；通过 `execute()` 产出 `SeedKnowledgeOutput`。

- **内部流程**：类定义时构造 `ToolSpec(...)` 并绑定到类属性 `spec`；`__init__` 只保存可选的 manager；`manager` 属性在需要时惰性构建；`execute` 负责把输入模型转成 `seed()` 的调用参数、再把结果字典转成输出模型。此外 `side_effect="write"`、`idempotent=True`、`parallel_safe=False` 这三个声明会影响运行时对它的调度策略（写操作、可安全重复、不可并行）。

- **异常/边界**：类本身不抛异常；`spec` 里声明的 `timeout_seconds=120.0` 与 `permissions=()` 是给运行时的约束，不是代码里的强制检查。若 `spec` 字段写错，错误会在 `ToolSpec` 构造阶段暴露。

- **同文件关系**：它引用本文件的 `SeedKnowledgeInput`、`SeedKnowledgeOutput`，其 `execute` 调用本文件的 `seed()`；它被本文件的 `create_tool()` 实例化。此外它通过 `from ._memory import build_default_manager` 引用同包内另一个模块（不属于本文件）。

### `__init__(self, manager: MemoryManager | None = None) -> None` （第 171 行）

- **作用**：工具类的构造方法，只做一件事——把外部传入的（可选的）记忆管理器存到实例私有属性 `self._manager` 上，除此之外不建连接、不读文件、不校验参数。之所以把 manager 设计成可选，是为了让工具在注册阶段能被无依赖地创建（例如工具注册表只想要一个 `BaseTool` 实例），而真正的记忆库实例推迟到 `manager` 属性第一次被访问时再惰性构建；同时在测试或应用启动钩子里也可以注入一个已经建好的 manager，避免重复初始化。它不设置任何其它实例状态，因此这个类是「无状态外壳 + 惰性依赖」的典型写法。

- **参数**：
  - `self`：实例本身。
  - `manager: MemoryManager | None = None`：可选的记忆管理器。传具体实例时后续所有读写都用它；传 `None`（默认）时 `self._manager` 保持 `None`，等 `manager` 属性被访问时再通过 `_memory.build_default_manager()` 构建。没有任何类型检查，传错类型会在真正调用 `add` / `list` 时才报错。

- **返回**：无返回值（`None`）。

- **内部流程**：单条赋值 `self._manager = manager`，没有分支、没有循环、不调用任何其它函数。

- **异常/边界**：无特殊处理，不会抛异常。

- **同文件关系**：它被本文件的 `create_tool()` 间接触发（`SeedKnowledgeTool()` 不带参数调用）；它设置的 `self._manager` 被本文件的 `manager` 属性读取。

### `manager` （property，第 174-175 行）

- **作用**：这是一个只读属性，用来给工具提供「一定会拿到一个可用 `MemoryManager`」的访问入口。它实现了惰性初始化：第一次访问时如果 `self._manager` 还是 `None`，就在函数内部 `from ._memory import build_default_manager` 导入默认管理器工厂并构建一个，缓存回 `self._manager`；之后再访问就直接返回缓存，不会重复构建。这样设计的好处是：工具对象可以在没有任何记忆库依赖的情况下被注册和持有，而记忆库的初始化（可能涉及打开文件、建索引等较重操作）只在实际执行播种时才发生一次。把 import 放在函数体内而不是模块顶部，也是为了不在模块导入阶段就拉起记忆子系统，避免循环导入与不必要的启动开销。

- **参数**：无（属性访问不需要参数，隐式的 `self` 是工具实例）。

- **返回**：返回 `MemoryManager` 实例——要么是构造时注入的那个，要么是惰性构建并缓存的那个。正常情况下永远不返回 `None`。

- **内部流程**：
  1. 判断 `self._manager is None`。
  2. 若为 `None`，执行局部导入 `from ._memory import build_default_manager`。
  3. 调用 `build_default_manager()` 并把结果赋给 `self._manager`。
  4. 返回 `self._manager`。
  5. 若不为 `None`，直接返回已有的 `self._manager`，不进入导入分支。

- **异常/边界**：如果同包的 `_memory` 模块缺失或 `build_default_manager` 不存在，局部导入会抛 `ImportError`；如果 `build_default_manager()` 自身构建失败（例如存储路径不可写），异常会原样冒泡。没有缓存失效机制：一旦构建成功就一直复用同一个实例，外部替换 `self._manager` 也只能通过直接改私有属性实现（属性本身没有 setter）。`None` 的情况不会返回给调用者，因为分支保证了构建。

- **同文件关系**：它读取并写入 `__init__` 设置的 `self._manager`；它被本文件的 `SeedKnowledgeTool.execute` 通过 `self.manager` 调用，从而把管理器传给 `seed()`。它引用了同包 `._memory` 模块的 `build_default_manager`（不属于本文件）。

### `execute(self, arguments: SeedKnowledgeInput) -> SeedKnowledgeOutput` （第 182 行）

- **作用**：这是工具的执行入口，由运行时在 Agent 决定调用 `knowledge.seed` 时触发。它做的是「适配」工作而不是业务逻辑：先把输入模型里的 `path` 转成 `seed()` 需要的参数形式，再把 `seed()` 返回的字典逐字段搬进输出模型。其中 `arguments.path or None` 是关键转换——空串是输入模型的默认值、表示「用仓库内默认种子文件」，而 `seed()` 用 `None` 表示同一个意思，这行代码把两种约定对齐，避免空串被 `Path("")` 解释成当前目录。所有 `bool(...)`、`str(... or "")`、`int(... or 0)` 的写法则是为了兼容 `seed()` 各条返回分支缺键的情况：缺键时 `dict.get` 返回 `None`，被 `or` 兜成空串或 0，再交给 pydantic 校验。整个方法不抛业务异常，也不做重试。

- **参数**：
  - `self`：工具实例。
  - `arguments: SeedKnowledgeInput`：已经过 pydantic 校验的输入对象，其中 `path` 是字符串（默认 `""`）。运行时负责保证类型正确；如果直接手写调用传了别的类型，`arguments.path` 的取值行为取决于传入对象，代码里没有额外防御。

- **返回**：返回 `SeedKnowledgeOutput` 实例，七个字段全部填好：`seeded` 来自 `result["seeded"]` 的布尔化，`reason` / `source` 缺省为空串，`existing` / `entities` / `relations` / `notes` 缺省为 0。成功播种时 `seeded=True` 且三个计数反映真实写入量；文件缺失或已播种时 `seeded=False` 且 `reason` 有值（已播种时还带 `existing`）。

- **内部流程**：
  1. 调用 `seed(self.manager, arguments.path or None)`，其中 `self.manager` 触发 `manager` 属性的惰性初始化；把结果存进局部变量 `result`。
  2. 用 `SeedKnowledgeOutput(...)` 构造返回对象：`seeded=bool(result.get("seeded"))`（缺失或假值都变 `False`）；`reason=str(result.get("reason") or "")`；`source=str(result.get("source") or "")`；`existing=int(result.get("existing") or 0)`；`entities=int(result.get("entities") or 0)`；`relations=int(result.get("relations") or 0)`；`notes=int(result.get("notes") or 0)`。
  3. 返回该对象。

- **异常/边界**：`seed()` 内部的异常（JSON 解析失败、记忆库写入失败、`float` 转换失败等）在这里**不被捕获**，会直接冒泡给运行时；因此调用方看到的是运行时统一包装的错误，而不是 `SeedKnowledgeOutput`。`arguments` 为 `None` 时访问 `arguments.path` 会抛 `AttributeError`，无特殊处理。`result` 中数值字段为 `None` 或缺失时被兜成 0；若数值是字符串（例如 `"3"`），`int(...)` 能转换成功，若不可转换则抛 `ValueError`。没有重试逻辑——这也与 `spec.guidance` 里「不要因为 seeded 为 false 就重试」的约束一致。

- **同文件关系**：它调用本文件的 `seed()`（传入 `self.manager`），读取本文件的 `manager` 属性，构造并返回本文件的 `SeedKnowledgeOutput`；它接收本文件的 `SeedKnowledgeInput` 作为参数类型。它被运行时（工具调度层）调用，本文件内没有其它函数调用它。

### `create_tool() -> BaseTool` （第 195 行）

- **作用**：这是工具工厂函数，供工具注册/发现机制按约定调用：注册器通常只需要一个「无参、返回 `BaseTool` 实例」的入口，就能把本模块的工具挂进运行时。它不接收任何配置，内部直接 `SeedKnowledgeTool()` 无参构造，也就是创建一个 manager 为 `None`、真正执行时才惰性构建记忆管理器的工具实例。返回类型标注为 `BaseTool`（父类）而不是 `SeedKnowledgeTool`，说明调用方只应依赖抽象接口。因为每次调用都新建实例，多次调用会得到互不共享 `_manager` 缓存的独立对象。

- **参数**：无。

- **返回**：返回一个新的 `SeedKnowledgeTool` 实例，静态类型标注为 `BaseTool`。永远不返回 `None`。

- **内部流程**：单条 `return SeedKnowledgeTool()`，没有分支、没有缓存、没有参数校验。

- **异常/边界**：无特殊处理。唯一可能的失败是 `SeedKnowledgeTool` 类体在导入时构造 `ToolSpec` 失败（例如 spec 字段非法），那属于模块导入期错误，与这个函数无关。不会返回 `None`，也不需要清理资源。

- **同文件关系**：它实例化本文件的 `SeedKnowledgeTool`；被外部工具注册机制调用，本文件内没有任何函数调用它。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `seed(manager, path)` | 读取种子 JSON，按实体/关系/备注三类幂等地写入语义记忆，并返回写入统计或跳过原因。 |
| `SeedKnowledgeInput` | 工具入参模型，只有一个 `path` 字段，空串表示使用仓库内默认种子文件。 |
| `SeedKnowledgeOutput` | 工具出参模型，用 `seeded`/`reason`/`source`/`existing`/`entities`/`relations`/`notes` 描述本次播种结果。 |
| `SeedKnowledgeTool` | 把 `seed()` 包装成名为 `knowledge.seed` 的写工具，声明超时、幂等、不可并行与使用指引，并惰性持有记忆管理器。 |
| `SeedKnowledgeTool.__init__(manager)` | 只把可选注入的记忆管理器存到 `self._manager`，不做任何初始化工作。 |
| `SeedKnowledgeTool.manager` | 只读属性，首次访问时惰性构建并缓存默认 `MemoryManager`，之后直接复用。 |
| `SeedKnowledgeTool.execute(arguments)` | 把输入模型的 `path` 适配成 `seed()` 的参数、调用 `seed()`，再把结果字典转成 `SeedKnowledgeOutput`。 |
| `create_tool()` | 无参工厂，返回一个新的 `SeedKnowledgeTool` 实例供工具注册器使用。 |
