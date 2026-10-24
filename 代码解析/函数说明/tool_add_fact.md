# tool/add_fact.py

## 一、这个文件是干什么的

这个文件实现了一个「写工具」：把一条结构化的语义事实（主语 subject、谓语 predicate、宾语 object 的三元组）写进记忆系统。它是知识图谱里一条「边」的最小可审计写入单元：一次调用会同时落到真值源（语义记忆行）与图投影（有向关系），所以 Agent 可以不去跑整条「写文本再让 LLM 抽取」的重管道，而是把用户明确陈述的事实立刻固化下来。文件顶部用模块 docstring 明确交代了它与 `memory.add` 的分工：`memory.add` 写的是「内容」（一段文本加任意元数据），本文件写的是「结构化三元组」，强制三件套齐全、带领域与置信度、并且保证图里长出对应的边，两者语义不同因而刻意不合并。本模块是这段逻辑的唯一实现，Web 层的 `POST /api/facts` 端点已经把原先内联的 `semantic.add_fact` 调用删掉，端点自身只保留请求校验与错误码映射，真正的写动作统一走这里。文件里包含：一个模块级常量 `TOOL_ENABLED`、一个纯函数 `add_fact`（真正干活的实现）、两个 Pydantic 数据模型 `AddFactInput` / `AddFactOutput`（分别描述工具入参与出参，并被 `strict=True, extra="forbid"` 收紧）、一个继承 `BaseTool` 的工具类 `AddFactTool`（携带 `ToolSpec` 元信息并实现 `execute`），以及一个工厂函数 `create_tool`。运行期它通常被工具注册表按名 `knowledge.add_fact` 发现并由 `create_tool()` 实例化，模型决定调用后由框架校验参数、调用 `execute`，`execute` 再转调模块级 `add_fact`，最终落到 `MemoryManager.semantic.add_fact` 上。

## 二、函数与类逐条详解

### `add_fact(manager, *, subject, predicate, object, domain="", note="", confidence=1.0) -> MemoryItem` （第 41 行）

- **作用**：这是本文件真正的业务实现，负责把一条三元组事实写进语义记忆并顺带更新知识图谱投影。它本身不做任何校验、不做字符串清洗，也不自己拼装记忆对象，而是把「怎么写」完全委托给 `manager.semantic.add_fact`，自己只负责把工具层的入参翻译成底层需要的形状：把 `domain` 和 `note` 打包成一个 `metadata` 字典，并把 `confidence` 原样透传。之所以需要它独立存在，是为了让「写事实」这件事有一个不依赖 Web 框架、不依赖 Pydantic、也不依赖工具运行时的调用点，从而任何内部代码（脚本、批处理、其它工具）都能直接复用同一份语义。它被 `AddFactTool.execute` 调用，是「Agent 声明一条事实」这条链路的最后一跳；也被 `__all__` 导出，说明它是本模块对外承诺的公共接口之一。
- **参数**：
  - `manager: MemoryManager`：记忆管理器实例，位置参数且无默认值，必须非 `None`；它是唯一的依赖注入点，函数体内直接取它的 `semantic` 子模块。
  - `subject: str`：主语实体名，必填，关键字限定（`*` 之后的参数都只能按关键字传）。本函数自身不检查空串或长度，约束由上层 `AddFactInput` 承担。
  - `predicate: str`：谓语/关系名，必填，关键字限定。同样不做本地校验。
  - `object: str`：宾语实体名，必填，关键字限定。同样不做本地校验。
  - `domain: str = ""`：所属领域，默认空串表示「未分类」；函数内部用 `domain or DEFAULT_DOMAIN` 做兜底，因此空串会被替换成常量 `DEFAULT_DOMAIN`。
  - `note: str = ""`：备注或证据说明，默认空串；内部用 `note or ""` 兜底，即空串仍然保持空串，只保证不会是 `None`。
  - `confidence: float = 1.0`：置信度，默认 1.0，取值范围 0~1 由上层 `AddFactInput` 用 `ge=0, le=1` 约束；本函数原样透传，不夹取也不校验。
- **返回**：返回 `manager.semantic.add_fact(...)` 的返回值，类型标注为 `MemoryItem`（来自 `memory.base`）。这个对象就是刚写入的语义记忆项，调用方随后通过它的 `.id` 取到记忆项 id——该 id 同时也是图关系上的 `memory_id`，因此它能作为「这条事实」的稳定句柄被返回给上层。
- **内部流程**：整个函数体只有一条 `return` 语句，没有局部变量、没有分支、没有循环。它把六个参数重新组织后调用 `manager.semantic.add_fact`：前三个位置参数按原顺序传 `subject`、`predicate`、`object`；然后以关键字传 `metadata={"domain": domain or DEFAULT_DOMAIN, "note": note or ""}`，也就是在字典构造的那一刻完成领域兜底与备注兜底；最后以关键字传 `confidence=confidence`。真正的持久化、图投影（长出有向边）以及时间戳等细节全部发生在被调用的 `semantic.add_fact` 内部，本文件不参与。
- **异常/边界**：本函数不做任何显式异常处理，也不抛出自定义异常。`manager` 为 `None` 时会在取 `.semantic` 属性处抛 `AttributeError`；`manager.semantic.add_fact` 内部对三元组为空、类型不对、存储不可用等情况的反应完全取决于底层实现，本文件既不拦截也不包装。`domain` 传 `None` 或空串都会被 `or` 兜底成 `DEFAULT_DOMAIN`，`note` 传 `None` 或空串都会被兜底成 `""`；`confidence` 传 `None` 或越界值不会被本函数纠正，会直接透传到底层。
- **同文件关系**：它调用了本文件之外的 `DEFAULT_DOMAIN` 常量（第 26 行导入）；被本文件的 `AddFactTool.execute` 调用（第 119 行），并被 `__all__` 列为公开导出。它不调用本文件内任何其它函数。

### `class AddFactInput(BaseModel)` （第 62 行）

- **作用**：这是工具入参的 Pydantic 模型，用来在真正写库之前把模型（LLM）给出的参数整体校验一遍。它存在的意义是把「格式与取值范围」的守门职责从业务实现里剥离出来：一旦校验通过，`add_fact` 就可以放心地不做二次检查。它通过 `Field` 给每个字段附上中文 `description`，这些描述会进入工具的参数 schema，帮助模型理解该填什么。字段上的 `min_length=1` 保证三元组三项都非空，`max_length` 引用 `constants` 里的 `WEB_FACT_*_MAX` 常量，说明长度上限与 Web 端点共用同一套配置，避免两处各写一个数字而对不上。`confidence` 用 `ge=0, le=1` 把置信度锁在闭区间内。
- **参数**：本类不是可调用函数，没有运行期参数；它的「参数」是类级字段定义，逐个说明如下。
  - `model_config`：不是字段，而是 `ConfigDict(extra="forbid", strict=True)`。`extra="forbid"` 表示传入任何未声明字段都会直接报错，防止模型幻觉出多余键被静默忽略；`strict=True` 表示不做宽松的隐式类型转换（例如不会把字符串 `"1"` 悄悄当成数字 1）。
  - `subject: str`：必填，`min_length=1`、`max_length=WEB_FACT_SUBJECT_MAX`，说明为「主语实体名。」。
  - `predicate: str`：必填，`min_length=1`、`max_length=WEB_FACT_PREDICATE_MAX`，说明为「谓语/关系名。」。
  - `object: str`：必填，`min_length=1`、`max_length=WEB_FACT_OBJECT_MAX`，说明为「宾语实体名。」。
  - `domain: str = ""`：默认空串，`max_length=WEB_FACT_DOMAIN_MAX`，说明为「所属领域；空串表示未分类。」。注意这里没有 `min_length`，所以空串合法，代表未分类。
  - `note: str = ""`：默认空串，`max_length=WEB_FACT_NOTE_MAX`，说明为「备注/证据说明。」。
  - `confidence: float = 1.0`：默认 1.0，`ge=0`、`le=1`，说明为「置信度 0~1。」。
- **返回**：类本身没有返回值；实例化 `AddFactInput(...)` 成功时返回一个已校验的不可变约束对象（Pydantic 模型实例），校验失败时抛 `pydantic.ValidationError`，不会返回任何对象。上层框架正是靠「构造成功/抛错」来区分参数是否可用。
- **内部流程**：没有方法体，只有类级声明。运行时行为由 Pydantic 的元类在类创建阶段完成：读取 `model_config` 设置模型级策略，把每个带 `Field` 的注解编译成字段校验器（包括长度约束与数值上下界），并据此生成 JSON Schema 供工具框架暴露给模型。实例化时 Pydantic 依次校验每个字段的存在性、类型、长度与数值范围，并按 `extra="forbid"` 检查是否存在未知键。
- **异常/边界**：参数缺失、类型不符、超长、`confidence` 越界、出现未声明字段，都会由 Pydantic 抛 `ValidationError`（含逐字段的错误明细），本类不捕获、不转换。空串边界由 `min_length=1` 覆盖（`subject`/`predicate`/`object` 不允许空串，`domain`/`note` 允许空串）。
- **同文件关系**：被 `AddFactTool.spec` 以 `input_model=AddFactInput` 引用（第 93 行），并被 `AddFactTool.execute` 的形参类型标注使用（第 118 行）；被 `__all__` 导出。它不调用本文件内任何函数。

### `class AddFactOutput(BaseModel)` （第 73 行）

- **作用**：这是工具出参的 Pydantic 模型，规定 `execute` 必须返回什么形状的结果。它存在的意义是让工具返回值对上层（工具运行时、Web 端点、模型）而言是自描述且稳定的：模型只需要读 `item_id` 就知道写入成功并拿到句柄，其余字段把这次写入的三元组与领域、置信度原样回显，便于调用方核对或直接用于后续引用。和入参一样，它用 `ConfigDict(extra="forbid", strict=True)` 锁死字段集合与类型，避免实现里多塞字段或返回错类型。
- **参数**：本类不是可调用函数，没有运行期参数；它的字段定义逐个说明如下。
  - `model_config`：`ConfigDict(extra="forbid", strict=True)`，禁止额外字段、禁止隐式类型转换。
  - `item_id: str`：必填，无默认值，说明为「写入的语义记忆项 id（同时也是图关系的 memory_id）。」。它是本类唯一带描述文字的字段，也是调用方最关心的字段。
  - `subject: str`：必填，回显写入时使用的主语。
  - `predicate: str`：必填，回显写入时使用的谓语。
  - `object: str`：必填，回显写入时使用的宾语。
  - `domain: str`：必填，回显最终生效的领域（注意 `execute` 传进来的是兜底后的值，而不是原始空串）。
  - `confidence: float`：必填，回显写入时使用的置信度。
- **返回**：类本身没有返回值；实例化成功返回校验通过的输出对象，字段缺失或类型不符时抛 `pydantic.ValidationError`。在本文件中它被 `AddFactTool.execute` 构造并返回。
- **内部流程**：与 `AddFactInput` 类似，没有方法体，全部行为由 Pydantic 在类创建与实例化阶段完成：编译字段校验器、生成 JSON Schema、按 `extra="forbid"` 与 `strict=True` 校验传入的关键字参数。由于所有字段都没有默认值，实例化时必须把六个字段全部传齐。
- **异常/边界**：少传任一字段、传错类型、传多余字段都会抛 `ValidationError`，本类不做兜底也不捕获。没有长度或数值范围约束，因此这里不会因为内容过长或置信度越界而报错（那些约束在入参侧已经挡住）。
- **同文件关系**：被 `AddFactTool.spec` 以 `output_model=AddFactOutput` 引用（第 94 行），并被 `AddFactTool.execute` 作为构造与返回类型使用（第 128、135 行）；被 `__all__` 导出。它不调用本文件内任何函数。

### `class AddFactTool(BaseTool)` （第 84 行）

- **作用**：这是把「写事实」这个能力包装成运行时可用工具的门面类。它继承 `BaseTool`（来自 `core`），通过类属性 `spec`（一个 `ToolSpec` 实例）向框架声明自己的身份与行为契约：工具名 `knowledge.add_fact`、给模型看的英文 `description`、版本 `1.0.0`、入参模型 `AddFactInput`、出参模型 `AddFactOutput`、副作用类型 `side_effect="write"`、空权限元组 `permissions=()`、超时 60 秒、`idempotent=False`、`parallel_safe=False`、标签 `("knowledge", "fact", "graph", "write")`，以及一段中文 `guidance` 指导模型什么时候该用、什么时候改用 `memory.rag` 或 `memory.add`。把「写操作、非幂等、不可并行」明确标注出来，是为了让调度层知道重复调用会产生新记录、且不能与其他写操作并发跑；`guidance` 里还强调「关系有更新时写新事实（旧值会被取代），不要试图改写历史」，这是给模型的策略约束。类本身不实现存储细节，只负责持有 `MemoryManager` 并把调用转给模块级 `add_fact`。
- **参数**：类不是函数；它的类属性 `spec` 是构造好的 `ToolSpec` 常量，字段含义见上（name、description、version、input_model、output_model、side_effect、permissions、timeout_seconds、idempotent、parallel_safe、tags、guidance）。实例化参数见 `__init__`。
- **返回**：类本身没有返回值；`AddFactTool()` 返回工具实例，由框架在注册/调用时使用。
- **内部流程**：类体在定义时先执行 `spec = ToolSpec(...)`，把契约固化成类级常量（所有实例共享同一份 spec）；随后定义 `__init__`、`manager` 属性与 `execute` 方法。运行期由 `BaseTool` 的通用流程读取 `spec` 做参数校验与结果包装，本类只补上「怎么真正执行」这一块。
- **异常/边界**：类定义阶段若 `ToolSpec` 的参数不合法会立即报错；实例化阶段本身不抛异常（`manager` 允许为 `None`）。执行期的异常来自 `execute` 与底层写入，本类不额外捕获。
- **同文件关系**：它引用了本文件的 `AddFactInput`、`AddFactOutput`（作为 spec 与 `execute` 的类型）以及模块级 `add_fact`（在 `execute` 中调用）和 `DEFAULT_DOMAIN`（在 `execute` 中兜底）；被本文件的 `create_tool` 实例化并返回；被 `__all__` 导出。它内部还会惰性导入本包的 `._memory.build_default_manager`。

#### `__init__(self, manager: MemoryManager | None = None) -> None` （第 107 行）

- **作用**：构造函数，只做一件极简的事——把外部可选的 `MemoryManager` 存进实例私有属性 `self._manager`。允许传 `None` 是刻意设计：这样工具可以在「还没准备好记忆管理器」的注册阶段被安全地创建出来（`create_tool()` 正是无参调用），把真正获取管理器的成本推迟到第一次实际使用时，由 `manager` 属性惰性构建。它不做任何 I/O、不校验类型、不建立连接，因此构造工具对象几乎零成本，适合在工具清单里批量实例化。
- **参数**：
  - `self`：实例自身，Python 隐式传入。
  - `manager: MemoryManager | None = None`：可选的位置/关键字参数，默认 `None`。传具体实例则后续所有写入都用它；传 `None` 表示「暂时没有」，等 `manager` 属性被访问时再惰性创建默认管理器。
- **返回**：`None`。构造函数不返回实例（Python 语义），实例由 `__new__` 产出。
- **内部流程**：唯一动作是 `self._manager = manager`，把入参原样保存为私有属性，不做转换、不做默认值替换（`None` 会被原样存下来，正是为了让 `manager` 属性能够判断「是否需要惰性创建」）。
- **异常/边界**：无特殊处理。传入非 `MemoryManager` 的任意对象也不会在这里报错，问题会推迟到 `execute` 调用 `manager.semantic.add_fact` 时才暴露。
- **同文件关系**：它设置的状态 `self._manager` 被本文件的 `manager` 属性（读并可能写）和 `execute`（通过属性间接使用）消费；被 `create_tool` 间接调用（`AddFactTool()` 走默认参数）。它不调用本文件内任何其它函数。

#### `manager` (property) -> `MemoryManager` （第 110 行）

- **作用**：这是一个只读属性（`@property` 装饰），充当实例的「惰性依赖获取器」。外部代码（以及本类自己的 `execute`）写 `self.manager` 时，如果构造时没有注入管理器，它会在此刻才去构建一个默认的 `MemoryManager` 并缓存进 `self._manager`，之后再次访问就直接返回缓存值，不会重复构建。之所以要这样做，是因为默认管理器的构建可能涉及配置读取、存储连接等较重动作，而这些在工具注册阶段既无必要也可能条件不满足；把构建推迟到真正要写事实的那一刻，既避免浪费也避免注册期失败。它让「必须显式注入」与「零配置开箱可用」两种用法共存。
- **参数**：只有隐式的 `self`，没有其它参数（属性不接受调用参数）。
- **返回**：返回 `MemoryManager` 实例。若构造时注入过，则原样返回注入的那个；若 `self._manager` 为 `None`，则先构建默认管理器并赋值给 `self._manager`，再返回它。类型标注保证返回类型是 `MemoryManager`。
- **内部流程**：先判断 `if self._manager is None:`；条件成立时执行一次函数内局部导入 `from ._memory import build_default_manager`（相对导入，放在函数体内是为了避免模块导入期的循环依赖与不必要的开销），调用它得到管理器对象并写入 `self._manager`；条件不成立时跳过整个 if 块。最后 `return self._manager`，此时它必然非 `None`。整体是「检查—惰性创建—缓存—返回」的经典模式。
- **异常/边界**：无显式异常处理。若 `build_default_manager()` 因配置缺失、依赖未安装或存储不可用而抛错，异常会直接向上冒泡到访问 `manager` 的调用点（通常是 `execute`）。没有并发保护：若多个线程同时首次访问，理论上可能各自构建一次并互相覆盖缓存，但因为该工具在 spec 里被标为 `parallel_safe=False`，这种并发场景被框架层面排除。
- **同文件关系**：它读取并写入 `__init__` 设置的 `self._manager`，内部调用本包 `._memory` 模块的 `build_default_manager`（本文件之外的函数）；被本文件的 `AddFactTool.execute` 通过 `self.manager` 调用。它不调用本文件内的其它函数。

#### `execute(self, arguments: AddFactInput) -> AddFactOutput` （第 118 行）

- **作用**：这是工具的执行入口，框架在参数校验通过后调用它来完成一次「写事实」。它承担两件事：一是把经过 Pydantic 校验的 `AddFactInput` 拆解成模块级 `add_fact` 需要的关键字参数并触发真正的写入；二是把写入结果的 id 与本次请求的字段组装成 `AddFactOutput` 返回给框架。它本身不写库、不校验长度、不做异常包装，是薄薄的一层适配（adapter），把「工具协议」与「记忆系统 API」对接起来。注意返回的 `domain` 用的是 `arguments.domain or DEFAULT_DOMAIN`，即回显的是最终生效值而非原始空串，保证调用方看到的结果与真正落库的领域一致。
- **参数**：
  - `self`：实例自身。
  - `arguments: AddFactInput`：已经过校验的入参模型实例（类型标注即来自本文件的 `AddFactInput`）。框架保证其字段完整且满足长度与范围约束，因此本方法内部不再重复检查。属性包括 `subject`、`predicate`、`object`、`domain`、`note`、`confidence`。
- **返回**：返回一个 `AddFactOutput` 实例，字段为：`item_id=item.id`（刚写入的语义记忆项 id）、`subject`/`predicate`/`object`（与入参一致）、`domain=arguments.domain or DEFAULT_DOMAIN`（兜底后的领域）、`confidence=arguments.confidence`（原样回显）。若底层写入抛错，本方法不会返回对象，异常向上传播。
- **内部流程**：第一步调用 `self.manager` 取得（或惰性构建）`MemoryManager`，与其余关键字参数一起传给模块级 `add_fact`，把 `arguments` 的六个字段逐个显式转发，返回值保存到局部变量 `item`。第二步用 `item.id` 作为 `item_id`，连同入参的 `subject`、`predicate`、`object`、`domain or DEFAULT_DOMAIN`、`confidence` 构造 `AddFactOutput` 并返回。整个过程没有分支与循环，只有两次调用与一次对象构造。
- **异常/边界**：无 `try/except`。可能冒出的异常包括：`manager` 属性构建默认管理器失败（配置或依赖问题）、`manager.semantic.add_fact` 因存储层问题或参数不被接受而失败、以及极端情况下 `item` 缺少 `id` 属性导致的 `AttributeError`；这些都会原样向上抛给框架，由框架按工具调用失败的约定处理（本文件不做错误码映射，映射在 Web 端点侧）。边界方面：`arguments.domain` 为空串时返回里会变成 `DEFAULT_DOMAIN`，`arguments.note` 不参与返回值因此无边界问题。
- **同文件关系**：它调用本文件的模块级函数 `add_fact`（第 119 行）、使用本文件的 `AddFactInput` 与 `AddFactOutput` 两个模型，并通过 `self.manager` 间接触发 `manager` 属性；常量 `DEFAULT_DOMAIN` 在此处被再次引用做兜底。它被框架（`BaseTool` 的调用流程）调用，本文件内没有其它函数调用它。

### `create_tool() -> BaseTool` （第 138 行）

- **作用**：这是工具的工厂函数，供注册表或插件加载逻辑按名调用，用来拿到一个全新的 `AddFactTool` 实例。它存在的意义是给框架一个统一的、无参的构造入口：注册代码不必知道 `AddFactTool` 的构造签名，也不必关心它内部是否允许注入 `MemoryManager`，只要调用 `create_tool()` 就能得到一个可直接注册的工具对象。因为返回的是新实例，每个调用方拿到的是独立的 `_manager` 缓存槽，互不干扰。返回类型标注为基类 `BaseTool`，说明调用方只需要按基类协议使用它（读 `spec`、调 `execute`），不依赖具体子类。
- **参数**：无参数。
- **返回**：返回 `BaseTool`（实际是 `AddFactTool` 的新实例）。构造使用默认参数，因此该实例的 `_manager` 初始为 `None`，会在第一次访问 `manager` 属性时才惰性构建默认管理器。
- **内部流程**：函数体只有一条语句 `return AddFactTool()`，直接调用本文件工具类的无参构造并返回结果；不缓存实例、不做单例、不做注册或副作用。
- **异常/边界**：无特殊处理。正常情况下不会抛异常（`__init__` 只做属性赋值）；只有在 `AddFactTool` 类定义本身有问题（例如导入期 `ToolSpec` 构造失败）时才会在模块导入阶段就暴露，而不是在这里。
- **同文件关系**：它实例化本文件的 `AddFactTool`，是本文件内部唯一调用该类的构造的地方；被本文件之外的注册/加载逻辑调用，并被 `__all__` 导出。它不调用本文件内的其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `add_fact` | 把一条 (主语, 谓语, 宾语) 事实连同领域、备注、置信度写进语义记忆并更新图投影，返回写入的 `MemoryItem`。 |
| `AddFactInput` | 工具入参的 Pydantic 模型，强制三元组非空、长度受限、置信度在 0~1 且禁止多余字段。 |
| `AddFactOutput` | 工具出参的 Pydantic 模型，回显 `item_id` 与本次写入的三元组、领域、置信度。 |
| `AddFactTool` | 继承 `BaseTool` 的工具门面类，用 `ToolSpec` 声明名称、描述、超时、写副作用等契约并转调实现。 |
| `AddFactTool.__init__` | 构造函数，把可选注入的 `MemoryManager` 存进 `self._manager`，允许为 `None` 以支持惰性构建。 |
| `AddFactTool.manager` | 只读属性，`self._manager` 为空时惰性构建并缓存默认 `MemoryManager`，否则直接返回。 |
| `AddFactTool.execute` | 执行入口，把校验后的入参转给模块级 `add_fact`，再用返回的 `item.id` 组装 `AddFactOutput`。 |
| `create_tool` | 无参工厂函数，返回一个新的 `AddFactTool` 实例供工具注册表使用。 |
