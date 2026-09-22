# core/registry.py

## 一、这个文件是干什么的

这个文件是整个 Agent 运行时里「工具（tool）注册与查找」的中枢，定义了两样东西：抽象基类 `BaseTool` 和具体实现类 `ToolRegistry`。`BaseTool` 规定了一个可被运行时执行的工具必须长什么样——它必须带一个 `ToolSpec` 类型的 `spec` 属性（描述名字、版本、输入输出 Pydantic 模型、确认键、schema 哈希等契约信息），并且必须实现 `execute(arguments)` 方法，接收一个已经通过校验的参数模型、返回与 `output_model` 匹配的数据。`ToolRegistry` 则是一个线程安全的、以工具名为键的容器：它内部维护 `_tools`（名字到工具实例的映射）和 `_generations`（名字到注册代数的计数），并用一把 `threading.RLock` 保护所有读写。

它的核心设计意图是「可热替换 + 可审计」：每当同名工具被重新注册（`replace=True`），该名字的 generation 计数器就加一，于是任何依赖具体实现的凭证（例如破坏性操作的确认键、调用级确认键）都会自动失效，不会出现「用户批准的是旧实现，实际执行的是新实现」这种漏洞。`unregister` 特意保留 generation 计数，就是为了让代数序列在「注销—再注册」循环中保持单调递增，不会被回退成重复值。

文件里还有两类辅助能力：一是「稳定的快照」——`snapshot` 在锁内一次性取出一批工具及其代数的视图，保证一次 provider 请求期间看到的工具集合不会中途变化；二是「状态查询」——`is_registered` 和 `registration_status` 用版本号与 schema 哈希做契约匹配检查，方便上层在调用前判断当前注册的实现是否还是自己期望的那个版本。注册与注销动作会通过 `core.activity_log.log_tool_registration` 写一条活动日志，因此这个文件也是工具生命周期审计的入口。

## 二、函数与类逐条详解

### `BaseTool(ABC)` （第 16 行）

- **作用**：这是所有可执行工具的抽象基类，定义了运行时对「一个工具」的最小契约。它本身不提供任何执行逻辑，只声明两件事：第一，任何子类实例都必须拥有一个类型为 `ToolSpec` 的类属性（或实例属性）`spec`，用来描述该工具对外暴露的元数据与输入输出模型；第二，子类必须实现 `execute` 方法，供运行时在参数校验通过后真正执行工具动作。之所以需要它，是因为 `ToolRegistry.register` 会做 `isinstance(tool, BaseTool)` 的强校验，从而把「随便一个带 spec 的对象」挡在注册表之外，保证注册进来的东西一定是可执行的、契约完整的。它继承自 `abc.ABC`，因此只要子类没有实现全部抽象方法，实例化时就会抛 `TypeError`，把契约缺失提前暴露在开发期而不是运行期。
- **参数**：无（类定义本身不接收参数；`spec` 是类型注解为 `ToolSpec` 的类属性声明，不含默认值，实际由子类提供）。
- **返回**：不适用（构造 `BaseTool` 本身不可行，因为它含未实现的抽象方法）。
- **内部流程**：类体只有两条语句：一条 `spec: ToolSpec` 的注解式属性声明（只声明类型，不在基类赋值），一条被 `@abstractmethod` 装饰的 `execute` 定义。`ABC` 元类在创建子类时会收集 `__abstractmethods__`，任何未覆写 `execute` 的具体子类都无法被实例化。
- **异常/边界**：直接实例化 `BaseTool` 会因存在抽象方法而抛 `TypeError`；子类若忘记实现 `execute` 同样在实例化时报错。对 `spec` 的存在性，基类不做运行时检查，检查发生在 `ToolRegistry.register` 中。
- **同文件关系**：被 `ToolRegistry` 的所有方法在类型注解与运行时校验中引用（`register` 的 `isinstance` 校验、`resolve`/`maybe_resolve`/`snapshot`/`get` 的返回类型等）；它的抽象方法 `execute` 由本文件之外的具体工具子类实现，本文件内部不调用。

### `BaseTool.execute(self, arguments: BaseModel) -> BaseModel | Any` （第 19 行）

- **作用**：这是工具的执行入口契约。运行时在完成参数校验（通常是先由 `ToolSpec.input_model` 校验原始 JSON 参数）之后，会把校验后的参数模型交给这个方法，由具体工具完成实际动作，例如读写文件、发起网络请求、操作记忆系统等。返回值约定为「与 `output_model` 匹配的数据」，可以是 Pydantic 模型实例，也可以是任意可被序列化的普通 Python 值（因此返回类型写作 `BaseModel | Any`）。在基类里它只是一个占位声明，作用是强制每个工具子类都提供这个入口，并统一签名，使注册表可以在不知道具体工具类型的情况下统一调度。文档字符串明确写出「Execute validated arguments and return data matching output_model」，这句话同时也是对实现者的约束：参数已经是校验过的，不要在这里重复做类型转换。
- **参数**：`arguments`（`pydantic.BaseModel`）：已经过 `input_model` 校验的参数对象，不是原始 dict，也不是 JSON 字符串；具体字段取决于该工具自己的 `input_model` 定义。基类不检查其类型，不做默认值处理。
- **返回**：`BaseModel | Any`：与工具 `output_model` 匹配的结果。基类实现只抛异常，永不真正返回。
- **内部流程**：基类实现体内只有一句 `raise NotImplementedError`，作为「必须覆写」的兜底提示。被 `@abstractmethod` 装饰后，`ABC` 机制会在实例化阶段就拦截未覆写的子类，因此这句 `raise` 在正常情况下不会被触发，它的价值在于显式表达意图并防止 `super().execute(...)` 被误当成有效实现。
- **异常/边界**：基类实现必然抛 `NotImplementedError`；子类实现应自行处理参数为空、外部依赖失败等边界，基类不做任何兜底。参数为 `None` 时基类也不会提前报错，会一路走到 `NotImplementedError`。
- **同文件关系**：被 `BaseTool` 自身声明，被 `ToolRegistry` 间接依赖（注册表只保存实例、从不直接调用 `execute`，真正调用发生在本文件之外的调度层）；无本文件内的调用者。

### `ToolRegistry` （第 25 行）

- **作用**：这是工具注册表本体，也是本文件的主类。它以工具名 `name` 为唯一键，维护「名字 → 工具实例」的 `_tools` 字典和「名字 → 注册代数」的 `_generations` 字典，所有公开方法都在 `threading.RLock` 保护下读写，因此可以被多个线程（例如 Web 请求线程与后台任务线程）并发使用。它的职责可以概括为四类：注册与注销（`register`、`unregister`）、查找与稳定快照（`resolve`、`maybe_resolve`、`get`、`maybe_get`、`snapshot`、`specs`）、契约一致性校验（`is_registered`、`registration_status`）、以及供审批流程使用的确认键生成（`confirmation_key`、`call_confirmation_key`）。选择 `RLock` 而不是普通 `Lock`，是因为 `call_confirmation_key` 会先调 `resolve`、再调 `confirmation_key`，而 `confirmation_key` 内部又调 `resolve`，可重入锁避免了自死锁。这个类本身不做参数校验之外的业务逻辑，它是纯内存、无持久化的，进程重启后注册内容全部丢失，需要由上层启动流程重新注册。
- **参数**：无（类定义不接收参数）。
- **返回**：不适用。
- **内部流程**：类体依次定义 `__init__` 建立三个内部字段，随后按「写操作 → 键生成 → 查询解析 → 批量视图 → 便捷取值 → 契约校验 → 状态描述 → 列表导出 → 协议方法」的顺序组织方法。所有涉及 `_tools`/`_generations` 的访问都包在 `with self._lock:` 中，保证检查与写入是一个原子步骤（例如 `register` 中「判断是否已存在」和「写入」在同一个临界区内）。
- **异常/边界**：类本身不抛异常，但它的方法会按契约抛 `TypeError`、`ValueError`、`KeyError`（详见各方法条目）。内部字段初始为空，因此新建的注册表长度为 0、任何查找都返回「未注册」。
- **同文件关系**：内部方法之间互相调用关系紧密：`get` 调 `resolve`；`maybe_get` 调 `maybe_resolve`；`confirmation_key` 调 `resolve`；`call_confirmation_key` 调 `resolve` 与 `confirmation_key`；`is_registered` 与 `registration_status` 调 `maybe_resolve`；`specs` 调 `snapshot`；`__contains__` 与 `__len__` 直接读 `_tools`。外部由应用启动装配与调度层调用。

### `ToolRegistry.__init__(self) -> None` （第 26 行）

- **作用**：构造一个空注册表，建立后续所有操作依赖的三个内部字段。它不做任何 I/O、不注册任何默认工具、不读取配置，是一个纯粹的初始化步骤，因此可以廉价地在需要隔离的测试里反复创建。`_generations` 初始为空字典，意味着某个工具名第一次注册时其 generation 会被计算为 1（见 `register` 中的 `get(name, 0) + 1`）。`_lock` 在实例创建时就分配，保证此后所有并发访问共享同一把可重入锁。
- **参数**：`self`（`ToolRegistry`）：实例本身，隐式传入。
- **返回**：`None`。
- **内部流程**：依次赋值 `self._tools = {}`（`dict[str, BaseTool]`）、`self._generations = {}`（`dict[str, int]`）、`self._lock = threading.RLock()`。三步之间没有判断、没有循环、没有异常分支。
- **异常/边界**：无特殊处理（分配字典与锁在正常情况下不会失败；极端内存耗尽时由解释器抛 `MemoryError`，本方法不捕获）。
- **同文件关系**：被 `ToolRegistry` 的所有其他方法间接依赖（它们都读写这里建立的 `_tools`、`_generations`、`_lock`）；本方法不调用本文件里的任何函数。

### `ToolRegistry.register(self, tool: BaseTool, *, replace: bool = False) -> None` （第 31 行）

- **作用**：把一个工具实例登记进注册表，是工具生命周期的起点。它先做四层防御性校验（实例类型、`replace` 类型、`spec` 类型、输入输出模型必须是类），确保注册进去的东西在调度阶段不会因为契约缺失而爆炸，然后才在锁内写入并把该名字的 generation 加一。`replace=False`（默认）时同名工具会被拒绝，防止两个模块互相覆盖；`replace=True` 用于热替换场景（例如插件重载、配置切换后重建工具），此时旧实例被直接覆盖，且 generation 递增使得所有基于旧实现的确认键立即失效。写入完成后它会在锁外调用活动日志，把名字、新代数、当前全部已注册名字记录下来，形成可追溯的审计线索。
- **参数**：`tool`（`BaseTool`）：要注册的工具实例，必须是 `BaseTool` 的子类实例，且其 `spec` 必须是 `ToolSpec` 实例、`spec.input_model` 与 `spec.output_model` 必须都是类（Pydantic 模型类）；`replace`（`bool`，仅关键字参数，默认 `False`）：同名已存在时是否允许覆盖，必须是布尔值，传非布尔值（例如 `1`）会抛 `TypeError`。
- **返回**：`None`（成功时无返回值，失败时抛异常）。
- **内部流程**：第一步 `isinstance(tool, BaseTool)` 校验，失败抛 `TypeError("tool must be a BaseTool instance")`；第二步 `isinstance(replace, bool)` 校验，失败抛 `TypeError("replace must be a boolean")`；第三步用 `getattr(tool, "spec", None)` 取 spec 并检查是否为 `ToolSpec` 实例，失败抛 `TypeError("tool.spec must be a ToolSpec instance")`；第四步用 `inspect.isclass` 检查 `spec.input_model` 与 `spec.output_model` 都是类，失败抛 `TypeError`；接着取出 `name = spec.name`，进入 `with self._lock`：若 `name` 已在 `_tools` 中且 `replace` 为假，抛 `ValueError(f"tool '{name}' is already registered")`；否则写入 `self._tools[name] = tool`，并把 `self._generations[name]` 设为原值加一（`self._generations.get(name, 0) + 1`，所以首次注册得到 1）；随后把新代数存入局部变量 `generation`，并用 `tuple(self._tools)` 在锁内冻结当前全部名字，形成 `registered_names`；退出锁后调用 `log_tool_registration(name, generation, registered_names)`。
- **异常/边界**：`TypeError` 覆盖 tool 类型不对、replace 不是布尔、spec 缺失或类型不对、输入输出模型不是类四种情况；`ValueError` 覆盖同名重复注册且未允许替换的情况；`spec.name` 为空字符串或非字符串时本方法不校验（校验责任在 `ToolSpec` 自身），因此理论上可以注册出空名工具，后续按名查找会因空名校验而失败。空值（`None`）传入会先撞上 `isinstance` 校验并抛 `TypeError`。
- **同文件关系**：不调用本文件中的其他方法，只读写 `self._tools` 与 `self._generations`；被本文件之外的装配/插件加载代码调用，其结果被本文件所有查询方法消费。

### `ToolRegistry.unregister(self, name: str) -> BaseTool` （第 53 行）

- **作用**：按名字从注册表里移除一个工具，并把被移除的实例返回给调用方，方便调用方做后续清理（例如关闭该工具持有的连接、通知订阅者）。文档字符串特别强调「per-name generation 计数器会被保留」，这是有意为之：如果注销时把 `_generations[name]` 一并删掉，那么重新注册同名工具时代数会从 1 重新开始，先前签发的确认键就可能与新实现碰撞，造成审批绕过；保留计数器则让代数严格单调递增。与 `register` 不同，它在移除时不写活动日志，因为日志函数只覆盖注册事件。它是「热卸载」能力的基础，插件被禁用或工具实现被废弃时会用到。
- **参数**：`name`（`str`）：要移除的工具名，必须是非空字符串；传空字符串或非字符串会抛 `ValueError`。
- **返回**：`BaseTool`：被移除的那个工具实例（与注册时传入的是同一个对象）。
- **内部流程**：先做 `not isinstance(name, str) or not name` 校验，失败抛 `ValueError("tool name must be a non-empty string")`；然后 `with self._lock:`，用 `self._tools.pop(name, None)` 尝试弹出；若结果为 `None`（说明该名字从未注册），在锁内抛 `KeyError(f"tool '{name}' is not registered")`；否则把 `tool` 返回。注意 `_generations` 在此过程中完全不被修改。
- **异常/边界**：非字符串或空字符串抛 `ValueError`；名字不存在抛 `KeyError`；由于 `pop` 的默认值是 `None`，注册表中不可能存在值为 `None` 的条目，所以「值恰为 None」不会造成误判。重复注销同一名字第二次必然抛 `KeyError`。
- **同文件关系**：不调用本文件中的其他方法；被本文件之外的卸载/重载逻辑调用。它的存在与 `register` 的 generation 递增机制配套，共同保证 `confirmation_key` 与 `call_confirmation_key` 的失效语义。

### `ToolRegistry.confirmation_key(self, name: str) -> str` （第 69 行）

- **作用**：为某个工具生成「实现级确认键」。返回值由工具 `spec` 上的 `confirmation_key` 字段与当前注册代数拼成，形如 `"<spec.confirmation_key>:<generation>"`。它的语义是：只要该名字下注册的实现发生了变化（重新注册使 generation 增加），这个键就必然改变，于是任何以旧键为前提做出的用户审批（例如「允许执行这个破坏性工具」）都会自动作废，必须重新征求确认。这是把「用户批准」与「具体实现版本」绑定起来的关键机制，防止工具被悄悄替换后沿用旧的授权。
- **参数**：`name`（`str`）：工具名，必须是非空字符串；校验由内部调用的 `resolve` 完成，非法值会由 `resolve` 抛 `ValueError`。
- **返回**：`str`：`f"{tool.spec.confirmation_key}:{generation}"`，即实现标识与代数用冒号连接的字符串。
- **内部流程**：调用 `self.resolve(name)`，在锁内一次性拿到 `(tool, generation)` 这一对原子结果（避免先取工具再取代数时中间被重新注册打断）；然后直接做 f-string 拼接并返回。
- **异常/边界**：`name` 非字符串或为空时由 `resolve` 抛 `ValueError`；名字未注册时由 `resolve` 抛 `KeyError`；若 `spec.confirmation_key` 本身是空字符串，本方法不报错，只会返回形如 `":3"` 的键，是否合法取决于 `ToolSpec` 的定义。
- **同文件关系**：调用本文件的 `resolve`；被本文件的 `call_confirmation_key` 调用；也被本文件之外的审批/权限模块调用。

### `ToolRegistry.call_confirmation_key(self, name: str, arguments: dict[str, Any]) -> str` （第 74 行）

- **作用**：生成「调用级确认键」，把破坏性操作的授权精确绑定到一次具体的参数对象上。它先取实现级确认键（含 spec 标识与代数），再对参数做规范化序列化并取 SHA-256 摘要，最终返回 `"<实现级确认键>:<参数摘要>"`。这样做的意义是：同一个工具、同一个实现版本，只要参数有一丁点不同（例如删除的文件路径不同），确认键就完全不同，用户对 A 参数的批准绝不会被复用到 B 参数上。规范化过程保证了「语义相同但写法不同」的输入（键顺序不同、JSON 表示不同）得到同一个摘要，从而避免因字典顺序抖动导致合法审批被误判为失效。
- **参数**：`name`（`str`）：工具名，非空字符串，非法值由 `resolve` 抛 `ValueError`、未注册由 `resolve` 抛 `KeyError`；`arguments`（`dict[str, Any]`）：待执行的原始参数字典，必须是 `dict` 类型（传 `None`、列表、Pydantic 模型都会抛 `TypeError`），其内容必须能通过该工具 `spec.input_model` 的严格校验。
- **返回**：`str`：`f"{实现级确认键}:{sha256 十六进制摘要}"`。
- **内部流程**：第一步 `isinstance(arguments, dict)` 校验，失败抛 `TypeError("arguments must be a dict")`；第二步 `tool, _ = self.resolve(name)` 取工具（代数在这里被丢弃，因为随后的 `confirmation_key` 会重新取一次，保证键里带的是同一时刻的代数）；第三步 `tool.spec.input_model.model_validate(arguments, strict=True)` 做严格模式校验，得到模型实例；第四步 `.model_dump(mode="json")` 转成纯 JSON 兼容结构（枚举、日期等被转成可序列化形式）；第五步 `json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)` 做确定性序列化——键排序消除顺序差异、紧凑分隔符消除空白差异、`ensure_ascii` 统一非 ASCII 转义、`allow_nan=False` 拒绝 NaN/Infinity 这类非标准 JSON 值；第六步 `hashlib.sha256(encoded.encode("utf-8")).hexdigest()` 得到摘要；最后与 `self.confirmation_key(name)` 拼接返回。
- **异常/边界**：`arguments` 不是 dict 抛 `TypeError`；名字非法抛 `ValueError`、未注册抛 `KeyError`；参数不符合 `input_model` 时由 Pydantic 抛 `ValidationError`（严格模式下类型不匹配也会被拒）；参数中含 `float('nan')` 或无穷大时 `json.dumps` 抛 `ValueError`；模型中含无法 JSON 化的对象时 `model_dump(mode="json")` 或 `json.dumps` 会抛序列化错误。以上异常均不捕获、直接向上传播。
- **同文件关系**：调用本文件的 `resolve` 和 `confirmation_key`；未被本文件内其他方法调用，由外部的审批流程使用。

### `ToolRegistry.resolve(self, name: str) -> tuple[BaseTool, int]` （第 93 行）

- **作用**：这是注册表的「严格查找」原语：在锁内一次性返回工具实例与它当前的注册代数，保证这两个值来自同一个一致的状态快照。之所以要返回元组而不是只返回工具，是因为调用方经常需要「我拿到的这个实现是不是还是我以为的那一版」这个信息（例如决定确认键、决定是否复用缓存的能力描述）。与 `maybe_resolve` 的区别在于名字不存在时它直接抛 `KeyError`，属于「必须存在」的语义，适合在已经确认过工具存在、或缺失即视为编程错误的路径上使用。
- **参数**：`name`（`str`）：工具名，必须是非空字符串；空字符串或非字符串抛 `ValueError`。
- **返回**：`tuple[BaseTool, int]`：`(工具实例, 当前注册代数)`，代数为正整数（首次注册为 1，之后每次 `replace` 注册递增）。
- **内部流程**：先做非空字符串校验，失败抛 `ValueError("tool name must be a non-empty string")`；然后 `with self._lock:`，在 `try` 块中返回 `self._tools[name], self._generations[name]`；若任一字典缺键触发 `KeyError`，用 `raise KeyError(f"tool '{name}' is not registered") from exc` 重新抛出，保留原始异常链（`from exc`）以便调试。
- **异常/边界**：`ValueError`（名字非法）、`KeyError`（未注册）。由于 `_tools` 与 `_generations` 在 `register` 中总是成对写入，且 `unregister` 只删 `_tools` 不删 `_generations`，所以实际缺键只可能发生在 `_tools` 上；`_generations` 的读取不会单独失败。
- **同文件关系**：被本文件的 `confirmation_key`、`call_confirmation_key`、`get` 调用；它本身只读 `_tools` 与 `_generations`，不调用本文件其他方法。

### `ToolRegistry.maybe_resolve(self, name: str) -> tuple[BaseTool, int] | None` （第 103 行）

- **作用**：这是注册表的「宽松查找」原语，语义与 `resolve` 相同但把「不存在」当作正常结果返回 `None`，而不是抛异常。它适用于探测型场景：例如上层想知道某个可选工具是否可用、或在批量处理中跳过缺失项，此时用异常控制流程既慢又难读。它对非法名字（非字符串或空串）也选择直接返回 `None` 而不是抛 `ValueError`，这使得调用方可以无脑地把任意用户输入或配置值传进来做存在性探测，不会因为脏输入炸掉整条链路。
- **参数**：`name`（`str`）：工具名；传入非字符串或空字符串时不做报错，直接返回 `None`。
- **返回**：`tuple[BaseTool, int] | None`：注册存在时返回 `(工具实例, 代数)`；名字非法或未注册时返回 `None`。
- **内部流程**：先判断 `not isinstance(name, str) or not name`，成立则立即返回 `None`；否则 `with self._lock:`，用 `self._tools.get(name)` 取工具，若为 `None` 返回 `None`；否则返回 `(tool, self._generations[name])`。整个「取值 + 读代数」在同一临界区内完成，保持原子性。
- **异常/边界**：正常路径不抛任何异常；只有极端情况（例如 `_generations` 缺键，理论上不会发生）才可能抛 `KeyError`。空值、非字符串一律返回 `None`，不做日志、不做提示。
- **同文件关系**：被本文件的 `maybe_get`、`is_registered`、`registration_status` 调用；自身只读内部字典。

### `ToolRegistry.snapshot(self, names: list[str] | tuple[str, ...] | None = None) -> dict[str, tuple[BaseTool, int]]` （第 112 行）

- **作用**：在锁内一次性抓取「名字 → (工具, 代数)」的完整视图，供一次 provider 请求期间使用。它的价值在于稳定性：调用方拿到快照后，即使其他线程在请求处理过程中注册或替换了工具，快照里的内容也不会变，从而保证一次请求内工具集合与版本完全自洽，不会出现「前半段用旧实现、后半段用新实现」的错乱。`names` 为 `None` 时返回全量快照；传入名字列表/元组时只返回这些名字，且只要有一个名字缺失就整体失败，属于「要么全给、要么不给」的严格语义，避免调用方在拿到部分结果后误以为请求的工具都已就绪。
- **参数**：`names`（`list[str] | tuple[str, ...] | None`，默认 `None`）：`None` 表示导出全部已注册工具；传列表或元组表示只导出指定名字，元素必须都是非空字符串。传其他类型（如 `set`、`str`、`dict`）或元素含空串/非字符串时抛 `TypeError`。
- **返回**：`dict[str, tuple[BaseTool, int]]`：键为工具名，值为 `(工具实例, 代数)` 元组。`names` 给定时，返回字典的键集合与传入名字集合一致（但顺序按传入顺序构建），重复名字会被后写覆盖，因此结果长度可能小于传入长度。
- **内部流程**：先做参数校验：若 `names is not None` 且（不是 `list`/`tuple`，或存在元素不是非空字符串），抛 `TypeError("names must be a list of non-empty strings")`；然后 `with self._lock:`；若 `names is None`，用字典推导遍历 `self._tools.items()`，为每个名字配上 `self._generations[name]` 返回；否则先用列表推导收集 `missing = [name for name in names if name not in self._tools]`，若非空则以第一个缺失名字抛 `KeyError(f"tool '{missing[0]}' is not registered")`；否则用字典推导按 `names` 顺序构建结果返回。
- **异常/边界**：`TypeError`（names 类型或元素非法）、`KeyError`（指定的某个名字未注册，只报告第一个缺失的名字）。传入空列表 `[]` 会合法返回空字典；`names=None` 且注册表为空时返回空字典。
- **同文件关系**：被本文件的 `specs` 调用（`specs` 用 `snapshot()` 取全量视图再提取 spec）；自身只读 `_tools` 与 `_generations`，不调用本文件其他方法。

### `ToolRegistry.get(self, name: str) -> BaseTool` （第 134 行）

- **作用**：这是最常用的便捷取值方法，只关心工具实例、不关心代数。它把 `resolve` 返回的元组丢掉第二个元素，等价于 `resolve(name)[0]`，语义是「名字必须存在，否则报错」。适合在已经确定工具已注册、只想拿到实例去读 `spec` 或交给调度层的场景；相比 `maybe_get`，它把缺失视为异常，能让配置或装配错误尽早暴露，而不是静默变成 `None` 后在更远的地方引发 `AttributeError`。
- **参数**：`name`（`str`）：工具名，必须是非空字符串（由 `resolve` 校验）。
- **返回**：`BaseTool`：对应名字注册的工具实例。
- **内部流程**：单行实现：调用 `self.resolve(name)`，取返回元组的下标 0 返回。锁与一致性保证全部由 `resolve` 提供。
- **异常/边界**：名字非法抛 `ValueError`，未注册抛 `KeyError`，二者都由 `resolve` 抛出，本方法不额外处理。
- **同文件关系**：调用本文件的 `resolve`；本文件内没有其他方法调用 `get`，它主要由外部代码使用。

### `ToolRegistry.maybe_get(self, name: str) -> BaseTool | None` （第 137 行）

- **作用**：`get` 的宽松版本，只返回工具实例；名字非法或未注册时返回 `None` 而不抛异常。它把 `maybe_resolve` 的元组结果解包，供「探测可选工具」的代码路径使用，例如运行时想尝试调用一个可能被插件提供的增强工具，没有就降级走默认逻辑。使用它时调用方必须显式判断 `None`，因此不会意外吞掉「工具不存在」这一信息，只是把异常变成了返回值。
- **参数**：`name`（`str`）：工具名；非字符串或空字符串不会报错，会得到 `None`。
- **返回**：`BaseTool | None`：存在时返回工具实例，否则返回 `None`。
- **内部流程**：调用 `self.maybe_resolve(name)` 得到 `registration`；若为 `None` 返回 `None`，否则返回 `registration[0]`（工具实例，代数被丢弃）。
- **异常/边界**：正常路径不抛异常；空值/非法名字返回 `None`。
- **同文件关系**：调用本文件的 `maybe_resolve`；本文件内无其他调用者，由外部代码使用。

### `ToolRegistry.is_registered(self, name: str, *, version: str | None = None, schema_hash: str | None = None) -> bool` （第 141 行）

- **作用**：判断某个名字当前注册的工具是否满足指定的契约条件——版本号匹配、schema 哈希匹配。两个条件都是可选的，且采用「给定了才比较」的语义：只给 `version` 就只比版本，只给 `schema_hash` 就只比哈希，两个都不给就退化为纯存在性检查。这在需要跨进程或跨模块协作时很有用：调用方记录了自己期望的工具版本与 schema 指纹，用这个方法确认当前运行时挂载的确实是同一份契约，避免因为版本漂移导致参数结构对不上却仍然发起调用。它对未注册的名字返回 `False` 而非抛异常，因此可以安全地用作前置守卫。
- **参数**：`name`（`str`）：工具名，必须是非空字符串，否则抛 `ValueError`；`version`（`str | None`，仅关键字，默认 `None`）：期望的版本号，给定时必须是非空白字符串，否则抛 `ValueError`；`schema_hash`（`str | None`，仅关键字，默认 `None`）：期望的 schema 哈希，给定时必须是非空白字符串，否则抛 `ValueError`。
- **返回**：`bool`：`True` 表示该名字已注册且所有给定条件都匹配；`False` 表示未注册，或注册了但版本/哈希与期望不符。
- **内部流程**：依次校验 `name` 为非空字符串、`version` 为 `None` 或非空白字符串（用 `.strip()` 判空）、`schema_hash` 为 `None` 或非空白字符串；随后调 `self.maybe_resolve(name)`，为 `None` 直接返回 `False`；否则解包 `tool, _`（代数被忽略，因为这里比较的是契约而非实现代数），返回布尔表达式 `(version is None or tool.spec.version == version) and (schema_hash is None or tool.spec.schema_hash == schema_hash)`。两个条件是与关系，必须同时满足。
- **异常/边界**：`ValueError`（name 非法、version 为空白字符串、schema_hash 为空白字符串）；未注册返回 `False` 而不是抛 `KeyError`；`spec.version` 或 `spec.schema_hash` 为 `None` 时，与给定字符串比较会得到 `False`（视为不匹配），不会抛异常。
- **同文件关系**：调用本文件的 `maybe_resolve`；本文件内无其他调用者，由外部契约校验逻辑使用。

### `ToolRegistry.registration_status(self, name: str) -> dict[str, Any]` （第 167 行）

- **作用**：把一个工具注册状态整理成 JSON 友好的字典，供诊断接口、状态页或日志使用。它统一了「已注册」与「未注册」两种情况的输出结构：未注册时也返回包含全部键的字典（只是 `registered` 为 `False`、其余字段为 `None`），这样消费方可以无条件地按固定字段取值，不必先判空。已注册时它额外给出实现来源标识 `implementation`，格式为 `"模块名:限定类名"`，用于回答「当前挂载的到底是哪个类的实现」这个运维上非常实际的问题（例如同一个工具名被两个插件各自实现）。
- **参数**：`name`（`str`）：工具名，必须是非空字符串，否则抛 `ValueError`。
- **返回**：`dict[str, Any]`：固定包含 `name`、`registered`、`version`、`schema_hash`、`generation`、`implementation` 六个键。未注册时 `registered=False`，`version`/`schema_hash`/`generation`/`implementation` 均为 `None`；已注册时 `registered=True`，`version` 与 `schema_hash` 取自 `tool.spec`，`generation` 为当前代数，`implementation` 为 `f"{type(tool).__module__}:{type(tool).__qualname__}"`。
- **内部流程**：先校验 `name` 为非空字符串，失败抛 `ValueError("tool name must be a non-empty string")`；调 `self.maybe_resolve(name)`；若为 `None`，直接返回预置的未注册字典；否则解包 `tool, generation`，用 `type(tool).__module__` 与 `type(tool).__qualname__` 拼出实现标识（用 `__qualname__` 而非 `__name__`，因此嵌套类也能得到完整路径），组装并返回已注册字典。
- **异常/边界**：仅 `ValueError`（name 非法）；未注册是正常分支，返回结构化结果而非异常。若 `tool.spec.version` 或 `schema_hash` 为 `None`，会原样写入字典，调用方需自行处理。
- **同文件关系**：调用本文件的 `maybe_resolve`；本文件内无其他调用者，由外部状态查询/诊断代码使用。

### `ToolRegistry.specs(self) -> list[ToolSpec]` （第 192 行）

- **作用**：导出当前所有已注册工具的 `ToolSpec` 列表，典型用途是生成发给模型 provider 的工具声明（tool schema）、渲染工具清单页面、或做整体契约审计。它通过 `snapshot()` 取全量视图后再提取 spec，因此整个过程在一次锁内完成，得到的列表是同一时刻的一致快照，不会因为并发注册而出现「列表长度与内容不匹配」的问题。返回顺序为 `_tools` 字典的插入顺序，即注册先后顺序，通常对上层展示是可预期的。
- **参数**：无（只有 `self`）。
- **返回**：`list[ToolSpec]`：按注册顺序排列的全部工具规格对象列表；注册表为空时返回空列表。
- **内部流程**：调用 `self.snapshot()`（无参，取得全量 `dict[str, tuple[BaseTool, int]]`），对其 `.values()` 做列表推导，每项解包为 `tool, _`（代数被丢弃），取出 `tool.spec` 组成列表返回。
- **异常/边界**：无特殊处理；空注册表返回 `[]`；不校验 spec 内容（校验在 `register` 阶段已完成）。
- **同文件关系**：调用本文件的 `snapshot`；本文件内无其他调用者，由外部 provider 适配层或展示层使用。

### `ToolRegistry.__contains__(self, name: str) -> bool` （第 195 行）

- **作用**：实现 `in` 运算符，使 `name in registry` 这种写法生效。它只做纯粹的键存在性判断，不涉及任何契约校验、也不检查代数，是「这个工具名有没有被注册」的最轻量问法。相比 `is_registered`，它没有版本/哈希过滤、没有参数校验（非字符串也照样走字典查找并返回 `False`），适合在条件表达式里快速分流。加锁是为了避免在另一个线程正在写入 `_tools` 时读到中间状态。
- **参数**：`name`（`str`）：任意对象都可传入（注解标注为 `str`，但运行时不做类型校验）；不可哈希的对象（如 `list`）会导致字典查找抛 `TypeError`。
- **返回**：`bool`：`True` 表示 `_tools` 中存在该键，否则 `False`。
- **内部流程**：`with self._lock:` 内执行 `return name in self._tools`，依赖字典的哈希查找，平均 O(1)。
- **异常/边界**：空字符串返回 `False`；`None` 返回 `False`（`None` 可哈希且不在字典中）；不可哈希类型抛 `TypeError`，本方法不捕获。
- **同文件关系**：不调用本文件其他方法，直接读 `self._tools`；由外部代码通过 `in` 语法触发。

### `ToolRegistry.__len__(self) -> int` （第 199 行）

- **作用**：实现 `len()`，返回当前注册表中工具的总数，用于监控、日志、断言和状态展示。它统计的是唯一的工具名数量，因此同名工具被替换注册时计数不变（替换是覆盖而非追加），只有新名字加入或名字被注销才会改变结果。加锁保证在并发注册/注销时读到的不是撕裂的中间值。
- **参数**：无（只有 `self`）。
- **返回**：`int`：`_tools` 中键的数量，非负整数；空注册表返回 0。
- **内部流程**：`with self._lock:` 内执行 `return len(self._tools)`。
- **异常/边界**：无特殊处理，不抛异常。
- **同文件关系**：不调用本文件其他方法，直接读 `self._tools`；由外部代码通过 `len()` 触发。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `BaseTool` | 抽象基类，规定工具必须带 `ToolSpec` 类型的 `spec` 并实现 `execute`。 |
| `BaseTool.execute` | 抽象执行入口，接收已校验的参数模型并返回匹配 `output_model` 的结果。 |
| `ToolRegistry` | 线程安全的工具注册表，按名字管理工具实例、注册代数与契约校验。 |
| `ToolRegistry.__init__` | 初始化空的 `_tools`、`_generations` 字典与可重入锁 `_lock`。 |
| `ToolRegistry.register` | 校验并登记工具实例，同名按 `replace` 决定是否覆盖，并把代数加一、写注册日志。 |
| `ToolRegistry.unregister` | 按名移除工具并返回该实例，保留代数计数器以维持代数单调。 |
| `ToolRegistry.confirmation_key` | 生成「spec 确认键 + 当前代数」的实现级确认键，实现变更即失效。 |
| `ToolRegistry.call_confirmation_key` | 把实现级确认键与规范化参数的 SHA-256 摘要绑定，生成调用级确认键。 |
| `ToolRegistry.resolve` | 在锁内原子返回工具实例与其当前代数，名字缺失时抛 `KeyError`。 |
| `ToolRegistry.maybe_resolve` | `resolve` 的宽松版，名字非法或未注册时返回 `None`。 |
| `ToolRegistry.snapshot` | 锁内抓取全量或指定名字的「工具 + 代数」稳定视图，缺失名字整体报错。 |
| `ToolRegistry.get` | 便捷取值，返回工具实例，缺失即抛异常。 |
| `ToolRegistry.maybe_get` | `get` 的宽松版，缺失或名字非法时返回 `None`。 |
| `ToolRegistry.is_registered` | 按可选的版本号与 schema 哈希判断当前注册是否满足期望契约。 |
| `ToolRegistry.registration_status` | 输出 JSON 友好的注册详情，含是否注册、版本、哈希、代数与实现类路径。 |
| `ToolRegistry.specs` | 导出全部已注册工具的 `ToolSpec` 列表，用于生成 provider 工具声明。 |
| `ToolRegistry.__contains__` | 实现 `in`，快速判断某工具名是否已注册。 |
| `ToolRegistry.__len__` | 实现 `len()`，返回当前注册的工具名数量。 |
