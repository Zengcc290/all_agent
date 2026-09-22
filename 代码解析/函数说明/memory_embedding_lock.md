# memory/embedding_lock.py

## 一、这个文件是干什么的

这个文件是记忆系统里「嵌入模型身份锁」的唯一权威实现，核心职责是防止不同嵌入空间被悄悄混用。项目的设计前提是：SQLite 是真相源（truth），向量库（Qdrant 之类的 vector_store）只是 SQLite 数据的一份「投影」（projection）；如果写入时用的嵌入模型或维度跟库里已有向量不一致，检索结果会被静默污染，而且很难事后发现。因此本模块把「当前锁定使用的 (model, dimension)」持久化到 SQLite 里（通过 `DocumentRepository` 的 `EmbeddingLockRecord`），在每一次向量写入前做一次闸门校验，不一致就直接拒绝写入并抛出可转成 HTTP 409 的错误。

文件里主要包含四类东西：第一是数据结构与异常，即冻结数据类 `EmbeddingIdentity` 和异常类 `EmbeddingLockMismatch`（它自带中文提示文案和可序列化成接口详情的 `to_detail()`）；第二是「探测当前身份」的辅助函数 `embedding_model_name`、`resolve_embedding_identity`、`live_vector_dimension`；第三是核心闸门与检查逻辑 `inspect_embedding_lock`（只读、不写）和 `apply_embedding_lock`（会写、会抛错、可触发重建）；第四是重建与重灌流程 `reindex_vector_projection`（把 SQLite 里每一个唯一 id 重新投影一次）以及它的便捷封装 `rebuild_vector_projection`。

典型使用路径是：Web 层或记忆写入路径在写向量之前调用 `apply_embedding_lock`；如果它抛出 `EmbeddingLockMismatch`，上层把异常转成 409 并把 `to_detail()` 的结果返回给前端，前端提示用户「继续将重建向量投影并全量重灌 / 取消则保持锁定配置」；用户确认后再次调用时带上 `confirm_rebuild=True`，这时模块会重建向量集合、按 SQLite 真相重新嵌入并重灌全部数据，最后把锁推进到新的 (model, dimension)。另外 `mismatch_from_exception` 用来把 Qdrant 自己抛出的维度错误（例如 "dimension mismatch"）归一化成同一种 409 载荷，保证用户无论从哪条路径撞上不一致，看到的都是同一套提示。

文件顶部的两个哨兵常量承担特殊语义：`UNKNOWN_EMBEDDING_MODEL`（`"__unlocked_existing_data__"`）表示「库里有数据但锁是空的、来源模型未知」，`REBUILDING_EMBEDDING_MODEL`（`"__rebuild_in_progress__"`）表示「正在进行重灌，锁处于中间态」。文件末尾的 `__all__` 声明了对外公开的 10 个名字，说明上述 API 就是这个模块的完整对外契约。

## 二、函数与类逐条详解

### `EmbeddingIdentity` （第 23 行）
- **作用**：这是一个用 `@dataclass(frozen=True)` 声明的不可变数据类，用来表达「当前活跃的嵌入身份」，也就是一对 (模型名, 向量维度)。它的存在意义是给「当前配置的嵌入器」和「库里锁定的嵌入器」提供一个统一的、可直接比较的值对象：因为是 frozen 的，实例创建后不能被改写，可以安全地在函数之间传递而不用担心被下游偷偷修改；又因为 dataclass 默认生成了 `__eq__`，两个 `EmbeddingIdentity` 可以直接用 `==` 判断模型和维度是否完全一致。本文件里它被 `resolve_embedding_identity` 构造出来，作为 `EmbeddingLockMismatch` 的 `current` 字段、`inspect_embedding_lock` 快照里的 `_current`、以及 `reindex_vector_projection` 的 `identity` 入参使用。它本身不含任何业务逻辑，纯粹是数据载体。
- **参数**：作为 dataclass，其构造参数就是两个字段。`model: str`，嵌入模型的名字，通常是配置里的模型标识（例如某个 bge / text-embedding 模型名），也可能是回退后的类名；没有默认值，必须显式传入。`dimension: int`，向量维度，必须是正整数（语义上的约束，由使用方保证）；没有默认值，必须显式传入。
- **返回**：它是类，实例化后返回一个 `EmbeddingIdentity` 对象；不返回其他值。
- **内部流程**：由 dataclass 装饰器自动生成 `__init__`、`__repr__`、`__eq__`，并由 `frozen=True` 使实例在 `__setattr__` 层面被冻结，任何对 `model` 或 `dimension` 的赋值都会抛 `FrozenInstanceError`（`dataclasses.FrozenInstanceError`，`AttributeError` 的子类）。类体内只声明了两个带类型注解的字段，没有自定义方法。
- **异常/边界**：构造时若缺少参数会抛 `TypeError`；构造后尝试修改字段会抛 `dataclasses.FrozenInstanceError`。对 `model` 为空字符串或 `dimension` 为 0/负数没有做校验，本类不做任何合法性检查，交由 `resolve_embedding_identity` 之类的调用方把关。
- **同文件关系**：被 `resolve_embedding_identity` 构造（唯一构造点）；作为类型注解出现在 `EmbeddingLockMismatch.__init__`、`apply_embedding_lock`、`mismatch_from_exception`、`reindex_vector_projection`、`rebuild_vector_projection` 的签名中；被 `EmbeddingLockMismatch` 保存为 `current` 字段。它不调用本文件里的任何函数。

### `EmbeddingLockMismatch` （第 29 行）
- **作用**：这是继承自 `ValueError` 的自定义异常类，代表「当前配置的嵌入与 SQLite 里锁定的嵌入不一致」这一业务错误。它不只是一个空壳异常，而是把两边身份（`locked` 与 `current`）都携带在实例上，并提供一个人类可读的中文提示 `message`，以及一个可直接作为 HTTP 响应体 / 错误详情返回的结构化字典 `to_detail()`。这样设计的好处是：写入路径只需要抛出这一个异常类型，Web 层既能拿到 `except ValueError` 的兼容性（因为父类是 `ValueError`），又能通过 `to_detail()` 直接生成带 `code` 的 409 载荷，不需要在 Web 层重新拼装文案。它还承担「是否需要用户确认重建」的语义载体作用：文案明确告诉用户「继续将重建向量投影并全量重灌；取消则保持锁定配置」。
- **参数**：作为类，其构造签名由 `__init__` 定义（见下一条）；类本身没有其他类属性。
- **返回**：类，实例化返回异常对象；不返回业务数据。
- **内部流程**：类体内定义了 `__init__`、`message` 属性、`to_detail` 方法三个成员；异常基类为 `ValueError`，因此 `except ValueError` 与 `except Exception` 都能捕获它。类文档字符串明确其语义为「Current embedding does not match the SQLite lock.」。
- **异常/边界**：它本身就是异常类型，构造过程不额外抛错（除了 `__init__` 中调用的 `super().__init__` 与 `message` 属性求值可能因字段缺失而失败）。没有对 `locked` / `current` 为 `None` 做防护，传入 `None` 会在 `message` 求值时抛 `AttributeError`。
- **同文件关系**：被 `apply_embedding_lock` 与 `mismatch_from_exception` 抛出/构造；`mismatch_from_exception` 也会把它原样透传。它自身调用 `self.message` 与 `self.to_detail` 的内部成员，不调用本文件的其他模块级函数。

#### `__init__(self, locked: EmbeddingLockRecord, current: EmbeddingIdentity) -> None` （第 32 行）
- **作用**：构造异常实例，把「库内锁定记录」和「当前实际身份」这两个对照物保存到实例属性上，然后把人类可读的中文提示交给 `ValueError` 基类，使 `str(exc)` 直接就是那段提示文案。之所以在 `__init__` 里调用 `super().__init__(self.message)` 而不是先算好字符串存起来，是因为 `message` 被实现成 property（每次读取时根据 `locked`/`current` 现算），必须先把两个字段赋值好、再取 property 才能拿到正确文案。这样后续 `to_detail()` 与日志打印都能拿到一致的描述。
- **参数**：`locked: EmbeddingLockRecord`，从 SQLite 读到的锁定记录，必须带 `model` 与 `dimension` 属性（以及可能的 `updated_at`）；无默认值，通常来自 `repository.get_embedding_lock()` 或在 `inspect_embedding_lock` 里临时构造的 `EmbeddingLockRecord`。`current: EmbeddingIdentity`，当前活跃嵌入身份，必须带 `model` 与 `dimension`；无默认值，通常来自 `resolve_embedding_identity`。
- **返回**：`None`；作为构造器返回新建的异常实例。
- **内部流程**：第一步把入参 `locked` 绑定到 `self.locked`；第二步把 `current` 绑定到 `self.current`；第三步调用 `super().__init__(self.message)`，即先通过 property 求出中文提示字符串，再把它作为 `ValueError` 的 args[0]，从而让 `str(exc)`、`repr(exc)` 都带上这段文案。
- **异常/边界**：如果传入的 `locked` 或 `current` 是 `None` 或缺少 `model` / `dimension` 属性，第三步求值 `self.message` 时会抛 `AttributeError`。不校验模型名是否为空、维度是否为正。无其他特殊处理。
- **同文件关系**：调用 `self.message`（同文件内 `message` property）；被 `apply_embedding_lock` 直接构造抛出，也被 `mismatch_from_exception` 构造返回；`to_detail` 依赖本方法设置的 `self.locked` / `self.current`。

#### `message` （第 38 行，`@property`）
- **作用**：这是一个只读属性，用来生成面向用户的中文不一致提示。它把两边身份拼成一句话：「嵌入锁定不一致：库内是 X / N 维，当前是 Y / M 维。继续将重建向量投影并全量重灌；取消则保持锁定配置。」这句话是给前端确认弹窗用的，既说明了冲突事实，也说明了用户的两个选择及各自后果，因此不只是一条日志文案，而是产品交互的一部分。由于它是 property，每次访问都会基于当前 `self.locked` / `self.current` 重新拼接，不会出现「构造后字段被改而文案过期」的问题（尽管本类字段一般不再变动）。`__init__` 与 `to_detail` 都依赖它。
- **参数**：无参数（只有隐式 `self`）。属性访问时不接受任何实参，写成 `exc.message`。
- **返回**：返回 `str`，一段包含库内模型名与维度、当前模型名与维度以及两个操作选项的中文提示；只要 `self.locked` 与 `self.current` 上存在 `model` / `dimension`，就总能返回字符串，不存在返回 `None` 的分支。
- **内部流程**：使用一个隐式字符串拼接表达式，按顺序拼接四段：固定的「嵌入锁定不一致：库内是 」、`self.locked.model`、`" / "`、`self.locked.dimension` 与 `" 维，"`；随后是「当前是 」、`self.current.model`、`" / "`、`self.current.dimension`、`" 维。"`；最后接上「继续将重建向量投影并全量重灌；取消则保持锁定配置。」其中 `dimension` 是 `int`，靠 f-string 自动转成文本。整个方法无分支、无循环、无 I/O。
- **异常/边界**：若 `self.locked` 或 `self.current` 为 `None`，或对象上没有 `model` / `dimension` 属性，会抛 `AttributeError`。不处理空模型名、不处理负数维度（这些会被原样拼进文案）。无其他特殊处理。
- **同文件关系**：被 `__init__` 调用（用于设置异常 args）、被 `to_detail` 调用（作为 `message` 字段的值）；自身不调用本文件的其他函数。

#### `to_detail(self) -> dict[str, Any]` （第 45 行）
- **作用**：把异常转换成一个结构化字典，供上层（通常是 FastAPI 的异常处理器）直接作为 HTTP 409 的响应体返回。字典里带一个稳定的机器可读错误码 `embedding_lock_mismatch`、一段人类可读的 `message`，以及 `locked` 与 `current` 两个子对象（各含 `model` 与 `dimension`）。这样前端可以按 `code` 做逻辑分支（例如弹「确认重建」对话框），也可以按 `locked` / `current` 展示对比表格，而不必去解析中文句子。它是异常类与 Web 协议层之间的适配器。
- **参数**：无参数（只有隐式 `self`）。
- **返回**：返回 `dict[str, Any]`，固定包含四个键：`"code"`（恒为字符串 `"embedding_lock_mismatch"`）、`"message"`（等于 `self.message` 的中文提示）、`"locked"`（字典，含 `"model"` 与 `"dimension"`）、`"current"`（字典，同样含 `"model"` 与 `"dimension"`）。没有条件分支，任何情况下都返回这四个键。
- **内部流程**：直接构造并返回一个字典字面量：`code` 写死；`message` 读取 property `self.message`；`locked` 从 `self.locked.model` 和 `self.locked.dimension` 取；`current` 从 `self.current.model` 和 `self.current.dimension` 取。不做序列化（不调用 `json.dumps`），假定字段值本身可 JSON 序列化（模型名是 `str`，维度是 `int`，满足该假设）。
- **异常/边界**：若 `self.locked` / `self.current` 缺失或为 `None`，访问属性会抛 `AttributeError`；若 `dimension` 不是可序列化类型（例如被换成了自定义对象），则不是本方法抛错，而是上层序列化时才失败。不处理空值兜底，也没有默认值填充。
- **同文件关系**：调用 `self.message`（同文件 property）；不被本文件其他函数调用，供模块外部的 Web / API 层使用。

### `embedding_model_name(embedding: BaseEmbedding) -> str` （第 54 行）
- **作用**：从任意嵌入器对象上「尽力」取出一个可读的模型名字符串。因为项目里的嵌入器可能来自不同后端（本地模型、远程 API、测试用的假实现），有的实例带 `model` 属性、有的不带，直接访问会 `AttributeError`，所以这里用 `getattr(..., None)` 做安全探测。当实例上有非空字符串的 `model` 时就用它（并顺手 `strip()` 去掉首尾空白，避免「看起来一样其实带空格」导致锁比较失败）；否则退化为用类名（`type(embedding).__name__`）作为模型标识。这个函数是身份识别的基础，被 `resolve_embedding_identity` 复用，从而保证「当前身份」和「锁记录」的模型名比较用的是同一套规则。
- **参数**：`embedding: BaseEmbedding`，任何嵌入器实例，类型注解是 `memory.embedding.BaseEmbedding`，但实现上只要求它是对象，不强制校验类型。无默认值。约束：若其 `model` 属性存在且是「非空白字符串」则被采用；否则走类名回退。
- **返回**：返回 `str`。两种情况：`model` 属性是 `str` 且 `strip()` 后非空时，返回去除首尾空白后的模型名；否则返回 `type(embedding).__name__`（例如 `"MockEmbedding"`、`"BaseEmbedding"`）。不会返回空字符串，也不会返回 `None`。
- **内部流程**：第一步 `getattr(embedding, "model", None)` 安全取属性，缺失时得到 `None`；第二步 `isinstance(model, str) and model.strip()` 判断「是字符串且去空白后非空」（注意 `strip()` 的结果被用作真值判断，非空字符串为真）；第三步命中则 `return model.strip()`；第四步否则 `return type(embedding).__name__`。全程无循环、无 I/O、无副作用。
- **异常/边界**：若 `model` 属性的读取本身通过 `property` 实现且抛异常，`getattr` 不会吞掉该异常，会向上传播（此处未做 try 包裹）。若 `model` 是数字、`bytes` 等非 `str` 类型，会被判为不合法而回退到类名。若 `model` 是纯空白字符串（如 `"   "`），`strip()` 后为空，同样回退到类名。`embedding` 为 `None` 时不报错，`getattr(None, "model", None)` 得 `None`，最终返回 `"NoneType"`。
- **同文件关系**：被 `resolve_embedding_identity` 调用；自身不调用本文件的其他函数。

### `resolve_embedding_identity(embedding: BaseEmbedding) -> EmbeddingIdentity` （第 61 行）
- **作用**：把「当前活跃的嵌入器」解析成一个标准的 `EmbeddingIdentity`，也就是统一的 (模型名, 维度) 对。难点在于维度并不总是现成的：有些嵌入器在初始化后不会填 `dimension`（或填 0），这时函数会真的做一次探测调用 `embedding.embed("embedding-lock-probe")`，用返回向量的长度当作维度，并在原来 `dimension` 为假值的情况下把探测结果回写到 `embedding.dimension`，避免后续每次都要再探测一遍。这个函数是所有锁比较的起点：`inspect_embedding_lock` 第一件事就是调它拿到 `current`。之所以要探测而不是信任配置，是因为配置声明的维度与真实输出维度不一致正是需要被锁机制拦住的情形之一。
- **参数**：`embedding: BaseEmbedding`，待解析的嵌入器实例，必须至少提供 `embed(text) -> 向量（可求 len）` 这个方法；`model` 与 `dimension` 属性是可选的。无默认值。
- **返回**：返回 `EmbeddingIdentity`（frozen dataclass），其 `model` 来自 `embedding_model_name`，`dimension` 为正整数。无论维度是直接读到的还是探测得到的，最终都保证 `dimension >= 1`，否则抛异常而不是返回。
- **内部流程**：第一步调用 `embedding_model_name(embedding)` 得到 `model`；第二步 `int(getattr(embedding, "dimension", 0) or 0)` 安全读取维度并强制转成 `int`（`or 0` 同时兜住 `None`、`0`、空值等假值，`int()` 兜住字符串数字，若值无法转 `int` 会抛 `ValueError`/`TypeError`）；第三步若 `dimension < 1`，调用 `embedding.embed("embedding-lock-probe")` 拿一个向量，用 `len(vector)` 得到维度；第四步若此时 `getattr(embedding, "dimension", 0)` 仍为假值，则把探测到的维度写回 `embedding.dimension`（对实例产生副作用）；第五步再次检查 `dimension < 1`，若仍小于 1 抛 `ValueError("embedding dimension must be a positive integer")`；第六步返回 `EmbeddingIdentity(model=model, dimension=dimension)`。函数体第一行的字符串是 docstring，不是逻辑。
- **异常/边界**：`embed` 探测失败（网络错误、模型加载失败等）会把底层异常原样抛出，不做包装；`embed` 返回不可求 `len` 的对象会抛 `TypeError`；返回空向量导致 `len == 0` 时，会走到最后的 `ValueError`。`dimension` 属性是无法转 `int` 的类型时抛 `ValueError` 或 `TypeError`。回写 `embedding.dimension` 时若该对象是只读属性（例如 `@property` 无 setter 或 frozen 对象），会抛 `AttributeError` 并向上传播——注意回写条件里先做了 `getattr(..., 0)` 的假值判断，但只读且已有非零维度的实例不会进入回写分支。没有重试逻辑，也没有超时控制（超时由嵌入器自身负责）。
- **同文件关系**：调用 `embedding_model_name`；构造 `EmbeddingIdentity`。被 `inspect_embedding_lock`（作为 `current` 的来源）和 `reindex_vector_projection`（当 `identity` 为 `None` 时）调用。

### `live_vector_dimension(manager: MemoryManager) -> int | None` （第 76 行）
- **作用**：尝试读取「活的向量库集合当前真实使用的维度」，用于发现 SQLite 锁记录已经过期或缺失、但向量库里其实还躺着一批老维度向量的情况。它是防御性的：先看向量库后端有没有暴露 `collection_dimension` 这个能力，没有就返回 `None`（表示「读不到」而不是「是 0」），有就调用它并转成 `int`。之所以要包一层 `try/except`，是因为集合可能还没被创建、后端可能抛网络异常，而这类「读不到投影尺寸」的情况不应该让整个锁检查流程失败，正确做法是退回完全依赖 SQLite 锁。`inspect_embedding_lock` 用它来让「活的 Qdrant 维度」优先于「陈旧的 SQLite 锁」。
- **参数**：`manager: MemoryManager`，记忆管理器实例，类型注解为 `memory.manager.MemoryManager`；函数只访问它的 `vector_store` 属性，并要求该属性存在（否则 `AttributeError` 会向上抛，因为这一行不在 `try` 里）。无默认值。
- **返回**：返回 `int | None`。当 `vector_store` 上没有可调用的 `collection_dimension` 时返回 `None`；当 getter 抛任何异常时返回 `None`；当 getter 返回 `None` 时返回 `None`；否则返回 `int(size)`。不会返回 0 以外的假值兜底——若 getter 返回 0，则返回 `0`（调用方自行判断）。
- **内部流程**：第一步 `getattr(manager.vector_store, "collection_dimension", None)` 取属性；第二步 `callable(getter)` 判断是否可调用，不可调用直接 `return None`；第三步在 `try` 里执行 `size = getter()`，捕获 `Exception`（带 `# noqa: BLE001` 注释，说明是有意宽泛捕获）并返回 `None`；第四步 `return int(size) if size is not None else None`，把结果规范成 `int`。无循环、无重试。
- **异常/边界**：`manager` 为 `None` 或没有 `vector_store` 属性时，`getattr` 表达式会先抛 `AttributeError`（未被捕获）；`int(size)` 在 size 是不可转 `int` 的类型时会抛 `ValueError`/`TypeError`（同样未被捕获，因为它位于 `try` 之外）。getter 内部的 `KeyboardInterrupt`、`SystemExit` 等 `BaseException` 子类不会被捕获（只捕获 `Exception`），会向上传播。返回值 `None` 的语义是「未知」，调用方必须区分它和 0。
- **同文件关系**：被 `inspect_embedding_lock` 调用，结果写入快照的 `"qdrant_dimension"` 字段；自身不调用本文件的其他函数。

### `inspect_embedding_lock(manager: MemoryManager, repository: DocumentRepository | None) -> dict[str, Any]` （第 89 行）
- **作用**：这是整个模块的「只读体检」函数：不改动任何状态，只回答「当前嵌入、SQLite 锁、活的向量库维度这三者是否自洽」并给出一份结构化快照。它的核心设计原则写在 docstring 里——活的向量库维度优先于陈旧的 SQLite 锁，因为一个空的或错误的锁绝不能掩盖住「向量库里已经存在 1024 维集合」这个事实。它区分了三类问题：库里已有数据但完全没有锁（`unlocked_data`，模型未知，不能安全地宣称归属）、向量库维度与当前配置维度不一致（`projected != current.dimension`）、以及锁存在但与当前配置不匹配（模型名或维度不同）。Web 层通常用它来做「设置页/诊断页」的展示，`apply_embedding_lock` 和 `mismatch_from_exception` 则用它拿到判定结果与内部对象。
- **参数**：`manager: MemoryManager`，记忆管理器，需要 `embedding`（用于解析当前身份）、`document_store`（用于 `list(include_expired=True)` 数记忆）、`vector_store`（用于 `collection_dimension` 与 `list_ids`）。`repository: DocumentRepository | None`，文档仓库，可为 `None`（例如 `:memory:` 或未配置持久化时），为 `None` 时锁与 chunk 信息一律视为不存在。两个参数都没有默认值。
- **返回**：返回 `dict[str, Any]`，固定包含九个键。`"current"`：`{"model", "dimension"}`，当前活跃身份；`"locked"`：`None` 或 `{"model", "dimension", "updated_at"}`，SQLite 里原始锁记录；`"projection"`：`None` 或 `{"model", "dimension", "updated_at"}`，判定后「应当视为有效」的投影身份（可能与 `locked` 不同，例如被升级成 `UNKNOWN_EMBEDDING_MODEL` 或 Qdrant 维度）；`"qdrant_dimension"`：`int | None`，活的向量库维度；`"mismatch"`：`bool`，是否存在需要用户确认的不一致；`"_current"`：内部用的 `EmbeddingIdentity` 对象（不是纯字典，不能直接 JSON 序列化）；`"_effective"`：内部用的 `EmbeddingLockRecord | None`。前六个键面向展示，带下划线的两个键是给同模块其他函数复用的「带外」数据。
- **内部流程**：第一步 `current = resolve_embedding_identity(manager.embedding)` 解析当前身份（可能触发一次探测嵌入）。第二步 `locked = repository.get_embedding_lock() if repository is not None else None` 读锁。第三步 `projected = live_vector_dimension(manager)` 读活维度。第四步初始化 `effective = locked`、`mismatch = False`、`unlocked_data = False`。第五步：若 `locked is None`，则分别探测三处是否已有数据——`memories_exist = bool(manager.document_store.list(include_expired=True))`、`chunks_exist = bool(repository.chunk_ids()) if repository is not None else False`、以及投影是否非空（取 `manager.vector_store.list_ids`，若可调用则先尝试 `list_ids(limit=1)`，遇到 `TypeError` 时回退成无参 `list_ids()`，结果与已有判断做 `or`）；三者任一为真即 `unlocked_data = True`。第六步分支判定：`unlocked_data` 为真时置 `mismatch = True`，并把 `effective` 换成模型为 `UNKNOWN_EMBEDDING_MODEL`、维度为 `projected or current.dimension` 的临时 `EmbeddingLockRecord`（即优先用活维度，否则用当前维度）；否则若 `projected is not None and projected != current.dimension`，置 `mismatch = True`，`effective` 换成模型沿用 `locked.model`（没有锁时写死 `"Qdrant"`）、维度为 `projected`、`updated_at` 沿用锁或空串的记录；否则若锁存在且 `locked.model != current.model or locked.dimension != current.dimension`，置 `mismatch = True` 并把 `effective` 保持为 `locked`。第七步组装并返回字典，其中 `locked`、`projection` 分别做 `None` 判断后展开字段，`_current` 直接放对象、`_effective` 直接放记录。
- **异常/边界**：`manager.embedding` 缺失或 `resolve_embedding_identity` 失败（探测嵌入抛错、维度非法）会把异常抛出；`manager.document_store.list(...)` 或 `repository.chunk_ids()` 抛错不会被捕获，会向上传播；`list_ids` 只捕获 `TypeError`（用于兼容签名不同），其它异常照抛。`repository is None` 时不读锁、不数 chunk，`locked`/`projection` 都可能为 `None`，且 `mismatch` 仍可能因为活维度与当前维度不同而为 `True`。返回的字典含有非 JSON 原生对象（`_current`、`_effective`），直接丢给 `json.dumps` 会失败，使用方必须自行剥离或使用 `to_detail()` 那类已序列化的结构。无空值兜底异常处理，无重试。
- **同文件关系**：调用 `resolve_embedding_identity`、`live_vector_dimension`；构造 `EmbeddingLockRecord`（来自 `storage.document_repo`）；被 `apply_embedding_lock`（读 `snapshot["mismatch"]`、`snapshot["_current"]`、`snapshot["_effective"]`）和 `mismatch_from_exception`（读 `_current`、`_effective`、`qdrant_dimension`）调用。

### `apply_embedding_lock(manager: MemoryManager, repository: DocumentRepository | None, *, confirm_rebuild: bool = False) -> EmbeddingLockRecord | None` （第 159 行）
- **作用**：这是写入路径上的闸门函数，也是本模块最核心的「决策点」。每次准备向向量库写数据前调用它，它会检查「当前嵌入」与「SQLite 锁 / 活向量库」是否一致：一致就放行并返回锁记录；不一致且用户没有确认重建，就抛 `EmbeddingLockMismatch`（Web 层转成 409，让用户决定）；不一致且 `confirm_rebuild=True`，就执行重建 + 全量重灌，然后把新的锁返回。它还有一个刻意的例外：`repository is None` 时直接返回 `None` 做无操作，这样单元测试可以继续使用互相隔离的内存存储，而不必配置锁。另一个关键规则是「空锁只在活的集合缺失或已经匹配时才被认领」——这是防止把一批来源不明的老向量错误地记成当前模型的产物。
- **参数**：`manager: MemoryManager`，记忆管理器。`repository: DocumentRepository | None`，文档仓库，`None` 表示不做锁管理（直接返回 `None`）。`confirm_rebuild: bool`，仅关键字参数（签名中的 `*` 使其必须写成 `confirm_rebuild=True`），默认 `False`；为 `True` 时允许在检测到不一致后执行破坏性的重建与重灌。三个参数没有其他取值范围约束，`confirm_rebuild` 非布尔真值（如 `1`）也会按真处理。
- **返回**：返回 `EmbeddingLockRecord | None`。`repository is None` 时返回 `None`；检测到不一致并成功重建后返回 `repository.get_embedding_lock()`（重建流程已把锁推进到新身份）；检测到不一致但 `confirm_rebuild=False` 时抛异常、不返回；无不一致时，若锁不存在则返回 `repository.set_embedding_lock(current.model, current.dimension)` 的写入结果，若锁已存在则原样返回 `locked`。
- **内部流程**：第一步 `if repository is None: return None` 短路。第二步 `snapshot = inspect_embedding_lock(manager, repository)` 做体检，取出 `current: EmbeddingIdentity = snapshot["_current"]`。第三步若 `snapshot["mismatch"]` 为真：取 `effective = snapshot["_effective"]`，若它不是 `EmbeddingLockRecord`（例如 `None`）则用当前身份临时构造一个 `EmbeddingLockRecord(current.model, current.dimension)` 供异常携带；随后若 `not confirm_rebuild` 则 `raise EmbeddingLockMismatch(effective, current)`；否则调用 `rebuild_vector_projection(manager, repository, current)` 执行破坏性重建，完成后返回 `repository.get_embedding_lock()`。第四步（无不一致时）重新读一次 `locked = repository.get_embedding_lock()`；若为 `None`（库里干净且无锁）则调用 `repository.set_embedding_lock(current.model, current.dimension)` 认领并返回写入结果；否则直接返回 `locked`。
- **异常/边界**：主要抛出 `EmbeddingLockMismatch`（不一致且未确认重建）。此外 `inspect_embedding_lock` 内部的任何异常（嵌入探测失败、存储读取失败）都会透传；`rebuild_vector_projection` 失败时也会透传（例如 `document_store` 不支持原子写入会抛 `RuntimeError`，嵌入失败会抛底层异常），此时锁可能停留在 `REBUILDING_EMBEDDING_MODEL` 中间态，需要调用方后续重新检查。`repository is None` 的短路意味着这种配置下完全不做维度校验，属于有意为之的测试便利。不处理并发（没有加锁），两个进程同时认领空锁时依赖仓库层的写入语义。
- **同文件关系**：调用 `inspect_embedding_lock`、`rebuild_vector_projection`，并构造 `EmbeddingLockMismatch` 与 `EmbeddingLockRecord`；被模块外部的记忆写入路径 / Web 层调用。

### `mismatch_from_exception(exc: BaseException, manager: MemoryManager, repository: DocumentRepository | None) -> EmbeddingLockMismatch | None` （第 192 行）
- **作用**：这是一个「异常归一化」函数，用来把向量库（Qdrant）自己抛出的维度错误翻译成本模块的 `EmbeddingLockMismatch`，从而让用户看到与闸门路径完全相同的 409 载荷和确认文案。之所以需要它，是因为维度不一致可能不是被 `apply_embedding_lock` 提前拦住的，而是写入或查询时由 Qdrant 直接报错（错误文本里通常带 `dimension mismatch` 或 `expected dim`），如果不在这一层翻译，用户会收到一个晦涩的后端错误。函数先做「是否已经是目标异常」的透传，再看错误文本是否命中关键字，命中才去重新体检并构造异常；不命中就返回 `None`，表示「这不是嵌入不一致问题，交给别的错误处理分支」。
- **参数**：`exc: BaseException`，待判定的原始异常，可能是 Qdrant 客户端异常、`ValueError`、`RuntimeError` 等；也允许直接传入已经是 `EmbeddingLockMismatch` 的实例（会被原样返回）。`manager: MemoryManager`，用于重新体检当前身份与活维度。`repository: DocumentRepository | None`，可为 `None`，透传给 `inspect_embedding_lock`。三个参数都没有默认值。
- **返回**：返回 `EmbeddingLockMismatch | None`。若 `exc` 本身是 `EmbeddingLockMismatch`，原样返回它；若 `str(exc)` 小写后既不含 `"dimension mismatch"` 也不含 `"expected dim"`，返回 `None`；否则返回新构造的 `EmbeddingLockMismatch`（其 `locked` 来自体检的 `_effective`，或退化为模型名 `"Qdrant"`、维度为 `snapshot["qdrant_dimension"] or current.dimension` 的临时记录）。
- **内部流程**：第一步 `isinstance(exc, EmbeddingLockMismatch)` 判断并透传。第二步 `text = str(exc).casefold()` 把错误信息小写化（`casefold()` 比 `lower()` 更彻底，能处理更多 Unicode 情形）。第三步做两次子串包含判断，都不命中则 `return None`。第四步 `snapshot = inspect_embedding_lock(manager, repository)` 重新体检（注意：这里可能再次触发嵌入探测）。第五步取 `current = snapshot["_current"]` 与 `effective = snapshot["_effective"]`；若 `effective` 不是 `EmbeddingLockRecord`，则取 `projected = snapshot["qdrant_dimension"] or current.dimension` 并构造 `EmbeddingLockRecord("Qdrant", int(projected))` 作为锁定侧。第六步返回 `EmbeddingLockMismatch(effective, current)`。
- **异常/边界**：`str(exc)` 对极少数自定义异常可能抛错（罕见）；`inspect_embedding_lock` 内部异常会向上传播，也就是说「为了报错而再次报错」是可能的，调用方需容忍；`int(projected)` 在 `projected` 是奇怪类型时抛 `ValueError`。不命中的错误文本一律返回 `None`，不做模糊匹配，因此 Qdrant 若改了错误措辞，这里会退化成「识别不出」而不是误报。`repository` 为 `None` 时 `inspect_embedding_lock` 仍可工作，只是锁信息缺失。
- **同文件关系**：调用 `inspect_embedding_lock`，构造 `EmbeddingLockMismatch` 与 `EmbeddingLockRecord`；不被本文件其他函数调用，供模块外部的 Web / 写入路径在捕获向量库异常时调用。

### `reindex_vector_projection(manager: MemoryManager, repository: DocumentRepository, identity: EmbeddingIdentity | None = None, *, recreate_collection: bool = False, continue_on_error: bool = False) -> dict[str, Any]` （第 213 行）
- **作用**：这是「全量重灌」的实现主体：把 SQLite 里的每一个唯一 id 重新嵌入一次并写回向量库，最后把锁推进到新身份。它要解决的核心麻烦是「同一个 id 可能同时存在于 `chunks` 表和 `memories` 表」——正常摄入流程会在两张表里用同一个 id 存同一条内容，如果无脑遍历两张表就会把同一条数据嵌入两次并互相覆盖，所以它先以 chunks 为主遍历（拿到更丰富的 chunk 载荷：文档 id、chunk 序号、来源），把处理过的 id 记进 `chunk_ids`，之后再遍历 `memories` 并跳过已处理的 id。它还能救回「孤儿 chunk」——那些只进了 `chunks` 表、从未进入 `memories` 表的数据，依然能从 chunk 真相源里恢复出来。整个流程是「先暂存、全部成功才提交」的两阶段式：嵌入结果收集到 `staged_memories`，只有 `failures` 为空时才原子写回文档存储、把 chunk 状态标成 `indexed`、并推进嵌入锁；只要有任何失败，锁就不会前进，避免出现「锁说已经切到新模型，但库里还有一半旧向量」的撕裂状态。
- **参数**：`manager: MemoryManager`，需要 `embedding`（提供 `embed` 与 `embed_item`）、`document_store`（提供 `list` 与可选的 `upsert_many`）、`vector_store`（提供可选的 `recreate_collection`、可选的 `upsert_chunk`、以及必需的 `upsert`）。`repository: DocumentRepository`，必须非 `None`，用于 `list_all_chunks`、`get_document`、`set_chunk_vector_status`、`set_embedding_lock`。`identity: EmbeddingIdentity | None = None`，目标身份；为 `None` 时内部用 `resolve_embedding_identity(manager.embedding)` 现场解析。`recreate_collection: bool = False`（仅关键字），为 `True` 且向量库暴露可调用的 `recreate_collection` 时，会先用新维度重建集合，从而清空所有旧向量。`continue_on_error: bool = False`（仅关键字），为 `False` 时任何一条失败都立刻抛异常（fail-fast）；为 `True` 时把失败收集进结果列表继续跑（best-effort）。
- **返回**：返回 `dict[str, Any]`，包含 `"model"`（目标模型名）、`"dimension"`（目标维度）、`"chunks"`（成功处理的 chunk 条数，含被识别为 chunk 的 memory 对）、`"memories"`（成功处理的 memory 条数，统计口径包含与 chunk 同 id 的那批）、`"failed"`（失败列表，每项是 `{"id": ..., "error": "异常类名: 异常信息"}`）。成功路径下锁已被推进到 `(current.model, current.dimension)`；`continue_on_error=True` 且有失败时也会返回字典，但锁保持在 `REBUILDING_EMBEDDING_MODEL` 中间态（因为 `if not failures` 分支未进入）。
- **内部流程**：第一步 `current = identity or resolve_embedding_identity(manager.embedding)` 确定目标身份。第二步 `repository.set_embedding_lock(REBUILDING_EMBEDDING_MODEL, current.dimension)` 先立起「重灌中」的哨兵锁，防止重灌期间被别的写入当成已完成状态。第三步取 `recreate = getattr(manager.vector_store, "recreate_collection", None)`，仅当 `recreate_collection` 为真且 `recreate` 可调用时执行 `recreate(current.dimension)`。第四步读取全部记忆 `memories = manager.document_store.list(include_expired=True)`，建立 `memory_by_id = {item.id: item for item in memories}`，并初始化 `chunk_ids: set[str]`、`indexed_chunk_ids: list[str]`、`staged_memories: list[MemoryItem]`、`failures: list[dict[str, str]]`、计数器 `chunks_done = memories_done = 0`、以及 `upsert_chunk = getattr(manager.vector_store, "upsert_chunk", None)`。第五步定义嵌套函数 `record_failure(item_id, exc)`（见下条）。第六步遍历 `repository.list_all_chunks()`：把 `chunk.chunk_id` 加入 `chunk_ids`，从 `memory_by_id` 查同 id 的记忆；在 `try` 内先取 `document = repository.get_document(chunk.document_id)`；若 `item is None`（孤儿 chunk）则用 `manager.embedding.embed(chunk.text)` 生成向量，并新建一个 `MemoryItem`（id 为 chunk id、内容为 chunk 文本、`memory_type=MemoryType.SEMANTIC`、`metadata` 含 `kind="chunk"`、`document_id`、`chunk_index`、`source`（文档缺失时为空串）、`embedding=vector`）；若 `item` 存在则用 `manager.embedding.embed_item(item.content, payload=item.payload, modality=item.modality)` 重算向量并赋给 `item.embedding`。随后把 item 追加到 `staged_memories`；若 `upsert_chunk` 可调用，则以 chunk 更丰富的载荷调用它（`chunk.chunk_id`、`vector`、`document_id`、`chunk_index`、`source`、`memory_type="semantic"`），否则退回 `manager.vector_store.upsert(item)`；最后记录 `indexed_chunk_ids.append(chunk.chunk_id)` 并把 `chunks_done` 与 `memories_done` 各加一。若期间抛异常，则把该 chunk 的向量状态置为 `"failed"` 并交给 `record_failure`。第七步遍历 `memories`，跳过 `item.id in chunk_ids` 的（已在上一轮处理），其余在 `try` 内用 `embed_item` 重算向量、追加 `staged_memories`、`manager.vector_store.upsert(item)`、`memories_done += 1`；异常同样交给 `record_failure`。第八步 `if not failures:` 进入提交阶段：取 `upsert_many = getattr(manager.document_store, "upsert_many", None)`，不可调用则抛 `RuntimeError("document store does not support atomic projection commits")`；可调用则用它原子写入 `staged_memories`，然后对 `indexed_chunk_ids` 逐个 `repository.set_chunk_vector_status(chunk_id, "indexed")`，最后 `repository.set_embedding_lock(current.model, current.dimension)` 推进锁。第九步返回统计字典。
- **异常/边界**：`continue_on_error=False` 时任何单条失败都会在 `record_failure` 里立刻重新抛出，此时锁停留在 `REBUILDING_EMBEDDING_MODEL`，且已经写入向量库的部分条目不会回滚（向量库的 upsert 是先做的），需要调用方重试。`continue_on_error=True` 时失败被记录但锁不推进。`document_store` 缺少 `upsert_many` 时抛 `RuntimeError`。`repository` 为 `None` 时会在 `set_embedding_lock` 处抛 `AttributeError`（函数没有做 `None` 防护，签名也要求非空）。空库情况：`list_all_chunks()` 与 `list()` 都为空时，循环不执行，`failures` 为空，于是走到提交阶段写入空的 `staged_memories`（取决于 `upsert_many` 是否接受空列表）并推进锁——即空库也会被认领为当前身份。`recreate_collection` 为真但后端没有该方法时静默跳过重建，只是重灌覆盖。`MemoryItem` 的 `metadata` 中 `document.source` 在文档不存在时用空串兜底。
- **同文件关系**：调用 `resolve_embedding_identity`（当 `identity` 为 `None`）、内部嵌套函数 `record_failure`，并构造 `MemoryItem`；被 `rebuild_vector_projection` 调用（后者固定传 `recreate_collection=True`），也被 `apply_embedding_lock` 间接调用。

#### `record_failure(item_id: str, exc: Exception) -> None` （第 245 行，嵌套函数）
- **作用**：这是定义在 `reindex_vector_projection` 内部的小闭包，用来统一处理「某一条数据重灌失败」的情况：把失败信息以 `{"id": ..., "error": "异常类名: 异常信息"}` 的形式追加进外层的 `failures` 列表，并根据外层参数 `continue_on_error` 决定是继续还是立刻把原异常重新抛出。把它抽成嵌套函数是为了让两个循环体（chunk 循环和 memory 循环）的失败处理逻辑完全一致，避免复制粘贴出两套略有差异的错误格式。它依赖闭包捕获外层的 `failures` 与 `continue_on_error`，因此只在 `reindex_vector_projection` 执行期间有意义。
- **参数**：`item_id: str`，失败条目的 id（chunk id 或 memory id），用于让上层知道是哪一条出了问题。`exc: Exception`，捕获到的原始异常对象，用于取类型名与字符串信息。无默认值。
- **返回**：返回 `None`。函数没有返回值；其效果体现在修改外层 `failures` 列表，以及可能抛出的异常上。
- **内部流程**：第一步构造错误描述字符串 `f"{type(exc).__name__}: {exc}"`（例如 `"ConnectionError: ..."`），并和 `item_id` 一起 append 进外层 `failures`。第二步判断 `if not continue_on_error:`，为真时执行 `raise exc`，把原始异常重新抛出（用裸 `raise exc` 而不是 `raise`，因此 traceback 会从这一行重新起算）。
- **异常/边界**：当 `continue_on_error` 为 `False` 时，它一定会抛出传入的 `exc`，使整个重灌流程中止；为 `True` 时永不抛错（除了 append 本身的内存错误）。不区分异常类型，也不做任何降级或重试；如果 `exc` 是 `BaseException` 而非 `Exception`（例如 `KeyboardInterrupt`），调用点的 `except Exception` 不会捕获它，本函数也不会被调用。
- **同文件关系**：被 `reindex_vector_projection` 内部的两处 `except` 分支调用（chunk 循环与 memory 循环各一处）；它自身不调用本文件的其他函数，只操作外层作用域的 `failures` 与 `continue_on_error`。

### `rebuild_vector_projection(manager: MemoryManager, repository: DocumentRepository, identity: EmbeddingIdentity | None = None) -> dict[str, Any]` （第 330 行）
- **作用**：这是重灌流程的便捷封装，语义是「先重建（清空）向量集合，再把 SQLite 真相完整投影一遍」。它本身几乎不做事，只是把 `reindex_vector_projection` 以 `recreate_collection=True` 调起来，从而把「重建」与「重灌」两个动作绑定成一个不可分割的调用。`apply_embedding_lock` 在用户确认重建后调用的就是它，所以它是「用户点确认」这条路径上真正执行破坏性操作的入口；单独暴露它也是为了让运维脚本或管理接口能在不经过闸门的情况下主动重建。因为它没有 `continue_on_error` 参数，所以走的是 fail-fast 语义：任何一条失败都会中止并抛异常。
- **参数**：`manager: MemoryManager`，记忆管理器，透传给 `reindex_vector_projection`。`repository: DocumentRepository`，文档仓库，必须非 `None`，透传。`identity: EmbeddingIdentity | None = None`，目标身份，为 `None` 时由下游 `resolve_embedding_identity` 现场解析；透传。三个参数都没有除 `identity` 之外的默认值，且没有关键字专用参数。
- **返回**：返回 `dict[str, Any]`，就是 `reindex_vector_projection` 的返回值，包含 `"model"`、`"dimension"`、`"chunks"`、`"memories"`、`"failed"` 五个键（见该函数的返回说明）。
- **内部流程**：函数体只有一条 `return` 语句：调用 `reindex_vector_projection(manager, repository, identity, recreate_collection=True)`，即前三个参数位置透传，并把 `recreate_collection` 强制置为 `True`；`continue_on_error` 保持默认的 `False`。没有任何额外的前置检查、日志或异常包装。
- **异常/边界**：完全继承 `reindex_vector_projection` 的异常行为：嵌入失败、存储写入失败、`document_store` 缺少 `upsert_many`（抛 `RuntimeError`）、`repository` 为 `None`（抛 `AttributeError`）等都会原样向上传播；重建失败时锁会停在 `REBUILDING_EMBEDDING_MODEL`。由于它强制 `recreate_collection=True`，一旦后端支持重建，调用它就是破坏性的——旧向量会被清空，这也是为什么它只应该在用户明确确认后调用。无自身特殊处理。
- **同文件关系**：调用 `reindex_vector_projection`；被 `apply_embedding_lock`（`confirm_rebuild=True` 路径）调用，并作为公开 API 列入 `__all__` 供外部使用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `EmbeddingIdentity` | 不可变的 (模型名, 维度) 值对象，用于统一表示并比较嵌入身份。 |
| `EmbeddingLockMismatch` | 表示嵌入与 SQLite 锁不一致的自定义 `ValueError` 异常，携带双方身份与可返回给前端的详情。 |
| `EmbeddingLockMismatch.__init__` | 保存 locked/current 两个字段并把中文提示交给异常基类作为 args。 |
| `EmbeddingLockMismatch.message` | 只读属性，现算出「库内 vs 当前」的中文不一致提示与两个操作选项。 |
| `EmbeddingLockMismatch.to_detail` | 把异常转成带 `code`、`message`、`locked`、`current` 的结构化字典，供 409 响应体使用。 |
| `embedding_model_name` | 安全地从嵌入器取模型名，取不到就回退成类名。 |
| `resolve_embedding_identity` | 解析当前活跃嵌入身份，维度未知时做一次探测嵌入并回写维度。 |
| `live_vector_dimension` | 尽力读取向量库集合的真实维度，读不到就返回 `None` 表示未知。 |
| `inspect_embedding_lock` | 只读体检：对比当前嵌入、SQLite 锁与活向量库维度，输出快照与 mismatch 判定。 |
| `apply_embedding_lock` | 写入前的闸门：一致则放行/认领空锁，不一致则抛 409 异常或按确认执行重建重灌。 |
| `mismatch_from_exception` | 把向量库的维度错误文本归一化成同一种 `EmbeddingLockMismatch`，否则返回 `None`。 |
| `reindex_vector_projection` | 按唯一 id 把 SQLite 真相全量重新嵌入并投影一次，全部成功才推进嵌入锁。 |
| `reindex_vector_projection.record_failure` | 嵌套闭包：记录单条失败信息，并按 `continue_on_error` 决定继续或立刻抛错。 |
| `rebuild_vector_projection` | 先重建（清空）向量集合再全量重灌的便捷封装，走 fail-fast 语义。 |
