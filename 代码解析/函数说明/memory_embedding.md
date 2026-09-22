# memory/embedding.py

## 一、这个文件是干什么的

这个文件是整个项目里唯一的「文本/图像 → 向量」统一入口，也就是嵌入（embedding）能力的抽象层与实现层。它把「怎么把一段文字或一张图片变成一串浮点数」这件事收敛成两个可互换的实现：一个是走云端 OpenAI 兼容 `/embeddings` 接口的 `APIEmbedding`（SiliconFlow、DashScope、OpenAI、智谱、本地 vLLM 都走这条路径），另一个是完全离线、确定性的 `HashEmbedding`（把词袋哈希到固定维度），后者保证在没有配置任何云端嵌入服务时，记忆层、Agent 工具与 Web 应用仍然可以跑起来，同时也是整个测试套件注入的替身实现。

文件对外暴露三个类加一个常量：抽象基类 `BaseEmbedding` 定义所有提供方必须实现的接口（`embed` 抽象方法，以及 `embed_batch`、`embed_item` 两个默认实现），`HashEmbedding` 与 `APIEmbedding` 各自给出离线与云端的具体实现，`VL_EMBEDDING_MODEL_MARKER` 用来识别视觉语言（VL）多模态嵌入模型。

在运行期，记忆系统每次写入一条记忆、做一次语义检索、或者 Web 端做 RAG 召回，都会经由 `BaseMemory.add` 走到 `embed_item` 这个钩子上，所以这个文件是四层记忆系统与向量库 `QdrantVectorStore` 之间的桥：`QdrantVectorStore` 的维度守卫依赖这里学到的 `dimension`，切换嵌入模型后必须重建投影，而维度的唯一开关是配置里的 `[embedding].model`。

文件里还包含一整套围绕 HTTP 调用的辅助函数：请求构造与发送（`_request`）、响应解析（`_extract_embedding_vectors`）、批内单次请求（`_embed_batch_once`）、向量形状校验与维度学习（`_learn_dimension`），以及按图片魔数猜 MIME 的小工具（`_image_mime_type`）。历史实现（Gemini `:embedContent`、本机转发网关 `/embed`、SSH 隧道自动拉起、`.env` 装载）都已经按「配置只认 config/ 与 constants.py」的约定删除，端点、密钥、模型名统一来自 `config/services.toml` 的 `[embedding]` 段。

## 二、函数与类逐条详解

### `class BaseEmbedding(ABC)` （第 48 行）
- **作用**：这是所有嵌入提供方的抽象基类，作用是把「嵌入服务」这件事定义成一个极小的接口契约，让记忆层、向量库和 Web 层只依赖这个契约，而不关心背后是云端 API 还是离线哈希。它声明了一个类属性 `dimension`（默认为 0，表示尚未知晓维度）和一个抽象方法 `embed`，任何子类都必须实现 `embed` 才能被实例化。同时它给出了两个带默认实现的便捷方法 `embed_batch` 与 `embed_item`，使得只实现单条嵌入的子类也能立刻拥有批处理和「记忆条目级」的嵌入能力。`embed_item` 是它与记忆系统的关键结合点：`BaseMemory.add` 会把每一次写入都路由到这个钩子上，多模态后端只要重写它就能把 `payload`/`modality` 折叠进同一个向量，从而让一张存下来的图片可以被它自己的内容检索到。不支持 `payload` 的提供方可以原样忽略它，继续只嵌入 `text`。它被继承为 `HashEmbedding` 与 `APIEmbedding`，是这两个实现的公共父类。
- **参数**：无（这是类定义本身；`ABC` 是继承自 `abc` 模块的抽象基类元类，不是运行期参数）。
- **返回**：无（类对象）。
- **内部流程**：类体里先定义类属性 `dimension: int = 0` 作为默认维度占位；接着用 `@abstractmethod` 装饰 `embed` 并在方法体里 `raise NotImplementedError`，形成必须被子类覆盖的契约；然后定义 `embed_batch`，把传入的可迭代对象实体化成列表、做全量类型检查、再逐条调用 `self.embed`；最后定义 `embed_item`，用 `del payload, modality` 显式丢弃这两个仅用于多模态覆盖的参数，再转调 `self.embed(text)`。
- **异常/边界**：直接实例化 `BaseEmbedding` 会因为存在未实现的抽象方法 `embed` 而由 `ABC` 机制抛出 `TypeError`；`embed_batch` 遇到非字符串元素会抛 `TypeError("texts must contain strings")`；`embed_item` 本身不抛异常，只是把 `payload` 与 `modality` 丢弃。
- **同文件关系**：它是 `HashEmbedding`（第 77 行）与 `APIEmbedding`（第 116 行）的父类；`embed_batch` 内部调用本类的 `embed`；`embed_item` 内部也调用本类的 `embed`。它不调用本文件里的模块级辅助函数。

### `BaseEmbedding.embed(self, text: str) -> list[float]` （第 54 行）
- **作用**：这是整个嵌入体系唯一的抽象方法，规定了「把一条文本变成有限数值向量」这个最小能力。之所以需要它，是因为上层代码（记忆写入、语义检索、RAG 召回）只应该知道「给我一个向量」，而不该知道背后是 HTTP 请求还是本地哈希。它被 `@abstractmethod` 装饰，因此任何继承 `BaseEmbedding` 的具体提供方都必须给出自己的实现，否则无法被实例化。它在运行期由 `embed_batch` 和 `embed_item` 的默认实现调用，也被子类各自的批量逻辑间接依赖。方法体里的 `raise NotImplementedError` 只是形式上的兜底，真正的约束由抽象基类机制提供。它是本文件里所有具体嵌入实现的统一签名来源。
- **参数**：`text: str` —— 待嵌入的原始文本，约定为 Python 字符串；具体实现会额外做类型检查（例如 `HashEmbedding.embed` 与 `APIEmbedding.embed` 都会对非字符串抛 `TypeError`）。
- **返回**：`list[float]` —— 一条文本对应的数值向量，长度即该提供方的 `dimension`；抽象签名不承诺归一化，是否单位化由具体实现决定。
- **内部流程**：抽象方法没有真实流程，唯一语句是 `raise NotImplementedError`，用于在子类忘记实现时（在绕过 ABC 检查的极端情况下）给出明确错误而不是静默返回错误结果。
- **异常/边界**：方法体本身会抛 `NotImplementedError`；在正常的 ABC 使用方式下，子类未实现时实例化阶段就会抛 `TypeError`，轮不到这里执行。对空字符串没有规定，交由具体实现决定。
- **同文件关系**：被本类的 `embed_batch` 与 `embed_item` 调用；被 `HashEmbedding.embed`（第 100 行）与 `APIEmbedding.embed`（第 181 行）覆盖实现。

### `BaseEmbedding.embed_batch(self, texts: Iterable[str]) -> list[list[float]]` （第 58 行）
- **作用**：这是批处理的默认实现，作用是把「一批文本」的嵌入退化成逐条调用 `embed`，从而让只实现了单条能力的提供方也能被批量接口调用。之所以需要它，是因为上层写入记忆或做批量索引时天然是按列表组织的，而并非所有后端都值得或能够实现真正的批量 HTTP 请求，默认实现保证了接口的普适性。它同时也是 `APIEmbedding` 需要覆盖的方法——后者会把它替换成按 `batch_size` 分片、真正走批量请求的版本。它在运行期由记忆层的批量写入路径触发。它体现了「接口最小化 + 默认实现补全」的设计取向。它不做任何去重、缓存或并发，纯粹是顺序映射。
- **参数**：`texts: Iterable[str]` —— 任意可迭代的字符串集合，可以是列表、元组、生成器等；函数内部会先实体化成列表，因此生成器只会被消费一次。
- **返回**：`list[list[float]]` —— 与输入顺序一一对应的向量列表，长度等于输入元素个数；输入为空可迭代对象时返回空列表 `[]`。
- **内部流程**：先用 `list(texts)` 把可迭代对象物化为 `values`，以便后续可以多次遍历；然后用 `all(isinstance(text, str) for text in values)` 做整体类型校验，只要有一个元素不是字符串就立刻抛错，避免部分成功；最后用列表推导 `[self.embed(text) for text in values]` 顺序逐条嵌入并返回结果列表。
- **异常/边界**：任一元素不是 `str` 时抛 `TypeError("texts must contain strings")`；空输入返回空列表而不报错；不处理 `None` 输入（传 `None` 会在 `list(None)` 处抛 `TypeError`）；不做长度上限控制，超大批次会逐条串行处理，可能很慢。
- **同文件关系**：内部调用本类的抽象方法 `embed`；被 `APIEmbedding.embed_batch`（第 186 行）覆盖实现；`HashEmbedding` 直接继承使用这一默认实现，不再覆盖。

### `BaseEmbedding.embed_item(self, text: str, *, payload: Any = None, modality: str | None = None) -> list[float]` （第 64 行）
- **作用**：这是「记忆条目级」的嵌入钩子，也是本文件与四层记忆系统耦合最紧的方法。`BaseMemory.add` 会把每一次写入都经由这个钩子，因此它决定了「一条记忆被存进向量库时到底用什么内容生成向量」。默认实现是纯文本的：它显式丢弃 `payload` 与 `modality`，只嵌入 `text`。这样设计的意义在于，多模态后端（例如带 VL 模型的 `APIEmbedding`）可以覆盖它，把图片或其它载荷折叠进同一个向量，从而让一张存下来的图片能够被它自己的内容检索到；而不支持载荷的提供方则完全不需要改动行为。它保证了记忆层的写入路径在单模态和多模态两种情况下都只有一种调用方式。它也是「同一份代码在纯文本模型下行为不变」的兼容性保障。
- **参数**：`text: str` —— 记忆条目的文本表示，通常是标题、正文或摘要的拼接结果。`payload: Any = None`（关键字专用）—— 与该条目关联的附加载荷，例如图片的字节、URL 或 base64 字符串；默认 `None` 表示没有附加内容。`modality: str | None = None`（关键字专用）—— 模态标记，用于提示这条记忆是文本、图像还是混合；默认 `None`。
- **返回**：`list[float]` —— 一条向量；默认实现下就是 `self.embed(text)` 的结果，长度等于该提供方的 `dimension`。
- **内部流程**：第一步用 `del payload, modality` 显式删除两个参数，一方面表达「基类不消费它们」的意图，另一方面避免静态检查器报未使用变量；第二步直接 `return self.embed(text)`，把工作完全委托给单条嵌入方法。
- **异常/边界**：本方法自身不抛异常，也不校验 `text` 类型（非字符串会在 `embed` 内部被拒绝）；`payload` 为 `None` 或非 `None` 都不影响默认行为；`modality` 同样被忽略。
- **同文件关系**：内部调用本类的抽象方法 `embed`；被 `APIEmbedding.embed_item`（第 259 行）覆盖实现，后者会根据 `payload` 是否为空与 `multimodal` 标志决定是否走多模态路径；`HashEmbedding` 继承默认实现。

### `class HashEmbedding(BaseEmbedding)` （第 77 行）
- **作用**：这是一个确定性的离线嵌入实现，把词袋（bag-of-words）哈希到固定维度上，作为没有云端配置时的兜底方案。它的存在意义是让记忆层、Agent 工具和 Web 应用在完全没有任何云端嵌入服务的情况下依旧可用，同时也是整个测试套件注入的替身（test double），保证测试不依赖网络与密钥。它的向量空间与 `APIEmbedding` 完全不兼容，因此只应该用于测试与离线场景，不能和云端向量混在同一个集合里检索。它的确定性意味着同样的文本永远得到同样的向量，便于断言与复现。它继承 `BaseEmbedding`，因此天然具备 `embed_batch` 与 `embed_item` 两个默认能力。它的 `to_dict` 输出会被序列化进健康检查或配置回显，方便看出当前跑的是哪种嵌入。
- **参数**：类定义无参数；构造参数见 `__init__`（第 85 行）。
- **返回**：无（类对象）。
- **内部流程**：类体定义了构造与校验（`__init__`）、分词（`tokenize`）、哈希取模（`_index`）、向量生成（`embed`），以及序列化与调试表示（`to_dict`、`__repr__`）。整体流程是：文本 → 小写化并正则切词 → 统计词频 → 每个词的哈希位置累加 `1 + ln(词频)` → 向量 L2 归一化。
- **异常/边界**：构造时维度非正整数抛 `ValueError`；`embed`/`tokenize` 收到非字符串抛 `TypeError`；空文本或全是标点的文本会产生全零向量，此时跳过归一化直接返回零向量（避免除零）。
- **同文件关系**：继承 `BaseEmbedding`（第 48 行），复用其 `embed_batch` 与 `embed_item`；`embed` 内部调用本类的 `tokenize` 与 `_index`；`_index` 内部不做跨类调用；`to_dict` 与 `__repr__` 只读自身属性。

### `HashEmbedding.__init__(self, dimension: int = MEMORY_EMBEDDING_DIMENSION) -> None` （第 85 行）
- **作用**：构造一个离线哈希嵌入器，并确定输出向量的维度。之所以需要显式构造，是因为向量维度必须与向量库集合的维度严格一致，一旦不匹配，写入或检索就会被维度守卫拒绝，所以这里把维度作为可配置项而不是硬编码。默认值取自 `constants.MEMORY_EMBEDDING_DIMENSION`，保证记忆层的默认配置能直接对上。它会在赋值前做严格的类型与取值校验，把「传了布尔值、传了浮点数、传了 0 或负数」这类错误尽早暴露成明确的 `ValueError`，而不是等到后面生成向量时才出现莫名其妙的失败。它是所有测试与离线运行场景下记忆系统的向量维度来源。构造完成后实例就不可变地持有该维度。
- **参数**：`dimension: int = MEMORY_EMBEDDING_DIMENSION` —— 输出向量的维度，必须是正整数；默认值来自 `constants.py` 中的 `MEMORY_EMBEDDING_DIMENSION`。布尔值 `True`/`False` 会被刻意拒绝（因为在 Python 里 `bool` 是 `int` 的子类，容易被误传）。
- **返回**：`None`（构造函数不返回值，只初始化实例状态）。
- **内部流程**：先用一条复合条件 `isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1` 做三重校验：排除布尔值、排除非整数（含浮点与字符串）、排除小于 1 的值；任一命中就抛 `ValueError("dimension must be a positive integer")`；校验通过后把 `self.dimension = dimension` 写入实例属性，覆盖基类上默认为 0 的类属性。
- **异常/边界**：`dimension` 为 `bool`、非 `int`、或小于 1 时抛 `ValueError`；没有对超大维度做上限限制，传一个极大的维度会显著增加内存与计算开销（向量按该长度初始化为零）。
- **同文件关系**：被 `HashEmbedding.embed` 与 `HashEmbedding._index` 通过 `self.dimension` 间接使用；`to_dict` 与 `__repr__` 读取该属性；与 `APIEmbedding.__init__` 是同类结构但校验范围不同（后者维度允许 `None`）。

### `HashEmbedding.tokenize(text: str) -> list[str]` （第 91 行）
- **作用**：把一段文本切成用于统计词频的 token 列表，是哈希嵌入的前置步骤。它被声明为静态方法，因为切词逻辑不依赖实例状态（与维度无关），这样既可以从实例调用也可以直接 `HashEmbedding.tokenize(...)` 调用，便于测试。实现上先做 `casefold()` 大小写折叠，使得 `Hello` 与 `hello` 落到同一个 token，从而获得大小写不敏感的词袋表示；再用正则 `\w+` 提取由字母、数字、下划线组成的连续片段，配合 `re.UNICODE` 让中文、日文等非 ASCII 文字也能被当作 token 提取出来，而不是被丢弃。标点与空白自然成为分隔符。它决定了 `HashEmbedding` 的语义粒度：相同 token 集合的文本会得到相同向量。
- **参数**：`text: str` —— 待分词的原始文本；必须是字符串，否则直接报错。
- **返回**：`list[str]` —— 分词结果，按在原文中出现的顺序排列，可能包含重复项（去重与计数由调用方 `Counter` 负责）；纯标点或空文本会返回空列表。
- **内部流程**：第一步类型校验，`not isinstance(text, str)` 时抛 `TypeError("text must be a string")`；第二步调用 `text.casefold()` 做比 `lower()` 更彻底的大小写折叠（能处理德语 ß 之类的特殊映射）；第三步用 `re.findall(r"\w+", ...)` 配合 `flags=re.UNICODE` 提取所有单词片段并作为列表返回。
- **异常/边界**：非字符串输入抛 `TypeError`；`None` 同样会触发该 `TypeError`；空字符串返回 `[]`（不会报错），后续 `embed` 会因此产生零向量；不做停用词过滤、不做词干化。
- **同文件关系**：被 `HashEmbedding.embed`（第 100 行）调用，其结果被 `Counter` 统计；不被本文件其它函数调用。

### `HashEmbedding._index(self, token: str) -> int` （第 96 行）
- **作用**：把单个 token 稳定地映射到 `[0, dimension)` 区间内的一个下标，也就是决定这个词「打在向量的哪一维上」。之所以用哈希而不是维护一张词表，是为了让离线嵌入器不需要任何训练、不需要保存词表文件、也不需要网络，同时对任意新词都能立即给出位置。之所以选用 `blake2b` 且指定 `digest_size=8`，是因为它速度快、确定性好、且 8 字节（64 位）的摘要空间足够大，能把取模后的碰撞概率压得很低。之所以固定用 `"big"` 字节序解析，是为了保证跨平台（大小端机器）得到完全一致的结果，从而维持确定性。它是 `HashEmbedding` 能被称为「确定性」的关键环节。
- **参数**：`token: str` —— 已经过 `tokenize` 处理的单个词元；本方法不重复做类型校验，若传入非字符串会在 `.encode("utf-8")` 处抛出 `AttributeError`。
- **返回**：`int` —— 落在 `0` 到 `self.dimension - 1` 之间的向量下标，作为 `embed` 中累加位置的索引。
- **内部流程**：先用 `token.encode("utf-8")` 把字符串编码为字节（UTF-8 保证同一文本在任何平台得到同一字节序列）；再用 `hashlib.blake2b(..., digest_size=8).digest()` 得到 8 字节摘要；最后用 `int.from_bytes(digest, "big")` 把字节按大端解析成整数，并对 `self.dimension` 取模后返回。
- **异常/边界**：传入非字符串（例如 `None`、`bytes`）会在编码或哈希阶段抛 `AttributeError`/`TypeError`；不处理哈希碰撞（两个不同 token 落到同一维，表现为该维计数叠加，这是词袋哈希的固有取舍）；`dimension` 为 0 时不会发生除零，因为构造函数已禁止该取值。
- **同文件关系**：只被 `HashEmbedding.embed`（第 100 行）调用；内部不调用本文件其它函数，只依赖标准库 `hashlib`。

### `HashEmbedding.embed(self, text: str) -> list[float]` （第 100 行）
- **作用**：这是 `HashEmbedding` 的核心，把一段文本转换成一个 L2 归一化的定长稠密向量。它的算法是经典的「哈希词袋 + 次线性词频加权」：先统计每个 token 出现的次数，再按 `1 + ln(count)` 把权重累加到该 token 对应的维度上，最后整体除以向量的欧氏长度，使结果落在单位球面上。之所以用对数加权而不是直接用原始计数，是为了抑制高频词对向量方向的支配，让长文档与短文本的向量更具可比性；之所以做归一化，是因为下游向量检索通常用余弦相似度，归一化后内积即等价于余弦，能显著简化与加速计算。它对空文本返回全零向量而不是报错，保证记忆层在遇到空内容时不会崩。它与 `APIEmbedding.embed` 签名一致，因此在配置层面可以互相替换（但不能混用同一向量空间）。
- **参数**：`text: str` —— 待嵌入的文本；必须是字符串，否则抛 `TypeError`。允许为空字符串或纯标点（结果为零向量）。
- **返回**：`list[float]` —— 长度等于 `self.dimension` 的浮点列表。正常情况下是单位向量（各元素平方和约为 1）；当文本不产生任何 token、导致原始向量全零时，返回未归一化的零向量。
- **内部流程**：第一步校验 `text` 是字符串，否则抛 `TypeError("text must be a string")`；第二步用 `[0.0] * self.dimension` 初始化零向量；第三步调用 `self.tokenize(text)` 得到 token 列表，交给 `collections.Counter` 统计词频，然后遍历 `(token, count)` 对，用 `self._index(token)` 求出下标，并执行 `vector[index] += 1.0 + math.log(float(count))` 累加权重；第四步计算 L2 范数 `norm = math.sqrt(sum(value * value for value in vector))`；第五步用条件表达式返回 `[value / norm for value in vector]`（范数非零）或原样的零向量（范数为零）。
- **异常/边界**：非字符串输入抛 `TypeError`；空文本或纯标点产生零向量并被原样返回（刻意避免 `ZeroDivisionError`）；`count` 恒为正整数，因此 `math.log` 不会遇到 0 或负数；不做向量截断，维度越大返回列表越长。
- **同文件关系**：调用本类的 `tokenize`（第 91 行）与 `_index`（第 96 行）；它是对 `BaseEmbedding.embed`（第 54 行）的具体实现，因此也会被基类的 `embed_batch`、`embed_item` 间接调用。

### `HashEmbedding.to_dict(self) -> dict[str, Any]` （第 109 行）
- **作用**：把当前嵌入器的类型与关键配置导出成普通字典，用于序列化、日志、健康检查或配置回显。之所以需要它，是因为上层（例如 `/api/health` 或记忆系统的自检输出）往往需要一个 JSON 可编码的「当前用的是什么嵌入、维度多少」的摘要，而不是去读对象属性或解析 `repr`。它只暴露两个字段：类名与维度，故意不包含任何敏感信息（`HashEmbedding` 本来也没有密钥）。它同时提供了与 `APIEmbedding.to_dict` 一致的调用方式，使得上层可以用多态的方式统一收集嵌入后端信息，而不必做 `isinstance` 判断。它的输出结构稳定，可以安全地写进日志或返回给前端。
- **参数**：无（除 `self`）。
- **返回**：`dict[str, Any]` —— 固定包含两个键：`"type"` 取 `type(self).__name__`（即 `"HashEmbedding"`，使用动态类名使得子类也能得到正确标识），`"dimension"` 为当前维度整数。
- **内部流程**：直接构造并返回一个字典字面量，不做校验、不做缓存、不访问外部资源；`type(self).__name__` 保证子类调用时返回子类名而非写死的父类名。
- **异常/边界**：无特殊处理；不涉及 I/O、不抛异常；返回值是新建的可变字典，调用方修改不会影响实例状态。
- **同文件关系**：只读取自身属性；与 `APIEmbedding.to_dict`（第 269 行）形成同构接口，便于上层统一处理；不被本文件其它函数调用。

### `HashEmbedding.__repr__(self) -> str` （第 112 行）
- **作用**：提供便于调试与日志阅读的对象表示，把类名与维度拼成一个简短字符串。之所以需要它，是因为在交互式排查或异常日志里，默认的 `repr` 会输出 `<memory.embedding.HashEmbedding object at 0x...>` 这种没有信息量的内存地址，而维度恰恰是这个类最关键的状态。它让「当前这个嵌入器是几维」可以一眼看出，尤其在对比测试期望维度与实际维度时非常有用。它不包含任何敏感信息。它是纯只读方法，不改变对象状态。
- **参数**：无（除 `self`）。
- **返回**：`str` —— 形如 `HashEmbedding(dimension=768)` 的字符串，维度值来自 `self.dimension`。
- **内部流程**：使用 f-string 拼装，直接内联 `self.dimension`；不做转义或截断；每次调用都会重新生成字符串（不缓存）。
- **异常/边界**：无特殊处理；若实例属性被外部篡改（例如删掉 `dimension`）会抛 `AttributeError`，正常情况下不会发生。
- **同文件关系**：只读取自身属性；与 `APIEmbedding.__repr__`（第 280 行）风格一致；不被本文件其它函数调用。

### `class APIEmbedding(BaseEmbedding)` （第 116 行）
- **作用**：这是面向任意云端（或本地）OpenAI 兼容 `/embeddings` 端点的嵌入客户端，是生产环境实际使用的实现。它支持 SiliconFlow、DashScope、OpenAI、智谱以及本地 vLLM 等所有遵循 OpenAI 请求/响应形状的服务：请求体是 `{"model": ..., "input": ...}`，响应从 `data[].embedding` 里取向量，同时解析器还额外兼容 DashScope 原生的 `output.embeddings` 布局，因此两种网关都能直接工作。密钥来自 `config/services.toml` 的 `[embedding].api_key`（由 `memory.base.make_default_embedding` 解析后传入），绝不从环境变量读取，这符合项目「配置只认 config/ 与 constants.py」的约定。它还内建了视觉语言（VL）多模态支持：模型名里含 `vl` 时（如 `Qwen/Qwen3-VL-Embedding-*`），`input` 除了字符串还能接受 `{"text": ...}`、`{"image": ...}` 内容对象以及它们的混合列表，一次请求把列表融合成单个向量；`multimodal` 属性由模型名推导，用于在 `embed_item` 里自动路由，而 `embed_image`/`embed_multimodal` 则无论如何都能手动调用。为了便于测试，`client` 参数允许注入替身，既可以是可调用对象，也可以是暴露 `embeddings.create(...)` 的对象。
- **参数**：类定义无参数；构造参数见 `__init__`（第 141 行）。
- **返回**：无（类对象）。
- **内部流程**：类体依次定义了构造与校验、单条与批量嵌入（`embed`、`embed_batch`）、VL 内容对象构造（静态方法 `text_input`、`image_input`）、VL 输入组装与发送（`inputs_for`、`embed_inputs`、`embed_image`、`embed_multimodal`）、记忆条目路由（`embed_item`），以及序列化与调试表示（`to_dict`、`__repr__`）。真正的网络交互全部委托给模块级辅助函数 `_request`、`_extract_embedding_vectors`、`_learn_dimension`、`_embed_batch_once`。
- **异常/边界**：构造期对 `model`、`base_url`、`dimension`、`timeout`、`batch_size`、`api_key` 逐项校验并抛出 `ValueError`/`RuntimeError`；运行期网络与解析错误统一由 `_request` 与 `_extract_embedding_vectors` 转成 `RuntimeError`。
- **同文件关系**：继承 `BaseEmbedding`（第 48 行）；`embed` 调用自身 `embed_batch`，`embed_batch` 调用模块级 `_embed_batch_once`，后者调用 `_request`、`_extract_embedding_vectors`、`_learn_dimension`；`embed_inputs` 直接调用 `_request`、`_extract_embedding_vectors`、`_learn_dimension`；`image_input` 调用模块级 `_image_mime_type`；`embed_item` 调用 `embed`、`embed_multimodal`、`embed_image`。

### `APIEmbedding.__init__(self, api_key: str | None = None, *, model: str = DEFAULT_EMBEDDING_MODEL, base_url: str = "", dimension: int | None = None, timeout: float = 30.0, batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE, client: Any = None) -> None` （第 141 行）
- **作用**：构造一个云端嵌入客户端，把配置（密钥、模型名、端点根地址、维度、超时、批大小、可选测试客户端）一次性校验并固化到实例上。之所以把校验集中放在构造函数里，是因为嵌入配置一旦出错（比如 base_url 漏了 `https://`、密钥是空串、批大小写成 0），故障会出现在很远的检索路径上且难以定位，所以在构造期就用明确的异常把问题拦下来。`base_url` 期望的是不带尾部 `/embeddings` 的提供方根地址（例如 `https://api.siliconflow.cn/v1`），因为路径拼接由 `_request` 统一负责。`dimension` 允许为 `None`，表示「还不知道维度」，此时初始化为 0，并在第一次成功响应后由 `_learn_dimension` 自动学习并写回。`multimodal` 由模型名里是否含 `vl` 推导得出，可以事后显式改写以支持名字里没有 `vl` 的多模态模型。
- **参数**：`api_key: str | None = None` —— 提供方密钥，必填且不能是空白字符串；空值会抛 `RuntimeError`，提示去 `config/services.toml [embedding].api_key` 配置。`model: str = DEFAULT_EMBEDDING_MODEL`（关键字专用）—— 嵌入模型 ID，必须是非空字符串，会做 `strip()`；它同时决定 `multimodal` 标志。`base_url: str = ""`（关键字专用）—— 提供方根地址，必须是非空字符串且是绝对 HTTP(S) URL（含 scheme 与 netloc），否则抛 `ValueError`；末尾斜杠会被去掉。`dimension: int | None = None`（关键字专用）—— 期望向量维度；`None` 表示未知（记为 0，后续自动学习），显式给值则必须是正整数，且会作为后续响应的强校验依据（不匹配时 `_learn_dimension` 会报错）。`timeout: float = 30.0`（关键字专用）—— 单次 HTTP 请求超时秒数，必须是正数（`int` 或 `float`，拒绝布尔值）。`batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE`（关键字专用）—— 批量嵌入时每片的最大条数，必须是正整数。`client: Any = None`（关键字专用）—— 可注入的测试替身；可以是 `callable(payload, *, model=...)`，或暴露 `embed(values, *, model=...)` 的对象，或暴露 `embeddings.create(input=..., model=...)` 的对象；为 `None` 时走真实 `urllib` 请求。
- **返回**：`None`（构造函数，只初始化实例属性）。
- **内部流程**：第一步校验 `model` 是非空字符串；第二步校验 `base_url` 非空，并用 `urlsplit` 解析后确认 scheme 属于 `{"http", "https"}` 且 `netloc` 非空，否则抛 `ValueError("base_url must be an absolute HTTP(S) URL")`；第三步在 `dimension is not None` 时校验其为非布尔正整数；第四步校验 `timeout` 是正数且非布尔；第五步校验 `batch_size` 是非布尔正整数；第六步校验 `api_key` 非空，否则抛 `RuntimeError` 并给出配置位置提示；随后逐项赋值：`api_key` 做 `strip()`、`model` 做 `strip()`、`base_url` 用 `rstrip("/")` 去掉尾部斜杠、`timeout` 转 `float`、`batch_size` 原样保存、`client` 原样保存、`self.dimension = dimension or 0`（把 `None` 与 0 都归一成 0）；最后用 `VL_EMBEDDING_MODEL_MARKER in self.model.casefold()` 计算 `self.multimodal`。
- **异常/边界**：`model` 为空或非字符串抛 `ValueError`；`base_url` 为空、非字符串或不是绝对 HTTP(S) URL 抛 `ValueError`；`dimension` 为布尔/非整数/小于 1 抛 `ValueError`；`timeout` 为布尔/非数值/不大于 0 抛 `ValueError`；`batch_size` 为布尔/非整数/小于 1 抛 `ValueError`；`api_key` 为 `None`、空串或纯空白抛 `RuntimeError`。注意 `dimension=0` 会被 `or` 折叠成 0（等价于未知），不会报错。
- **同文件关系**：写入的 `self.dimension` 被 `_learn_dimension`（第 302 行）读取与更新；`self.model`、`self.client` 被 `_request`（第 318 行）使用；`self.batch_size` 被 `APIEmbedding.embed_batch`（第 186 行）使用；`self.timeout`、`self.base_url`、`self.api_key` 被 `_request` 使用；`self.multimodal` 被 `embed_item`（第 259 行）与 `to_dict`（第 269 行）使用。

### `APIEmbedding.embed(self, text: str) -> list[float]` （第 181 行）
- **作用**：把单条文本嵌入成向量，是 `BaseEmbedding.embed` 契约在本实现下的落地。它的实现非常薄：把文本包成单元素列表交给批量方法，再取回第一个结果。这样做的价值在于所有网络交互、分片、响应解析、维度学习逻辑只在 `embed_batch` → `_embed_batch_once` 一条路径上实现一次，单条嵌入不会形成第二套代码分支，减少行为不一致的风险。它在运行期被记忆层的逐条写入路径以及 `embed_item` 的纯文本分支调用。它也保证了 `embed` 与 `embed_batch([text])[0]` 的结果完全一致。它是整个类里最常用的入口之一。
- **参数**：`text: str` —— 待嵌入的文本；必须是字符串，否则在进入批量方法前就先抛 `TypeError`。
- **返回**：`list[float]` —— 该文本对应的向量，长度等于提供方实际返回的维度（首次调用后会被记录到 `self.dimension`）。
- **内部流程**：第一步类型校验，非字符串抛 `TypeError("text must be a string")`；第二步构造单元素列表 `[text]` 调用 `self.embed_batch(...)`；第三步取下标 `[0]` 返回唯一的向量。由于传入了非空列表，`embed_batch` 内部的空列表短路分支不会被触发，因此不存在索引越界。
- **异常/边界**：非字符串输入抛 `TypeError`；网络错误、HTTP 错误、响应格式异常、维度不一致等都会从 `embed_batch` → `_embed_batch_once` → `_request`/`_extract_embedding_vectors`/`_learn_dimension` 一路上抛为 `RuntimeError`（`_request` 还会把底层 `HTTPError`/`URLError`/JSON 解析错误包装成带状态码与响应片段的 `RuntimeError`）。
- **同文件关系**：调用 `APIEmbedding.embed_batch`（第 186 行），进而间接调用 `_embed_batch_once`、`_request`、`_extract_embedding_vectors`、`_learn_dimension`；被 `APIEmbedding.embed_item`（第 259 行）在纯文本分支调用；是对 `BaseEmbedding.embed`（第 54 行）的实现。

### `APIEmbedding.embed_batch(self, texts: Iterable[str]) -> list[list[float]]` （第 186 行）
- **作用**：批量嵌入文本，是真正走网络请求的批量入口，覆盖了基类的逐条默认实现。它把输入列表按 `batch_size` 切分成多个分片，每个分片发一次 HTTP 请求，再把各分片的结果按顺序拼接起来返回。之所以要分片，是因为提供方对单次请求的 `input` 条数通常有上限，一次塞进去几千条会被拒绝或超时，分片能在保持批量效率的同时避免越界。之所以空输入直接返回 `[]` 而不是发一次空请求，是为了避免对端返回不可预期的结果以及无谓的网络往返。它保证了返回列表与输入列表严格一一对应、顺序不变，这是上层把向量与文本配对的前提。它是批量索引与批量检索性能的关键路径。
- **参数**：`texts: Iterable[str]` —— 任意可迭代的字符串集合；内部先物化成列表，因此生成器只会被消费一次；允许为空。
- **返回**：`list[list[float]]` —— 与输入顺序一一对应的向量列表；输入为空时返回空列表 `[]`；每个向量的长度等于提供方返回的维度。
- **内部流程**：第一步 `values = list(texts)` 物化输入；第二步用 `all(isinstance(text, str) for text in values)` 整体校验元素类型，任一非字符串即抛 `TypeError("texts must contain strings")`；第三步对空列表短路返回 `[]`；第四步初始化 `result: list[list[float]] = []`，用 `range(0, len(values), self.batch_size)` 生成分片起点，对每个分片调用模块级 `_embed_batch_once(self, values[start:start + self.batch_size])`，并用 `result.extend(...)` 追加；第五步返回拼接好的 `result`。
- **异常/边界**：元素含非字符串抛 `TypeError`；空输入返回 `[]`；任一分片请求失败（网络、HTTP、超限、JSON 非法、条数不匹配、维度不一致、非有限值）都会抛出 `RuntimeError` 并中断整个批量，已完成的 `result` 不会被返回（无部分成功语义）；`batch_size` 已在构造期保证为正整数，因此 `range` 步长不会非法。
- **同文件关系**：调用模块级 `_embed_batch_once`（第 284 行），后者再调用 `_request`、`_extract_embedding_vectors`、`_learn_dimension`；被 `APIEmbedding.embed`（第 181 行）以单元素列表调用；覆盖了 `BaseEmbedding.embed_batch`（第 58 行）的默认实现。

### `APIEmbedding.text_input(text: str) -> dict[str, str]` （第 200 行）
- **作用**：构造 VL（视觉语言）嵌入接口所需的 `{"text": ...}` 内容对象，是组装多模态 `input` 列表的积木之一。之所以需要它，是因为 VL 模型的 `/embeddings` 端点不接受裸字符串，而要求 `input` 的元素是带类型标记的内容对象；把这一构造收敛成一个静态方法，可以避免在多个调用点手写字典字面量而写错键名。它被声明为静态方法，因为构造逻辑与实例配置无关（不需要知道模型名或端点）。它同时承担了输入类型守卫的职责，让非字符串在最早的位置被拒绝。它与 `image_input` 成对使用，共同构成 `inputs_for` 的输入源。任何需要手工拼装 VL 输入的调用方都可以直接复用它。
- **参数**：`text: str` —— 要放进内容对象的文本；必须是字符串，允许为空字符串（本方法不检查空白，是否发送由 `inputs_for` 决定）。
- **返回**：`dict[str, str]` —— 形如 `{"text": text}` 的单键字典，键名固定为 `"text"`，值就是原样传入的文本（不做 `strip`）。
- **内部流程**：第一步类型校验，`not isinstance(text, str)` 时抛 `TypeError("text must be a string")`；第二步返回字典字面量 `{"text": text}`，不做任何转换、编码或截断。
- **异常/边界**：非字符串输入抛 `TypeError`；空字符串会被正常包装（不报错），后续是否被发送取决于 `inputs_for` 中的 `text.strip()` 判断；不做长度限制。
- **同文件关系**：被 `APIEmbedding.inputs_for`（第 222 行）调用；与 `APIEmbedding.image_input`（第 206 行）配对；不被本文件其它函数调用。

### `APIEmbedding.image_input(data: Any, *, mime_type: str | None = None) -> dict[str, str]` （第 206 行）
- **作用**：构造 VL 嵌入接口所需的 `{"image": ...}` 内容对象，把多种形态的图片数据统一成提供方要求的字符串形式。它的策略是「尽量不越俎代庖」：如果调用方给的是 URL 或已经编码好的 base64 字符串（裸 base64 或 data URI 都行），就原样传递，因为调用方最清楚提供方期望的编码形式；只有拿到 `bytes`/`bytearray`/`memoryview` 这种二进制数据时，才由本方法负责 base64 编码并拼成 `data:<mime>;base64,...` 的 data URI，因为裸的 base64 串不自描述、服务端无法判断类型。MIME 类型优先使用调用方显式传入的 `mime_type`，否则按魔数猜测（见 `_image_mime_type`），这是因为声明错 MIME 会被服务端拒收。它是 `embed_image` 与 `embed_multimodal` 的底层依赖。
- **参数**：`data: Any` —— 图片数据，支持三种形态：`bytes`/`bytearray`/`memoryview` 二进制；非空字符串（URL 或 base64）；其它类型一律拒绝。`mime_type: str | None = None`（关键字专用）—— 仅在传入二进制时生效的 MIME 覆盖值，例如 `"image/jpeg"`；为 `None` 时按魔数自动判断。
- **返回**：`dict[str, str]` —— 形如 `{"image": "data:image/png;base64,...."}`（二进制输入）或 `{"image": "<原样字符串>"}`（字符串输入）的单键字典，键名固定为 `"image"`。
- **内部流程**：第一步判断是否为 `bytes`/`bytearray`/`memoryview`：是则用 `mime_type or _image_mime_type(data)` 决定 MIME，用 `base64.b64encode(bytes(data)).decode("ascii")` 编码，再拼出 `f"data:{mime};base64,{encoded}"` 返回；第二步否则判断是否为「字符串且 `strip()` 后非空」：是则返回 `{"image": data.strip()}`（去掉首尾空白，避免 URL 里混入换行导致请求失败）；第三步两者都不满足则抛 `TypeError("image data must be bytes, or a non-empty URL/base64 string")`。
- **异常/边界**：`data` 为 `None`、数字、空字符串或纯空白字符串时抛 `TypeError`；`memoryview` 会被 `bytes(...)` 正确转换；不做 base64 合法性校验（假定调用方给的是有效编码）；不做图片大小上限检查；不校验 URL 是否可达。
- **同文件关系**：调用模块级 `_image_mime_type`（第 424 行）做魔数判断；被 `APIEmbedding.inputs_for`（第 222 行）、`embed_image`（第 251 行）、`embed_multimodal`（第 255 行，经 `inputs_for`）调用；与 `text_input`（第 200 行）配对。

### `APIEmbedding.inputs_for(self, text: str = "", *, image: Any = None, mime_type: str | None = None) -> list[Any]` （第 222 行）
- **作用**：把「文本」与「图片」两种可选输入组装成 VL 模型所需的 `input` 列表，是 `embed_multimodal` 的前置步骤。它的核心语义是：非空文本会被包装成 `{"text": ...}` 追加，非 `None` 的图片会被包装成 `{"image": ...}` 追加，两者都提供时列表里就有两个元素——这个列表在后续请求里会被模型融合成单个向量（图文联合嵌入）。之所以要求「至少提供一个」，是因为空列表发出去毫无意义且会被服务端报错，所以这里提前抛 `ValueError` 而不是浪费一次网络往返。它用 `text.strip()` 来判断文本是否「有内容」，因此纯空白文本会被静默忽略，只发图片。它保证了调用方无需自己判断哪一路输入存在。它是 `embed_multimodal` 与外部手工构造 VL 请求时的公共入口。
- **参数**：`text: str = ""` —— 文本部分，默认空字符串表示不提供；非字符串会被忽略（不报错）；纯空白会被忽略。`image: Any = None`（关键字专用）—— 图片部分，默认 `None` 表示不提供；非 `None` 时会交给 `image_input` 处理，因此类型要求与 `image_input` 一致（二进制或非空字符串）。`mime_type: str | None = None`（关键字专用）—— 图片 MIME 覆盖值，仅在 `image` 为二进制时被 `image_input` 使用。
- **返回**：`list[Any]` —— 元素为 `{"text": ...}` 与/或 `{"image": ...}` 字典的列表，顺序固定为「先文本后图片」；长度只可能是 1 或 2。
- **内部流程**：第一步初始化空列表 `items`；第二步判断 `isinstance(text, str) and text.strip()`，成立则 `items.append(self.text_input(text))`（注意这里把原始 `text` 传入，不预先 strip，最终由服务端处理首尾空白）；第三步判断 `image is not None`，成立则 `items.append(self.image_input(image, mime_type=mime_type))`；第四步若 `items` 仍为空则抛 `ValueError("at least one of text/image must be provided")`；第五步返回 `items`。
- **异常/边界**：文本与图片都为空（或文本是纯空白、图片是 `None`）时抛 `ValueError`；`image` 存在但类型非法时，异常由 `image_input` 抛出 `TypeError`；`text` 为非字符串时被静默忽略而不报错，这是与 `text_input` 的严格校验不同的宽松点。
- **同文件关系**：调用 `APIEmbedding.text_input`（第 200 行）与 `APIEmbedding.image_input`（第 206 行）；被 `APIEmbedding.embed_multimodal`（第 255 行）调用；不被本文件其它函数调用。

### `APIEmbedding.embed_inputs(self, items: list[Any]) -> list[float]` （第 233 行）
- **作用**：发送一次 VL 内容对象列表请求，并把模型融合后的单个向量取回来。它与 `_embed_batch_once` 的关键区别在于「多进一出」：`[{"text": ...}, {"image": ...}]` 这样的混合列表是一次请求、一个向量，而 `_embed_batch_once` 处理的是「一进一出」的批量场景，因此这里不能复用批量路径上的条数一致性检查，必须自己断言响应里只有一条向量。之所以要显式检查 `len(vectors) != 1`，是因为如果对端把混合列表当成多条输入分别返回向量，静默取第一条会导致图文向量语义错乱（拿到的可能只是纯文本向量），属于难以察觉的严重错误，必须立刻报错。它在成功后同样会通过 `_learn_dimension` 记录并校验维度，使首次调用就能确定 `self.dimension`。它是 `embed_image` 与 `embed_multimodal` 的共同实现底座。
- **参数**：`items: list[Any]` —— 待发送的 VL `input` 列表，元素应为 `{"text": ...}` 或 `{"image": ...}` 内容对象（通常由 `text_input`/`image_input`/`inputs_for` 生成）；必须是列表且非空。
- **返回**：`list[float]` —— 唯一的一条融合向量，长度等于提供方维度。
- **内部流程**：第一步校验 `items` 是 `list` 且非空，否则抛 `ValueError("items must be a non-empty list")`；第二步调用模块级 `_request(self, items)` 发送请求并拿到已解析的 JSON；第三步调用 `_extract_embedding_vectors(...)` 把响应归一化成向量列表；第四步若 `len(vectors) != 1` 则抛 `RuntimeError`，消息里带上实际条数与「VL 输入列表期望单个融合向量」的说明；第五步调用 `_learn_dimension(self, vectors)` 校验形状、有限性与维度一致性并写回维度；第六步返回 `vectors[0]`。
- **异常/边界**：`items` 非列表或为空抛 `ValueError`；响应条数不为 1 抛 `RuntimeError`；响应为空向量、维度不一致、含非有限值（`NaN`/`inf`）或与已知 `dimension` 冲突，都会由 `_learn_dimension` 抛 `RuntimeError`；网络与解析错误由 `_request` 包装成 `RuntimeError`；不做多请求分片（VL 列表本来就是一次请求）。
- **同文件关系**：调用模块级 `_request`（第 318 行）、`_extract_embedding_vectors`（第 369 行）、`_learn_dimension`（第 302 行）；被 `APIEmbedding.embed_image`（第 251 行）与 `APIEmbedding.embed_multimodal`（第 255 行）调用。

### `APIEmbedding.embed_image(self, data: Any, *, mime_type: str | None = None) -> list[float]` （第 251 行）
- **作用**：只嵌入一张图片、不附带任何文本，得到纯图像向量。它的用途是让存进记忆的图片在没有标题或描述的情况下也能被检索到，或者在做以图搜图的场景下直接产生查询向量。它在实现上把图片包装成单元素 VL 列表交给 `embed_inputs`，因此完全复用了请求、解析、维度学习这套逻辑，不需要重复代码。它不受 `multimodal` 标志限制——即使当前模型名里不含 `vl`，调用方也可以显式调用它（能否成功取决于服务端是否真的支持图像输入）。它是 `embed_item` 在「无文本但有载荷且是多模态模型」分支下的实际执行者。它把「图片 → 向量」这件事从记忆层解耦出来，成为一个可单独测试的能力。
- **参数**：`data: Any` —— 图片数据，要求与 `image_input` 一致：`bytes`/`bytearray`/`memoryview`，或非空字符串（URL 或 base64）。`mime_type: str | None = None`（关键字专用）—— MIME 覆盖值，仅在二进制输入时生效。
- **返回**：`list[float]` —— 该图片的嵌入向量，长度等于提供方维度。
- **内部流程**：先调用 `self.image_input(data, mime_type=mime_type)` 得到 `{"image": ...}` 字典，把它放进单元素列表，再调用 `self.embed_inputs([...])` 并直接返回其结果。全流程没有额外的分支或后处理。
- **异常/边界**：`data` 类型非法（`None`、数字、空字符串）时由 `image_input` 抛 `TypeError`；服务端返回条数不是 1、维度不一致、含非有限值等由 `embed_inputs`/`_learn_dimension` 抛 `RuntimeError`；网络失败由 `_request` 抛 `RuntimeError`；不做图片大小与格式支持性预检。
- **同文件关系**：调用 `APIEmbedding.image_input`（第 206 行）与 `APIEmbedding.embed_inputs`（第 233 行）；被 `APIEmbedding.embed_item`（第 259 行）在无文本分支调用。

### `APIEmbedding.embed_multimodal(self, text: str, data: Any, *, mime_type: str | None = None) -> list[float]` （第 255 行）
- **作用**：把一段文本和一张图片一起送入模型，得到图文融合后的单个向量。这在记忆系统里对应「一条记忆同时有文字描述和配图」的情况：把两者融合进同一个向量，意味着检索时用文字描述或图片内容都能命中同一条记忆，而不必维护两套索引。它同样不受 `multimodal` 标志限制，可以手动调用。实现上它先通过 `inputs_for` 组装出「文本对象 + 图片对象」的列表，再交给 `embed_inputs` 完成请求与解析，因此当 `text` 为空或纯空白时会退化成纯图片嵌入，当图片非法时异常来自 `image_input`。它是 `embed_item` 在「有文本且有载荷且是多模态模型」分支下的执行者。它把融合逻辑完全交给服务端模型，本地不做任何向量拼接或加权。
- **参数**：`text: str` —— 与图片配对的文本描述；为空字符串或纯空白时该路被忽略，只剩图片。`data: Any` —— 图片数据，要求与 `image_input` 一致（二进制或非空字符串）。`mime_type: str | None = None`（关键字专用）—— 图片 MIME 覆盖值，仅在二进制输入时生效。
- **返回**：`list[float]` —— 图文融合后的单个向量，长度等于提供方维度。
- **内部流程**：第一步调用 `self.inputs_for(text, image=data, mime_type=mime_type)` 组装 VL `input` 列表（内部会按需生成文本对象与图片对象，并保证至少有一项）；第二步把该列表交给 `self.embed_inputs(...)`；第三步返回其唯一向量。若 `text` 为纯空白而 `data` 有效，`inputs_for` 只会产出图片对象，行为等价于 `embed_image`。
- **异常/边界**：`text` 与 `data` 同时为空（`text` 为空白且 `data` 为 `None`）时，`inputs_for` 抛 `ValueError`；`data` 类型非法时 `image_input` 抛 `TypeError`；响应条数不为 1、维度不一致、非有限值由 `embed_inputs`/`_learn_dimension` 抛 `RuntimeError`；`text` 为非字符串时被 `inputs_for` 静默忽略（不会报错），可能意外退化为纯图片嵌入。
- **同文件关系**：调用 `APIEmbedding.inputs_for`（第 222 行）与 `APIEmbedding.embed_inputs`（第 233 行）；被 `APIEmbedding.embed_item`（第 259 行）在「有文本 + 有载荷 + 多模态」分支调用。

### `APIEmbedding.embed_item(self, text: str, *, payload: Any = None, modality: str | None = None) -> list[float]` （第 259 行）
- **作用**：这是 `BaseEmbedding.embed_item` 钩子在云端实现下的覆盖版本，负责在记忆写入时自动决定「这次该用纯文本嵌入还是多模态嵌入」。判断逻辑只有两条：如果 `payload` 为 `None`，或者当前模型不是多模态（`self.multimodal` 为假），就退回普通文本嵌入，行为与旧版本完全一致；否则看文本是否有实质内容——有文本就做图文融合嵌入，没文本（空或纯空白）就只嵌入图片。这样设计的意义在于：记忆层的写入路径只有一种调用形式，而「这条记忆带不带图、当前模型支不支持图」的决策被收敛在这里，上层不需要写任何 `if`。它刻意忽略了 `modality` 参数，因为是否走多模态由 `payload` 是否存在和模型能力共同决定，比调用方传入的模态标记更可靠。它是记忆系统「存下来的图片能按自身内容被检索到」这一能力的最后一块拼图。
- **参数**：`text: str` —— 记忆条目的文本表示。`payload: Any = None`（关键字专用）—— 附加载荷，在记忆场景里通常是图片的字节、URL 或 base64 字符串；为 `None` 时走纯文本路径。`modality: str | None = None`（关键字专用）—— 模态标记，本实现中不参与任何判断，被完全忽略。
- **返回**：`list[float]` —— 一条向量：纯文本路径下是文本向量，多模态路径下是图文融合向量或纯图像向量；长度均等于提供方维度。
- **内部流程**：第一步判断 `payload is None or not self.multimodal`，成立则直接 `return self.embed(text)`（此时文本类型非法会由 `embed` 抛 `TypeError`）；第二步判断 `isinstance(text, str) and text.strip()`，成立则 `return self.embed_multimodal(text, payload)`（`mime_type` 未传，二进制图片的 MIME 由 `_image_mime_type` 按魔数推断）；第三步否则（有载荷、多模态模型、但文本为空或纯空白）`return self.embed_image(payload)`。
- **异常/边界**：`payload` 非 `None` 且模型是多模态、但 `payload` 类型不是二进制也不是非空字符串时，`embed_image`/`image_input` 抛 `TypeError`；`text` 为 `None` 且 `payload` 非 `None` 且是多模态时，会走 `embed_image` 分支（`isinstance(None, str)` 为假）而不报错；网络与响应异常由下游 `RuntimeError` 抛出；`modality` 参数无论取值如何都不影响结果。
- **同文件关系**：调用 `APIEmbedding.embed`（第 181 行）、`APIEmbedding.embed_multimodal`（第 255 行）、`APIEmbedding.embed_image`（第 251 行），并间接依赖 `inputs_for`、`image_input`、`embed_inputs`、`_request`、`_extract_embedding_vectors`、`_learn_dimension`、`_image_mime_type`；覆盖 `BaseEmbedding.embed_item`（第 64 行）。

### `APIEmbedding.to_dict(self) -> dict[str, Any]` （第 269 行）
- **作用**：把云端嵌入客户端的当前配置导出成普通字典，用于健康检查、日志与配置回显。它包含六个键：类名、模型名、端点根地址、当前已知维度、批大小，以及 `multimodal` 标志——最后这一项的存在意义是让 `/api/health` 能直接看出「当前是不是按 VL 内容对象在发请求」，而不必去翻配置或看日志。它刻意不包含 `api_key`，避免密钥泄漏到响应或日志里。它与 `HashEmbedding.to_dict` 保持同构（都至少有 `type` 与 `dimension`），使上层可以用统一方式收集嵌入后端信息。它是运行期自检与故障排查的重要信息源。
- **参数**：无（除 `self`）。
- **返回**：`dict[str, Any]` —— 包含 `"type"`（`type(self).__name__`，即 `"APIEmbedding"`）、`"model"`（模型 ID）、`"base_url"`（已去掉尾部斜杠的根地址）、`"dimension"`（当前维度整数；若尚未成功调用过则为 0）、`"batch_size"`（批大小整数）、`"multimodal"`（布尔，是否按 VL 内容对象发请求）的字典。
- **内部流程**：直接构造并返回字典字面量；`type(self).__name__` 保证子类得到正确类名；所有值都取自实例属性，不做网络调用、不做缓存、不做脱敏以外的处理。
- **异常/边界**：无特殊处理；不含 `api_key` 与 `timeout`；返回的是新建可变字典，修改它不影响实例。
- **同文件关系**：只读取自身属性；与 `HashEmbedding.to_dict`（第 109 行）形成同构接口；不被本文件其它函数调用。

### `APIEmbedding.__repr__(self) -> str` （第 280 行）
- **作用**：给出便于调试的对象表示，把模型名、维度与端点地址拼成一个简短字符串。之所以需要它，是因为在排查「连到了哪个端点、当前维度是多少」这类问题时，直接在日志或交互式终端里看到这三个关键值能省去大量猜测，而默认 `repr` 只显示内存地址。它刻意不打印 `api_key`，避免密钥通过日志或异常回溯泄漏。它对排查「维度守卫拒绝写入」这类问题特别有用，因为维度是否已经学到、是否与预期一致，一眼可见。它是纯只读方法，不改变状态、不发网络请求。
- **参数**：无（除 `self`）。
- **返回**：`str` —— 形如 `APIEmbedding(model='BAAI/bge-m3', dimension=1024, base_url='https://api.siliconflow.cn/v1')` 的字符串；`model` 与 `base_url` 用 `!r` 加引号，便于看出空白字符。
- **内部流程**：使用 f-string 拼装，`self.model!r` 与 `self.base_url!r` 使用 repr 形式，`self.dimension` 直接内联；不截断、不脱敏模型名（模型名不敏感）、不输出密钥。
- **异常/边界**：无特殊处理；若实例属性缺失会抛 `AttributeError`，正常情况下不会发生。
- **同文件关系**：只读取自身属性；与 `HashEmbedding.__repr__`（第 112 行）风格一致；不被本文件其它函数调用。

### `_embed_batch_once(embedding: APIEmbedding, values: list[Any]) -> list[list[float]]` （第 284 行）
- **作用**：这是模块级的「发一次批量请求并把响应归一化成向量列表」的辅助函数，是 `APIEmbedding.embed_batch` 分片循环的实际执行体。它被拆成模块级私有函数而不是方法，是因为它只依赖传入的 `embedding` 实例，逻辑上属于「协议适配」而非对象状态操作，这样也更便于单独测试。它同时服务两种输入：文本模型传的是纯字符串列表，VL 模型传的是 `{"text": ...}`/`{"image": ...}` 内容对象列表——两者都塞进同一个 OpenAI 兼容的 `input` 字段，因此这里不需要分支判断。它严格执行「一进一出」的条数一致性检查，这是批量场景下防止向量与文本错位的关键防线。成功后会调用 `_learn_dimension` 记录维度。它是批量路径与 VL 融合路径的分界点：融合列表由 `APIEmbedding.embed_inputs` 单独处理，不走这里。
- **参数**：`embedding: APIEmbedding` —— 提供端点、密钥、模型、超时与可选测试客户端的嵌入实例，函数从中读取配置并写回学到的维度。`values: list[Any]` —— 本分片的输入列表，元素为字符串（文本模型）或 VL 内容对象字典（多模态模型）；调用方保证非空（`embed_batch` 已短路空输入，且分片长度至少为 1）。
- **返回**：`list[list[float]]` —— 与 `values` 等长、顺序一致的向量列表；每个向量是浮点列表。
- **内部流程**：第一步调用 `_request(embedding, values)` 完成 HTTP 调用（或走注入的测试客户端）并得到解析后的 JSON；第二步调用 `_extract_embedding_vectors(response)` 把响应统一成向量列表（兼容 OpenAI 与 DashScope 两种布局，并按 `index`/`text_index` 重排）；第三步比较 `len(vectors)` 与 `len(values)`，不相等则抛 `RuntimeError`，消息里同时给出实际条数与输入条数；第四步调用 `_learn_dimension(embedding, vectors)` 校验形状、有限性与维度一致性并写回 `embedding.dimension`；第五步返回 `vectors`。
- **异常/边界**：条数不匹配抛 `RuntimeError`；响应无 `data`/`output.embeddings`、元素缺少 `embedding`、向量为空、向量含非数值、索引不完整等由 `_extract_embedding_vectors` 抛 `RuntimeError`；空向量、维度不一致、非有限值、与已知维度冲突由 `_learn_dimension` 抛 `RuntimeError`；网络与 HTTP 错误由 `_request` 包装成 `RuntimeError`；不做重试、不做退避。
- **同文件关系**：调用模块级 `_request`（第 318 行）、`_extract_embedding_vectors`（第 369 行）、`_learn_dimension`（第 302 行）；被 `APIEmbedding.embed_batch`（第 186 行）调用；与 `APIEmbedding.embed_inputs`（第 233 行）是并列的两条请求路径（后者用于融合列表）。

### `_learn_dimension(embedding: APIEmbedding, vectors: list[list[float]]) -> None` （第 302 行）
- **作用**：这是响应向量的「形状与数值体检」函数，同时负责把从响应中学到的维度记录回嵌入实例。它的存在是向量库维度守卫的前提：`QdrantVectorStore` 要求写入的向量维度与集合维度严格一致，如果嵌入服务返回了意外维度（比如模型被换成了另一个版本），必须在写入向量库之前就报错，而不是让 Qdrant 抛出难以理解的错误。它依次检查四件事：向量是否为空、所有向量长度是否一致、数值是否都是有限值、以及是否与实例上已知的 `dimension` 冲突；全部通过后把 `embedding.dimension` 更新为实测维度。这样第一次调用会「学习」维度，后续调用则变成强校验。它对 `NaN`/`inf` 的检查尤其重要，因为这类值会污染整个向量索引且事后很难排查。它是所有成功响应路径的必经关卡。
- **参数**：`embedding: APIEmbedding` —— 目标嵌入实例；函数会读取其 `dimension` 属性做一致性校验，并在通过后写回新值。`vectors: list[list[float]]` —— 待校验的向量列表，可能为空列表（此时会直接报错）。
- **返回**：`None` —— 无返回值，效果是校验通过后把维度写回 `embedding.dimension`。
- **内部流程**：第一步 `dimension = len(vectors[0]) if vectors else 0`，用第一条向量的长度作为基准维度，空列表则取 0；第二步若 `dimension == 0` 抛 `RuntimeError("embedding response contained empty vectors")`（同时覆盖「列表为空」和「第一条向量为空」两种情况）；第三步用 `any(len(vector) != dimension for vector in vectors)` 检查所有向量长度一致，不一致抛 `RuntimeError("embedding response contained inconsistent dimensions")`；第四步用 `any(not math.isfinite(value) for vector in vectors for value in vector)` 扫描所有元素，出现非有限值抛 `RuntimeError("embedding response contained non-finite values")`；第五步若 `embedding.dimension` 非 0 且不等于实测 `dimension`，抛 `RuntimeError`，消息同时给出实际维度与期望维度；第六步 `embedding.dimension = dimension` 写回。
- **异常/边界**：`vectors` 为空、首条向量为空、长度不一致、含 `NaN`/`inf`/`-inf`、与已知维度不符，五种情况都抛 `RuntimeError`；实例维度为 0（未知）时不报错而是接受并学习；不做元素类型校验（元素通常已在 `_extract_embedding_vectors` 中被转成 `float`，若传入字符串会在 `math.isfinite` 处抛 `TypeError`）。
- **同文件关系**：调用标准库 `math.isfinite`；被 `_embed_batch_once`（第 284 行）与 `APIEmbedding.embed_inputs`（第 233 行）调用；它读取并写入 `APIEmbedding.__init__`（第 141 行）设置的 `dimension` 属性。

### `_request(embedding: APIEmbedding, values: list[Any]) -> Any` （第 318 行）
- **作用**：这是整个文件里唯一真正发起网络 I/O 的函数，负责组装请求体、选择真实 HTTP 调用或注入的测试客户端、发送请求、读取响应并解析 JSON。把网络逻辑集中在一个模块级函数里，使得 `APIEmbedding` 的各个嵌入方法只关心「输入什么、拿回什么」，也让测试可以通过注入 `client` 完全绕开网络。它构造的请求严格遵循 OpenAI 形状：`POST {base_url}/embeddings`，`Authorization: Bearer <api_key>`，`Content-Type: application/json`，体为 `{"model": ..., "input": values}`。它显式限制响应体大小（`EMBEDDING_RESPONSE_MAX_BYTES`，8 MiB），避免对端行为异常时把超大响应整段读进内存；为此还兼容了只暴露无参 `read()` 的测试替身。它把所有底层异常统一翻译成带上下文的 `RuntimeError`，让上层的错误信息更有诊断价值。它是 `_embed_batch_once` 与 `APIEmbedding.embed_inputs` 的共同底座。
- **参数**：`embedding: APIEmbedding` —— 提供 `model`、`client`、`base_url`、`api_key`、`timeout` 的实例。`values: list[Any]` —— 要放进 `input` 字段的内容列表，元素为字符串或 VL 内容对象字典；调用方保证非空。
- **返回**：`Any` —— 解析后的 JSON 对象。真实 HTTP 路径下是 `json.loads` 的结果（通常是 `dict`）；注入客户端路径下是客户端自行返回的任意已解析对象（可以是 `dict`，也可以是带属性的对象，`_extract_embedding_vectors` 两种都支持）。
- **内部流程**：第一步构造 `payload = {"model": embedding.model, "input": values}`；第二步检查 `embedding.client`：若不为 `None`，依次尝试三种注入形式——`callable(client)` 则调用 `client(payload, model=embedding.model)`；否则若有可调用的 `client.embed` 则调用 `client.embed(values, model=embedding.model)`；否则若有 `client.embeddings.create` 则调用 `client.embeddings.create(input=values, model=embedding.model)`；三者都不满足则抛 `TypeError("client must be callable or expose embed()/embeddings.create()")`；第三步无客户端时，用 `urlsplit` 再次校验 `base_url` 是绝对 HTTP(S) URL（防御性二次校验）；第四步 `json.dumps(payload).encode("utf-8")` 得到请求体字节；第五步用 `urllib.request.Request` 构造 `POST {base_url}/embeddings` 请求并带上两个头部；第六步在 `try` 中 `urllib.request.urlopen(request, timeout=embedding.timeout)`，用 `with` 管理连接；第七步尝试 `response.read(EMBEDDING_RESPONSE_MAX_BYTES + 1)`（多读 1 字节以便判断是否超限），若抛 `TypeError`（测试替身只有无参 `read()`）则退回 `response.read()`；第八步若 `len(body) > EMBEDDING_RESPONSE_MAX_BYTES` 抛 `OverflowError`；第九步 `json.loads(body.decode("utf-8"))` 返回结果。异常处理部分依次捕获：`urllib.error.HTTPError` 读取前 500 字符响应体并抛 `RuntimeError`（带状态码，`from exc` 保留因果链）；`urllib.error.URLError` 抛 `RuntimeError`（带 `exc.reason`）；`ValueError`（JSON 解析失败）抛 `RuntimeError("embedding API returned invalid JSON: ...")`；`OverflowError` 抛 `RuntimeError("embedding API response is too large: ...")`。
- **异常/边界**：注入客户端形式非法抛 `TypeError`；`base_url` 非法抛 `ValueError`；HTTP 错误、连接失败/DNS 失败/超时（`URLError`，超时在 urllib 中表现为 `socket.timeout` 被包装进 `URLError` 或直接抛出，此处只显式处理 `URLError`）、JSON 非法、响应体超 8 MiB 分别抛对应 `RuntimeError`；`OverflowError` 是先抛出再被捕获并重新包装，因此对外呈现为 `RuntimeError`；不做重试、不做退避、不做并发；读取响应时最多读 `8 MiB + 1` 字节，超限即拒绝。
- **同文件关系**：读取 `APIEmbedding` 实例的 `model`、`client`、`base_url`、`api_key`、`timeout`；被 `_embed_batch_once`（第 284 行）与 `APIEmbedding.embed_inputs`（第 233 行）调用；使用模块级常量 `EMBEDDING_RESPONSE_MAX_BYTES`（第 45 行）。

### `_extract_embedding_vectors(response: Any) -> list[list[float]]` （第 369 行）
- **作用**：这是响应解析器，负责把不同提供方五花八门的响应结构统一成「按输入顺序排列的浮点向量列表」。它同时兼容两种布局：OpenAI 风格的 `data[].embedding`，以及 DashScope 原生的 `output.embeddings`，因此同一个 `APIEmbedding` 可以无缝对接这两类网关。它还兼容两种对象形态：`dict`（真实 `json.loads` 的结果）和带属性的对象（注入的测试替身可能返回 SDK 风格对象），并且对元素也同时支持 `item["embedding"]` 与 `item.embedding`。它最关键的一步是「按显式索引重排」：部分提供方会在每个结果里带上 `index` 或 `text_index`，但返回顺序未必与请求顺序一致，因此它先按显式索引（没有则用枚举位置）收集 `(位置, 向量)` 对，排序后再校验位置序列必须恰好是 `0..n-1`，从而保证「向量与输入严格对应」这一不变量。任何结构异常都会转成明确的 `RuntimeError`。它是所有嵌入路径（批量与 VL）的公共解析入口。
- **参数**：`response: Any` —— 待解析的响应对象。可以是 `dict`（期望含 `data` 或 `output.embeddings`），也可以是带 `data`/`output` 属性的任意对象；元素可以是含 `embedding`/`index`/`text_index` 键的 `dict`，也可以是带同名属性的对象，甚至是可直接转 `float` 的序列。
- **返回**：`list[list[float]]` —— 按索引升序排列的向量列表，每个向量是 `float` 列表；顺序已保证与请求输入顺序一致。
- **内部流程**：第一步判断 `isinstance(response, dict)`：是字典则取 `response.get("data")`，为 `None` 时再取 `response.get("output", {})` 并从中取 `embeddings`（要求 `output` 是字典，否则置 `None`）；不是字典则用 `getattr(response, "data", None)`，为 `None` 时用 `getattr(response, "output", None)` 再取 `embeddings` 属性。第二步若最终 `data is None` 抛 `RuntimeError("embedding response contained no data/embeddings list")`。第三步遍历 `enumerate(data)`，对每个元素：是字典则取 `values = item.get("embedding")`、`explicit = item.get("index", item.get("text_index"))`；否则取 `values = getattr(item, "embedding", item)`（没有 `embedding` 属性时把元素本身当向量）、`explicit = getattr(item, "index", None)`，若为 `None` 再试 `getattr(item, "text_index", None)`。第四步若 `values is None` 抛 `RuntimeError("embedding response item contained no embedding vector")`；第五步用列表推导 `[float(value) for value in values]` 转成浮点，捕获 `TypeError`/`ValueError` 并抛 `RuntimeError("embedding response item contained an invalid vector")`；第六步若向量为空抛 `RuntimeError("embedding response item contained an empty vector")`；第七步按 `explicit` 是否存在决定记录 `(int(explicit), vector)` 还是 `(index, vector)`。第八步 `indexed.sort(key=lambda pair: pair[0])` 按位置排序，取出 `positions` 列表，若 `positions != list(range(len(indexed)))` 则抛 `RuntimeError("embedding response indices were incomplete")`（用于发现索引缺失、重复或从 1 开始编号等问题）；第九步返回按序排列的纯向量列表。
- **异常/边界**：响应没有 `data`/`output.embeddings` 抛 `RuntimeError`；元素缺 `embedding` 抛 `RuntimeError`；元素向量含非数值抛 `RuntimeError`；元素向量为空抛 `RuntimeError`；索引不构成完整的 `0..n-1` 序列抛 `RuntimeError`；`explicit` 为非整数字符串时 `int(explicit)` 会抛 `ValueError`（未被捕获，会直接冒出）；`data` 不是可迭代对象时 `enumerate` 抛 `TypeError`（未被捕获）；不做条数与输入数量的比较（那是 `_embed_batch_once` 的职责）；不做数值有限性检查（那是 `_learn_dimension` 的职责）。
- **同文件关系**：被 `_embed_batch_once`（第 284 行）与 `APIEmbedding.embed_inputs`（第 233 行）调用；其输出随后交给 `_learn_dimension`（第 302 行）校验。

### `_image_mime_type(data: Any, fallback: str = "image/png") -> str` （第 424 行）
- **作用**：按文件头魔数猜测图片的 MIME 类型，用于在把二进制图片转成 data URI 时填上正确的媒体类型。之所以不能一律写 `image/png`，是因为声明错误的 MIME 会被服务端拒收（例如实际是 JPEG 却声称 PNG），导致多模态嵌入请求失败，因此这里宁可花几个字节的比较去猜。它支持的魔数由模块级常量 `_IMAGE_MAGIC` 定义：PNG 的 `\x89PNG\r\n\x1a\n`、JPEG 的 `\xff\xd8\xff`、GIF 的 `GIF87a` 与 `GIF89a`，另外单独识别 RIFF 容器里的 WebP（`RIFF` 开头且第 8 到 12 字节为 `WEBP`）。猜不出时返回 `fallback`（默认 `image/png`），保证函数总有返回值而不抛异常。它是 `image_input` 在二进制输入路径上的依赖。
- **参数**：`data: Any` —— 图片数据；只有 `bytes`/`bytearray`/`memoryview` 会被真正检查，其它类型一律直接返回 `fallback`。`fallback: str = "image/png"` —— 无法识别时返回的 MIME 字符串，默认 `"image/png"`，调用方可以改成别的默认值。
- **返回**：`str` —— 识别出的 MIME 字符串，取值范围为 `"image/png"`、`"image/jpeg"`、`"image/gif"`、`"image/webp"`，或未识别时返回传入的 `fallback`。
- **内部流程**：第一步判断 `data` 是否为 `bytes`/`bytearray`/`memoryview`；不是则直接跳过整个检查逻辑返回 `fallback`。第二步 `head = bytes(data)[:12]` 取前 12 字节（WebP 判断需要看到第 8 到 12 字节）。第三步遍历 `_IMAGE_MAGIC` 中的 `(magic, mime)` 对，用 `head.startswith(magic)` 逐一比对，命中即返回对应 MIME。第四步若都没有命中，检查 `head[:4] == b"RIFF"` 且 `head[8:12] == b"WEBP"`，成立则返回 `"image/webp"`。第五步否则返回 `fallback`。
- **异常/边界**：对非二进制输入、空字节串、不足 12 字节的短数据都不抛异常（短数据在切片比较时自然不匹配，返回 `fallback`）；不识别 BMP、TIFF、AVIF、HEIC 等格式（会落到 `fallback`）；`memoryview` 会被 `bytes(...)` 复制成字节串；不做真实解码校验，只看文件头。
- **同文件关系**：被 `APIEmbedding.image_input`（第 206 行）调用；读取模块级常量 `_IMAGE_MAGIC`（第 416 行）；不被本文件其它函数调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `BaseEmbedding` | 所有嵌入提供方的抽象基类，定义 `embed` 契约并提供 `embed_batch`/`embed_item` 默认实现。 |
| `BaseEmbedding.embed` | 抽象方法，规定「一条文本 → 一个有限数值向量」的最小能力，未实现即报错。 |
| `BaseEmbedding.embed_batch` | 默认批处理实现：校验元素全为字符串后逐条调用 `embed` 顺序映射。 |
| `BaseEmbedding.embed_item` | 记忆条目级嵌入钩子，默认丢弃 `payload`/`modality` 只嵌入文本。 |
| `HashEmbedding` | 确定性离线哈希词袋嵌入器，作为无云端配置时的兜底与测试替身。 |
| `HashEmbedding.__init__` | 校验并保存输出维度，只接受非布尔的负整数以外的正整数。 |
| `HashEmbedding.tokenize` | 大小写折叠后用 `\w+` 正则切词，供词频统计使用。 |
| `HashEmbedding._index` | 用 blake2b 摘要取模把 token 映射到向量下标。 |
| `HashEmbedding.embed` | 统计词频、按 `1 + ln(count)` 累加到哈希维度，再做 L2 归一化。 |
| `HashEmbedding.to_dict` | 导出 `{type, dimension}` 供序列化与自检。 |
| `HashEmbedding.__repr__` | 输出 `HashEmbedding(dimension=...)` 便于调试。 |
| `APIEmbedding` | OpenAI 兼容 `/embeddings` 客户端，支持云端多提供方与 VL 多模态内容对象。 |
| `APIEmbedding.__init__` | 逐项校验并保存密钥、模型、端点、维度、超时、批大小与可选测试客户端，并推导 `multimodal`。 |
| `APIEmbedding.embed` | 单条文本嵌入，委托给 `embed_batch([text])` 取第一个结果。 |
| `APIEmbedding.embed_batch` | 按 `batch_size` 分片批量嵌入，空输入短路返回空列表。 |
| `APIEmbedding.text_input` | 构造 VL 的 `{"text": ...}` 内容对象。 |
| `APIEmbedding.image_input` | 构造 VL 的 `{"image": ...}` 内容对象，二进制自动转 data URI。 |
| `APIEmbedding.inputs_for` | 组装 VL `input` 列表（文本与/或图片），全空时抛错。 |
| `APIEmbedding.embed_inputs` | 发送 VL 列表并断言只返回一个融合向量。 |
| `APIEmbedding.embed_image` | 仅嵌入一张图片得到纯图像向量。 |
| `APIEmbedding.embed_multimodal` | 文本与图片一起送入，得到图文融合向量。 |
| `APIEmbedding.embed_item` | 按 `payload` 与 `multimodal` 自动路由到纯文本、图文融合或纯图像嵌入。 |
| `APIEmbedding.to_dict` | 导出类型、模型、端点、维度、批大小与 `multimodal` 标志（不含密钥）。 |
| `APIEmbedding.__repr__` | 输出模型名、维度与端点地址，便于排查连接与维度问题。 |
| `_embed_batch_once` | 发一次批量请求，解析响应并校验条数一致后学习维度。 |
| `_learn_dimension` | 校验向量非空、长度一致、数值有限、与已知维度不冲突，并写回维度。 |
| `_request` | 组装并发送 `/embeddings` 请求（或调用注入客户端），限流响应体并统一包装异常。 |
| `_extract_embedding_vectors` | 兼容 OpenAI 与 DashScope 两种布局，按显式索引重排并转成浮点向量列表。 |
| `_image_mime_type` | 按文件头魔数猜图片 MIME，猜不出返回 fallback。 |
