# memory/storage/vector.py

## 一、这个文件是干什么的

这个文件是记忆系统的「向量索引存储层」，负责把已经带嵌入向量（embedding）的记忆条目放进一个可按相似度检索的索引里，并提供统一的增删查接口。它定义了一个抽象基类 `BaseVectorStore`，用来约束所有向量库实现必须提供的三个动作：`upsert`（写入或更新）、`delete`（按 id 删除）、`search`（按查询向量做近邻检索并返回 `(条目 id, 相似度分数)` 列表）。文件里给出了一个纯内存的具体实现 `InMemoryVectorStore`，用普通字典保存 `id -> (向量, 记忆类型)`，配合一把可重入锁 `threading.RLock` 保证多线程下的读写安全，适合作为默认实现以及单元测试用的轻量索引。此外文件导出一个模块级工具函数 `cosine_similarity`，用纯 Python 计算两个等长向量的余弦相似度，是所有检索打分的核心。整个文件不依赖任何第三方库（没有 numpy、没有 faiss），只用 `math`、`threading` 和 `abc`，因此零依赖、可移植、易于在测试环境直接构造。上层的记忆管理器在写入记忆时调用 `upsert`，在删除记忆时调用 `delete`，在语义检索时调用 `search`；当嵌入模型或向量维度发生变化、需要重建向量空间时，会调用 `recreate_collection` 清空索引以便重新灌入全部向量。文件末尾用 `__all__` 显式声明对外暴露的三个名字：`BaseVectorStore`、`InMemoryVectorStore`、`cosine_similarity`。注意 `recreate_collection` 只定义在内存实现上，并未出现在抽象基类的接口里。

## 二、函数与类逐条详解

### `class BaseVectorStore(ABC)` （第 12 行）
- **作用**：这是整个向量存储层的抽象基类，作用是给「向量索引」这件事定下一份统一契约，让上层记忆系统不需要知道底层到底用的是内存字典、本地向量库还是远程向量服务，只要面向这三个方法编程即可。它继承自 `abc.ABC`，因此本身不能被实例化，只能被具体的向量库实现继承。把接口抽出来的直接好处是可替换性：单元测试里用内存实现，生产环境里换成远程实现，调用方代码一行都不用改。同时它也是类型标注的锚点，函数签名里写 `BaseVectorStore` 就能表达「任何符合该协议的向量库」。类体内没有保存任何状态，也不包含任何具体逻辑，纯接口。
- **参数**：类本身无构造参数（未定义 `__init__`，由 `ABC` 提供默认行为，且因存在抽象方法而无法直接实例化）。
- **返回**：类是类型对象本身；实例化会因抽象方法未实现而抛出 `TypeError`。
- **内部流程**：类体只声明三个被 `@abstractmethod` 装饰的方法签名，方法体统一写成 `...`（Ellipsis），表示「只占位、不实现」。任何子类必须同时实现这三个方法，否则实例化时 Python 的 ABC 机制会拒绝并报错。
- **异常/边界**：直接 `BaseVectorStore()` 会抛出 `TypeError: Can't instantiate abstract class ... with abstract methods delete, search, upsert`；子类只实现一部分同样无法实例化。除此之外无其它异常。
- **同文件关系**：被同文件的 `InMemoryVectorStore` 继承；其抽象方法 `upsert`、`delete`、`search` 分别被 `InMemoryVectorStore.upsert`、`InMemoryVectorStore.delete`、`InMemoryVectorStore.search` 覆盖实现。

### `upsert(self, item: MemoryItem) -> None` （第 14 行）
- **作用**：抽象方法，声明「把一个记忆条目写入向量索引」这一能力。命名为 upsert（update + insert）是因为它同时承担新增和覆盖更新两种语义：条目 id 已存在就替换其向量，不存在就新增。之所以需要这个动作，是因为记忆内容可能被修正或重新嵌入，索引必须能就地覆盖而不是产生重复项。它是上层「记忆写入」路径的终点之一。返回值刻意设计为 `None`，表示写入失败不应通过返回值表达，而应当直接抛异常。该签名不关心条目里是否真的有向量，这个判断留给具体实现。
- **参数**：`item: MemoryItem` —— 待写入的记忆条目对象，类型来自 `..base` 模块。抽象签名只约束类型，不约束其字段取值；具体实现自行决定当 `item.embedding` 为 `None` 或为空时如何处理。
- **返回**：声明返回 `None`，无任何返回数据。
- **内部流程**：方法体是 `...`，没有任何执行逻辑，仅作为接口占位与文档说明。
- **异常/边界**：抽象方法自身不抛异常；具体实现可以自行决定异常策略（同文件的实现会在向量非法时抛 `ValueError`）。
- **同文件关系**：被 `InMemoryVectorStore.upsert` 覆盖实现；自身不调用本文件其它函数。

### `delete(self, item_id: str) -> bool` （第 16 行）
- **作用**：抽象方法，声明「按条目 id 从向量索引中移除」这一能力。记忆被删除、被淘汰或过期清理时，需要同步把它的向量从索引里去掉，否则检索会命中已经不存在的内容，产生「幽灵结果」。返回布尔值是为了让调用方能区分「确实删掉了一条」和「本来就没有这条」，便于上层统计或做幂等处理。这是三个抽象方法里唯一有返回值语义的一个。抽象签名不关心底层是物理删除还是标记删除。
- **参数**：`item_id: str` —— 记忆条目的唯一标识，通常对应 `MemoryItem.id`；抽象签名未对空字符串或格式做约束，由具体实现决定是否校验。
- **返回**：声明返回 `bool` —— 具体实现约定：删除成功返回 `True`，目标不存在返回 `False`。
- **内部流程**：方法体是 `...`，无实际执行逻辑。
- **异常/边界**：抽象方法自身不抛异常；对不存在的 id 的处理方式由实现定义（同文件实现返回 `False` 而不报错）。
- **同文件关系**：被 `InMemoryVectorStore.delete` 覆盖实现；自身不调用本文件其它函数。

### `search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]` （第 20 行）
- **作用**：抽象方法，声明「用查询向量在索引里找最相似的若干条记忆」这一能力，是整个向量层最终要交付的价值所在。返回 `(条目 id, 相似度分数)` 组成的列表，而不是直接返回完整记忆对象，是为了让向量层与内容存储层解耦：向量层只负责排序和召回 id，上层再拿 id 去取真实内容。`limit` 用关键字参数（`*` 之后的参数只能按名传递）强制调用方写明数量，避免位置参数误传。`memory_type` 允许按记忆类型过滤，使「只在某一层记忆里做语义检索」成为可能。默认值 `limit=10` 提供了一个开箱可用的合理召回规模。
- **参数**：
  - `vector: list[float]` —— 查询向量，通常是用户查询文本经过同一嵌入模型得到的向量；抽象签名不约束维度，具体实现需要保证与库内向量维度一致才有意义。
  - `limit: int = 10` —— 关键字限定参数，返回结果的最大条数，默认 10；约定应为正整数。
  - `memory_type: MemoryType | str | None = None` —— 关键字限定参数，可传枚举成员、其字符串形式，或 `None`；为 `None` 时表示不做类型过滤，检索全部记忆类型。
- **返回**：声明返回 `list[tuple[str, float]]`，即 `(条目 id, 相似度)` 二元组列表，按相似度从高到低排列（具体实现约定）。
- **内部流程**：方法体是 `...`，无实际执行逻辑，仅作为接口占位。
- **异常/边界**：抽象方法自身不抛异常；对非法 `limit`、非法 `memory_type`、维度不匹配的查询向量如何处理由具体实现定义。
- **同文件关系**：被 `InMemoryVectorStore.search` 覆盖实现；其实现内部会调用本文件的 `cosine_similarity`。

### `cosine_similarity(left: list[float], right: list[float]) -> float` （第 23 行）
- **作用**：计算两个浮点向量的余弦相似度，是内存向量索引打分排序的核心数学函数。余弦相似度衡量的是两个向量方向上的夹角，取值在 `[-1, 1]` 之间（对非负向量则落在 `[0, 1]`），因此天然不受向量长度（模长）影响，非常适合文本嵌入这种「只关心语义方向、不关心绝对幅度」的场景。它被设计成模块级纯函数而不是类方法，是因为打分逻辑与存储介质无关，抽出来便于单独测试、复用和被其它实现共享。实现刻意没有引入 numpy，用纯 Python 的 `sum` 与 `zip` 完成点积和模长计算，好处是零依赖、行为可预测。函数把各种异常输入统一「软处理」为返回 `0.0`，使调用方无需在检索循环里写 try/except，保证检索过程不会因为个别脏向量整体失败。
- **参数**：
  - `left: list[float]` —— 第一个向量，通常是查询向量；元素应为有限实数（整数或浮点数皆可，会被 `float()` 转换）。
  - `right: list[float]` —— 第二个向量，通常是索引中已存储的记忆向量；元素同样应为有限实数。
- **返回**：返回 `float` 类型的余弦相似度：
  - 正常情况返回点积除以两个模长之积的结果，落在 `[-1, 1]` 区间；
  - 当 `left` 或 `right` 为空、或两者长度不相等时，返回 `0.0`；
  - 当任一向量中含有 `nan`、`inf`、`-inf` 等非有限值时，返回 `0.0`；
  - 当任一向量模长为 `0`（全零向量）时，返回 `0.0`，从而避免除零。
- **内部流程**：
  1. 首先做长度与空值校验：`if not left or not right or len(left) != len(right): return 0.0`，把空向量和维度不匹配直接判定为「不相似」。
  2. 再做有限性校验：用生成器表达式遍历 `(*left, *right)`（把两个列表解包拼成一个临时元组，一次性检查两侧），只要发现任何一个值经 `float()` 转换后不是有限值（`math.isfinite` 为假），立即返回 `0.0`。这一步能拦住 `nan`/`inf` 这类会污染整个排序结果的脏数据。
  3. 计算点积 `dot`：用 `zip(left, right)` 成对遍历，累加 `a * b`。
  4. 分别计算两侧的欧几里得模长 `norm_left`、`norm_right`：对每个元素求平方后求和，再开平方（`math.sqrt`）。
  5. 最后用条件表达式返回：当 `norm_left` 和 `norm_right` 都非零（即都非全零向量）时返回 `dot / (norm_left * norm_right)`，否则返回 `0.0` 以防除零。
- **异常/边界**：函数本身不主动抛异常，所有非法输入都被归一化为 `0.0`。需要注意的边界是：若元素不是可被 `float()` 转换的类型（例如字符串或 `None`），`float(value)` 仍会抛出 `TypeError` 或 `ValueError`，这一步的健壮性依赖上游数据干净；维度不一致按 0 分处理而不是报错；全零向量按 0 分处理。
- **同文件关系**：不调用本文件其它函数；被 `InMemoryVectorStore.search` 在列表推导中对每个候选向量逐条调用，用于生成相似度分数。

### `class InMemoryVectorStore(BaseVectorStore)` （第 34 行）
- **作用**：`BaseVectorStore` 的纯内存具体实现，用一个普通字典在进程内保存全部向量，是这个文件里唯一可直接使用的向量库。它的定位是「快速本地向量索引」，正如类文档字符串所说，既适合作为系统默认实现，也适合在单元测试里免去启动外部服务的麻烦。因为数据只存在于内存中，进程重启即丢失，所以它更适合作为缓存或测试替身，而不是长期唯一存储。它用一把 `threading.RLock` 保护字典，让并发写入与并发检索不会踩踏，这对运行在多线程 Web 服务（FastAPI 工作线程）里的场景是必要的。它实现了抽象基类要求的全部三个方法（`upsert`、`delete`、`search`），并额外提供一个 `recreate_collection` 方法用于清空索引。它把 `MemoryType` 也一起存进字典，从而支持按记忆类型过滤检索。
- **参数**：类本身无显式构造参数；实例化时执行 `__init__`，不接受任何入参。
- **返回**：返回一个可用的 `InMemoryVectorStore` 实例。
- **内部流程**：类体按顺序定义 `__init__`（初始化字典与锁）、`upsert`（带校验的写入）、`recreate_collection`（清空）、`delete`（弹出并返回是否命中）、`search`（打分排序截断）。所有对 `self._vectors` 的读写都在 `with self._lock:` 保护下进行。
- **异常/边界**：继承自 ABC，只有把三个抽象方法全部实现后才能实例化；实例本身不做维度一致性校验，不同维度的向量可以共存于同一实例中（此时检索会因长度不等而得到 0 分）。
- **同文件关系**：继承 `BaseVectorStore`；`__init__` 建立状态，`upsert`/`delete`/`search` 覆盖抽象方法，`search` 内部调用模块级 `cosine_similarity`；`recreate_collection` 是本类独有、不在基类接口中的扩展方法。

### `__init__(self) -> None` （第 37 行）
- **作用**：构造内存向量库实例，负责准备两样运行时必需品：存放向量的字典和保护它的锁。字典 `self._vectors` 的键是记忆条目 id，值是 `(向量副本, 记忆类型)` 二元组；之所以同时存类型，是为了让 `search` 能在不访问外部存储的情况下直接按 `memory_type` 过滤。锁使用 `threading.RLock`（可重入锁）而不是普通 `Lock`，是因为可重入锁允许同一线程重复获取而不会自锁，给未来可能出现的「方法内部再调用另一个加锁方法」留出安全余量。该方法是零参数构造，不需要配置维度、路径或连接串，体现了内存实现的即开即用特性。
- **参数**：无（仅隐式 `self`）。
- **返回**：`None`；构造完成后实例状态就绪。
- **内部流程**：
  1. `self._vectors: dict[str, tuple[list[float], MemoryType]] = {}` —— 初始化空字典，类型标注明确了键值结构。
  2. `self._lock = threading.RLock()` —— 创建可重入锁实例。
- **异常/边界**：无特殊处理；构造过程不会失败（不涉及 IO 或网络），也不会做任何维度校验或容量限制。
- **同文件关系**：被本类的 `upsert`、`recreate_collection`、`delete`、`search` 使用，它们都通过 `self._lock` 与 `self._vectors` 读写状态；不调用本文件其它函数。

### `upsert(self, item: MemoryItem) -> None` （第 41 行）
- **作用**：把一条记忆条目写入内存索引，是 `BaseVectorStore.upsert` 的具体实现。它的关键设计是「没有向量就静默跳过」：如果 `item.embedding` 为 `None`，函数直接返回、不写入任何东西，也不会报错，因为在一个混合了「已嵌入」和「未嵌入」条目的系统里，未嵌入条目本就不该出现在向量索引中，静默跳过比抛异常更符合预期。反过来，如果条目确实带了向量但向量是空列表或含有 `nan`/`inf`，则视为数据错误，抛出 `ValueError`，避免脏数据污染后续所有检索打分。写入时用 `list(item.embedding)` 做了一次浅拷贝，防止外部持有并修改原列表导致索引内容被意外篡改。整个过程在锁内完成，保证并发安全。
- **参数**：`item: MemoryItem` —— 待写入的记忆条目，函数会读取两个字段：`item.embedding`（可选，`None` 或浮点列表）和 `item.memory_type`（记忆类型枚举，用于过滤检索）；另外使用 `item.id` 作为字典键。
- **返回**：始终返回 `None`；无返回值表达写入结果（写入失败通过异常体现）。
- **内部流程**：
  1. 判断 `if item.embedding is not None:` —— 只有带向量的条目才继续，未嵌入条目直接跳过。
  2. 在分支内做合法性校验：`if not item.embedding or any(not math.isfinite(float(value)) for value in item.embedding)` —— 空向量或含非有限值即 `raise ValueError("item embedding must contain finite values")`。
  3. `with self._lock:` 获取可重入锁，保证写入临界区互斥。
  4. `self._vectors[item.id] = (list(item.embedding), item.memory_type)` —— 以 id 为键覆盖写入（同 id 即更新），值为「向量浅拷贝 + 记忆类型」元组。
- **异常/边界**：当 `item.embedding` 是空列表（`[]`）或含 `nan`/`inf`/`-inf` 时抛 `ValueError`；`item.embedding` 为 `None` 时不写入、不报错；不校验不同条目之间的向量维度是否一致，也不限制向量条数（内存无上限）。
- **同文件关系**：覆盖 `BaseVectorStore.upsert`；依赖 `__init__` 建立的 `self._lock` 与 `self._vectors`；不调用本文件其它函数（`math.isfinite` 是标准库调用）。

### `recreate_collection(self, dimension: int) -> None` （第 48 行）
- **作用**：清空内存中已保存的全部向量，语义上等价于「重建向量集合/向量空间」。当嵌入模型被更换、或嵌入维度发生变化、或系统确认需要重新计算全部嵌入时，旧向量与新查询向量已经不在同一个向量空间里，继续保留只会产生无意义的相似度，因此需要先整体丢弃再重新灌入。方法文档字符串明确写出它服务于「a confirmed embedding-space rebuild can reindex」（确认过的嵌入空间重建后重新索引）。它虽然接收 `dimension` 参数，但在内存实现里维度并不需要真正使用——字典结构不依赖维度——保留该参数是为了与远程向量库（例如需要按维度创建集合的实现）保持一致的调用签名。这是本类独有的扩展方法，抽象基类并未要求。
- **参数**：`dimension: int` —— 目标向量维度，语义上是新嵌入空间的维度。约束为「正整数且必须是真正的整数」：不能是布尔值（`bool` 是 `int` 的子类，会被显式排除）、不能是 `float` 等其它类型、不能小于 1。
- **返回**：返回 `None`；清空后索引为空字典。
- **内部流程**：
  1. 参数校验：`if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1: raise ValueError("dimension must be a positive integer")` —— 三段判断依次排除布尔值、非整数类型、非正数。
  2. `with self._lock:` 进入临界区。
  3. `self._vectors.clear()` —— 就地清空字典，保留字典对象本身与锁对象（因此其它方法持有的 `self` 引用仍然有效）。
- **异常/边界**：`dimension` 为 `True`/`False`、浮点数、字符串或小于 1 的整数时抛 `ValueError("dimension must be a positive integer")`；清空是幂等的，索引本来就空时调用也不会出错；不返回被丢弃的数据条数。
- **同文件关系**：不覆盖任何抽象方法（基类没有该方法）；依赖 `__init__` 建立的 `self._vectors` 与 `self._lock`；不调用本文件其它函数；清空后由 `upsert` 重新填充，由 `search`/`delete` 使用新状态。

### `delete(self, item_id: str) -> bool` （第 56 行）
- **作用**：按 id 从内存索引中删除一条向量，是 `BaseVectorStore.delete` 的具体实现。它用 `dict.pop(key, None)` 的惯用写法实现「删除并判断是否存在」：pop 在键存在时返回被删除的值，不存在时返回默认值 `None`，于是 `is not None` 就精确回答了「到底删掉了没有」。这个布尔结果让上层可以做幂等删除——重复删除同一条记忆不会报错，只是第二次返回 `False`。整个操作在锁内执行，避免与 `upsert` 的写入或 `search` 的遍历并发冲突。函数只删向量索引里的记录，不负责删除记忆正文。
- **参数**：`item_id: str` —— 要删除的记忆条目 id，对应 `upsert` 时使用的 `item.id`。不校验是否为 `None` 或空字符串；对不存在的键（包括从未写入的 id）走「返回 False」路径。
- **返回**：返回 `bool` —— 该 id 此前存在于索引中并已被移除时返回 `True`；该 id 不存在（从未写入或已被删除）时返回 `False`。
- **内部流程**：
  1. `with self._lock:` 获取锁。
  2. `self._vectors.pop(item_id, None)` —— 从字典中弹出该键对应的 `(向量, 记忆类型)` 元组，缺失时得到 `None`。
  3. `... is not None` —— 把弹出结果是否为 `None` 转成布尔值并作为返回值。
- **异常/边界**：无特殊处理，不会因为 id 不存在而抛 `KeyError`（因为传了默认值）；若传入不可哈希的 id（例如列表）则 `pop` 会抛 `TypeError`，但按类型标注应为 `str`；不校验空字符串，空字符串会被当作普通键处理。
- **同文件关系**：覆盖 `BaseVectorStore.delete`；依赖 `__init__` 建立的 `self._lock` 与 `self._vectors`；不调用本文件其它函数。

### `search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]` （第 60 行）
- **作用**：内存索引的语义检索实现，也是这个文件最核心的功能方法。它拿到一个查询向量后，遍历索引中（可按类型过滤后的）全部候选，用 `cosine_similarity` 逐个打分，再按分数降序排列，最后截取前 `limit` 条返回 `(条目 id, 分数)` 列表。之所以先在全量候选上算分再排序，是因为内存实现的定位就是「小规模、快速、无索引结构」的暴力检索（brute force），实现简单且结果精确，没有近似算法的召回损失。它支持按 `memory_type` 过滤，使得四层记忆系统可以只在某一层内做语义检索。`limit` 和 `memory_type` 都设计为关键字限定参数（`*` 之后），强制调用点写明语义，防止位置参数传错。返回 id 而非完整对象，是为了与存储层解耦，由上层按 id 取内容。排序键 `(-row[1], row[0])` 还额外保证了同分时按 id 字典序稳定排列，使输出可复现。
- **参数**：
  - `vector: list[float]` —— 查询向量，与索引中向量的维度一致时才有意义；维度不一致时 `cosine_similarity` 会返回 `0.0`，该候选会被排到末尾而不是报错。
  - `limit: int = 10` —— 关键字限定参数，返回条数上限。约束为正整数：不能是布尔值、不能是非 `int` 类型、不能小于 1，否则抛 `ValueError`。
  - `memory_type: MemoryType | str | None = None` —— 关键字限定参数，类型过滤条件。为 `None` 时不过滤；为 `MemoryType` 枚举成员时直接使用；为字符串时会被 `MemoryType(memory_type)` 转换为枚举（非法字符串会转换失败）。
- **返回**：返回 `list[tuple[str, float]]` —— 每个元素是 `(item_id, similarity)`；列表已按相似度从高到低排序（同分按 id 升序），长度不超过 `limit`；索引为空或无候选通过过滤时返回空列表 `[]`。
- **内部流程**：
  1. 参数校验 `limit`：`if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1: raise ValueError("limit must be a positive integer")`。
  2. 归一化类型过滤条件：`wanted = MemoryType(memory_type) if memory_type is not None else None`，把字符串统一成枚举，`None` 保持 `None`。
  3. `with self._lock:` 进入临界区，在锁内构建候选列表 `scores`（列表推导）：遍历 `self._vectors.items()`，解包出 `(stored, stored_type)`，当 `wanted is None or stored_type == wanted` 时，调用 `cosine_similarity(vector, stored)` 计算分数，生成三元组 `(item_id, score, stored_type)`。把打分放在锁内是为了避免遍历期间字典被并发修改。
  4. 退出锁后排序：`scores.sort(key=lambda row: (-row[1], row[0]))` —— 主键取分数取负实现降序，次键取 id 实现同分稳定升序。
  5. 切片并投影：`scores[:limit]` 截取前 limit 条，列表推导只保留 `(item_id, score)` 两元组，丢弃内部的 `stored_type`，作为最终返回值。
- **异常/边界**：`limit` 为 `True`/`False`、浮点数、字符串或小于 1 时抛 `ValueError`；`memory_type` 传入无法被 `MemoryType` 识别的字符串时，`MemoryType(memory_type)` 会抛 `ValueError`（由枚举构造抛出）；查询向量为空、含非有限值或与某条存储向量维度不一致时，对应候选分数为 `0.0` 而不会报错；索引为空时返回 `[]`；`limit` 大于候选总数时返回全部候选；本方法不修改索引状态（只读）。
- **同文件关系**：覆盖 `BaseVectorStore.search`；内部调用模块级 `cosine_similarity` 打分；依赖 `__init__` 建立的 `self._vectors` 与 `self._lock`；其结果与 `upsert` 写入、`delete` 删除、`recreate_collection` 清空后的索引状态直接相关。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `BaseVectorStore` | 向量存储的抽象基类，规定 upsert / delete / search 三个必须实现的接口。 |
| `BaseVectorStore.upsert` | 抽象声明「写入或覆盖更新一条记忆向量」的能力。 |
| `BaseVectorStore.delete` | 抽象声明「按 id 删除向量并返回是否命中」的能力。 |
| `BaseVectorStore.search` | 抽象声明「按查询向量做近邻检索、可按类型过滤并限制条数」的能力。 |
| `cosine_similarity` | 纯 Python 计算两个等长向量的余弦相似度，非法或退化输入一律返回 0.0。 |
| `InMemoryVectorStore` | 基于字典加可重入锁的进程内向量索引，是默认实现与测试用实现。 |
| `InMemoryVectorStore.__init__` | 初始化空向量字典与 `threading.RLock`。 |
| `InMemoryVectorStore.upsert` | 校验向量有限性后按 id 覆盖写入向量与记忆类型，无向量则静默跳过。 |
| `InMemoryVectorStore.recreate_collection` | 校验维度为正整数后清空全部内存向量，用于嵌入空间重建。 |
| `InMemoryVectorStore.delete` | 用 `dict.pop` 删除指定 id 并返回此前是否存在。 |
| `InMemoryVectorStore.search` | 过滤候选、逐条算余弦相似度、按分数降序取前 limit 条返回 (id, 分数)。 |
