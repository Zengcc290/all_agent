# memory/storage/qdrant.py

## 一、这个文件是干什么的

这个文件是记忆系统里「向量投影层」的 Qdrant 适配器，整个文件只服务于一件事：把 Agent 记忆条目的 embedding 写进 Qdrant 向量库，并在需要时按向量相似度把命中的记忆 id 取回来。它在项目里的定位是 BaseVectorStore 的一个具体实现，`QdrantVectorStore` 继承自同包 `.vector` 里的 `BaseVectorStore`，因此对上层（MemoryManager、重索引脚本、reconcile 接口）来说它只是「一个能 upsert / search / delete / list_ids 的向量库」。

文件顶部先放了四个模块级私有小工具函数：`_vector_size_of`、`_collection_vector_size`、`_is_dimension_error`、`_dimension_mismatch_message`，它们全部围绕「向量维度」这一个主题——因为 Qdrant 的集合维度一旦建好就不能改，而 embedding 模型却可能被换掉，维度不一致是这个适配器最容易踩的坑，所以这里专门用几个函数去读集合真实维度、识别维度类报错、并生成可操作的中文修复指引。

主体类 `QdrantVectorStore` 的设计要点有三个：第一，集合是「懒创建」的，第一次写入时才知道维度并顺带建集合（`_ensure_collection`），这样嵌入维度是从真实数据里发现的，而不是只依赖配置；第二，它写入的 payload 是「轻量投影」，只保留 id、memory_type、created_at、expires_at 这类回查和过滤必需的字段，正文和 metadata 仍然以 SQLite 为真值源（`_light_payload`），避免在 Qdrant 里再存一份会漂移的真相；第三，所有点都必须带 `namespace`，检索时强制按 `namespace` 过滤，所以这个键缺失的点等于永远检索不到。

另外它还处理了两个运行环境的现实问题：一是本机回环地址绝不允许走系统代理（用户开 Clash 之类代理时 127.0.0.1 请求会被转发导致 502），所以 `__init__` 里对 localhost / 127.0.0.1 / ::1 显式 `trust_env=False`；二是 Qdrant Cloud 的 HTTP API 会对没有 payload index 的字段做过滤查询直接返回 400，所以 `_ensure_payload_indexes` 会幂等地给 `namespace` 和 `memory_type` 建 keyword 索引，且建索引失败不致命（索引是检索需求，不是写入需求）。

还有一个「点 id」的细节：Qdrant 只接受 UUID 或整数作为点 id，而应用层的记忆 id 可能是任意字符串，所以 `_point_id` 用 `uuid5` 把任意字符串确定性地映射成 UUID，同时把原始 id 放在 payload 里，回查时优先从 payload 取回应用层 id。

文件末尾的 `__all__ = ["QdrantVectorStore"]` 只导出这个类，四个模块级工具函数和所有下划线方法都是实现细节，不对外承诺。

## 二、函数与类逐条详解

### `_vector_size_of(value: Any) -> int | None` （第 14 行）

- **作用**：从各种「形态不一」的输入里把向量维度（size）抠出来。Qdrant 的 `VectorParams` 有 `.size` 属性，而新版客户端返回的可能是命名向量字典 `{"text": VectorParams(...)}`，测试替身又可能直接给一个 int 或者 `{"size": 1024}` 这样的普通 dict，这个函数就是用来统一消化这些差异的。它是整个维度探测链路的最后一环，`_collection_vector_size` 拿到 `config.params.vectors` 之后就是靠它把最终数字读出来。因为它会被 `_collection_vector_size` 在 dict 分支里递归调用，所以它也承担了「字典里再套字典」这种嵌套结构的解包工作。
- **参数**：
  - `value`：任意类型，`Any`，无默认值。可能是 `None`、`bool`、`int`、`VectorParams` 之类的对象（只要有 `.size` 属性）、普通 `dict`（可能含 `"size"` 键，也可能是「名字 → 尺寸描述」的映射）。
- **返回**：返回 `int | None`。传入 `None` 或 `bool` 返回 `None`；传入 `int` 直接原样返回（注意这里不校验正负，负数也会原样返回）；对象带非 `None` 的 `.size` 时返回 `int(size)`；dict 里有非 `None` 的 `"size"` 键时返回 `int(value["size"])`；dict 非空但没有 `"size"` 键时递归处理 `next(iter(value.values()))` 即第一个值；其余情况（空 dict、既无 `.size` 也非 dict）返回 `None`。
- **内部流程**：第一步先做「毒值」过滤，`value is None or isinstance(value, bool)` 直接返回 `None`——单独判 bool 是因为 Python 里 `True` 是 `int` 的子类，如果不先拦掉，`True` 会被当成尺寸 `1`。第二步 `isinstance(value, int)` 直接返回，覆盖「替身直接给数字」的情况。第三步 `getattr(value, "size", None)`，拿到非 `None` 就 `int(size)` 返回，这是 `VectorParams` 的正常路径。第四步如果 `value` 是 dict：先看 `"size"` 键且其值非 `None` 就 `int` 化返回；否则如果 dict 非空，就对 `next(iter(value.values()))` 递归调用自身，用于解命名向量字典。最后兜底 `return None`。
- **异常/边界**：函数本身不主动抛异常。但 `int(size)` 在 size 是无法转成整数的字符串（如 `"big"`）时会抛 `ValueError`，代码没有捕获。`value` 是 dict 但某个值是 `None` 时，递归进去第一行就返回 `None`，不会崩。dict 为空时 `if value:` 为假，直接走到末尾返回 `None`，不会对空 dict 调 `next(iter(...))`。
- **同文件关系**：被 `_collection_vector_size` 调用（在 dict 分支和属性分支各调一次），并且在自身 dict 分支里递归调用自己。不调用本文件其它函数。

### `_collection_vector_size(info: Any) -> int | None` （第 32 行）

- **作用**：从 Qdrant 的「集合信息」对象里读出这个集合当前真实的向量维度。之所以需要它，是因为「配置里写的维度」和「集合里已有的维度」是两件事：`[embedding].dimension` 只是配置，集合建好之后维度就固定了，判断是否冲突必须看集合的真实尺寸。真实路径是 `info.config.params.vectors`（可能是 `VectorParams` 也可能是命名向量字典），但测试替身可能把尺寸放在扁平的 `info.config.params.size` 上，所以这里同时兼容两条路径。它只负责「读」，不负责判断冲突，判断交给 `_ensure_collection`。
- **参数**：
  - `info`：`Any`，无默认值。期望是 `client.get_collection()` 的返回值，可能是对象（带 `.config` / `.config.params`）也可能是普通 `dict`（形如 `{"config": {"params": {...}}}`）。
- **返回**：返回 `int | None`。`info` 为 `None` 或结构不匹配、`params` 取不到、`vectors`/`size` 都读不出数字时返回 `None`；能读出时返回维度整数。注意 dict 分支里用的是 `or`，所以读出的维度为 `0`（假值）时会被继续尝试 `size` 分支。
- **内部流程**：第一步，如果 `info` 是 dict，用 `type("Info", (), info)()` 动态造一个临时类并实例化，把 dict 的键变成对象属性，这样后面就可以统一用属性访问。第二步取 `params`：先按对象路径 `getattr(getattr(info, "config", None), "params", None)` 取；紧接着如果 `info.config` 本身是 dict（说明是「对象外壳里套 dict」的混合形态），改用 `info.config.get("params")` 覆盖。第三步 `params is None` 就直接返回 `None`。第四步分两条：`params` 是 dict 时返回 `_vector_size_of(params.get("vectors")) or _vector_size_of(params.get("size"))`；`params` 是对象时返回 `_vector_size_of(getattr(params, "vectors", None)) or _vector_size_of(getattr(params, "size", None))`。
- **异常/边界**：不主动抛异常。`type("Info", (), info)()` 要求 `info` 的键都是合法标识符字符串，否则理论上会出问题，但真实 Qdrant 返回的字段名都是合法标识符。dict 值无法被 `getattr` 取出时返回 `None`，由调用方按「读不到尺寸」降级处理。文件里的注释特别强调：不要拿配置文件里的 `[embedding].dimension` 当集合维度来比对。
- **同文件关系**：调用 `_vector_size_of`（两次，且由于 `or` 的短路语义可能只调一次）；被 `QdrantVectorStore._live_collection_size` 调用。

### `_is_dimension_error(exc: BaseException) -> bool` （第 54 行）

- **作用**：判断一个异常到底是不是「向量维度不匹配」类错误。这个判断的意义在于：写入失败时，如果是维度问题，就不应该原样抛出底层客户端的原始异常，而应该改抛一个带修复指引的 `ValueError`，告诉使用者「要么换回原模型，要么重建集合」；如果是别的错误（网络、权限、集合不存在等），就应该原样上抛，不要用维度信息掩盖真实故障。它靠字符串匹配实现，因为 qdrant-client 在本地模式、HTTP 模式、不同版本下抛出的异常类型不一致，只有错误文本是相对稳定的。
- **参数**：
  - `exc`：`BaseException`，无默认值。任意异常实例；只要能被 `str()` 转换即可。
- **返回**：返回 `bool`。异常文本（转小写后）包含 `"expected dim"`、`"vector dimension error"` 或 `"dimension mismatch"` 三者中任意一个时返回 `True`，否则 `False`。
- **内部流程**：`str(exc).casefold()` 先把异常转成字符串并做大小写无关化（`casefold` 比 `lower` 更彻底），然后在 `return` 语句里用三个 `in` 判断做逻辑或。没有缓存、没有正则、没有副作用，是一个纯函数。
- **异常/边界**：`str(exc)` 对几乎所有异常都安全；若某个异常自定义了会抛错的 `__str__`，这里会把这个新异常抛出，代码未做保护。传入 `None` 时 `str(None)` 得到 `"none"`，返回 `False`，不会崩。
- **同文件关系**：被 `_ensure_collection`（捕获写入/建集合异常后判断是否转成维度冲突）、`upsert`、`upsert_chunk` 调用。不调用本文件其它函数。

### `_dimension_mismatch_message(collection: str, existing: int, actual: int) -> str` （第 59 行）

- **作用**：生成维度冲突时给用户看的中文修复指引文本。它存在的价值是把「底层一个冷冰冰的维度错误」翻译成「你现在该做哪两件事」，因为维度冲突无法靠改单个配置项解决：集合维度是既成事实，只能换回原 embedding 模型，或者确认后重建向量投影。消息里还顺带给出了命令行等价操作，方便运维不用去翻代码就知道怎么修。它是一个纯字符串拼装函数，不涉及任何 IO。
- **参数**：
  - `collection`：`str`，无默认值。集合名，会以 `!r`（带引号）形式嵌入消息，便于辨认。
  - `existing`：`int`，无默认值。集合当前已有的维度。调用方通常会把 `None` 归一成 `0` 再传进来。
  - `actual`：`int`，无默认值。当前 embedding 模型实际输出的维度。
- **返回**：返回 `str`，一段完整的中文说明，内容包含：集合名与两个维度的对比、维度不能单独改集合或配置的结论、两条可选修复路径（把 `config/services.toml` 的 `[embedding].model` 换回原模型；或确认重建向量投影，并说明 SQLite 是真值可全量重灌）、以及两个命令行等价命令 `python scripts/migrate_to_cloud.py --recreate-collection` 与 `python scripts/reindex_embeddings.py`。
- **内部流程**：没有分支和循环，直接用 f-string 拼接多行相邻字符串字面量返回。`{collection!r}` 走 `repr`，`{existing}`、`{actual}` 走 `str`。
- **异常/边界**：无特殊处理。传入 `existing=0` 时消息会显示「是 0 维」，这正是调用方在「读不到已有维度」时的兜底表现。
- **同文件关系**：只被 `QdrantVectorStore._raise_dimension_mismatch` 调用，是那条抛错路径的文案来源。不调用本文件其它函数。

### `class QdrantVectorStore(BaseVectorStore)` （第 73 行）

- **作用**：这是本文件唯一的类，也是唯一的对外导出物，扮演「记忆系统的 Qdrant 向量库实现」这一角色。它继承 `BaseVectorStore`，因此必须提供 `upsert`、`search`、`delete`、`list_ids` 这套接口，供上层 MemoryManager、reconcile 接口、重索引脚本调用。类的文档字符串点明了两个关键设计：`client` 可以被注入（传一个 `qdrant_client.QdrantClient` 兼容对象）以便测试；集合是懒创建的，第一次 upsert 时才根据条目的 embedding 长度建集合，从而让嵌入维度从真实数据里被发现。除了标准接口，它还额外提供 `collection_dimension`（读集合真实维度）、`recreate_collection`（确认后重建投影）、`upsert_chunk`（写 chunk 级投影）这些给运维和重索引用的方法。
- **参数**：类本身不接收参数；构造参数见 `__init__`。继承自 `BaseVectorStore`，因此也继承了基类提供的能力（本文件未覆写的部分由基类负责）。
- **返回**：类，实例化后得到向量库对象。
- **内部流程**：类的运行依赖几个内部状态：`self.client`（Qdrant 客户端或测试替身）、`self.collection_name`（集合名）、`self.dimension`（已知维度，可能来自构造参数，也可能由写入路径发现）、`self.namespace`（命名空间，所有点和检索都必须带）、`self._ready`（是否已经确认过集合存在且维度核对通过，只由写入路径置位）。整体调用链是：写入走 `upsert` / `upsert_chunk` → `_ensure_collection`（必要时建集合、建索引、核对维度）→ `_point_id` 生成点 id → `client.upsert`；读取走 `search` / `list_ids` / `delete` → `_ensure_ready_for_read`（只挂载不创建）→ `client.search` / `client.scroll` / `client.delete`。
- **异常/边界**：构造期会抛 `ValueError`（namespace 为空、collection_name 为空、dimension 非法）和 `RuntimeError`（缺少 qdrant-client 依赖）。写入期维度冲突抛带指引的 `ValueError`，其它初始化失败抛 `RuntimeError`。读取期对异常一律降级为空结果。
- **同文件关系**：类内部方法互相调用，详见各方法条目。类依赖模块级函数 `_collection_vector_size`、`_is_dimension_error`、`_dimension_mismatch_message`。

### `__init__(self, url: str | None = None, collection_name: str = MEMORY_QDRANT_COLLECTION, *, api_key: str | None = None, client: Any = None, dimension: int | None = None, namespace: str = "memory", proxy_url: str | None = None) -> None` （第 81 行）

- **作用**：构造向量库实例，并在这里完成「客户端从哪来」和「参数合法性」这两件最容易出错的事。如果调用方直接注入了 `client`（测试替身或已建好的客户端），就完全跳过连接逻辑；否则按 `url` 的情况创建真实的 `QdrantClient`。它特别处理了代理问题：本机回环地址（127.0.0.1 / localhost / ::1）绝不走系统代理，因为用户开着 Clash 之类的代理时 httpx 会信任环境变量或注册表里的代理设置，把回环请求转发到代理端口，而代理对回环目标会返回 502；云端端点则在给了 `proxy_url` 时显式走本地转发代理，没给时也关掉 `trust_env`，避免畸形 `NO_PROXY` 之类的环境变量干扰。构造完还会做参数校验并把状态初始化成「尚未就绪」。
- **参数**：
  - `url`：`str | None`，默认 `None`。Qdrant 服务地址。为 `None` 或空串时创建内存实例 `QdrantClient(path=":memory:")`（本地嵌入式模式，适合测试）；非空时按其 hostname 决定是否走代理。
  - `collection_name`：`str`，默认 `MEMORY_QDRANT_COLLECTION`（从 `constants` 导入的模块级常量）。集合名，必须是非空且非纯空白的字符串。
  - `api_key`：`str | None`，仅关键字参数，默认 `None`。云端鉴权密钥，透传给 `QdrantClient`；本地内存模式不使用。
  - `client`：`Any`，仅关键字参数，默认 `None`。可注入的客户端/替身；非 `None` 时完全跳过 `qdrant_client` 的导入和创建，因此测试环境即使没装 qdrant-client 也能跑。
  - `dimension`：`int | None`，仅关键字参数，默认 `None`。已知的 embedding 维度护栏。给了就校验必须是正整数（`bool` 被显式排除，`< 1` 被拒）；注意它只是「护栏」，不会被当成集合已有维度。
  - `namespace`：`str`，仅关键字参数，默认 `"memory"`。命名空间，必须是非空且非纯空白字符串；它会写进每个点的 payload 并作为检索的强制过滤条件。
  - `proxy_url`：`str | None`，仅关键字参数，默认 `None`。云端端点要走的本地转发代理地址，透传给 `QdrantClient(proxy=...)`，最终透传到 httpx。
- **返回**：`None`。副作用是设置 `self.client`、`self.collection_name`、`self.dimension`、`self.namespace`，并把 `self._ready` 置为 `False`。
- **内部流程**：第一步校验 `namespace`：不是 `str` 或 `strip()` 后为空就抛 `ValueError`。第二步，若 `client is None`，先 `from qdrant_client import QdrantClient`，`ImportError` 时包装成 `RuntimeError("QdrantVectorStore requires qdrant-client")` 并保留 `from exc` 链。第三步分三种情况建客户端：`url` 为真时用 `urlparse` 取 hostname，命中 `("127.0.0.1", "localhost", "::1")` 就 `QdrantClient(url=url, api_key=api_key, trust_env=False)`；否则若 `proxy_url` 为真就 `QdrantClient(url=url, api_key=api_key, proxy=proxy_url)`；否则 `QdrantClient(url=url, api_key=api_key, trust_env=False)`。`url` 为假时用 `QdrantClient(path=":memory:")`。第四步校验 `collection_name` 非空。第五步校验 `dimension`：显式排除 `bool`、要求 `int` 且 `>= 1`。第六步一次性赋值四个属性并把 `_ready` 置 `False`。
- **异常/边界**：`ValueError`——`namespace` 非字符串或空白、`collection_name` 非字符串或空白、`dimension` 是 bool / 非 int / 小于 1。`RuntimeError`——未安装 qdrant-client。`url` 是空串按「无 url」处理走内存模式。`api_key` 在内存模式下被忽略。若 `urlparse` 得到的 hostname 是 `None`（畸形 URL），不会命中回环分支，会落到 `proxy_url` 或 `trust_env=False` 分支。
- **同文件关系**：不调用本文件其它函数或方法；被外部调用方实例化。它设置的 `self._ready` 被 `_ensure_collection`、`_ensure_ready_for_read` 读取。

### `_live_collection_size(self) -> int | None` （第 115 行）

- **作用**：向 Qdrant 问一次「这个集合现在到底是多少维」。它是类内部所有维度判断的统一入口，把「客户端有没有 `get_collection` 方法」和「读失败怎么办」这两件事包起来，让上层（`collection_dimension`、`_ensure_collection`、以及写入失败后的错误信息）不用各自写 try/except。它刻意吞掉所有异常，因为读不到尺寸并不致命——真正的写入路径还会再核对一次，读不到只意味着「暂时不知道」，不应该让探测失败变成硬错误。
- **参数**：无（除 `self`）。
- **返回**：`int | None`。客户端没有可调用的 `get_collection` 属性时返回 `None`；`get_collection` 抛任何异常时返回 `None`；成功时返回 `_collection_vector_size` 解析出的维度（可能仍是 `None`，表示解析不出尺寸）。
- **内部流程**：用 `getattr(self.client, "get_collection", None)` 取方法并 `callable` 检查，不可调用就直接返回 `None`（测试替身可能只实现 upsert/search）。然后 `try` 里调用 `get_collection(collection_name=self.collection_name)` 并把结果交给 `_collection_vector_size`；`except Exception` 捕获全部异常返回 `None`，注释说明「读不到尺寸时由写入路径再核对」。
- **异常/边界**：不向外抛异常，所有异常都被吞掉（`# noqa: BLE001`）。这种设计是有意的降级：探测失败等同于「不知道维度」，后续逻辑会退回到保守路径。
- **同文件关系**：调用模块级函数 `_collection_vector_size`；被 `collection_dimension`、`_ensure_collection` 调用，也被 `upsert`、`upsert_chunk` 在维度报错时用来取「已有维度」喂给 `_raise_dimension_mismatch`。

### `collection_dimension(self) -> int | None` （第 124 行）

- **作用**：对外的「查集合现有维度」接口，供运维脚本、健康检查或 reconcile 流程判断当前投影和配置是否一致。它和 `_live_collection_size` 的区别是：它先确认集合是否存在，集合不存在时明确返回 `None`（表示「还没有投影」），而不是像内部方法那样只是「读不到」。探测 `collection_exists` 失败时也返回 `None`，并且刻意不把这种情况当成「已有投影」，以免在集合真的存在却暂时探测不到时误判。
- **参数**：无（除 `self`）。
- **返回**：`int | None`。`collection_exists` 抛异常返回 `None`；集合不存在返回 `None`；集合存在时返回 `_live_collection_size()` 的结果（可能因解析不出尺寸而为 `None`）。
- **内部流程**：先 `try` 调用 `self.client.collection_exists(collection_name=self.collection_name)`，`except Exception` 直接 `return None`（注释：探测失败时不当成已有投影）。然后 `if not exists: return None`。最后 `return self._live_collection_size()`。注意它不会去调用 `_ensure_ready_for_read`，也不会改动 `self._ready`。
- **异常/边界**：不向外抛异常，全部降级为 `None`。集合存在但尺寸读不出时返回 `None`，调用方需要把 `None` 理解为「未知」而不是「0 维」。
- **同文件关系**：调用 `_live_collection_size`。本文件内部没有任何地方调用它，它是给外部调用方用的只读查询接口。

### `_raise_dimension_mismatch(self, existing: int | None, actual: int, cause: BaseException | None = None) -> None` （第 135 行）

- **作用**：把「维度冲突」这件事统一抛成一个带修复指引的 `ValueError`。它是所有维度冲突路径的收口点，好处是错误文案只有一处来源，且能根据是否携带原始异常决定要不要保留异常链。之所以需要 `cause` 参数，是因为有的冲突是「我们主动比对出来的」（没有底层异常，`cause=None`），有的是「客户端抛了维度错我们翻译过来的」（有底层异常，需要 `from cause` 保留调用栈线索）。
- **参数**：
  - `existing`：`int | None`，无默认值。集合已有维度；为 `None` 时按 `0` 处理（`int(existing or 0)`），用于「读不到已有维度」的兜底。
  - `actual`：`int`，无默认值。当前实际的 embedding 维度。
  - `cause`：`BaseException | None`，默认 `None`。原始异常；为 `None` 时不设置异常链，非 `None` 时用 `raise ... from cause` 保留因果链。
- **返回**：不返回（`None`），因为它总会抛异常。
- **内部流程**：先调用 `_dimension_mismatch_message(self.collection_name, int(existing or 0), actual)` 拿到文案，然后分两支：`cause is None` 时 `raise ValueError(message)`；否则 `raise ValueError(message) from cause`。
- **异常/边界**：一定抛 `ValueError`，消息是 `_dimension_mismatch_message` 的产物。`existing` 传 `None` 或 `0` 时消息里显示「是 0 维」。
- **同文件关系**：调用模块级函数 `_dimension_mismatch_message`；被 `_ensure_collection`（主动比对不一致、以及捕获到维度类异常时）、`upsert`、`upsert_chunk` 调用。

### `_ensure_collection(self, dimension: int) -> None` （第 141 行）

- **作用**：写入路径上的「集合就绪」保障：确保集合存在、维度一致、payload 索引已建，并刷新 `self.dimension` 与 `self._ready`。它实现了懒创建——集合不存在就用给定维度按余弦距离建集合；已存在就只跟集合的**真实**尺寸比对，不一致就抛维度冲突。这里刻意不拿 `self.dimension` 当「集合已有维度」，因为 `self.dimension` 可能来自构造时的配置护栏，换模型后拿它比对会误报 1024/4096 这类冲突。它还有一个快速路径：如果已经 `_ready` 且读到的尺寸要么读不到、要么正好等于目标维度，就直接更新 `self.dimension` 返回，避免每次写入都打一次建集合/建索引的调用。
- **参数**：
  - `dimension`：`int`，无默认值。本次写入使用的向量维度，由调用方从 `len(item.embedding)` 或 `len(vector)` 得到。本方法不做正整数校验，假定调用方已经保证非零。
- **返回**：`None`。副作用是可能创建集合、创建 payload 索引、更新 `self.dimension`、把 `self._ready` 置为 `True`。
- **内部流程**：第一步，`existing = self._live_collection_size() if self._ready else None`——只有在已经就绪过的情况下才先去探一次真实尺寸。第二步快速路径：`self._ready and (existing is None or existing == dimension)` 成立时更新 `self.dimension = dimension` 并返回。第三步进入 `try`：从 `qdrant_client.models` 导入 `Distance, VectorParams`；调 `collection_exists`；不存在就 `create_collection(collection_name=..., vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))`；存在则先补探 `existing`（若为 `None` 再调一次 `_live_collection_size`），若 `existing is not None and existing != dimension` 就调 `_raise_dimension_mismatch(existing, dimension)`。第四步无论新建还是复用，都设置 `self.dimension = dimension`、调 `_ensure_payload_indexes()`、把 `self._ready = True`。第五步异常处理：`except ValueError: raise` 让维度冲突原样透出；`except Exception as exc` 先用 `_is_dimension_error(exc)` 判断，是维度错误就调 `_raise_dimension_mismatch(self._live_collection_size(), dimension, exc)` 翻译成指引型 `ValueError`，否则抛 `RuntimeError(f"unable to initialize Qdrant collection: {exc}")` 并保留 `from exc`。
- **异常/边界**：会抛 `ValueError`（维度冲突，来自 `_raise_dimension_mismatch`）和 `RuntimeError`（其它初始化失败，如网络不通、权限不足）。`existing` 读不到（`None`）时不报冲突，选择相信调用方给的维度继续。`qdrant_client.models` 导入失败会落入 `except Exception` 变成 `RuntimeError`。`_ensure_payload_indexes` 内部吞异常，所以索引失败不会影响本方法的成功判定。
- **同文件关系**：调用 `_live_collection_size`、`_raise_dimension_mismatch`、`_ensure_payload_indexes`，并使用模块级函数 `_is_dimension_error`；被 `upsert` 和 `upsert_chunk` 调用。

### `_ensure_payload_indexes(self) -> None` （第 168 行）

- **作用**：给 `namespace` 和 `memory_type` 两个字段建 keyword payload 索引，让云端过滤检索能用。这个需求的来源是：本地 Qdrant 服务会按需扫描没有索引的 payload，所以不带索引也能过滤；但 Qdrant Cloud 的 HTTP API 对「在无 payload 索引的字段上做过滤查询」直接返回 HTTP 400，而本类的 `search` 永远会按 `namespace` 过滤、按需按 `memory_type` 过滤，所以在云端必须先把索引建出来。这个操作是幂等的（索引已存在不会报错），而且失败是非致命的：索引是检索需求而不是写入需求，一个建不了索引的集合仍然必须能接受 upsert，所以这里把所有异常都吞掉。
- **参数**：无（除 `self`）。
- **返回**：`None`。副作用是可能发起两次建索引请求（每个字段一次），没有任何返回值或状态变化。
- **内部流程**：`create = getattr(self.client, "create_payload_index", None)`，不是可调用对象就直接 `return`（替身可能不实现）。然后 `for field in ("namespace", "memory_type")` 循环，每轮在 `try` 里调 `create(collection_name=self.collection_name, field_name=field, field_schema="keyword", wait=False)`；`except Exception: continue`，注释说明「已存在 / 集群不支持 / 权限不足都按已处理」。`wait=False` 表示不等待索引构建完成，避免写入路径被索引构建阻塞。
- **异常/边界**：不向外抛任何异常，全部在循环内被吞掉并继续下一个字段。客户端缺少该方法时静默返回。
- **同文件关系**：被 `_ensure_collection`（集合就绪后）和 `recreate_collection`（重建集合后）调用。不调用本文件其它函数。

### `recreate_collection(self, dimension: int) -> None` （第 193 行）

- **作用**：把集合删掉再按给定维度重建，用于「换了嵌入模型之后重建向量投影」这个运维动作。它之所以存在，是因为 Qdrant 集合的维度不可修改，换模型只能重建。文档字符串明确了安全边界：向量只是 SQLite 真值的投影，删除集合只丢投影不丢数据；因此调用方必须**先得到用户确认**，再全量重灌，而这个方法本身从不在写入路径上被静默触发。重建完成后同样会补建 payload 索引并把 `_ready` 置位，让后续写入不需要再走一次懒创建。
- **参数**：
  - `dimension`：`int`，无默认值。新集合的向量维度，必须是正整数。
- **返回**：`None`。副作用是可能删除旧集合、创建新集合、更新 `self.dimension`、建索引、把 `self._ready` 置为 `True`。
- **内部流程**：第一步校验 `dimension`：`isinstance(dimension, bool)` 或非 `int` 或 `< 1` 一律 `raise ValueError("dimension must be a positive integer")`。第二步 `try`：导入 `Distance, VectorParams`；`collection_exists` 为真时先 `delete_collection`；然后 `create_collection(collection_name=..., vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))`；接着 `self.dimension = dimension`、`_ensure_payload_indexes()`、`self._ready = True`。第三步异常处理：`except ValueError: raise`（保留自己的校验错误），`except Exception as exc` 抛 `RuntimeError(f"unable to recreate Qdrant collection: {exc}") from exc`。
- **异常/边界**：`ValueError`——维度非法。`RuntimeError`——删除或创建失败（网络、权限、服务不可用等）。集合本来就不存在时不会报错，直接走创建分支。注意：如果删除成功但创建失败，会留下「集合不存在」的状态，此时 `self._ready` 仍是旧值（可能为 `True`），下一次写入会在 `_ensure_collection` 里重新发现集合并重建。删除数据不可恢复，所以文档强调必须先确认。
- **同文件关系**：调用 `_ensure_payload_indexes`。本文件内部没有其它地方调用它，属于给外部运维/脚本使用的显式重建入口。

### `upsert(self, item: MemoryItem) -> None` （第 219 行）

- **作用**：把一个记忆条目写进（或覆盖）Qdrant 集合，是记忆入库的主写入路径。它的职责边界很清晰：只负责「把 embedding 和轻量 payload 变成 Qdrant 的一个点」，正文和 metadata 不在这里存。它会在写之前用 `len(item.embedding)` 触发集合懒创建与维度核对，写完之后如果客户端抛的是维度类错误，就把错误翻译成带修复指引的 `ValueError`，而不是把底层原始异常直接抛给上层。没有 embedding 的条目会被静默跳过——因为向量库里存一个没有向量的点没有意义。
- **参数**：
  - `item`：`MemoryItem`（从 `..base` 导入），无默认值。记忆条目，需要用到它的 `.embedding`（列表，非空才写入）、`.id`（应用层 id，用于生成点 id）以及 `to_dict()` 返回的字段。
- **返回**：`None`。成功时不返回任何东西；`item.embedding` 为空时直接返回（什么都不做）。
- **内部流程**：第一步 `if not item.embedding: return`，空列表、`None` 都跳过。第二步 `self._ensure_collection(len(item.embedding))`，把维度交给集合保障逻辑。第三步从 `qdrant_client.models` 导入 `PointStruct`。第四步 `payload = self._light_payload(item)` 拿轻量投影，然后手动补上 `payload["namespace"] = self.namespace`——这一步很关键，`search` 强制按 namespace 过滤，缺这个键的点永远检索不到。第五步 `try` 调 `self.client.upsert(collection_name=..., points=[PointStruct(id=self._point_id(item.id), vector=item.embedding, payload=payload)])`。第六步异常处理：`_is_dimension_error(exc)` 为假就 `raise` 原样上抛；为真则调 `self._raise_dimension_mismatch(self._live_collection_size() or self.dimension, len(item.embedding), exc)`，即优先用实时读到的集合维度，读不到就退回 `self.dimension`。
- **异常/边界**：可能抛 `ValueError`（维度冲突，来自 `_raise_dimension_mismatch`，或来自 `_ensure_collection`）、`RuntimeError`（集合初始化失败）、以及客户端本身的其它异常（非维度错误时原样上抛）。`item.embedding` 为空/`None` 时静默返回，不报错。`item.id` 是任意字符串时由 `_point_id` 保证能变成合法点 id。
- **同文件关系**：调用 `_ensure_collection`、`_light_payload`、`_point_id`、`_live_collection_size`、`_raise_dimension_mismatch`，并使用模块级函数 `_is_dimension_error`。本文件内部没有其它方法调用它，它是给上层（如 MemoryManager.add）用的写入接口。

### `upsert_chunk(self, chunk_id: str, vector: list[float], *, document_id: str = "", chunk_index: int = 0, source: str = "", memory_type: str = "semantic") -> None` （第 233 行）

- **作用**：写入一个 chunk 级别的向量点，payload 用的是「窄 chunk 结构」（chunk_id / document_id / chunk_index / source / memory_type / namespace）。它主要服务于向量投影的重索引路径：入库路径仍然走 `upsert`（也就是 MemoryManager.add），而重索引路径走这个方法；两条路径最终都会落到同一个 `chunk_id` 上，所以后续用 id 回查 SQLite 真值时结果是一致的。它和 `upsert` 的另一处一致是：同样补上 `namespace`，因为 `search` 强制按它过滤，缺了这个键的点会永远检索不到。
- **参数**：
  - `chunk_id`：`str`，无默认值。chunk 的应用层 id，既用于生成点 id，也写进 payload 供回查使用。
  - `vector`：`list[float]`，无默认值。chunk 的嵌入向量，空列表会被跳过。
  - `document_id`：`str`，仅关键字参数，默认 `""`。所属文档 id。
  - `chunk_index`：`int`，仅关键字参数，默认 `0`。chunk 在文档内的序号。
  - `source`：`str`，仅关键字参数，默认 `""`。来源标识。
  - `memory_type`：`str`，仅关键字参数，默认 `"semantic"`。记忆类型字符串，注意这里是字符串而不是 `MemoryType` 枚举，直接原样写进 payload。
- **返回**：`None`。成功时不返回；`vector` 为空时直接返回。
- **内部流程**：第一步 `if not vector: return`。第二步 `self._ensure_collection(len(vector))`。第三步导入 `PointStruct`。第四步构造 payload 字典，键依次是 `chunk_id`、`document_id`、`chunk_index`、`source`、`memory_type`，最后补 `"namespace": self.namespace`。第五步 `try` 调 `self.client.upsert(collection_name=..., points=[PointStruct(id=self._point_id(chunk_id), vector=vector, payload=payload)])`。第六步异常处理与 `upsert` 完全同构：非维度错误 `raise`，维度错误则 `_raise_dimension_mismatch(self._live_collection_size() or self.dimension, len(vector), exc)`。
- **异常/边界**：与 `upsert` 相同：`ValueError`（维度冲突）、`RuntimeError`（集合初始化失败）、其它客户端异常原样上抛。`vector` 为空时静默返回。默认参数允许只给 `chunk_id` 和 `vector` 就能写入，其余字段落成空串/0/`"semantic"`。
- **同文件关系**：调用 `_ensure_collection`、`_point_id`、`_live_collection_size`、`_raise_dimension_mismatch`，并使用模块级函数 `_is_dimension_error`。本文件内部没有其它方法调用它，供外部重索引脚本使用。

### `_light_payload(item: MemoryItem) -> dict[str, Any]` （第 261 行，`@staticmethod`）

- **作用**：把完整的 `MemoryItem` 压缩成 Qdrant 里真正需要的那几个字段，形成「轻量投影」。这个设计解决的是「双重真相」问题：chunk 正文和 metadata 已经存在 SQLite（真值源）里，而 1024 个 float 的 embedding 才是每个点的大头；如果在 Qdrant 里再存一份完整副本，就会多出一份会漂移的真相，还得同步维护。所以这里只保留回查和过滤必需的四项：`id`、`memory_type`、`created_at`、`expires_at`。它是静态方法，因为它不需要访问实例状态。
- **参数**：
  - `item`：`MemoryItem`，无默认值。需要它实现 `to_dict()` 并返回包含上述四个键的字典。
- **返回**：`dict[str, Any]`，恰好包含 `"id"`、`"memory_type"`、`"created_at"`、`"expires_at"` 四个键。注意用的是 `full.get(key)`，所以 `to_dict()` 里缺失的键会以 `None` 值出现在结果里（键存在、值为 `None`），而不是被丢掉。
- **内部流程**：`full = item.to_dict()` 拿到完整字典，然后用字典推导 `{key: full.get(key) for key in ("id", "memory_type", "created_at", "expires_at")}` 逐键取值返回。没有分支、没有循环体、没有异常处理。
- **异常/边界**：无特殊处理。`item` 没有 `to_dict()` 会抛 `AttributeError`；`to_dict()` 返回非 dict 会在 `.get` 处抛 `AttributeError`。缺失的键静默变成 `None`。
- **同文件关系**：只被 `upsert` 调用。它不调用本文件其它函数（`item.to_dict()` 属于外部对象的方法）。

### `delete(self, item_id: str) -> bool` （第 272 行）

- **作用**：按应用层 id 删除集合里的一个点，用于记忆被删除时同步清理向量投影。它先走 `_ensure_ready_for_read`，这个检查的含义是「集合存在我才删」——如果集合不存在（比如从没建过、或已经被 recreate 掉），删无可删，直接返回 `False`，而不是让客户端去抛一个「集合不存在」的异常。注意它不检查 `_ready` 状态是否由写入置位，只关心集合此刻是否存在，这样纯读取/纯删除的进程也能正确工作。
- **参数**：
  - `item_id`：`str`，无默认值。应用层记忆 id，会经 `_point_id` 转成 Qdrant 能接受的点 id。
- **返回**：`bool`。集合不存在或探测失败时返回 `False`；删除请求成功发出后返回 `True`。注意返回 `True` 只表示「删除调用成功」，不表示一定删掉了一个存在的点——Qdrant 删除不存在的 id 也不报错。
- **内部流程**：第一步 `if not self._ensure_ready_for_read(): return False`。第二步从 `qdrant_client.models` 导入 `PointIdsList`。第三步 `self.client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=[self._point_id(item_id)]))`。第四步 `return True`。没有 try/except，客户端异常会向上抛。
- **异常/边界**：不捕获异常，删除失败（网络、权限、集合刚被删掉导致的竞态）会把客户端异常抛出。集合不存在时返回 `False` 而不抛异常。传空字符串 `item_id` 不会报错，`_point_id` 会把它映射成一个确定的 UUID（`uuid5` 对空串也能算）。
- **同文件关系**：调用 `_ensure_ready_for_read` 和 `_point_id`。本文件内部没有其它方法调用它，供上层记忆删除流程使用。

### `_ensure_ready_for_read(self) -> bool` （第 279 行）

- **作用**：读路径的「挂载检查」：确认集合存在，但**绝不创建**集合。它解决的问题是：`_ready` 只会被写入路径置位，所以一个只读的进程（reconcile 接口、UI 会话）如果只看 `_ready`，会把一个真实存在的集合误报成「没有向量」，导致 reconcile 得出错误结论。另一个同样重要的点是：读路径即使确认了集合存在，也刻意**不把 `_ready` 置位**，因为置位会让后续写入跳过 `_ensure_collection` 里的维度核对，那正是换模型后最容易出问题的地方。探测本身失败时按「读不到」处理，让调用方降级为空结果。
- **参数**：无（除 `self`）。
- **返回**：`bool`。`self._ready` 已为 `True` 时直接返回 `True`（快速路径，不再打客户端）；`collection_exists` 抛异常返回 `False`；集合不存在返回 `False`；集合存在返回 `True`。
- **内部流程**：第一步 `if self._ready: return True`。第二步 `try`：若 `not self.client.collection_exists(collection_name=self.collection_name)` 就 `return False`。第三步 `except Exception: return False`，注释说明「探测失败按读不到处理，由调用方降级」。第四步返回 `True`。整个方法不修改任何实例状态，是纯查询。
- **异常/边界**：不向外抛异常，探测异常一律降级为 `False`。客户端没有 `collection_exists` 方法时会抛 `AttributeError`，被 `except Exception` 捕获返回 `False`。
- **同文件关系**：被 `delete`、`list_ids`、`search` 三个读路径方法调用。它不调用本文件其它函数。

### `list_ids(self, *, limit: int = 10000) -> list[str]` （第 297 行）

- **作用**：把集合里所有点的「应用层 id」列出来，供 reconcile（对账）流程使用——对账需要拿到 Qdrant 侧的完整 id 集合，才能和 SQLite 真值比对出「多了哪些、少了哪些」。它用 `scroll` 而不是 `search`，因为这里不需要向量相似度，只需要遍历。返回时优先取 payload 里的 `chunk_id`，其次是 `id`，两者都没有才退回 Qdrant 的点 id，这样新旧两种写入路径产生的点都能被正确还原成应用层 id。
- **参数**：
  - `limit`：`int`，仅关键字参数，默认 `10000`。一次 scroll 最多返回的点数，也就是这个方法能列出的上限。本方法不校验它，传 0 或负数会由客户端决定行为。
- **返回**：`list[str]`。集合不存在、客户端没有可调用的 `scroll`、或集合为空时返回空列表；否则返回至多 `limit` 个字符串 id，顺序由 Qdrant 返回顺序决定（不保证稳定）。
- **内部流程**：第一步 `if not self._ensure_ready_for_read(): return []`。第二步 `scroll = getattr(self.client, "scroll", None)`，不可调用就返回 `[]`。第三步解包调用结果：`points, _ = scroll(collection_name=..., limit=limit, with_payload=True, with_vectors=False)`——只要 payload 不要向量，避免拉回大量浮点数据；第二个返回值（下一个翻页 offset）被丢弃，所以这是单页查询。第四步遍历 `points`，对每个点 `payload = point.payload or {}`，然后 `ids.append(str(payload.get("chunk_id") or payload.get("id") or point.id))`。第五步返回 `ids`。
- **异常/边界**：`scroll` 调用本身没有 try/except，客户端异常会向上抛。`point.payload` 为 `None` 时用 `or {}` 兜底。payload 里 `chunk_id`/`id` 都是空串等假值时会退回 `point.id`。超过 `limit` 的点不会被翻页取到，调用方需要自己意识到这是单页查询。
- **同文件关系**：调用 `_ensure_ready_for_read`。本文件内部没有其它方法调用它，供外部 reconcile / 对账逻辑使用。

### `search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]` （第 317 行）

- **作用**：按向量相似度检索记忆，返回「应用层 id + 相似度分数」的列表，是记忆召回的核心读接口。它强制按 `namespace` 过滤，保证不同命名空间的记忆不会串味；`memory_type` 给了就再加一个过滤条件，用于「只召回某一类记忆」。为了兼容不同版本的 qdrant-client，它先试旧的 `client.search`，遇到 `AttributeError`（新版客户端移除了该方法）就改用 `query_points` 并从结果里取 `.points`。返回的 id 解析顺序和 `list_ids` 一致：chunk_id 优先、老点用 id、都没有才退回点 id。
- **参数**：
  - `vector`：`list[float]`，无默认值。查询向量，必须与集合维度一致，否则底层会报维度错。
  - `limit`：`int`，仅关键字参数，默认 `10`。返回条数上限，必须是正整数。
  - `memory_type`：`MemoryType | str | None`，仅关键字参数，默认 `None`。为 `None` 时不过滤类型；否则会经 `MemoryType(memory_type).value` 归一成枚举值字符串再过滤，因此传字符串或枚举都可以，但传一个枚举里不存在的字符串会抛 `ValueError`。
- **返回**：`list[tuple[str, float]]`。集合不存在或探测失败时返回空列表；否则每个元素是 `(id, score)`，`score` 由 `float(point.score)` 得到，顺序按 Qdrant 返回的相似度排序（默认最相似在前）。
- **内部流程**：第一步校验 `limit`：`isinstance(limit, bool)` 或非 `int` 或 `< 1` 抛 `ValueError("limit must be a positive integer")`（单独排除 bool 是因为 `True` 也是 int）。第二步 `if not self._ensure_ready_for_read(): return []`。第三步导入 `FieldCondition, Filter, MatchValue`，构造 `conditions`：第一条永远是 `FieldCondition(key="namespace", match=MatchValue(value=self.namespace))`；`memory_type is not None` 时追加 `FieldCondition(key="memory_type", match=MatchValue(value=MemoryType(memory_type).value))`。第四步 `query_filter = Filter(must=conditions)`。第五步 `try` 调 `self.client.search(collection_name=..., query_vector=vector, query_filter=query_filter, limit=limit)`；`except AttributeError` 时改调 `self.client.query_points(collection_name=..., query=vector, query_filter=query_filter, limit=limit).points`。第六步遍历 `points`，`payload = point.payload or {}`，`hits.append((str(payload.get("chunk_id") or payload.get("id") or point.id), float(point.score)))`。第七步返回 `hits`。
- **异常/边界**：`ValueError`——`limit` 非法；以及 `MemoryType(memory_type)` 对非法类型字符串抛的 `ValueError`。`AttributeError` 被专门用来做客户端版本兼容，但如果 `query_points` 也不存在，会在 `except` 块里再抛 `AttributeError` 向上传播。集合不存在或探测失败时返回空列表，不抛异常。查询向量维度不对时底层异常会原样上抛，这里不做维度翻译（维度翻译只做在写入路径）。
- **同文件关系**：调用 `_ensure_ready_for_read`。本文件内部没有其它方法调用它，供上层记忆召回逻辑使用。

### `_point_id(item_id: str) -> str` （第 338 行，`@staticmethod`）

- **作用**：把任意应用层 id 转成 Qdrant 能接受的点 id。Qdrant 的点 id 只允许 UUID 或整数，而应用层的记忆 id / chunk id 可能是任意字符串（比如带前缀的自定义格式），直接传会被客户端拒绝，所以需要这个转换。策略是：如果原字符串本身就是一个合法 UUID，就原样使用（保持可读性和幂等）；否则用 `uuid.uuid5(uuid.NAMESPACE_URL, f"helloagents-memory:{item_id}")` 做一个带命名空间的确定性哈希。确定性这一点很关键——同一个应用层 id 每次都必须映射到同一个点 id，否则 upsert 会变成「每次新增一个点」而不是覆盖，delete 也删不掉。原始 id 仍然保存在 payload 里，读取时优先从 payload 还原。
- **参数**：
  - `item_id`：`str`，无默认值。任意应用层 id；传非字符串会在 `uuid.UUID` 处抛 `AttributeError` 并被内部捕获，然后走 `str(item_id)` 的哈希路径。
- **返回**：`str`。若 `item_id` 是合法 UUID 字符串则原样返回；否则返回 `uuid5` 生成的 UUID 字符串形式。
- **内部流程**：`try` 里执行 `uuid.UUID(item_id)`——这个构造只做解析校验，结果不保存；解析成功直接 `return item_id`。`except (ValueError, AttributeError)` 时返回 `str(uuid.uuid5(uuid.NAMESPACE_URL, f"helloagents-memory:{item_id}"))`。前缀 `helloagents-memory:` 用于把这类派生 UUID 和其它用途的 UUID 区分开，降低碰撞语义上的歧义。
- **异常/边界**：`ValueError`（不是合法 UUID 文本）和 `AttributeError`（不是字符串、没有合适类型）都被捕获，所以不会向外抛。空字符串会走哈希路径得到一个确定的 UUID。`uuid5` 理论上可能碰撞，但概率可忽略。注意大写形式的合法 UUID 字符串会被原样返回（不做规范化），而 `uuid.UUID()` 解析本身对大小写不敏感，所以大小写不同的同一 UUID 会被视为合法并原样保留，可能产生两个不同的点 id 字符串——这是这个实现的一个隐含边界。
- **同文件关系**：被 `upsert`、`upsert_chunk`、`delete` 三处调用。它不调用本文件其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_vector_size_of` | 从 `VectorParams`、命名向量字典或普通 dict 里把向量维度 size 统一抠成 int 或 None。 |
| `_collection_vector_size` | 从 Qdrant 集合信息里读出集合当前真实的向量维度，兼容对象与 dict 两种形态。 |
| `_is_dimension_error` | 用异常文本判断这是不是「向量维度不匹配」类错误，决定是翻译成指引还是原样上抛。 |
| `_dimension_mismatch_message` | 生成维度冲突时的中文修复指引，说明只能换回原模型或确认后重建向量投影。 |
| `QdrantVectorStore` | 记忆系统的 Qdrant 向量库适配器，继承 `BaseVectorStore`，实现懒建集合的向量写入与检索。 |
| `QdrantVectorStore.__init__` | 校验 namespace / 集合名 / 维度，按 URL 与代理配置创建或注入 Qdrant 客户端，并初始化未就绪状态。 |
| `QdrantVectorStore._live_collection_size` | 吞掉所有异常地向 Qdrant 问一次集合真实维度，读不到就返回 None。 |
| `QdrantVectorStore.collection_dimension` | 对外只读接口：集合存在则返回其真实维度，不存在或探测失败返回 None。 |
| `QdrantVectorStore._raise_dimension_mismatch` | 把所有维度冲突统一抛成带修复指引的 `ValueError`，可按需保留原始异常链。 |
| `QdrantVectorStore._ensure_collection` | 写入前保障集合存在、维度一致、payload 索引就绪，并刷新 `dimension` 与 `_ready`。 |
| `QdrantVectorStore._ensure_payload_indexes` | 幂等地给 `namespace` 与 `memory_type` 建 keyword 索引，失败不致命以满足云端过滤检索。 |
| `QdrantVectorStore.recreate_collection` | 校验维度后删除并重建集合，用于换嵌入模型后重建向量投影，且绝不在写入路径静默触发。 |
| `QdrantVectorStore.upsert` | 把一个记忆条目的 embedding 与轻量 payload 写入集合，维度冲突时给出修复指引。 |
| `QdrantVectorStore.upsert_chunk` | 以窄 chunk payload 写入单个 chunk 向量，服务于投影重索引路径并与入库路径共用同一个 chunk_id。 |
| `QdrantVectorStore._light_payload` | 把 `MemoryItem` 压缩成只含 id、memory_type、created_at、expires_at 的轻量投影 payload。 |
| `QdrantVectorStore.delete` | 集合存在时按应用层 id 删除对应点，集合不存在则返回 False 而不报错。 |
| `QdrantVectorStore._ensure_ready_for_read` | 读路径检查集合是否存在但绝不创建、也绝不置位 `_ready`，探测失败按读不到降级。 |
| `QdrantVectorStore.list_ids` | 用单页 scroll 列出集合里所有点的应用层 id，供对账流程比对真值。 |
| `QdrantVectorStore.search` | 强制按 namespace（可选 memory_type）过滤做向量检索，返回 id 与相似度分数，并兼容新旧客户端 API。 |
| `QdrantVectorStore._point_id` | 把任意应用层 id 确定性地映射成合法 UUID 点 id，合法 UUID 则原样保留。 |
