# memory/manager.py

## 一、这个文件是干什么的

这个文件是四层记忆系统（工作记忆 WorkingMemory、情景记忆 EpisodicMemory、语义记忆 SemanticMemory、感知记忆 PerceptualMemory）的**统一协调层**，对外只暴露 `MemoryManager` 这一个入口类，把「文档存储 / 向量存储 / 图存储 / 嵌入模型」四类底层组件组装起来并分发给四种记忆实现。

它的核心职责有三个：第一，**依赖注入与默认装配**——调用方可以自己传入 embedding、document_store、vector_store、graph_store，也可以什么都不传，由 `MemoryManager.__init__` 按 `MemoryConfig` 自动挑选 SQLite / Qdrant / InMemory / Neo4j 的具体实现；第二，**按类型路由**——`add`、`get`、`delete`、`search`、`list`、`clear` 六个业务方法都支持可选的 `memory_type` 参数，传了就转交给对应那一层记忆，不传就走跨层逻辑（比如 `search` 会把四层结果合并后按分数重排，`get`/`delete` 会先去文档存储里反查这条记忆属于哪一层）；第三，**生命周期管理**——`close` 统一关闭三种存储，配合 `__enter__`/`__exit__` 支持 `with` 语句。

文件里另外还有两个模块级的小工具：`_is_loopback_endpoint` 判断一个地址是不是本机回环地址，`cloud_proxy_url` 决定访问云端 Qdrant / Neo4j 时要不要挂代理（云端默认走本机 7890 代理，回环地址和显式配置的代理除外），这两个函数只在 `__init__` 装配向量库和图库时被调用。

文件末尾的 `__all__` 把 `MemoryManager` 和 `cloud_proxy_url` 标记为对外公开符号，`_is_loopback_endpoint` 因为是私有工具没有导出。整个文件不含任何业务算法，纯粹是「装配 + 转发 + 合并排序」的胶水层，Agent 运行时和 FastAPI Web 应用在使用记忆功能时，通常先构造一个 `MemoryManager`，然后全程通过它读写记忆。

## 二、函数与类逐条详解

### `_is_loopback_endpoint(value: str | None) -> bool` （第 25 行）
- **作用**：判断传入的地址字符串解析出来的主机名是不是本机回环地址，用来区分「连本机服务」和「连云端服务」这两种情况。之所以需要它，是因为 `cloud_proxy_url` 要决定是否给连接挂代理：如果目标是本机（127.0.0.1 / localhost / ::1），再走代理既没有意义又容易出错，所以必须先做这个判定。它只关心主机名，不看端口、路径、用户名密码等部分。传入 `None` 或空字符串时会被当作「不是回环」处理。这个函数是本模块私有工具，只在 `cloud_proxy_url` 内部被调用一次。
- **参数**：
  - `value`：`str | None`，待判断的地址，可以是完整的 URL（如 `http://127.0.0.1:6333`）、带协议的 URI（如 `bolt://localhost:7687`），也可以是空字符串或 `None`。没有默认值，位置参数。
- **返回**：`bool`。解析出的主机名（统一转小写后）属于 `{"127.0.0.1", "localhost", "::1"}` 三者之一时返回 `True`，否则返回 `False`；当 `value` 为 `None`、空串或解析不出主机名时，`urlsplit` 得到的 `hostname` 为 `None`，经 `or ""` 兜底后为空字符串，最终返回 `False`。
- **内部流程**：第一步 `urlsplit(value or "")`，用 `value or ""` 把 `None` 和空串统一成空字符串，避免 `urlsplit` 收到 `None` 报错；第二步取 `.hostname` 属性并用 `or ""` 兜底，防止没有主机名时出现 `None`；第三步调用 `.casefold()` 做大小写无关化，保证 `LOCALHOST`、`LocalHost` 也能命中；第四步用 `in` 判断是否落在三个回环主机的集合里并直接返回该布尔结果。
- **异常/边界**：本函数不主动抛异常。`urlsplit` 对畸形字符串通常不会抛错，只是可能解析不出主机名；主机名为 `None` 时用 `or ""` 处理，因此不会出现 `None.casefold()` 这类属性错误。注意它不识别 `0.0.0.0`、`[::]`、`127.0.0.2` 等其他本机/回环写法，这些会被判为「非回环」。`IPv6` 地址需要带方括号才能被 `urlsplit` 正确解析出主机名（解析结果形如 `::1`，不带方括号）。
- **同文件关系**：只被本文件的 `cloud_proxy_url` 调用；自身不调用本文件里的任何其它函数。

### `cloud_proxy_url(config: MemoryConfig, endpoint: str | None) -> str | None` （第 30 行）
- **作用**：为云端 Qdrant / Neo4j 连接计算应该使用的代理地址。项目约定「云端向量库和图库默认走本机 7890 代理」，而访问本机服务不应该挂代理，所以这个函数把三种情形区分开：没有地址、地址是回环地址，都返回 `None`（表示直连、不用代理）；地址是远程的且用户在 `MemoryConfig` 里显式配置了 `proxy_url`，就用用户配置的那个；地址是远程但没有显式配置，就退回模块级默认值 `DEFAULT_PROXY_URL`。它是本模块的公开工具函数（列在 `__all__` 里），实际调用点只有 `MemoryManager.__init__` 装配 `QdrantVectorStore` 和 `Neo4jGraphStore` 的两处。这样设计让「本地开发不折腾代理、云端部署自动走代理」成为默认行为，同时保留显式覆盖的能力。
- **参数**：
  - `config`：`MemoryConfig`，记忆系统配置对象，函数只读取它的 `proxy_url` 字段（`str | None`），不修改它。没有默认值。
  - `endpoint`：`str | None`，目标服务地址，例如 `config.qdrant_url` 或 `config.neo4j_uri`。可以是 `None` 或空字符串，此时视为「无地址」直接返回 `None`。没有默认值。
- **返回**：`str | None`。`endpoint` 为空或为回环地址时返回 `None`；否则若 `config.proxy_url` 有值则返回该值；否则返回 `DEFAULT_PROXY_URL`（从 `constants` 模块导入的默认代理地址，通常形如 `http://127.0.0.1:7890`）。
- **内部流程**：第一步 `if not endpoint or _is_loopback_endpoint(endpoint): return None`，用短路求值一次性处理「空地址」和「本机地址」两种情况；第二步 `if config.proxy_url: return config.proxy_url`，显式配置优先；第三步兜底 `return DEFAULT_PROXY_URL`，把「远程地址 + 未配置代理」的默认行为固定成本机 7890。
- **异常/边界**：不主动抛异常。`endpoint` 为 `None` 或空串走第一条分支安全返回；`config` 缺少 `proxy_url` 属性会抛 `AttributeError`（由 `MemoryConfig` 的定义保证不会发生）；`config.proxy_url` 为空字符串会被当作「未配置」，从而回退到默认代理。注意 `DEFAULT_PROXY_URL` 本身如果被配置成空值，函数会原样返回它，不做二次判断。
- **同文件关系**：调用了本文件的私有函数 `_is_loopback_endpoint`；被本文件的 `MemoryManager.__init__` 调用（两处：构造 `QdrantVectorStore` 与构造 `Neo4jGraphStore`）。

### `class MemoryManager` （第 40 行）
- **作用**：整个记忆子系统的唯一门面（facade）类，对外代表「四层记忆 + 三种存储」的整体。它把 `BaseMemory` 的四个具体实现（`WorkingMemory`、`EpisodicMemory`、`SemanticMemory`、`PerceptualMemory`）与共享的 `document_store`、`vector_store`、`embedding`、`config` 装配在一起，并用 `self.memories` 字典按 `MemoryType` 枚举建立索引。上层代码（Agent 运行时、FastAPI 路由）只需要持有一个 `MemoryManager` 实例，就可以用统一签名读写任意一层的记忆，也能跨层搜索。类本身只定义方法、不定义类属性，实例状态全部在 `__init__` 里创建。它实现了上下文管理器协议，因此推荐用 `with MemoryManager(...) as mm:` 的方式使用，退出时自动释放三种存储的连接。
- **参数**：类本身不接受参数，参数由 `__init__` 定义。
- **返回**：类不是函数，实例化后返回 `MemoryManager` 对象。
- **内部流程**：类体内依次定义 `__init__`、`for_type`、`add`、`get`、`delete`、`search`、`list`、`clear`、`close`、`__enter__`、`__exit__` 共 11 个方法，没有类变量、没有继承基类（隐式继承 `object`）、没有装饰器。
- **异常/边界**：类本身不抛异常；实例化过程中的异常见 `__init__` 条目。
- **同文件关系**：类内各方法互相调用，关系详见各方法条目；它是模块级唯一导出的类。

### `MemoryManager.__init__(self, config: MemoryConfig | None = None, *, embedding: BaseEmbedding | None = None, embedding_service: BaseEmbedding | None = None, document_store: BaseDocumentStore | None = None, vector_store: BaseVectorStore | None = None, graph_store: Neo4jGraphStore | None = None) -> None` （第 43 行）
- **作用**：构造记忆管理器并完成全部依赖装配。它先确定配置对象，再按「显式传入优先、否则按配置自动挑选、最后兜底内存实现」的优先级决定四个共享组件，然后把它们打包成 `common` 字典分发给四种记忆实现，建立 `MemoryType -> 实例` 的映射表，最后在向量库是进程内实现时把文档存储里的历史数据回灌进去。这个「回灌」只在 `InMemoryVectorStore` 时才做，因为远程/持久化向量库本身已经持有向量，启动时再全量 upsert 一遍纯属浪费启动时间（代码里保留了这段注释说明该设计意图）。它是整个文件里逻辑最重、分支最多的方法。
- **参数**：
  - `config`：`MemoryConfig | None`，默认 `None`。为 `None` 时内部新建一个默认 `MemoryConfig()`；传入时直接使用，不会拷贝或修改。它决定了 sqlite 路径、qdrant 地址/集合名/api key、neo4j 连接三元组、工作记忆容量、搜索条数上限、代理地址等。
  - `embedding`：`BaseEmbedding | None`，仅关键字参数，默认 `None`。显式指定嵌入模型实例，优先级最高。与 `embedding_service` 互斥。
  - `embedding_service`：`BaseEmbedding | None`，仅关键字参数，默认 `None`。`embedding` 的别名/兼容写法，语义相同，优先级低于 `embedding`。同时传两者会抛 `ValueError`。
  - `document_store`：`BaseDocumentStore | None`，仅关键字参数，默认 `None`。显式指定文档存储，传入时完全接管，忽略 `config.sqlite_path`；为 `None` 时按 `config.sqlite_path` 新建 `SQLiteDocumentStore`。
  - `vector_store`：`BaseVectorStore | None`，仅关键字参数，默认 `None`。显式指定向量存储；为 `None` 时若 `config.qdrant_url` 有值则新建 `QdrantVectorStore`（并带上经 `cloud_proxy_url` 计算出的代理），否则新建进程内 `InMemoryVectorStore`。
  - `graph_store`：`Neo4jGraphStore | None`，仅关键字参数，默认 `None`。显式指定图存储；为 `None` 时按 `config.neo4j_uri` / `neo4j_username` / `neo4j_password` 新建 `Neo4jGraphStore`，代理同样由 `cloud_proxy_url` 决定。
- **返回**：`None`。构造完的实例状态写入 `self.config`、`self.embedding`、`self.document_store`、`self.vector_store`、`self.graph_store`、`self.working`、`self.episodic`、`self.semantic`、`self.perceptual`、`self.memories`。
- **内部流程**：第一步 `self.config = config if config is not None else MemoryConfig()`，注意用的是 `is not None` 而不是真值判断，避免把合法的空配置对象替换掉；第二步校验 `embedding` 与 `embedding_service` 不能同时给出，同时给出则 `raise ValueError("provide either embedding or embedding_service, not both")`；第三步用嵌套三元表达式确定 `self.embedding`，优先级为 `embedding` > `embedding_service` > `make_default_embedding(self.config)`；第四步确定 `self.document_store`，显式传入优先，否则用 `SQLiteDocumentStore(self.config.sqlite_path)`；第五步确定 `self.vector_store`，显式传入优先，否则有 `qdrant_url` 时构造 `QdrantVectorStore`（参数含 `url`、`collection_name=config.qdrant_collection`、`api_key=config.qdrant_api_key`、`proxy_url=cloud_proxy_url(self.config, self.config.qdrant_url)`），都没有时用 `InMemoryVectorStore()`；第六步确定 `self.graph_store`，显式传入优先，否则构造 `Neo4jGraphStore(config.neo4j_uri, config.neo4j_username, config.neo4j_password, proxy_url=cloud_proxy_url(self.config, self.config.neo4j_uri))`；第七步组装 `common` 字典，键为 `document_store`、`vector_store`、`embedding`、`config`，值为刚确定的四个组件；第八步用 `**common` 展开分别构造四种记忆：`WorkingMemory(capacity=self.config.working_memory_capacity, **common)`、`EpisodicMemory(**common)`、`SemanticMemory(graph_store=self.graph_store, **common)`（只有语义记忆额外需要图存储）、`PerceptualMemory(**common)`；第九步建立 `self.memories` 字典，用 `MemoryType.WORKING/EPISODIC/SEMANTIC/PERCEPTUAL` 四个枚举做键映射到上面四个实例；第十步 `isinstance(self.vector_store, InMemoryVectorStore)` 判断，成立时遍历 `self.document_store.list()` 并对每个 `item` 执行 `self.vector_store.upsert(item)`，把本地索引重建起来。
- **异常/边界**：`embedding` 和 `embedding_service` 同时非 `None` 时抛 `ValueError`；底层存储构造失败（例如 Qdrant 地址不可达、Neo4j 认证失败、SQLite 路径不可写）会把底层异常原样向上抛出，本方法不做捕获和降级；`config` 为 `None` 时用默认配置兜底；`config.qdrant_url` 为空（`None` 或空串）时走内存向量库分支；回灌循环里如果某条 `item` 的 upsert 失败，异常会中断构造过程，导致实例不可用；`self.memories` 固定包含四个键，不存在的类型不会出现在字典里。注意此处没有对底层连接做连通性预检，也没有回滚已创建资源的清理逻辑。
- **同文件关系**：调用了本文件的 `cloud_proxy_url`（两次）；构造并持有 `WorkingMemory`、`EpisodicMemory`、`SemanticMemory`、`PerceptualMemory` 四个外部类实例；被上层通过 `MemoryManager(...)` 直接调用，也被 `__enter__` 所在的 `with` 语法隐式触发。

### `MemoryManager.for_type(self, memory_type: MemoryType | str) -> BaseMemory` （第 103 行）
- **作用**：把「可能是枚举、也可能是字符串」的记忆类型统一解析成 `MemoryType` 枚举，并返回对应的记忆实现实例，是 `add`/`get`/`delete`/`search`/`list`/`clear` 六个业务方法共用的路由底座。之所以需要它，是因为调用方经常用字符串（例如从 HTTP 请求 JSON 里读到的 `"working"`、`"semantic"`）来指定记忆类型，而内部索引字典的键是枚举，必须做一次规范化。解析失败时它会把底层的 `KeyError`/`ValueError` 统一改写成信息更清楚的 `ValueError`，并把原异常挂到 `__cause__` 上便于排查。
- **参数**：
  - `memory_type`：`MemoryType | str`，必填位置参数。可以是 `MemoryType` 枚举成员本身，也可以是能构造出枚举的字符串（大小写需与枚举值定义一致，具体取决于 `MemoryType` 的取值定义），还可以是任意非法值用于触发错误分支。
- **返回**：`BaseMemory`。返回 `self.memories` 中该类型对应的实例（实际运行时是 `WorkingMemory`、`EpisodicMemory`、`SemanticMemory`、`PerceptualMemory` 之一，它们的共同基类为 `BaseMemory`）。
- **内部流程**：第一步 `MemoryType(memory_type)`，用枚举的构造能力把字符串或枚举统一成枚举成员（传枚举成员时是幂等的）；第二步用结果作为键去 `self.memories` 取实例并返回；整个过程包在 `try/except (KeyError, ValueError)` 中，捕获到任一异常时执行 `raise ValueError(f"unknown memory type: {memory_type}") from exc`，保留原始异常链。
- **异常/边界**：`memory_type` 无法转成合法 `MemoryType` 时抛 `ValueError`，消息格式为 `unknown memory type: <原值>`；`memory_type` 能转成合法枚举但不在 `self.memories` 字典中时同样抛 `ValueError`（先被 `KeyError` 捕获再改写）。传入 `None` 会由 `MemoryType(None)` 抛出 `ValueError` 并被改写。本方法不返回 `None`，不存在「找不到就静默返回空」的行为。
- **同文件关系**：读取 `__init__` 建立的 `self.memories`；被本文件的 `add`、`get`、`delete`、`search`、`list`、`clear` 六个方法在传入 `memory_type` 时调用；自身不调用本文件里的其它函数。

### `MemoryManager.add(self, content: str, *, memory_type: MemoryType | str = MemoryType.WORKING, **kwargs: Any) -> MemoryItem` （第 109 行）
- **作用**：向指定层的记忆写入一条新记忆，是记忆系统最主要的写入口。它本身不做任何加工，只负责按 `memory_type` 找到对应实现并把 `content` 与其余关键字参数原样转发给那一层的 `add`，因此各层记忆自己特有的参数（例如重要度、标签、元数据、来源、时间戳等）都通过 `**kwargs` 透传，无需在这里逐一声明。默认写入工作记忆，符合「随手记录先放短期缓冲」的常见用法。
- **参数**：
  - `content`：`str`，必填位置参数，要记住的文本内容，由具体记忆层决定如何切分、向量化与存储。
  - `memory_type`：`MemoryType | str`，仅关键字参数，默认 `MemoryType.WORKING`。可以是枚举也可以是字符串，最终交给 `for_type` 解析；传非法值会在路由阶段抛 `ValueError`。
  - `**kwargs`：`Any`，其余任意关键字参数，全部原样转发给对应记忆实现的 `add`，例如元数据、重要度、标签、时间等，具体支持哪些由各记忆层的签名决定。
- **返回**：`MemoryItem`，即该层 `add` 返回的写入结果对象（通常包含生成的 id、内容、类型、创建时间等字段）。
- **内部流程**：唯一一步是 `return self.for_type(memory_type).add(content, **kwargs)`，即先解析类型拿到记忆实例，再把位置参数 `content` 与关键字参数 `kwargs` 透传调用其 `add` 方法并返回结果。
- **异常/边界**：`memory_type` 非法时由 `for_type` 抛 `ValueError`；`kwargs` 里出现目标 `add` 不认识的参数时抛 `TypeError`；`content` 为 `None` 或非字符串不会被本方法拦截，是否报错取决于具体记忆层的实现；本方法不做去重、不做长度校验、不做异常包装。
- **同文件关系**：调用本文件的 `for_type`；被上层业务代码（Agent 运行时、Web 路由）调用；自身不调用本文件其它方法。

### `MemoryManager.get(self, item_id: str, *, memory_type: MemoryType | str | None = None) -> MemoryItem | None` （第 112 行）
- **作用**：按 id 读取单条记忆。指定了 `memory_type` 时直接交给那一层去查；没指定时先在共享的 `document_store` 里按 id 反查这条记忆，从而得知它究竟属于哪一层，然后再判断是否已过期——如果已经过期，就顺手把它删掉并返回 `None`，避免调用方读到过期数据。这个「读到过期就清理」的顺手删除是懒清理（lazy expiration）策略，让过期条目在真正被访问时才付出删除成本，而不是靠后台定时任务扫描。它是 `delete` 无类型分支的反向操作（`delete` 也是先反查再删）。
- **参数**：
  - `item_id`：`str`，必填位置参数，记忆条目的唯一标识。
  - `memory_type`：`MemoryType | str | None`，仅关键字参数，默认 `None`。为 `None` 时走跨层反查逻辑；非 `None` 时限定在该层查找，找不到由该层返回 `None`。
- **返回**：`MemoryItem | None`。找到且未过期时返回该条目；`memory_type` 分支下由对应记忆层决定返回值（通常找不到返回 `None`）；无类型分支下文档存储里没有该 id 时返回 `None`，条目已过期时删除后返回 `None`。
- **内部流程**：第一步判断 `memory_type is not None`，成立则直接 `return self.for_type(memory_type).get(item_id)`；第二步 `item = self.document_store.get(item_id)` 在文档存储里反查；第三步判断 `item is not None and item.is_expired`，成立时调用 `self.delete(item_id)`（无类型版本，即跨层删除）再 `return None`；第四步返回 `item`（此时可能是 `None`，也可能是未过期的条目）。
- **异常/边界**：`memory_type` 非法时由 `for_type` 抛 `ValueError`；`item_id` 不存在时返回 `None` 而不报错；条目过期时先删除再返回 `None`，如果删除过程出错，异常会向上抛出（而不是吞掉）；`item.is_expired` 由 `MemoryItem` 提供，本方法不自行计算过期时间。无类型分支不校验 `item_id` 的格式。
- **同文件关系**：调用本文件的 `for_type`（有类型分支）和 `delete`（清理过期条目时，且是无类型调用，因此不会递归回 `get`）；被上层业务代码调用；与 `delete` 在「先查文档存储再路由」这一模式上对称。

### `MemoryManager.delete(self, item_id: str, *, memory_type: MemoryType | str | None = None) -> bool` （第 121 行）
- **作用**：按 id 删除一条记忆。指定了 `memory_type` 时直接在该层删除；没指定时先在文档存储里反查条目以确定它属于哪一层，然后路由到正确的层去删。之所以不能无脑遍历四层调用删除，是因为要保证返回值语义清晰（文档存储里根本没有这个 id 时直接返回 `False`，避免四次无意义查询）。它同时被 `get` 的过期清理逻辑复用，形成「发现过期即删除」的闭环。
- **参数**：
  - `item_id`：`str`，必填位置参数，要删除的记忆条目 id。
  - `memory_type`：`MemoryType | str | None`，仅关键字参数，默认 `None`。为 `None` 时自动识别条目所属层；非 `None` 时限定在该层删除。
- **返回**：`bool`。删除成功返回 `True`；`memory_type` 分支下由对应记忆层返回其布尔结果；无类型分支下文档存储里没有该 id 时返回 `False`。
- **内部流程**：第一步判断 `memory_type is not None`，成立则 `return self.for_type(memory_type).delete(item_id)`；第二步 `item = self.document_store.get(item_id)` 反查条目；第三步 `if item is None: return False`，提前短路；第四步 `return self.for_type(item.memory_type).delete(item_id)`，用条目自身的 `memory_type` 字段路由到正确层执行删除并返回其布尔结果。
- **异常/边界**：`memory_type` 非法时抛 `ValueError`；`item_id` 不存在返回 `False` 而不报错；如果条目存在于文档存储但其 `memory_type` 字段是非法值，`for_type` 会抛 `ValueError`（这种数据不一致不会被静默忽略）；本方法不级联删除向量库或图库中的关联数据，那属于各层 `delete` 的内部职责。
- **同文件关系**：调用本文件的 `for_type`；被本文件的 `get` 调用（处理过期条目）；也被上层业务代码直接调用。

### `MemoryManager.search(self, query: str, *, memory_type: MemoryType | str | None = None, limit: int | None = None, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[MemorySearchResult]` （第 129 行）
- **作用**：语义检索入口。指定 `memory_type` 时只搜那一层；不指定时会在四层记忆里各搜一遍，把结果汇总后按分数从高到低排序（同分时按创建时间升序，即更早的排前面），最后截断到 `limit` 条。这样设计的好处是四层记忆共用同一个向量集合，仍能给调用方保留「按类型过滤」和「全局融合检索」两种用法。参数校验在这里集中做：`limit` 必须是正整数（显式排除 `bool`，因为 `True` 是 `int` 的子类，`limit=True` 属于明显的误用），否则抛 `ValueError`；`limit` 为 `None` 时取配置里的 `config.search_limit`。
- **参数**：
  - `query`：`str`，必填位置参数，检索用的自然语言查询文本，由各记忆层负责向量化后做相似度匹配。
  - `memory_type`：`MemoryType | str | None`，仅关键字参数，默认 `None`。指定时只搜该层并直接把该层结果返回（此时 `limit`/`threshold`/`metadata` 原样透传，不做本方法的 limit 校验）；为 `None` 时执行跨层融合检索。
  - `limit`：`int | None`，仅关键字参数，默认 `None`。为 `None` 时使用 `self.config.search_limit`；非 `None` 时必须是非布尔的正整数（`>= 1`），否则抛 `ValueError`。融合分支中它既决定最终返回条数，也决定每层各自的检索条数（`per_type_limit = max(limit, 1)`）。
  - `threshold`：`float | None`，仅关键字参数，默认 `None`。相似度阈值，原样透传给各记忆层的 `search`，`None` 表示由各层使用自己的默认阈值。
  - `metadata`：`Mapping[str, Any] | None`，仅关键字参数，默认 `None`。元数据过滤条件，原样透传给各层，`None` 表示不过滤。
- **返回**：`list[MemorySearchResult]`。`memory_type` 分支下直接返回该层的搜索结果列表；融合分支下返回合并排序并截断后的列表，长度最多为 `limit`，没有任何命中时返回空列表 `[]`（不会返回 `None`）。
- **内部流程**：第一步判断 `memory_type is not None`，成立则 `return self.for_type(memory_type).search(query, limit=limit, threshold=threshold, metadata=metadata)`，注意此处不做 limit 校验；第二步 `limit = self.config.search_limit if limit is None else limit` 填默认值；第三步 `if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1: raise ValueError("limit must be a positive integer")`，把布尔值、非整数（如字符串、浮点）、小于 1 的整数全部拒绝；第四步 `per_type_limit = max(limit, 1)` 计算每层检索条数（因为已经校验过 `limit >= 1`，这里实际等于 `limit`，属于防御式写法）；第五步初始化空列表 `found`，遍历 `self.memories.values()`（四个记忆层实例），对每个 `memory` 调用 `memory.search(query, limit=per_type_limit, threshold=threshold, metadata=metadata)` 并把返回结果 `extend` 进 `found`；第六步 `found.sort(key=lambda result: (-result.score, result.item.created_at))`，用「分数取负」实现降序、同分时用创建时间升序；第七步 `return found[:limit]` 截断到最终条数。
- **异常/边界**：`memory_type` 非法时由 `for_type` 抛 `ValueError`；融合分支下 `limit` 为 `bool`、非 `int` 或 `< 1` 时抛 `ValueError("limit must be a positive integer")`（注意有类型分支时不做这个校验，非法 `limit` 会被直接透传给底层）；`query` 为空字符串不会被本方法拦截；某个记忆层的 `search` 抛异常会中断整个融合检索，前面的结果全部丢弃；`found.sort` 假定每个结果都有 `score` 属性、其 `item` 有 `created_at` 属性，缺任何一个都会抛 `AttributeError`；结果分数相同时依赖 `created_at` 可比较。
- **同文件关系**：调用本文件的 `for_type`（有类型分支）；遍历读取 `__init__` 建立的 `self.memories`，并调用其中四个记忆实例的 `search`；读取 `self.config.search_limit`；被上层业务代码调用。

### `MemoryManager.list(self, *, memory_type: MemoryType | str | None = None, include_expired: bool = False) -> list[MemoryItem]` （第 144 行）
- **作用**：列出记忆条目，用于后台管理、调试页面、导出或统计等场景。指定 `memory_type` 时列出那一层的条目；不指定时直接向共享的 `document_store` 要全量列表——因为文档存储是四层记忆共同的持久化底座，一次查询就能覆盖所有层，比逐层调用再拼接更省事。`include_expired` 控制是否把已过期的条目也算进来，默认 `False`，符合「列表里不该出现失效数据」的直觉。
- **参数**：
  - `memory_type`：`MemoryType | str | None`，仅关键字参数，默认 `None`。指定时只列该层；为 `None` 时列文档存储中的全部条目。
  - `include_expired`：`bool`，仅关键字参数，默认 `False`。为 `False` 时过滤掉已过期条目，为 `True` 时包含它们；本方法只做透传，具体过滤由记忆层或文档存储实现。
- **返回**：`list[MemoryItem]`。无命中时返回空列表 `[]`，不返回 `None`。
- **内部流程**：第一步判断 `memory_type is not None`，成立则 `return self.for_type(memory_type).list(include_expired=include_expired)`；第二步 `return self.document_store.list(include_expired=include_expired)`，把过滤责任交给文档存储。
- **异常/边界**：`memory_type` 非法时由 `for_type` 抛 `ValueError`；文档存储为空时返回空列表；`include_expired` 传入非布尔真值会被原样透传，是否报错取决于底层实现；本方法不做分页、不做排序保证（顺序由底层存储决定）。
- **同文件关系**：调用本文件的 `for_type`（有类型分支），读取 `self.document_store`（无类型分支）；被上层业务代码调用；与 `__init__` 里的启动回灌循环共用同一个 `document_store.list()`（但那里调用的是不带参数的默认形式）。

### `MemoryManager.clear(self, *, memory_type: MemoryType | str | None = None) -> int` （第 149 行）
- **作用**：清空记忆，返回被清掉的条目数量。指定 `memory_type` 时只清那一层；不指定时把四层记忆全部清空，并用 `sum(...)` 把各层返回的删除条数累加成总数。它常用于测试隔离、会话重置、管理接口的「一键清库」等场景。返回总数而不是布尔值，方便调用方做日志和统计。
- **参数**：
  - `memory_type`：`MemoryType | str | None`，仅关键字参数，默认 `None`。指定时只清该层；为 `None` 时清空 `self.memories` 中的全部四层。
- **返回**：`int`。有类型分支下返回该层 `clear` 的结果（通常是被删除的条数）；无类型分支下返回四层删除条数之和，全部为空时返回 `0`。
- **内部流程**：第一步判断 `memory_type is not None`，成立则 `return self.for_type(memory_type).clear()`；第二步 `return sum(memory.clear() for memory in self.memories.values())`，用生成器表达式遍历四个记忆实例、逐个调用 `clear()`，再对返回值求和。注意这里是逐层调用而不是直接清空文档存储，因此各层的向量索引、图数据等也由各层自己同步清理。
- **异常/边界**：`memory_type` 非法时由 `for_type` 抛 `ValueError`；某层 `clear` 返回非数值时 `sum` 会抛 `TypeError`；遍历过程中某层 `clear` 抛异常会中断整个清空，此前已清掉的层不会回滚；空管理器返回 `0`。本方法不清空 `self.memories` 字典本身，也不重建实例，只清数据。
- **同文件关系**：调用本文件的 `for_type`（有类型分支）；遍历调用 `self.memories` 中四个记忆实例的 `clear`；被上层业务代码调用。

### `MemoryManager.close(self) -> None` （第 154 行）
- **作用**：释放三种存储占用的资源，是管理器的统一收尾动作。文档存储的 `close` 被直接调用（因为 `BaseDocumentStore` 保证有这个方法），而向量存储和图存储则先用 `getattr(..., "close", None)` 探测是否存在可调用的 `close`，只有存在才调用——这样即使某个实现（比如纯内存的 `InMemoryVectorStore`）没有定义 `close`，或者用了同步/异步不同的接口风格，也不会因为属性缺失而报 `AttributeError`。它被 `__exit__` 调用，因此 `with` 语句退出时会自动执行。
- **参数**：无（除 `self`）。
- **返回**：`None`。
- **内部流程**：第一步 `self.document_store.close()` 直接关闭文档存储；第二步 `close_vector = getattr(self.vector_store, "close", None)`，第三步 `if callable(close_vector): close_vector()`，存在且可调用才关向量存储；第四步 `close = getattr(self.graph_store, "close", None)`，第五步 `if callable(close): close()`，同样方式关闭图存储。注意局部变量名复用了 `close`，第二、三步的向量关闭函数保存在 `close_vector` 里以免被覆盖。
- **异常/边界**：文档存储没有 `close` 属性时抛 `AttributeError`（不做防御）；向量/图存储缺少 `close` 或 `close` 不可调用时静默跳过，不报错也不提示；某个 `close` 内部抛异常会中断后续关闭动作，导致后面的存储无法释放（本方法不做 `try/finally` 保护）；重复调用 `close` 的行为取决于底层实现，本方法不记录「已关闭」状态、不做幂等保护。
- **同文件关系**：读取 `self.document_store`、`self.vector_store`、`self.graph_store`（均在 `__init__` 中创建）；被本文件的 `__exit__` 调用，也可被上层显式调用。

### `MemoryManager.__enter__(self) -> Self` （第 163 行）
- **作用**：实现上下文管理器协议的进入动作，让 `with MemoryManager(...) as mm:` 这种写法成立。它不做事前初始化、不做连接预检、不做任何资源申请，只是把自身返回出去，保证 `with ... as mm` 里的 `mm` 就是刚构造好的管理器实例。之所以只写一行 `return self`，是因为真正的装配工作已经在 `__init__` 里完成，进入上下文时无需额外准备。
- **参数**：无（除 `self`）。
- **返回**：`Self`，即 `MemoryManager` 实例本身（类型标注使用 `typing.Self`，表示返回当前类）。
- **内部流程**：唯一一步 `return self`。
- **异常/边界**：无特殊处理，不会抛异常，也没有对「重复进入」做限制（同一个实例可以被多个 `with` 块先后使用）。
- **同文件关系**：与 `__exit__` 配对构成上下文管理器协议；不调用本文件里的其它函数；被 `with` 语句隐式调用。

### `MemoryManager.__exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None` （第 166 行）
- **作用**：实现上下文管理器协议的退出动作，无论 `with` 代码块是正常结束还是抛异常退出，都会执行，用于确保三种存储被释放。它只做一件事：调用 `self.close()`。返回 `None`（假值）意味着**不吞掉异常**——如果代码块里抛了异常，异常会继续向上传播，`close` 只负责清理，不负责改变控制流。这是典型的「资源清理型」上下文管理器实现。
- **参数**：
  - `exc_type`：`type[BaseException] | None`，异常类型；正常退出时为 `None`。本方法不读取该参数。
  - `exc_value`：`BaseException | None`，异常实例；正常退出时为 `None`。本方法不读取该参数。
  - `traceback`：`TracebackType | None`，异常回溯对象；正常退出时为 `None`。本方法不读取该参数。
- **返回**：`None`。返回 `None` 表示不抑制异常，异常继续传播。
- **内部流程**：唯一一步 `self.close()`，忽略三个参数。
- **异常/边界**：`close()` 内部抛出的异常会覆盖代码块中原有的异常向外传播（本方法不做 `try/except` 保护，也不判断 `exc_type`）；三个参数可为 `None`，因为根本没用它们，所以不存在空值问题。
- **同文件关系**：调用本文件的 `close`；与 `__enter__` 配对；被 `with` 语句在退出时隐式调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_is_loopback_endpoint` | 判断地址解析出的主机名是否为 127.0.0.1 / localhost / ::1 这类本机回环地址。 |
| `cloud_proxy_url` | 为云端 Qdrant/Neo4j 计算代理地址：空地址或回环地址返回 `None`，否则优先用配置的代理、没有就用默认 7890 代理。 |
| `MemoryManager` | 四层记忆系统的统一门面类，负责装配存储组件、按类型路由读写请求并提供上下文管理能力。 |
| `MemoryManager.__init__` | 按「显式传入 > 配置自动挑选 > 内存兜底」确定配置、嵌入模型与三种存储，构造四种记忆实例并建立类型映射，必要时把文档存储数据回灌进内存向量库。 |
| `MemoryManager.for_type` | 把枚举或字符串形式的记忆类型规范化，返回对应的记忆实现实例，非法类型统一抛 `ValueError`。 |
| `MemoryManager.add` | 把内容与附加关键字参数转发给指定层（默认工作记忆）的 `add`，写入一条新记忆。 |
| `MemoryManager.get` | 按 id 取单条记忆，未指定类型时先反查文档存储确定所属层，发现已过期则删除并返回 `None`。 |
| `MemoryManager.delete` | 按 id 删除单条记忆，未指定类型时先反查确定所属层，id 不存在时返回 `False`。 |
| `MemoryManager.search` | 语义检索：指定类型则只搜该层，否则四层全搜后按分数降序、同分按创建时间升序合并并截断到 `limit` 条。 |
| `MemoryManager.list` | 列出记忆条目，指定类型则列该层，否则直接列文档存储全量，可选用 `include_expired` 控制是否包含过期条目。 |
| `MemoryManager.clear` | 清空指定层或全部四层记忆，并返回被清除条目的总数。 |
| `MemoryManager.close` | 关闭文档存储，并在向量存储、图存储确实提供可调用 `close` 时一并关闭它们。 |
| `MemoryManager.__enter__` | 上下文管理器入口，原样返回 `self`。 |
| `MemoryManager.__exit__` | 上下文管理器出口，调用 `close()` 释放资源并返回 `None` 以不抑制异常。 |
