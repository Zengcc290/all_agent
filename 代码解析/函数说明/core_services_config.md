# core/services_config.py

## 一、这个文件是干什么的

这个文件是项目里「所有外部 API / 云服务调用」的**集中配置加载器**。它唯一的输入是 `config/services.toml`（私有运行时文件，已被 gitignore；可公开发布的模板是 `config/services.example.toml`），唯一的输出是一个不可变的 `ServicesConfig` 数据对象。

它描述并承载六类外部服务的连接信息：`embedding`（向量化嵌入服务）、`vision`（视觉语言模型，用于图片→实体/关系抽取）、`search`（联网搜索）、`qdrant`（Qdrant Cloud 向量库）、`neo4j`（Neo4j Aura 图数据库）、`proxy`（本地正向代理，例如 Clash 的 7890 端口）。

秘密信息遵循 `provider.toml` 的约定：某个 section 要么直接写明文值（因为文件本身被 gitignore），要么通过 `*_env` 键名指定一个环境变量，由本模块在加载时用 `os.getenv` 解析出来。

优先级被刻意简化为两层：**显式传入的参数 > services.toml**。历史上曾经存在的 `.env` / `HELLOAGENTS_MEMORY_*` 环境变量优先级层已被移除，配置只住在 `config/` 目录和 `constants.py` 里。

整个文件的设计原则是「缺失不报错、写错要报错」：配置文件不存在时返回一个全 None 的空配置，让项目在离线状态下仍能用本地记忆回退、只是没有搜索能力；而 TOML 语法错误、类型错误、非法 URL scheme、非正数等**本地手改坏了**的情况则一律抛异常，尽早暴露。

结构上它由三部分组成：一批 `frozen=True` 的 `@dataclass` 数据类（每个服务一个）、一个聚合类 `ServicesConfig`、以及一组以 `_` 开头的私有解析/校验辅助函数（`_table`、`_parse_*`、`_resolve_secret`、`_optional_string`、`_optional_positive_int`、`_optional_positive_float`、`_require_scheme`），最后用 `__all__` 显式声明对外公开的名字。

---

## 二、函数与类逐条详解

### 模块级常量 `_HTTP_SCHEMES` / `_NEO4J_SCHEMES` / `_EMBEDDING_PROVIDERS` （第 30-32 行）

- **作用**：这三个 `frozenset` 是模块级的校验白名单，供下面的解析函数做合法性判断。`_HTTP_SCHEMES` 只允许 `http` 与 `https`，用于 embedding、search、qdrant、proxy 四个 section 的 URL 校验。`_NEO4J_SCHEMES` 额外允许 `bolt`、`bolt+s`、`neo4j`、`neo4j+s`，因为 Neo4j Aura 用的是加密的 `bolt+s`/`neo4j+s` scheme，而 LLM provider 的解析器是故意拒绝这些 scheme 的，这也正是 Neo4j 配置要单独放一个文件、单独做校验的原因之一。`_EMBEDDING_PROVIDERS` 允许 `auto`、`openai`、`hash` 三个取值，约束 `[embedding].provider` 字段。用 `frozenset` 而不是 `list` 是为了让「是否属于集合」的判断是 O(1)，同时防止运行期被意外修改。
- **参数**：不是函数，无参数。
- **返回**：不是函数，无返回值；它们本身就是被引用的常量集合。
- **内部流程**：模块导入时被创建一次，之后 `_require_scheme` 通过 `in` 做成员测试，`_parse_embedding` 通过 `casefold() not in` 做成员测试。
- **异常/边界**：无特殊处理（纯常量定义，不会抛异常）。
- **同文件关系**：被 `_parse_embedding`、`_parse_search`、`_parse_qdrant`、`_parse_neo4j`、`_parse_proxy` 通过 `_require_scheme` 间接使用；被 `_parse_embedding` 直接使用。不被任何函数调用式地「调用」。

---

### `class EmbeddingService` （第 35-43 行）

- **作用**：这是一个 `@dataclass(frozen=True)`，用来承载向量化嵌入服务的配置。项目要把文本转成向量写进 Qdrant，就需要知道「用哪个 provider、打到哪个 base_url、用哪个 model、带什么 api_key、向量维度多少、一次批量多少条、超时多久」。它本身不含任何行为，只是把 `[embedding]` 这个 TOML section 解析后的结果装起来，供上层的记忆系统读取。因为它是 frozen 的，创建之后不可修改，避免了运行时被别处偷偷改写导致行为不一致。
- **参数**：dataclass 自动生成的 `__init__` 有七个可选参数，全部默认 `None`：`provider: str | None`（嵌入后端类型，合法值被约束为 `auto` / `openai` / `hash`，解析时会被 `casefold()` 转成小写）；`base_url: str | None`（OpenAI 兼容的嵌入接口地址，必须是 http/https 绝对 URL）；`model: str | None`（嵌入模型名，例如某个 text-embedding 模型）；`api_key: str | None`（调用凭证，可能来自明文或环境变量解析结果）；`dimension: int | None`（向量维度，必须为正整数）；`batch_size: int | None`（批量条数，必须为正整数）；`timeout: float | None`（请求超时秒数，必须为正数）。
- **返回**：构造时返回 `EmbeddingService` 实例；因为 frozen，实例的字段是只读属性。字段未配置时值为 `None`，表示「交给上层用默认值或走回退逻辑」。
- **内部流程**：由 `@dataclass` 装饰器自动生成 `__init__`、`__repr__`、`__eq__`；`frozen=True` 让 `__setattr__` 被拦截，赋值时抛 `FrozenInstanceError`。类体内只有字段声明，没有自定义方法。
- **异常/边界**：类本身不校验任何值，所以直接手动构造时传非法值（例如 `dimension=-1`）不会报错；校验责任在 `_parse_embedding` 里。用位置参数或关键字参数构造都可以。
- **同文件关系**：被 `_parse_embedding` 构造，被 `ServicesConfig` 作为字段 `embedding` 持有，被 `ServicesConfig.configured` 通过 `__dict__.values()` 遍历，也被 `load_services_config` 间接使用；`__all__` 里导出。

---

### `class VisionService` （第 46-55 行）

- **作用**：这是一个 `@dataclass(frozen=True)`，承载视觉语言模型的配置，用于「图片 → 实体/关系抽取」这条链路。与其它服务不同，它**只保存模型名**：因为端点和凭证都来自当前生效的 `config/provider.toml` profile（一个 OpenAI 兼容的 chat API），所以这里只需要选一个「同一个 provider 上具备视觉能力的模型」就够了。这样设计避免了同一份凭证在两个文件里重复维护、互相打架。
- **参数**：dataclass 自动生成的 `__init__` 只有一个可选参数 `model: str | None`，默认 `None`，表示要使用的视觉模型名称；未配置时上层应当回退到默认模型或不启用视觉抽取。
- **返回**：构造时返回 `VisionService` 实例；`model` 未配置时为 `None`。
- **内部流程**：类体内有 docstring 说明设计意图，随后只声明 `model` 字段；`@dataclass` 自动生成 `__init__` / `__repr__` / `__eq__`，`frozen=True` 使其不可变。
- **异常/边界**：类本身不校验；由 `_parse_vision` 保证只会塞进 `str | None`。无特殊异常处理。
- **同文件关系**：被 `_parse_vision` 构造，被 `ServicesConfig` 作为字段 `vision` 持有，被 `ServicesConfig.configured` 遍历；`__all__` 里导出。

---

### `class SearchService` （第 58-62 行）

- **作用**：这是一个 `@dataclass(frozen=True)`，承载联网搜索服务的配置。项目在需要「查网上的实时信息」时会用它拿到的 `base_url` 和 `api_key` 去请求搜索接口，并用 `timeout` 控制单次请求的最长等待时间。它不保存搜索引擎的类型或结果条数等策略参数，只保存「打到哪、用什么凭证、等多久」这三件连接层的事。
- **参数**：dataclass 自动生成的 `__init__` 有三个可选参数，默认均为 `None`：`base_url: str | None`（搜索接口的绝对 URL，必须是 http/https）；`api_key: str | None`（调用凭证）；`timeout: float | None`（超时秒数，必须为正数）。
- **返回**：构造时返回 `SearchService` 实例；字段未配置时为 `None`。
- **内部流程**：类体只声明三个字段，`@dataclass` 自动生成 `__init__` / `__repr__` / `__eq__`，`frozen=True` 保证不可变。
- **异常/边界**：类本身不校验；非法 scheme、非正超时由 `_parse_search` 在构造前拦下。无特殊异常处理。
- **同文件关系**：被 `_parse_search` 构造，被 `ServicesConfig` 作为字段 `search` 持有，被 `ServicesConfig.configured` 遍历；`__all__` 里导出。

---

### `class QdrantService` （第 65-69 行）

- **作用**：这是一个 `@dataclass(frozen=True)`，承载 Qdrant Cloud 向量数据库的配置。四层记忆系统里的语义记忆需要把向量存进 Qdrant 并在检索时查询，因此需要集群 URL、API key 以及要操作的 collection 名字。把 collection 也放在配置里，是为了让同一份代码可以指向不同的集合（例如不同实验、不同环境）而不用改代码。
- **参数**：dataclass 自动生成的 `__init__` 有三个可选参数，默认均为 `None`：`url: str | None`（Qdrant 集群的绝对 URL，必须是 http/https，解析时由 `_require_scheme` 用 `_HTTP_SCHEMES` 校验）；`api_key: str | None`（访问凭证）；`collection: str | None`（集合名称）。
- **返回**：构造时返回 `QdrantService` 实例；未配置字段为 `None`。
- **内部流程**：类体只声明三个字段；`@dataclass` 自动生成 `__init__` / `__repr__` / `__eq__`，`frozen=True` 保证不可变。
- **异常/边界**：类本身不校验；URL 与凭证的合法性由 `_parse_qdrant` 负责。无特殊异常处理。
- **同文件关系**：被 `_parse_qdrant` 构造，被 `ServicesConfig` 作为字段 `qdrant` 持有，被 `ServicesConfig.configured` 遍历；`__all__` 里导出。

---

### `class Neo4jService` （第 72-76 行）

- **作用**：这是一个 `@dataclass(frozen=True)`，承载 Neo4j Aura 图数据库的配置。记忆系统里的实体-关系图谱要落到 Neo4j，所以需要 `uri`（连接串）、`username`、`password`。Neo4j 的 URI 支持 `bolt` / `bolt+s` / `neo4j` / `neo4j+s` 等专用 scheme，这一点和其它 HTTP 服务完全不同，所以它独立成一个类、并用 `_NEO4J_SCHEMES` 单独校验。
- **参数**：dataclass 自动生成的 `__init__` 有三个可选参数，默认均为 `None`：`uri: str | None`（Neo4j 连接 URI，scheme 必须在 `_NEO4J_SCHEMES` 内且必须有 netloc）；`username: str | None`（用户名）；`password: str | None`（密码，可以通过明文 `password` 或环境变量名 `password_env` 两种方式提供）。
- **返回**：构造时返回 `Neo4jService` 实例；未配置字段为 `None`。
- **内部流程**：类体只声明三个字段；`@dataclass` 自动生成 `__init__` / `__repr__` / `__eq__`，`frozen=True` 保证不可变。
- **异常/边界**：类本身不校验；`uri` 的 scheme 与绝对性由 `_parse_neo4j` 校验，密码由 `_resolve_secret` 以「明文优先、其次环境变量」的方式解析。无特殊异常处理。
- **同文件关系**：被 `_parse_neo4j` 构造，被 `ServicesConfig` 作为字段 `neo4j` 持有，被 `ServicesConfig.configured` 遍历；`__all__` 里导出。

---

### `class ProxyService` （第 79-88 行）

- **作用**：这是一个 `@dataclass(frozen=True)`，承载本地正向代理的配置（例如运行在 7890 端口的 Clash）。之所以需要它，是因为在国内网络环境下访问 Qdrant Cloud、Neo4j Aura 等境外服务往往必须走代理。`url` 是一个 `http://host:port` 形式的 CONNECT 代理地址。Neo4j Aura 之所以也要走它，是因为 Neo4j 官方驱动本身**没有原生的代理支持**，只能通过隧道方式绕过去；而 Qdrant Cloud 则是把这个地址直接透传给底层 HTTP 客户端。
- **参数**：dataclass 自动生成的 `__init__` 只有一个可选参数 `url: str | None`，默认 `None`，表示代理地址；必须是带 netloc 的 http/https 绝对 URL（由 `_parse_proxy` 用 `_HTTP_SCHEMES` 校验）。
- **返回**：构造时返回 `ProxyService` 实例；未配置时 `url` 为 `None`，表示不使用代理。
- **内部流程**：类体内含说明代理用途与两种消费方式的 docstring，随后只声明 `url` 字段；`@dataclass` 自动生成 `__init__` / `__repr__` / `__eq__`，`frozen=True` 保证不可变。
- **异常/边界**：类本身不校验；URL 合法性由 `_parse_proxy` 负责。无特殊异常处理。
- **同文件关系**：被 `_parse_proxy` 构造，被 `ServicesConfig` 作为字段 `proxy` 持有，被 `ServicesConfig.configured` 遍历；`__all__` 里导出。

---

### `class ServicesConfig` （第 91-100 行）

- **作用**：这是整个模块的核心聚合类型，是一个 `@dataclass(frozen=True)`，代表「对 `config/services.toml` 的一份完整解析视图」。它把六个服务配置对象组合在一起，作为加载结果在项目里传递。它的 docstring 明确说明：缺失的 section 会全部变成 None 字段，也就是说即使 TOML 里什么都没写，也能得到一个字段全为 `None` 的合法对象，调用方只需要判断字段是否为 `None`，不用到处写「配置是否存在」的防御代码。用 frozen dataclass 保证这份配置在进程内被多处读取时不会被某一方篡改。
- **参数**：dataclass 自动生成的 `__init__` 有六个可选参数，默认值都是各自数据类的空实例（这是 dataclass 允许的写法，因为这些类本身也是 frozen 的、可以作为不可变默认值）：`embedding: EmbeddingService = EmbeddingService()`、`vision: VisionService = VisionService()`、`search: SearchService = SearchService()`、`qdrant: QdrantService = QdrantService()`、`neo4j: Neo4jService = Neo4jService()`、`proxy: ProxyService = ProxyService()`。每个参数类型就是对应的服务配置类。
- **返回**：构造时返回 `ServicesConfig` 实例；字段可以通过属性读取（例如 `config.embedding.model`）。
- **内部流程**：类体内声明六个带默认值的字段，随后定义了一个 `@property` 方法 `configured`（见下一条）。`@dataclass` 自动生成 `__init__` / `__repr__` / `__eq__`，`frozen=True` 使其实例不可变。
- **异常/边界**：类本身不做校验；不过因为六个字段的默认值是在类定义时就求值的空实例，所有未显式传入的字段都会共享同一个空实例对象——由于这些实例是 frozen 且无状态，共享是安全的。无特殊异常处理。
- **同文件关系**：被 `load_services_config` 构造（传入六个 `_parse_*` 的结果），被 `ServicesConfig.configured` 读取自身字段；`__all__` 里导出。

---

### `ServicesConfig.configured` （第 102-110 行）

- **作用**：这是一个只读的 `@property`，用来回答「这份配置到底有没有被真正使用/填写过」这个问题。当 `config/services.toml` 不存在，或者存在但六个 section 全是空占位符时，加载结果是所有字段都为 `None`，此时 `configured` 返回 `False`，上层就可以据此判断「当前是离线/无外部服务的运行模式」，从而启用本地回退逻辑（本地记忆回退、不联网搜索）而不是去尝试连接一个根本不存在的服务。它相当于给调用方一个廉价的「配置是否为空」的总开关判断。
- **参数**：无参数（只有 `self`，通过属性访问触发，例如 `config.configured`）。
- **返回**：返回 `bool`。只要六个 section 中任意一个 section 的任意一个字段不是 `None`，就返回 `True`；所有字段全为 `None` 时返回 `False`。
- **内部流程**：返回一个 `any(...)` 表达式的结果，内部是一个双层生成器表达式：外层遍历 `(self.embedding, self.vision, self.search, self.qdrant, self.neo4j, self.proxy)` 这六个 section 对象，内层用 `section.__dict__.values()` 取出每个 dataclass 实例的所有字段值，然后对每个值判断 `value is not None`。因为是 `any` 的短路求值，一旦发现第一个非 None 值就立刻返回 `True`，不会继续遍历后面的 section。注意这里用的是 `__dict__`，所以是逐字段扫描，而不是判断 section 对象本身是否为 None（section 对象永远不是 None，因为默认值是空实例）。
- **异常/边界**：无特殊处理。生成器表达式对空集合是安全的；即使某个 section 一个字段都没有（当前六个类都至少有一个字段，不会发生），`any` 也会返回 `False`。不会抛异常。
- **同文件关系**：它读取 `ServicesConfig` 自身的六个字段（`embedding` / `vision` / `search` / `qdrant` / `neo4j` / `proxy`），以及这些类（`EmbeddingService`、`VisionService`、`SearchService`、`QdrantService`、`Neo4jService`、`ProxyService`）的实例属性；本文件内没有其它函数调用它，它是给外部调用方用的公开接口。

---

### `default_config_path() -> Path` （第 113-121 行）

- **作用**：这个函数负责回答「默认该去哪个文件读服务配置」。它的策略是**优先使用私有的运行时文件 `services.toml`，找不到就退回到可公开发布的模板 `services.example.toml`**。这样设计的好处是：开发者在本地把真实凭证填进 `services.toml`（被 gitignore，不会泄露），而仓库里保留一份模板供他人参考；即使用户没有创建私有文件，只 clone 了仓库，程序也仍能读到模板并正常启动（模板里的空占位符会被解析成 None，不会崩）。
- **参数**：无参数。
- **返回**：返回 `pathlib.Path`。具体地：先在 `config` 目录里按顺序检查 `services.toml`、`services.example.toml`，返回第一个 `is_file()` 为真的路径；如果两个都不存在，则返回 `config_dir / "services.toml"` 这个（不存在的）路径，让后续逻辑按「文件不存在」处理。
- **内部流程**：第一步用 `Path(__file__).resolve().parent.parent / "config"` 定位 config 目录——`__file__` 是 `core/services_config.py`，`resolve()` 拿到绝对路径，`parent` 是 `core/`，再 `parent` 是项目根，然后拼上 `config`，所以不依赖当前工作目录，无论从哪里启动都对。第二步用 `for filename in ("services.toml", "services.example.toml")` 按优先级遍历，对每个候选路径做 `candidate.is_file()` 检查，命中就 `return`。第三步是循环结束后的兜底 `return`。
- **异常/边界**：无特殊异常处理。它只做路径拼接和存在性判断，不会因为文件缺失而抛错——缺失情况通过返回一个不存在的路径来表达，由 `load_services_config` 决定如何处理。若 `config` 目录本身不存在，`is_file()` 返回 `False`，走兜底返回。权限异常（如目录不可读）会由 `Path.is_file()` 传播为 `OSError`，本函数不做捕获。
- **同文件关系**：被 `load_services_config` 调用（当 `path` 参数为 `None` 时作为默认路径）；它自身不调用本文件里的任何其它函数。`__all__` 里导出。

---

### `load_services_config(path: str | Path | None = None) -> ServicesConfig` （第 124-149 行）

- **作用**：这是本模块对外的**主入口函数**，负责把磁盘上的 TOML 文件读进来、解析成 `ServicesConfig` 对象。它是整个「服务配置」能力的唯一加载点，项目启动时（例如 `python -m web.app` 的初始化阶段）会调用它一次，把结果传给嵌入、视觉、搜索、Qdrant、Neo4j、代理等各个消费方。它的一个重要设计承诺是：**文件不存在绝不抛异常**，因为项目必须在没有任何外部服务配置文件的情况下依然可用（走离线记忆回退、不提供搜索）；但**格式错误的 TOML 会抛异常**，因为那说明本地编辑被改坏了，属于必须立刻暴露的错误。
- **参数**：`path: str | Path | None = None`。含义是配置文件路径，默认 `None`。取值约束：传 `None` 时使用 `default_config_path()` 的结果（优先 `config/services.toml`，其次模板）；传字符串或 `Path` 时按该路径读取，会先经过 `Path(...).expanduser()` 展开 `~`，再 `.resolve()` 转成绝对路径（因此相对路径是相对当前工作目录解析的）。没有其它取值限制。
- **返回**：返回 `ServicesConfig` 实例。三种情况：文件不存在 → 返回默认构造的 `ServicesConfig()`（六个 section 全为 None 字段）；文件存在且合法 → 返回由六个 `_parse_*` 结果拼装的 `ServicesConfig`；文件存在但解析失败 → 不返回，直接抛异常。
- **内部流程**：第一步按上面的规则算出 `config_path`。第二步用 `config_path.is_file()` 判断文件是否存在，不存在就立即返回空配置。第三步在 `try` 里用 `config_path.open("rb")` 以**二进制模式**打开文件（`tomllib.load` 要求二进制句柄），调用 `tomllib.load(handle)` 得到字典 `document`；`tomllib.TOMLDecodeError` 会被 `except` 捕获，并重新抛出为 `ValueError("invalid services TOML: <路径>")`，用 `from exc` 保留原始异常链。第四步做一次防御性类型检查：如果 `document` 不是 `dict`（理论上 `tomllib.load` 顶层总是 dict，这里属于兜底），抛 `TypeError`。第五步构造并返回 `ServicesConfig`，六个字段分别由 `_parse_embedding(_table(document, "embedding"))`、`_parse_vision(_table(document, "vision"))`、`_parse_search(_table(document, "search"))`、`_parse_qdrant(_table(document, "qdrant"))`、`_parse_neo4j(_table(document, "neo4j"))`、`_parse_proxy(_table(document, "proxy"))` 产生——注意 `_table` 负责取出同名子表并在类型不对时报错，`_parse_*` 负责逐字段解析校验。
- **异常/边界**：文件缺失 → 不报错，返回空配置；TOML 语法错误 → `ValueError`（由 `TOMLDecodeError` 转换而来，`__cause__` 保留原异常）；顶层不是表 → `TypeError`；某个 section 不是表 → 由 `_table` 抛 `TypeError`；字段值非法（provider 不在白名单、URL scheme 不对、非正数等）→ 由各 `_parse_*` 抛 `ValueError`；文件打开失败（权限、路径是目录等）→ 由 `open()` 抛出的 `OSError` 直接向上传播，本函数不捕获。
- **同文件关系**：它调用了 `default_config_path`、`_table`、`_parse_embedding`、`_parse_vision`、`_parse_search`、`_parse_qdrant`、`_parse_neo4j`、`_parse_proxy`，并构造 `ServicesConfig`；本文件内没有其它函数调用它，它是公开 API，`__all__` 里导出。

---

### `_table(document: dict[str, Any], name: str) -> dict[str, Any]` （第 152-156 行）

- **作用**：这是一个私有小工具，负责从整份 TOML 文档字典里取出指定名字的子表，并保证「取到的东西确实是一张表」。它让 `load_services_config` 里那六行解析代码保持整齐，同时把「section 写成了非表类型」这种错误的检查集中在一处。TOML 里 `[embedding]` 这样写才是表；如果用户误写成 `embedding = "xxx"` 这种标量，`document.get("embedding")` 拿到的就是字符串，这里会立刻报错而不是让后续解析拿到一个莫名其妙的类型。
- **参数**：`document: dict[str, Any]`，整份 TOML 解析后的顶层字典；`name: str`，要取的 section 名字，调用处传入的是 `"embedding"`、`"vision"`、`"search"`、`"qdrant"`、`"neo4j"`、`"proxy"` 六个固定值。
- **返回**：返回 `dict[str, Any]`。如果文档里有同名子表，返回该子表；如果没有（`document.get(name, {})` 落到默认值），返回一个**新的空字典 `{}`**，表示「这个 section 没配置」，后续 `_parse_*` 会把所有字段解析成 None。
- **内部流程**：第一步 `document.get(name, {})` 取值，缺失时用空字典兜底。第二步 `isinstance(table, dict)` 判断类型，不是字典就抛 `TypeError`，错误信息是 `f"[{name}] must be a TOML table"`，明确指出是哪个 section 写错了。第三步返回该字典。
- **异常/边界**：section 缺失 → 返回空字典，不报错（这是「缺失不报错」原则的体现）；section 存在但不是表 → `TypeError`。空表是合法输入，会正常返回。无其它特殊处理。
- **同文件关系**：被 `load_services_config` 调用六次；它自身不调用本文件里的任何其它函数。

---

### `_parse_embedding(table: dict[str, Any]) -> EmbeddingService` （第 159-181 行）

- **作用**：把 `[embedding]` 这张表解析成 `EmbeddingService` 对象，并对每个字段做类型与取值校验。它是六个解析函数里逻辑最复杂的一个，因为它需要处理 provider 白名单校验、base_url 的 URL scheme 校验、api_key 的「明文或环境变量」解析，以及三个正数型数值字段（dimension、batch_size、timeout）的校验。这样上层拿到的 `EmbeddingService` 就是一个已经「洗干净」的对象，不用再自己防御非法值。
- **参数**：`table: dict[str, Any]`，`[embedding]` 子表的内容；如果该 section 未配置，这里是空字典（由 `_table` 保证）。函数会尝试读取 `provider`、`base_url`、`model`、`api_key`（或 `api_key_env`）、`dimension`、`batch_size`、`timeout` 这些键，其余键一律忽略。
- **返回**：返回 `EmbeddingService` 实例。字段规则：`provider` 若配置了会被 `casefold()` 转成小写后存入，未配置则为 `None`；`base_url`、`model`、`api_key` 按解析结果或 `None`；`dimension`、`batch_size` 为 int 或 `None`；`timeout` 为 float 或 `None`。
- **内部流程**：按顺序执行以下步骤。第一步读 `provider = _optional_string(table, "provider")`；若不为 `None` 且 `provider.casefold()` 不在 `_EMBEDDING_PROVIDERS`（`auto` / `openai` / `hash`）里，抛 `ValueError`，错误信息里用 `", ".join(sorted(_EMBEDDING_PROVIDERS))` 列出所有合法取值（排序保证错误信息稳定）。第二步读 `base_url`，不为 `None` 时调用 `_require_scheme("embedding", "base_url", base_url, _HTTP_SCHEMES)` 要求它是 http/https 绝对 URL。第三步读 `model`。第四步 `api_key = _resolve_secret(table)` 用默认参数解析凭证。第五步依次用 `_optional_positive_int(table, "dimension", "[embedding]")`、`_optional_positive_int(table, "batch_size", "[embedding]")`、`_optional_positive_float(table, "timeout", "[embedding]")` 解析三个数值字段，location 参数统一是 `"[embedding]"` 以便错误信息可读。第六步构造 `EmbeddingService`，注意 `provider=provider.casefold() if provider else None`——这里再做一次 casefold 是因为前面那次只用于校验、没有改变量；`if provider` 同时挡住了 `None` 和空字符串（空字符串已经被 `_optional_string` 变成 `None` 了）。
- **异常/边界**：`provider` 非法 → `ValueError`；`base_url` 不是 http/https 绝对 URL（例如写成 `ftp://`、或者只有 `host:port` 没有 scheme）→ `_require_scheme` 抛 `ValueError`；`dimension` / `batch_size` 不是正整数（包括传了 `True`/`False` 布尔值、传了浮点数、传了 0 或负数）→ `_optional_positive_int` 抛 `ValueError`；`timeout` 不是正数 → `_optional_positive_float` 抛 `ValueError`；空白字符串值 → 被 `_optional_string` 视为未配置变成 `None`，不报错；`api_key_env` 指向的环境变量不存在或为空白 → 返回 `None`，不报错。
- **同文件关系**：它调用了 `_optional_string`、`_require_scheme`、`_resolve_secret`、`_optional_positive_int`、`_optional_positive_float`，并直接引用常量 `_EMBEDDING_PROVIDERS` 与 `_HTTP_SCHEMES`，构造 `EmbeddingService`；被 `load_services_config` 调用。本文件内没有其它函数调用它。

---

### `_parse_vision(table: dict[str, Any]) -> VisionService` （第 184-185 行）

- **作用**：把 `[vision]` 这张表解析成 `VisionService`。因为视觉服务的端点和凭证都来自 `provider.toml`，这里只关心一个字段——模型名，所以整个函数就是一行「读 `model` 然后构造对象」。它的存在让六个 section 的解析保持对称，`load_services_config` 不必为 vision 写特殊分支。
- **参数**：`table: dict[str, Any]`，`[vision]` 子表内容；section 未配置时是空字典。只读取 `model` 键，其余键忽略。
- **返回**：返回 `VisionService` 实例，其中 `model` 是去掉首尾空白后的字符串，或者 `None`（未配置或值为空白时）。
- **内部流程**：直接调用 `_optional_string(table, "model")` 取值，并用关键字参数 `model=` 传给 `VisionService(...)` 构造，然后 `return` 该对象。没有任何条件分支或循环。
- **异常/边界**：无特殊处理——`model` 若不是字符串或为空白，`_optional_string` 会静默返回 `None` 而**不报错**（这是六个解析函数里唯一一个不做任何主动校验的，因为它只有一个字符串字段）。不会抛异常。
- **同文件关系**：调用 `_optional_string`，构造 `VisionService`；被 `load_services_config` 调用。

---

### `_parse_search(table: dict[str, Any]) -> SearchService` （第 188-196 行）

- **作用**：把 `[search]` 这张表解析成 `SearchService`，并对搜索接口地址做 URL scheme 校验、对超时做正数校验、对凭证做「明文或环境变量」解析。搜索是联网能力，地址写错会导致运行时请求失败，所以在加载阶段就把它拦下来，比等到真正搜索时才报错要好得多。
- **参数**：`table: dict[str, Any]`，`[search]` 子表内容；section 未配置时是空字典。读取 `base_url`、`api_key` / `api_key_env`、`timeout` 三个键，其余忽略。
- **返回**：返回 `SearchService` 实例；`base_url` 为字符串或 `None`，`api_key` 为字符串或 `None`，`timeout` 为 float 或 `None`。
- **内部流程**：第一步 `base_url = _optional_string(table, "base_url")`。第二步若 `base_url is not None`，调用 `_require_scheme("search", "base_url", base_url, _HTTP_SCHEMES)` 校验它是 http/https 绝对 URL。第三步构造 `SearchService`，其中 `base_url` 用已校验的值，`api_key` 由 `_resolve_secret(table)` 用默认键名解析，`timeout` 由 `_optional_positive_float(table, "timeout", "[search]")` 解析（location 为 `"[search]"`，用于错误信息）。
- **异常/边界**：`base_url` scheme 不合法或缺少 netloc → `ValueError`（由 `_require_scheme` 抛出）；`timeout` 不是正数（含布尔值、0、负数）→ `ValueError`；`base_url` / `api_key` 为空白字符串 → 视为未配置，返回 `None`；`api_key_env` 指向的环境变量缺失 → 返回 `None`。不会因字段缺失而报错。
- **同文件关系**：调用 `_optional_string`、`_require_scheme`、`_resolve_secret`、`_optional_positive_float`，引用 `_HTTP_SCHEMES`，构造 `SearchService`；被 `load_services_config` 调用。

---

### `_parse_qdrant(table: dict[str, Any]) -> QdrantService` （第 199-207 行）

- **作用**：把 `[qdrant]` 这张表解析成 `QdrantService`。它校验集群 URL 必须是 http/https 绝对地址（Qdrant Cloud 走的是 HTTPS REST/gRPC 网关，所以不接受 bolt 之类的 scheme），解析 API key，并读取 collection 名。有了它，向量记忆模块拿到的就是一个可直接用于建连的配置对象。
- **参数**：`table: dict[str, Any]`，`[qdrant]` 子表内容；section 未配置时是空字典。读取 `url`、`api_key` / `api_key_env`、`collection` 三个键，其余忽略。
- **返回**：返回 `QdrantService` 实例；`url` 为字符串或 `None`，`api_key` 为字符串或 `None`，`collection` 为字符串或 `None`。
- **内部流程**：第一步 `url = _optional_string(table, "url")`。第二步若 `url is not None`，调用 `_require_scheme("qdrant", "url", url, _HTTP_SCHEMES)` 校验（注意这里的 location 是 `"qdrant"`、key 是 `"url"`，与其它 section 用的 `base_url` 不同，错误信息会写成 `[qdrant].url`）。第三步构造 `QdrantService`：`url` 用已校验值，`api_key` 由 `_resolve_secret(table)` 用默认键名解析，`collection` 由 `_optional_string(table, "collection")` 取值。
- **异常/边界**：`url` scheme 不在 `{http, https}` 内或缺少 netloc → `ValueError`；`api_key` / `collection` 为空白 → 变成 `None`；`api_key_env` 指向的环境变量不存在或为空白 → `None`。字段缺失不报错。
- **同文件关系**：调用 `_optional_string`、`_require_scheme`、`_resolve_secret`，引用 `_HTTP_SCHEMES`，构造 `QdrantService`；被 `load_services_config` 调用。

---

### `_parse_neo4j(table: dict[str, Any]) -> Neo4jService` （第 210-218 行）

- **作用**：把 `[neo4j]` 这张表解析成 `Neo4jService`。与其它 HTTP 服务最大的区别在于它的 scheme 白名单是 `_NEO4J_SCHEMES`，额外接受 `bolt`、`bolt+s`、`neo4j`、`neo4j+s`——因为 Neo4j Aura 实际使用的就是加密的 `neo4j+s` / `bolt+s` 连接串，而项目里 LLM provider 的解析器是明确拒绝这些 scheme 的，这正是 Neo4j 配置要独立成文件、独立校验的原因。另外它的凭证字段叫 `password` 而不是 `api_key`，环境变量键名对应 `password_env`，所以调用 `_resolve_secret` 时必须显式覆盖这两个参数。
- **参数**：`table: dict[str, Any]`，`[neo4j]` 子表内容；section 未配置时是空字典。读取 `uri`、`username`、`password` / `password_env` 三个键（环境变量键名是 `password_env`），其余忽略。
- **返回**：返回 `Neo4jService` 实例；`uri`、`username`、`password` 分别为字符串或 `None`。
- **内部流程**：第一步 `uri = _optional_string(table, "uri")`。第二步若 `uri is not None`，调用 `_require_scheme("neo4j", "uri", uri, _NEO4J_SCHEMES)` 校验 scheme 在 Neo4j 白名单内且存在 netloc。第三步构造 `Neo4jService`：`uri` 用已校验值，`username` 由 `_optional_string(table, "username")` 取值，`password` 由 `_resolve_secret(table, key="password", env_key="password_env")` 解析——这里用关键字参数把默认的 `api_key` / `api_key_env` 换成了 Neo4j 专用的键名。
- **异常/边界**：`uri` scheme 不在 `_NEO4J_SCHEMES` 内或缺少 netloc → `ValueError`；`username`、`password` 为空白 → `None`；`password_env` 指向的环境变量不存在或为空白 → `None`；字段缺失不报错。注意明文 `password` 优先于 `password_env`。
- **同文件关系**：调用 `_optional_string`、`_require_scheme`、`_resolve_secret`，引用 `_NEO4J_SCHEMES`，构造 `Neo4jService`；被 `load_services_config` 调用。

---

### `_parse_proxy(table: dict[str, Any]) -> ProxyService` （第 221-225 行）

- **作用**：把 `[proxy]` 这张表解析成 `ProxyService`。它只处理一个字段 `url`，并强制要求它是带 netloc 的 http/https 绝对 URL（因为这是要交给 HTTP 客户端的 CONNECT 代理地址，写成裸 `host:port` 或 `socks5://` 都会在运行期失败）。在加载阶段校验可以避免「配置看起来有值、实际连不通」的隐性故障。
- **参数**：`table: dict[str, Any]`，`[proxy]` 子表内容；section 未配置时是空字典。只读取 `url` 键，其余忽略。
- **返回**：返回 `ProxyService` 实例；`url` 为校验通过的字符串，或 `None`（未配置 / 值为空白）。
- **内部流程**：第一步 `url = _optional_string(table, "url")`。第二步若 `url is not None`，调用 `_require_scheme("proxy", "url", url, _HTTP_SCHEMES)` 校验。第三步 `return ProxyService(url=url)`。
- **异常/边界**：`url` scheme 不合法或缺少 netloc（例如写成 `127.0.0.1:7890`）→ `ValueError`；值为空白字符串 → 视为未配置返回 `None`；字段缺失不报错。
- **同文件关系**：调用 `_optional_string`、`_require_scheme`，引用 `_HTTP_SCHEMES`，构造 `ProxyService`；被 `load_services_config` 调用。

---

### `_resolve_secret(table: dict[str, Any], *, key: str = "api_key", env_key: str = "api_key_env") -> str | None` （第 228-242 行）

- **作用**：这是「秘密解析」的公共逻辑，被 embedding、search、qdrant、neo4j 四个 section 复用。它实现了项目文档里写的那条约定：**某个 section 要么直接写明文值，要么通过 `*_env` 键名指定一个环境变量，由加载时用 `os.getenv` 解析**。这里还明确规定了两者的优先级——明文值优先；只有明文没写时才去看环境变量。这样开发者可以本地直接写明文（文件被 gitignore），也可以把密钥放在环境变量里供 CI/服务器使用，两种方式共用同一份解析代码。
- **参数**：`table: dict[str, Any]`，待解析的 section 表；`key: str = "api_key"`，明文值的键名，关键字限定（`*` 表示只能按关键字传），Neo4j 传 `"password"`；`env_key: str = "api_key_env"`，环境变量名所在键的键名，Neo4j 传 `"password_env"`。两个键名参数都只接受非空字符串（调用处都是字面量）。
- **返回**：返回 `str | None`。明文值存在（去空白后非空）时返回该明文；否则若 `env_key` 指向的环境变量名存在且 `os.getenv` 取到非空白值，返回 `resolved.strip()`；否则返回 `None`。
- **内部流程**：第一步 `value = _optional_string(table, key)` 取明文，如果 `value is not None` 直接 `return value`（明文优先，不再看环境变量）。第二步 `env_name = _optional_string(table, env_key)` 取环境变量名，如果 `env_name is None`（没写或写了空白）就 `return None`。第三步 `resolved = os.getenv(env_name)` 真正去环境里查；第四步判断 `resolved is None or not resolved.strip()`，只要缺失或全是空白就 `return None`；第五步 `return resolved.strip()` 返回去掉首尾空白后的密钥。注意返回值**没有做任何长度、格式或前缀校验**，任何非空字符串都会被原样接受。
- **异常/边界**：`env_name` 不是合法环境变量名不会报错，`os.getenv` 只会返回 `None`，然后走「返回 None」分支；环境变量存在但值为空白（例如 `KEY="   "`）→ 视为未配置返回 `None`；明文值为空白 → 被 `_optional_string` 变成 `None`，于是继续尝试环境变量。函数自身不会抛异常（除非 `table` 不是字典，那会在 `table.get` 处抛 `AttributeError`，但调用链上 `_table` 已保证是字典）。无超时、无 IO。
- **同文件关系**：调用 `_optional_string`；被 `_parse_embedding`、`_parse_search`、`_parse_qdrant`、`_parse_neo4j` 四个解析函数调用（其中 Neo4j 覆盖了 `key` 与 `env_key`）。它不构造任何数据类。

---

### `_optional_string(table: dict[str, Any], key: str) -> str | None` （第 245-253 行）

- **作用**：这是整个模块里被调用次数最多的字符串读取工具。它统一了「可选字符串字段」的语义：字段缺失、值为 `None`、值不是字符串、值是空白字符串，这四种情况一律视为「未配置」并返回 `None`；只有真正有内容的字符串才返回其 `strip()` 后的结果。注释里点明了这条设计的原因——空白值在 `.env` 约定里就代表「没配置」，这样一份只含空占位符的模板文件（`services.example.toml`）永远不会让消费方崩溃。它同时也起到了类型守门的作用：TOML 里若把某个字符串字段写成了整数或数组，这里不会抛错而是静默忽略，保证加载流程不会因为一个可有可无的字段而整体失败。
- **参数**：`table: dict[str, Any]`，待读取的表；`key: str`，要读的键名。没有默认值。
- **返回**：返回 `str | None`。键不存在或值为 `None` → `None`；值不是 `str` 类型 → `None`；值是空字符串或只含空白（`not value.strip()`）→ `None`；其余情况返回 `value.strip()`（已去掉首尾空白）。
- **内部流程**：第一步 `value = table.get(key)`（键不存在时得到 `None`）。第二步 `if value is None: return None`。第三步 `if not isinstance(value, str) or not value.strip(): return None`——用 `or` 短路，非字符串直接返回，字符串则检查去空白后是否为空。第四步 `return value.strip()`。
- **异常/边界**：无特殊处理，也不会抛异常：非字符串、空白、缺失全部被静默转成 `None`。注意它**不**做任何内容校验（不查 scheme、不查白名单），校验是调用方的责任。
- **同文件关系**：被 `_parse_embedding`、`_parse_vision`、`_parse_search`、`_parse_qdrant`、`_parse_neo4j`、`_parse_proxy` 以及 `_resolve_secret` 调用；它自身不调用本文件里的任何其它函数。

---

### `_optional_positive_int(table: dict[str, Any], key: str, location: str) -> int | None` （第 256-262 行）

- **作用**：这是「可选正整数字段」的读取与校验工具，目前用于 `[embedding].dimension` 和 `[embedding].batch_size`。这两个值都必须 ≥ 1：维度为 0 或负数的向量没有意义，批量为 0 会导致一次都不发请求，所以这里不是简单转类型，而是真的做合法性检查并在非法时抛出带位置的错误信息。它同时刻意排除了 `bool` 类型——在 Python 里 `True`/`False` 是 `int` 的子类，如果不显式排除，TOML 里写 `dimension = true` 会被悄悄当成 1 通过校验。
- **参数**：`table: dict[str, Any]`，待读取的表；`key: str`，字段名（如 `"dimension"`、`"batch_size"`）；`location: str`，用于拼错误信息的位置前缀，调用处统一传 `"[embedding]"`，最终错误信息形如 `[embedding].dimension must be a positive integer`。
- **返回**：返回 `int | None`。字段缺失或值为 `None` → 返回 `None`（表示未配置）；值是 ≥ 1 的整数 → 原样返回该整数。
- **内部流程**：第一步 `value = table.get(key)`。第二步 `if value is None: return None`。第三步是一个组合条件：`isinstance(value, bool) or not isinstance(value, int) or value < 1`，任一成立就抛 `ValueError(f"{location}.{key} must be a positive integer")`——先排除布尔值，再排除非整数（例如浮点 `1.5`、字符串 `"5"`），最后排除 0 与负数。第四步 `return value`。
- **异常/边界**：值为布尔 → `ValueError`；值为非 int（float、str、list、dict）→ `ValueError`；值 < 1 → `ValueError`；缺失或 `None` → 返回 `None`，不报错。注意 TOML 里的浮点 `1.0` 也会被拒（因为 `isinstance(1.0, int)` 为假），这是有意的严格。
- **同文件关系**：被 `_parse_embedding` 调用两次（`dimension` 与 `batch_size`）；它自身不调用本文件里的任何其它函数。

---

### `_optional_positive_float(table: dict[str, Any], key: str, location: str) -> float | None` （第 265-271 行）

- **作用**：这是「可选正数字段」的读取与校验工具，用于 `[embedding].timeout` 和 `[search].timeout`。超时必须大于 0，否则意味着立即超时或无限等待（0 或负数在多数 HTTP 客户端里语义混乱甚至报错），所以这里强制 `> 0`。与整数版本不同，它**接受整数也接受浮点**（TOML 里写 `timeout = 30` 和 `timeout = 30.0` 都合法），并把结果统一转成 `float` 返回，让下游拿到类型一致的值。同样地，它显式排除 `bool`，避免 `timeout = true` 被当成 1.0 混过去。
- **参数**：`table: dict[str, Any]`，待读取的表；`key: str`，字段名（如 `"timeout"`）；`location: str`，错误信息的位置前缀，调用处传 `"[embedding]"` 或 `"[search]"`，错误信息形如 `[search].timeout must be a positive number`。
- **返回**：返回 `float | None`。字段缺失或值为 `None` → `None`；值是正数（int 或 float，且 `float(value) > 0`）→ 返回 `float(value)`；其它情况抛异常。
- **内部流程**：第一步 `value = table.get(key)`。第二步 `if value is None: return None`。第三步组合条件 `isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0`，任一成立抛 `ValueError(f"{location}.{key} must be a positive number")`。第四步 `return float(value)`，完成 int → float 的统一转换。
- **异常/边界**：布尔值 → `ValueError`；字符串、列表等非数字 → `ValueError`；0 或负数 → `ValueError`；注意 `float("nan")` 这类值在 TOML 中不以该形式出现，但若通过 Python 直接构造调用，`nan <= 0` 为 `False`，理论上会通过校验——这是唯一的边界缝隙，实际从 TOML 加载不会触发。缺失或 `None` → 返回 `None`。
- **同文件关系**：被 `_parse_embedding`（`timeout`）和 `_parse_search`（`timeout`）调用；它自身不调用本文件里的任何其它函数。

---

### `_require_scheme(location: str, key: str, url: str, schemes: frozenset[str]) -> None` （第 274-278 行）

- **作用**：这是 URL 合法性校验的统一入口，被 embedding、search、qdrant、neo4j、proxy 五个解析函数调用。它做两件事：一是确认 URL 的 scheme 在允许的白名单里，二是确认 URL 有 netloc（即真的有主机名/地址，而不是像 `http:///path` 这种残缺写法）。这两点保证了下游 HTTP/驱动客户端拿到的地址至少形式上可用，把「地址写错」这类问题在加载配置的阶段就暴露出来，而不是等到运行期某个请求失败。
- **参数**：`location: str`，位置名（section 名），用于错误信息，调用处传 `"embedding"`、`"search"`、`"qdrant"`、`"neo4j"`、`"proxy"`；`key: str`，字段名，多为 `"base_url"`，qdrant/neo4j/proxy 传 `"url"` / `"uri"`；`url: str`，待校验的 URL 字符串（调用方已保证非空、已 strip）；`schemes: frozenset[str]`，允许的 scheme 集合，HTTP 类服务传 `_HTTP_SCHEMES`，Neo4j 传 `_NEO4J_SCHEMES`。
- **返回**：没有返回值（返回 `None`）。校验通过时静默返回，让调用方继续构造配置对象。
- **内部流程**：第一步 `parsed = urlsplit(url)` 用标准库 `urllib.parse.urlsplit` 把 URL 拆成 scheme、netloc、path 等部分——注意 `urlsplit` 不会对非法字符报错，只会尽力解析。第二步 `if parsed.scheme not in schemes or not parsed.netloc:` 判断，只要 scheme 不在白名单**或** netloc 为空就进入错误分支。第三步在错误分支里用 `allowed = ", ".join(sorted(schemes))` 把允许的 scheme 排序后拼成逗号分隔字符串（排序是为了让错误信息稳定、便于测试），然后抛 `ValueError(f"[{location}].{key} must be an absolute URL with scheme in: {allowed}")`。
- **异常/边界**：scheme 不合法 → `ValueError`；缺少 netloc（如 `http:///x`、`https://`）→ `ValueError`；大小写方面，`urlsplit` 会把 scheme 规范化为小写，所以写 `HTTP://...` 也能通过校验；函数不校验 host 是否真实存在、端口是否合法范围，也不做网络请求。`url` 为空字符串的情况在调用链上不会出现，因为调用方都先经过 `_optional_string` 把空白转成了 `None` 并跳过校验。
- **同文件关系**：被 `_parse_embedding`、`_parse_search`、`_parse_qdrant`、`_parse_neo4j`、`_parse_proxy` 调用，并配合常量 `_HTTP_SCHEMES`、`_NEO4J_SCHEMES` 使用；它自身不调用本文件里的任何其它函数，只使用标准库 `urlsplit`。

---

### 模块级 `__all__` （第 281-291 行）

- **作用**：显式声明本模块对外公开的名字，包含六个数据类（`EmbeddingService`、`Neo4jService`、`ProxyService`、`QdrantService`、`SearchService`、`ServicesConfig`、`VisionService`，共七个类）以及两个函数（`default_config_path`、`load_services_config`）。它同时起到了「文档」作用：告诉使用者哪些是稳定 API，哪些是内部实现细节——所有 `_parse_*`、`_table`、`_resolve_secret`、`_optional_*`、`_require_scheme` 都不在列表里，说明它们是私有实现，外部不应直接依赖。名字按字母序排列，便于人工核对。
- **参数**：不是函数，无参数。
- **返回**：不是函数，无返回值；它只是一个 `list[str]` 常量，供 `from module import *` 和静态检查工具使用。
- **内部流程**：模块导入时创建该列表，无其它逻辑。
- **异常/边界**：无特殊处理。
- **同文件关系**：列出本文件里定义的所有公开类与函数（`EmbeddingService`、`Neo4jService`、`ProxyService`、`QdrantService`、`SearchService`、`ServicesConfig`、`VisionService`、`default_config_path`、`load_services_config`），不包含任何私有辅助函数。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_HTTP_SCHEMES`（模块常量） | 允许 `http`/`https` 的 scheme 白名单，供 embedding、search、qdrant、proxy 的 URL 校验使用。 |
| `_NEO4J_SCHEMES`（模块常量） | Neo4j 专用 scheme 白名单，额外接受 `bolt`、`bolt+s`、`neo4j`、`neo4j+s`。 |
| `_EMBEDDING_PROVIDERS`（模块常量） | embedding provider 的合法取值集合：`auto`、`openai`、`hash`。 |
| `EmbeddingService` | 承载嵌入服务的 provider、base_url、model、api_key、dimension、batch_size、timeout 的不可变数据类。 |
| `VisionService` | 只承载视觉模型名的不可变数据类，端点与凭证来自 provider.toml。 |
| `SearchService` | 承载联网搜索的 base_url、api_key、timeout 的不可变数据类。 |
| `QdrantService` | 承载 Qdrant Cloud 的 url、api_key、collection 的不可变数据类。 |
| `Neo4jService` | 承载 Neo4j Aura 的 uri、username、password 的不可变数据类。 |
| `ProxyService` | 承载本地正向代理 url 的不可变数据类，供 Neo4j 隧道与 Qdrant HTTP 客户端使用。 |
| `ServicesConfig` | 把六个服务配置聚合成一份完整的 `services.toml` 解析视图，缺失 section 全为 None。 |
| `ServicesConfig.configured` | 只读属性，判断六个 section 中是否有任意字段非 None，即配置是否真的被填写过。 |
| `default_config_path` | 优先返回 `config/services.toml`，否则退回 `config/services.example.toml`，都不存在则返回前者路径。 |
| `load_services_config` | 模块主入口：读取并解析 TOML，文件缺失返回空配置，TOML 语法错误抛 ValueError。 |
| `_table` | 从 TOML 文档中取出指定名字的子表，缺失时返回空字典，类型不对时抛 TypeError。 |
| `_parse_embedding` | 解析 `[embedding]` 表：校验 provider 白名单、URL scheme、正整数与正数，并解析密钥。 |
| `_parse_vision` | 解析 `[vision]` 表，只读取 `model` 一个字段。 |
| `_parse_search` | 解析 `[search]` 表：校验 base_url scheme、解析密钥与正数 timeout。 |
| `_parse_qdrant` | 解析 `[qdrant]` 表：校验 url scheme、解析密钥与 collection。 |
| `_parse_neo4j` | 解析 `[neo4j]` 表：用 Neo4j 专用 scheme 白名单校验 uri，并用 `password`/`password_env` 解析密码。 |
| `_parse_proxy` | 解析 `[proxy]` 表：校验代理 url 必须是 http/https 绝对地址。 |
| `_resolve_secret` | 密钥解析公共逻辑：明文优先，其次按 `*_env` 指定的环境变量取值，都没有则返回 None。 |
| `_optional_string` | 读取可选字符串字段，把缺失、None、非字符串、空白值统一视为未配置返回 None。 |
| `_optional_positive_int` | 读取可选正整数字段，缺失返回 None，布尔/非整数/小于 1 一律抛 ValueError。 |
| `_optional_positive_float` | 读取可选正数字段，缺失返回 None，布尔/非数字/不大于 0 抛 ValueError，结果统一转 float。 |
| `_require_scheme` | 校验 URL 的 scheme 在白名单内且有 netloc，否则抛带位置信息的 ValueError。 |
| `__all__`（模块常量） | 声明对外公开的七个数据类与两个函数，屏蔽所有私有辅助函数。 |
