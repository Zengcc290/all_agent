# tool/search.py

## 一、这个文件是干什么的

这个文件实现的是项目里唯一一个"联网网页搜索"工具 `web.search`，底层对接 AnySearch 提供的公开搜索 API（文档化的入口是 `POST /v1/search`，响应信封把搜索条目放在 `data.results` 里）。它把 AnySearch 特有的请求构造、鉴权头、响应结构差异全部封装在文件内部，对外只暴露项目统一的"单文件工具协议"：一个继承 `core.BaseTool` 的工具类、一份 `ToolSpec` 元数据、以及一个 `create_tool()` 工厂函数供自动发现使用。

文件里主要包含四类东西：一是输入契约 `SearchInput`（Pydantic 模型，同时承担参数校验与"宽松兼容"的归一化职责，因为很多 OpenAI 兼容模型会把可空字段写成 `""`、`"None"`、`"{}"` 这类脏值）；二是输出契约 `SearchItem` / `SearchOutput`（把 provider 返回的任意结构裁剪成稳定的 title/url/snippet 三元组）；三是工具主体 `SearchTool`（构造时从 `config/services.toml` 的 `[search]` 段加载 base_url / api_key / timeout，执行时发 HTTP 请求、解析 JSON、归一化结果）；四是一组模块级私有辅助函数（端点拼接、老式 GET 端点降级、响应体大小限制读取、响应归一化、文本兜底转换）。

运行时的调用链大致是：工具发现机制看到模块级 `TOOL_ENABLED = True` 后调用 `create_tool()` 造出 `SearchTool` 实例；Agent 决定联网搜索时，运行时把模型给出的参数喂给 `SearchInput` 校验，再调用 `SearchTool.execute()`；`execute()` 拼出端点、发请求、读响应、解析 JSON，最后交给 `_normalize_response()` 产出 `SearchOutput`，由框架序列化回给模型。这个文件不落盘、不写数据库，声明 `side_effect="read"`，属于只读型外部访问工具。

## 二、函数与类逐条详解

### `SearchInput`（第 36 行）
- **作用**：这是 `web.search` 工具的入参模型，定义了 AnySearch `/v1/search` 接口能接受的字段集合，同时也是给模型看的 JSON Schema 来源。它存在的意义有两层：第一层是把 `query` / `max_results` / `tag` / `zone` / `language` / `params` / `format` 这些字段的长度、范围、枚举约束写死，避免模型传出越界参数直接打到上游；第二层是充当"脏值清洗层"，因为 OpenAI 兼容模型经常把可空字符串字段填成 `""` 或 `"None"`、把 JSON 对象序列化成字符串，如果不在这里做兼容，请求会因为严格校验失败而白白浪费一次工具调用。它被 `SearchTool.spec` 的 `input_model` 引用，因此运行时在任何搜索发生前都会先用它校验参数。
- **参数**：类本身没有构造参数，字段即参数。`model_config` 设定 `extra="forbid"`（拒绝未知字段）、`strict=True`（严格类型，不做隐式转换）、`populate_by_name=True`（允许用字段名或别名填充）。字段包括：`query: str`（必填，长度 1–500）；`max_results: int`（默认 10，取值 1–10，校验别名 `limit`，序列化别名固定为 `max_results`）；`tag: str | None`（默认 None，若给则长度 1–100）；`zone: Literal["cn","intl"] | None`（默认 None）；`language: str | None`（默认 None，若给则长度 1–32）；`params: dict[str, Any] | None`（默认 None，用 `WithJsonSchema` 覆盖对外 schema 为 `object|null`）；`format: Literal["json","markdown"] | None`（默认 None）。
- **返回**：不返回；实例化成功即得到经过校验与归一化后的参数对象，实例化失败由 Pydantic 抛 `ValidationError`。
- **内部流程**：Pydantic 在构造时先跑三个 `mode="before"` 的字段校验器（`normalize_tag`、`normalize_language`、`normalize_provider_params`）把原始值清洗一遍，然后按字段约束做类型与范围校验，最后跑 `mode="after"` 的 `reject_blank_text` 拒绝纯空白 `query`。类还提供一个只读属性 `limit` 作为历史参数名的向后兼容视图。
- **异常/边界**：字段越界、类型不符、出现未声明字段都会触发 `pydantic.ValidationError`；`query` 为纯空白时由 `reject_blank_text` 抛 `ValueError`（Pydantic 包装成 `ValidationError`）；`params` 传任意非字典字符串时不会被静默接受，而是留给严格校验报错。
- **同文件关系**：它的三个字段校验器调用了模块级 `_normalize_nullable_text()`；它被 `SearchTool.execute()` 做类型断言、被 `_legacy_search_endpoint()` 读取 `query`/`max_results`、被 `_normalize_response()` 使用 `max_results` 做截断。

### `SearchInput.normalize_tag(value) -> Any`（第 97 行）
- **作用**：这是 `tag` 字段的 `mode="before"` 校验器，专门处理"模型把可空字符串字段填坏"的情况。它的核心工作是把 `""`、`"None"`、`"null"`（大小写不敏感、允许首尾空白）统一视作"未提供"，从而让请求体里干脆不出现 `tag` 键。除此之外它还承担一个防呆职责：当模型误把函数名当成能力标签塞进来（`web`、`web.search`、`web__search`），这些值不是 AnySearch 的合法 capability tag，必须一并抹掉，否则上游会因为未知 tag 报错。没有这个校验器，搜索工具在弱模型下会出现大量"参数看起来有值但请求非法"的失败。
- **参数**：`value: Any` —— 模型原始给出的 `tag` 值，可能是 `None`、字符串、数字、列表等任意类型。
- **返回**：返回 `Any`。归一化后为 `None` 时返回 `None`（表示字段缺省）；命中工具名别名集合时也返回 `None`；否则返回去掉首尾空白后的字符串；非字符串类型原样返回，交给后续严格校验去拒绝。
- **内部流程**：先调用 `_normalize_nullable_text(value)` 做通用清洗；若结果为 `None` 直接返回 `None`；随后把结果 `casefold()` 后与集合 `{"web", "web.search", "web__search"}` 比较，命中即返回 `None`；都不命中则返回清洗后的字符串。
- **异常/边界**：自身不抛异常；对 `None`、空串、`"None"`、非字符串类型都安全放行，把类型错误的最终裁决权交给 Pydantic 的 `strict=True`。
- **同文件关系**：调用模块级 `_normalize_nullable_text()`；被 `SearchInput` 在字段校验阶段自动调用，不由其他函数显式调用。

### `SearchInput.normalize_language(value) -> Any`（第 116 行）
- **作用**：这是 `language` 字段的 `mode="before"` 校验器，处理方式比 `tag` 更单纯：只做"空值哨兵"清洗，不额外过滤任何内容。原因是 `language` 是一个自由文本字段，AnySearch 对它的取值没有枚举约束，所以只需要把模型常见的 `""`、`"None"`、`"null"` 转成"未提供"，避免把无意义的空串发给上游。它不承担工具名别名过滤职责，因为语言字段不存在"误填函数名"的典型场景。
- **参数**：`value: Any` —— 模型原始给出的 `language` 值，理论上应为字符串或 `None`。
- **返回**：返回 `Any`。空值哨兵（`None`、空白串、`"none"`/`"null"` 各种大小写）归一化为 `None`；其他字符串返回去除首尾空白后的结果；非字符串原样返回。
- **内部流程**：整个函数体只有一行，直接 `return _normalize_nullable_text(value)`，把所有判断逻辑委托给模块级辅助函数。
- **异常/边界**：自身不抛异常；非字符串值原样透传，由 Pydantic 严格校验决定是否报错。
- **同文件关系**：调用模块级 `_normalize_nullable_text()`；被 `SearchInput` 在字段校验阶段自动调用。

### `SearchInput.normalize_provider_params(value) -> Any`（第 123 行）
- **作用**：这是 `params` 字段的 `mode="before"` 校验器，用来兼容"把 JSON 对象当成字符串返回"的弱工具调用客户端。对外公开契约始终是 `object | null`，但现实中不少模型会把 `{}` 序列化成字符串 `"{}"`，把空值写成 `"None"`。这个校验器就是那道窄缝：只对"能被 `json.loads` 解析成字典的字符串"和"null 哨兵字符串"做宽容处理，其它任意字符串一律原样放行，让严格校验去报错，从而既不削弱契约、又不至于因为一次格式瑕疵整次搜索失败。
- **参数**：`value: Any` —— 模型原始给出的 `params` 值，可能是 `None`、字典、字符串，也可能是别的类型。
- **返回**：返回 `Any`。`value` 为 `None` 或非字符串时原样返回；空串或 `"none"`/`"null"`（大小写不敏感）返回 `None`；能解析成 JSON 且解析结果是 `dict` 的字符串返回该字典；其余情况（解析失败、解析结果不是字典）返回原始字符串。
- **内部流程**：先判断 `value is None or not isinstance(value, str)`，命中则直接返回原值；否则 `strip()` 得到 `normalized`；若为空或属于 `{"none", "null"}` 则返回 `None`；再用 `json.loads` 尝试解析，捕获 `TypeError`/`ValueError` 后返回原字符串；解析成功且 `isinstance(parsed, dict)` 时返回该字典，否则返回原字符串。
- **异常/边界**：内部捕获 `json.loads` 的 `TypeError` 与 `ValueError`，因此不会向外抛解析异常；边界上刻意不把 `"[]"`、`"123"` 这类非对象 JSON 当作有效值，而是交回原始字符串让严格校验拒绝。
- **同文件关系**：只使用标准库 `json`；被 `SearchInput` 在字段校验阶段自动调用，不依赖本文件其他函数。

### `SearchInput.reject_blank_text(value) -> str | None`（第 146 行）
- **作用**：这是 `query` 字段的校验器（默认 `mode="after"`，即类型校验通过之后运行），职责是拒绝"只由空白字符组成"的查询串。`query` 的 `min_length=1` 只能挡住长度为零的字符串，但 `"   "` 这种长度大于零、语义上完全无效的输入会漏过去；如果直接发给 AnySearch，就会变成一次必然失败的无效网络请求。这个校验器把这类输入在本地直接拦下，错误信息明确指向"必须包含非空白字符"。它允许 `None` 通过（因为 `query` 字段本身是必填字符串，`None` 会在更早的类型校验阶段就被拒绝），所以 `None` 分支实际上只是为了签名兼容。
- **参数**：`value: str | None` —— 已经通过类型校验的 `query` 值，正常情况下是非空字符串。
- **返回**：返回 `str | None`。`value` 为 `None` 时原样返回 `None`；`value.strip()` 后非空时返回原始（未去空白的）`value`；`strip()` 后为空时抛异常。
- **内部流程**：单分支判断 `if value is not None and not value.strip():` 命中即 `raise ValueError("text values must contain a non-whitespace character")`，否则 `return value`。
- **异常/边界**：对纯空白（空格、制表符、换行等 `strip()` 可去除的字符）抛 `ValueError`，由 Pydantic 包装成 `ValidationError` 向上传播；`None` 不处理、直接放行。
- **同文件关系**：无对本文件其他函数的调用；被 `SearchInput` 在字段校验阶段自动调用。

### `SearchInput.limit`（property，第 153 行）
- **作用**：这是一个只读属性，为已经被弃用的历史参数名 `limit` 提供向后兼容的读取视图。项目早期的本地搜索工具用 `limit` 表示结果条数，迁移到 AnySearch 后对外字段改名为 `max_results`；为了让仍在读 `arguments.limit` 的旧调用方不炸掉，这里用一个属性把 `max_results` 暴露成 `limit`。它只读不写，说明"写入"路径已经彻底统一到 `max_results`（写入兼容由字段的 `validation_alias=AliasChoices("max_results", "limit")` 承担）。
- **参数**：无（除 `self` 外没有参数）。
- **返回**：返回 `int`，即 `self.max_results` 的当前值，取值范围 1–10。
- **内部流程**：函数体只有 `return self.max_results` 一行，没有任何判断或副作用。
- **异常/边界**：无特殊处理；因为 `max_results` 是必填字段，属性访问不会遇到缺失值。
- **同文件关系**：无调用关系；它读取的是 `SearchInput` 自身字段，本文件其他函数未使用该属性。

### `SearchItem`（第 160 行）
- **作用**：这是单条搜索结果的输出模型，定义了对外稳定暴露的三个字段：`title`、`url`、`snippet`。AnySearch 上游的每条结果可能带有几十个字段（站点名、发布时间、打分、原始 HTML 等），如果原样透传给模型，既浪费 token 又会让模型分心；`SearchItem` 通过 `extra="forbid"` 把结构锁死成三项，保证无论上游怎么演进，Agent 看到的结果形状都不变。三个字段都给了默认空串，意味着即便上游某条结果缺标题或缺链接，也不会导致整批结果校验失败，只会得到一个空字段。它被 `SearchOutput.items` 引用，在 `_normalize_response()` 里逐条构造。
- **参数**：无构造参数，字段即参数。`title: str`（默认 `""`，最大长度 1000）、`url: str`（默认 `""`，最大长度 2000）、`snippet: str`（默认 `""`，最大长度 5000）。`model_config` 为 `extra="forbid"`、`strict=True`。
- **返回**：不返回；构造成功得到一个结果条目对象。
- **内部流程**：`_normalize_response()` 在循环里对每个原始条目取 `title`、`url`（回退 `link`）、`snippet`（回退 `description`），经 `_text()` 转成字符串并按字段长度上限切片后，作为关键字参数构造本类。
- **异常/边界**：字段超长会被 Pydantic 拒绝，所以构造方必须自己先切片（`_normalize_response()` 确实做了 `[:1000]` / `[:2000]` / `[:5000]`）；传入额外字段会触发 `ValidationError`；`strict=True` 意味着传 `None` 或数字给 `title` 会直接报错。
- **同文件关系**：被 `_normalize_response()` 构造、被 `SearchOutput` 作为列表元素类型引用；本文件其他函数不直接使用。

### `SearchOutput`（第 170 行）
- **作用**：这是 `web.search` 工具的整体返回契约，结构非常简单：一个 `items` 列表，元素是 `SearchItem`。它存在的意义是给运行时的输出校验与序列化提供唯一的形状定义，同时用一个 `results` 属性兼容 AnySearch 文档中"结果叫 results"的术语习惯。所有搜索路径（正常 POST、501 降级的 GET、上游返回老式顶层 `results`/`items`）最终都会被 `_normalize_response()` 收敛成本类的实例，因此模型侧永远只看到一种结构。
- **参数**：无构造参数，字段即参数。`items: list[SearchItem]`（必填，无默认值）。`model_config` 为 `extra="forbid"`、`strict=True`。
- **返回**：不返回；构造成功得到归一化输出对象。
- **内部流程**：`SearchTool.execute()` 的最后一步是 `return _normalize_response(payload, arguments.max_results)`，由该函数构造并返回本类实例；本类自身除属性 `results` 外没有额外逻辑。
- **异常/边界**：`items` 缺失或不是列表会触发 `ValidationError`；列表为空是合法状态（表示搜索成功但无结果）；`extra="forbid"` 保证上游多余字段不会渗入输出。
- **同文件关系**：被 `_normalize_response()` 构造并作为 `SearchTool.execute()` 的返回类型标注；`results` 属性读取自身 `items`。

### `SearchOutput.results`（property，第 177 行）
- **作用**：这是一个只读属性，把 `items` 以 AnySearch 的术语 `results` 再暴露一次。AnySearch 的响应信封里结果字段就叫 `data.results`，项目内部统一叫 `items`；当有代码（例如调试脚本、日志打印、旧适配层）按 provider 术语去读 `.results` 时，这个属性避免它们失败。它不参与序列化，纯粹是读取别名。
- **参数**：无。
- **返回**：返回 `list[SearchItem]`，即 `self.items` 本身（同一个列表对象，不是副本）。
- **内部流程**：函数体只有 `return self.items`。
- **异常/边界**：无特殊处理；因为 `items` 是必填字段，访问不会缺失。返回的是引用，调用方若原地修改列表会影响原对象。
- **同文件关系**：无调用关系；被本文件外部代码使用，`_normalize_response()` 未使用它（那里直接构造 `SearchOutput(items=...)`）。

### `SearchTool`（第 184 行）
- **作用**：这是工具的实体类，继承 `core.BaseTool`，负责真正向 AnySearch 发起带鉴权的搜索请求。类上挂着一个类属性 `spec`（`ToolSpec` 实例），声明了工具名 `web.search`、版本 `1.1`、入参/出参模型、`side_effect="read"`、30 秒超时、幂等、可并行、最大并发 8、标签 `("web","search")`、建议前置工具 `system.current_time`，以及一段中文使用指引（只在联网模式可用、本地知识优先、引用要给标题与链接）。工具发现机制正是靠这个 `spec` 和模块级 `TOOL_ENABLED` 把它注册进运行时。实例化时它从 `config/services.toml` 的 `[search]` 段读取 base_url、api_key、timeout，并把最终 timeout 回写到实例自己的 `spec` 上。
- **参数**：无构造参数（字段即参数，见 `__init__`）。类属性 `spec` 是关键配置载体，权限元数据 `permissions=()` 被刻意留空，表示当前公开工具部署不做权限过滤。
- **返回**：不返回；由 `create_tool()` 构造实例供框架使用。
- **内部流程**：类体只包含 `spec` 声明与两个方法 `__init__`、`execute`；`__init__` 完成配置加载与校验，`execute` 完成一次完整搜索调用。
- **异常/边界**：类本身不抛异常；配置缺失或非法的处理集中在 `__init__`，运行期错误集中在 `execute`。
- **同文件关系**：引用 `SearchInput`、`SearchOutput` 作为 spec 的输入输出模型；`__init__` 调用 `core.services_config.load_services_config()`；`execute` 调用 `_search_endpoint()`、`_read_response_body()`、`_legacy_search_endpoint()`、`_normalize_response()`；被 `create_tool()` 构造。

### `SearchTool.__init__(base_url=None, api_key=None, *, timeout=None) -> None`（第 212 行）
- **作用**：这是工具的构造函数，做三件事：加载配置、补齐缺省值、校验参数合法性。它刻意把配置读取放在构造阶段而不是模块导入阶段，注释写得很明确——这样工具发现（import 模块）就不会带上配置读取与 I/O 副作用，只有真正要造实例时才碰配置文件。配置来源是唯一的：`config/services.toml` 的 `[search]` 段，历史上通过 `SEARCH_*` / `ANYSEARCH_*` 环境变量注入的入口已经被删除。构造完成后，它把最终生效的 timeout 通过 `dataclasses.replace` 回写到实例的 `spec.timeout_seconds`，让执行管理器与 urllib 使用同一个截止时间。
- **参数**：`base_url: str | None = None`（AnySearch 服务基址，形如 `https://host` 或 `https://host/v1`，为 None 时从配置读取）；`api_key: str | None = None`（Bearer 令牌，为 None 时从配置读取）；仅关键字参数 `timeout: float | None = None`（请求超时秒数，为 None 时从配置读取，配置里也没有则回退 30.0）。
- **返回**：返回 `None`；副作用是给实例设置 `base_url`、`api_key`、`timeout` 三个属性并替换 `self.spec`。
- **内部流程**：第一步判断 `base_url is None or api_key is None or timeout is None`，只要有一个没给就调用 `load_services_config().search` 拿到 services 对象，否则 services 置为 `None`（避免无谓的配置 I/O）；第二步分别用 services 的 `base_url`、`api_key`、`timeout` 填补三个仍为 None 的参数，timeout 的兜底是 30.0；第三步做类型与取值校验——`base_url` 必须是 `str`（否则 `TypeError`），且 `strip()` 后不能为空（否则 `ValueError`），`api_key` 必须是 `str`（否则 `TypeError`）并 `strip()`；第四步校验 timeout：布尔值、非 `int/float`、非有限值（`math.isfinite` 为假，即 NaN/Inf）、小于等于 0 全部判为非法并抛 `ValueError`，合法则 `float(timeout)` 存回；最后用 `replace(type(self).spec, timeout_seconds=self.timeout)` 生成新的 spec，保证子类重写 spec 时也被正确继承。
- **异常/边界**：`base_url` 非字符串抛 `TypeError`；`base_url` 为空串或纯空白抛 `ValueError`；`api_key` 非字符串抛 `TypeError`（但空串是允许的，因为空 api_key 会在 `execute()` 阶段被判定为"未配置"）；timeout 非法抛 `ValueError`。特别注意 `isinstance(timeout, bool)` 被显式排除，因为 Python 里 `True` 也是 `int`。配置项本身缺失不会在这里报错，而是留给 `execute()` 抛 `RuntimeError`。
- **同文件关系**：调用 `core.services_config.load_services_config()`；使用标准库 `math.isfinite` 与 `dataclasses.replace`；读取类属性 `SearchTool.spec`；被 `create_tool()` 调用（`SearchTool()` 无参形式）。

### `SearchTool.execute(arguments) -> SearchOutput`（第 259 行）
- **作用**：这是工具的执行入口，一次调用完成"校验参数 → 拼端点 → 组装请求体 → 序列化 → 发 POST → 读响应 → 解析 JSON → 归一化结果"的全流程。它是运行时真正被调度的函数，`side_effect="read"` 与 `idempotent=True` 的声明都由它的行为背书：它只发一次带鉴权的读请求，不修改任何服务端状态。请求体只包含非 None 的可选字段，这样上游不会因为收到一堆 null 而困惑；序列化时强制 `ensure_ascii=False`（中文查询不转义）、紧凑分隔符、`allow_nan=False`（拒绝 NaN/Infinity 这类非法 JSON 字面量）。解析阶段特意传入 `parse_constant=reject_json_constant`，防止上游返回 `NaN` 被 Python 的宽松解析悄悄接受。
- **参数**：`arguments: SearchInput` —— 已经过校验的入参对象，必须是 `SearchInput` 实例，否则抛 `TypeError`。字段含义见 `SearchInput`：`query` 必填，`max_results` 1–10，其余可选。
- **返回**：返回 `SearchOutput` 实例，内含最多 `arguments.max_results` 条 `SearchItem`（每条含 title/url/snippet）。若上游成功但没有结果，返回 `SearchOutput(items=[])`。
- **内部流程**：①`isinstance` 断言参数类型；②检查 `self.base_url` 与 `self.api_key` 是否已配置，缺失即抛 `RuntimeError` 并指明配置文件位置；③`_search_endpoint(self.base_url)` 得到规范化 POST 端点；④构造 `request_body` 字典，先放必填的 `query` 与 `max_results`，再遍历 `("tag","zone","language","params","format")`，用 `getattr` 取值、只把非 None 的塞进去；⑤`json.dumps(..., ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")` 得到字节体，捕获 `TypeError`/`ValueError` 转成"参数必须可 JSON 序列化"的 `ValueError`；⑥构造 `urllib.request.Request`，POST，带 `Authorization: Bearer <api_key>`、`Content-Type: application/json`、`Accept: application/json`；⑦`urlopen(request, timeout=self.timeout)` 打开响应，用 `_read_response_body()` 读回字节；⑧若抛 `HTTPError` 且状态码正好是 501，关闭该异常对象，调用 `_legacy_search_endpoint()` 生成老式 GET 地址，再用 `urlopen` 发一次只带 `Authorization` 与 `Accept` 的 GET 请求并读取响应体；其它 HTTP 状态码直接向上抛；⑨`json.loads(raw_body.decode("utf-8"), parse_constant=reject_json_constant)` 解析，捕获 `UnicodeDecodeError` 与 `ValueError` 并转成 `ValueError("search response was not valid UTF-8 JSON")`；⑩`return _normalize_response(payload, arguments.max_results)`。
- **异常/边界**：`TypeError`（参数类型不对）；`RuntimeError`（base_url 或 api_key 未配置）；`ValueError`（请求体不可序列化、响应不是合法 UTF-8 JSON、响应过大）；`HTTPError`（非 501 的 HTTP 错误原样抛出，由上层处理）；`URLError`/`socket.timeout` 等网络异常未捕获，会直接冒泡给运行时；空结果集返回空列表而非报错；响应体超过 2,000,000 字节由 `_read_response_body()` 抛 `ValueError`；上游返回的条目不是字典时在归一化阶段被静默跳过。
- **同文件关系**：调用 `_search_endpoint()`、`_read_response_body()`、`_legacy_search_endpoint()`、`_normalize_response()`；使用 `SearchInput`（类型断言与属性读取）与 `SearchOutput`（返回类型）；被运行时/执行管理器调用，本文件内没有函数调用它。

### `_normalize_nullable_text(value) -> Any`（第 328 行）
- **作用**：这是模块级的通用"可空文本"清洗函数，被 `SearchInput` 的 `normalize_tag` 与 `normalize_language` 两个字段校验器共用。它解决的是一类非常具体的现实问题：OpenAI 兼容模型在表达"这个可选参数没有值"时，会写出 `None`、`""`、`"   "`、`"None"`、`"null"`、`"NULL"` 等各种形式。如果这些值原样进入请求体，AnySearch 会认为 `tag` 或 `language` 被显式设置成了一个非法值。把这类哨兵统一折叠成 Python 的 `None`，就能让 `execute()` 里的"非 None 才放进请求体"逻辑自然生效。它对非字符串类型保持中立，原样返回，从而不破坏 `strict=True` 的类型校验语义。
- **参数**：`value: Any` —— 待清洗的原始值，可能是 `None`、字符串，也可能是数字、布尔、列表、字典等。
- **返回**：返回 `Any`。`value` 为 `None` 时返回 `None`；非字符串时原样返回；字符串 `strip()` 后为空、或 `casefold()` 后等于 `"none"`/`"null"` 时返回 `None`；其余返回 `strip()` 后的字符串。
- **内部流程**：先判断 `value is None or not isinstance(value, str)`，命中直接返回原值；否则 `normalized = value.strip()`；再判断 `if not normalized or normalized.casefold() in {"none", "null"}`，命中返回 `None`；最后 `return normalized`。
- **异常/边界**：不会抛异常；对 `None`、空串、纯空白、大小写变体都安全；注意它会把合法的纯文本 `"None"` 也当成空值丢弃，这是刻意的兼容取舍。
- **同文件关系**：被 `SearchInput.normalize_tag()` 和 `SearchInput.normalize_language()` 调用；自身只使用内置的 `str.strip()` 与 `str.casefold()`，不调用本文件其他函数。

### `_search_endpoint(base_url) -> str`（第 337 行）
- **作用**：这是端点规范化函数，把用户/配置里写的各种形式的 base_url 统一成可以直接 POST 的搜索地址。现实配置里可能出现三种写法：`https://host`（只有主机名）、`https://host/v1`（带版本前缀）、`https://host/v1/search`（已经是完整端点）。这个函数把前两种自动补全成 `.../v1/search`，第三种保持不变。它同时承担一道安全职责：校验 scheme 必须是 `http` 或 `https` 且必须有 netloc，这样后续 `urlopen` 就不会被 `file://`、`ftp://` 之类的协议欺骗（代码里 `# nosec B310` 的注释正是说明这一点）。它还会去掉 URL 里的 fragment，避免把 `#anchor` 混进请求目标。
- **参数**：`base_url: str` —— 配置或构造参数传入的服务基址字符串，理论上应为绝对 HTTP(S) URL。
- **返回**：返回 `str`，规范化后的完整搜索端点，例如 `https://api.example.com/v1/search`。若原路径以 `/` 结尾会先 `rstrip("/")`，若原路径以 `/v1` 结尾会追加 `/search`。
- **内部流程**：①`urlsplit(base_url)` 拆成 scheme/netloc/path/query/fragment；②判断 `parsed.scheme not in {"http", "https"} or not parsed.netloc`，命中即 `raise ValueError("search base_url must be an absolute HTTP(S) URL")`；③`path = parsed.path.rstrip("/")`；④若 `path` 为空则设为 `/v1/search`；⑤否则若 `path.endswith("/v1")` 则 `path += "/search"`；⑥`urlunsplit(parsed._replace(path=path, fragment=""))` 重新拼装并返回（query 部分被保留）。
- **异常/边界**：非 http/https 协议或缺少主机名抛 `ValueError`；空字符串会被判为无 netloc 而抛 `ValueError`；尾随斜杠被规范化；fragment 被丢弃；query 参数保留不删（因为某些网关可能在基址上带必要查询参数）。
- **同文件关系**：被 `SearchTool.execute()` 调用；`_legacy_search_endpoint()` 接收它的输出作为输入；依赖标准库 `urlsplit` / `urlunsplit`。

### `_legacy_search_endpoint(endpoint, arguments) -> str`（第 349 行）
- **作用**：这是为"极老的 AnySearch 兼容网关"准备的后备端点构造器。有些早期部署只实现了 GET 形式的搜索接口，对文档化的 POST 会返回 501 Not Implemented；`SearchTool.execute()` 捕获到 501 后就会调用本函数，把查询参数编码进 URL 再发一次 GET。它把查询串参数名定为 `q`（老接口惯例）而把条数参数定为 `limit`（老接口术语，而不是新接口的 `max_results`），这是两代接口命名的真实差异。它只在降级路径上被用到，正常路径永远走 POST。
- **参数**：`endpoint: str` —— 已经过 `_search_endpoint()` 规范化的 POST 端点 URL；`arguments: SearchInput` —— 原始入参对象，用于读取 `query` 与 `max_results`。
- **返回**：返回 `str`，带查询串的 GET 端点。若原端点自身已带 query，则用 `&` 拼接而不是覆盖，例如 `https://host/v1/search?a=1&q=xxx&limit=10`。
- **内部流程**：①`urlsplit(endpoint)` 拆解；②`urlencode({"q": arguments.query, "limit": arguments.max_results})` 生成编码后的查询串（自动做百分号转义，中文查询也安全）；③`merged_query = f"{parsed.query}&{query}" if parsed.query else query`，即已有查询串则追加、否则直接使用；④`urlunsplit(parsed._replace(query=merged_query, fragment=""))` 返回新 URL。
- **异常/边界**：自身不抛异常；不做协议校验（因为输入已经由 `_search_endpoint()` 校验过）；`arguments` 若缺少 `query`/`max_results` 属性会抛 `AttributeError`，但类型注解已约束为 `SearchInput`，实际不会发生。
- **同文件关系**：被 `SearchTool.execute()` 在 HTTP 501 分支中调用；读取 `SearchInput` 实例属性；依赖标准库 `urlencode` / `urlsplit` / `urlunsplit`。

### `_read_response_body(response) -> bytes`（第 356 行）
- **作用**：这是响应体读取函数，核心目的是给搜索响应加一道尺寸上限，防止上游（或中间人）返回一个超大响应把进程内存吃满。它先看 `Content-Length` 头：如果有且能被解析成正整数、且不超过 2,000,000 字节，就认为安全；如果声明值超过上限，直接抛错而不去读。读的时候用 `response.read(2_000_001)` 多读一个字节，这样即使服务端谎报或省略了 `Content-Length`，也能通过"读回来的长度是否超过 2,000,000"来判断超限。它还兼容了只实现无参 `read()` 的测试替身与部分 urllib 兼容适配器。函数带 `Any` 类型的 `response` 参数，是为了同时接受真实 `http.client.HTTPResponse` 和测试里的假对象。
- **参数**：`response: Any` —— 一个类文件对象，需具备 `read()` 方法；可选具备 `headers` 属性（支持 `.get()` 查询）。
- **返回**：返回 `bytes`，即响应体原始字节（未做解码），长度保证不超过 2,000,000 字节。
- **内部流程**：①`headers = getattr(response, "headers", None)`；②若 headers 存在，读取 `"Content-Length"` 或小写 `"content-length"`；③若有长度值，`int()` 转换失败抛 `TypeError("invalid Content-Length header")`，值为负抛同样 `TypeError`，值大于 2,000,000 抛 `ValueError("search response is too large")`；④`try: body = response.read(2_000_001)`，若抛 `TypeError`（说明该对象只支持无参 `read()`）则退化为 `response.read()`；⑤若 `len(body) > 2_000_000` 抛 `ValueError("search response is too large")`；⑥返回 `body`。
- **异常/边界**：非法 `Content-Length`（非数字、负数）抛 `TypeError`；响应过大（头部声明或实际长度超过 2,000,000 字节）抛 `ValueError`；缺失 `Content-Length` 不做拒绝，靠读取长度兜底；`response.read` 自身抛出的网络异常不捕获，直接冒泡。
- **同文件关系**：被 `SearchTool.execute()` 在 POST 正常路径与 501 降级 GET 路径中两次调用；不调用本文件其他函数。

### `_normalize_response(payload, max_results) -> SearchOutput`（第 382 行）
- **作用**：这是响应归一化函数，把 AnySearch 返回的任意信封形状收敛成项目统一的 `SearchOutput`。它按文档化的信封 `code/message/request_id/data` 做严格校验：`code` 必须是整数（显式排除布尔，因为 Python 里 `bool` 是 `int` 的子类），非 0 视为业务失败；`data` 必须是对象；`data.results` 必须是列表。同时它对老式网关留了一条窄兼容：当 `data` 缺失但顶层直接有 `items` 或 `results` 时，也接受并从顶层取结果。逐条构造 `SearchItem` 时它做了两处字段回退——`url` 回退到 `link`、`snippet` 回退到 `description`，并且对每个字段先转字符串再按模型长度上限切片，避免因为上游返回超长文本或 `null` 而让整个响应校验失败。它还用 `raw_items[:max_results]` 做了二次截断，保证返回条数不超过调用方要求。
- **参数**：`payload: Any` —— 已经 `json.loads` 出来的响应对象，正常情况是 `dict`；`max_results: int` —— 调用方要求的结果条数上限，来自 `SearchInput.max_results`（1–10）。
- **返回**：返回 `SearchOutput` 实例，`items` 为最多 `max_results` 条 `SearchItem`。上游成功但无结果、或所有条目都不是字典时，返回 `SearchOutput(items=[])`。
- **内部流程**：①`isinstance(payload, dict)` 不成立即抛 `TypeError("search response must be a JSON object")`；②读 `code = payload.get("code")`，若不为 None 且是 `bool` 或不是 `int` 抛 `TypeError("search response code must be an integer")`；③若 `code not in (None, 0)` 抛 `RuntimeError("AnySearch search request failed")`；④读 `data = payload.get("data")`，若不为 None 且不是 `dict` 抛 `TypeError("search response data must be an object")`；⑤若 `data is None` 且顶层既没有 `"items"` 也没有 `"results"` 键，抛 `TypeError("search response is missing data.results")`；⑥取 `raw_items`：`data` 是 dict 时用 `data.get("results", [])`，否则用 `payload.get("items", payload.get("results", []))`；⑦若 `raw_items` 不是 `list` 抛 `TypeError("search response results must be a list")`；⑧初始化空 `items` 列表，遍历 `raw_items[:max_results]`，非 dict 的条目 `continue` 跳过，否则用 `_text()` 取值并按 `[:1000]`/`[:2000]`/`[:5000]` 切片，构造 `SearchItem` 追加；⑨返回 `SearchOutput(items=items)`。
- **异常/边界**：`TypeError`（payload 非对象、code 非整数、data 非对象、缺少 data.results、results 非列表）；`RuntimeError`（业务 code 非 0，注意这里不携带上游 message，避免把不可信文本直接透出）；条目级别的脏数据（非字典）被静默跳过而不是报错；`title`/`url`/`snippet` 为 `None` 时由 `_text()` 转成空串；超长字段被切片而非报错。
- **同文件关系**：调用 `_text()` 构造字段、构造 `SearchItem` 与 `SearchOutput`；被 `SearchTool.execute()` 作为最后一步调用。

### `_text(value) -> str`（第 422 行）
- **作用**：这是最小的字段兜底转换函数，把任意值安全地变成字符串。上游结果里的 `title`、`url`、`snippet` 有时是 `null`，有时是数字（例如把年份当标题），有时干脆缺键（`item.get()` 返回 `None`）。如果直接 `str()`，`None` 会变成字面量 `"None"` 混进结果里污染模型上下文；这个函数先用 `None` 判断挡掉这种情况，其余值才走 `str()`。它让 `_normalize_response()` 的循环体保持一行一个字段的紧凑写法。
- **参数**：`value: Any` —— 任意待转换的值，常见为 `str`、`None`、`int`，也可能是 list/dict。
- **返回**：返回 `str`。`value is None` 时返回空字符串 `""`；否则返回 `str(value)`（数字会变成十进制字符串，字典/列表会变成 Python repr 形式）。
- **内部流程**：单行表达式 `return "" if value is None else str(value)`，无条件分支、无循环、无副作用。
- **异常/边界**：几乎不抛异常；只有当值的 `__str__` 实现本身报错时才会传播（上游 JSON 解析出来的类型都是内置类型，不会发生）。
- **同文件关系**：被 `_normalize_response()` 调用三次（分别处理 title、url、snippet）；自身不调用本文件其他函数。

### `create_tool() -> BaseTool`（第 426 行）
- **作用**：这是工具发现的工厂函数，返回一个无参构造的 `SearchTool` 实例。项目里的自动发现机制约定：模块暴露 `create_tool()` 就调用它来拿工具对象，而不是自己去猜类名或构造签名。因为 `SearchTool.__init__` 的三个参数都有默认值，无参调用会自动走 `config/services.toml` 的 `[search]` 配置，这正是运行时的期望行为。把实例化集中在一个工厂里，也让测试或替换实现时只需覆盖这一个入口。
- **参数**：无。
- **返回**：返回 `BaseTool`（实际运行时类型是 `SearchTool`），已完成配置加载与参数校验。
- **内部流程**：函数体只有 `return SearchTool()` 一行，没有任何条件分支或异常处理。
- **异常/边界**：自身不抛异常，但 `SearchTool()` 的构造可能因为配置非法抛 `TypeError` 或 `ValueError`（例如 timeout 为负、base_url 为空白串），这些异常会原样向上传播给发现机制。
- **同文件关系**：调用 `SearchTool.__init__()`（间接经由 `SearchTool()` 构造）；被本文件外部的工具发现机制调用，本文件内没有函数调用它。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `SearchInput` | `web.search` 的入参 Pydantic 模型，定义并校验 query/max_results/tag/zone/language/params/format 七个字段 |
| `SearchInput.normalize_tag` | 把 tag 的空值哨兵和误填的工具名别名统一折叠成 None |
| `SearchInput.normalize_language` | 把 language 的空值哨兵折叠成 None，不做别名过滤 |
| `SearchInput.normalize_provider_params` | 兼容把 JSON 对象写成字符串的弱客户端，只放行能解析成字典的字符串 |
| `SearchInput.reject_blank_text` | 拒绝只由空白字符组成的 query |
| `SearchInput.limit` | 只读属性，把 max_results 以历史名称 limit 暴露出来 |
| `SearchItem` | 单条搜索结果的稳定输出结构：title/url/snippet |
| `SearchOutput` | 工具整体输出契约，只含一个 SearchItem 列表 items |
| `SearchOutput.results` | 只读属性，用 provider 术语 results 暴露 items |
| `SearchTool` | 继承 BaseTool 的搜索工具实体，持有 spec 元数据并完成真实 HTTP 调用 |
| `SearchTool.__init__` | 加载 [search] 配置、校验 base_url/api_key/timeout，并把 timeout 回写到 spec |
| `SearchTool.execute` | 校验参数、拼端点、发 POST（501 时降级 GET）、解析并归一化响应 |
| `_normalize_nullable_text` | 通用可空文本清洗，把空串与 None/null 哨兵折叠成 None |
| `_search_endpoint` | 校验 base_url 为绝对 HTTP(S) URL 并补全成 /v1/search 端点 |
| `_legacy_search_endpoint` | 为老式网关生成带 q/limit 查询串的 GET 端点 |
| `_read_response_body` | 校验 Content-Length 并按 2,000,000 字节上限读取响应体 |
| `_normalize_response` | 校验 AnySearch 信封并把结果收敛成 SearchOutput |
| `_text` | 把任意值安全转成字符串，None 转成空串 |
| `create_tool` | 工具发现入口，返回无参构造的 SearchTool 实例 |
