# constants.py

## 一、这个文件是干什么的

`constants.py` 是整个项目的「常量与默认值单一事实来源」（single source of truth）。它不定义任何函数、方法或类，全篇只有模块级文档字符串、`from __future__ import annotations` 一行导入，以及按主题分区排列的模块级常量赋值（含少量 f-string 拼装出来的派生常量）。它的职责是把原本散落在 `agents/`、`core/`、`memory/`、`web/`、`tool/` 各处、以及原本藏在环境变量里的数字阈值、路径名、端点、开关和调色板，集中到一处并加注释说明出处，从而让「改一个默认值」只需要改一个文件，也让审计者能从常量反查到它服务的模块。

文件顶部用大段注释划清了边界：它刻意**不**收纳 `tool/*.py` 的 `TOOL_ENABLED`（那是工具发现协议要求的「每模块各自布尔开关」，集中后会破坏协议设计）、`memory/storage/document_repo.py` 的状态枚举（与同文件 DDL 同源，分开会漂移）、各工具自己的 `MAX_*` 协议上限、`tool/domain_classify.py` 的 `DOMAIN_KEYWORDS` 词表、`web/seed.py` 的 `SEED_MARK`、`agents/message_utils.py` 的 `DEFAULT_TOOL_NAME` / `MAX_TOOL_*`、以及 `scripts/*.py` 的 `ROOT`。这些「不收」的说明本身就是本文件价值的一部分：它把「什么是全局默认值」和「什么是模块私有契约」明确区分开。

在运行时的使用方式上，本文件是被动消费型模块：其它模块用 `from constants import XXX` 直接取用，或者把这里的值当作 `MemoryConfig`、`APIEmbedding`、FastAPI 路由校验等处的字段默认值。少数常量带有运行语义而非纯数值语义，例如 `MEMORY_EDGE_REINFORCE`、`MEMORY_HYBRID`、`WEB_AUTOSEED`、`WEB_QA_EXTRACT`、`WEB_QA_EXTRACT_SYNC` 是行为总开关，测试会通过 `monkeypatch` 直接替换它们来关闭边强化、混合检索、自动播种或后台抽取线程，因此这些常量同时充当「测试隔离开关」。

文件里的分区顺序（Agent/LLM 默认值 → 工具循环 → 更新日志读取 → 历史压缩 → 数据库文件名 → 连接与端点 → 嵌入服务 → 记忆层 → RAG → F1 边强化 → 运行开关 → 知识抽取 → ReAct 容错 → Web API → 星图构建 → 领域分类器）基本对应真实运行入口 `python -m web.app` 从启动、连服务、装配记忆层、跑 RAG/抽取、到渲染星图的调用链，因此这份文件也可以当作一张「项目默认参数地图」来读。

## 二、函数与类逐条详解

**重要说明：本文件没有任何函数、方法、类或嵌套函数。** 全篇 336 行中，第 1–25 行是模块级文档字符串，第 27 行是 `from __future__ import annotations`，其余全部是模块级常量赋值语句与注释。因此不存在可以按 `### 函数名(参数) -> 返回类型 （第 N 行）` 格式逐条讲解的可调用对象，也不存在 `__init__`、dunder 方法、私有函数或模块级 class。

为了让这一节仍然具备「逐条详解」的同等信息量，下面按**文件中出现的先后顺序**，逐个讲解每一个模块级常量（共 79 个，含 f-string 派生的 `DEFAULT_QDRANT_URL` / `DEFAULT_NEO4J_URI` / `DEFAULT_PROXY_URL`）。每个条目给出所在行号、类型与取值、作用与使用时机、边界语义，以及与其它常量的派生/依赖关系（对应「同文件关系」字段）。

### `DEFAULT_MAX_RETRIES` （第 35 行）
- **类型与取值**：`int`，值为 `3`。
- **作用**：OpenAI SDK 与项目内 LLM 客户端共用的最大重试次数默认值。当模型请求遇到瞬时网络抖动、限流或 5xx 时，客户端按这个次数重试，超过才把异常向上抛给 Agent 循环。
- **使用时机**：构造 LLM 客户端 / 调用聊天补全接口时若调用方没有显式传 `max_retries`，就用它兜底。
- **边界**：值为 `3` 表示「首次调用 + 最多 3 次重试」这类语义由 SDK 决定；置 `0` 会关闭重试，置负数不具业务意义（SDK 行为未在本文件约束）。本文件不做校验。
- **同文件关系**：无派生、无依赖。

### `DEFAULT_TEMPERATURE` （第 38 行）
- **类型与取值**：`float`，值为 `0.7`。
- **作用**：模型采样温度的统一默认值，控制回答的随机性与发散程度。`0.7` 是「有创造性但不至于跑题」的通用档位，适用于对话、ReAct 推理与知识抽取等需要一定表达多样性的场景。
- **使用时机**：所有未显式指定温度的 LLM 请求。
- **边界**：文件未约束取值范围，惯例是 `0.0`–`2.0`；填 `0.0` 趋向确定性输出。本文件不做校验。
- **同文件关系**：与 `DEFAULT_TIMEOUT` 同属「模型采样温度与请求超时」注释块，彼此无计算关系。

### `DEFAULT_TIMEOUT` （第 39 行）
- **类型与取值**：`int`，值为 `60`，单位秒。
- **作用**：LLM 请求超时的统一默认值，防止一次模型调用无限挂住整个 Agent 轮次。
- **使用时机**：未显式传 `timeout` 的模型调用。
- **边界**：`60` 秒对长上下文生成可能偏紧，需要更长时可被上层覆盖；置 `None` 的语义（是否禁用超时）由客户端实现决定，本文件不约束。
- **同文件关系**：与 `DEFAULT_TEMPERATURE` 同属一个注释块。

### `PROMPT_CACHE_KEY_VERSION` （第 42 行）
- **类型与取值**：`str`，值为 `"pc-v1"`。
- **作用**：prompt-cache（提示词缓存）路由键的命名空间版本号。缓存键里带上这个版本串，可以让「同一份提示词」在新旧版本之间被区分开，改动此值等于一次性作废旧缓存键，避免旧缓存命中到已经变更的提示词结构。
- **使用时机**：构造缓存键、查询缓存、写入缓存时都会拼接它。
- **边界**：改动后旧缓存键全部失效，会带来一次冷启动式的缓存未命中高峰，这是刻意的；本文件不做格式校验。
- **同文件关系**：无派生、无依赖。

### `TOOL_LOOP_SAFETY_LIMIT` （第 50 行）
- **类型与取值**：`int`，值为 `64`。
- **作用**：工具循环（ToolLoop）在调用方没有指定 `max_rounds` 时允许的最大轮数，是一条防止失控 provider 无限请求的保险丝。它约束的是「模型 → 工具 → 模型」这个循环最多转多少圈。
- **使用时机**：`core/tool_loop.py` 构造循环时读取；注释说明 `core/tool_loop.py` 中另有 `DEFAULT_SAFETY_LIMIT` 作为兼容别名指向同义值。
- **边界**：`64` 轮已相当宽松，正常任务远不会触顶；触顶通常意味着模型反复调用工具不收敛，此时循环应终止并给出兜底回答。
- **同文件关系**：与 `REACT_*_RETRY_LIMIT` 属于不同层面的「防死循环」参数（后者管的是协议格式重试，本常量管的是轮数），二者不互相计算。

### `UPDATE_LOG_READ_RANGE_MAX` （第 59 行）
- **类型与取值**：`int`，值为 `100`。
- **作用**：`tool/read_update_logs.py` 单次批量读取 update_log 的最大记录数。该工具一次调用会把整个范围内的记录解码进模型上下文，不设上限时可能一次拉爆内存或超时（注释指出该工具超时为 10 秒），因此用这个上限做硬性截断。
- **使用时机**：工具解析请求范围时，用它夹取或拒绝过大的区间。
- **边界**：请求范围超过 `100` 时的具体行为（截断还是报错）由该工具实现决定，本文件只提供数值。
- **同文件关系**：与 `DEFAULT_UPDATE_LOG_FILENAME` 服务于同一模块，但彼此无计算关系。

### `OBSERVATION_COMPRESS_THRESHOLD` （第 68 行）
- **类型与取值**：`int`，值为 `12_000`（即 12000 个字符）。
- **作用**：历史压缩阈值。超过该长度的 Observation（工具返回结果）在写入 profile 历史时会被压缩成桩文本，而正在运行的那一轮内仍然保留完整负载，这样既不影响当轮推理，又避免长历史把上下文撑爆。
- **使用时机**：保存对话 / 落盘历史时逐条检查 Observation 长度。
- **边界**：阈值按字符数而非 token 数计算，属于粗略但稳定的度量；恰好等于阈值时通常不压缩（严格大于才压缩，具体由实现决定）。
- **同文件关系**：与 `OBSERVATION_STUB_PREFIX`、`OBSERVATION_PREVIEW_CHARS` 构成同一套压缩机制的三件套——本常量决定「压不压」，后两者决定「压成什么样」。

### `OBSERVATION_STUB_PREFIX` （第 69 行）
- **类型与取值**：`str`，值为 `"[已压缩的历史工具结果"`（注意结尾没有右方括号，是刻意留出的前缀）。
- **作用**：压缩桩文本的前缀标记。被压缩的 Observation 会以此串开头，模型与开发者都能一眼识别「这里是历史压缩产物，不是原始工具输出」。
- **使用时机**：生成桩文本时拼接；反向识别桩文本时做前缀匹配。
- **边界**：作为前缀使用，因此包含方括号但不成对；若其它地方想解析出原始长度等信息，需要依赖该前缀之后拼接的额外内容。
- **同文件关系**：配合 `OBSERVATION_COMPRESS_THRESHOLD`（何时用）与 `OBSERVATION_PREVIEW_CHARS`（保留多少原文）。

### `OBSERVATION_PREVIEW_CHARS` （第 70 行）
- **类型与取值**：`int`，值为 `400`。
- **作用**：压缩后的桩文本中保留的原文预览字符数。保留开头一小段原文能让模型仍看到工具结果的开头线索（例如 JSON 头部、错误信息首行），从而在压缩后仍可做基本判断。
- **使用时机**：生成桩文本时截取原文前 400 个字符。
- **边界**：原文本身就短于 400 时不会触发压缩路径（因为长度达不到阈值）；按字符截取可能切断多字节字符之外的内容，属可接受的近似。
- **同文件关系**：三件套之一，依赖 `OBSERVATION_COMPRESS_THRESHOLD` 先行判定。

### `HISTORY_MAX_MESSAGES` （第 74 行）
- **类型与取值**：`int`，值为 `60`。
- **作用**：单条会话历史保留的最大消息数。裁剪时保留开头的 system 前缀，只裁掉旧的对话轮，从而在长会话中把每轮重发给模型的历史长度控制在有界范围内，避免 token 成本随轮次线性上升。
- **使用时机**：每次组装请求消息列表前，对历史做裁剪。
- **边界**：`60` 条包含 system 前缀在内还是之外，由实现决定；裁剪必须保证不破坏「用户/助手」交替结构，否则部分厂商 API 会报错。本文件只给数值。
- **同文件关系**：与 `OBSERVATION_COMPRESS_THRESHOLD` 同属「历史压缩」分区，前者按条数裁剪、后者按单条长度压缩，是两级互补策略。

### `DEFAULT_TOOLS_DB_FILENAME` （第 81 行）
- **类型与取值**：`str`，值为 `"tools.sqlite3"`。
- **作用**：工具相关数据的 SQLite 数据库文件名默认值，只给文件名不给目录，实际路径由运行时基目录与环境变量组合得出（分区注释说明「环境变量可覆盖运行时路径」）。
- **使用时机**：`memory/base.py`、`web/support.py`、`tool/update_log.py`、`core/repository.py` 等在拼装数据库路径时读取。
- **边界**：改这个值不会自动迁移旧文件，旧库会「消失」在新路径之外；本文件不做存在性检查。
- **同文件关系**：与 `DEFAULT_UPDATE_LOG_FILENAME`、`DEFAULT_MEMORY_DB_FILENAME` 同属数据库文件名三兄弟，彼此独立。

### `DEFAULT_UPDATE_LOG_FILENAME` （第 82 行）
- **类型与取值**：`str`，值为 `"update_log.sqlite3"`。
- **作用**：更新日志（update_log）数据库文件名默认值，供 `tool/update_log.py` 与 `tool/read_update_logs.py` 定位存储。
- **使用时机**：写日志与读日志时拼路径。
- **边界**：同 `DEFAULT_TOOLS_DB_FILENAME`，改名等于换库。
- **同文件关系**：与 `UPDATE_LOG_READ_RANGE_MAX` 服务同一业务域；与另两个文件名常量并列。

### `DEFAULT_MEMORY_DB_FILENAME` （第 83 行）
- **类型与取值**：`str`，值为 `"memory.sqlite3"`。
- **作用**：四层记忆系统的持久化数据库文件名默认值，承载文档、分块、事实、 episodic 记录等表。
- **使用时机**：`MemoryConfig` 未指定 sqlite 路径时由上层据此拼装。
- **边界**：注意它和 `MEMORY_SQLITE_DEFAULT`（值为 `":memory:"`）是两个不同层面的默认值——前者是「持久化时的文件名」，后者是记忆库配置字段的出厂默认（不持久化）；实际用哪个取决于配置装配路径。
- **同文件关系**：与 `MEMORY_SQLITE_DEFAULT` 语义相关但取值相反，容易混淆，改任一个都要检查另一处。

### `LOCALHOST` （第 101 行）
- **类型与取值**：`str`，值为 `"127.0.0.1"`。
- **作用**：本机服务统一绑定的回环地址。注释明确解释了为什么用字面 IP 而不用 `"localhost"`：后者在部分 Windows 环境会解析到 IPv6 的 `::1`，而服务只监听 IPv4，表现出的症状是「明明启动了却连不上」。
- **使用时机**：所有本机端点常量、Web 服务监听地址都以它为基座。
- **边界**：只适用于本机回环场景；对外提供服务需要换成 `0.0.0.0` 或具体网卡地址，那不属于本常量的职责。
- **同文件关系**：被 `DEFAULT_QDRANT_URL`、`DEFAULT_NEO4J_URI`、`DEFAULT_PROXY_URL` 三个 f-string 常量直接引用，是本文件里被依赖最多的常量之一。

### `DEFAULT_QDRANT_PORT` （第 107 行）
- **类型与取值**：`int`，值为 `6333`。
- **作用**：本地 Qdrant HTTP 服务的端口，属「本机回环端口分配表」的一部分（注释强调改端口只改这里）。
- **使用时机**：拼装 `DEFAULT_QDRANT_URL`。
- **边界**：这是文档化的本机默认端口，不代表 Qdrant 一定被启用（见 `DEFAULT_QDRANT_URL` 条目）。
- **同文件关系**：被 `DEFAULT_QDRANT_URL` 引用；与 `DEFAULT_NEO4J_BOLT_PORT`、`DEFAULT_WEB_PORT` 并列构成端口分配表。

### `DEFAULT_NEO4J_BOLT_PORT` （第 108 行）
- **类型与取值**：`int`，值为 `7687`。
- **作用**：本地 Neo4j bolt 协议端口，用于图数据库连接。
- **使用时机**：拼装 `DEFAULT_NEO4J_URI`。
- **边界**：同上，属文档化默认值而非启用开关。
- **同文件关系**：被 `DEFAULT_NEO4J_URI` 引用。

### `DEFAULT_WEB_PORT` （第 109 行）
- **类型与取值**：`int`，值为 `8765`。
- **作用**：本地 Web 服务（知识星云 FastAPI 应用）的监听端口。真实运行入口 `python -m web.app` 会用它作为绑定端口。
- **使用时机**：`web/app.py` 启动 uvicorn / 绑定 socket 时读取。
- **边界**：端口被占用时启动会失败，需要外部改配置或改本常量；本文件不做占用检测。
- **同文件关系**：与 `LOCALHOST` 配合决定服务监听地址；与另两个端口常量并列。

### `DEFAULT_QDRANT_URL` （第 113 行）
- **类型与取值**：`str`，f-string 派生，值为 `"http://127.0.0.1:6333"`。
- **作用**：本地 Qdrant 端点。注释给出启用方式：在 `config/services.toml` 的 `[qdrant]` 段写 `url = "http://127.0.0.1:6333"`。它同时被标注为本机「服务连哪里」的唯一事实来源之一。
- **使用时机**：`memory/storage/qdrant.py`（默认 collection 相关）与 `web/app.py` 等直接 import；间接消费方 `memory/manager.py` 依据 `MemoryConfig.qdrant_url` 是否非空决定建真存储还是内存回退。
- **边界（关键）**：这是「本机文档化默认值」，**不是默认启用**。出厂 `MemoryConfig.qdrant_url` 仍为 `None`，即不连接、走内存回退；只有显式配置（`services.toml` 或构造参数）才连真服务。想接 Qdrant Cloud 时端点与密钥走 `config/services.toml`，不要改本常量。
- **同文件关系**：依赖 `LOCALHOST` 与 `DEFAULT_QDRANT_PORT`；与 `MEMORY_QDRANT_COLLECTION`（集合名）配合使用。

### `DEFAULT_NEO4J_URI` （第 117 行）
- **类型与取值**：`str`，f-string 派生，值为 `"bolt://127.0.0.1:7687"`。
- **作用**：本地 Neo4j 端点。注释给出启用方式：`config/services.toml` 的 `[neo4j]` 段写 `uri = "bolt://127.0.0.1:7687"`，用户名密码同段配置。
- **使用时机**：间接消费方 `memory/manager.py`（`neo4j_uri` 非空才建真图存储，否则内存回退）与 `memory/storage/graph.py`（URI 由 `MemoryConfig` 传参）。
- **边界**：同样是「文档化默认值而非默认启用」，出厂为 `None`；Neo4j Aura 等云端实例的 URI 与凭据走 `services.toml`。
- **同文件关系**：依赖 `LOCALHOST` 与 `DEFAULT_NEO4J_BOLT_PORT`。

### `DEFAULT_PROXY_PORT` （第 121 行）
- **类型与取值**：`int`，值为 `7890`。
- **作用**：本机转发代理端口（注释说明是 Clash 默认端口）。当云端 Qdrant / Neo4j 没有显式配置 `[proxy]` 时，走这个本机代理出去。
- **使用时机**：拼装 `DEFAULT_PROXY_URL`；网络客户端在需要访问境外服务且未显式配置代理时读取。
- **边界**：注释强调「本机回环地址绝不走代理」——即访问 `127.0.0.1` 的请求必须绕过代理，否则本地服务会被误转发到代理端口而连不上。
- **同文件关系**：被 `DEFAULT_PROXY_URL` 引用。

### `DEFAULT_PROXY_URL` （第 122 行）
- **类型与取值**：`str`，f-string 派生，值为 `"http://127.0.0.1:7890"`。
- **作用**：默认代理 URL 的完整形式，供 HTTP 客户端（requests / httpx / SDK）设置 `proxies` 或 `http_proxy` 时使用。
- **使用时机**：云端嵌入、Qdrant Cloud、Neo4j Aura、联网搜索等需要出网的调用在未显式配置 `[proxy]` 时兜底。
- **边界**：代理没启动时出网调用会连接被拒；同时必须配合「回环不走代理」的例外规则。本文件只给 URL，不做连通性检查。
- **同文件关系**：依赖 `LOCALHOST` 与 `DEFAULT_PROXY_PORT`。

### `DEFAULT_EMBEDDING_MODEL` （第 132 行）
- **类型与取值**：`str`，值为 `"qwen3-embedding-0.6b"`。
- **作用**：嵌入模型名的兜底默认值，仅在 `config/services.toml` 的 `[embedding].model` 未配置时生效（注释建议显式配置）。注释同时指出：维度的唯一开关就是 `[embedding].model`，因为维度由模型输出决定，若与已有 Qdrant 集合不匹配会被守卫拦下。
- **使用时机**：`memory/embedding.py` 的 `APIEmbedding` 与 `memory/base.py` 的 `MemoryConfig` 装配时。
- **边界**：换模型通常意味着换向量维度，需要重建集合，否则集合维度校验失败；本文件不做校验。
- **同文件关系**：与 `DEFAULT_EMBEDDING_BATCH_SIZE` 同属嵌入服务分区；与 `MEMORY_EMBEDDING_DIMENSION_REMOTE` 的「自动识别维度」策略相互配合。

### `DEFAULT_EMBEDDING_BATCH_SIZE` （第 135 行）
- **类型与取值**：`int`，值为 `10`。
- **作用**：DashScope 风格的批处理上限，即一次嵌入请求最多带多少条文本。其它厂商可在 `[embedding].batch_size` 调整。
- **使用时机**：批量嵌入文档分块时，把分块列表切片成每批最多 10 条。
- **边界**：设为过大值可能触发厂商侧单请求上限报错；设为 `1` 会退化为逐条请求、明显变慢。本文件不校验正负。
- **同文件关系**：与 `DEFAULT_EMBEDDING_MODEL` 并列。

### `MEMORY_DEFAULT_TTL_SECONDS` （第 143 行）
- **类型与取值**：`float`，值为 `3600.0`，单位秒（即 1 小时）。
- **作用**：四层记忆中「工作记忆」的默认存活时间（TTL）。条目写入时若未显式指定过期时间，就按这个值计算到期时刻，超时条目在读取/清理时被淘汰。
- **使用时机**：`memory/base.py` 的 `MemoryConfig` 字段默认值。
- **边界**：用 `float` 而非 `int`，便于配置成小数秒（测试里常用很小的值来快速验证过期逻辑）；设为 `0` 或负数的语义（立即过期？永不过期？）由记忆层实现决定，本文件不约束。
- **同文件关系**：与 `MEMORY_WORKING_CAPACITY` 共同定义工作记忆的「时间维 + 容量维」双重边界。

### `MEMORY_WORKING_CAPACITY` （第 146 行）
- **类型与取值**：`int`，值为 `100`。
- **作用**：工作记忆可容纳的条目上限。超过容量时通常按 LRU 或时间顺序淘汰最旧条目，保证工作记忆始终是「近期活跃上下文」而不是无限增长的缓存。
- **使用时机**：工作记忆写入路径上做容量检查。
- **边界**：与 TTL 同时生效，两个条件谁先触发谁先淘汰；本文件不规定淘汰策略。
- **同文件关系**：与 `MEMORY_DEFAULT_TTL_SECONDS` 配对使用。

### `MEMORY_SEARCH_LIMIT` （第 149 行）
- **类型与取值**：`int`，值为 `10`。
- **作用**：语义检索默认返回条数，即一次相似度查询取 top-k 中的 k。控制注入模型上下文的记忆条数与 token 成本。
- **使用时机**：调用方未显式传 limit 的语义检索（如记忆层 recall）。
- **边界**：设为过大值会让上下文变长、噪声变多；`0` 的语义（返回空还是用默认）由实现决定。
- **同文件关系**：与 `MEMORY_SIMILARITY_THRESHOLD` 组成检索的两大默认参数（取多少条、门槛多高）。

### `MEMORY_SIMILARITY_THRESHOLD` （第 152 行）
- **类型与取值**：`float`，值为 `0.0`。
- **作用**：语义检索默认相似度阈值。注释明确 `0` 表示「不过滤」，即只要进入 top-k 就返回，不按相似度剔除。
- **使用时机**：检索结果后置过滤阶段。
- **边界**：设为接近 `1.0` 会非常严格、可能经常返回空；由于不同嵌入模型的相似度分布差异大，出厂选择不过滤是稳妥做法。
- **同文件关系**：与 `MEMORY_SEARCH_LIMIT` 配对。

### `MEMORY_EMBEDDING_DIMENSION` （第 155 行）
- **类型与取值**：`int`，值为 `1024`。
- **作用**：离线 `HashEmbedding` 的默认向量维度，仅用于测试与「无云端配置」的兜底场景。离线哈希嵌入必须有一个固定维度才能生成向量。
- **使用时机**：`memory/embedding.py` 构造 `HashEmbedding` 时。
- **边界**：与远端模型的真实维度无关；若在离线兜底与云端嵌入之间切换，向量维度不同会导致检索结果不可比（通常需要重建集合）。
- **同文件关系**：与 `MEMORY_EMBEDDING_DIMENSION_REMOTE` 是「离线 / 远端」两个对照常量，注意不要混用。

### `MEMORY_EMBEDDING_DIMENSION_REMOTE` （第 160 行）
- **类型与取值**：`int | None`，出厂值为 `None`（显式带类型注解）。
- **作用**：远端嵌入的期望向量维度。出厂留空（`None`）表示「不预设」：首次调用时从响应里自动识别维度（`APIEmbedding` 把 `0` 当作「尚未知」），因此接任意嵌入平台都不必先查文档。要固定维度就在 `[embedding].dimension` 填整数。
- **使用时机**：`APIEmbedding` 初始化与首次响应解析；Qdrant 集合创建/校验时也会用到最终确定的维度。
- **边界**：`None` 与 `0` 是两种不同的「未知」表示——配置层用 `None`，运行时对象内部用 `0` 当哨兵；若显式填了一个与模型实际输出不符的整数，会导致维度校验失败。这也是本文件里唯一带 `int | None` 注解的常量。
- **同文件关系**：与 `MEMORY_EMBEDDING_DIMENSION`（离线维度）对照；与 `DEFAULT_EMBEDDING_MODEL` 的「维度由模型决定」策略一致。

### `MEMORY_EMBEDDING_PROVIDER_DEFAULT` （第 165 行）
- **类型与取值**：`str`，值为 `"auto"`。
- **作用**：嵌入提供方的默认选择。`auto` 表示：配置了 `[embedding]` 端点 + 密钥就走云端 `APIEmbedding`，否则走离线兜底；这是「零配置也能跑、配了就升级」的策略。
- **使用时机**：`MemoryConfig` / `APIEmbedding` 装配时决定选型。
- **边界**：取值必须落在 `MEMORY_EMBEDDING_PROVIDERS` 三元组内，否则行为未定义；本文件不做校验。
- **同文件关系**：与 `MEMORY_EMBEDDING_PROVIDERS` 严格配套——后者是合法取值集合。

### `MEMORY_EMBEDDING_PROVIDERS` （第 166 行）
- **类型与取值**：`tuple[str, ...]`，值为 `("auto", "openai", "hash")`。
- **作用**：嵌入提供方的合法取值白名单。`auto` 是自动判定；`openai` 强制走云端（缺配置时报错并回退）；`hash` 强制走离线哈希嵌入。用元组而非列表，是因为元组不可变，能防止被意外修改。
- **使用时机**：配置校验、界面下拉选项、文档枚举。
- **边界**：白名单只约束取值，不描述各值的行为差异；顺序在文档意义上表示「默认在前」。
- **同文件关系**：与 `MEMORY_EMBEDDING_PROVIDER_DEFAULT` 配套，后者必须是前者之一。

### `MEMORY_EMBEDDING_TIMEOUT` （第 169 行）
- **类型与取值**：`float`，值为 `30.0`，单位秒。
- **作用**：嵌入请求超时时间。注意它比 LLM 的 `DEFAULT_TIMEOUT`（60 秒）更短，因为嵌入通常是批量短请求，超时应当更快暴露问题。
- **使用时机**：`APIEmbedding` 发起 HTTP 请求时。
- **边界**：大批量嵌入在慢网络上可能触顶；本文件不做重试配置（重试次数走 `DEFAULT_MAX_RETRIES` 一类的客户端参数）。
- **同文件关系**：与 `DEFAULT_EMBEDDING_BATCH_SIZE` 共同决定「批量嵌入一次请求的规模与耐心」。

### `MEMORY_SQLITE_DEFAULT` （第 172 行）
- **类型与取值**：`str`，值为 `":memory:"`。
- **作用**：记忆库 SQLite 的默认连接目标，`":memory:"` 是 SQLite 的特殊路径，表示进程内内存数据库、不做持久化。出厂这样设置可以让项目「开箱即跑、不留副作用」。
- **使用时机**：`MemoryConfig` 未指定 sqlite 路径时。
- **边界**：进程退出数据即丢失；要持久化必须显式配置文件路径（此时文件名参考 `DEFAULT_MEMORY_DB_FILENAME`）。
- **同文件关系**：与 `DEFAULT_MEMORY_DB_FILENAME` 语义互补但取值相反（内存 vs 文件名），是本文件中最容易被误读的一对。

### `MEMORY_QDRANT_COLLECTION` （第 175 行）
- **类型与取值**：`str`，值为 `"helloagents_memory"`。
- **作用**：可选 Qdrant 后端的默认集合名。所有记忆向量写入/检索都落在这个 collection 里。
- **使用时机**：`memory/storage/qdrant.py` 初始化客户端与默认集合时读取。
- **边界**：集合一旦创建就绑定了向量维度与距离度量，换嵌入模型往往需要换集合名或重建；本文件不校验集合是否存在。
- **同文件关系**：与 `DEFAULT_QDRANT_URL` 配合构成「连哪个 Qdrant、用哪个集合」。

### `RAG_CHUNK_SIZE` （第 183 行）
- **类型与取值**：`int`，值为 `1000`。
- **作用**：RAG 文档切块的默认块长（字符数）。决定每个 chunk 承载多少原文，进而影响检索粒度与嵌入调用次数。
- **使用时机**：`memory/rag/document.py`、`memory/rag/pipeline.py` 切块时；`web/ingest_queue.py` 的入库切块另有更细的 `WEB_INGEST_CHUNK_SIZE`。
- **边界**：块太大则检索命中后上下文冗余、精度下降；块太小则语义不完整、召回过碎。必须大于 `RAG_CHUNK_OVERLAP`，否则切块会死循环或产生异常结果。
- **同文件关系**：与 `RAG_CHUNK_OVERLAP` 配对；与 `WEB_INGEST_CHUNK_SIZE`、`QA_EXTRACT_CHUNK_SIZE` 是三个不同场景的块长。

### `RAG_CHUNK_OVERLAP` （第 184 行）
- **类型与取值**：`int`，值为 `100`。
- **作用**：相邻块之间的重叠长度（字符数）。重叠的作用是避免关键句子恰好被切在块边界上而丢失语义，让跨边界的表述在至少一个块里完整出现。
- **使用时机**：切块循环计算下一块起点时（步长通常为 `RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP`）。
- **边界**：必须小于 `RAG_CHUNK_SIZE`；重叠越大索引冗余越多、嵌入成本越高。
- **同文件关系**：与 `RAG_CHUNK_SIZE` 强耦合，改动需成对考虑。

### `RAG_RETRIEVE_LIMIT` （第 187 行）
- **类型与取值**：`int`，值为 `5`。
- **作用**：检索与上下文装配默认返回条数。比记忆层的 `MEMORY_SEARCH_LIMIT`（10）更小，因为 RAG 返回的是一整块文档原文，占用的上下文明显更多。
- **使用时机**：`memory/rag/pipeline.py` 装配检索结果时。
- **边界**：设为过大值会迅速吃满上下文窗口；本文件不设上限。
- **同文件关系**：与 `RAG_CONTEXT_MAX_CHARS` 一起构成 RAG 上下文的「条数 + 总字符」双限。

### `RAG_GRAPH_HOPS` （第 190 行）
- **类型与取值**：`int`，值为 `1`。
- **作用**：图检索默认跳数，即从命中实体出发向外扩展几层邻居。默认 1 跳意味着只取直接相邻节点，召回范围可控、噪声低。
- **使用时机**：`memory/rag/graph_rag.py` 做图扩展时。
- **边界**：跳数增加会让结果规模指数增长，因此另设 `RAG_GRAPH_MAX_HOPS` 上限；必须 ≤ 该上限。
- **同文件关系**：与 `RAG_GRAPH_MAX_HOPS`、`RAG_GRAPH_PATH_LIMIT` 组成图检索的三个规模控制参数。

### `RAG_GRAPH_MAX_HOPS` （第 191 行）
- **类型与取值**：`int`，值为 `3`。
- **作用**：图检索跳数的允许上限，用于夹取调用方传入的过大跳数，防止一次查询把整张图拉出来。
- **使用时机**：校验 / 夹取请求参数时。
- **边界**：即使 3 跳在图密集时也可能爆炸，因此还有路径数上限兜底。
- **同文件关系**：与 `RAG_GRAPH_HOPS` 成对（默认值与上限）。

### `RAG_GRAPH_PATH_LIMIT` （第 194 行）
- **类型与取值**：`int`，值为 `20`。
- **作用**：图路径展开上限，限制一次图检索最多展开多少条路径，是防止组合爆炸的第二道闸。
- **使用时机**：图遍历 / 路径枚举时计数截断。
- **边界**：达到上限后通常直接停止扩展，可能导致结果不完整，这是「宁可少也不要卡死」的取舍。
- **同文件关系**：与 `RAG_GRAPH_HOPS`、`RAG_GRAPH_MAX_HOPS` 同组。

### `RAG_CONTEXT_MAX_CHARS` （第 197 行）
- **类型与取值**：`int`，值为 `12000`。
- **作用**：图检索上下文拼装的最大字符数。把多条路径 / 关系拼成一段提示词文本时，用它做总长度截断，保证注入模型的上下文有界。
- **使用时机**：`memory/rag/graph_rag.py` 拼装上下文末尾。
- **边界**：按字符截断可能切断句子；被截断后模型看到的信息不完整，属可接受的近似。
- **同文件关系**：与 `RAG_RETRIEVE_LIMIT` 共同约束 RAG 上下文体积；与 `GRAPH_CONTEXT_MAX_RELATIONS`（条数上限）是「字符 + 条数」两种截断方式。

### `MEMORY_EDGE_WEIGHT_GROWTH` （第 205 行）
- **类型与取值**：`float`，值为 `1.5`。
- **作用**：F1 边强化（「回忆即强化」）机制中，一次「回忆」把图边权重乘以的倍数。边权重初始为 `1.0`，每次被检索命中就乘 1.5，模拟「越常被想起的关系越牢固」。
- **使用时机**：`memory/storage/graph.py` 的边递增逻辑。
- **边界**：大于 1 才叫强化；设为 1.0 等于关闭强化效果（但排序逻辑仍可能变化）；与上限 `MEMORY_EDGE_WEIGHT_MAX` 配合防止无限增长。
- **同文件关系**：与 `MEMORY_EDGE_WEIGHT_MAX`、`MEMORY_EDGE_REINFORCE` 同属 F1 分区，三者构成「倍率 + 上限 + 开关」。

### `MEMORY_EDGE_WEIGHT_MAX` （第 208 行）
- **类型与取值**：`float`，值为 `8.0`。
- **作用**：边权重上限，防止「富者越富」效应把热门边权重推到无穷大，导致检索排序被少数边垄断。
- **使用时机**：边权重递增后立即做 `min(weight, MAX)` 式夹取。
- **边界**：必须 ≥ 初始权重 `1.0`，否则新边一开始就被压到上限；本文件不做校验。
- **同文件关系**：与 `MEMORY_EDGE_WEIGHT_GROWTH` 配套。

### `MEMORY_EDGE_REINFORCE` （第 212 行）
- **类型与取值**：`bool`，值为 `True`。
- **作用**：F1 边强化总开关。默认开启；置 `False` 时检索完全不触发递增、排序退回纯 `confidence`，用于还原旧行为与回归对照。注释特别说明测试里会 `monkeypatch` 本常量。
- **使用时机**：`memory/rag/graph_rag.py` 的排序与触发路径。
- **边界**：这是行为开关而非数值参数，运行中不应被改写；作为模块级变量被 monkeypatch 时只影响测试进程。
- **同文件关系**：与 `MEMORY_EDGE_WEIGHT_GROWTH`、`MEMORY_EDGE_WEIGHT_MAX` 同组，是这组机制的「闸门」。

### `MEMORY_HYBRID` （第 220 行）
- **类型与取值**：`bool`，值为 `True`。
- **作用**：RAG 混合检索（FTS5 关键词检索 × 向量检索的 RRF 融合）总开关；`False` 表示退化为纯向量检索。注释指出它原本是散落的环境变量，现收拢到本文件以便测试 monkeypatch。
- **使用时机**：`memory/rag/pipeline.py`、`memory/rag/graph_rag.py`、`web/ingest_queue.py` 在检索路径上读取。
- **边界**：关闭后关键词精确命中能力丢失，专有名词、代码标识符类查询召回会变差；不影响已入库数据。
- **同文件关系**：与 `MEMORY_EDGE_REINFORCE` 同属「运行开关」分区，都是测试隔离用的模块级开关。

### `ENTITY_SIMILARITY_THRESHOLD` （第 228 行）
- **类型与取值**：`float`，值为 `0.88`。
- **作用**：知识抽取中实体名相似度合并的阈值，取值方式是 `SequenceMatcher` 相似度与 token 重叠度取最大值后与该阈值比较。超过阈值就认为两个实体名指同一个实体，从而合并、避免图谱里出现大量近义重复节点。
- **使用时机**：`memory/rag/knowledge.py` 实体解析与合并阶段。
- **边界**：调高会漏合并（图谱碎片化），调低会错合并（把不同实体并成一个）；`0.88` 是偏保守的取值。
- **同文件关系**：与 `ENTITY_PREFIX_MIN_LENGTH` 一起控制合并的激进程度；与 `ENTITY_DEFAULT_TYPE`、`ENTITY_DEFAULT_CONFIDENCE` 同属知识抽取分区。

### `ENTITY_DEFAULT_CONFIDENCE` （第 231 行）
- **类型与取值**：`float`，值为 `0.8`。
- **作用**：实体解析时若没有更具体的置信度信息，就给新实体 / 新关系赋这个默认置信度。它是图检索排序（尤其关闭边强化后的纯 confidence 排序）的基准分。
- **使用时机**：抽取流水线写入实体与关系时。
- **边界**：取值惯例在 `0.0`–`1.0`；高于真实确定性会让低质量抽取结果在排序中占优。
- **同文件关系**：与 `ENTITY_DEFAULT_TYPE` 是「置信度 + 类型」两个默认属性。

### `ENTITY_DEFAULT_TYPE` （第 232 行）
- **类型与取值**：`str`，值为 `"概念"`。
- **作用**：默认实体类型。当抽取器无法判定实体属于人物、组织、技术等哪一类时，落到这个中文兜底类型，保证图谱节点一定有类型、不会出现空类型。
- **使用时机**：`memory/rag/knowledge.py` 构造实体记录时。
- **边界**：中文取值意味着前端展示与筛选需与之保持一致；本文件不做枚举约束。
- **同文件关系**：与 `ENTITY_DEFAULT_CONFIDENCE` 配对；与星图分区的 `DEFAULT_DOMAIN`（领域兜底）语义相似但作用对象不同（实体类型 vs 领域）。

### `ENTITY_NAME_MAX_LENGTH` （第 235 行）
- **类型与取值**：`int`，值为 `200`。
- **作用**：实体名长度上限，解析时超长直接截断。防止模型抽出整段句子当实体名，把图谱节点名撑得无法展示、也破坏相似度比较。
- **使用时机**：写入实体前的清洗阶段。
- **边界**：截断是简单裁剪，可能把有意义的尾部信息丢掉；这是为防止异常输入而做的取舍。
- **同文件关系**：与 `ENTITY_CLEAN_MAX_LENGTH` 是「名字上限 / 正文上限」两个不同字段的截断长度。

### `ENTITY_CLEAN_MAX_LENGTH` （第 236 行）
- **类型与取值**：`int`，值为 `500`。
- **作用**：文本清理默认上限，用于实体描述、上下文片段等较长文本字段的截断，避免单条记录体积失控。
- **使用时机**：抽取结果落库前的清洗。
- **边界**：与 `ENTITY_NAME_MAX_LENGTH` 是两套独立限制，不要互相套用；截断后信息不完整属预期。
- **同文件关系**：与 `ENTITY_NAME_MAX_LENGTH` 并列。

### `ENTITY_PREFIX_MIN_LENGTH` （第 240 行）
- **类型与取值**：`int`，值为 `3`。
- **作用**：别名前缀匹配所需的最短公共前缀长度。低于该长度只接受精确匹配，注释给了具体反例：避免「we」这类过短前缀把「web 中转站」和「web 网关」错误合并成同一实体。
- **使用时机**：实体合并判断中的前缀分支。
- **边界**：设为 `1` 会极其激进地误合并；设为很大的值等于关闭前缀匹配（退化为只靠相似度与精确匹配）。
- **同文件关系**：与 `ENTITY_SIMILARITY_THRESHOLD` 共同决定合并策略的宽严。

### `GRAPH_CONTEXT_MAX_RELATIONS` （第 243 行）
- **类型与取值**：`int`，值为 `60`。
- **作用**：抽取前注入提示词的子图关系条数上限，按置信度排序后取前 60 条。作用是让模型在抽取新知识时能看到既有图谱上下文（便于复用已有实体名、避免重复建点），同时不让提示词无限膨胀。
- **使用时机**：`memory/rag/knowledge.py` 组装抽取提示词时。
- **边界**：超过 60 条的关系被丢弃，可能漏掉低置信度但相关的上下文；与 `RAG_CONTEXT_MAX_CHARS` 的字符上限是两道不同维度的闸。
- **同文件关系**：与 `ENTITY_*` 系列同属知识抽取分区；与 `RAG_CONTEXT_MAX_CHARS` 在「上下文截断」思路上呼应。

### `REACT_UNMARKED_ANSWER_RETRY_LIMIT` （第 251 行）
- **类型与取值**：`int`，值为 `3`。
- **作用**：ReAct 同步 / 文本协议容错中，「轮内未标记答案」的重试上限。当模型输出的答案没有按协议打标记（例如缺少约定的 Answer 标识）时，运行时会要求模型重新输出，最多重试 3 次，避免因格式问题直接失败或无限重试。
- **使用时机**：`agents/react.py` 解析模型输出后判定格式不合法时。
- **边界**：达到上限后如何收场（报错、降级为兜底答案）由 `agents/react.py` 决定；本文件只给次数。
- **同文件关系**：与 `REACT_MALFORMED_ANSWER_RETRY_LIMIT` 是两种不同格式错误的并列上限。

### `REACT_MALFORMED_ANSWER_RETRY_LIMIT` （第 252 行）
- **类型与取值**：`int`，值为 `3`。
- **作用**：ReAct 轮内「畸形答案」的重试上限。畸形指结构上无法解析（例如动作与参数缺字段、JSON 截断）的输出，同样最多重试 3 次。
- **使用时机**：`agents/react.py` 解析失败分支。
- **边界**：与未标记答案的判定是不同分支，但共享「重试几次后放弃」的设计哲学；`3` 次也是与 `DEFAULT_MAX_RETRIES` 相呼应的数值。
- **同文件关系**：与 `REACT_UNMARKED_ANSWER_RETRY_LIMIT` 成对。

### `MAX_UPLOAD_BYTES` （第 260 行）
- **类型与取值**：`int`，值为 `64 * 1024 * 1024`，即 67108864 字节（64MB）。用乘法表达式书写是为了让人一眼读出「64MB」。
- **作用**：`/api/ingest` 与 `/api/import` 上传大小上限。超过则拒绝请求，防止一次上传把内存或磁盘打爆。
- **使用时机**：FastAPI 路由读取上传体时做前置校验（通常在读取前检查 `Content-Length`）。
- **边界**：拒绝时返回的具体 HTTP 状态码与错误体由 `web/app.py` 决定；分块上传 / 流式上传是否受此限制取决于实现。
- **同文件关系**：与 Web API 分区其它长度上限并列，是唯一以「字节」为单位的限制（其余多为字符数）。

### `WEB_CHAT_MAX_CHARS` （第 263 行）
- **类型与取值**：`int`，值为 `8000`。
- **作用**：`/api/chat` 消息长度上限（字符数）。超长消息直接拒绝，避免单轮对话把上下文打满或造成昂贵计费。
- **使用时机**：聊天接口入参校验。
- **边界**：`8000` 字符对中文约相当于数千 token，加上历史与工具结果后仍可能接近模型上限；本文件只约束单条消息。
- **同文件关系**：与 `WEB_KNOWLEDGE_MAX_CHARS`、`WEB_GRAPH_RAG_QUERY_MAX` 同属 Web 入参长度限制族，但各自对应不同端点。

### `WEB_FACT_SUBJECT_MAX` （第 266 行）
- **类型与取值**：`int`，值为 `200`。
- **作用**：`/api/facts` 三元组中「主语」字段的长度上限。手工录入事实时防止超长字符串污染图谱。
- **使用时机**：事实写入接口的字段校验。
- **边界**：与知识抽取侧的 `ENTITY_NAME_MAX_LENGTH`（同为 200）取值一致，属有意的对齐；本文件不做联动校验。
- **同文件关系**：与下面四个 `WEB_FACT_*` 常量共同定义三元组各字段上限。

### `WEB_FACT_PREDICATE_MAX` （第 267 行）
- **类型与取值**：`int`，值为 `100`。
- **作用**：`/api/facts` 三元组中「谓语 / 关系」字段的长度上限。关系名通常比实体名短，因此上限设为 100。
- **使用时机**：事实写入接口字段校验。
- **边界**：超长被拒；本文件不规定是否截断（由路由决定）。
- **同文件关系**：与另四个 `WEB_FACT_*` 并列。

### `WEB_FACT_OBJECT_MAX` （第 268 行）
- **类型与取值**：`int`，值为 `200`。
- **作用**：`/api/facts` 三元组中「宾语」字段的长度上限，与主语上限一致，保证三元组两端对等。
- **使用时机**：事实写入接口字段校验。
- **边界**：同上。
- **同文件关系**：与另四个 `WEB_FACT_*` 并列。

### `WEB_FACT_DOMAIN_MAX` （第 269 行）
- **类型与取值**：`int`，值为 `100`。
- **作用**：`/api/facts` 三元组中「领域」字段的长度上限。领域名用于星图着色与筛选（对应 `NEBULA_PALETTE` 与 `DEFAULT_DOMAIN`），需要短且规整，因此限制为 100。
- **使用时机**：事实写入接口字段校验。
- **边界**：空领域时通常回落到 `DEFAULT_DOMAIN`（「未分类」），该回落逻辑在业务代码而非本文件。
- **同文件关系**：与 `DEFAULT_DOMAIN`、`NEBULA_PALETTE` 在业务上相关（领域 → 配色），本文件层面无计算关系。

### `WEB_FACT_NOTE_MAX` （第 270 行）
- **类型与取值**：`int`，值为 `4000`。
- **作用**：`/api/facts` 三元组中「备注」字段的长度上限。备注是自由文本、信息量最大，因此上限明显高于其它字段（4000），但仍需封顶。
- **使用时机**：事实写入接口字段校验。
- **边界**：`4000` 字符仍可能被后续切块逻辑影响（备注是否会进 RAG 索引取决于实现）。
- **同文件关系**：与另四个 `WEB_FACT_*` 并列，是其中最大的一项。

### `WEB_GRAPH_RAG_QUERY_MAX` （第 273 行）
- **类型与取值**：`int`，值为 `8000`。
- **作用**：`/api/graph-rag` 查询文本的长度上限。与 `WEB_CHAT_MAX_CHARS` 同为 8000，保持「一次查询不超过 8000 字符」的一致体验。
- **使用时机**：图检索接口入参校验。
- **边界**：超长拒绝；查询本身不写入记忆库（区别于问答留痕）。
- **同文件关系**：与 `WEB_GRAPH_RAG_LIMIT_MAX` 是同一条路由的两个参数上限。

### `WEB_GRAPH_RAG_LIMIT_MAX` （第 274 行）
- **类型与取值**：`int`，值为 `50`。
- **作用**：`/api/graph-rag` 中 `limit` 参数的上限，防止调用方传 `limit=100000` 拉爆响应与检索耗时。
- **使用时机**：图检索接口参数夹取 / 校验。
- **边界**：与记忆层的 `MEMORY_SEARCH_LIMIT`、RAG 的 `RAG_RETRIEVE_LIMIT` 是不同层面的默认值——这里是「外部 API 允许的最大值」，不改变内部默认。
- **同文件关系**：与 `WEB_GRAPH_RAG_QUERY_MAX` 成对；与 `WEB_DOCUMENTS_PAGE_SIZE_MAX` 同属「分页 / 条数上限」思路。

### `WEB_KNOWLEDGE_MAX_CHARS` （第 277 行）
- **类型与取值**：`int`，值为 `20000`。
- **作用**：`/api/knowledge` 一句话文本长度上限。这是全文件里最大的文本上限，因为该端点接收的是整段知识正文而非对话消息。
- **使用时机**：知识写入接口入参校验。
- **边界**：`20000` 字符会进一步被切块（参考 `WEB_INGEST_CHUNK_SIZE`）后入库；本文件只做入口封顶。
- **同文件关系**：与 `WEB_INGEST_CHUNK_SIZE` 在业务流程上前后衔接。

### `WEB_IMPORT_ERRORS_MAX` （第 280 行）
- **类型与取值**：`int`，值为 `20`。
- **作用**：`/api/import` 响应中回传的错误明细条数上限。批量导入时错误可能成百上千条，全量回传会让响应体过大、前端渲染卡顿，因此只回传前 20 条（通常还会带上总数）。
- **使用时机**：组装导入响应体时对错误列表切片。
- **边界**：只影响回传内容，不影响实际导入结果与日志记录；本文件不规定「是否有总数」。
- **同文件关系**：与 `MAX_UPLOAD_BYTES` 同属 `/api/import` 相关的保护性上限。

### `WEB_DOCUMENTS_PAGE_SIZE_MAX` （第 283 行）
- **类型与取值**：`int`，值为 `100`。
- **作用**：`/api/documents` 分页的每页条目上限，注释直白说明是为了防止 `page_size=100000` 把响应拉爆。
- **使用时机**：文档列表接口分页参数校验 / 夹取。
- **边界**：超过上限时是夹到 100 还是报错由路由实现决定；默认 page_size 不在此处定义。
- **同文件关系**：与 `WEB_GRAPH_RAG_LIMIT_MAX` 同属「条数上限」族。

### `WEB_QA_QUESTION_MAX_CHARS` （第 287 行）
- **类型与取值**：`int`，值为 `2000`。
- **作用**：问答留痕写入 episodic 记忆时，问题原文的截断上限。注释说明目的是避免超长对话把记忆库无限撑大，同时保证检索仍能命中问题/答案的关键句。
- **使用时机**：问答完成后写 episodic 记录时。
- **边界**：截断只作用于「留痕副本」，不影响当轮真实对话；截断方式（从头截）由实现决定。
- **同文件关系**：与 `WEB_QA_ANSWER_MAX_CHARS` 成对（问题 2000 / 答案 8000）。

### `WEB_QA_ANSWER_MAX_CHARS` （第 288 行）
- **类型与取值**：`int`，值为 `8000`。
- **作用**：问答留痕写入 episodic 记忆时，答案原文的截断上限。答案通常比问题长，因此上限是问题的 4 倍。
- **使用时机**：问答留痕写入路径。
- **边界**：与 `WEB_CHAT_MAX_CHARS` 数值相同但语义不同——后者限制「输入消息」，本常量限制「留痕答案」，不要混用。
- **同文件关系**：与 `WEB_QA_QUESTION_MAX_CHARS` 成对。

### `WEB_INGEST_CHUNK_SIZE` （第 291 行）
- **类型与取值**：`int`，值为 `800`。
- **作用**：`/api/ingest` 的 RAG 切块块长。注释说明它比默认 `RAG_CHUNK_SIZE`（1000）更细，专门用于文档，目的是让文档检索的粒度更精准。
- **使用时机**：`web/ingest_queue.py` 入库切块时覆盖默认块长。
- **边界**：更小的块意味着同样文档产生更多 chunk、更多嵌入调用与更大的索引；重叠长度是否同步调整由实现决定（本文件未定义 ingest 专用的 overlap）。
- **同文件关系**：是 `RAG_CHUNK_SIZE` 的场景化覆盖值；与 `QA_EXTRACT_CHUNK_SIZE` 并列为另外两种切块场景。

### `QA_EXTRACT_CHUNK_SIZE` （第 294 行）
- **类型与取值**：`int`，值为 `2000`。
- **作用**：问答抽取的切块块长。注释说明问答通常很短，不需要按文档大小切，因此取更大的块（2000）以减少 chunk 数量、降低抽取调用开销。
- **使用时机**：问答后自动抽取图事实的切块阶段。
- **边界**：块大意味着单次抽取提示词更长，可能触及模型上下文上限；由抽取实现保证不超限。
- **同文件关系**：与 `WEB_INGEST_CHUNK_SIZE`、`RAG_CHUNK_SIZE` 构成三种切块粒度；与 `WEB_QA_EXTRACT` 开关在业务流程上衔接。

### `WEB_AUTOSEED` （第 297 行）
- **类型与取值**：`bool`，值为 `True`。
- **作用**：Web 启动时是否自动播种演示数据（首次启动让星图有内容）。注释明确测试会 monkeypatch 它。
- **使用时机**：`web/app.py` 启动钩子（lifespan / startup 事件）里判断是否调用 `web/seed.py`。
- **边界**：开启时若重复播种可能产生重复数据，因此实现侧通常用种子标记（`web/seed.py` 的 `SEED_MARK`，本文件刻意未收录该常量）做幂等判断；本文件只给开关。
- **同文件关系**：与 `WEB_QA_EXTRACT`、`WEB_QA_EXTRACT_SYNC` 同属「Web 运行开关」三件套。

### `WEB_QA_EXTRACT` （第 300 行）
- **类型与取值**：`bool`，值为 `True`。
- **作用**：问答后自动抽取图事实（后台线程）的总开关。注释说明测试会 monkeypatch 关闭它以做隔离，避免后台线程干扰断言。
- **使用时机**：`web/app.py` 问答处理完成后决定是否触发抽取。
- **边界**：关闭后问答仍正常，只是不再自动往图谱里加事实；与 `WEB_QA_EXTRACT_SYNC` 是「开不开」与「同步还是异步」两个正交维度。
- **同文件关系**：与 `WEB_QA_EXTRACT_SYNC` 配合；与 `QA_EXTRACT_CHUNK_SIZE` 在抽取流程上衔接。

### `WEB_QA_EXTRACT_SYNC` （第 303 行）
- **类型与取值**：`bool`，值为 `False`。
- **作用**：抽取线程是否改为同步执行。仅调试 / 测试用，避免后台线程竞争（异步时断言可能跑在抽取完成之前，或线程与测试共享资源产生竞态）。
- **使用时机**：`web/app.py` 触发抽取时选择调用方式。
- **边界**：置 `True` 会让问答请求的响应时间包含抽取耗时，生产环境不应开启；只有 `WEB_QA_EXTRACT` 为 `True` 时本开关才有意义。
- **同文件关系**：与 `WEB_QA_EXTRACT` 成对（前者是总开关，本项是执行模式）。

### `NEBULA_PALETTE` （第 311 行）
- **类型与取值**：`list[str]`，8 个十六进制颜色值：`"#38bdf8"`（青）、`"#c084fc"`（紫）、`"#f43f5e"`（玫红）、`"#fbbf24"`（金）、`"#34d399"`（绿）、`"#60a5fa"`（蓝）、`"#f472b6"`（粉）、`"#a3e635"`（黄绿）。注释说明与 Aetheria 深空青紫主题协调。
- **作用**：星图（知识星云）的领域配色板。不同领域按顺序（或哈希取模）分配颜色，让同一领域的节点在图上颜色一致、视觉上可分组。
- **使用时机**：`tool/graph_snapshot.py` 生成星图快照时按领域取色。
- **边界**：领域数超过 8 时必然出现颜色复用，属可接受设计；顺序即取色顺序，调整顺序会改变既有星图的观感（但不影响数据）。使用 `list` 而非 `tuple`，意味着理论上可被运行时修改，改动需谨慎。
- **同文件关系**：与 `DEFAULT_DOMAIN`（兜底领域名）在展示层配合；与 `WEB_FACT_DOMAIN_MAX`（领域字段长度）无计算关系。

### `DEFAULT_DOMAIN` （第 323 行）
- **类型与取值**：`str`，值为 `"未分类"`。
- **作用**：兜底领域名。当文档 / 事实无法判定领域、或领域字段为空时，统一归到「未分类」，保证星图与筛选器里不会出现空领域分组。
- **使用时机**：领域分类失败路径、事实写入时的空值兜底。
- **边界**：中文取值需与前端展示 / 筛选保持一致；本文件不做枚举约束。
- **同文件关系**：与 `NEBULA_PALETTE`（配色）、`DOMAIN_TITLE_WEIGHT`（分类加权）同属领域分类与星图展示链路。

### `NEBULA_CONTENT_PREVIEW_CHARS` （第 326 行）
- **类型与取值**：`int`，值为 `400`。
- **作用**：星图节点正文预览的截断长度。节点上只显示一小段正文摘要，鼠标悬停或点击才看全文，因此需要截断以控制渲染体积与视觉噪声。
- **使用时机**：`tool/graph_snapshot.py` 生成节点文案时。
- **边界**：数值与 `OBSERVATION_PREVIEW_CHARS` 相同（都是 400），但用途完全不同（星图预览 vs 历史压缩预览），改一个不要顺手改另一个。
- **同文件关系**：与 `NEBULA_EVENT_TITLE_CHARS`、`NEBULA_DATE_CHARS` 同属星图文案截断三兄弟。

### `NEBULA_EVENT_TITLE_CHARS` （第 327 行）
- **类型与取值**：`int`，值为 `24`。
- **作用**：星图事件标题的截断长度。事件标题在图上以短标签形式呈现，24 个字符大致能容纳一个中文短句，超出则截断。
- **使用时机**：`tool/graph_snapshot.py` 生成事件节点标题时。
- **边界**：按字符而非显示宽度计算，中英混排时视觉长度会不一致；属可接受的近似。
- **同文件关系**：与 `NEBULA_CONTENT_PREVIEW_CHARS`、`NEBULA_DATE_CHARS` 同组。

### `NEBULA_DATE_CHARS` （第 328 行）
- **类型与取值**：`int`，值为 `10`。
- **作用**：星图中日期字符串的截断长度。10 个字符正好是 `YYYY-MM-DD` 的长度，因此它实际上是把 ISO 时间戳裁成「只保留日期部分」。
- **使用时机**：`tool/graph_snapshot.py` 处理事件时间字段时。
- **边界**：如果时间戳格式不是 `YYYY-MM-DD...` 开头（例如带时区前缀或本地化格式），裁 10 位会得到无意义字符串；本文件不校验格式。
- **同文件关系**：与另两个 `NEBULA_*_CHARS` 同组。

### `DOMAIN_TITLE_WEIGHT` （第 336 行）
- **类型与取值**：`int`，值为 `3`。
- **作用**：领域分类器中「标题命中」的加权系数。注释给出理由：文件名常含主题词（如「c语言笔记.txt」），标题里出现的领域关键词比正文里的更能代表文档主题，因此命中一次按 3 倍计分。
- **使用时机**：`tool/domain_classify.py` 累计各领域得分时。
- **边界**：设得过大时标题里的一个偶然词就会决定整个文档领域；设为 `1` 等于取消标题加权。本文件只给权重，关键词词表本身（`DOMAIN_KEYWORDS`）刻意未收录。
- **同文件关系**：与 `DEFAULT_DOMAIN` 在分类流程上衔接（加权打分后取最高，失败或全零则落到「未分类」）。

## 三、一句话总览表

说明：本文件不含任何函数、方法或类（无 `def`、无 `class`、无嵌套函数、无 dunder 方法），下表列出文件内全部 79 个模块级常量，一个不漏。

| 函数/类 | 一句话作用 |
| --- | --- |
| `DEFAULT_MAX_RETRIES`（常量） | LLM 客户端共用的最大重试次数默认值 3。 |
| `DEFAULT_TEMPERATURE`（常量） | 模型采样温度统一默认值 0.7。 |
| `DEFAULT_TIMEOUT`（常量） | LLM 请求超时默认值 60 秒。 |
| `PROMPT_CACHE_KEY_VERSION`（常量） | prompt-cache 路由键命名空间版本，改动即作废旧缓存键。 |
| `TOOL_LOOP_SAFETY_LIMIT`（常量） | 未指定 max_rounds 时工具循环允许的最大轮数 64。 |
| `UPDATE_LOG_READ_RANGE_MAX`（常量） | 单次批量读取 update_log 的最大记录数 100。 |
| `OBSERVATION_COMPRESS_THRESHOLD`（常量） | 超过 12000 字符的 Observation 写入历史时被压缩。 |
| `OBSERVATION_STUB_PREFIX`（常量） | 压缩桩文本的前缀标记「[已压缩的历史工具结果」。 |
| `OBSERVATION_PREVIEW_CHARS`（常量） | 压缩后保留的原文预览字符数 400。 |
| `HISTORY_MAX_MESSAGES`（常量） | 单条会话历史保留的最大消息数 60（保留 system 前缀）。 |
| `DEFAULT_TOOLS_DB_FILENAME`（常量） | 工具数据库文件名默认值 tools.sqlite3。 |
| `DEFAULT_UPDATE_LOG_FILENAME`（常量） | 更新日志数据库文件名默认值 update_log.sqlite3。 |
| `DEFAULT_MEMORY_DB_FILENAME`（常量） | 记忆库数据库文件名默认值 memory.sqlite3。 |
| `LOCALHOST`（常量） | 本机服务统一绑定的回环地址 127.0.0.1（避免 localhost 解析到 ::1）。 |
| `DEFAULT_QDRANT_PORT`（常量） | 本地 Qdrant HTTP 端口 6333。 |
| `DEFAULT_NEO4J_BOLT_PORT`（常量） | 本地 Neo4j bolt 端口 7687。 |
| `DEFAULT_WEB_PORT`（常量） | 本地 Web 服务端口 8765。 |
| `DEFAULT_QDRANT_URL`（常量） | 由 LOCALHOST+端口派生的本地 Qdrant 端点（文档化默认，非默认启用）。 |
| `DEFAULT_NEO4J_URI`（常量） | 由 LOCALHOST+端口派生的本地 Neo4j 端点（文档化默认，非默认启用）。 |
| `DEFAULT_PROXY_PORT`（常量） | 本机转发代理端口 7890（Clash 默认）。 |
| `DEFAULT_PROXY_URL`（常量） | 由 LOCALHOST+端口派生的默认代理 URL。 |
| `DEFAULT_EMBEDDING_MODEL`（常量） | 嵌入模型名兜底默认值 qwen3-embedding-0.6b。 |
| `DEFAULT_EMBEDDING_BATCH_SIZE`（常量） | DashScope 风格嵌入批处理上限 10。 |
| `MEMORY_DEFAULT_TTL_SECONDS`（常量） | 工作记忆默认 TTL 3600 秒。 |
| `MEMORY_WORKING_CAPACITY`（常量） | 工作记忆条目容量上限 100。 |
| `MEMORY_SEARCH_LIMIT`（常量） | 语义检索默认返回条数 10。 |
| `MEMORY_SIMILARITY_THRESHOLD`（常量） | 语义检索默认相似度阈值 0.0（0 表示不过滤）。 |
| `MEMORY_EMBEDDING_DIMENSION`（常量） | 离线 HashEmbedding 默认维度 1024。 |
| `MEMORY_EMBEDDING_DIMENSION_REMOTE`（常量） | 远端嵌入期望维度，出厂 None = 首次调用自动识别。 |
| `MEMORY_EMBEDDING_PROVIDER_DEFAULT`（常量） | 嵌入提供方默认 auto（有配置走云端，否则离线兜底）。 |
| `MEMORY_EMBEDDING_PROVIDERS`（常量） | 嵌入提供方合法取值白名单 ("auto","openai","hash")。 |
| `MEMORY_EMBEDDING_TIMEOUT`（常量） | 嵌入请求超时 30 秒。 |
| `MEMORY_SQLITE_DEFAULT`（常量） | 记忆库 SQLite 默认目标 ":memory:"（不持久化）。 |
| `MEMORY_QDRANT_COLLECTION`（常量） | Qdrant 后端默认集合名 helloagents_memory。 |
| `RAG_CHUNK_SIZE`（常量） | RAG 文档切块默认块长 1000 字符。 |
| `RAG_CHUNK_OVERLAP`（常量） | 相邻块重叠长度 100 字符。 |
| `RAG_RETRIEVE_LIMIT`（常量） | 检索/上下文装配默认返回条数 5。 |
| `RAG_GRAPH_HOPS`（常量） | 图检索默认跳数 1。 |
| `RAG_GRAPH_MAX_HOPS`（常量） | 图检索跳数允许上限 3。 |
| `RAG_GRAPH_PATH_LIMIT`（常量） | 图路径展开上限 20。 |
| `RAG_CONTEXT_MAX_CHARS`（常量） | 图检索上下文拼装最大字符数 12000。 |
| `MEMORY_EDGE_WEIGHT_GROWTH`（常量） | 一次回忆把边权重乘以 1.5 倍。 |
| `MEMORY_EDGE_WEIGHT_MAX`（常量） | 边权重上限 8.0，防富者越富。 |
| `MEMORY_EDGE_REINFORCE`（常量） | F1 边强化总开关，默认 True，测试 monkeypatch。 |
| `MEMORY_HYBRID`（常量） | 混合检索（FTS5 × 向量 RRF）总开关，False 则纯向量。 |
| `ENTITY_SIMILARITY_THRESHOLD`（常量） | 实体名相似度合并阈值 0.88。 |
| `ENTITY_DEFAULT_CONFIDENCE`（常量） | 实体解析默认置信度 0.8。 |
| `ENTITY_DEFAULT_TYPE`（常量） | 默认实体类型「概念」。 |
| `ENTITY_NAME_MAX_LENGTH`（常量） | 实体名长度上限 200（解析时截断）。 |
| `ENTITY_CLEAN_MAX_LENGTH`（常量） | 文本清理默认上限 500。 |
| `ENTITY_PREFIX_MIN_LENGTH`（常量） | 别名前缀匹配最短公共前缀 3，避免过短前缀误合并。 |
| `GRAPH_CONTEXT_MAX_RELATIONS`（常量） | 抽取前注入提示词的子图关系条数上限 60。 |
| `REACT_UNMARKED_ANSWER_RETRY_LIMIT`（常量） | ReAct 未标记答案的重试上限 3。 |
| `REACT_MALFORMED_ANSWER_RETRY_LIMIT`（常量） | ReAct 畸形答案的重试上限 3。 |
| `MAX_UPLOAD_BYTES`（常量） | /api/ingest 与 /api/import 上传上限 64MB。 |
| `WEB_CHAT_MAX_CHARS`（常量） | /api/chat 消息长度上限 8000。 |
| `WEB_FACT_SUBJECT_MAX`（常量） | 事实三元组主语长度上限 200。 |
| `WEB_FACT_PREDICATE_MAX`（常量） | 事实三元组谓语长度上限 100。 |
| `WEB_FACT_OBJECT_MAX`（常量） | 事实三元组宾语长度上限 200。 |
| `WEB_FACT_DOMAIN_MAX`（常量） | 事实三元组领域长度上限 100。 |
| `WEB_FACT_NOTE_MAX`（常量） | 事实三元组备注长度上限 4000。 |
| `WEB_GRAPH_RAG_QUERY_MAX`（常量） | /api/graph-rag 查询长度上限 8000。 |
| `WEB_GRAPH_RAG_LIMIT_MAX`（常量） | /api/graph-rag 的 limit 上限 50。 |
| `WEB_KNOWLEDGE_MAX_CHARS`（常量） | /api/knowledge 一句话文本长度上限 20000。 |
| `WEB_IMPORT_ERRORS_MAX`（常量） | /api/import 响应回传错误明细条数上限 20。 |
| `WEB_DOCUMENTS_PAGE_SIZE_MAX`（常量） | /api/documents 每页条目上限 100。 |
| `WEB_QA_QUESTION_MAX_CHARS`（常量） | 问答留痕问题原文截断上限 2000。 |
| `WEB_QA_ANSWER_MAX_CHARS`（常量） | 问答留痕答案原文截断上限 8000。 |
| `WEB_INGEST_CHUNK_SIZE`（常量） | /api/ingest 的 RAG 切块块长 800（比默认更细）。 |
| `QA_EXTRACT_CHUNK_SIZE`（常量） | 问答抽取切块块长 2000。 |
| `WEB_AUTOSEED`（常量） | Web 启动时是否自动播种演示数据，默认 True。 |
| `WEB_QA_EXTRACT`（常量） | 问答后自动抽取图事实（后台线程）总开关，默认 True。 |
| `WEB_QA_EXTRACT_SYNC`（常量） | 抽取是否改为同步执行，仅调试/测试用，默认 False。 |
| `NEBULA_PALETTE`（常量） | 星图领域配色板，8 个深空青紫色系颜色。 |
| `DEFAULT_DOMAIN`（常量） | 兜底领域名「未分类」。 |
| `NEBULA_CONTENT_PREVIEW_CHARS`（常量） | 星图节点正文预览截断长度 400。 |
| `NEBULA_EVENT_TITLE_CHARS`（常量） | 星图事件标题截断长度 24。 |
| `NEBULA_DATE_CHARS`（常量） | 星图日期字符串截断长度 10（YYYY-MM-DD）。 |
| `DOMAIN_TITLE_WEIGHT`（常量） | 领域分类中标题命中的加权系数 3。 |
