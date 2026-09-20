# 02 · core/ —— 工具协议内核

`core/` 是整项目的「工具运行时」：定义工具契约、发现机制、注册表、执行调度、轮次控制，以及两块基础设施（外部服务配置加载、云端代理隧道、更新日志仓储）。上层 `agents/` 与 `tool/` 都建立在这套协议上。

包入口 `core/__init__.py` 只做集中再导出（`ToolCall`、`ToolExecutionManager`、`ToolRegistry`、`ToolLoop`、`discover_tools`、`ToolCatalogTool`、`ToolSpecRepository`、`UpdateLogRepository`、解析函数等），无自身逻辑。

---

## 2.1 core/models.py —— 数据模型（工具世界的「法律」）

全部协议模型都继承 `StrictModel`（`ConfigDict(extra="forbid", strict=True, validate_assignment=True)`：禁未知字段、严格类型、赋值时也校验）。

| 类 | 字段/方法 | 说明 |
| --- | --- | --- |
| `ToolCall` | `type="tool_call"`、`call_id`(1–128)、`tool_name`(1–200)、`schema_version`、`schema_hash`、`registry_generation`(≥1, 可空)、`arguments: dict`、`depends_on: list[str]` | 模型侧发起的一次工具调用请求。depends_on 声明依赖的 call_id，用于分层执行 |
| `ToolError` | `code`(1–64)、`message`(≤2000)、`retryable`(默认 False) | 结构化错误；code 取值见执行管理器 |
| `ToolResult` | `type="tool_result"`、`call_id`、`tool_name`、`ok`、`data`、`error`；`validate_consistency()` | 单个调用结果；模型校验器强制：成功不能带 error、失败必须带 error、失败不能带 data |
| `BatchToolResult` | `results: list[ToolResult]`；`validate_unique_call_ids()` | 批量结果；校验 call_id 唯一，唯一合法例外是运行时对重复 ID 返回的 `DUPLICATE_CALL_ID` 诊断 |
| `ToolSpec`（frozen dataclass） | 见下 | 工具静态元数据；Pydantic 输入/输出模型才是唯一真相 |
| `ExecutionContext`（frozen dataclass） | `subject="default"`、`permissions:frozenset`、`confirmed_side_effects:frozenset` | 一次工具请求的执行上下文。confirmed_side_effects 里放注册表给出的、绑定了代次的确认键；permissions 仅作兼容/审计元数据，不参与访问控制 |

`ToolSpec` 字段与约束（`__post_init__` 逐项校验）：

| 字段 | 约束 | 含义 |
| --- | --- | --- |
| `name` | 必须匹配 `命名空间.名字` 正则 | 工具名，如 `web.search`；OpenAI 接口映射时转成 `web__search` |
| `description` | 非空、≤2000 字符 | 模型可见描述 |
| `version` | 1–32 字符 | Schema 版本 |
| `input_model` / `output_model` | 必须是 Pydantic 模型类 | 输入/输出 Schema 的唯一真相 |
| `side_effect` | 非空 ≤32 字符，默认 `"read"` | 副作用声明；非 read 的执行前必须确认 |
| `permissions` | 字符串元组 | 兼容性元数据，不被强制 |
| `timeout_seconds` | 有限正数，默认 30.0 | 单次执行超时 |
| `idempotent` / `parallel_safe` | bool，默认 True | 幂等性 / 是否可并行 |
| `max_concurrency` | 正整数或 None | 单工具并发上限（None = 用全局上限） |
| `tags` | 字符串元组 | 目录检索标签 |
| `recommended_before_tools` | 命名空间工具名元组 | 建议前置工具；仅作提示，运行时不构成依赖、不阻止直接调用 |
| `_schema_hash` | `init=False`，自动计算 | 输入/输出 JSON Schema 序列化后取 SHA-256 |

`ToolSpec` 的关键属性/方法：
- `schema_hash`：返回上述哈希；调用方在 ToolCall 里携带它，执行时比对防「旧 schema 打新工具」；
- `confirmation_key`：`f"{name}@{version}#{schema_hash}"`，把副作用确认绑定到确切契约；
- `input_schema` / `output_schema`：Pydantic 生成的 JSON Schema；
- `summary()`：目录检索用的摘要字典；
- `model_description`：给模型看的描述，附带前置工具建议（明确标注 advisory only）。

---

## 2.2 core/registry.py —— 工具注册表

`BaseTool`（ABC）：只要求类属性 `spec: ToolSpec` 和抽象方法 `execute(arguments) -> BaseModel | Any`。

`ToolRegistry`：线程安全（`RLock`）的名字→工具表 + 每工具「代次」计数器。

| 方法 | 作用 |
| --- | --- |
| `register(tool, replace=False)` | 注册工具；同名已存在且非 replace 抛错；每次注册该工具代次 +1（重注册也单调递增） |
| `unregister(name)` | 注销并返回实例；保留代次计数，使 unregister/re-register 循环中代次序列保持单调 |
| `resolve(name)` | 原子返回 `(tool, generation)`；未注册抛 KeyError |
| `maybe_resolve(name)` | 同上但未注册返回 None |
| `get / maybe_get` | 取工具实例 |
| `is_registered(name, version, schema_hash)` | 当前注册是否匹配给定契约 |
| `registration_status(name)` | JSON 友好的注册详情（版本/哈希/代次/实现位置 `模块:类名`） |
| `confirmation_key(name)` | 返回 `spec.confirmation_key:generation`——实现一换，旧确认键立即失效 |
| `resolve_all(names=None)` | 批量解析 |
| `specs()` | 全部 ToolSpec 列表 |
| `__contains__ / __len__` | 支持 `in` 与 `len()` |

---

## 2.3 core/discovery.py —— 单文件工具自动发现

`discover_tools(registry, *, package="tool", repository=None, replace=False, strict=False, reload_modules=False, metadata_only=False) -> ToolDiscoveryReport`

扫描规则（源码注释即协议）：
- 只扫 `tool` 包的**直接子模块**；子目录（`ispkg`）不加载；
- 文件名以 `_` 开头或叫 `base.py` 的跳过，记 `ignored`；
- 导入失败记 `error` 并继续扫其它模块（插件是隔离边界，一个坏插件不能拖垮全家）；
- `TOOL_ENABLED` 必须是 bool：缺失/非 bool 记 `error`，为 False 记 `disabled`；
- 必须存在零参可调用的 `create_tool()`，否则 `error`；
- `metadata_only=True` 时：只校验工厂、把类级 `ToolSpec` 写进 repository，**不把可执行代码注册进 registry**（lazy 工作流的基础）；
- 同名不同实现：spec 完全相同记 `already_registered`；不同且 `replace=False` 抛错；
- `strict=True` 且有任一条 error 时抛 `ToolDiscoveryError`；
- 返回 `ToolDiscoveryReport(package_name, records)`，提供 `errors/registered/ok/for_tool/as_dict` 视图。

辅助函数：
- `_load_package`：加载并校验目标包（必须有 `__path__`）；
- `_register_discovered_tool`：工厂产出→类型检查→注册→`_save_spec` 持久化元数据；
- `_register_discovered_metadata`：从模块命名空间里找「恰好一个」本模块定义的 BaseTool 子类（带类级 ToolSpec），只存元数据；
- `_save_spec / _save_metadata`：把 `implementation_ref`（`模块:类名`）连同 spec 写入 `ToolSpecRepository`；
- `_candidate_path`：推模块文件路径（供报告展示）；
- `_error_record`：统一错误记录格式 `前缀: 异常类型: 信息`。

`ToolDiscoveryRecord`（frozen dataclass）：`module / path / enabled / status / tool_name / version / generation / error`。`status ∈ {registered, already_registered, disabled, ignored, error}`。

---

## 2.4 core/runtime.py —— 工具执行管理器（调度核心）

`ToolExecutionManager(registry, max_concurrency=8)`，入口 `execute_batch(calls, context=None, *, failure_policy="continue")`。

执行流水线（批次级）：
1. **准入校验**（逐调用，收集错误码，不中断整批）：
   - `DUPLICATE_CALL_ID`：批内 call_id 重复；
   - `UNKNOWN_TOOL`：工具未注册；
   - `SCHEMA_MISMATCH`：schema_version / schema_hash / registry_generation 与当前注册不一致（防陈旧调用）；
   - `CONFIRMATION_REQUIRED`：`side_effect != "read"` 且确认键不在 `context.confirmed_side_effects`；
   - `INVALID_ARGUMENTS`：输入模型严格校验失败（错误信息只回字段路径+消息，**不回显用户输入值**，防把未信任参数/密钥回泄给模型）；
2. **依赖解析**：`depends_on` 按 call_id 匹配，错误码 `UNKNOWN_DEPENDENCY` / `AMBIGUOUS_DEPENDENCY` / `DEPENDENCY_CYCLE`（自依赖）；
3. **分层调度**：`while remaining` 循环——只有全部依赖已完成的调用才「就绪」；就绪集合里若有 `parallel_safe=False` 的工具，本批只放行它一个（隔离）；否则整批 `asyncio.gather` 并发；
4. **失败策略**：`fail_fast` 时任何一调用失败即把剩余全部标记 `ABORTED`；默认 `continue`；
5. **循环检测**：没有任何就绪调用而仍有剩余 = 依赖成环，全部标记 `DEPENDENCY_CYCLE`。

单调用执行 `_run_one(...)`：
- 依赖失败 → `DEPENDENCY_FAILED`（不执行）；
- 先拿工具信号量、再拿全局信号量（等工具槽位不消耗全局槽位）；
- 按 `spec.timeout_seconds` 设截止时间，用 `asyncio.wait` 等待；
- `tool.execute` 是协程 → 直接超时取消；同步函数 → `asyncio.to_thread` 丢线程，超时不取消线程，而是**保留两个信号量直到线程退出**（`_release_after_background_work` 回调释放），避免超时造成隐藏并发；
- 执行签名自省：`execute` 若声明 `context` 参数，按位置/关键字自动注入；
- 返回过 `output_model.model_validate(strict=True)`，失败 → `INVALID_OUTPUT`；
- 异常归一为 `EXECUTION_ERROR`；超时为 `TIMEOUT`；`retryable` 取 `spec.idempotent`；
- `_limited_message`：把 Pydantic 校验错误压成 `字段路径: 消息`，截断到 2000 字符。

`_loop_state`：每事件循环一份 `_LoopExecutionState(global_semaphore, tool_semaphores)`，让同一个 manager 可以被多次顺序 `asyncio.run` 安全复用；已关闭的循环状态会被清理。

---

## 2.5 core/tool_loop.py —— 对话轮次控制

`ToolLoop(max_rounds=None, safety_limit=64)`（`DEFAULT_SAFETY_LIMIT` 是 `TOOL_LOOP_SAFETY_LIMIT` 的兼容别名）：
- `rounds()`：生成 1 开始的轮号，直到 max_rounds 或安全上限；
- `record_call(name, arguments)`：把调用签名（name+arguments 的规范化 JSON）计数，**同一签名出现第 4 次**时抛 RuntimeError——专门治「模型鬼打墙重复同一调用」。

---

## 2.6 core/catalog.py —— 工具目录（system.tool_catalog）

`ToolCatalogTool`：只读门面，**绝不执行任意 SQL**，`side_effect="read"`。

`CatalogInput`：`action ∈ {search, get_spec, resolve}`、`intent`(≤500)、`tool_name`、`version`、`limit`(1–20)；`model_validator` 校验 get_spec 必须带 tool_name、resolve 必须带 intent。
`CatalogOutput`：`candidates`（摘要列表）、`spec`（兼容旧调用方的单个）、`specs`（全部命中的完整契约）。

`execute` 行为：
- `get_spec`：先查活动注册；未加载但 repository 里有元数据时返回存储的完整 schema（这正是 lazy 工作流：模型能发现并加载还没注册实现的工具）；版本不符/元数据不同步 → ValueError；
- `search`/`resolve`：有 repository 且（`repository_only` 或 registry 里除目录外没加载过工具）时走仓储搜索；否则走活动注册表搜索 `_search_specs`；
- `resolve` 返回全部命中的完整契约（一次调用可加载多个能力）；无命中抛 `no matching tool found`。

内部方法：
- `_active_generation(spec)`：取活动代次，期间 spec 变化则抛「catalog changed while the request was running」；
- `_has_loaded_tools()`：registry 里除目录本身外是否有可执行工具；
- `_stored_generation(stored)`：仓储代次；未加载返回 0（= catalog-only）；
- `_search_specs`：对 name/description/tags 做关键词命中计数排序；
- `_intent_terms(intent)`：**中英双语能力词归一**——把「日志/记录」映射到 log、audit；「查询/查看」映射到 read/retrieve/search 等。解决「模型用中文描述能力、工具契约是英文」导致的检索全部落空；
- `_full_spec / _full_stored_spec`：把 ToolSpec 或仓储行转成完整契约字典（含版本、哈希、代次、输入输出 schema、权限、超时、并发标记）。

---

## 2.7 core/parser.py —— 模型输出解析（容错 JSON）

| 函数 | 作用 |
| --- | --- |
| `parse_tool_calls(payload)` | 解析原生或回退 JSON 调用：字符串先过 `loads_model_json`；dict 取 `tool_calls` 键或本身；list 逐条严格校验成 ToolCall。只解析，不执行 |
| `parse_openai_tool_calls(tool_calls, registry, name_map=None, registrations=None)` | 把 OpenAI 原生 tool_calls 转成内部带版本封套的 ToolCall：支持 `web__search → web.search` 名字映射；call_id 缺失时生成稳定的 `native-call-N`；通过注册表解析出版本/哈希/代次；未知工具抛 ValueError |
| `_get(value, key, default)` | 同时支持 dict 与对象属性取值（网关响应两种形态都可能） |
| `loads_model_json(value, object_only=False)` | 三级容错：①原文直解；②截取第一个配平的 `{...}`/`[...]` 块（模型在 JSON 外加白话时）；③Python 风格单引号字面量修复。`NaN/Infinity` 经 `parse_constant=reject_json_constant` 拒绝；所有失败统一抛 `JSONDecodeError`，调用方只需处理一种错误。`object_only=True` 时只找对象块（ReAct 的 Action Input 必须是对象，防止前面的数组块抢先匹配） |
| `_balanced_json_substring` | 手写状态机找配平 JSON 块（正确处理字符串内转义） |
| `_repair_single_quoted_object` | `ast.literal_eval` 安全求值后转标准 JSON |
| `reject_json_constant` | `NaN`/`Infinity` 拒绝钩子 |

---

## 2.8 core/repository.py —— 工具元数据仓储（SQLite）

`ToolSpecRepository(path="tools.sqlite3")`：持久化**可发现工具的元数据，绝不存可执行代码**。

- `_connect / _connection / close`：文件库每次短连接、内存库共享连接；`RLock` 串行化；建库时自动 `mkdir -p` 父目录；
- `_initialize`：建 `tool_specs` 表（主键 `(tool_name, version)`；列：description、schema_hash、input/output_schema、side_effect、permissions、timeout_seconds、idempotent、parallel_safe、max_concurrency、tags、recommended_before_tools、enabled、implementation_ref），并对旧库做 `ALTER TABLE` 增量迁移；
- `save(spec, implementation_ref, replace)`：插入/替换一行元数据；
- `get(tool_name, version)`：取指定或最高版本（`_version_key` 语义化排序：数字段按数值比，`1.0` 新于 `1.0-alpha`）；
- `search(intent, limit)`：关键词命中排序；先按工具名取最新版再去重；
- `active_tool_names()`：全部启用工具名（不受 search 的 limit 限制）；
- `_decode`：JSON 列解码 + bool 还原。

---

## 2.9 core/services_config.py —— 外部服务配置加载

「所有外部 API 调用只有这一个入口」。读取 `config/services.toml`（不入库；模板 `config/services.example.toml`），优先级：**显式参数 > services.toml**（历史上的 .env 优先级层已移除）。

数据类（frozen dataclass，缺省全 None）：
- `EmbeddingService`：provider/base_url/model/api_key/dimension/batch_size/timeout
- `VisionService`：只有 model（端点与凭证复用 provider.toml 的聊天 API）
- `SearchService`：base_url/api_key/timeout
- `QdrantService`：url/api_key/collection
- `Neo4jService`：uri/username/password
- `ProxyService`：url（本机 CONNECT 转发代理）
- `ServicesConfig`：六个段 + `configured` 属性（任一段有值即 True）

函数：
- `default_config_path()`：优先 `services.toml`，回落模板；
- `load_services_config(path)`：文件不存在返回空配置（项目必须离线可用）；TOML 语法错误才抛错；
- `_parse_embedding/_parse_vision/_parse_search/_parse_qdrant/_parse_neo4j/_parse_proxy`：逐段解析并校验 URL scheme（Neo4j 额外允许 bolt/bolt+s/neo4j/neo4j+s）；
- `_resolve_secret(table, key, env_key)`：明文值优先；否则解析 `<key>_env` 指向的环境变量；
- `_optional_string/_optional_positive_int/_optional_positive_float`：空串视为「未配置」，正值约束，bool 不算整数；
- `_require_scheme`：URL 必须是白名单 scheme + 有 netloc。

---

## 2.10 core/proxy_tunnel.py —— 云端 Neo4j 代理隧道（纯标准库）

背景：Neo4j Python 驱动没有原生代理支持；Neo4j Aura 的 bolt 端口要经本机代理（如 Clash:7890）出去。

- `_should_proxy_host(host)`：只代理 `*.neo4j.io` / `*.databases.neo4j.io`，回环地址绝不代理；
- `ConnectTunnel`：每个主机名一个回环监听器；`_accept_loop` 接受连接 → 每连接两个 daemon 线程 `_pipe` 双向搬运；`_relay` 先向代理发 `CONNECT host:port`，校验 `HTTP/1.x 200` 后开始中转；`_read_connect_response` 只给 CONNECT 响应设 10 秒读超时，随后恢复阻塞模式（否则空闲会把 Neo4j 长连接误判为断链）；
- `ProxyBroker`：`ensure(host, port)` 按 (host, port) 缓存隧道（Aura 路由会下发成员主机名，一个入口隧道不够）；`resolve(address)` 与 `remap_getaddrinfo(original)` 把 Aura 主机映射到 `127.0.0.1:<隧道端口>`，同时保留真实主机名给驱动做 SNI/证书校验（直接改 DNS 会导致路由失败）；
- `_numeric_port`：端口归一化，非法值回落 7687；
- `_pipe(source, sink, half)`：单向搬运，源关闭时对 sink 半闭（`shutdown(SHUT_WR)`），保证长连接语义。

---

## 2.11 core/update_log.py —— 项目更新日志仓储

`UpdateLogRepository(path=None)`：默认读 `UPDATE_LOG_DB_PATH` 环境变量或 `update_log.sqlite3`；`RLock` + 内存库共享连接；`busy_timeout=10000`、`foreign_keys=ON`。

- `_initialize`：建 `update_logs` 表：`update_id INTEGER PRIMARY KEY AUTOINCREMENT`、timestamp、system_name、executor、update_type、title、task_background、update_details、added_features、files_json、behavior_impact、validation、risks、follow_up；
- `append(...)`：**一次一行、事务内写入**；timeout/system_name 仅数据迁移时可传，正常调用由仓储填真实 UTC 时间与 `platform.system()`；files 逐项校验 path/action/description 非空；返回 `{update_id, timestamp, system_name, next_update_id}` 的简短确认（历史正文不进模型上下文）；
- `get(update_id)`：审计/测试用单条查询；
- `latest_id()`：当前最大 ID；
- `get_range(start_id, end_id)`：一次连接一条 SQL 取闭区间（替代旧的「每个 ID 开一次连接」）；
- `_decode`：`files_json` 还原为 `files` 键。

---

## 2.12 core/activity_log.py —— 控制台活动日志

logger 名 `all_agent.activity`，自定义 `_ConsoleLogHandler` 用 `print` 输出（交互用户总能看到），格式 `[HH:MM:SS] 消息`，`propagate=False`（不污染根 logger）。

| 函数 | 行为 |
| --- | --- |
| `log_tool_registration` / `log_discovery_summary` / `log_react_round_started` / `log_react_final_answer` | **故意静默**（注册、发现摘要、轮次开始、最终答案都属于噪音或已由调用方返回） |
| `log_model_completed` | 每轮模型耗时 |
| `log_model_first_chunk` | 流式首字延迟 |
| `log_react_thought` | 模型思考（截断 2000 字符） |
| `log_tool_call_started / completed` | 只打印工具名与耗时，**永不打印参数与返回值**（大结果不刷屏） |
| `log_react_parse_issue` | 协议错误 warning，便于诊断卡死 |
| `_tool_names / _clean_text` | 去重排序 / 折叠空白截断 |
