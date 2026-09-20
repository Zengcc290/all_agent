# 05 · tool/ —— 单文件工具插件层

`tool/` 里每个 `.py` 文件都是一个可被自动发现的插件。发现规则在 `core/discovery.py`（详见 `02-core.md`）：**文件名不能以 `_` 开头、不能叫 `base.py`、不能放子目录；模块必须定义布尔常量 `TOOL_ENABLED`（严格等于 `True` 才会被调用工厂）和一个零参数 `create_tool()` 工厂**。

包入口 `tool/__init__.py` 只静态导出 `BaseTool`/`ToolRegistry`（从 `core.registry` 转出，保持历史导入路径可用），`SearchInput/SearchItem/SearchOutput/SearchTool` 通过 `__getattr__` **懒加载** `tool.search` —— 这样某一个坏掉的实现不会让整个包在发现器隔离它之前就 import 失败。

---

## 5.1 模块协议（来自 tool/tool_template.py 的规范正文）

`tool/tool_template.py` 既是模板也是给代码生成模型的完整规范，其 docstring 逐条列出了硬约束，这里摘录要点：

**模块协议**：直接放在 `tool/` 下；`TOOL_ENABLED` 必须是布尔 `True`（不能是字符串 `"true"`）；`create_tool()` 必须零参数且返回 `BaseTool` 实例（不能返回类/字典/函数/None）；一个文件只提供一个发现工厂；导入阶段只能做轻量操作（不能访问网络、写文件、起线程、改数据库）。

**Input/Output 模型规则**：必须继承 `BaseModel` 且 `ConfigDict(extra="forbid", strict=True)`；Pydantic 是唯一协议来源，不要另写 JSON Schema、不要在 `execute()` 里再解析自然语言；每个字段都要有 `description` 与长度/范围约束；OpenAI strict function schema 会把所有 properties 声明为 required，所以不要依赖「LLM 会省略带默认值字段」；**不要用允许任意键的 `dict[str, ...]`**（严格 schema 不允许 `additionalProperties`）；Output 必须稳定、精简、可序列化，不返回第三方 SDK 对象/HTTP Response/异常/含密钥内容。

**ToolSpec 字段选型**：`name` 形如 `namespace.tool_name`（点会映射成双下划线，映射后 ≤64 字符，不要把版本写进名字）；`description` 面向 LLM 说明何时用、不能做什么；`version` 在 schema 或可观察语义变化时升级；`side_effect` 只有精确的 `"read"` 被当作无写副作用，其它非空值都会要求调用方给确认；`permissions` 当前部署不执行授权过滤（历史兼容/审计元数据，新工具通常写 `()`）；`timeout_seconds` 是单次执行的 Runtime 截止时间，且工具内部 I/O 也要设不超它的超时；`idempotent` 决定错误是否标记可重试（不代表 Runtime 自动重试）；`parallel_safe=False` 时实际并发被限制为 1；`max_concurrency=None` 表示用 Runtime 全局上限；`tags` 供 Catalog 搜索；`recommended_before_tools` 只是提示，运行时不会自动调用或校验顺序。

**execute() 规则**：默认同步签名 `execute(self, arguments: XxxInput) -> XxxOutput`（Runtime 会把同步实现放进工作线程）；原生异步可直接 `async def`；需要主体/权限信息时用 `execute(self, arguments, *, context: ExecutionContext)`（Runtime 自动注入，**绝不允许 LLM 在 Input 里伪造权限或确认字段**）；不要自行返回 `ToolResult`，正常返回 Output，失败抛异常由 Runtime 转成结构化 `EXECUTION_ERROR`；写操作尽量幂等；不要在工具内做无限重试；避免类级共享可变状态。

**运行时链路**：Agent 发现模块 → 检查开关 → 调用工厂 → Registry 注册 → Input 生成工具 Schema → LLM 返回 tool_call → Parser 绑定版本/Schema 哈希/注册代次 → Runtime 检查副作用确认并验证 Input → `execute()` → Runtime 验证 Output → ToolResult 返回 LLM。

模板自带示例 `template.text_statistics`（`TOOL_ENABLED = False`，不会被当成业务工具）：统计文本的 Unicode 字符数与按空白分词的词数。

---

## 5.2 tool/base.py 与 tool/_shared.py / tool/_memory.py

### tool/base.py（3 行）

纯兼容转出：`from core.registry import BaseTool, ToolRegistry`。新代码直接用 `core.BaseTool`。

### tool/_shared.py —— 文件系统工具的路径沙箱（**不可被发现**）

以 `_` 开头、没有 `TOOL_ENABLED` 也没有 `create_tool()`，因此发现器会忽略它；导入期无任何网络/写文件/起进程副作用。

| 名称 | 作用 |
| --- | --- |
| `WORKSPACE_ROOT_ENV = "WORKSPACE_ROOT"` | 工作区根目录环境变量 |
| `ALLOW_OUTSIDE_ENV = "WORKSPACE_ALLOW_OUTSIDE"` | 是否允许越界开关 |
| `workspace_root()` | 有配置取配置（expanduser + resolve），否则进程 CWD |
| `allow_outside_workspace()` | 读环境开关，`1/true/yes/on`（大小写不敏感）为真 |
| `resolve_path(base_dir, path, allow_outside)` | 相对路径锚定 `base_dir`；绝对路径必须落在解析后的 `base_dir` 内，否则抛错。`allow_outside` 是**部署决策**，由工具所有者决定，绝不由 LLM 通过输入字段决定 |
| `_is_within(base, candidate)` | `os.path.normcase` 归一后比较前缀（Windows 大小写不敏感） |

### tool/_memory.py —— 记忆工具共享件（**不可被发现**）

同样以 `_` 开头；导入期不开库、不联网，后端都在首次使用时才构建。

| 名称 | 作用 |
| --- | --- |
| `MemoryScope` | `Literal["working","episodic","semantic","perceptual"]` |
| `MemoryMetadata` | `{"key": str, "value": str}`，`extra="forbid", strict=True` |
| `normalize_metadata_payload(v)` | 兼容模型常吐的 `{"metadata": {"k": "v"}}` 映射形态，转成列表形态 |
| `metadata_dict(entries)` | 列表 → dict（manager 需要的形态） |
| `build_default_manager()` | 打开共享记忆库：`MemoryConfig.from_config()` + `default_sqlite_path()`（与 `web.support.get_manager` 同一口径，保证 Agent 工具与 Web API 共用同一个库） |
| `build_default_pipeline()` | 构建默认 RAG 管道。配置了真实 provider 就用 `LLMKnowledgeExtractor`（视觉模型名来自 `config/services.toml` 的 `[vision]` 段）；缺 key / 占位 key / 配置读失败一律降级为 `NullKnowledgeExtractor` 并把原因写进日志 |

---

## 5.3 系统类工具

### tool/current_time.py —— `system.current_time`

- `CurrentTimeInput`：空模型（该工具不需要参数）；
- `CurrentTimeOutput`：`local_time`（ISO 8601 带 UTC 偏移）、`timezone_name`（取不到回落 `"UTC"`）、`unix_timestamp`；
- `CurrentTimeTool.execute`：`datetime.now().astimezone()`，`del arguments` 后返回；
- spec：`side_effect="read"`、`timeout=5s`、`idempotent=True`、`parallel_safe=True`。

### tool/update_log.py —— `system.update_log`

只写不读的追加式更新日志（AI 调用方只拿回新 ID，历史日志不占模型上下文）。

- `UpdateLogFileChange`：`path` / `action`(added/modified/deleted/renamed/generated) / `description`；
- `UpdateLogInput`：`executor`、`update_type`、`title`、`task_background`、`update_details`、`added_features`、`files`(1–100)、`behavior_impact`、`validation`、`risks`、`follow_up`，全部带长度上限；
- `UpdateLogOutput`：`update_id`、`next_update_id`、`timestamp`、`system_name`、`recorded=True`；
- `execute`：调 `UpdateLogRepository.append(...)`，`system_name=platform.system()`；
- spec：`side_effect="write"`、`permissions=()`（写确认已足够，不再叠权限门）、`max_concurrency=1`、`parallel_safe=False`、`idempotent=False`。

### tool/read_update_log.py —— `system.read_update_log`

按单个数字 ID 读一条完整记录。**故意只做单条查找**，绝不列表/全量加载历史。输出除全部字段外还带 `latest_update_id`，调用方可据此从 1 顺序逐条审计。`update_id` 不存在抛 `LookupError`。`timeout=5s`、`max_concurrency=8`。

### tool/read_update_logs.py —— `system.read_update_logs`

一次读一段连续区间（上限 `constants.UPDATE_LOG_READ_RANGE_MAX`）。仓储侧仍是一条一条查（`get_range` 存在则用之，否则逐条 `get` 兼容轻量替身）；返回前校验条数一致，缺哪条就把那条 ID 报出来（`LookupError`）。这样一次完整审计只需少数几次模型工具调用。

---

## 5.4 记忆类工具

四个记忆工具共享同一套后端注入模式：`__init__(manager=None)` 只存字段，`manager` 是**惰性 property**（首次访问才 `build_default_manager()`）——导入/发现阶段绝不打开 SQLite。

### tool/memory_query.py —— `memory.query`（只读）

`MemoryQueryInput`：`action ∈ {search, get, list}`、`memory_type`（search 时留空表示四层全搜）、`item_id`、`query`、`metadata`、`limit`(1–100)。

`execute` 分三支：`search` 走 `manager.search`（`query` 必填）；`get` 走 `manager.get`（`item_id` 必填）；`list` 走 `manager.list` 后截断到 `limit`。返回 `MemoryQueryOutput(action, count, items)`。spec：`side_effect="read"`、`permissions=("memory.read",)`、`parallel_safe=True`、`idempotent=True`。

### tool/memory_add.py —— `memory.add`（写）

`MemoryAddInput`：`content`、`memory_type`（默认 working，描述里说明持久化用户事实应放 episodic/semantic）、`item_id`、`metadata`、`importance`(0–1)、`ttl_seconds`(>0)；`model_validator(mode="before")` 做 metadata 形态归一。

`execute`：`manager.add(...)` 后返回 `{action:"add", count:1, items:[item.to_dict()]}`。spec：`side_effect="write"`、`permissions=("memory.write",)`、`parallel_safe=False`、`idempotent=False`。

### tool/memory_tool.py —— `memory.manage`（写，管理面）

`MemoryManageInput`：`action ∈ {delete, clear}`、`memory_type`（默认 `working`，**「clear 默认不抹掉整个库」是有意的防误删设计**）、`item_id`。

`execute`：`delete` 必须给 `item_id`，返回删除条数；`clear` 调 `manager.clear(memory_type=...)` 返回清掉条数。spec：`side_effect="write"`、`permissions=("memory.write",)`。

### tool/memory_propose_delete.py —— `memory.propose_delete`（F2，读语义）

LLM 必须先检索才允许提议删除；提议本身**从不删除任何东西**，只落一条等待用户确认的记录（执行走 `execute_deletion`）。因此它声明为 `side_effect="read"`：聊天链路无需写确认，同时也没有任何人能通过它直接删除。

`MemoryProposeDeleteInput`：`action ∈ {by_query, by_ids, by_relation}`、`target`（分别是检索串 / item_id 列表 / `subject predicate object` 三元组文本）、`reason`（必填，没有理由的提议直接被拒）。

`_candidates(arguments)`：三种取候选方式，返回 `item_id/content_preview(120 字)/memory_type/why` 列表；查不到就返回空并附说明、不建提议。拿到候选后创建 `DeletionProposalStore`（`:memory:` 时复用 `manager.document_store.connection` 与 memories 同库），返回 `proposal_id`/`confirm_token`/`expires_at`/`items`。

### tool/rag_search.py —— `memory.rag_search`（只读）

`RAGSearchInput`：`action ∈ {retrieve, context, graph_retrieve, graph_context}`、`query`、`limit`(1–50)、`hops`(0–3)。

`execute` 四支：`retrieve` → `pipeline.retrieve`；`graph_retrieve` → `pipeline.graph_retrieve`（返回证据、实体、路径与 `build_context()`）；`graph_context` → `pipeline.graph_context`；默认 `context` → `pipeline.build_context`。`timeout=30s`、`parallel_safe=True`。

### tool/rag_tool.py —— `memory.rag`（写）

`RAGToolInput`：`action="ingest"`、`text` 与 `source` **二选一**（都给了或都没给都报错）、`chunk_size`(1–100000，默认 1000)、`overlap`(≥0，默认 100)。

`execute`：`text` 直接 `pipeline.ingest(Document(text))`；`source` 先过 `resolve_path(workspace_root(), source)`（**模型给的路径也走同一套工作区沙箱**）再 `pipeline.ingest_source(...)`。返回 `count/items/report(pipeline.last_ingest_report)`。spec：`side_effect="write"`、`permissions=("memory.write",)`、`parallel_safe=False`。

---

## 5.5 tool/search.py —— `web.search`（AnySearch）

把 AnySearch 的 `POST /v1/search` 封装成标准单文件工具。

### SearchInput

`query`(1–500)、`max_results`(1–10，`validation_alias=AliasChoices("max_results","limit")` 兼容旧参数名、`serialization_alias="max_results"`)、`tag`、`zone(cn/intl)`、`language`、`params`、`format(json/markdown)`。

三个 `mode="before"` 校验器专门收拾弱工具调用客户端：

- `normalize_tag`：把 `""`/`"None"` 当空值；模型误把工具名（`web`/`web.search`/`web__search`）抄进 `tag` 时直接丢弃（工具名不是能力标签）；
- `normalize_language`：同样归一空值，但不接受工具名别名；
- `normalize_provider_params`：接受 JSON 字符串形式的对象（如 `"{}"`）并解析；任意其它字符串原样留下、交给 strict Pydantic 拒绝。

`params` 用 `Annotated[..., WithJsonSchema({...additionalProperties: False...})]`：公开 schema 保持合法，Pydantic 仍按 dict 校验（严格函数 schema 不允许 `additionalProperties: true`）。`limit` property 是旧字段的兼容视图。

### SearchItem / SearchOutput

`SearchItem`：`title`(≤1000)/`url`(≤2000)/`snippet`(≤5000)；`SearchOutput.items`，`results` 是别名属性。

### SearchTool

构造时读 `config/services.toml` 的 `[search]` 段（**唯一配置来源**，历史 `SEARCH_*`/`ANYSEARCH_*` 环境变量入口已删除）；`base_url`/`api_key`/`timeout` 都可显式注入。校验 `base_url` 非空、`timeout` 有限正数，并用 `dataclasses.replace(type(self).spec, timeout_seconds=self.timeout)` 生成实例级 spec（配置确实改变 Runtime 合约时才这么做，schema/版本/权限不悄悄漂移）。spec 还带 `recommended_before_tools=("system.current_time",)`。

`execute`：组请求体（只带非 None 字段）→ `json.dumps(..., allow_nan=False)` → POST；`HTTPError` 且码为 501 时回落旧式 GET 端点（兼容很老的 AnySearch 网关）→ 用 `core.parser.reject_json_constant` 解析（拒绝 NaN/Infinity，注意它抛的是裸 `ValueError`，必须和 `JSONDecodeError` 一起捕获）→ `_normalize_response`。

模块函数：

- `_services_search()`：取 `[search]` 段，文件缺失/空则全 None；
- `_normalize_nullable_text(v)`：空串 / `"none"` / `"null"` → None；
- `_search_endpoint(base_url)`：强制 `http/https` + netloc；路径为空补 `/v1/search`，以 `/v1` 结尾补 `/search`；
- `_legacy_search_endpoint(endpoint, args)`：旧 GET 的 `?q=&limit=` 形态；
- `_read_response_body(response)`：按 `Content-Length` 预检上限 2,000,000 字节，读取也按上限多读 1 字节判断；兼容只暴露无参 `read()` 的测试替身；
- `_normalize_response(payload, max_results)`：校验 `code ∈ {None, 0}`；结果取 `data.results`，兼容旧网关的顶层 `items`/`results`；逐条映射 `title/url(or link)/snippet(or description)` 并截断；
- `_text(v)`：None → 空串；
- `create_tool()`：返回配置好的实例。

---

## 5.6 工具清单速查

| 工具名 | 文件 | 副作用 | 权限标记 | 并发 |
| --- | --- | --- | --- | --- |
| `system.current_time` | `current_time.py` | read | `()` | 可并行 |
| `system.update_log` | `update_log.py` | write | `()` | 1 |
| `system.read_update_log` | `read_update_log.py` | read | `()` | 8 |
| `system.read_update_logs` | `read_update_logs.py` | read | `()` | 4 |
| `memory.query` | `memory_query.py` | read | `memory.read` | 可并行 |
| `memory.add` | `memory_add.py` | write | `memory.write` | 串行 |
| `memory.manage` | `memory_tool.py` | write | `memory.write` | 串行 |
| `memory.propose_delete` | `memory_propose_delete.py` | read | `memory.read` | 可并行 |
| `memory.rag_search` | `rag_search.py` | read | `memory.read` | 可并行 |
| `memory.rag` | `rag_tool.py` | write | `memory.write` | 串行 |
| `web.search` | `search.py` | read | `()` | 8 |
| `template.text_statistics` | `tool_template.py` | read | `()` | 8（**默认关闭**） |
