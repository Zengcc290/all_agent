# 06 · web/ —— FastAPI 应用层（知识星云）

`web/` 把 `memory/` 四层记忆系统暴露成 HTTP API，并以静态文件方式托管星云图前端（`web/static/index.html`）。入口 `web/__init__.py` 只导出 `create_app`。

运行方式：`python -m web.app`，默认监听 `constants.LOCALHOST:8765`（`uvicorn.run(app, host=LOCALHOST, port=DEFAULT_WEB_PORT)`）。

---

## 6.1 路由总览（web/app.py 顶部 docstring 与代码对照）

| 方法与路径 | 函数 | 作用 |
| --- | --- | --- |
| `GET /api/graph` | `graph(since, at)` | 全图 nodes+edges（星云图数据源），带 revision 缓存 |
| `GET /api/graph-rag` | `graph_rag(body)` | 向量证据 + 图关系路径混合检索 |
| `POST /api/chat` | `chat(body)` | 与知识管家对话（未配置聊天模型时 503） |
| `POST /api/ingest` | `ingest(file, confirm_rebuild)` | 上传文档 → RAG 切块入库（星云长出新星星） |
| `POST /api/facts` | `add_fact(body)` | 手工添加三元组知识 |
| `POST /api/knowledge` | `add_knowledge(body, confirm_rebuild)` | 一句话入库：原文向量化 + LLM 自动抽取实体/关系 |
| `GET /api/knowledge/jobs` | `list_knowledge_jobs(status, limit)` | 一句话入库历史（后台队列状态） |
| `POST /api/knowledge/jobs/{job_id}/retry` | `retry_knowledge_job(...)` | 失败任务重新入队 |
| `POST /api/knowledge/image` | `add_image_knowledge(...)` | 图片/相机 → VL 嵌入 + 视觉模型抽取 |
| `POST /api/seed` | `reseed()` | （重新）播种 Aetheria 种子数据（幂等） |
| `GET /api/export` | `export(request)` | 导出全部记忆为 JSON 文件 |
| `POST /api/import` | `import_file(file)` | 导入此前导出的 JSON |
| `GET /api/documents` | `list_documents(...)` | 文档中心列表（分页 + 标签/状态过滤） |
| `GET /api/documents/{document_id}` | `get_document(...)` | 单个文档 + 其全部 chunk |
| `POST /api/documents/{document_id}/revectorize` | `revectorize_document(...)` | 重建该文档的向量投影 |
| `GET /api/stats` | `stats()` | 计数统计 |
| `GET /api/reconcile` | `reconcile()` | 三库对账（只看不改） |
| `POST /api/reconcile` | `reconcile_repair(body)` | 对账修复（幂等自愈） |
| `POST /api/embedding/rebuild` | `rebuild_embedding(...)` | 确认后重建向量投影 |
| `GET /api/health` | `health()` | 健康检查：嵌入模式、聊天可用性、三存储模式 |
| `/`（静态挂载） | — | 星云图前端（注册在 API 路由之后，避免吞掉 `/api/*`） |

---

## 6.2 web/app.py —— 应用工厂与路由实现

### 模块级辅助

| 名称 | 作用 |
| --- | --- |
| `_reject_json_constant(value)` | 拒绝 `NaN/Infinity`（导入 JSON 严格有限） |
| `ReconcileBody` | `{repair: list[str]}`，`repair` 为空表示只报告不修 |
| `_is_fact_item(item)` | 判断语义条目是否为 `(subject, predicate, object)` 三元组事实 |
| `embedding_config_hint(manager)` | 云端嵌入未配置时的指引文案（供 `/api/health` 的 degraded 字段；历史版本的「网关不可达 → 503 门禁」已删除，端点失败在请求时自然报错） |
| `ChatBody` | `message`(1–`WEB_CHAT_MAX_CHARS`) + `mode`(offline/online) |
| `FactBody` | `subject/predicate/object/domain/note/confidence`，全部有长度/范围约束（`WEB_FACT_*` 常量） |
| `GraphRAGBody` | `query`(1–`WEB_GRAPH_RAG_QUERY_MAX`) + `limit`(1–`WEB_GRAPH_RAG_LIMIT_MAX`) + `hops`(0–`RAG_GRAPH_MAX_HOPS`) + `at`(≤80) |
| `KnowledgeBody` | `text`(1–`WEB_KNOWLEDGE_MAX_CHARS`) + `event_at`(≤80) + `wait`(默认 True：同步等结果；False 提交后台队列立即返回) |
| `_save_upload(file, prefix, suffix)` | 流式写临时文件并强制 `MAX_UPLOAD_BYTES`：1MB 分块读，超限 413、空文件 400；**两个上传端点共用**（历史只有 /api/ingest 有界，`file.read()` 整读可能撑爆进程）；调用方负责 unlink |

### create_app(manager=None) —— 应用工厂

`manager` 可注入（测试用内存库），默认用共享单例 `get_manager()`。

**lifespan**（启动/关闭钩子）：

1. `app.state.manager` = 注入或 `get_manager()`；
2. `app.state.pipeline` = `RAGPipeline(manager, extractor=build_knowledge_extractor())`；
3. `app.state.ingest_queue` = `IngestJobQueue(manager, pipeline.extractor, on_progress=lambda _: invalidate_graph())` 并 `start()`（`:memory:` 测试库下 `available=False`）；
4. `app.state.chat_lock = asyncio.Lock()`：知识管家是**进程级单例且对话历史是共享可变状态**，并发问答会互相覆盖历史，所以串行化聊天请求（单人本地应用，排队可接受）；
5. `WEB_AUTOSEED` 开时 `seed(app.state.manager)`（首启自动播种）；
6. `yield` 后：`ingest_queue.shutdown()`，若 manager 是自建则 `close_manager()`。

**闭包辅助**：

- `the_manager()` → `app.state.manager`；
- `guard_embedding(confirm_rebuild=False)`：写向量前的闸门，`apply_embedding_lock` 不一致时抛 409（`EmbeddingLockMismatch.to_detail()`）；
- `raise_embedding_http(exc)`：把 Qdrant 维度异常归一成同一个 409 载荷；
- 星云图缓存 `graph_cache = {"external": graph_revision(), "payload": None}`：任何写操作调 `invalidate_graph()`（`bump_graph_revision()` + 置空 payload）；后台抽取线程只递增 `support.GRAPH_REVISION`，读请求比对两个 revision，落后才重建（数据量大时 `build_graph` 是全库 O(N) 遍历）；
- `invalidate_graph()`：本地写入与后台抽取共用同一个进程级计数 —— `/api/graph?since=` 只有一个真相来源，否则刚写完就会被 since 判成「无变化」。

### 各路由实现要点

**GET /api/graph**：缓存键 = revision + as-of 时间；`since >= 0` 且等于当前 revision 时返回 `{unchanged: true}`（前端轮询省流量）；`at` 参数选历史时刻。`TypeError/ValueError` → 422。

**POST /api/graph-rag**：`pipeline.graph_retrieve(...)` 后 `result.to_dict() | {"context": result.build_context()}`。

**POST /api/chat**：`chat_ready()` 不过则 503（带中文原因）；`online` 模式只在 AnySearch 已配置时真正开放 `web.search`（`chat_tool_names`），否则按 offline 处理；`ExecutionContext(confirmed_side_effects=chat_confirmed_side_effects(agent))` —— 只为 `memory.add` 预置写确认（用户这轮明确「记住」才能落库，删除/清空/入库仍需人工确认）；`async with app.state.chat_lock` 内 `await asyncio.to_thread(agent.run, ...)`（同步阻塞调用丢线程避免卡事件循环）；任何异常归一为 502。回答后：`record_qa` 写 episodic 留痕 → `schedule_qa_extraction` 后台把问答转成图补丁 → `invalidate_graph` → `graph_retrieve` 拿图证据（失败降级为空 `GraphRAGResult`，保持 200）→ `hybrid_retrieve` 做 U4 溯源（向量分/关键词分/RRF 分明细给前端「依据」面板；网关不可用时退化为纯关键词，让降级原因对用户可见）。返回 `{answer, mode, sources, paths, retrieval}`。

**POST /api/ingest**：`_save_upload` → `guard_embedding` → `pipeline.ingest_source(tmp_path, metadata={source, filename}, chunk_size=WEB_INGEST_CHUNK_SIZE, overlap=RAG_CHUNK_OVERLAP)` → 写 episodic「上传并导入了文档…」→ `invalidate_graph`。解析失败归一 422（附错误类型），`HTTPException` 原样上抛，`finally` 删临时文件。

**POST /api/facts**：`the_manager().semantic.add_fact(...)` → `invalidate_graph` → `{ok, item_id}`。

**POST /api/knowledge**：空文本 422；`guard_embedding`；有持久化队列时 `queue.submit(text, event_at)`——`wait=false` 立即返回 `{async: true, job_id, status}`；`wait=true` 阻塞 `queue.wait(job_id)` 后解析 `result` JSON（解析失败给可读 500，不把解析异常漏成内部错误），取 `report` 返回。回退分支（`:memory:` 测试库）保持旧同步行为。两种路径都写 episodic「添加了一条知识…」并 `invalidate_graph`。

**GET /api/knowledge/jobs**：状态白名单校验（422）；队列不可用返回 `{items:[], available:false}`；limit 在 API 层夹到 1–200（仓储层内部还有 200 硬上限）。

**POST /api/knowledge/jobs/{job_id}/retry**：先 `guard_embedding`；`LookupError` → 404，`ValueError` → 400，成功返回 `{ok, async, ...job_to_dict}`。

**POST /api/knowledge/image**：MIME 必须 `image/*`（否则 415）；`text` 超长 422；`_save_upload` 读 bytes；`pipeline.ingest_media(image, text, mime_type, metadata={source, filename, captured_at, reference_time, modality:"image"})`；`TypeError/ValueError` → 422；返回里带 `warning`（当前 embedding 不是 VL 模型时提示「图片已留存且已识图，但向量只使用文字说明」）。

**POST /api/seed**：`seed(the_manager())` + `invalidate_graph`。

**GET /api/export**：全部条目 `manager.list(include_expired=False)` → `{format: "knowledge-nebula-export/v1", exported_at, counts, items}`；响应头 `Content-Disposition` 用本地时间戳命名 `knowledge_export_<stamp>.json`。

**POST /api/import**：严格 JSON（`parse_constant=_reject_json_constant` 拒绝 NaN/Infinity，防止把非有限数值写进记忆库）；条目必须是列表；逐条：缺 `id/content` 跳过；`manager.get(item_id)` 已存在跳过；`memory_type` 必须合法；三元组事实按 `(subject, predicate, object)` 幂等去重（避免重复导入长重边）后 `add_fact`，其余 `manager.add`；单条失败只跳过该条并把原因记进 `errors`（上限 `WEB_IMPORT_ERRORS_MAX`，历史上静默吞异常只显示 skipped 计数）；导入成功才 `invalidate_graph`。返回 `{imported, skipped, errors}`。

**三库对账区**：

- `the_repository()`：复用 `app.state.pipeline.document_repo()`；`None`（内存模式）→ 400；
- `fact_items()`：全部三元组事实；
- `projected_vector_ids()`：`vector_store.list_ids()` 全集；读不到返回 None（不误报漂移）；
- `projected_edge_ids()`：`graph_store.relation_memory_ids()` 全集；
- `reconcile_report()`：三库计数 + 两类漂移——`missing_vector`（SQLite 标 indexed 但 Qdrant 没有）、`orphan_vector`（Qdrant 有但真值源和 memories 都没有）、`missing_edge`（事实没有对应图边）；
- `repair_drift(kinds)`：幂等自愈，只补缺失投影绝不删改真值源。`missing_vector`：重嵌入 chunk → `upsert_chunk` + `set_chunk_vector_status("indexed")`，文档状态 `parsed → vectorized`；`missing_edge`：对缺失的事实重放 `add_fact`（带原 `item_id`）。未知类型 422。

**GET /api/documents**：分页参数校验（`page_size` 1–`WEB_DOCUMENTS_PAGE_SIZE_MAX`，布尔值拒绝）；`repository.list_documents` + `chunk_counts` 组装。

**GET /api/documents/{document_id}**：404 或返回文档全字段 + 全部 chunks。

**POST /api/documents/{document_id}/revectorize**：`guard_embedding` → 取 chunk → `embed_batch` → 逐条 `upsert_chunk` + 状态置 indexed → 文档状态 `vectorized`（原是 extracted 则保持）；嵌入失败 `raise_embedding_http` 归一 409/502。

**GET /api/stats**：`repository.stats()` + `facts` 数 + `memories_total`。

**GET/POST /api/reconcile**：`reconcile_report()` / `repair_drift(body.repair)`。

**POST /api/embedding/rebuild**：`confirm_rebuild` 为真才真的重建（`guard_embedding(confirm_rebuild=True)`），否则只做门禁检查；返回 `{ok, rebuilt, embedding_lock, qdrant_dimension}`。

**GET /api/health**：报告**实际生效**的嵌入实现（看实例属性 `base_url`，不是猜配置——调用方可注入自定义 embedding）；`embedding_mode` = api/hash；`store_modes` 区分「本地真值 / 本地投影」（document: memory/sqlite、vector: 类名、graph: neo4j/inmemory）；`degraded` 字段给 UI 提示（云端嵌入未配置时检索只能走 FTS5，`keyword_fallback: true`）；`embedding_lock` 快照全量字段；`knowledge_extractor` 类名。

---

## 6.3 web/support.py —— 进程级单例与共享装配

模块 docstring 的设计要点：嵌入选型只认 `config/services.toml` 的 `[embedding]` 段（规则在 `memory.base.make_default_embedding`，Web 与 Agent 工具同一份）；`DB_PATH` 在导入时固定为 `default_sqlite_path()` 的返回值（`MEMORY_DB_PATH` 是唯一路径覆盖入口）——保证 Agent 工具（memory.query / memory.add / memory.rag）与 Web API **共享同一个记忆库**。

### 常量与路径

| 名称 | 含义 |
| --- | --- |
| `PROJECT_ROOT` / `WEB_DIR` / `STATIC_DIR` / `SEED_FILE` | 项目根、web 目录、静态目录、种子文件 `web/seed_data.json` |
| `DB_PATH` | `Path(default_sqlite_path())`，导入期固定 |
| `GRAPH_REVISION` + `_graph_revision_lock` | 全局图版本号（后台抽取与本地写入共用一个计数）；`bump_graph_revision()` / `graph_revision()` |
| `SYSTEM_PROMPT` | 知识管家行为约束：先 `memory.rag_search` 的 graph_retrieve/context 检索知识库，再用 `memory.query` 的 search 补记忆检索（**search 不指定 memory_type 会跨全部四层，提问历史与经历都在 episodic**）；中文简洁回答并注明来源；不编造；「记住」用 `memory.add` 写 episodic；关系更新只采信当前有效值（supersede 后的新值） |
| `CHAT_CONFIRMED_TOOLS = ("memory.add",)` | 聊天回合只为「记住这件事」这一个增量写入提供确认；delete/clear/ingest 仍需人工确认 |
| `SEARCH_TOOL_NAME = "web.search"` | 联网搜索注册名 |
| `CHAT_TOOL_ALLOWLIST` | `memory.query / memory.add / memory.rag_search / memory.rag / web.search` —— 前端知识管家只暴露这些；项目脚手架工具不进这一层 |

### 函数

| 函数 | 行为 |
| --- | --- |
| `build_embedding(config)` | **只转发** `make_default_embedding`，不重复选型逻辑 |
| `build_knowledge_extractor()` | 有真实 `provider.toml` 且 key 非占位符 → `LLMKnowledgeExtractor(client.complete, model, vision_model)`（视觉模型名来自 services.toml `[vision]` 段）；否则 `NullKnowledgeExtractor`（日志记原因） |
| `get_manager()` | 进程级单例 `MemoryManager`：`MemoryConfig.from_config()` + `sqlite_path=DB_PATH`，`MemoryManager(config, embedding=build_embedding(config))`；双检锁（`_manager_lock`） |
| `close_manager()` | 关单例并置 None |
| `get_pipeline()` | 进程级单例 `RAGPipeline(manager, extractor=build_knowledge_extractor())`（双检锁 `_manager_lock` 保护） |
| `get_agent()` | 懒加载 `ReActAgent("knowledge-butler", auto_discover_tools=False)` 单例：`set_system_prompt(SYSTEM_PROMPT)`；**不自动发现**（避免 fs/update_log/current_time 等脚手架进来），显式 `register_tool(..., replace=True)` 注册四个记忆工具（都注入 Web 单例后端，避免发现机制各自建的默认连接与 Web API 漂移）+ `SearchTool()` |
| `chat_confirmed_side_effects(agent)` | 返回本聊天回合允许的写确认键集合。Fails closed：未注册/未知工具不贡献任何键，Runtime 继续要求确认而不是静默放行 |
| `search_available()` | `[search]` 段 `base_url` 与 `api_key` 同时存在才视为可用 |
| `chat_tool_names(agent, online)` | 按模式返回可见工具名：永远只暴露记忆/图/向量检索工具；online 且 AnySearch 已配置才加 `web.search`。`None` 不再表示「发现到的全部工具」 |
| `record_qa(manager, question, answer, mode)` | 问答写入 episodic：正文「问：…\n答：…」，metadata 带 `kind=qa / type=qa / title / question / answer / mode / asked_at`，正文与字段都有字符上限（`WEB_QA_QUESTION_MAX_CHARS` 等） |
| `knowledge_extract_enabled()` | 问答是否触发图抽取（`constants.WEB_QA_EXTRACT`，测试 monkeypatch 该常量） |
| `extract_graph_patches(question, answer, manager)` | 把问答转成图补丁并落库：文本包装成「【用户陈述】+【助手回答（只抽取其中依据知识库给出的事实，推测性表述不要抽取）】」喂 `pipeline.ingest`；**刻意排除在聊天延迟之外**（响应返回后再调用）；抽取失败不能影响问答本身（记日志返回 None）；有实体/关系产出才 `bump_graph_revision`。**调用方必须传入自己的 manager**：后台线程绝不能再调 `get_manager()`（Web 关闭时全局单例会先被关闭，后台线程再抢同一把锁会永久挂住） |
| `extract_graph_patches_async(...)` | `await asyncio.to_thread(...)` 把阻塞的 LLM + SQLite 工作挪出事件循环 |
| `schedule_qa_extraction(...)` | fire-and-forget：没有真实聊天模型直接跳过；`WEB_QA_EXTRACT_SYNC` 为真则内联执行（测试/脚本拿确定顺序）；无事件循环也只内联执行（避免创建永远不跑的协程） |
| `chat_ready()` | 检测真实聊天模型：只有存在**真实的** `provider.toml` 且 key 非占位符才放行（ProviderRegistry 在文件缺失时会回退 provider.example.toml，必须显式区分）；返回 `(ready, reason)` |

---

## 6.4 web/graph_builder.py —— 星云图构建

模块 docstring 给出映射规则（与前端 `web/static/index.html` 布局约定一致）：

| kind | 天体 | 说明 |
| --- | --- | --- |
| `domain` | 恒星 | level 1，星系定位 |
| `entity` | 行星 | level 2，绕所属领域公转；**全局同名同一个** |
| `chunk` | 行星 | 原句：一个文档凝聚为一行星，含有向「提及」边连到全部相关实体 |
| `fact` | 卫星 | 多元观察保留，挂主语实体下 |
| `note` | 卫星 | 绕所属实体公转，不产生边 |
| `event` | 卫星 | 绕所属领域公转 |

**关系不再作为节点**：二元关系是一条有向边 `source→target`，`relation` 字段是谓词；时序观察把 `event_at/cardinality=temporal` 附在边上，同一实体的多条时序边并列保留。

分类优先级：`subject/predicate/object` 齐全 → fact；`kind=entity` → 显式实体；`kind=note` → 备注；`document_id + chunk_index` → RAG 知识块；其余 → 事件。

### 模块函数

| 名称 | 作用 |
| --- | --- |
| `domain_color(name)` | 领域 → 稳定颜色：`NEBULA_PALETTE[crc32(name) % len]`（crc32 跨进程稳定，Python 内建 hash 不稳定） |
| `_node(node_id, kind, title, ...)` | 造一个节点字典（`id/kind/title/content/domain/date/importance/color/parent/source/meta`） |
| `_entity_aliases(manager)` | 实体名 → 别名，一次取回；图库读不到退回空表，不能因此整图失败 |
| `_graph_snapshot(manager, at)` | 从图后端读拓扑（`graph_snapshot(at=at)`）；异常时返回 None → 走 SQLite 兼容回退 |
| `_date(item)` | 取 `created_at` 的日期字符串 |

### build_graph(manager, at)

多遍扫描内存条目 + 图快照，产出 `{graph_source, as_of, stats, nodes, edges}`：

1. 显式实体（`kind=entity`）先建节点（含别名 meta）；
2. RAG chunks：按 `document_id` 归组 `doc_meta`，逐文档算 `chunk_domain`（`classify_domain(content, title=filename)`）；
3. SQLite 三元组事实收集 + `attach_doc_entities`（事实把 subject/object 挂进所属文档的 entity_set）；
4. 实体 → 所属文档的「提及」关系准备（**即便实体还没有任何关系边也要连原句，避免实体在图上变孤儿**）；
5. `kind=note` → 挂到所属实体（`parent=entity_ids[...]`）的卫星；
6. 其余 → 事件卫星（过滤掉「一句话入库」/「添加了一条知识」这类系统噪音）；
7. **图快照优先**：实体节点补充 `aliases/entity_type`；observation 遍历——subject/object 两参与者的普通观察走 `link_triple` 直接成边；带额外角色的 n 元观察建成 `fact` 卫星节点（`edge:{observation_id}:subject` + 各参与角色边），`active=False` 的观察计入 `historical_facts` 且不渲染；随后遍历 `relations`（`active=False` 或已作为观察渲染过的跳过）；
8. SQLite 事实兜底（图快照缺失时）；非 active 的只计数不建边；
9. 文档去重（按 preview/filename 归并 `unique_docs`）→ 建 `chunk` 行星节点 + 「提及」边连全部相关实体；
10. **恒星 ↔ 行星引力桥**：每个 entity/chunk 行星都有一条指向所属领域恒星的 `属于` 有向边（`structural: true`，`confidence: 1.0`）；
11. 统计 `kinds` 计数 + `find_orphan_entities` 的孤儿数，返回 `graph_source`（`neo4j` / `inmemory` / `sqlite-fallback`）。

内部闭包：

- `domain_node(name)`：恒星节点去重建；
- `entity_node(name, domain, item, title)`：行星节点去重建（有 item 用 item.id 作 node_id，否则 `ent:<name>`）；别名来自图库；
- `link_triple(subject, predicate, obj, domain, extra)`：`(s,p,o)` 与 `(source_id, predicate, target_id)` 双重去重后追加边；
- `attach_doc_entities(doc_id, *names)`：给文档的 entity_set 追加实体名。

---

## 6.5 web/ingest_queue.py —— 一句话后台入库队列

模块 docstring：提交立即返回（前端即刻清空输入框），后台线程按提交顺序完成向量化与 LLM 抽取。**后一次入库可能要检索/用上上一次的图与向量，所以不能并行**（固定单 worker）。每条任务 `pending → running → done/failed` 状态写在 memories 同一个 SQLite 文件（`ingest_jobs` 表）；进程异常退出后 `start()` 把遗留 `running` 复位为 `pending` 并重新入队（`attempts` 封顶防死循环）。

| 名称 | 含义 |
| --- | --- |
| `MAX_ATTEMPTS = 3` | 单条任务跨重启最大尝试次数（含首次） |
| `JOB_LABELS` | 状态 → 中文展示文案（API/前端共用一份口径） |
| `job_to_dict(job, text_preview)` | 记录 → API 字典（`result` 列是 JSON 字符串，先解析成 dict），附 `label` 与 `retryable`(仅 failed) |

### IngestJobQueue

构造：`manager/extractor/on_progress`；`path = manager.document_store.path`，`:memory:` 或空则 `_repo=None` → `available=False`；文件库建 `DocumentRepository(path)`（每次调用独立连接，线程安全）。

| 方法 | 行为 |
| --- | --- |
| `start()` | 单 worker `ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest-job")`；`restart_stale_ingest_jobs()` 恢复上次退出时未完成的任务；对每个 `pending` 任务 `_enqueue` |
| `submit(text, event_at)` | 落库 `pending` 任务并立即入队，返回记录供 API 回显 |
| `job(job_id)` / `list(status, limit)` | 查询 |
| `retry(job_id)` | 只有 failed 可重试；`reset_failed_ingest_job` 清零 attempts（用户重试不卡在封顶）；重排队 |
| `wait(job_id, timeout=180)` | 50ms 轮询到终态或超时（同步调用方用） |
| `shutdown(wait=True)` | 停止 worker；`wait=True` 等在跑任务收尾（**v1.65 之前异步关闭会残留仍在访问已关闭 manager 的后台线程，进程退出竞争**）；`wait=False` 留给下次启动复位续跑 |
| `_enqueue(job_id)` | `executor.submit(_run_job, job_id)` |
| `_run_job(job_id)` | 终态/done 直接返回；`attempts >= MAX_ATTEMPTS` → failed「重试次数超限」；置 running → **每任务独立 `RAGPipeline`**（串行执行时 `last_ingest_report` 不串味）→ `pipeline.ingest(Document(text, metadata={source:"一句话入库", filename: 首行前 40 字, note, event_at, reference_time, ingest_job_id}))` → `report.errors` 非空则 failed，否则 episodic 记录「添加了一条知识…」并 done；任何异常 failed 并记 `类型: 消息` |
| `_notify(job_id)` | 调 `on_progress`（失败只警告不中断） |

---

## 6.6 web/domain_classifier.py —— 本地主题领域分类器

设计目标：零依赖纯规则、不联网不耗 LLM key（未配置 key 也能用）；确定性（同样文本永远同样领域，跨进程稳定可测试）；轻量（一次关键词表遍历 O(len(keywords))）。

| 名称 | 含义 |
| --- | --- |
| `DOMAIN_KEYWORDS` | 领域 → 关键词表（8 类：编程开发 / 数学 / 物理 / 化学 / 生物医学 / 历史人文 / 经济管理 / 文学艺术），中英混合；太通用的词（如「数据」「系统」）容易串类，故不收录 |
| `KNOWN_DOMAINS` | `tuple(DOMAIN_KEYWORDS.keys())`，供 UI/测试/文档用 |
| `classify_domain(text, title, default)` | 打分：统计关键词出现次数，`title` 命中额外加权（`DOMAIN_TITLE_WEIGHT`，文件名常含主题词如「c语言笔记.txt」）；最高分胜出；全 0 返回 `DEFAULT`（"未分类"） |
| `majority_domain(domains, default)` | 取众数领域（文档实体按多数知识块的领域挂恒星系）；空输入回退 default |

---

## 6.7 web/seed.py —— Aetheria 种子数据导入

幂等标记 `SEED_MARK = "aetheria-seed-v1"`。三种运行方式：`python -m web.seed` 命令行手动播种；应用启动自动播种（`constants.WEB_AUTOSEED`，默认开）；`POST /api/seed` 强制检查。

`seed(manager, path=None)`：

1. 种子文件不存在 → `{seeded: false, reason}`；
2. 已存在带 `seed == SEED_MARK` 的语义条目 → 幂等跳过；
3. `entities`：`manager.add(name, semantic, metadata={kind:"entity", title, domain, seed}, importance)`；
4. `relations`：`manager.semantic.add_fact(subject, predicate, object, metadata={domain, note, date, seed}, confidence)`；
5. `notes`：`manager.add(content, semantic, metadata={kind:"note", entity, domain, title, date, seed}, importance=0.4)`；
6. 返回 `{seeded, source, entities, relations, notes}`。

---

## 6.8 web/cleanup.py —— 孤立实体清理

「完全孤立」判定（四个条件**全部满足**才算）：

1. 没有任何活跃关系边（既不是任何事实的 subject，也不是 object）；
2. 没有被任何原句（chunk）通过「提及」边引用（`source_ids` 里没有 chunk id）；
3. 没有备注（`kind=note`）挂靠在它名下；
4. 不是 seed 播种的实体（避免把种子星图当噪音清掉）。

| 函数 | 行为 |
| --- | --- |
| `find_orphan_entities(manager, items)` | 复用调用方已取回的 semantic 列表（避免图构建时二次全表扫描）；返回孤儿条目 |
| `propose_orphan_cleanup(manager, requested_by, reason)` | 为所有孤儿创建一条待确认删除提案（只写提案不删数据；调用方拿到 `proposal_id + confirm_token` 后由显式确认流程调 `execute_deletion` 才真删）；没有孤儿返回说明 |

真正删除始终走 `memory.storage.document_repo.execute_deletion` 的确认闸门，绝不绕过。

---

## 6.9 一次「问答」的完整链路（web 视角）

1. `POST /api/chat` → `chat_ready()`（无真实模型 503）→ `get_agent()`（单例，五个工具）→ `ExecutionContext`（只给 `memory.add` 写确认）→ `chat_lock` 串行化 → `agent.run(message, tool_names, context)`；
2. Agent 内部先 `memory.rag_search(graph_retrieve/context)` 检索知识库、`memory.query(search)` 搜记忆、需要时 `memory.add` 记住、`web.search` 联网；
3. 回答返回后 `record_qa` 写 episodic → `schedule_qa_extraction` 后台抽取图补丁（独立 manager 注入，绝不碰全局单例）→ `invalidate_graph`；
4. 响应附带 `graph_retrieve` 证据 + `hybrid_retrieve` 溯源明细（向量/RRF/关键词分），前端渲染「依据」面板；
5. 前端 `/api/graph?since=<revision>` 轮询，后台抽取写图后 revision 递增，星云图自动刷新。
