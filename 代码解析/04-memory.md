# 04 · memory/ —— 四层记忆系统与 RAG

`memory/` 是整个项目的数据核心：所有「记住的东西」都住在这里，Web 前端与 Agent 工具共用同一份记忆库（`MEMORY_DB_PATH` 是唯一路径覆盖入口）。

包结构（见 `memory/__init__.py`）：

| 子模块 | 职责 |
| --- | --- |
| `base` | 数据结构（`MemoryItem` / `MemoryConfig`）、`BaseMemory` 通用读写、嵌入服务选型 |
| `embedding` | 云端 OpenAI 兼容嵌入客户端 + 离线确定性 `HashEmbedding` |
| `embedding_lock` | 向量空间锁：换嵌入模型时必须显式确认重建投影 |
| `ids` | 实体/关系/观察的稳定 ID 派生（SQLite 与 Neo4j 共用） |
| `types` | 四层记忆实现：working / episodic / semantic / perceptual |
| `storage` | SQLite 文档库、`documents`/`chunks` 真值源、内存/Qdrant 向量库、Neo4j 图库 |
| `rag` | 文档解析与切块、LLM 知识抽取落地、图检索、混合检索管线 |
| `manager` | `MemoryManager`：四层 + 三个存储的协调入口 |

---

## 4.1 memory/base.py —— 数据结构与通用读写

### MemoryType / 路径与时间工具

| 名称 | 含义 |
| --- | --- |
| `MemoryType.WORKING` | 工作记忆：会话级、带 TTL |
| `MemoryType.EPISODIC` | 情景记忆：带时间戳的事件/问答 |
| `MemoryType.SEMANTIC` | 语义记忆：实体、三元组事实、RAG 知识块 |
| `MemoryType.PERCEPTUAL` | 感知记忆：多模态负载（图片） |
| `default_sqlite_path()` | `MEMORY_DB_PATH` 环境变量，否则项目根目录下 `memory.sqlite3` |
| `utc_now()` | 当前 UTC 时间（带时区） |
| `ensure_datetime(v)` | `datetime`/ISO 字符串/None → 统一为 UTC aware datetime；无时区按 UTC 补齐；其他类型抛 `TypeError` |

### make_default_embedding(config) —— 嵌入服务三级选型

1. `embedding_provider == "hash"` → 强制离线 `HashEmbedding`；
2. 云端配置齐全（`[embedding].base_url` + `api_key`）→ `APIEmbedding`；
3. 其他情况 → 退回 `HashEmbedding` 而不是抛错，保证离线可用。

关键约束：**不同模型的向量空间互不兼容**，换模型必须重建投影（见 4.3）。

### _merge_services_into(values, services)

把 `config/services.toml` 的值补进 `values`，只在字段缺席时补 —— 显式构造参数永远优先。共补 14 个字段：`embedding_provider/base_url/model/api_key/dimension/batch_size/timeout`、`qdrant_url/collection/api_key`、`neo4j_uri/username/password`、`proxy_url`。

### MemoryItem（dataclass）

| 字段 | 含义 |
| --- | --- |
| `content` | 记忆正文 |
| `memory_type` | 所属层，构造后强制转 `MemoryType` |
| `id` | 记忆 ID，默认 `uuid4`；非空字符串，否则 `ValueError` |
| `metadata` | 结构化元数据（`subject/predicate/object`、`kind`、`domain`、`aliases`…） |
| `importance` | 重要度，0–1，非有限数/越界报错 |
| `created_at` / `updated_at` | UTC 时间；构造时统一走 `ensure_datetime` |
| `expires_at` | 过期时间；配合 `is_expired` 实现 TTL |
| `timestamp` | 业务发生时间（区别于写入时间） |
| `embedding` | 向量（只允许有限数值序列，空列表/NaN 都拒绝） |
| `payload` | 任意负载（图片 bytes 等） |
| `modality` | 模态标记，如 `"image"` |
| `relations` | 关系列表（Mapping 序列） |

- `to_dict()`：导出为可 JSON 序列化的字典（`_json_safe` 把 bytes 转 base64 包装、不可序列化对象转字符串）；
- `_json_restore(value)`：`__bytes__` 包装还原为 bytes，读路径用。

### MemorySearchResult

`frozen dataclass`，字段 `item` + `score`；`to_dict()` = `item.to_dict()` 再加上 `score`。

### MemoryConfig（dataclass）

运行时配置，字段全部带默认值；`sqlite_path` 支持 `":memory:"`（测试用）。关键字段：

`sqlite_path` / `default_ttl_seconds` / `working_memory_capacity` / `search_limit` / `similarity_threshold` / `embedding_dimension`（None=首次响应自动识别）/ `embedding_provider`（`openai_compatible`/`hash`）/ `embedding_model` / `embedding_base_url` / `embedding_api_key` / `embedding_timeout` / `embedding_batch_size` / `qdrant_url` / `qdrant_collection` / `qdrant_api_key` / `neo4j_uri` / `neo4j_username` / `neo4j_password` / `proxy_url` / `extra`。

`__post_init__` 逐项做类型与范围校验（0–1 的小数、正整数、非空字符串等）。`from_config()` 的优先级只有两层：**显式构造参数 > services.toml**（历史环境变量层已删除）。

### BaseMemory —— 一层记忆的通用读写

构造函数注入 `document_store` / `vector_store` / `embedding` / `config` / `memory_type`，缺省时各自懒加载默认实现（在函数内 import，避免与 storage 包循环依赖）。

| 方法 | 行为 |
| --- | --- |
| `_embed_item(content, payload, modality)` | 委托 `embedding.embed_item`，让多模态后端把 `payload` 折进向量 |
| `_validate_embedding_dimension(v)` | 与配置中的期望维度比对，不一致即抛错 |
| `add(...)` | **先写真值再写向量**：`document_store.upsert` 成功后才 `vector_store.upsert`；向量失败则回滚（新条目删除、旧条目还原旧向量），保证不会出现「有索引无记录」的孤立向量。`ttl_seconds` 与 `expires_at` 互斥；WORKING 层未显式给 TTL 时套用 `config.default_ttl_seconds` |
| `get(item_id)` | 读取并顺带清理过期项；跨层不返回 |
| `delete(item_id)` | 先删向量索引，再删文档库记录 |
| `search(query, limit, threshold, metadata)` | 嵌入 query → 向量召回 `limit*4` 个候选（多取因为过滤会淘汰）→ 阈值/过期/metadata 过滤 → 按分数排序截断；`score<=0` 视为无意义匹配直接丢弃 |
| `list(include_expired)` | 只读文档库，不碰向量索引（重建索引是 manager 启动时的职责） |
| `clear()` | 遍历删除该层所有条目，返回删除条数 |

---

## 4.2 memory/embedding.py —— 嵌入服务

模块常量：`VL_EMBEDDING_MODEL_MARKER = "vl"`（模型名含 `vl` 视为视觉语言模型）、`EMBEDDING_RESPONSE_MAX_BYTES`（响应体上限）。

### BaseEmbedding（ABC）

`embed(text)` / `embed_batch(texts)` / `embed_item(text, payload, modality)` 三个抽象接口。

### HashEmbedding —— 离线确定性兜底

- `tokenize(text)`：`\w+` 小写切词；
- `_index(token)`：`blake2b(token, digest_size=8)` 取模映射到维度桶；
- `embed(text)`：词袋计数（`1 + log(count)` 加权）叠加后 L2 归一化；
- 与云端向量空间**不兼容**，仅用于离线可用与测试替身。

### APIEmbedding —— OpenAI 兼容客户端

构造参数 `api_key / model / base_url / dimension / timeout / batch_size / client`。`base_url` 是去掉 `/embeddings` 的根地址；`multimodal` 由模型名是否含 `vl` 推导，可显式改写。

| 方法 | 说明 |
| --- | --- |
| `embed` / `embed_batch` | 按 `batch_size` 分批调用，逐批走 `_embed_batch_once` |
| `text_input` / `image_input` | 构造 VL 内容对象；bytes 自动转 `data:<mime>;base64,...` |
| `inputs_for(text, image, mime_type)` | 组装 VL `input` 列表（文本对象 + 图片对象） |
| `embed_inputs(items)` | 一个融合列表只应得到 1 个向量，数量不符即报错 |
| `embed_image` / `embed_multimodal` | 纯图 / 图文融合 |
| `embed_item(text, payload, modality)` | 路由入口：`payload` 为空或非多模态模型 → 纯文本；否则图文融合 |
| `to_dict()` | 供 `/api/health` 展示 model/dimension/multimodal |

模块级辅助：

- `_embed_batch_once`：一次请求，校验「返回向量数 == 输入数」；
- `_learn_dimension`：校验维度一致、数值有限，并把学到的维度写回 `embedding.dimension`（与配置不符即报错）；
- `_request`：发 HTTP（`Authorization: Bearer`），也支持注入可调用 `client` 或带 `embed`/`embeddings.create` 的对象（测试替身）；HTTPError/URLError/JSON 错误/超大响应统一转成带原因的 `RuntimeError`；
- `_extract_embedding_vectors`：同时兼容 OpenAI `data[].embedding` 与 DashScope `output.embeddings`；按 `index`/`text_index` 排序，并校验索引连续；
- `_IMAGE_MAGIC` + `_image_mime_type(data, fallback)`：按魔数识别 PNG/JPEG/GIF，猜不出用 fallback（声明错 MIME 会被服务端拒收）。

---

## 4.3 memory/embedding_lock.py —— 向量空间锁

**背景**：每家的嵌入模型产出互不兼容的向量空间。若静默混用，检索会「查得到但全不相关」。因此把「当前库属于哪套向量空间」作为单行记录锁在 SQLite 里，与线上集合真实维度交叉验证。

| 名称 | 作用 |
| --- | --- |
| `EmbeddingIdentity` | frozen dataclass：`model` + `dimension` |
| `EmbeddingLockMismatch(ValueError)` | 不一致异常；`message` 给出中文修复指引；`to_detail()` 产出 HTTP 409 的 JSON 体（`code=embedding_lock_mismatch` + locked/current） |
| `embedding_model_name(e)` | 取 `model` 属性，没有则回落类名 |
| `resolve_embedding_identity(e)` | 拿实时 (model, dimension)；维度未知时用探针文本 `embedder-lock-probe` 试一次并回写 |
| `live_vector_dimension(manager)` | 读 Qdrant 集合真实尺寸（后端没实现 `collection_dimension` 时返回 None） |
| `inspect_embedding_lock(manager, repo)` | 只读快照：`current/locked/projection/qdrant_dimension/mismatch`。**线上集合维度优先于 SQLite 锁**：空锁不能掩盖一个已存在的 1024 维集合 |
| `apply_embedding_lock(manager, repo, confirm_rebuild)` | 写向量前的闸门：`None` 仓库（测试）直接放行；一致则补写空锁；不一致且未确认 → 抛 `EmbeddingLockMismatch`；确认后调 `rebuild_vector_projection` |
| `mismatch_from_exception(exc, ...)` | 把 Qdrant 的维度报错归一成同一个 409 载荷 |
| `rebuild_vector_projection(manager, repo, identity)` | 删集合并按新维度重建 → 重灌全部 chunk 向量（`upsert_chunk` + `vector_status=indexed`）→ 重灌全部记忆条目 → 写新锁。返回 `{model, dimension, chunks, memories}` |

---

## 4.4 memory/ids.py —— 稳定 ID 派生

所有写入方共用同一套 ID，保证「同一个实体/同一条事实」在 SQLite 与 Neo4j 里是同一个东西。

| 函数 | 输出 |
| --- | --- |
| `_clean_text(v, max_length)` | 压缩空白 + 截断 + 去首尾空白 |
| `normalize_entity_name(v)` | 转小写并剔除所有非「单词/汉字」字符 → 比较键（保留可读性由调用方负责） |
| `entity_id_for(name)` | `entity:<sha256(key)[:20]>` |
| `relation_id_for(s, p, o)` | `relation:<sha256(s\|p\|o)[:24]>` |
| `observation_id_for(...)` | `observation:<sha256(payload)[:24]>`；payload 含主语/谓语/宾语/全部 roles/`event_at`/`valid_from`/`valid_to`/`source_id`，因此**同一三元组在不同时间点的观察可以共存**，同一 chunk 重试则幂等 |
| `predicate_key_for(s, p)` | (主语, 谓语) 槽位键，不含宾语 —— 单值槽位（如「余额」）据此让新值淘汰旧值 |
| `legacy_fact_id_for(s, p, o)` | 旧版可读形式 `fact:s|p|o`，`add_fact` 回退它以便原地更新老数据 |

---

## 4.5 memory/types/ —— 四层记忆实现

| 文件 | 类 | 说明 |
| --- | --- | --- |
| `working.py` | `WorkingMemory` | 容量超限时驱逐：按 `(importance, updated_at)` 升序排序，先踢最不重要/最旧的 |
| `episodic.py` | `EpisodicMemory` | 额外提供 `record(content, metadata, timestamp, ...)`：面向事件记录的便捷入口 |
| `semantic.py` | `SemanticMemory` | 唯一持有 `graph_store` 的一层，负责把事实落成图 |
| `perceptual.py` | `PerceptualMemory` | 空壳，直接继承 `BaseMemory`（多模态负载随 `payload` 走通用路径） |

### SemanticMemory.add_fact(subject, predicate, object, metadata, confidence, item_id)

1. 三项都必须是非空字符串，`confidence ∈ [0,1]`；
2. ID 走统一方案 `relation_id_for(...)`；查不到且调用方未指定 `item_id` 时，回退 `legacy_fact_id_for` 命中老行（**老数据原地更新而不是重复插入**）；
3. 已存在：合并 metadata（`active`/`superseded_by`/`superseded_at` 只在没有显式指定时重置为「复活」状态，保证 `retract` 幂等），importance 取新旧最大值，然后 `add` + `_write_edge` 刷边（否则图里还留着旧 active 标记，与 SQLite 真值不一致）；
4. 新事实：写入 `subject/predicate/object/confidence` 元数据后同样 `_write_edge`。

### _write_edge / _graph_properties / _endpoint_attributes

- `_graph_properties(item, metadata)`：边属性集合（`memory_id`、`confidence`，以及 `evidence/source/source_document/chunk_id/predicate_key/action/cardinality/active/superseded_*/supersedes/valid_from/valid_to/status/event_at/captured_at/modality/observation_id` 中存在才写入）；
- `_endpoint_attributes(name)`：查该名字对应的实体条目，取出 `domain/aliases/importance/entity_type`（抽取管道先写实体、后写关系，所以通常命中；未命中就用默认值）；
- `_write_edge(...)`：先 `add_relation` 写二元兼容边；再把 subject/object 以及每个额外 role 组成 `participants` 调 `add_observation` 落 **n 元观察**；同时为每个额外角色补一条兼容 `RELATED` 边（`GraphRAG` 仍遍历 `RELATED`，权威形态是 `MemoryObservation + HAS_PARTICIPANT`）。

其他方法：`add_relation`（`add_fact` 的别名）、`delete`（先删向量再删图边）、`related(entity, relation, at)`、`facts(entity)`（无参返回全部事实，按 subject/object 过滤）。

---

## 4.6 memory/manager.py —— MemoryManager

构造函数按优先级拼装四个后端：

- `embedding` / `embedding_service`（二者只能给一个，否则 `ValueError`）；
- `document_store`：默认 `SQLiteDocumentStore(config.sqlite_path)`；
- `vector_store`：注入优先 → 配置了 `qdrant_url` 则 `QdrantVectorStore`（回环地址不走代理，云端地址默认走 `cloud_proxy_url` 计算出的本机 7890 或显式 `[proxy]`）→ 否则 `InMemoryVectorStore`；
- `graph_store`：默认 `Neo4jGraphStore(...)`（同样带上 `proxy_url`）；
- 四个记忆层共用同一组后端；`memories` 字典按 `MemoryType` 索引；
- **仅当向量库是内存实现时**，启动时把文档库全部条目重新 upsert 进索引（远端/Qdrant 已持久化，重灌纯属启动开销）。

| 方法 | 行为 |
| --- | --- |
| `for_type(t)` | 取某一层；未知类型抛 `ValueError` |
| `add(content, memory_type=working, **kw)` | 转发到对应层 |
| `get(item_id, memory_type=None)` | 指定层走层内 get；未指定走文档库，过期项顺带删除 |
| `delete(item_id, memory_type=None)` | 未指定类型时先查出条目再按其真实类型删 |
| `search(query, memory_type=None, ...)` | 指定层 → 单层检索；未指定 → **四层各查一次再按分数合并**（`(-score, created_at)` 排序后截断） |
| `list(memory_type, include_expired)` | 单层或全库 |
| `clear(memory_type)` | 单层或全部 |
| `close()` / `__enter__` / `__exit__` | 关文档库、向量库、图库 |

模块函数 `_is_loopback_endpoint(url)` / `cloud_proxy_url(config, endpoint)`：判断端点是否回环，决定是否挂代理。

---

## 4.7 memory/rag/document.py —— 文档解析与切块

### Document / ChunkSpan

- `Document`（frozen dataclass）：`content`（非空字符串）/ `id`（默认 uuid4）/ `metadata`；
- `ChunkSpan`：`chunk` + `char_start` + `char_end`。**偏移量索引的是 `normalized_text` 的结果**，也就是 `documents.raw_text` 存的那份文本 —— 两者同源，偏移才能用于重切与前端高亮（方案 2.3）。

### DocumentProcessor

| 方法 | 行为 |
| --- | --- |
| `parse(source, metadata)` | `Path`/`os.PathLike` → 读字节按扩展名解析；`bytes` → UTF-8 宽松解码；`TextIOBase` → 读流；`str` → **永远当字面文本**（绝不探测文件系统，防止模型给的路径越出沙箱） |
| `normalized_text(document)` | `\s+` → 单空格并 strip；全文只归一化这一次 |
| `chunks_with_spans(document, chunk_size, overlap)` | 定长滑窗切块（步长 = `chunk_size - overlap`），逐块记 `(char_start, char_end)`，metadata 带 `document_id` + `chunk_index` |
| `sentences_with_spans(document)` | F4 逐句切块：以 `。！？!?.;\n` 为边界，标记 `granularity="sentences"`，结尾残句也不丢 |
| `_parse_bytes(raw, extension)` | `.jsonl` 逐行 JSON；`.json` 美化输出；`.csv` 用 ` | ` 连接；`.html/.htm` 正则去标签；`.pdf` 用 pypdf 逐页抽文本；其余按 UTF-8 文本 |

`resolve_within(base_dir, path)`：解析并强制约束在 `base_dir` 内（越界抛错）。

---

## 4.8 memory/rag/knowledge.py —— 知识抽取与落地

### 抽取结果模型（Pydantic）

| 模型 | 关键字段 |
| --- | --- |
| `EntityCandidate` | `name`、`entity_type`、`description`、`confidence`、`aliases` |
| `RelationRole` | n 元观察的额外参与者：`role`、`value`、`entity_type` |
| `RelationCandidate` | `subject`/`predicate`/`object` + `action(assert/supersede/retract)` + `cardinality(single/multi/temporal)` + `roles` + `valid_from`/`valid_to` + `status(fact/plan/expired/uncertain)` + `event_at` + `confidence` + `evidence` |
| `ExtractionResult` | `domain`、`topics`、`entities`(≤50)、`relations`(≤80)、`keywords` |

均配 `extra="ignore", strict=True` 与字段清洗器（`_clean_text` 截断），非法 JSON 值会在模型层就被挡下。

### 抽取器

- `KnowledgeExtractor`（Protocol）：`extract(text, metadata, graph_context, image, mime_type) -> ExtractionResult`；
- `NullKnowledgeExtractor`：未配置聊天模型时的安全兜底，永远返回空结果（只做向量检索，不产图）；
- `LLMKnowledgeExtractor`：
  - `SYSTEM_PROMPT` 是一份很长的中文抽取协议，要求模型只输出一个 JSON 对象，并把任务定义为「对现有知识图打补丁」；
  - `extract(...)`：把 source/text/reference_time 以及 `captured_at/event_at/modality` 打包成 JSON prompt；有图片时改发 `[{type:text},{type:image_url}]` 并切到 `vision_model`；`temperature=0.0`；
  - `_image_data_url`：bytes → data URI，字符串原样透传；
  - `_content(response)`：从 dict/对象两种形态取 `choices[0].message.content`；
  - `_parse_json(raw)`：剥 Markdown 代码围栏 → `json.loads` → 失败时截取首尾花括号再解析；非对象报错。
- `QueryDecomposer` / `NullQueryDecomposer` / `LLMQueryDecomposer`（F3）：把一句问句拆成最多 6 条子查询（第一条必须是原句），失败时永远回退原句，「永不因分解失败而答不出来」。

### 实体消解 EntityResolver

构造时 `_load()` 全量载入 `kind == "entity"` 的语义条目，按 canonical name 与别名建索引。

| 方法 | 作用 |
| --- | --- |
| `_remember(item)` | 把条目按 canonical + 每个别名登记进索引 |
| `_exact(key)` | canonical 名或别名精确命中 |
| `_alias(key)` | 仅别名命中（用于回写别名） |
| `_prefix_candidate(key)` | 最长完整前缀命中（共享字符最长者胜），只读不改 |
| `_store_alias(item, name)` | 把新写法存为别名并**重定点所有指向旧快照的索引键**（否则下一次 resolve 读到过期快照会丢别名） |
| `match(name)` | 精确 → 别名 → 前缀 |
| `resolve(name, domain, ...)` | 完整解析：命中已有实体则合并元数据（`kind/title/canonical_name/entity_type/description/domain/aliases/source_ids`），importance 取新旧最大；未命中按模糊相似度（`_entity_similarity`：序列相似度与词集合 Jaccard 取大者，阈值 `ENTITY_SIMILARITY_THRESHOLD`）尝试归并；否则新建实体。返回 **canonical name** |

### materialize_extraction —— 补丁落地

一次 ingest 调用的核心写入口。抽取器只「提议」，这里负责确定性地执行：

1. 先 `resolver.resolve` 落全部实体（`canonical_by_key` 缓存本 chunk 的归并结果）；
2. 建 `(subject, predicate)` 槽位索引 `known_items`（`index_all_facts` 惰性全量扫描，`remember` 在每次写入后刷新缓存副本）；
3. 逐条 relation：
   - `confidence < relation_threshold`(0.6) 直接跳过并计数；
   - `retract`：把目标事实标 `active=False` + `superseded_at`，不删历史；
   - `retire`（`action == "supersede"` 或 `cardinality == "single"` 且非时序）：把同槽位其它 active 事实全部标记淘汰（不是只挑缓存里第一条）；
   - `assert`：写入/更新事实；
   - ID 选择：时序（有 `event_at`/`valid_from`/`valid_to`/`cardinality=temporal`/有额外 roles）→ `observation_id_for(...)`；否则 `relation_id_for(...)`；
   - `fact_metadata` 里带上 `predicate_key`、`observation_id`、`source_ids`、`evidence_items`（保留最近 20 条证据）、`roles`、时间与状态字段；
4. 返回报告 `{domain, topics, entities, relations, superseded, retracted, skipped_relations, relation_items}`。

### build_graph_context / _known_domains / _fit_lines

- `build_graph_context(manager, text, resolver, max_relations, max_chars)`：把**图上与本文相关的切片**渲染成文本喂给抽取器（canonical 名或别名出现在文本中，或共享显著前缀即算相关，命中实体再外扩一跳）。resolver 可注入，避免每个 chunk 重建索引；`max_chars` 用 `_fit_lines` 按行截断；
- `_known_domains()`：从 `web.domain_classifier.KNOWN_DOMAINS` 取已知领域清单，提示模型复用既有领域名。

---

## 4.9 memory/rag/graph_rag.py —— 图检索

### GraphPath / GraphRAGResult

- `GraphPath`：`source/target/relations/entities/confidence/evidence/effective/steps`；`effective = confidence × weight`（F1 边权重参与排序）；
- `GraphRAGResult`：`query/evidence/paths/entities`；`build_context(max_chars)` 用 `_fit_lines` 渲染成一段可注入 prompt 的上下文。

### GraphRAGPipeline

| 方法 | 行为 |
| --- | --- |
| `retrieve(query, limit, hops, threshold, path_limit, at)` | 先 `manager.search(semantic)` 取证据 → `_find_seed_entities` → `_expand`。**任何图后端异常都被捕获**，退化为「只有向量证据」，Aura 抖动不会拖垮整轮问答 |
| `build_context(...)` | `retrieve().build_context()` 的便捷包装 |
| `retrieve_multi(queries, ...)` | F3：每条子查询各自 seed+expand，证据按 id 去重、路径按 `(entities, relations)` 去重取 effective 最大者 |
| `_find_seed_entities(query, evidence)` | 证据里 `kind == "entity"` 的条目、以及 `subject`/`object` 出现在问句中的实体名，都作为种子 |
| `_expand(seeds, hops, path_limit, at)` | BFS：`visited` 按 `(邻居, 关系序列)` 去重、邻居不可重复入路径（简单路径）；每条边都要过 `_edge_is_active`（回到 SQLite 核对 memory_id 的 active，因为图里的边副本可能还是淘汰前的状态）、`_edge_in_window`、`status != "expired"`；`uncertain` 置信度减半；`effective` 沿路径取最小值；最后按 `(-effective, 关系数, target)` 排序截断，并对真正返回的路径调 `_reinforce` |
| `_weighted_edges(entity, at)` | 取邻居边并按 `weight` 降序（强边优先遍历，limit 截断时留下更强的路径） |
| `_reinforce(paths)` | 「回忆即强化」：对返回路径的每条边各 +1 次回忆（每条边每轮最多一次），调 `add_relation(..., bump=True)` |

---

## 4.10 memory/rag/pipeline.py —— RAG 主管线

`RetrievedChunk`（frozen dataclass）：`content/score/memory_id/metadata/detail`；`detail` 存混合检索分数明细（`rrf_score`/`vector_score`/`keyword_score`），供前端「依据」面板溯源。

辅助函数：`_accepts_parameter` / `_accepts_graph_context`（用 `inspect.signature` 判断抽取器是否支持新参数，兼容旧替身）、`_document_tags` / `_document_permission`、`_hybrid_enabled()`（读 `constants.MEMORY_HYBRID`）、`_rrf_fuse(rank_lists, k=60)`（RRF 只按名次计分，避免余弦与 BM25 两套量纲混算）、`_chunk_metadata(chunk)`（真值源元数据投影）。

### RAGPipeline

构造：`manager`（默认新建）/ `processor` / `extractor`（默认 `NullKnowledgeExtractor`）/ `auto_extract`；内部持 `GraphRAGPipeline`、`last_ingest_report`、`last_retrieval_note`、缓存 `_repository`。

| 方法 | 行为 |
| --- | --- |
| `document_repo()` | 返回 `documents`/`chunks` 真值源仓储；非 SQLite 后端或 `:memory:` 返回 `None`（第二个内存库与它要镜像的库不共享数据） |
| `ingest(documents, chunk_size, overlap, granularity)` | ① 写 `documents` 真值行（`status="parsed"`）；② 逐 chunk **先写真值再写向量**（`vector_status="indexed"`）；③ `build_graph_context` → `extractor.extract` → `materialize_extraction`；④ 任何 chunk 失败只记 `report["errors"]`，文档状态置 `failed`；**嵌入/真值写失败则 `delete_document` 回滚后原样抛出**；⑤ 汇总 report（`extractor` 类型、`extraction_skipped`） |
| `ingest_media(image, text, mime_type, metadata)` | 图片走 `PERCEPTUAL` 层，`payload=image`；图边只来自视觉抽取器的结构化输出，绝不来自向量相似度；report 带 `multimodal_embedding` 标记当前嵌入是否 VL 模型 |
| `ingest_source(source, base_dir, **kw)` | 按文件路径入库；给 `base_dir` 时强制路径包含约束 |
| `retrieve(query, ...)` | 纯向量检索 |
| `_vector_hits(...)` | 向量路；`ConnectionError/OSError/RuntimeError` → 记 `last_retrieval_note` 并返回空表（降级为纯关键词） |
| `hybrid_retrieve(...)` | 向量路 × FTS5 关键词路，各取 `limit*2`，RRF 融合后只返回**真值源里存在**的 chunk（孤立向量不外泄），并带分数明细 |
| `hybrid_retrieve_multi(queries, ...)` | F3：N 条子查询的两路 rank 列表一起丢给 RRF |
| `build_context` / `graph_retrieve` / `graph_retrieve_multi` / `graph_context` | 上下文拼装与图检索入口 |
| `answer(query, generator, limit)` | 检索 → 组上下文 → 交给外部生成器 |
| `delete_document(document_id)` | 删真值源文档、chunk、相关记忆与向量 |

---

## 4.11 memory/storage/document.py —— 记忆真值库

`BaseDocumentStore`（ABC）：`upsert/get/delete/list/close`。

`SQLiteDocumentStore(path)`

- 表 `memories`：一列一个 `MemoryItem` 字段，`embedding` 以 JSON 存；
- 连接模型：`:memory:` 时固定单条共享连接；文件库**每次调用开一条独立连接**（`check_same_thread` 友好），`RLock` 包住每个 `_connection_scope`；
- `connection` 属性暴露当前共享连接（`DeletionProposalStore`/`execute_deletion` 靠它实现同库复用）；
- `list(memory_type, include_expired)`：按 `memory_type` 过滤、按过期时间过滤；`_decode` 负责 JSON 还原（含 `_json_restore` 的 bytes 还原）。

---

## 4.12 memory/storage/document_repo.py —— documents/chunks 真值源

**定位**：`memories` 装四层记忆，`documents`/`chunks` 装语料本身（原文 + 分块边界）。向量库与图都是可从本表重建的**投影**，所以本模块从不访问它们——把原文留在磁盘上，才是换嵌入空间后还能重索引的前提。

### 模块常量（状态机）

`DOCUMENT_STATUSES = (uploaded, parsed, vectorized, extracted, failed)`、`INGEST_JOB_STATUSES = (pending, running, done, failed)`、`CHUNK_VECTOR_STATUSES = (pending, indexed, failed)`、`PERMISSIONS = (private, shared, public)`、`FTS_TOKENIZERS = (trigram, unicode61)`。

### 数据类

- `DocumentRecord`：`document_id/title/raw_text/source/tags/permission/status/error/created_at/updated_at`
- `ChunkRecord`：`chunk_id/document_id/chunk_index/char_start/char_end/text/vector_status`
- `EmbeddingLockRecord`（frozen）：`model/dimension/updated_at`（单行锁）
- `IngestJobRecord`：一句话后台入库任务，`status` 随时可查，`result` 存完成后的 JSON 摘要
- `DeletionProposal`：待确认删除提议（见下）

### DocumentRepository

连接与锁模型与 `SQLiteDocumentStore` 一致：文件库每作用域独立连接，`:memory:` 固定单连接（可注入与 `memories` 同库的连接）。

`_initialize()` 建表：`documents`（含 `idx_documents_status`）、`chunks`（含 `idx_chunks_document`）、`delete_proposals`、`ingest_jobs`（含状态索引）、`embedding_lock`（`CHECK (id = 1)` 强制单行），最后 `_initialize_fts`。

`_initialize_fts(connection)`：给 `chunks.text` 建 FTS5 外部内容表 + `AFTER INSERT/DELETE/UPDATE` 三个触发器同步索引；分词器优先 `trigram`（支持中文子串），不可用时退 `unicode61`；已存在的表沿用当初建成的分词器（避免把 unicode61 误报成 trigram）；**FTS5 不可用不阻止仓储打开**，只把 `fts_tokenizer` 置空，检索退化为纯向量（D8）。

| 分组 | 方法 |
| --- | --- |
| 文档 | `upsert_document`（`created_at` 只在首次插入落库，重跑 ingest 不刷新）、`get_document`、`list_documents(tag, status, page, page_size)`（分页 + `json_each` 按标签过滤，`total` 不计分页）、`count_documents`、`set_status`、`delete_document`（返回删除的 chunk 数） |
| 分块 | `upsert_chunk`、`upsert_chunks`、`_write_chunk`（`ON CONFLICT` upsert）、`get_chunk`、`list_chunks(document_id)`、`set_chunk_vector_status` |
| 关键词检索 | `_fts_query`（**把用户输入包成带引号的短语并双写内层引号**——裸 `abc-123` 会被 FTS5 当成列语法报 `no such column: 123`，`型号: X200` 同理，历史上因此返 500）、`search_keywords(query, limit)`（BM25，取负号让高分在前） |
| 报告 | `chunk_counts`（一次 GROUP BY 给出每文档块数）、`chunk_ids(vector_status)`、`list_all_chunks`（全量重建投影用） |
| 嵌入锁 | `get_embedding_lock` / `set_embedding_lock(model, dimension)` |
| 统计/关闭 | `stats()` → `{documents, chunks, chunks_indexed}`；`close()` |
| 入库队列 | `create_ingest_job`、`get_ingest_job`、`set_ingest_job_status`（转 `running` 时 `attempts + 1`）、`list_ingest_jobs(status, limit)`（200 条硬上限）、`restart_stale_ingest_jobs`（遗留 `running` → `pending`）、`reset_failed_ingest_job`（用户重试，清零 attempts） |
| 工具 | `_check_choice`（枚举校验）、`_decode_document` / `_decode_chunk` |

### DeletionProposalStore 与 execute_deletion

- 常量：`DELETION_PROPOSAL_STATUSES = (pending, confirmed, rejected, expired)`、`DELETION_PROPOSAL_TTL_MINUTES = 15`；
- `create(requested_by, reason, item_ids, ttl_minutes)`：生成 8 位大写 `confirm_token`；`item_ids` 去重；
- `get(proposal_id)`：读取时顺带把过期的 `pending` 提议标为 `expired`；
- `_set_status(...)`；
- `confirm(proposal_id, token)`：**单条带守卫的 UPDATE**（`status='pending' AND confirm_token=?`），并发下只有一个调用方能改到行；失败时再分诊给出「不存在 / 已过期 / token 不匹配 / 非 pending」四种可读错误（fail closed）；
- `execute_deletion(proposal_id, token, manager)`：确认后逐条 `manager.delete`，语义事实经 `SemanticMemory.delete` 硬删（连带 Neo4j 边），最后写一条 `deletion_audit` 的 episodic 记录留痕；重复确认同一提议直接返回（幂等，绝不删两次）。

---

## 4.13 memory/storage/vector.py —— 向量索引

- `BaseVectorStore`（ABC）：`upsert/delete/search`；
- `cosine_similarity(a, b)`：长度不等、含非有限值、任一零向量都返回 `0.0`；
- `InMemoryVectorStore`：`dict[id → (vector, memory_type)]` + `RLock`；`upsert` 拒绝空向量与非有限值；`search` 线性扫描后按 `(-score, id)` 排序；`recreate_collection(dimension)` 清空（供确认后的向量空间重建）。

---

## 4.14 memory/storage/qdrant.py —— Qdrant 适配器

模块函数：

- `_vector_size_of(v)` / `_collection_vector_size(info)`：从 `VectorParams`、命名向量字典或 dict 载荷里取出集合维度（**只认集合真实尺寸，配置文件里的 `[embedding].dimension` 不是集合现有维度，不能拿来比对**）；
- `_is_dimension_error(exc)`：识别 `expected dim` / `vector dimension error` / `dimension mismatch`；
- `_dimension_mismatch_message(...)`：维度冲突的中文修复指引（唯一的维度开关是 `[embedding].model`，要么改回原模型，要么确认重建投影）。

类 `QdrantVectorStore(url, collection_name, api_key, client, dimension, namespace, proxy_url)`：

- 客户端构造：未给 `client` 时按 URL 选择——**回环地址显式 `trust_env=False`**（用户开着 Clash 等代理时，httpx 默认信任环境/注册表代理会把 127.0.0.1 请求转发到代理端口并返回 502）；云端地址按 `[proxy]` 配置走本地转发代理；无 `url` 则本地 `path=":memory:"`；
- `collection_dimension()` / `_live_collection_size()`：读集合是否存在及其真实维度；
- `_ensure_collection(dimension)`：集合不存在则按 COSINE 距离创建；存在则与真实尺寸比对，不一致抛 `ValueError`；随后 `_ensure_payload_indexes`；
- `_ensure_payload_indexes()`：为 `namespace`/`memory_type` 建 keyword 载荷索引（Qdrant Cloud 的 HTTP API 拒绝在无索引字段上做过滤查询，HTTP 400）；索引是检索需求不是写入需求，建失败不影响 upsert；
- `recreate_collection(dimension)`：删除并重建集合；**向量只是 SQLite 真值的投影，删集合不丢数据**，且从不在写入路径上静默触发；
- `upsert(item)`：写入轻量 payload（只留 `id/memory_type/created_at/expires_at` + `namespace`）——chunk 文本与元数据已在 SQLite，1024 维浮点才是体积大头，存副本等于造第二份会漂移的真值；
- `upsert_chunk(chunk_id, vector, document_id, chunk_index, source, memory_type)`：重索引路径的窄 payload 写入；与 `upsert` 落到同一个 `chunk_id`，SQLite 回查口径一致；
- `_ensure_ready_for_read()`：**只读路径不创建集合、不置 `_ready`**（否则纯读进程会把「集合不存在」误判成空，还会跳过写入时的维度核对）；
- `list_ids(limit)`：滚动取全部 app 级 id（对账需要全集）；
- `search(vector, limit, memory_type)`：强制按 `namespace` 过滤；优先 `search`，老客户端回落 `query_points`；
- `_point_id(item_id)`：Qdrant 只接受 UUID/整数点 id，非 UUID 的 app id 用 `uuid5(NAMESPACE_URL, "helloagents-memory:"+id)` 稳定映射，原 id 保留在 payload 里。

---

## 4.15 memory/storage/graph.py —— Neo4j 图库（含内存回退）

### 进程级 socket.getaddrinfo 补丁

Neo4j 驱动在连接与路由阶段都会按成员主机名解析地址，因此补丁必须陪伴驱动整个生命周期。实现：引用计数 + 链式帧（`_socket_patch_lock`、`_socket_patch_original`、`_socket_patch_frames`）。

- `_chained_getaddrinfo(...)`：从最新帧往前依次尝试，任一帧返回非 None 即采用；都不命中走最初捕获的原函数；
- `_make_proxy_frame(broker)`：`_should_proxy_host(host)` 为真（Aura 域名）时，从 `ProxyBroker` 取/建隧道，把解析目标改成 `127.0.0.1:<local_port>`；其他主机返回 None；
- `_install_socket_patch(broker)` / `_uninstall_socket_patch(frame)`：首个帧负责替换 `socket.getaddrinfo`，末个帧负责还原——多实例共存时谁安装谁卸载，互不覆盖，同一时间不会有两份互相打架的全局补丁。

### Neo4jGraphStore

构造：`uri/username/password/driver/database/proxy_url`；`driver=None` 且给了 `uri` 时调 `_open_driver`。无 driver 时走内存回退结构：`_local`（出边）、`_reverse`（入边）、`_entities`（实体属性镜像）、`_observations`（n 元观察）。

| 方法 | 行为 |
| --- | --- |
| `_open_driver()` | 建 Bolt 驱动（`connection_timeout=15`、`max_connection_lifetime=60`、`liveness_check_timeout=10`）；Aura + `proxy_url` 时先装 socket 补丁，驱动构造失败立即卸载，避免泄漏进程级全局补丁 |
| `_discard_socket_patch()` / `_reopen_driver()` | 卸载补丁并关 broker；重连时先关旧驱动再重建（路由抖动时重试一次） |
| `_with_session(runner)` | 包一层 `session`；只对 `ServiceUnavailable`/`SessionExpired`/含 "routing information" 的异常重连一次，其他异常原样抛 |
| `add_relation(source, relation, target, properties, source_domain, target_domain, *_aliases, *_importance, bump)` | `bump=True` 时只调 `_bump_relation`（F1 回忆强化：不动实体、不造边）。常规写入用 `MERGE`：实体 `ON CREATE` 才写 domain/importance，`ON MATCH` 只补别名（后续边写入不该把首次抽取的属性覆盖成空值）；边 `ON CREATE` 才给 `weight=1.0 / recall_count=0 / last_accessed_at=""`（`SET r += $properties` 是覆盖语义，常规写入碰它们会把累计回忆计数清零） |
| `add_observation(observation_id, predicate, participants, properties)` | 幂等落一条 n 元观察：`MERGE (o:MemoryObservation {id})`，先删旧 `HAS_PARTICIPANT` 再按 participants 重建（带 `role`/`ordinal`/`entity_type`）。participants 至少 2 个且必须有一个 `subject`。独立观察标识让同一三元组在不同时间点的陈述不互相坍缩 |
| `_merge_entity(name, domain, aliases, importance)` | 内存回退里 Cypher 实体子句的孪生实现 |
| `_bump_relation(s, r, t)` | Cypher：`recall_count+1`、`last_accessed_at=now`、`weight = min(weight*GROWTH, MAX)`；边不存在返回 False（强化不能凭空造边） |
| `_apply_bump(props, at)` | 内存版同款逻辑 |
| `entity(name)` | 实体属性；有 driver 时**以 Neo4j 为准**（`_entities` 只是本进程镜像，新进程里是空的，直接读会静默返回「没有别名」） |
| `entity_aliases()` | 一条 Cypher 取回全部实体别名（星云图避免 N+1） |
| `graph_snapshot(at)` | 返回 `{mode, entities, observations, relations}`；无 `at` 返回全量（前端渲染完整时间线），有 `at` 则按时间过滤 |
| `_temporal_bounds(props, at)` | 由 `valid_from/valid_to/event_at` 算出「在 at 时刻是否可见」 |
| `_observations_at / _relations_at` | 非时序观察/边全保留；`cardinality == "temporal"` 的按 `(subject, predicate)` 槽位只保留 `event_at` 最新的一条；`status == "expired"` 一律不返回 |
| `relation_memory_ids()` | 全部边的 `memory_id`（对账用） |
| `get_relations(entity, relation, direction, at)` / `related` | 按方向（in/out/both）查边，再过时间过滤 |
| `path_query(start, target, max_depth, limit)` | 变长路径查询；`path_weight` 用 `reduce` 累加边权重，`ORDER BY path_weight DESC` 让 LIMIT 截断发生在权重排序之后（F1）。变长区间上界必须拼进语句，用 `int()` 收敛为整数 |
| `_neighbours(entity)` | `(对端, 关系, 是否出边, weight)`，按权重降序 |
| `_local_paths(...)` | 内存回退的 BFS 简单路径（队列保证短路径先出，不重复经过同一实体，天然无环） |
| `delete_memory_relation(memory_id)` | 删 `RELATED {memory_id}` 边并 `DETACH DELETE` 对应观察节点；内存版同步清理 `_local/_reverse/_observations` |
| `close()` | 关驱动并卸载补丁 |

---

## 4.16 一次「写入」的完整链路

1. 调用方（Web API / Agent 工具 / 脚本）调 `MemoryManager.add(...)` 或 `SemanticMemory.add_fact(...)`；
2. `BaseMemory.add` 先算向量（`APIEmbedding` 或 `HashEmbedding`）→ **先 `document_store.upsert` 写真值** → 再 `vector_store.upsert` 写投影，失败则回滚真值行；
3. `SemanticMemory._write_edge` 同步把事实写成 Neo4j 的 `RELATED` 边 + `MemoryObservation`/`HAS_PARTICIPANT` 多元观察；
4. `RAGPipeline.ingest` 额外先把原文与分块边界写进 `documents`/`chunks`，再让抽取器提议补丁、由 `materialize_extraction` 确定性落地。

## 4.17 一次「检索」的完整链路

1. `RAGPipeline.hybrid_retrieve`：向量路（`manager.search`）+ 关键词路（`search_keywords`，FTS5 BM25）→ `_rrf_fuse(k=60)` 融合 → 只返回真值源里存在的 chunk；
2. 向量路失败时把原因写进 `last_retrieval_note` 并降级为纯关键词（D8）；
3. `graph_retrieve` 再走 `GraphRAGPipeline.retrieve`：证据 → 种子实体 → BFS 多跳（过滤过期/撤回/时间窗外边）→ 按 `effective` 排序 → 对返回路径做回忆强化。
