# 01 · constants.py —— 全项目唯一常量源

模块定位：所有模块从这里 import 常量，禁止在别处重新定义同名值（唯一例外见文件头注释：`tool/*.py` 的 `TOOL_ENABLED` 是工具发现协议要求的每模块开关，必须留在工具文件里）。

## Agent / LLM 请求默认值（消费方 agents/llm.py、agents/agent.py、agents/react.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `DEFAULT_MAX_RETRIES` | 3 | OpenAI SDK 与 LLM 客户端共用的最大重试次数 |
| `DEFAULT_TEMPERATURE` | 0.7 | 模型采样温度默认值 |
| `DEFAULT_TIMEOUT` | 60 | 单次模型请求超时（秒） |
| `PROMPT_CACHE_KEY_VERSION` | "pc-v1" | prompt-cache 路由键命名空间版本；改动后旧缓存键全部失效 |

## 工具循环（消费方 core/tool_loop.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `TOOL_LOOP_SAFETY_LIMIT` | 64 | 未指定 max_rounds 时的最大轮数，防止 provider 失控无限请求 |

## 更新日志读取（消费方 tool/read_update_logs.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `UPDATE_LOG_READ_RANGE_MAX` | 100 | 单次批量读取 update_log 的最大记录数（防止一次拉爆内存/超时） |

## 历史压缩（消费方 agents/agent.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `OBSERVATION_COMPRESS_THRESHOLD` | 12000 | 超过该长度的 Observation 在写入 profile 历史时被打桩压缩；正在运行的一轮内仍保留完整负载 |
| `OBSERVATION_STUB_PREFIX` | "[已压缩的历史工具结果" | 压缩桩的前缀标记 |
| `OBSERVATION_PREVIEW_CHARS` | 400 | 压缩桩中保留的原文预览字符数 |
| `HISTORY_MAX_MESSAGES` | 60 | 单条会话历史保留的最大消息数（保留开头 system 前缀，只裁剪旧对话轮） |

## 数据库文件名（消费方 memory/base.py、web/support.py、tool/update_log.py、core/repository.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `DEFAULT_TOOLS_DB_FILENAME` | "tools.sqlite3" | 工具元数据目录库（ToolSpecRepository 默认路径） |
| `DEFAULT_UPDATE_LOG_FILENAME` | "update_log.sqlite3" | 项目更新日志库（可被 UPDATE_LOG_DB_PATH 覆盖） |
| `DEFAULT_MEMORY_DB_FILENAME` | "memory.sqlite3" | 记忆库默认路径（可被 MEMORY_DB_PATH 覆盖，这是唯一保留的路径覆盖入口） |

## 连接与端点（「服务连哪里」的唯一事实来源）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `LOCALHOST` | "127.0.0.1" | 本机服务统一绑定地址；用字面 IP 而非 localhost，避免部分 Windows 环境解析到 ::1 导致连不上 |
| `DEFAULT_QDRANT_PORT` | 6333 | 本地 Qdrant HTTP 端口（端口分配表） |
| `DEFAULT_NEO4J_BOLT_PORT` | 7687 | 本地 Neo4j bolt 端口 |
| `DEFAULT_WEB_PORT` | 8765 | Web 服务端口 |
| `DEFAULT_QDRANT_URL` | "http://127.0.0.1:6333" | 本地 Qdrant 端点（文档化默认值，默认不启用） |
| `DEFAULT_NEO4J_URI` | "bolt://127.0.0.1:7687" | 本地 Neo4j 端点（默认不启用） |
| `DEFAULT_PROXY_PORT` | 7890 | 本机转发代理（Clash）默认端口 |
| `DEFAULT_PROXY_URL` | "http://127.0.0.1:7890" | 云端 Qdrant/Neo4j 未显式配置 [proxy] 时走的代理；回环地址绝不走代理 |

> 注意：MemoryConfig.qdrant_url / neo4j_uri 出厂为 None（不连接、走内存回退），只有显式配置（services.toml 或构造参数）才连真服务。

## 嵌入服务（消费方 memory/embedding.py、memory/base.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `DEFAULT_EMBEDDING_MODEL` | "qwen3-embedding-0.6b" | 未配置 [embedding].model 时的兜底模型名 |
| `DEFAULT_EMBEDDING_BATCH_SIZE` | 10 | DashScope 风格批处理上限 |

## 记忆层（消费方 memory/base.py 的 MemoryConfig 字段默认值）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `MEMORY_DEFAULT_TTL_SECONDS` | 3600.0 | 工作记忆未显式指定过期时间时的 TTL（秒，默认 1 小时） |
| `MEMORY_WORKING_CAPACITY` | 100 | 工作记忆条目上限 |
| `MEMORY_SEARCH_LIMIT` | 10 | 语义检索默认返回条数 |
| `MEMORY_SIMILARITY_THRESHOLD` | 0.0 | 默认相似度阈值（0 = 不过滤） |
| `MEMORY_EMBEDDING_DIMENSION` | 1024 | 离线 HashEmbedding 的默认维度 |
| `MEMORY_EMBEDDING_DIMENSION_REMOTE` | None | 远端嵌入期望维度；None = 首次调用从响应自动识别（APIEmbedding 把 0 当「尚未知」） |
| `MEMORY_EMBEDDING_PROVIDER_DEFAULT` | "auto" | 嵌入选型默认策略 |
| `MEMORY_EMBEDDING_PROVIDERS` | ("auto","openai","hash") | 合法提供方枚举 |
| `MEMORY_EMBEDDING_TIMEOUT` | 30.0 | 嵌入请求超时（秒） |
| `MEMORY_SQLITE_DEFAULT` | ":memory:" | 记忆库 SQLite 默认路径（MemoryConfig 默认不持久化） |
| `MEMORY_QDRANT_COLLECTION` | "helloagents_memory" | Qdrant 默认集合名 |

## RAG 文档处理与检索（消费方 memory/rag/）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` | 1000 | 文档切块默认块长 |
| `RAG_CHUNK_OVERLAP` | 100 | 相邻块重叠字符数（防语义被切断） |
| `RAG_RETRIEVE_LIMIT` | 5 | 检索/上下文装配默认返回条数 |
| `RAG_GRAPH_HOPS` | 1 | 图检索默认跳数 |
| `RAG_GRAPH_MAX_HOPS` | 3 | 图检索允许的最大跳数 |
| `RAG_GRAPH_PATH_LIMIT` | 20 | 图路径展开上限 |
| `RAG_CONTEXT_MAX_CHARS` | 12000 | 图检索上下文拼装的最大字符数 |

## F1 边强化「回忆即强化」（消费方 memory/storage/graph.py、memory/rag/graph_rag.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `MEMORY_EDGE_WEIGHT_GROWTH` | 1.5 | 一次「回忆」把边权重乘以的倍数（初始 1.0） |
| `MEMORY_EDGE_WEIGHT_MAX` | 8.0 | 边权重上限，防止「富者越富」推到无穷 |
| `MEMORY_EDGE_REINFORCE` | True | 总开关；False 时排序退回纯 confidence（测试 monkeypatch 用） |

## 运行开关

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `MEMORY_HYBRID` | True | 混合检索（FTS5 关键词 × 向量 RRF 融合）总开关；False = 纯向量 |

## 知识抽取（消费方 memory/rag/knowledge.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `ENTITY_SIMILARITY_THRESHOLD` | 0.88 | 实体名相似度合并阈值（SequenceMatcher / token 重叠取最大） |
| `ENTITY_DEFAULT_CONFIDENCE` | 0.8 | 实体解析默认置信度 |
| `ENTITY_DEFAULT_TYPE` | "概念" | 默认实体类型 |
| `ENTITY_NAME_MAX_LENGTH` | 200 | 实体名长度上限（解析时截断） |
| `ENTITY_CLEAN_MAX_LENGTH` | 500 | 文本清理默认上限 |
| `ENTITY_PREFIX_MIN_LENGTH` | 3 | 别名前缀匹配的最短公共前缀；低于该长度只接受精确匹配（防止「we」误合并「web 中转站」与「web 网关」） |
| `GRAPH_CONTEXT_MAX_RELATIONS` | 60 | 抽取前注入提示词的子图关系条数上限（按置信度排序） |

## ReAct 文本协议容错（消费方 agents/react.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `REACT_UNMARKED_ANSWER_RETRY_LIMIT` | 3 | 轮内未标记答案的重试上限 |
| `REACT_MALFORMED_ANSWER_RETRY_LIMIT` | 3 | 畸形答案的重试上限 |

## Web API（消费方 web/app.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | 64MB | /api/ingest 与 /api/import 上传大小上限（流式校验，超限 413） |
| `WEB_CHAT_MAX_CHARS` | 8000 | /api/chat 消息长度上限 |
| `WEB_FACT_SUBJECT_MAX` | 200 | /api/facts 主语长度上限 |
| `WEB_FACT_PREDICATE_MAX` | 100 | 谓词长度上限 |
| `WEB_FACT_OBJECT_MAX` | 200 | 宾语长度上限 |
| `WEB_FACT_DOMAIN_MAX` | 100 | 领域长度上限 |
| `WEB_FACT_NOTE_MAX` | 4000 | 备注长度上限 |
| `WEB_GRAPH_RAG_QUERY_MAX` | 8000 | /api/graph-rag 查询长度上限 |
| `WEB_GRAPH_RAG_LIMIT_MAX` | 50 | /api/graph-rag limit 上限 |
| `WEB_KNOWLEDGE_MAX_CHARS` | 20000 | /api/knowledge 一句话文本长度上限 |
| `WEB_IMPORT_ERRORS_MAX` | 20 | /api/import 响应回传的错误明细条数上限 |
| `WEB_DOCUMENTS_PAGE_SIZE_MAX` | 100 | /api/documents 分页每页上限 |
| `WEB_QA_QUESTION_MAX_CHARS` | 2000 | 问答留痕写入 episodic 的问题截断上限 |
| `WEB_QA_ANSWER_MAX_CHARS` | 8000 | 问答留痕写入 episodic 的答案截断上限 |
| `WEB_INGEST_CHUNK_SIZE` | 800 | /api/ingest 的 RAG 切块块长（比默认 1000 更细） |
| `QA_EXTRACT_CHUNK_SIZE` | 2000 | 问答抽取的切块块长 |
| `WEB_AUTOSEED` | True | Web 启动时是否自动播种演示数据 |
| `WEB_QA_EXTRACT` | True | 问答后自动抽取图事实总开关 |
| `WEB_QA_EXTRACT_SYNC` | False | 抽取是否改同步执行（仅调试/测试） |

## 星图构建（消费方 web/graph_builder.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `NEBULA_PALETTE` | 8 色数组 | 领域配色板（与 Aetheria 深空青紫主题协调） |
| `DEFAULT_DOMAIN` | "未分类" | 兜底领域名 |
| `NEBULA_CONTENT_PREVIEW_CHARS` | 400 | 星图节点正文预览长度 |
| `NEBULA_EVENT_TITLE_CHARS` | 24 | 事件标题截断长度 |
| `NEBULA_DATE_CHARS` | 10 | 日期字符串截断长度 |

## 领域分类器（消费方 web/domain_classifier.py）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `DOMAIN_TITLE_WEIGHT` | 3 | 标题命中的加权系数（文件名常含主题词，如「c语言笔记.txt」） |
