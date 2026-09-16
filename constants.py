"""全项目统一的常量与默认值（单一事实来源）。

所有模块从这里导入常量，禁止在本模块之外重新定义同名值。
注意：``tool/*.py`` 的 ``TOOL_ENABLED`` 不在此处——它是工具发现协议
要求的"每个工具模块各自的布尔开关"，由 ``core.discovery`` 逐模块读取，
集中后所有工具会共享同一个开关，破坏协议设计。

每个常量都以注释标注它"所在/来自"的源文件，便于溯源。

下文「连接与端点」一节是**「服务连哪里」的唯一事实来源**（本机回环端口、
Qdrant/Neo4j 端点、嵌入网关与隧道端口）；部署机密的覆盖入口是 ``.env``
（不入库）。

刻意**不**收进来的（各有归属，集中反而割裂，列出出处便于查找）：
  - ``tool/*.py`` 的 ``TOOL_ENABLED``：工具发现协议要求的每模块开关（见上）。
  - ``memory/storage/document_repo.py`` 的状态枚举（``DOCUMENT_STATUSES`` /
    ``CHUNK_VECTOR_STATUSES`` / ``PERMISSIONS`` / ``FTS_TOKENIZERS``）：与同文件
    的 DDL 同源，改表结构就得改它，分开会漂移。
  - 各工具自己的协议上限（``tool/fs_*.py``、``tool/_shared.py`` 的 ``MAX_*`` /
    ``*_ENV``）：只被该工具读取，属工具契约的一部分。
  - ``web/domain_classifier.py`` 的 ``DOMAIN_KEYWORDS`` 词表、
    ``web/seed.py`` 的 ``SEED_MARK``、``agents/message_utils.py`` 的
    ``DEFAULT_TOOL_NAME`` / ``MAX_TOOL_*``、``scripts/*.py`` 的 ``ROOT``：
    分别是词表数据、种子标记、消息清洗上限与脚本自身锚点。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Agent / LLM 请求默认值
#   所在文件：agents/llm.py、agents/agent.py、agents/react.py
# ---------------------------------------------------------------------------

#: OpenAI SDK 与 LLM 客户端共用的最大重试次数。
DEFAULT_MAX_RETRIES = 3

#: 模型采样温度与请求超时（秒）的统一默认值。
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 60

#: prompt-cache 路由键的命名空间版本，改动后旧缓存键全部失效。
PROMPT_CACHE_KEY_VERSION = "pc-v1"

#: 技能包根目录的默认位置（相对于项目根）。
DEFAULT_SKILLS_ROOT = "skills"

# ---------------------------------------------------------------------------
# 工具循环（ToolLoop）
#   所在文件：core/tool_loop.py（DEFAULT_SAFETY_LIMIT 兼容别名）
# ---------------------------------------------------------------------------

#: 未指定 max_rounds 时允许的最大轮数，防止失控的 provider 无限请求。
TOOL_LOOP_SAFETY_LIMIT = 64

# ---------------------------------------------------------------------------
# 历史压缩（保存对话时的超大工具结果打桩）
#   所在文件：agents/profile 压缩逻辑
# ---------------------------------------------------------------------------

# 超过该长度的 Observation 在写入 profile 历史时被压缩；
# 正在运行的一轮内仍保留完整负载。
OBSERVATION_COMPRESS_THRESHOLD = 12_000
OBSERVATION_STUB_PREFIX = "[已压缩的历史工具结果"
OBSERVATION_PREVIEW_CHARS = 400

#: 单条会话历史保留的最大消息数（保留开头的 system 前缀，只裁剪旧对话轮）。
#: 不设上限时历史会无限增长：每轮都把全部历史重发给模型，token 成本线性上升。
HISTORY_MAX_MESSAGES = 60

# ---------------------------------------------------------------------------
# 技能包（skills/<name>.md）校验与发现
#   所在文件：core/skill_models.py、core/skill_discovery.py、core/skill_registry.py
# ---------------------------------------------------------------------------

#: 技能名规则：kebab-case ASCII，最长 64 字符。
SKILL_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

MAX_DESCRIPTION_CHARS = 2000
MAX_VERSION_CHARS = 32
MAX_TRIGGER_CHARS = 200

#: content_hash 必须是长度 64 的小写 SHA-256 十六进制摘要。
CONTENT_HASH_LENGTH = 64

SKILL_FILE_SUFFIX = ".md"
LEGACY_SKILL_ENTRY_FILENAME = "SKILL.md"
README_FILENAME = "README.md"
ENABLED_FIELD = "enabled"

# ---------------------------------------------------------------------------
# 数据库文件名默认值（环境变量可覆盖运行时路径）
#   所在文件：memory/base.py、web/support.py、tool/update_log.py、core/repository.py
# ---------------------------------------------------------------------------

DEFAULT_TOOLS_DB_FILENAME = "tools.sqlite3"
DEFAULT_UPDATE_LOG_FILENAME = "update_log.sqlite3"
DEFAULT_MEMORY_DB_FILENAME = "memory.sqlite3"

# ---------------------------------------------------------------------------
# 连接与端点（本机回环 / SSH 隧道）——「服务连哪里」的唯一事实来源
#   所在文件：constants.py（本段定义）→ 直接 import 方：
#     memory/base.py（MemoryConfig 连接字段的默认值与说明）、
#     memory/embedding.py（EmbedServerEmbedding 默认 base_url、gateway_reachable）、
#     memory/storage/qdrant.py（默认 collection）、web/app.py（服务监听 host/port）
#   间接消费方（不 import，由 MemoryConfig 传值决定选型）：
#     memory/manager.py（qdrant_url/neo4j_uri 非空才建真存储，否则内存回退）、
#     memory/storage/graph.py（URI 由 MemoryConfig 传参）
#   部署覆盖入口：.env（不入库；模板见仓库外，键名见下方各常量注释）
#   重要：下面的 QDRANT / NEO4J 端点是「本机文档化默认值」，**不是默认启用**。
#   MemoryConfig.qdrant_url / neo4j_uri 出厂仍为 None（= 不连接、走内存回退），
#   只有显式配置（.env 或构造参数）才连真服务；理由见方案 §12 与 F3。
# ---------------------------------------------------------------------------

#: 本机服务统一绑定的回环地址。用字面 IP 而不是 "localhost"：后者在
#: 部分 Windows 环境解析到 ::1，而服务只监听 IPv4，表现为"连不上"。
LOCALHOST = "127.0.0.1"

#: 本机回环端口分配表（改端口只改这里）：
#:   10800 = SSH 隧道入口，即 EMBEDDING_BASE_URL 的端口；
#:    6333 = 本地 Qdrant HTTP；
#:    7687 = 本地 Neo4j bolt；
#:    8765 = 本地 Web 服务（环境变量 NEBULA_PORT 可覆盖）。
DEFAULT_EMBEDDING_GATEWAY_PORT = 10800
DEFAULT_QDRANT_PORT = 6333
DEFAULT_NEO4J_BOLT_PORT = 7687
DEFAULT_WEB_PORT = 8765

#: 隧道另一端的 qwen-embed 网关端口。主机名与 SSH 端口属部署信息，**不写进
#: 仓库**：由 .env 的 EMBEDDING_TUNNEL_HINT 或部署文档提供（见 .env 注释）。
DEFAULT_EMBEDDING_TUNNEL_REMOTE_PORT = 18000

#: 本地 Qdrant 端点。启用方式（.env，不入库）：
#:   HELLOAGENTS_MEMORY_QDRANT_URL=http://127.0.0.1:6333
DEFAULT_QDRANT_URL = f"http://{LOCALHOST}:{DEFAULT_QDRANT_PORT}"

#: 本地 Neo4j 端点与账号。启用方式（.env，不入库）：
#:   HELLOAGENTS_MEMORY_NEO4J_URI=bolt://127.0.0.1:7687
#:   HELLOAGENTS_MEMORY_NEO4J_USERNAME=neo4j
#:   HELLOAGENTS_MEMORY_NEO4J_PASSWORD=<你的密码>
#: 密码这里只是 Neo4j 首次安装的出厂占位，真实密码只放 .env。
DEFAULT_NEO4J_URI = f"bolt://{LOCALHOST}:{DEFAULT_NEO4J_BOLT_PORT}"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "neo4j"

# ---------------------------------------------------------------------------
# 嵌入服务（memory/embedding.py）
#   所在文件：memory/embedding.py（APIEmbedding 默认参数）、memory/base.py（MemoryConfig）
# ---------------------------------------------------------------------------

#: 默认端点与模型（qwen3-embedding-0.6b，1024 维）。
#: 端点默认指向**本机隧道**（方案 §0.1/D8：网关是唯一嵌入来源；端口取自
#: 上面的 DEFAULT_EMBEDDING_GATEWAY_PORT，.env 里的 EMBEDDING_TUNNEL_HINT
#: 给出建隧道的命令）。不要改回公网厂商端点：
#: 那样在缺 .env 时会静默改用另一套向量空间（§12「向量空间不可互换」），
#: 而指向本机隧道时隧道不通会明确降级为关键词检索并提示隧道命令。
DEFAULT_EMBEDDING_BASE_URL = f"http://{LOCALHOST}:{DEFAULT_EMBEDDING_GATEWAY_PORT}"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"

#: DashScope 风格批处理上限；其他厂商可在构造时降低/提高 batch_size。
DEFAULT_EMBEDDING_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# 记忆层（memory/base.py 的 MemoryConfig 字段默认值）
#   所在文件：memory/base.py
# ---------------------------------------------------------------------------

#: 未显式指定过期时间时的工作记忆 TTL（秒）。
MEMORY_DEFAULT_TTL_SECONDS = 3600.0

#: 工作记忆可容纳的条目上限。
MEMORY_WORKING_CAPACITY = 100

#: 语义检索默认返回条数。
MEMORY_SEARCH_LIMIT = 10

#: 语义检索默认相似度阈值（0 表示不过滤）。
MEMORY_SIMILARITY_THRESHOLD = 0.0

#: qwen3-embedding-0.6b 的向量维度。
MEMORY_EMBEDDING_DIMENSION = 1024

#: 嵌入请求超时（秒）。
MEMORY_EMBEDDING_TIMEOUT = 30.0

#: 记忆库 SQLite 默认路径（":memory:" 表示不持久化）。
MEMORY_SQLITE_DEFAULT = ":memory:"

#: 可选 Qdrant 后端的默认集合名。
MEMORY_QDRANT_COLLECTION = "helloagents_memory"

# ---------------------------------------------------------------------------
# RAG 文档处理与检索（memory/rag/）
#   所在文件：memory/rag/document.py、memory/rag/pipeline.py、memory/rag/graph_rag.py
# ---------------------------------------------------------------------------

#: 文档切块默认块长与重叠长度。
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 100

#: 检索/上下文装配默认返回条数。
RAG_RETRIEVE_LIMIT = 5

#: 图检索默认跳数及其允许上限。
RAG_GRAPH_HOPS = 1
RAG_GRAPH_MAX_HOPS = 3

#: 图路径展开上限。
RAG_GRAPH_PATH_LIMIT = 20

#: 图检索上下文拼装的最大字符数。
RAG_CONTEXT_MAX_CHARS = 12000

#: rag 工具 ingest 允许的单次文本上限（tool/rag_tool.py 的 Field le）。
RAG_CHUNK_MAX = 100_000

# ---------------------------------------------------------------------------
# 知识抽取（memory/rag/knowledge.py）
#   所在文件：memory/rag/knowledge.py
# ---------------------------------------------------------------------------

#: 实体名相似度合并阈值（SequenceMatcher / token 重叠取最大）。
ENTITY_SIMILARITY_THRESHOLD = 0.88

#: 实体解析默认置信度与默认实体类型。
ENTITY_DEFAULT_CONFIDENCE = 0.8
ENTITY_DEFAULT_TYPE = "概念"

#: 实体名长度上限（解析时截断）与文本清理默认上限。
ENTITY_NAME_MAX_LENGTH = 200
ENTITY_CLEAN_MAX_LENGTH = 500

#: 别名前缀匹配的最短公共前缀。低于该长度只接受精确匹配，避免
#: 「we」这类过短前缀把「web 中转站」和「web 网关」错误合并成同一实体。
ENTITY_PREFIX_MIN_LENGTH = 3

#: 抽取前注入提示词的子图关系条数上限（按置信度排序）。
GRAPH_CONTEXT_MAX_RELATIONS = 60

# ---------------------------------------------------------------------------
# ReAct 同步/文本协议容错（agents/react.py）
#   所在文件：agents/react.py
# ---------------------------------------------------------------------------

#: ReAct 轮内未标记答案 / 畸形答案的重试上限。
REACT_UNMARKED_ANSWER_RETRY_LIMIT = 3
REACT_MALFORMED_ANSWER_RETRY_LIMIT = 3

# ---------------------------------------------------------------------------
# Web API（web/app.py）
#   所在文件：web/app.py
# ---------------------------------------------------------------------------

#: /api/ingest 与 /api/import 上传大小上限（字节，64MB）。
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

#: /api/chat 消息长度上限。
WEB_CHAT_MAX_CHARS = 8000

#: /api/facts 三元组各字段长度上限。
WEB_FACT_SUBJECT_MAX = 200
WEB_FACT_PREDICATE_MAX = 100
WEB_FACT_OBJECT_MAX = 200
WEB_FACT_DOMAIN_MAX = 100
WEB_FACT_NOTE_MAX = 4000

#: /api/graph-rag 查询长度上限与 limit 上限。
WEB_GRAPH_RAG_QUERY_MAX = 8000
WEB_GRAPH_RAG_LIMIT_MAX = 50

#: /api/knowledge 一句话文本长度上限。
WEB_KNOWLEDGE_MAX_CHARS = 20000

#: /api/import 响应中回传的错误明细条数上限（避免超长响应）。
WEB_IMPORT_ERRORS_MAX = 20

#: /api/ingest 的 RAG 切块块长（比默认 1000 更细，用于文档）。
WEB_INGEST_CHUNK_SIZE = 800

#: 问答抽取的切块块长。问答通常很短，不需要按文档大小切。
QA_EXTRACT_CHUNK_SIZE = 2000

# ---------------------------------------------------------------------------
# 星图构建（web/graph_builder.py）
#   所在文件：web/graph_builder.py
# ---------------------------------------------------------------------------

#: 领域配色板（与 Aetheria 深空青紫主题协调）。
NEBULA_PALETTE = [
    "#38bdf8",  # 青
    "#c084fc",  # 紫
    "#f43f5e",  # 玫红
    "#fbbf24",  # 金
    "#34d399",  # 绿
    "#60a5fa",  # 蓝
    "#f472b6",  # 粉
    "#a3e635",  # 黄绿
]

#: 兜底领域名与内置「事件时间线」的域名/实体名/节点 ID。
DEFAULT_DOMAIN = "未分类"
TIMELINE_DOMAIN = "时间线"
TIMELINE_ENTITY = "事件时间线"
TIMELINE_ID = "ent:__timeline__"

#: 星图节点文案截断长度（节点正文预览 / 事件标题 / 日期字符串）。
NEBULA_CONTENT_PREVIEW_CHARS = 400
NEBULA_EVENT_TITLE_CHARS = 24
NEBULA_DATE_CHARS = 10

# ---------------------------------------------------------------------------
# 领域分类器（web/domain_classifier.py）
#   所在文件：web/domain_classifier.py
# ---------------------------------------------------------------------------

#: 标题命中的加权系数（文件名常含主题词，如“c语言笔记.txt”）。
DOMAIN_TITLE_WEIGHT = 3