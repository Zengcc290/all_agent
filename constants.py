"""全项目统一的常量与默认值（单一事实来源）。

所有模块从这里导入常量，禁止在本模块之外重新定义同名值。
注意：``tool/*.py`` 的 ``TOOL_ENABLED`` 不在此处——它是工具发现协议
要求的"每个工具模块各自的布尔开关"，由 ``core.discovery`` 逐模块读取，
集中后所有工具会共享同一个开关，破坏协议设计。

每个常量都以注释标注它"所在/来自"的源文件，便于溯源。
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
# 嵌入服务（memory/embedding.py）
#   所在文件：memory/embedding.py（APIEmbedding 默认参数）、memory/base.py（MemoryConfig）
# ---------------------------------------------------------------------------

#: 默认厂商端点与模型（qwen3-embedding-0.6b，1024 维，DashScope 兼容端点）。
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
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

#: /api/ingest 上传大小上限（字节，64MB）。
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

#: /api/ingest 的 RAG 切块块长（比默认 1000 更细，用于文档）。
WEB_INGEST_CHUNK_SIZE = 800

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

#: 历史常量（保留以兼容旧引用）：文档知识块已改为按内容自动分类到主题恒星系。
DOC_DOMAIN = "文档库"

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