"""全项目统一的常量与默认值（单一事实来源）。

所有模块从这里导入常量，禁止在本模块之外重新定义同名值。
注意：``tool/*.py`` 的 ``TOOL_ENABLED`` 不在此处——它是工具发现协议
要求的"每个工具模块各自的布尔开关"，由 ``core.discovery`` 逐模块读取，
集中后所有工具会共享同一个开关，破坏协议设计。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Agent / LLM 请求默认值
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
# ---------------------------------------------------------------------------

#: 未指定 max_rounds 时允许的最大轮数，防止失控的 provider 无限请求。
TOOL_LOOP_SAFETY_LIMIT = 64

# ---------------------------------------------------------------------------
# 历史压缩（保存对话时的超大工具结果打桩）
# ---------------------------------------------------------------------------

# 超过该长度的 Observation 在写入 profile 历史时被压缩；
# 正在运行的一轮内仍保留完整负载。
OBSERVATION_COMPRESS_THRESHOLD = 12_000
OBSERVATION_STUB_PREFIX = "[已压缩的历史工具结果"
OBSERVATION_PREVIEW_CHARS = 400

# ---------------------------------------------------------------------------
# 技能包（skills/<name>.md）校验与发现
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
# ---------------------------------------------------------------------------

DEFAULT_TOOLS_DB_FILENAME = "tools.sqlite3"
DEFAULT_UPDATE_LOG_FILENAME = "update_log.sqlite3"
DEFAULT_MEMORY_DB_FILENAME = "memory.sqlite3"
