"""Web 层支撑设施：嵌入降级、共享单例与路径约定。

设计要点：
- 没有任何 API key 时整站仍可运行：``HashEmbedding`` 提供离线向量检索
  （质量有限，仅演示用）；填入 ``DASHSCOPE_API_KEY`` 后自动升级到
  qwen3-embedding-0.6b。
- ``MEMORY_DB_PATH`` 在导入时就被固定为项目根下的 ``memory.sqlite3``，
  保证 Agent 工具（memory.manage / memory.rag）与 Web API 共享同一个
  记忆库——「记忆共享」的数据面。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MEMORY_DB_FILENAME,
    MEMORY_EMBEDDING_DIMENSION,
    NEBULA_EVENT_TITLE_CHARS,
)

from memory import APIEmbedding, MemoryConfig, MemoryItem, MemoryManager, utc_now
from memory.rag import LLMKnowledgeExtractor, NullKnowledgeExtractor, RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
SEED_FILE = WEB_DIR / "seed_data.json"

# .env 中的 key 优先级高于环境变量（本地开发惯例）
load_dotenv(PROJECT_ROOT / ".env")

#: 统一记忆库路径：Web API 与 Agent 工具都读它。
DB_PATH = Path(os.getenv("MEMORY_DB_PATH") or (PROJECT_ROOT / DEFAULT_MEMORY_DB_FILENAME))
os.environ.setdefault("MEMORY_DB_PATH", str(DB_PATH))


class HashEmbedding:
    """离线确定性 embedding：未配置任何 API key 时的降级实现。

    token -> blake2b 哈希桶累加，再归一化。检索质量有限，但完全本地、
    可重复，适合课设演示与断网环境。
    """

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", (text or "").casefold(), flags=re.UNICODE)
        vector = [0.0] * self.dimension
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0 + math.log(float(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def build_embedding():
    """有 DashScope key 用真实嵌入，否则降级 HashEmbedding。"""
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if api_key:
        return APIEmbedding(
            api_key=api_key,
            model=DEFAULT_EMBEDDING_MODEL,
            dimension=MEMORY_EMBEDDING_DIMENSION,
        )
    return HashEmbedding()


def build_knowledge_extractor():
    """Create the configured extractor, or a safe offline no-op fallback."""

    from agents.llm import LLM
    from agents.providers import ProviderRegistry

    path = ProviderRegistry.default_config_path()
    if path.name != "provider.toml" or not path.is_file():
        return NullKnowledgeExtractor()
    try:
        registry = ProviderRegistry(path)
        profile = registry.get(registry.active_profile)
        api_key = registry.resolve_api_key(profile.name)
        if not api_key or api_key.startswith("replace-with"):
            return NullKnowledgeExtractor()
        client = LLM(
            api_key=api_key,
            base_url=profile.base_url,
            model=profile.default_model,
        )
        return LLMKnowledgeExtractor(client.complete, model=profile.default_model)
    except Exception:
        return NullKnowledgeExtractor()


_manager: MemoryManager | None = None
_manager_lock = Lock()


def get_manager() -> MemoryManager:
    """进程级单例 MemoryManager：Web API、RAG、种子脚本共用。"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = MemoryManager(
                    MemoryConfig(sqlite_path=str(DB_PATH)),
                    embedding=build_embedding(),
                )
    return _manager


def close_manager() -> None:
    global _manager
    if _manager is not None:
        _manager.close()
        _manager = None


_agent = None
_agent_lock = Lock()

#: 知识管家的行为约束：先检索记忆再回答。
SYSTEM_PROMPT = (
    "你是『星图』——用户的个人知识管家，管理着用户的知识库与记忆。遵守：\n"
    "1. 回答与用户知识、经历、文档相关的问题前，先用 memory.rag 的 graph_retrieve/context"
    " 行动检索知识库，必要时用 memory.manage 的 search 补充记忆检索。\n"
    "2. 用中文简洁回答；引用知识库内容时注明来源文件和关系证据（若有）。\n"
    "3. 不编造知识库里没有的内容；检索不到就如实说明。\n"
    "4. 用户明确让你记住某件事时，用 memory.manage 的 add 写入 episodic 记忆。"
)


def get_agent():
    """懒加载 ReActAgent 单例（真实聊天模型未配置时构造也能成功，
    调用前由 ``chat_ready`` 把关）。"""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                from agents import ReActAgent
                from tool.rag_tool import RAGTool

                agent = ReActAgent("knowledge-butler")
                agent.set_system_prompt(SYSTEM_PROMPT)
                agent.register_tool(
                    RAGTool(
                        pipeline=RAGPipeline(
                            get_manager(),
                            extractor=build_knowledge_extractor(),
                        )
                    ),
                    replace=True,
                )
                _agent = agent
    return _agent


#: 联网搜索工具的注册名；聊天「联网/非联网」开关据此决定是否可见。
SEARCH_TOOL_NAME = "web.search"


def search_available() -> bool:
    """AnySearch 是否已配置（base_url 与 api_key 同时存在才视为可用）。"""
    base_url = next(
        (
            value
            for value in (
                os.getenv("SEARCH_BASE_URL"),
                os.getenv("ANYSEARCH_BASE_URL"),
            )
            if value
        ),
        None,
    )
    api_key = next(
        (
            value
            for value in (
                os.getenv("SEARCH_API"),
                os.getenv("SEARCH_API_KEY"),
                os.getenv("ANYSEARCH_API_KEY"),
            )
            if value
        ),
        None,
    )
    return bool(base_url) and bool(api_key)


def chat_tool_names(agent, *, online: bool) -> list[str] | None:
    """按聊天模式返回可见工具名清单（传给 ``agent.run(tool_names=...)``）。

    - online 且 AnySearch 已配置：返回 ``None``（全部工具，含 web.search）。
    - 其余情况（offline 或联网未配置）：摘除 web.search，只留本地工具。
    """
    if online and search_available():
        return None
    return [name for name in agent.tools.snapshot() if name != SEARCH_TOOL_NAME]


def record_qa(
    manager: MemoryManager,
    question: str,
    answer: str,
    *,
    mode: str,
) -> MemoryItem:
    """把一次问答写入 episodic 记忆，带时间戳、可被 memory.manage/search 检索。

    写入内容以「问 / 答」为主，便于将来用「我这两天问过什么」这类问题检索；
    metadata 保留结构化字段，星图时间线上以「问：…」事件出现。
    """
    now = utc_now()
    return manager.episodic.record(
        f"问：{question}\n答：{answer}",
        metadata={
            "kind": "qa",
            "type": "qa",
            "title": f"问：{question[:NEBULA_EVENT_TITLE_CHARS]}",
            "question": question,
            "answer": answer,
            "mode": mode,
            "asked_at": now.isoformat(),
        },
        timestamp=now,
    )


def chat_ready() -> tuple[bool, str]:
    """检测是否已配置真实聊天模型。

    ProviderRegistry 在 provider.toml 缺失时会回退到 provider.example.toml
    （占位 key），因此这里必须显式区分：只有真实的 provider.toml 且 key
    非 placeholder 时才放行。
    """
    from agents.providers import ProviderRegistry

    path = ProviderRegistry.default_config_path()
    if path.name != "provider.toml" or not path.is_file():
        return False, (
            "未配置聊天模型：请复制 config/provider.example.toml 为 "
            "config/provider.toml，填入 api_key（或用 api_key_env 指向环境"
            "变量），然后重启服务。"
        )
    try:
        registry = ProviderRegistry()
        profile = registry.get(registry.active_profile)
        key = profile.api_key
        if not key and profile.api_key_env:
            key = os.getenv(profile.api_key_env, "")
        if not key or str(key).startswith("replace-with"):
            return False, "provider.toml 已存在但 api_key 为空/占位符，请填入真实 key。"
    except Exception as exc:  # 配置解析失败也要给用户可读的信息
        return False, f"provider.toml 解析失败：{type(exc).__name__}: {exc}"
    return True, ""
