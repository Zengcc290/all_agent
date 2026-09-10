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

from memory import APIEmbedding, MemoryConfig, MemoryManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
SEED_FILE = WEB_DIR / "seed_data.json"

# .env 中的 key 优先级高于环境变量（本地开发惯例）
load_dotenv(PROJECT_ROOT / ".env")

#: 统一记忆库路径：Web API 与 Agent 工具都读它。
DB_PATH = Path(os.getenv("MEMORY_DB_PATH") or (PROJECT_ROOT / "memory.sqlite3"))
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
        return APIEmbedding(api_key=api_key, model="qwen3-embedding-0.6b", dimension=1024)
    return HashEmbedding()


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
    "1. 回答与用户知识、经历、文档相关的问题前，先用 memory.rag 的 retrieve/context"
    " 行动检索知识库，必要时用 memory.manage 的 search 补充记忆检索。\n"
    "2. 用中文简洁回答；引用知识库内容时注明来源文件（若有）。\n"
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

                agent = ReActAgent("knowledge-butler")
                agent.set_system_prompt(SYSTEM_PROMPT)
                _agent = agent
    return _agent


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
