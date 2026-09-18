"""Web 层支撑设施：嵌入降级、共享单例与路径约定。

设计要点：
- 嵌入实现按优先级：``EMBEDDING_BASE_URL`` 指向的本地转发网关（自定义
  ``/embed`` 协议）→ ``DASHSCOPE_API_KEY`` 公网 qwen3-embedding-0.6b →
  ``memory.HashEmbedding`` 离线兜底。没有任何 key 时整站仍可运行。
- ``MEMORY_DB_PATH`` 在导入时就被固定为项目根下的 ``memory.sqlite3``，
  保证 Agent 工具（memory.query / memory.add / memory.rag）与 Web API 共享
  同一个记忆库——「记忆共享」的数据面。
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess  # 仅用于拉起本机 SSH 端口转发
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv

from constants import (
    DEFAULT_MEMORY_DB_FILENAME,
    NEBULA_EVENT_TITLE_CHARS,
    QA_EXTRACT_CHUNK_SIZE,
)
from core.services_config import SearchService, load_services_config
from memory import (
    MemoryConfig,
    MemoryItem,
    MemoryManager,
    make_default_embedding,
    utc_now,
)
from memory.embedding import gateway_reachable
from memory.rag import LLMKnowledgeExtractor, NullKnowledgeExtractor, RAGPipeline

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
SEED_FILE = WEB_DIR / "seed_data.json"

#: 环境变量优先于 .env（``load_dotenv`` 默认 ``override=False``）。
load_dotenv(PROJECT_ROOT / ".env")

#: 统一记忆库路径：Web API 与 Agent 工具都读它。
DB_PATH = Path(os.getenv("MEMORY_DB_PATH") or (PROJECT_ROOT / DEFAULT_MEMORY_DB_FILENAME))
os.environ.setdefault("MEMORY_DB_PATH", str(DB_PATH))


def build_embedding(config: MemoryConfig | None = None):
    """按 ``memory.base.make_default_embedding`` 的优先级选嵌入实现。

    这里**只做转发**，不重复一份选型逻辑：曾经这里硬编码 ``DASHSCOPE_API_KEY``
    + qwen 模型名 + 本机网关地址，导致 ``HELLOAGENTS_MEMORY_EMBEDDING_*`` 配好的
    远端端点（含 Gemini）在 Web 侧完全不生效，而 Agent 工具侧却生效——同一个进程
    两套向量空间。选型规则见 ``memory.base.make_default_embedding``。
    """
    return make_default_embedding(config)


def ensure_embedding_tunnel() -> bool:
    """按需自动建立嵌入隧道（EMBEDDING_BASE_URL 指向 10800 时）。

    设计（D8 的配套便利，不改变向量空间语义）：
    - 只在「配置了本地隧道网关」且「10800 不可达」时才拉起 SSH；
    - 隧道参数来自 .env：``EMBEDDING_TUNNEL_KEY`` 指定私钥路径，
      ``EMBEDDING_TUNNEL_HINT`` 里的服务器/端口作为兜底来源；
    - 幂等：进程内只启动一次；已监听 10800 时直接返回 True；
    - 不阻塞主流程：失败只记日志，嵌入仍按 D8 降级，绝不换向量空间。
    """
    global _tunnel_started
    server_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip()
    if not server_url:
        return True  # 没配隧道网关（DashScope/Hash），无需隧道
    if gateway_reachable(server_url):
        return True  # 隧道已通（或网关已在跑）
    with _tunnel_lock:
        if _tunnel_started:
            return False  # 本进程已试过一次，避免重复拉起
        _tunnel_started = True
    key_path = (os.getenv("EMBEDDING_TUNNEL_KEY") or "").strip()
    hint = (os.getenv("EMBEDDING_TUNNEL_HINT") or "").strip()
    if not key_path:
        LOGGER.warning("嵌入网关不可达且未配置 EMBEDDING_TUNNEL_KEY，跳过自动隧道：%s", hint)
        return False
    if not Path(key_path).is_file():
        LOGGER.warning("EMBEDDING_TUNNEL_KEY 指向的私钥不存在：%s", key_path)
        return False
    # 从 hint 解析转发参数（root@host -p port），解析不出就用默认值。
    import re as _re

    match = _re.search(r"(\S+)@(\S+)\s+-p\s+(\d+)", hint)
    if not match:
        LOGGER.warning("EMBEDDING_TUNNEL_HINT 无法解析出 user@host -p port：%s", hint)
        return False
    user, host, port = match.group(1), match.group(2), match.group(3)
    # 本地端口取自 EMBEDDING_BASE_URL 的端口；远端固定 18000（隧道另一端网关）。
    from urllib.parse import urlparse

    local_port = urlparse(server_url).port or 10800
    remote_port = os.getenv("EMBEDDING_TUNNEL_REMOTE_PORT", "18000").strip() or "18000"
    cmd = [
        "ssh", "-N",
        "-L", f"{local_port}:127.0.0.1:{remote_port}",
        f"{user}@{host}", "-p", port,
        "-i", key_path,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ExitOnForwardFailure=yes",
    ]
    try:
        subprocess.Popen(  # 参数全部来自本机 .env，非外部输入
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        LOGGER.warning("自动建立嵌入隧道失败：%s", exc)
        return False
    # 最多等 15 秒让隧道就绪（网络秒连时通常 1-2 秒）。
    import time

    for _ in range(15):
        time.sleep(1)
        if gateway_reachable(server_url):
            LOGGER.info("嵌入隧道已自动建立：%s -> %s@%s:%s", local_port, user, host, remote_port)
            return True
    LOGGER.warning("自动隧道已拉起但 %s 秒内未就绪，嵌入按 D8 降级", 15)
    return False


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
        vision_model = (
            os.getenv("KNOWLEDGE_VISION_MODEL")
            or os.getenv("VISION_MODEL")
            or ""
        ).strip()
        return LLMKnowledgeExtractor(
            client.complete,
            model=profile.default_model,
            vision_model=vision_model or profile.default_model,
        )
    except Exception:  # noqa: BLE001 - 任何配置问题都退回无抽取器的可用状态
        return NullKnowledgeExtractor()


_manager: MemoryManager | None = None
_manager_lock = Lock()
_pipeline: RAGPipeline | None = None

#: 嵌入隧道自动拉起的一次性标记（进程内幂等）。
_tunnel_started = False
_tunnel_lock = Lock()

#: 每次问答/文档抽取实际写入图事实时递增。图缓存据此失效，避免为后台
#: 抽取线程加锁，也避免抽取失败时白白重建星图。
GRAPH_REVISION = 0
_graph_revision_lock = Lock()


def bump_graph_revision() -> None:
    global GRAPH_REVISION
    with _graph_revision_lock:
        GRAPH_REVISION += 1


def graph_revision() -> int:
    with _graph_revision_lock:
        return GRAPH_REVISION


def get_manager() -> MemoryManager:
    """进程级单例 MemoryManager：Web API、RAG、种子脚本共用。"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                # 读 HELLOAGENTS_MEMORY_* 全套（Qdrant/Neo4j 开关在这里生效）；
                # 未配置时与旧行为完全一致（内存向量 + 内存图）。
                config = MemoryConfig.from_env()
                config.sqlite_path = str(DB_PATH)  # MEMORY_DB_PATH 优先级不变
                _manager = MemoryManager(config, embedding=build_embedding(config))
    return _manager


def close_manager() -> None:
    global _manager
    if _manager is not None:
        _manager.close()
        _manager = None


_agent = None
_agent_lock = Lock()

#: 知识管家的行为约束：先检索记忆再回答。只读工具（memory.rag_search /
#: memory.query）不触发写确认，因此这条链路无需任何用户额外授权。
SYSTEM_PROMPT = (
    "你是『星图』——用户的个人知识管家，管理着用户的知识库与记忆。遵守：\n"
    "1. 回答与用户知识、经历、文档相关的问题前，先用 memory.rag_search 的 graph_retrieve/context"
    " 行动检索知识库；再用 memory.query 的 search 补充记忆检索——search 不指定 "
    "memory_type 会跨全部四层搜索，用户的提问历史与经历都存在 episodic，"
    "回答「我这两天问过什么 / 计划是什么」这类问题时必须搜这里。\n"
    "2. 用中文简洁回答；引用知识库内容时注明来源文件和关系证据（若有）。\n"
    "3. 不编造知识库里没有的内容；检索不到就如实说明。\n"
    "4. 用户明确让你记住某件事时，用 memory.add 写入 episodic 记忆。\n"
    "5. 关系有更新时只采信当前有效值（supersede 后的新值）。检索到旧值或被标记为"
    "历史的记录，要说明它已被更新，不要把新旧值并列当作同时成立。"
)


def get_pipeline() -> RAGPipeline:
    """进程级 RAG 管道：Web API、知识管家工具、问答抽取共用同一份记忆。"""
    global _pipeline
    if _pipeline is None:
        with _manager_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline(
                    get_manager(),
                    extractor=build_knowledge_extractor(),
                )
    return _pipeline


def get_agent():
    """懒加载 ReActAgent 单例（真实聊天模型未配置时构造也能成功，
    调用前由 ``chat_ready`` 把关）。"""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                from agents import ReActAgent
                from tool.memory_add import MemoryAddTool
                from tool.memory_query import MemoryQueryTool
                from tool.rag_search import RAGSearchTool
                from tool.rag_tool import RAGTool

                agent = ReActAgent("knowledge-butler")
                agent.set_system_prompt(SYSTEM_PROMPT)
                # 四个记忆工具统一注入 Web 单例后端，避免发现机制各自创建的
                # 默认连接与嵌入配置和 Web API 漂移（同一份记忆库是硬要求）。
                agent.register_tool(
                    MemoryQueryTool(manager=get_manager()), replace=True
                )
                agent.register_tool(MemoryAddTool(manager=get_manager()), replace=True)
                agent.register_tool(
                    RAGSearchTool(pipeline=get_pipeline()), replace=True
                )
                agent.register_tool(RAGTool(pipeline=get_pipeline()), replace=True)
                _agent = agent
    return _agent


#: 聊天回合只为「记住这件事」这一个增量写入提供确认。delete/clear/ingest
#: 仍需人工确认，所以提示词注入最多让模型多记一条，不能删库或改库。
CHAT_CONFIRMED_TOOLS = ("memory.add",)


def chat_confirmed_side_effects(agent) -> frozenset[str]:
    """Return confirmation keys for the writes one user chat turn may perform.

    Fails closed: an unregistered or unknown tool simply contributes no key, so
    the runtime keeps asking for confirmation instead of silently allowing it.
    """

    keys: set[str] = set()
    for name in CHAT_CONFIRMED_TOOLS:
        for lookup in (
            getattr(agent, "tool_confirmation_key", None),
            getattr(getattr(agent, "tools", None), "confirmation_key", None),
        ):
            if not callable(lookup):
                continue
            try:
                keys.add(lookup(name))
            except (KeyError, ValueError, TypeError):
                continue
            else:
                break
        else:
            LOGGER.warning("chat confirmation key unavailable for tool %s", name)
    return frozenset(keys)


#: 联网搜索工具的注册名；聊天「联网/非联网」开关据此决定是否可见。
SEARCH_TOOL_NAME = "web.search"


def search_available() -> bool:
    """AnySearch 是否已配置（base_url 与 api_key 同时存在才视为可用）。

    环境变量优先（SEARCH_* / ANYSEARCH_*），其次看 config/services.toml 的
    [search] 段——那是外部 API 调用的集中配置。
    """
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
    if base_url and api_key:
        return True
    search = _services_search()
    return bool(search.base_url) and bool(search.api_key)


def _services_search() -> SearchService:
    """Search settings from config/services.toml; a blank/missing file yields all-None."""

    return load_services_config().search


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
    """把一次问答写入 episodic 记忆，带时间戳、可被 memory.query/search 检索。

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


def knowledge_extract_enabled() -> bool:
    """问答是否触发图抽取。无真实聊天模型时自动关闭（没有抽取器可用）。"""
    return os.getenv("WEB_QA_EXTRACT", "1").strip().casefold() not in {
        "0",
        "false",
        "no",
        "off",
    }


def extract_graph_patches(
    question: str,
    answer: str,
    *,
    manager: MemoryManager | None = None,
) -> dict[str, Any] | None:
    """把一次问答交给 LLM 转成图补丁并落库。

    问答原文先以 episodic 留痕（调用方负责），这里只做图侧增量。抽取被刻意
    排除在聊天延迟之外：调用方在响应返回后再调用本函数，且限定只从用户陈述
    和助手依据知识库给出的事实里抽取，避免把模型自己的推测固化成边。

    调用方传入自己的 ``manager``。后台线程绝不能再调 ``get_manager()``：
    Web 应用关闭时全局单例会先被关闭，后台线程再抢同一把锁就会永久挂住。
    """

    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    if not question.strip() or not answer.strip():
        return None
    if manager is None:
        manager = get_manager()
    pipeline = RAGPipeline(manager, extractor=build_knowledge_extractor())
    from memory.rag import Document

    text = (
        "【用户陈述】\n"
        f"{question.strip()}\n"
        "【助手回答（只抽取其中依据知识库给出的事实，推测性表述不要抽取）】\n"
        f"{answer.strip()}"
    )
    try:
        pipeline.ingest(
            Document(
                text,
                metadata={
                    "source": "问答抽取",
                    "filename": "问答抽取",
                    "kind": "qa",
                },
            ),
            chunk_size=QA_EXTRACT_CHUNK_SIZE,
            overlap=0,
        )
    except Exception:  # 抽取失败不能影响问答本身
        LOGGER.exception("QA graph extraction failed")
        return None
    report = dict(pipeline.last_ingest_report)
    if report.get("entities") or report.get("relations"):
        bump_graph_revision()
    return report


async def extract_graph_patches_async(
    question: str,
    answer: str,
    *,
    manager: MemoryManager | None = None,
) -> None:
    """Background wrapper: blocking LLM + SQLite work runs off the event loop."""
    await asyncio.to_thread(
        extract_graph_patches, question, answer, manager=manager
    )


def schedule_qa_extraction(
    question: str,
    answer: str,
    *,
    manager: MemoryManager | None = None,
) -> None:
    """Fire-and-forget QA graph extraction.

    - 没有真实聊天模型时直接跳过：那时没有抽取器，抽取只会白跑一次。
    - ``WEB_QA_EXTRACT_SYNC=1`` 改为内联执行；测试与脚本用它拿到确定顺序。
    - 无事件循环时也只内联执行，避免创建永远不跑的协程。
    """

    if not knowledge_extract_enabled():
        return
    ready, _ = chat_ready()
    if not ready:
        return
    if os.getenv("WEB_QA_EXTRACT_SYNC", "").strip().casefold() in {"1", "true", "yes", "on"}:
        extract_graph_patches(question, answer, manager=manager)
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        extract_graph_patches(question, answer, manager=manager)
        return
    asyncio.create_task(
        extract_graph_patches_async(question, answer, manager=manager)
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
    except Exception as exc:  # noqa: BLE001 - 配置解析失败也要给用户可读的信息
        return False, f"provider.toml 解析失败：{type(exc).__name__}: {exc}"
    return True, ""
