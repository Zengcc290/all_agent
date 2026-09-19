"""Web 层支撑设施：共享单例与路径约定。

设计要点：
- 嵌入选型只认 ``config/services.toml`` 的 ``[embedding]`` 段（端点+密钥配齐
  走云端 ``memory.APIEmbedding``，否则 ``memory.HashEmbedding`` 离线兜底），
  规则在 ``memory.base.make_default_embedding``，Web 与 Agent 工具同一份。
- ``DB_PATH`` 在导入时固定为 ``memory.base.default_sqlite_path()`` 的返回值
  （``MEMORY_DB_PATH`` 是唯一保留的路径覆盖入口），保证 Agent 工具
  （memory.query / memory.add / memory.rag）与 Web API 共享同一个记忆库
  ——「记忆共享」的数据面。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from constants import (
    NEBULA_EVENT_TITLE_CHARS,
    QA_EXTRACT_CHUNK_SIZE,
    WEB_QA_ANSWER_MAX_CHARS,
    WEB_QA_EXTRACT,
    WEB_QA_EXTRACT_SYNC,
    WEB_QA_QUESTION_MAX_CHARS,
)
from core.services_config import SearchService, load_services_config
from memory import (
    MemoryConfig,
    MemoryItem,
    MemoryManager,
    default_sqlite_path,
    make_default_embedding,
    utc_now,
)
from memory.rag import LLMKnowledgeExtractor, NullKnowledgeExtractor, RAGPipeline

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
SEED_FILE = WEB_DIR / "seed_data.json"

#: 统一记忆库路径：Web API 与 Agent 工具都读 memory.base.default_sqlite_path()
#: （单一事实来源；``MEMORY_DB_PATH`` 是唯一保留的路径覆盖入口）。
DB_PATH = Path(default_sqlite_path())


def build_embedding(config: MemoryConfig | None = None):
    """按 ``memory.base.make_default_embedding`` 的优先级选嵌入实现。

    这里**只做转发**，不重复一份选型逻辑：配置（端点/密钥/模型）统一来自
    config/services.toml 的 ``[embedding]`` 段，选型规则见
    ``memory.base.make_default_embedding``。
    """
    return make_default_embedding(config)


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
            LOGGER.warning("knowledge extractor disabled: provider api_key missing")
            return NullKnowledgeExtractor()
        client = LLM(
            api_key=api_key,
            base_url=profile.base_url,
            model=profile.default_model,
        )
        # 模型选择统一在 config/services.toml 的 [vision] 段（.env 不再承载模型名）。
        vision_model = load_services_config().vision.model or ""
        return LLMKnowledgeExtractor(
            client.complete,
            model=profile.default_model,
            vision_model=vision_model or profile.default_model,
        )
    except Exception:
        LOGGER.exception("knowledge extractor unavailable; using NullKnowledgeExtractor")
        return NullKnowledgeExtractor()


_manager: MemoryManager | None = None
_manager_lock = Lock()
_pipeline: RAGPipeline | None = None

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
                # 读 config/services.toml 全套（Qdrant/Neo4j 开关在这里生效）；
                # 未配置时与旧行为完全一致（内存向量 + 内存图）。
                config = MemoryConfig.from_config()
                config.sqlite_path = str(DB_PATH)
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
        manager = get_manager()
        with _manager_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline(
                    manager,
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
                from tool.search import SearchTool

                # 前端知识管家只挂检索/写入记忆的工具，不自动发现
                # fs / update_log / current_time 等项目脚手架。
                agent = ReActAgent("knowledge-butler", auto_discover_tools=False)
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
                agent.register_tool(SearchTool(), replace=True)
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

#: 前端知识管家允许暴露给 LLM 的工具。图检索 / 向量检索 / 增量记忆写入
#: 是星图聊天真正用到的能力；项目脚手架工具不进这一层。
CHAT_TOOL_ALLOWLIST = (
    "memory.query",
    "memory.add",
    "memory.rag_search",
    "memory.rag",
    SEARCH_TOOL_NAME,
)


def search_available() -> bool:
    """AnySearch 是否已配置（base_url 与 api_key 同时存在才视为可用）。

    唯一来源是 config/services.toml 的 ``[search]`` 段——外部 API 调用的
    集中配置（历史 SEARCH_* / ANYSEARCH_* 环境变量入口已删除）。
    """
    search = _services_search()
    return bool(search.base_url) and bool(search.api_key)


def _services_search() -> SearchService:
    """Search settings from config/services.toml; a blank/missing file yields all-None."""

    return load_services_config().search


def chat_tool_names(agent, *, online: bool) -> list[str]:
    """按聊天模式返回可见工具名清单（传给 ``agent.run(tool_names=...)``）。

    前端知识管家永远只暴露记忆/图/向量检索相关工具；``None`` 不再表示
    「发现到的全部工具」。联网且 AnySearch 已配置时额外开放 web.search。
    """
    names = [
        name
        for name in agent.tools.snapshot()
        if name in CHAT_TOOL_ALLOWLIST and name != SEARCH_TOOL_NAME
    ]
    if online and search_available() and SEARCH_TOOL_NAME in agent.tools.snapshot():
        names.append(SEARCH_TOOL_NAME)
    return names


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
    question = str(question or "")[:WEB_QA_QUESTION_MAX_CHARS]
    answer = str(answer or "")[:WEB_QA_ANSWER_MAX_CHARS]
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
    """问答是否触发图抽取。无真实聊天模型时自动关闭（没有抽取器可用）。

    开关在 constants.WEB_QA_EXTRACT（测试 monkeypatch 该常量）。
    """
    return WEB_QA_EXTRACT


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
    - ``constants.WEB_QA_EXTRACT_SYNC`` 为 True 时改为内联执行；测试与脚本
      用它拿到确定顺序（monkeypatch 该常量）。
    - 无事件循环时也只内联执行，避免创建永远不跑的协程。
    """

    if not knowledge_extract_enabled():
        return
    ready, _ = chat_ready()
    if not ready:
        return
    if WEB_QA_EXTRACT_SYNC:
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
    from agents.providers import ProviderRegistry, load_project_dotenv

    load_project_dotenv()
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
