"""多路混合召回工具：先把问句分解成多路，再分别召回并融合。

与 ``knowledge.hybrid_recall`` 的区别
=====================================

``hybrid_recall`` 只跑**一条**查询的两路（向量 × FTS5）。
本工具跑**多条**查询：用问句分解器把一句话拆成若干更短的探针查询（原句永远第一
条），每条各跑「向量 + 关键词」两路，把所有名次列表一起丢给 RRF 融合；图路同理，
每条子查询各做一次 seed+expand，路径按 ``effective`` 权重合并去重。

为什么多路更好：精确词（型号、编号）向量区分度差，转述又只有向量能召回；关系类
问句（「小红的亲戚是谁」）单靠原句往往一条都命中不了，拆出实体名和关系词后才召回
得到。任一子查询失败都不能让整体失败——图后端抖动时只丢路径、保留向量证据。

本模块是这条路径的**唯一实现**：``RAGPipeline.hybrid_retrieve_multi``、
``GraphRAGPipeline.retrieve_multi`` 与 ``memory/rag/knowledge.py`` 的
``QueryDecomposer``/``NullQueryDecomposer``/``LLMQueryDecomposer`` 已删除并收敛到这里。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from constants import (
    RAG_GRAPH_HOPS,
    RAG_GRAPH_PATH_LIMIT,
    RAG_RETRIEVE_LIMIT,
)
from core import BaseTool, ToolSpec
from memory import MemorySearchResult
from memory.rag import GraphPath, GraphRAGResult, RetrievedChunk
from memory.rag.knowledge import parse_json_object, response_content

from .hybrid_recall import (
    HybridHit,
    HybridRecallResult,
    fuse_hits,
    hybrid_enabled,
    vector_hits,
)

TOOL_ENABLED = True

LOGGER = logging.getLogger(__name__)

#: 分解器最多产出多少条子查询（含原句）。
MAX_SUB_QUERIES = 6


def _default_pipeline() -> Any:
    """Build the default pipeline (imported lazily: ``_memory`` pulls ``memory.rag``)."""

    from ._memory import build_default_pipeline

    return build_default_pipeline()


def _clean_query_text(value: Any, *, max_length: int = 500) -> str:
    """Collapse whitespace and truncate, mirroring ``memory.ids._clean_text``."""

    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_length].strip()


class QueryDecomposer(Protocol):
    """把一句话拆成多种待测查询（协议，与 KnowledgeExtractor 同构）。"""

    def decompose(self, query: str) -> list[str]: ...


class NullQueryDecomposer:
    """LLM 不可用时的安全降级：永不因分解失败而答不出来，返回原句。"""

    def decompose(self, query: str) -> list[str]:
        return [query.strip()] if isinstance(query, str) and query.strip() else []


class LLMQueryDecomposer:
    """Decompose one question into multiple probe queries through the chat client."""

    SYSTEM_PROMPT = (
        "你是检索问句分解器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。\n"
        "任务：把用户的问句拆成若干更短的待测查询，交给检索工具分别执行后融合。\n"
        "\n"
        "【要求】\n"
        "1. sub_queries 里每项是一条更短的查询；第一条必须是原句本身。\n"
        "2. 最多 6 条；去重；不要添加原文没有的实体或数字。\n"
        "3. 针对关系词、别称、上位词各给一条变体（如「小红的亲戚是谁」→「小红」、"
        "「小红 亲戚」、「小红 亲属 关系」）。\n"
        "\n"
        '字段格式：{"sub_queries": string[]}。'
    )

    MAX_SUB_QUERIES = MAX_SUB_QUERIES

    def __init__(
        self,
        complete: Callable[..., Any],
        *,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        self.complete = complete
        self.model = model
        self.timeout = timeout

    def decompose(self, query: str) -> list[str]:
        if not isinstance(query, str) or not query.strip():
            return []
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"query": query}, ensure_ascii=False)},
        ]
        try:
            response = self.complete(
                messages,
                model=self.model,
                temperature=0.0,
                timeout=self.timeout,
                stream=False,
            )
            raw = response_content(response)
            value = parse_json_object(raw)
        except Exception:  # noqa: BLE001 - 分解失败也不能丢原句（永不因分解失败而答不出来）
            return [query]
        return self._normalize(query, value)

    def _normalize(self, query: str, value: dict[str, Any]) -> list[str]:
        """原句永远第一条；去重、截断——分解失败也不能丢原句。"""

        items = value.get("sub_queries") if isinstance(value, dict) else None
        if not isinstance(items, list):
            return [query]
        cleaned = [
            _clean_query_text(item)
            for item in items
            if isinstance(item, str) and _clean_query_text(item)
        ]
        return [query, *dict.fromkeys(cleaned)][: self.MAX_SUB_QUERIES]


def hybrid_recall_multi(
    pipeline: Any,
    queries: list[str],
    *,
    limit: int = RAG_RETRIEVE_LIMIT,
    threshold: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HybridRecallResult:
    """F3：每条子查询各跑向量+关键词路，N 路 rank 列表一起丢给 RRF 融合。"""

    queries = [query for query in queries if isinstance(query, str) and query.strip()]
    if not queries:
        return HybridRecallResult()
    if len(queries) == 1:
        from .hybrid_recall import hybrid_recall

        return hybrid_recall(
            pipeline, queries[0], limit=limit, threshold=threshold, metadata=metadata
        )
    repository = pipeline.document_repo()
    if not hybrid_enabled() or repository is None:
        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for query in queries:
            for chunk in pipeline.retrieve(
                query, limit=limit, threshold=threshold, metadata=metadata
            ):
                if chunk.memory_id in seen:
                    continue
                seen.add(chunk.memory_id)
                chunks.append(chunk)
        return HybridRecallResult(chunks=chunks[:limit])
    rank_lists: list[list[str]] = []
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}
    note = ""
    for query in queries:
        hits, current_note = vector_hits(
            pipeline, query, limit=limit * 2, threshold=threshold, metadata=metadata
        )
        note = note or current_note
        keyword_hits = [
            (chunk_id, float(score))
            for chunk_id, score in repository.search_keywords(query, limit=limit * 2)
        ]
        rank_lists.append([chunk_id for chunk_id, _ in hits])
        rank_lists.append([chunk_id for chunk_id, _ in keyword_hits])
        vector_scores.update(hits)
        keyword_scores.update(keyword_hits)
    chunks = fuse_hits(
        repository, rank_lists, (vector_scores, keyword_scores), limit=limit
    )
    return HybridRecallResult(chunks=chunks, note=note, vector_available=not note)


def graph_recall_multi(
    pipeline: Any,
    queries: list[str],
    *,
    limit: int = RAG_RETRIEVE_LIMIT,
    hops: int = RAG_GRAPH_HOPS,
    threshold: float | None = None,
    path_limit: int = RAG_GRAPH_PATH_LIMIT,
) -> GraphRAGResult:
    """F3：每条子查询各做 seed+expand，路径按 effective 合并去重。"""

    queries = [query for query in queries if isinstance(query, str) and query.strip()]
    if not queries:
        raise ValueError("queries must be a non-empty list of strings")
    if len(queries) == 1:
        return pipeline.graph.retrieve(
            queries[0], limit=limit, hops=hops, threshold=threshold, path_limit=path_limit
        )
    evidence: dict[str, MemorySearchResult] = {}
    paths: dict[tuple[tuple[str, ...], tuple[str, ...]], GraphPath] = {}
    seeds: list[str] = []
    for query in queries:
        result = pipeline.graph.retrieve(
            query, limit=limit, hops=hops, threshold=threshold, path_limit=path_limit
        )
        for item in result.evidence:
            evidence.setdefault(item.item.id, item)
        seeds.extend(result.entities)
        for path in result.paths:
            key = (path.entities, path.relations)
            existing = paths.get(key)
            if existing is None or path.effective > existing.effective:
                paths[key] = path
    merged = sorted(
        paths.values(), key=lambda path: (-path.effective, len(path.relations), path.target)
    )[:path_limit]
    return GraphRAGResult(
        query=" | ".join(queries),
        evidence=list(evidence.values())[:limit],
        paths=merged,
        entities=list(dict.fromkeys(seeds)),
    )


def build_decomposer() -> QueryDecomposer:
    """Use the configured chat model when available, otherwise the null fallback."""

    try:
        from agents.llm import LLM
        from agents.providers import ProviderRegistry

        registry = ProviderRegistry()
        profile = registry.get(registry.active_profile)
        key = registry.resolve_api_key(profile.name)
        if key and not key.startswith("replace-with"):
            client = LLM(
                api_key=key, base_url=profile.base_url, model=profile.default_model
            )
            return LLMQueryDecomposer(client.complete, model=profile.default_model)
    except Exception:
        LOGGER.warning(
            "问句分解器不可用，多路召回退化为原句单路",
            exc_info=True,
        )
    return NullQueryDecomposer()


class MultiRecallInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=2000, description="原始检索问句。")
    mode: Literal["hybrid", "graph", "both"] = Field(
        default="both",
        description="hybrid=向量×关键词多路融合；graph=关系路径多路；both=两者都跑。",
    )
    decompose: bool = Field(
        default=True,
        description="是否先用 LLM 把问句拆成多条探针查询（关闭则只跑原句）。",
    )
    limit: int = Field(default=8, ge=1, le=50, description="每路返回条数上限。")
    hops: int = Field(default=2, ge=0, le=3, description="图路最大跳数。")


class MultiRecallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str
    sub_queries: list[str] = Field(description="实际执行的子查询；原句永远第一条。")
    note: str = Field(description="向量路降级说明；空串表示正常。")
    hits: list[HybridHit] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    paths: list[dict[str, Any]] = Field(default_factory=list)
    context: str = Field(default="", description="可直接喂给模型的关系路径上下文。")


class MultiRecallTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.multi_recall",
        description=(
            "Recall across MULTIPLE decomposed sub-queries at once: each probe "
            "runs the vector+FTS5 hybrid path and/or the graph path, then all "
            "ranked lists are fused (RRF for chunks, effective weight for graph "
            "paths). Use it for relational questions where one phrasing is not "
            "enough. Read-only."
        ),
        version="1.0.0",
        input_model=MultiRecallInput,
        output_model=MultiRecallOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=90.0,
        idempotent=True,
        parallel_safe=True,
        tags=("memory", "recall", "multi", "graph", "rrf", "read"),
    )

    def __init__(self, pipeline: Any = None, decomposer: QueryDecomposer | None = None) -> None:
        self._pipeline = pipeline
        self._decomposer = decomposer

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = _default_pipeline()
        return self._pipeline

    @property
    def decomposer(self) -> QueryDecomposer:
        if self._decomposer is None:
            self._decomposer = build_decomposer()
        return self._decomposer

    def execute(self, arguments: MultiRecallInput) -> MultiRecallOutput:
        queries = (
            self.decomposer.decompose(arguments.query)
            if arguments.decompose
            else NullQueryDecomposer().decompose(arguments.query)
        )
        if not queries:
            queries = [arguments.query]
        pipeline = self.pipeline
        hits: list[HybridHit] = []
        note = ""
        entities: list[str] = []
        paths: list[dict[str, Any]] = []
        context = ""
        if arguments.mode in {"hybrid", "both"}:
            hybrid = hybrid_recall_multi(
                pipeline, queries, limit=arguments.limit
            )
            note = hybrid.note
            hits = [
                HybridHit(
                    chunk_id=chunk.memory_id,
                    document_id=chunk.metadata.get("document_id"),
                    chunk_index=chunk.metadata.get("chunk_index"),
                    snippet=(chunk.content or "")[:200],
                    score=float(chunk.score),
                    rrf_score=float(chunk.detail.get("rrf_score") or chunk.score),
                    vector_score=chunk.detail.get("vector_score"),
                    keyword_score=chunk.detail.get("keyword_score"),
                )
                for chunk in hybrid.chunks
            ]
        if arguments.mode in {"graph", "both"}:
            graph = graph_recall_multi(
                pipeline, queries, limit=arguments.limit, hops=arguments.hops
            )
            entities = list(graph.entities)
            paths = [path.to_dict() for path in graph.paths]
            context = graph.build_context()
        return MultiRecallOutput(
            query=arguments.query,
            sub_queries=queries,
            note=note,
            hits=hits,
            entities=entities,
            paths=paths,
            context=context,
        )


def create_tool() -> BaseTool:
    return MultiRecallTool()


__all__ = [
    "MAX_SUB_QUERIES",
    "LLMQueryDecomposer",
    "MultiRecallInput",
    "MultiRecallOutput",
    "MultiRecallTool",
    "NullQueryDecomposer",
    "QueryDecomposer",
    "build_decomposer",
    "create_tool",
    "graph_recall_multi",
    "hybrid_recall_multi",
]