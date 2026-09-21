"""混合召回工具：向量路 × FTS5 关键词路，RRF 融合。

为什么需要两路
==============

精确词（型号、编号、代码标识符）在向量空间里区分度很差，转述又只有向量能召回，
两路互补。任一投影不可用都不能让检索整体失败：向量路异常时本工具降级为纯关键词
（FTS5），并把降级原因写进返回值的 ``note``，让 UI 和健康检查能如实展示。

融合用 Reciprocal Rank Fusion（只按名次计分），避免余弦相似度与 bm25 两套量纲
直接相加。

本模块是这条召回路径的**唯一实现**：``memory.rag.pipeline.RAGPipeline`` 不再
持有 ``hybrid_retrieve`` / ``_vector_hits`` / ``_rrf_fuse`` / ``last_retrieval_note``，
调用方（``web/app.py`` 的聊天溯源面板）改为调用 :func:`hybrid_recall`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from constants import MEMORY_HYBRID, RAG_RETRIEVE_LIMIT
from core import BaseTool, ToolSpec
from memory import MemoryType
from memory.rag import RetrievedChunk
from memory.storage.document_repo import ChunkRecord

TOOL_ENABLED = True

#: RRF 名次平滑常数（与信息检索常用取值一致）。
RRF_K = 60


def _default_pipeline() -> Any:
    """Build the default pipeline (imported lazily: ``_memory`` pulls ``memory.rag``)."""

    from ._memory import build_default_pipeline

    return build_default_pipeline()


@dataclass(frozen=True)
class HybridRecallResult:
    """Fused hits plus the reason the vector path was skipped, if it was."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    note: str = ""
    vector_available: bool = True


def rrf_fuse(rank_lists: list[list[str]], *, k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: 只按名次计分，避免余弦与 bm25 两套量纲混算。"""

    scores: dict[str, float] = {}
    for hits in rank_lists:
        for rank, chunk_id in enumerate(hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def chunk_metadata(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


def hybrid_enabled() -> bool:
    """混合检索开关（constants.MEMORY_HYBRID，默认开；关闭即回到纯向量）。"""

    return MEMORY_HYBRID


def vector_hits(
    pipeline: Any,
    query: str,
    *,
    limit: int,
    threshold: float | None,
    metadata: Mapping[str, Any] | None,
) -> tuple[list[tuple[str, float]], str]:
    """向量路 ``(chunk_id, 相似度)``；不可用时返回空表并给出降级原因。

    不再做请求前的 TCP 可达性探测（历史隧道网关的产物）：云端端点一次
    urlopen 的代价与探测相同，失败时异常本身就是降级信号。
    """

    try:
        results = pipeline.manager.search(
            query,
            memory_type=MemoryType.SEMANTIC,
            limit=limit,
            threshold=threshold,
            metadata=metadata,
        )
    except (ConnectionError, OSError, RuntimeError) as exc:
        note = (
            "向量检索失败，本次检索降级为纯关键词（FTS5）："
            f"{type(exc).__name__}: {exc}"
        )
        return [], note
    return [(result.item.id, float(result.score)) for result in results], ""


def fuse_hits(
    repository: Any,
    rank_lists: list[list[str]],
    scores: tuple[dict[str, float], dict[str, float]],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """RRF 融合并按真值源回查正文；真值源没有的分块不返回（孤立向量不外泄）。"""

    vector_scores, keyword_scores = scores
    results: list[RetrievedChunk] = []
    for chunk_id, score in rrf_fuse(rank_lists)[:limit]:
        chunk = repository.get_chunk(chunk_id)
        if chunk is None:
            continue
        results.append(
            RetrievedChunk(
                chunk.text,
                score,
                chunk.chunk_id,
                chunk_metadata(chunk),
                detail={
                    # 与对外 score 同源同值（不在这里四舍五入，展示精度交给前端）
                    "rrf_score": float(score),
                    "vector_score": vector_scores.get(chunk_id),
                    "keyword_score": keyword_scores.get(chunk_id),
                },
            )
        )
    return results


def hybrid_recall(
    pipeline: Any,
    query: str,
    *,
    limit: int = RAG_RETRIEVE_LIMIT,
    threshold: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HybridRecallResult:
    """Run the vector and keyword paths, then fuse them with RRF."""

    repository = pipeline.document_repo()
    if not hybrid_enabled() or repository is None:
        return HybridRecallResult(
            chunks=pipeline.retrieve(
                query, limit=limit, threshold=threshold, metadata=metadata
            ),
            note="",
            vector_available=True,
        )
    hits, note = vector_hits(
        pipeline, query, limit=limit * 2, threshold=threshold, metadata=metadata
    )
    keyword_hits = [
        (chunk_id, float(score))
        for chunk_id, score in repository.search_keywords(query, limit=limit * 2)
    ]
    # U4：两路的原始分数在融合前留一份，否则 RRF 只留下名次、贡献不可见。
    chunks = fuse_hits(
        repository,
        [[chunk_id for chunk_id, _ in hits], [chunk_id for chunk_id, _ in keyword_hits]],
        (dict(hits), dict(keyword_hits)),
        limit=limit,
    )
    return HybridRecallResult(chunks=chunks, note=note, vector_available=not note)


class HybridRecallInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="检索问句；精确词（型号/编号）与转述句都能召回。",
    )
    limit: int = Field(default=5, ge=1, le=50, description="返回条数上限。")


class HybridHit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    document_id: str | None = None
    chunk_index: int | None = None
    snippet: str
    score: float = Field(description="RRF 融合分，同时是排序依据。")
    rrf_score: float
    vector_score: float | None = Field(description="向量路原始相似度；降级时为 null。")
    keyword_score: float | None = Field(description="FTS5 bm25 原始分；无命中时为 null。")


class HybridRecallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str
    count: int
    note: str = Field(description="向量路降级说明；空串表示两路都正常。")
    vector_available: bool
    hits: list[HybridHit] = Field(default_factory=list)


class HybridRecallTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.hybrid_recall",
        description=(
            "Retrieve stored chunks by combining vector similarity with FTS5 "
            "keyword matching (reciprocal rank fusion). Use it for exact "
            "identifiers and paraphrased questions alike. Read-only: it never "
            "writes, and it degrades to keyword-only instead of failing when the "
            "embedding endpoint is down."
        ),
        version="1.0.0",
        input_model=HybridRecallInput,
        output_model=HybridRecallOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=30.0,
        idempotent=True,
        parallel_safe=True,
        tags=("memory", "recall", "hybrid", "rrf", "fts5", "read"),
    )

    def __init__(self, pipeline: Any = None) -> None:
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = _default_pipeline()
        return self._pipeline

    def execute(self, arguments: HybridRecallInput) -> HybridRecallOutput:
        result = hybrid_recall(self.pipeline, arguments.query, limit=arguments.limit)
        return HybridRecallOutput(
            query=arguments.query,
            count=len(result.chunks),
            note=result.note,
            vector_available=result.vector_available,
            hits=[
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
                for chunk in result.chunks
            ],
        )


def create_tool() -> BaseTool:
    return HybridRecallTool()


__all__ = [
    "HybridHit",
    "HybridRecallInput",
    "HybridRecallOutput",
    "HybridRecallResult",
    "HybridRecallTool",
    "chunk_metadata",
    "create_tool",
    "fuse_hits",
    "hybrid_enabled",
    "hybrid_recall",
    "rrf_fuse",
    "vector_hits",
]
