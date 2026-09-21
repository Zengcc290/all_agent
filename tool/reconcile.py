"""三库对账工具：真值源 ↔ 向量投影 ↔ 图投影（只读，只看不改）。

为什么这是一个独立能力
======================

本项目有**三个**存储：SQLite ``documents``/``chunks``（真值源）、向量库（投影）、
图存储（投影）。投影可能落后于真值源（云端嵌入当时不可达、进程被 kill、Neo4j 抖动），
所以必须有一个"对账"动作能回答三个问题：

1. 各库现在有多少东西（计数）；
2. 真值源里已标 ``indexed`` 的分块，向量库里是否真的都有；
3. 有边（fact）却没有对应图关系的，是哪些。

它只读、不写、不改任何数据，因此可以被随时调用、被 Agent 调用、被 UI 轮询；
真正的修复在 ``knowledge.repair_drift``（写工具）。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 里原先内联的
``_is_fact_item`` / ``fact_items`` / ``projected_vector_ids`` / ``projected_edge_ids`` /
``reconcile_report`` 已删除，``/api/reconcile``、``/api/stats`` 改为调用这里。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory.base import MemoryItem
from memory.manager import MemoryManager
from memory.storage.document_repo import DocumentRepository

TOOL_ENABLED = True

#: 可报告的漂移类型；修复工具只接受其中可自愈的两类。
DRIFT_KINDS = ("missing_vector", "orphan_vector", "missing_edge")

#: 可自愈的漂移类型（只补投影，绝不删改真值源）。
REPAIRABLE_KINDS = ("missing_vector", "missing_edge")


def is_fact_item(item: Any) -> bool:
    """A semantic row that represents one (subject, predicate, object) fact."""

    metadata = getattr(item, "metadata", {}) or {}
    return all(metadata.get(key) for key in ("subject", "predicate", "object"))


def fact_items(manager: MemoryManager) -> list[MemoryItem]:
    """Every semantic row that is a fact (has subject/predicate/object)."""

    return [item for item in manager.semantic.facts() if is_fact_item(item)]


def projected_vector_ids(manager: MemoryManager) -> set[str] | None:
    """向量库里的 app 级 id 集合；无法枚举（存储不支持或不可达）时返回 None。"""

    list_ids = getattr(manager.vector_store, "list_ids", None)
    if not callable(list_ids):
        return None
    try:
        return {str(value) for value in list_ids()}
    except Exception:  # noqa: BLE001 - 读不到就跳过向量对账，不误报漂移
        return None


def projected_edge_ids(manager: MemoryManager) -> set[str] | None:
    """图投影里的 memory_id 集合（内存回退与 Neo4j 都实现同一方法）。"""

    relation_ids = getattr(manager.graph_store, "relation_memory_ids", None)
    if not callable(relation_ids):
        return None
    try:
        return {str(value) for value in relation_ids() if str(value)}
    except Exception:  # noqa: BLE001 - 同上
        return None


def _drift_entry(kind: str, ids: list[str], *, include_ids: bool) -> dict[str, Any]:
    return {"kind": kind, "count": len(ids), "ids": list(ids) if include_ids else []}


def reconcile_report(
    manager: MemoryManager,
    repository: DocumentRepository | None,
    *,
    include_ids: bool = True,
) -> dict[str, Any]:
    """三库计数与漂移（只看不改）：真值源 ↔ 向量投影 ↔ 图投影。

    ``repository=None``（内存模式或非 SQLite 文档库）时真值源计数按 0 处理，
    向量/图对账仍照常进行——能对多少对多少，不因为一个库不可用就整体失败。
    """

    chunk_ids = set(repository.chunk_ids()) if repository is not None else set()
    indexed = (
        set(repository.chunk_ids(vector_status="indexed"))
        if repository is not None
        else set()
    )
    memory_ids = {item.id for item in manager.document_store.list(include_expired=True)}
    facts = fact_items(manager)
    vectors = projected_vector_ids(manager)
    edges = projected_edge_ids(manager)

    drift: list[dict[str, Any]] = []
    if vectors is not None:
        missing = sorted(indexed - vectors)
        orphan = sorted(vectors - chunk_ids - memory_ids)
        if missing:
            drift.append(_drift_entry("missing_vector", missing, include_ids=include_ids))
        if orphan:
            drift.append(_drift_entry("orphan_vector", orphan, include_ids=include_ids))
    if edges is not None:
        missing_edges = sorted({item.id for item in facts} - edges)
        if missing_edges:
            drift.append(
                _drift_entry("missing_edge", missing_edges, include_ids=include_ids)
            )
    return {
        "counts": {
            "chunks": len(chunk_ids),
            "chunks_indexed_sqlite": len(indexed),
            "qdrant_points": len(vectors) if vectors is not None else -1,
            "facts": len(facts),
            "neo4j_edges": len(edges) if edges is not None else -1,
        },
        "drift": drift,
    }


class ReconcileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    include_ids: bool = Field(
        default=True,
        description="是否返回漂移的具体 id 列表；只关心数量时传 false，避免长列表。",
    )


class ReconcileCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunks: int = Field(description="真值源 chunks 行数。")
    chunks_indexed_sqlite: int = Field(description="真值源里标记为已投影的分块数。")
    qdrant_points: int = Field(description="向量库里的点数；-1 表示无法枚举。")
    facts: int = Field(description="语义层里 (主语, 谓语, 宾语) 事实条数。")
    neo4j_edges: int = Field(description="图投影里的关系数；-1 表示无法枚举。")


class DriftEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: str = Field(description="missing_vector / orphan_vector / missing_edge。")
    count: int
    ids: list[str] = Field(default_factory=list, description="include_ids=false 时为空列表。")


class ReconcileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    counts: ReconcileCounts
    drift: list[DriftEntry] = Field(default_factory=list)
    consistent: bool = Field(description="true 表示三库计数无漂移。")


class ReconcileTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.reconcile",
        description=(
            "Check whether the knowledge base's three stores agree: SQLite "
            "chunks (truth source), the vector projection and the graph "
            "projection. Reports counts plus any drift (chunks marked indexed "
            "but missing from the vector store, orphan vectors, facts without a "
            "graph edge). Read-only: use knowledge.repair_drift to fix."
        ),
        version="1.0.0",
        input_model=ReconcileInput,
        output_model=ReconcileOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=60.0,
        idempotent=True,
        parallel_safe=True,
        tags=("knowledge", "reconcile", "storage", "drift", "read"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: ReconcileInput) -> ReconcileOutput:
        from .hybrid_index import repository_for

        report = reconcile_report(
            self.manager,
            repository_for(self.manager),
            include_ids=arguments.include_ids,
        )
        return ReconcileOutput(
            counts=ReconcileCounts(**report["counts"]),
            drift=[DriftEntry(**entry) for entry in report["drift"]],
            consistent=not report["drift"],
        )


def create_tool() -> BaseTool:
    return ReconcileTool()


__all__ = [
    "DRIFT_KINDS",
    "REPAIRABLE_KINDS",
    "DriftEntry",
    "ReconcileCounts",
    "ReconcileInput",
    "ReconcileOutput",
    "ReconcileTool",
    "create_tool",
    "fact_items",
    "is_fact_item",
    "projected_edge_ids",
    "projected_vector_ids",
    "reconcile_report",
]
