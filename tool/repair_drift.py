"""漂移自愈工具：只补缺失的投影，绝不删除或改写真值源。

与 ``knowledge.reconcile`` 的分工
================================

``knowledge.reconcile`` 只读地报告三库漂移；本工具执行**幂等**修复：

- ``missing_vector``：真值源标了 ``indexed`` 但向量库里没有的分块 → 重新嵌入并 upsert，
  再把分块置回 ``indexed``，并把仍是 ``parsed`` 的文档推进到 ``vectorized``；
- ``missing_edge``：语义层有 fact 但图里没有对应关系 → 用同一 ``item_id`` 重放
  ``semantic.add_fact``，图存储按 id 幂等。

刻意**不**处理 ``orphan_vector``：删除是不可逆动作，必须由人来判断，不能由一个
"修复"工具顺手删掉用户的向量。所以工具会明确拒绝这一项，而不是悄悄扩大副作用面。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 里原先内联的 ``repair_drift`` 已删除，
``POST /api/reconcile`` 改为调用这里，并把 ``ValueError`` 映射成 422。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory.base import MemoryType
from memory.manager import MemoryManager
from memory.storage.document_repo import DocumentRepository

from .hybrid_index import repository_for
from .reconcile import REPAIRABLE_KINDS, fact_items, reconcile_report

TOOL_ENABLED = True


def repair_drift(
    manager: MemoryManager,
    repository: DocumentRepository | None,
    kinds: list[str],
) -> dict[str, Any]:
    """幂等自愈：只补缺失的投影，绝不删除或改写真值源。

    未知类型抛 ``ValueError``（调用方负责映射成 4xx）。``repository=None``
    （内存模式或非 SQLite 文档库）时真值源里没有任何"已标记投影"的分块，
    因此 ``missing_vector`` 恒为 0——报告里的 ``chunks_indexed_sqlite`` 同样是 0，
    不存在"假装修好了"的空间。
    """

    requested = list(dict.fromkeys(kinds or []))
    unknown = [kind for kind in requested if kind not in REPAIRABLE_KINDS]
    if unknown:
        raise ValueError(f"不支持的修复类型：{', '.join(unknown)}")
    entries = {
        entry["kind"]: entry["ids"] for entry in reconcile_report(manager, repository)["drift"]
    }
    repaired = {"missing_vector": 0, "missing_edge": 0}

    if "missing_vector" in requested and entries.get("missing_vector") and repository is not None:
        ids = entries["missing_vector"]
        chunks = [
            chunk
            for chunk in (repository.get_chunk(chunk_id) for chunk_id in ids)
            if chunk is not None
        ]
        if chunks:
            vectors = manager.embedding.embed_batch([chunk.text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                manager.vector_store.upsert_chunk(
                    chunk.chunk_id,
                    vector,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    source="",
                    memory_type=MemoryType.SEMANTIC.value,
                )
                repository.set_chunk_vector_status(chunk.chunk_id, "indexed")
            repaired["missing_vector"] = len(chunks)
            for document_id in {chunk.document_id for chunk in chunks}:
                document = repository.get_document(document_id)
                if document is not None and document.status == "parsed":
                    repository.set_status(document_id, "vectorized")

    if "missing_edge" in requested and entries.get("missing_edge"):
        wanted = set(entries["missing_edge"])
        for item in fact_items(manager):
            if item.id not in wanted:
                continue
            manager.semantic.add_fact(
                str(item.metadata["subject"]),
                str(item.metadata["predicate"]),
                str(item.metadata["object"]),
                metadata=item.metadata,
                confidence=float(item.importance),
                item_id=item.id,
            )
            repaired["missing_edge"] += 1

    return {"repaired": repaired}


class RepairDriftInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    repair: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="要修复的漂移类型，可选项只有 missing_vector 与 missing_edge。",
    )


class RepairedCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    missing_vector: int = Field(description="本次补回的向量条数。")
    missing_edge: int = Field(description="本次补回的图关系条数。")


class RepairDriftOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    repaired: RepairedCounts


class RepairDriftTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.repair_drift",
        description=(
            "Idempotently repair storage drift: re-embed chunks that the truth "
            "source marked as indexed but the vector store is missing, and "
            "replay facts whose graph edge is missing. It only ever ADDS "
            "projections — never deletes or rewrites the truth source, and it "
            "refuses orphan_vector because deletion must be a human decision."
        ),
        version="1.0.0",
        input_model=RepairDriftInput,
        output_model=RepairDriftOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=300.0,
        idempotent=True,
        parallel_safe=False,
        tags=("knowledge", "reconcile", "repair", "storage", "write"),
        guidance=(
            "只在 knowledge.reconcile 报告了漂移、且用户同意修复时调用。它只补投影、绝不删除或改写真值源；orphan_vector 会被明确拒绝，因为删除必须由人决定。"
            "修复后再跑一次 reconcile 验证归零。"
        ),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: RepairDriftInput) -> RepairDriftOutput:
        manager = self.manager
        result = repair_drift(manager, repository_for(manager), arguments.repair)
        return RepairDriftOutput(repaired=RepairedCounts(**result["repaired"]))


def create_tool() -> BaseTool:
    return RepairDriftTool()


__all__ = [
    "RepairDriftInput",
    "RepairDriftOutput",
    "RepairDriftTool",
    "RepairedCounts",
    "create_tool",
    "repair_drift",
]
