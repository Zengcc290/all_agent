"""孤儿实体检测工具：找出「完全孤立」的实体（只读，只找不删）。

判定“完全孤立”（全部满足才算，缺一不可）：
1. 没有任何活跃关系边（既不是任何事实的 subject，也不是 object）；
2. 没有被任何原句（chunk）通过「提及」边引用（source_ids 里没有 chunk id）；
3. 没有备注（kind=note）挂靠在它名下；
4. 不是 seed 播种的实体（避免把种子星图当噪音清掉）。

为什么这是一个独立能力
======================

这是"体检"能力：它能告诉用户"图里有多少实体是白建的"。它只读、不删，
所以可以随时调用（星云图的 ``orphan_entities`` 统计就复用它，避免二次全表扫描）；
真正的清理走 ``knowledge.propose_cleanup`` 生成待确认的删除提案，
最终删除仍由 ``memory.storage.document_repo.execute_deletion`` 的确认闸门把关。

本模块是这段逻辑的**唯一实现**：``web/cleanup.py`` 里的 ``find_orphan_entities`` 已删除。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryItem, MemoryManager

TOOL_ENABLED = True


def find_orphan_entities(
    manager: MemoryManager, *, items: list[MemoryItem] | None = None
) -> list[MemoryItem]:
    """返回完全孤立的实体记忆项列表（判定口径见模块 docstring）。

    ``items`` 可复用调用方已取回的 semantic 列表，避免图构建时二次全表扫描。
    """

    items = manager.list(memory_type="semantic") if items is None else items
    entities = [item for item in items if item.metadata.get("kind") == "entity"]

    used_in_fact: set[str] = set()
    chunk_ids: set[str] = set()
    note_entities: set[str] = set()
    for item in items:
        metadata = item.metadata
        if metadata.get("subject"):
            used_in_fact.add(str(metadata["subject"]))
        if metadata.get("object"):
            used_in_fact.add(str(metadata["object"]))
        if metadata.get("document_id") is not None and "chunk_index" in metadata:
            chunk_ids.add(item.id)
        if metadata.get("kind") == "note" and metadata.get("entity"):
            note_entities.add(str(metadata["entity"]))

    orphans: list[MemoryItem] = []
    for item in entities:
        metadata = item.metadata
        if metadata.get("seed"):
            continue
        name = str(metadata.get("canonical_name") or metadata.get("title") or item.content)
        if name in used_in_fact:
            continue
        if name in note_entities:
            continue
        source_ids = [str(value) for value in metadata.get("source_ids") or []]
        if any(source_id in chunk_ids for source_id in source_ids):
            continue
        orphans.append(item)
    return orphans


def orphan_summary(item: MemoryItem) -> dict[str, object]:
    """One orphan entity as a small, model-friendly record."""

    metadata = item.metadata
    return {
        "id": item.id,
        "name": str(metadata.get("canonical_name") or metadata.get("title") or item.content),
        "domain": str(metadata.get("domain") or ""),
        "content": item.content,
        "importance": round(float(item.importance), 3),
    }


class OrphanEntitiesInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="最多返回多少个孤儿实体（count 始终是完整数量）。",
    )


class OrphanEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    name: str
    domain: str = ""
    content: str = ""
    importance: float = 0.5


class OrphanEntitiesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    count: int = Field(description="完全孤立实体的总数。")
    entities: list[OrphanEntity] = Field(default_factory=list, description="前 limit 个（按记忆库顺序）。")


class OrphanEntitiesTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.orphan_entities",
        description=(
            "Find entity nodes that are completely isolated: no relation edge, no "
            "chunk mention, no note attached, and not seeded. Read-only health "
            "check; use knowledge.propose_cleanup to open a confirmation-gated "
            "deletion proposal for them."
        ),
        version="1.0.0",
        input_model=OrphanEntitiesInput,
        output_model=OrphanEntitiesOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=60.0,
        idempotent=True,
        parallel_safe=True,
        tags=("knowledge", "graph", "orphan", "health", "read"),
        guidance=(
            "做知识库健康检查、找完全孤立实体时使用。判定包含四项：没有关系边、没有被分块提及、没有备注、且不是种子数据，所以种子实体不会被误判成垃圾。"
            "要清理必须走 knowledge.propose_cleanup（人工确认），不要自行删除。"
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

    def execute(self, arguments: OrphanEntitiesInput) -> OrphanEntitiesOutput:
        orphans = find_orphan_entities(self.manager)
        return OrphanEntitiesOutput(
            count=len(orphans),
            entities=[OrphanEntity(**orphan_summary(item)) for item in orphans[: arguments.limit]],
        )


def create_tool() -> BaseTool:
    return OrphanEntitiesTool()


__all__ = [
    "OrphanEntitiesInput",
    "OrphanEntitiesOutput",
    "OrphanEntitiesTool",
    "OrphanEntity",
    "create_tool",
    "find_orphan_entities",
    "orphan_summary",
]
