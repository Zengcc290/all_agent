"""孤立实体清理：找出完全没被用到的实体，走删除提案通道等待确认。

判定“完全孤立”（全部满足才算，缺一不可）：
1. 没有任何活跃关系边（既不是任何事实的 subject，也不是 object）；
2. 没有被任何原句（chunk）通过「提及」边引用（source_ids 里没有 chunk id）；
3. 没有备注（kind=note）挂靠在它名下；
4. 不是 seed 播种的实体（避免把种子星图当噪音清掉）。

实现上只负责「找出候选 + 生成删除提案」，真正的删除仍走
``memory.storage.document_repo.execute_deletion`` 的确认闸门，绝不绕过。
"""

from __future__ import annotations

from memory import MemoryItem, MemoryManager
from memory.storage.document_repo import DeletionProposalStore


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


def propose_orphan_cleanup(
    manager: MemoryManager,
    *,
    requested_by: str = "maintenance",
    reason: str = "完全孤立实体：无关系边、无原句提及、无备注挂靠",
) -> dict[str, object]:
    """为所有完全孤立实体创建一条待确认的删除提案。

    只写提案、不删数据；调用方拿到 proposal_id + confirm_token 后，由用户
    （或显式确认流程）调用 ``execute_deletion`` 才会真正删除。
    """

    orphans = find_orphan_entities(manager)
    item_ids = [item.id for item in orphans]
    if not item_ids:
        return {
            "proposal_id": "",
            "confirm_token": "",
            "count": 0,
            "item_ids": [],
            "note": "没有发现完全孤立的实体",
        }
    connection = getattr(manager.document_store, "connection", None)
    store = DeletionProposalStore(manager.document_store.path, connection=connection)
    proposal = store.create(requested_by=requested_by, reason=reason, item_ids=item_ids)
    return {
        "proposal_id": proposal.proposal_id,
        "confirm_token": proposal.confirm_token,
        "count": len(item_ids),
        "item_ids": item_ids,
        "note": f"已生成删除提案 {proposal.proposal_id[:8]}，等待确认",
    }


__all__ = ["find_orphan_entities", "propose_orphan_cleanup"]
