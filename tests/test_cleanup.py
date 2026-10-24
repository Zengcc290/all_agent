"""孤立实体只读检测的最小回归集。

只有完全孤立（无关系边、无原句提及、无备注挂靠、非 seed）的实体才会被列为候选；
有事实边、有提及、有备注、或 seed 播种的实体都不算孤立。
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from tool.orphan_entities import find_orphan_entities


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


def _add_entity(manager: MemoryManager, name: str, **metadata: object) -> None:
    manager.add(
        name,
        memory_type="semantic",
        metadata={"kind": "entity", "title": name, **metadata},
    )


def _add_chunk(manager: MemoryManager, chunk_id: str) -> None:
    manager.add(
        f"原句 {chunk_id}",
        memory_type="semantic",
        metadata={"kind": "chunk", "document_id": "doc-1", "chunk_index": 0, "title": chunk_id},
    )


def _orphan_names(manager: MemoryManager) -> set[str]:
    return {
        str(item.metadata.get("canonical_name") or item.metadata.get("title") or item.content)
        for item in find_orphan_entities(manager)
    }


def test_completely_orphan_entity_is_detected(manager: MemoryManager) -> None:
    """验收①：无任何使用痕迹的实体必须被列为候选。"""

    _add_entity(manager, "尘埃")
    assert _orphan_names(manager) == {"尘埃"}


def test_entities_with_usage_are_not_orphans(manager: MemoryManager) -> None:
    """验收②：有事实边/提及/备注/seed 的实体都不算孤立。"""

    # 参与事实的实体
    _add_entity(manager, "恒星")
    manager.semantic.add_fact("恒星", "照亮", "行星", confidence=0.9)
    _add_entity(manager, "行星")
    # 被原句提及的实体（source_ids 里必须是真实 chunk 的 item id）
    chunk = manager.add(
        "提到卫星的原句",
        memory_type="semantic",
        metadata={"kind": "chunk", "document_id": "doc-1", "chunk_index": 0},
    )
    _add_entity(manager, "卫星", source_ids=[chunk.id])
    # 挂有备注的实体
    _add_entity(manager, "陨石")
    manager.add(
        "备注内容",
        memory_type="semantic",
        metadata={"kind": "note", "entity": "陨石", "title": "档案"},
    )
    # seed 播种的实体，即使完全没被用也不能清
    _add_entity(manager, "种子", seed="aetheria-seed-v1")

    assert _orphan_names(manager) == set()


def test_chunk_mention_keeps_entity_alive(manager: MemoryManager) -> None:
    """被任意原句（chunk）通过 source_ids 提及的实体不是孤立。"""

    chunk = manager.add(
        "提到尘埃的原句",
        memory_type="semantic",
        metadata={"kind": "chunk", "document_id": "doc-2", "chunk_index": 0},
    )
    _add_entity(manager, "尘埃", source_ids=[chunk.id])
    assert _orphan_names(manager) == set()
