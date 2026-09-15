"""Phase 5: scripts/migrate_storage.py 的迁移与回填行为。

覆盖：分组与排序、raw_text 与 char 区间同源、dry-run 不写库、幂等、
memories 既有数据不被改动、无网关时不清空向量（防数据丢失）、备份可用。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, MemoryType
from memory.storage.document_repo import DocumentRepository

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import migrate_storage


@pytest.fixture()
def db_path(tmp_path) -> Path:
    return tmp_path / "memory.sqlite3"


@pytest.fixture()
def manager(db_path: Path) -> MemoryManager:
    instance = MemoryManager(MemoryConfig(sqlite_path=str(db_path)), embedding=HashEmbedding())
    yield instance
    instance.close()


def seed_legacy(manager: MemoryManager) -> None:
    """现库形态：分块是 semantic 条目（带 document_id + chunk_index），事实另存。"""

    for index, text in ((2, "第三段讲 Qdrant 投影。"), (0, "第一段讲 SQLite 真值源。"), (1, "第二段讲 Neo4j 图谱。")):
        manager.add(
            text,
            memory_type=MemoryType.SEMANTIC,
            metadata={"document_id": "doc-legacy", "chunk_index": index, "source": "旧文档.txt"},
            item_id=f"doc-legacy:{index}",
        )
    manager.semantic.add_fact("Qdrant", "是", "向量库", confidence=0.9)
    manager.add("工作记忆条目。", memory_type=MemoryType.WORKING)


def test_migration_groups_chunks_and_keeps_memories(manager: MemoryManager):
    seed_legacy(manager)

    report = migrate_storage.migrate(manager)

    assert report == {"documents": 1, "chunks": 3, "facts_kept": 1, "graph_replayed": 0}
    repo = DocumentRepository(manager.config.sqlite_path)
    try:
        document = repo.get_document("doc-legacy")
        chunks = repo.list_chunks("doc-legacy")
        assert document is not None
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]  # 按 chunk_index 排序
        assert document.raw_text == "\n".join(chunk.text for chunk in chunks)
        assert document.source == "旧文档.txt"
        for chunk in chunks:
            assert document.raw_text[chunk.char_start : chunk.char_end] == chunk.text
        assert repo.stats()["chunks_indexed"] == 0  # 尚未重嵌入，状态诚实为 pending
    finally:
        repo.close()

    # memories 一条不少：分块 3 + 事实 1 + 工作记忆 1
    assert len(manager.document_store.list(include_expired=True)) == 5


def test_migration_is_idempotent(manager: MemoryManager):
    seed_legacy(manager)
    migrate_storage.migrate(manager)

    second = migrate_storage.migrate(manager)

    assert second["documents"] == 0 and second["chunks"] == 0 and second["facts_kept"] == 1
    repo = DocumentRepository(manager.config.sqlite_path)
    try:
        assert repo.count_documents() == 1
        assert len(repo.list_chunks("doc-legacy")) == 3
    finally:
        repo.close()


def test_dry_run_writes_nothing(manager: MemoryManager):
    seed_legacy(manager)

    report = migrate_storage.migrate(manager, dry_run=True)

    assert report["documents"] == 1 and report["chunks"] == 3
    repo = DocumentRepository(manager.config.sqlite_path)
    try:
        assert repo.count_documents() == 0 and repo.stats()["chunks"] == 0
    finally:
        repo.close()


def test_reindex_skips_without_gateway_and_keeps_embeddings(manager: MemoryManager):
    """没有网关时既不能重嵌入，也就绝不能清空 memories.embedding（否则向量尽失）。"""

    seed_legacy(manager)
    migrate_storage.migrate(manager)

    outcome = migrate_storage.reindex(manager)

    assert outcome == {"reindexed": 0, "blanked": 0, "skipped": "no_gateway"}
    items = manager.document_store.list(include_expired=True)
    assert all(item.embedding for item in items)


def test_backup_database_is_restorable(db_path: Path, manager: MemoryManager):
    seed_legacy(manager)

    backup = migrate_storage.backup_database(db_path)

    assert backup.exists() and backup.name.endswith(migrate_storage.BACKUP_SUFFIX)
    with sqlite3.connect(backup) as connection:
        assert connection.execute("SELECT count(*) FROM memories").fetchone()[0] == 5
