"""SQLite embedding lock + confirmed Qdrant rebuild."""

from __future__ import annotations

import pytest

from memory import HashEmbedding, MemoryConfig, MemoryManager
from memory.embedding_lock import (
    EmbeddingLockMismatch,
    apply_embedding_lock,
    rebuild_vector_projection,
    resolve_embedding_identity,
)
from memory.rag import Document
from memory.rag.pipeline import RAGPipeline
from memory.storage.document_repo import DocumentRepository
from memory.storage.qdrant import QdrantVectorStore
from tests.test_qdrant_hybrid import FakeQdrantClient


def test_apply_lock_claims_empty_sqlite(tmp_path):
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(dimension=8),
    )
    repo = DocumentRepository(manager.document_store.path)
    try:
        locked = apply_embedding_lock(manager, repo)
        assert locked is not None
        assert locked.model == "HashEmbedding"
        assert locked.dimension == 8
        again = apply_embedding_lock(manager, repo)
        assert again.dimension == 8
    finally:
        repo.close()
        manager.close()


def test_apply_lock_raises_on_mismatch_without_confirm(tmp_path):
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(dimension=8),
    )
    repo = DocumentRepository(manager.document_store.path)
    try:
        apply_embedding_lock(manager, repo)
        manager.embedding = HashEmbedding(dimension=16)
        with pytest.raises(EmbeddingLockMismatch) as excinfo:
            apply_embedding_lock(manager, repo)
        detail = excinfo.value.to_detail()
        assert detail["code"] == "embedding_lock_mismatch"
        assert detail["locked"] == {"model": "HashEmbedding", "dimension": 8}
        assert detail["current"] == {"model": "HashEmbedding", "dimension": 16}
        assert repo.get_embedding_lock().dimension == 8
    finally:
        repo.close()
        manager.close()


def test_confirm_rebuild_recreates_qdrant_and_reindexes(tmp_path):
    client = FakeQdrantClient()
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(dimension=8),
        vector_store=QdrantVectorStore(client=client, namespace="tests"),
    )
    pipeline = RAGPipeline(manager, auto_extract=False)
    try:
        pipeline.ingest(Document("锁定 8 维后的原文。" * 8, id="doc-lock"), chunk_size=80, overlap=10)
        assert client.existing_size == 8
        old_points = list(client.points)
        assert old_points

        manager.embedding = HashEmbedding(dimension=16)
        repo = pipeline.document_repo()
        with pytest.raises(EmbeddingLockMismatch):
            apply_embedding_lock(manager, repo)

        apply_embedding_lock(manager, repo, confirm_rebuild=True)

        assert client.deleted_collections == [manager.vector_store.collection_name]
        assert client.existing_size == 16
        assert repo.get_embedding_lock().dimension == 16
        assert client.points
        assert all(len(point.vector) == 16 for point in client.points)
        identity = resolve_embedding_identity(manager.embedding)
        assert identity.dimension == 16
    finally:
        pipeline.close()
        manager.close()


def test_rebuild_helper_updates_lock(tmp_path):
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(dimension=8),
    )
    repo = DocumentRepository(manager.document_store.path)
    try:
        repo.set_embedding_lock("HashEmbedding", 8)
        manager.embedding = HashEmbedding(dimension=32)
        report = rebuild_vector_projection(manager, repo)
        assert report["dimension"] == 32
        assert repo.get_embedding_lock().dimension == 32
    finally:
        repo.close()
        manager.close()


def test_live_qdrant_size_overrides_stale_sqlite_lock(tmp_path):
    """SQLite 锁被误写成新维度时，仍须按 Qdrant 现有集合抬错。"""

    client = FakeQdrantClient(exists=True, existing_size=1024)
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(dimension=4096),
        vector_store=QdrantVectorStore(client=client, namespace="tests"),
    )
    repo = DocumentRepository(manager.document_store.path)
    try:
        repo.set_embedding_lock("HashEmbedding", 4096)
        with pytest.raises(EmbeddingLockMismatch) as excinfo:
            apply_embedding_lock(manager, repo)
        detail = excinfo.value.to_detail()
        assert detail["locked"]["dimension"] == 1024
        assert detail["current"]["dimension"] == 4096
        assert repo.get_embedding_lock().dimension == 4096  # 未确认前不改 SQLite 锁
    finally:
        repo.close()
        manager.close()
