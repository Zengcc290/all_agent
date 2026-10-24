"""SQLite embedding lock + confirmed Qdrant rebuild."""

from __future__ import annotations

import pytest

from memory import HashEmbedding, MemoryConfig, MemoryManager
from memory.embedding_lock import (
    EmbeddingLockMismatch,
    apply_embedding_lock,
    rebuild_vector_projection,
    reindex_vector_projection,
    resolve_embedding_identity,
)
from memory.rag import Document
from memory.rag.pipeline import RAGPipeline
from memory.storage.document_repo import ChunkRecord, DocumentRecord, DocumentRepository
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
        # Each chunk id is projected once. A second generic MemoryItem upsert
        # would overwrite document_id/chunk_index/source in the Qdrant payload.
        assert len(client.points) == len({point.id for point in client.points})
        assert all("chunk_id" in point.payload for point in client.points)
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


@pytest.mark.parametrize("recreate_collection", [False, True])
def test_reindex_projects_each_unique_id_once_and_recovers_orphan_chunks(
    tmp_path, recreate_collection
):
    class CountingEmbedding(HashEmbedding):
        def __init__(self) -> None:
            super().__init__(dimension=8)
            self.calls: list[tuple[str, str, object, object]] = []

        def embed(self, text):
            self.calls.append(("chunk", text, None, None))
            return HashEmbedding.embed(self, text)

        def embed_item(self, text, *, payload=None, modality=None):
            self.calls.append(("memory", text, payload, modality))
            return HashEmbedding.embed(self, text)

    client = FakeQdrantClient()
    embedding = CountingEmbedding()
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=embedding,
        vector_store=QdrantVectorStore(client=client, namespace="tests"),
    )
    repo = DocumentRepository(manager.document_store.path)
    try:
        repo.upsert_document(DocumentRecord("doc", raw_text="成对分块孤立分块", source="doc.txt"))
        repo.upsert_chunk(ChunkRecord("paired", "doc", 0, 0, 4, "成对分块"))
        manager.add("成对分块", memory_type="semantic", item_id="paired")
        repo.upsert_chunk(ChunkRecord("orphan", "doc", 1, 4, 8, "孤立分块"))
        manager.add("普通记忆", memory_type="semantic", item_id="ordinary")
        manager.add(
            "相机画面",
            memory_type="perceptual",
            item_id="image",
            payload=b"image-bytes",
            modality="image",
        )
        embedding.calls.clear()
        client.points.clear()
        client.deleted_collections.clear()

        report = reindex_vector_projection(
            manager,
            repo,
            recreate_collection=recreate_collection,
        )

        assert report["chunks"] == 2
        assert report["memories"] == 4
        assert report["failed"] == []
        assert len(embedding.calls) == 4
        assert [call[0] for call in embedding.calls].count("chunk") == 1
        assert ("memory", "相机画面", b"image-bytes", "image") in embedding.calls
        assert len(client.points) == 4
        assert sum("chunk_id" in point.payload for point in client.points) == 2
        assert all(chunk.vector_status == "indexed" for chunk in repo.list_all_chunks())
        recovered = manager.document_store.get("orphan")
        assert recovered is not None
        assert recovered.metadata["document_id"] == "doc"
        assert bool(client.deleted_collections) is recreate_collection
    finally:
        repo.close()
        manager.close()


def test_unlocked_populated_store_requires_confirmed_rebuild(tmp_path):
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(dimension=8),
    )
    legacy = manager.add("旧空间向量", memory_type="semantic")
    repo = DocumentRepository(manager.document_store.path)
    try:
        assert repo.get_embedding_lock() is None
        with pytest.raises(EmbeddingLockMismatch) as excinfo:
            apply_embedding_lock(manager, repo)
        assert excinfo.value.locked.model == "__unlocked_existing_data__"

        apply_embedding_lock(manager, repo, confirm_rebuild=True)

        assert repo.get_embedding_lock().model == "HashEmbedding"
        assert manager.get(legacy.id) is not None
    finally:
        repo.close()
        manager.close()


def test_failed_reindex_does_not_commit_staged_sqlite_embeddings(tmp_path):
    class FailingEmbedding(HashEmbedding):
        def __init__(self) -> None:
            super().__init__(dimension=16)
            self.calls = 0

        def embed_item(self, text, *, payload=None, modality=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second item failed")
            return super().embed_item(text, payload=payload, modality=modality)

    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(dimension=8),
    )
    first = manager.add("第一条", memory_type="semantic", item_id="first")
    second = manager.add("第二条", memory_type="semantic", item_id="second")
    old_embeddings = {first.id: first.embedding, second.id: second.embedding}
    repo = DocumentRepository(manager.document_store.path)
    repo.set_embedding_lock("HashEmbedding", 8)
    manager.embedding = FailingEmbedding()
    try:
        report = reindex_vector_projection(manager, repo, continue_on_error=True)

        assert len(report["failed"]) == 1
        assert repo.get_embedding_lock().model == "__rebuild_in_progress__"
        assert manager.document_store.get("first").embedding == old_embeddings["first"]
        assert manager.document_store.get("second").embedding == old_embeddings["second"]
    finally:
        repo.close()
        manager.close()


def test_recovered_orphan_chunk_survives_inmemory_restart(tmp_path):
    path = str(tmp_path / "memory.sqlite3")
    manager = MemoryManager(MemoryConfig(sqlite_path=path), embedding=HashEmbedding(dimension=8))
    repo = DocumentRepository(path)
    try:
        repo.upsert_document(DocumentRecord("doc", raw_text="唯一孤立词", source="doc.txt"))
        repo.upsert_chunk(ChunkRecord("orphan", "doc", 0, 0, 5, "唯一孤立词"))
        reindex_vector_projection(manager, repo)
    finally:
        repo.close()
        manager.close()

    restarted = MemoryManager(MemoryConfig(sqlite_path=path), embedding=HashEmbedding(dimension=8))
    try:
        assert restarted.get("orphan") is not None
        assert [result.item.id for result in restarted.search("唯一孤立词", memory_type="semantic")] == [
            "orphan"
        ]
    finally:
        restarted.close()


@pytest.mark.parametrize("mismatch", [False, True])
def test_reindex_script_routes_both_branches_through_unique_projection(
    tmp_path, monkeypatch, mismatch
):
    from types import SimpleNamespace

    import scripts.reindex_embeddings as script

    db_path = tmp_path / "memory.sqlite3"
    db_path.touch()
    calls: list[tuple[str, bool]] = []

    class ConfigFactory:
        @classmethod
        def from_config(cls):
            return SimpleNamespace(sqlite_path="")

    class CloudEmbedding:
        timeout = 1.0

    class FakeManager:
        def __init__(self, config, *, embedding):
            self.vector_store = object()
            self.closed = False

        def close(self):
            self.closed = True

    class FakeRepository:
        def __init__(self, path):
            self.closed = False

        def close(self):
            self.closed = True

    def apply(manager, repo, *, confirm_rebuild=False):
        calls.append(("rebuild", confirm_rebuild))

    def reindex(manager, repo, *, recreate_collection, continue_on_error):
        calls.append(("reindex", recreate_collection))
        assert continue_on_error is True
        return {"chunks": 2, "memories": 3, "failed": []}

    monkeypatch.setattr(script, "MemoryConfig", ConfigFactory)
    monkeypatch.setattr(script, "MemoryManager", FakeManager)
    monkeypatch.setattr(script, "DocumentRepository", FakeRepository)
    monkeypatch.setattr(script, "make_default_embedding", lambda config: CloudEmbedding())
    monkeypatch.setattr(script, "default_sqlite_path", lambda: str(db_path))
    monkeypatch.setattr(script, "inspect_embedding_lock", lambda manager, repo: {"mismatch": mismatch})
    monkeypatch.setattr(script, "apply_embedding_lock", apply)
    monkeypatch.setattr(script, "reindex_vector_projection", reindex)
    monkeypatch.setattr(
        script.sys,
        "argv",
        ["reindex_embeddings.py", "--confirm-rebuild"] if mismatch else ["reindex_embeddings.py"],
    )

    assert script.main() == 0
    expected = [("rebuild", True)] if mismatch else [("reindex", False)]
    assert calls == expected


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
