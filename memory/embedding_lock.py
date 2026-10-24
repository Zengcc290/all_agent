"""SQLite embedding lock: persist (model, dimension) and rebuild on confirm.

The vector store is a projection of SQLite truth. Mixing embedding spaces
silently (or recreating Qdrant on the write path) corrupts retrieval. This
module keeps the locked identity in SQLite, refuses mismatched writes, and
only recreates + reindexes after an explicit confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import MemoryItem, MemoryType
from .embedding import BaseEmbedding
from .manager import MemoryManager
from .storage.document_repo import DocumentRepository, EmbeddingLockRecord

UNKNOWN_EMBEDDING_MODEL = "__unlocked_existing_data__"
REBUILDING_EMBEDDING_MODEL = "__rebuild_in_progress__"


@dataclass(frozen=True)
class EmbeddingIdentity:
    model: str
    dimension: int


class EmbeddingLockMismatch(ValueError):
    """Current embedding does not match the SQLite lock."""

    def __init__(self, locked: EmbeddingLockRecord, current: EmbeddingIdentity) -> None:
        self.locked = locked
        self.current = current
        super().__init__(self.message)

    @property
    def message(self) -> str:
        return (
            f"嵌入锁定不一致：库内是 {self.locked.model} / {self.locked.dimension} 维，"
            f"当前是 {self.current.model} / {self.current.dimension} 维。"
            "继续将重建向量投影并全量重灌；取消则保持锁定配置。"
        )

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": "embedding_lock_mismatch",
            "message": self.message,
            "locked": {"model": self.locked.model, "dimension": self.locked.dimension},
            "current": {"model": self.current.model, "dimension": self.current.dimension},
        }


def embedding_model_name(embedding: BaseEmbedding) -> str:
    model = getattr(embedding, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return type(embedding).__name__


def resolve_embedding_identity(embedding: BaseEmbedding) -> EmbeddingIdentity:
    """Return the live (model, dimension); probe once if dimension is unknown."""

    model = embedding_model_name(embedding)
    dimension = int(getattr(embedding, "dimension", 0) or 0)
    if dimension < 1:
        vector = embedding.embed("embedding-lock-probe")
        dimension = len(vector)
        if not getattr(embedding, "dimension", 0):
            embedding.dimension = dimension
    if dimension < 1:
        raise ValueError("embedding dimension must be a positive integer")
    return EmbeddingIdentity(model=model, dimension=dimension)


def live_vector_dimension(manager: MemoryManager) -> int | None:
    """Read the live vector-store collection size, if the backend exposes one."""

    getter = getattr(manager.vector_store, "collection_dimension", None)
    if not callable(getter):
        return None
    try:
        size = getter()
    except Exception:  # noqa: BLE001 - 读不到投影尺寸时退回 SQLite 锁
        return None
    return int(size) if size is not None else None


def inspect_embedding_lock(
    manager: MemoryManager,
    repository: DocumentRepository | None,
) -> dict[str, Any]:
    """Describe current embedding vs SQLite lock vs live Qdrant size.

    Does not write. Live Qdrant dimension wins over a stale SQLite lock: an
    empty/wrong lock must not hide an existing 1024-d collection.
    """

    current = resolve_embedding_identity(manager.embedding)
    locked = repository.get_embedding_lock() if repository is not None else None
    projected = live_vector_dimension(manager)
    effective = locked
    mismatch = False
    unlocked_data = False
    if locked is None:
        memories_exist = bool(manager.document_store.list(include_expired=True))
        chunks_exist = bool(repository.chunk_ids()) if repository is not None else False
        # A non-empty live collection with an unknown model is not safe to claim
        # even if its dimension happens to match the configured model.
        projection_exists = False
        list_ids = getattr(manager.vector_store, "list_ids", None)
        if callable(list_ids):
            try:
                projection_exists = projection_exists or bool(list_ids(limit=1))
            except TypeError:
                projection_exists = projection_exists or bool(list_ids())
        unlocked_data = memories_exist or chunks_exist or projection_exists
    if unlocked_data:
        mismatch = True
        effective = EmbeddingLockRecord(
            model=UNKNOWN_EMBEDDING_MODEL,
            dimension=projected or current.dimension,
        )
    elif projected is not None and projected != current.dimension:
        mismatch = True
        effective = EmbeddingLockRecord(
            model=locked.model if locked is not None else "Qdrant",
            dimension=projected,
            updated_at=locked.updated_at if locked is not None else "",
        )
    elif locked is not None and (
        locked.model != current.model or locked.dimension != current.dimension
    ):
        mismatch = True
        effective = locked
    return {
        "current": {"model": current.model, "dimension": current.dimension},
        "locked": (
            None
            if locked is None
            else {"model": locked.model, "dimension": locked.dimension, "updated_at": locked.updated_at}
        ),
        "projection": (
            None
            if effective is None
            else {
                "model": effective.model,
                "dimension": effective.dimension,
                "updated_at": effective.updated_at,
            }
        ),
        "qdrant_dimension": projected,
        "mismatch": mismatch,
        "_current": current,
        "_effective": effective,
    }


def apply_embedding_lock(
    manager: MemoryManager,
    repository: DocumentRepository | None,
    *,
    confirm_rebuild: bool = False,
) -> EmbeddingLockRecord | None:
    """Gate a vector write against SQLite lock and the live Qdrant collection.

    ``:memory:`` / missing repository is a no-op so unit tests keep using
    isolated in-memory stores. An empty lock is claimed only when the live
    collection is missing or already matches. A mismatch raises unless
    ``confirm_rebuild`` is set, in which case the vector collection is
    recreated and SQLite truth is reindexed.
    """

    if repository is None:
        return None
    snapshot = inspect_embedding_lock(manager, repository)
    current: EmbeddingIdentity = snapshot["_current"]
    if snapshot["mismatch"]:
        effective = snapshot["_effective"]
        if not isinstance(effective, EmbeddingLockRecord):
            effective = EmbeddingLockRecord(current.model, current.dimension)
        if not confirm_rebuild:
            raise EmbeddingLockMismatch(effective, current)
        rebuild_vector_projection(manager, repository, current)
        return repository.get_embedding_lock()
    locked = repository.get_embedding_lock()
    if locked is None:
        return repository.set_embedding_lock(current.model, current.dimension)
    return locked


def mismatch_from_exception(
    exc: BaseException,
    manager: MemoryManager,
    repository: DocumentRepository | None,
) -> EmbeddingLockMismatch | None:
    """Turn a Qdrant dimension error into the same 409 payload as the lock gate."""

    if isinstance(exc, EmbeddingLockMismatch):
        return exc
    text = str(exc).casefold()
    if "dimension mismatch" not in text and "expected dim" not in text:
        return None
    snapshot = inspect_embedding_lock(manager, repository)
    current: EmbeddingIdentity = snapshot["_current"]
    effective = snapshot["_effective"]
    if not isinstance(effective, EmbeddingLockRecord):
        projected = snapshot["qdrant_dimension"] or current.dimension
        effective = EmbeddingLockRecord("Qdrant", int(projected))
    return EmbeddingLockMismatch(effective, current)


def reindex_vector_projection(
    manager: MemoryManager,
    repository: DocumentRepository,
    identity: EmbeddingIdentity | None = None,
    *,
    recreate_collection: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Project every unique SQLite id once, including orphan/pending chunks.

    Normal ingestion stores a chunk in both ``chunks`` and ``memories`` under
    the same id. That pair is embedded once via ``embed_item`` and written with
    the richer chunk payload; ordinary memories follow afterwards. A chunk that
    never reached ``memories`` is still recoverable from the chunks truth source.
    The embedding lock advances only when every projection write succeeds.
    """

    current = identity or resolve_embedding_identity(manager.embedding)
    repository.set_embedding_lock(REBUILDING_EMBEDDING_MODEL, current.dimension)
    recreate = getattr(manager.vector_store, "recreate_collection", None)
    if recreate_collection and callable(recreate):
        recreate(current.dimension)

    memories = manager.document_store.list(include_expired=True)
    memory_by_id = {item.id: item for item in memories}
    chunk_ids: set[str] = set()
    indexed_chunk_ids: list[str] = []
    staged_memories: list[MemoryItem] = []
    failures: list[dict[str, str]] = []
    chunks_done = memories_done = 0
    upsert_chunk = getattr(manager.vector_store, "upsert_chunk", None)

    def record_failure(item_id: str, exc: Exception) -> None:
        failures.append(
            {"id": item_id, "error": f"{type(exc).__name__}: {exc}"}
        )
        if not continue_on_error:
            raise exc

    for chunk in repository.list_all_chunks():
        chunk_ids.add(chunk.chunk_id)
        item = memory_by_id.get(chunk.chunk_id)
        try:
            document = repository.get_document(chunk.document_id)
            if item is None:
                vector = manager.embedding.embed(chunk.text)
                item = MemoryItem(
                    id=chunk.chunk_id,
                    content=chunk.text,
                    memory_type=MemoryType.SEMANTIC,
                    metadata={
                        "kind": "chunk",
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "source": document.source if document is not None else "",
                    },
                    embedding=vector,
                )
            else:
                vector = manager.embedding.embed_item(
                    item.content,
                    payload=item.payload,
                    modality=item.modality,
                )
                item.embedding = vector
            staged_memories.append(item)

            if callable(upsert_chunk):
                upsert_chunk(
                    chunk.chunk_id,
                    vector,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    source=document.source if document is not None else "",
                    memory_type="semantic",
                )
            else:
                manager.vector_store.upsert(item)
            indexed_chunk_ids.append(chunk.chunk_id)
            chunks_done += 1
            memories_done += 1
        except Exception as exc:  # noqa: BLE001 - caller chooses fail-fast/best-effort
            repository.set_chunk_vector_status(chunk.chunk_id, "failed")
            record_failure(chunk.chunk_id, exc)

    for item in memories:
        if item.id in chunk_ids:
            continue
        try:
            item.embedding = manager.embedding.embed_item(
                item.content,
                payload=item.payload,
                modality=item.modality,
            )
            staged_memories.append(item)
            manager.vector_store.upsert(item)
            memories_done += 1
        except Exception as exc:  # noqa: BLE001 - caller chooses fail-fast/best-effort
            record_failure(item.id, exc)

    if not failures:
        upsert_many = getattr(manager.document_store, "upsert_many", None)
        if not callable(upsert_many):
            raise RuntimeError("document store does not support atomic projection commits")
        upsert_many(staged_memories)
        for chunk_id in indexed_chunk_ids:
            repository.set_chunk_vector_status(chunk_id, "indexed")
        repository.set_embedding_lock(current.model, current.dimension)
    return {
        "model": current.model,
        "dimension": current.dimension,
        "chunks": chunks_done,
        "memories": memories_done,
        "failed": failures,
    }


def rebuild_vector_projection(
    manager: MemoryManager,
    repository: DocumentRepository,
    identity: EmbeddingIdentity | None = None,
) -> dict[str, Any]:
    """Recreate the vector collection, then project every unique SQLite id."""

    return reindex_vector_projection(
        manager,
        repository,
        identity,
        recreate_collection=True,
    )


__all__ = [
    "EmbeddingIdentity",
    "EmbeddingLockMismatch",
    "apply_embedding_lock",
    "embedding_model_name",
    "inspect_embedding_lock",
    "live_vector_dimension",
    "mismatch_from_exception",
    "rebuild_vector_projection",
    "reindex_vector_projection",
    "resolve_embedding_identity",
]
