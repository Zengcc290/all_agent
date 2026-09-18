"""SQLite embedding lock: persist (model, dimension) and rebuild on confirm.

The vector store is a projection of SQLite truth. Mixing embedding spaces
silently (or recreating Qdrant on the write path) corrupts retrieval. This
module keeps the locked identity in SQLite, refuses mismatched writes, and
only recreates + reindexes after an explicit confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embedding import BaseEmbedding
from .manager import MemoryManager
from .storage.document_repo import DocumentRepository, EmbeddingLockRecord


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
    if projected is not None and projected != current.dimension:
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


def rebuild_vector_projection(
    manager: MemoryManager,
    repository: DocumentRepository,
    identity: EmbeddingIdentity | None = None,
) -> dict[str, Any]:
    """Recreate the vector collection and re-embed every SQLite chunk/memory."""

    current = identity or resolve_embedding_identity(manager.embedding)
    recreate = getattr(manager.vector_store, "recreate_collection", None)
    if callable(recreate):
        recreate(current.dimension)
    chunks_done = 0
    upsert_chunk = getattr(manager.vector_store, "upsert_chunk", None)
    for chunk in repository.list_all_chunks():
        vector = manager.embedding.embed(chunk.text)
        if callable(upsert_chunk):
            document = repository.get_document(chunk.document_id)
            upsert_chunk(
                chunk.chunk_id,
                vector,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source=document.source if document is not None else "",
                memory_type="semantic",
            )
        repository.set_chunk_vector_status(chunk.chunk_id, "indexed")
        chunks_done += 1
    memories_done = 0
    for item in manager.document_store.list(include_expired=True):
        item.embedding = manager.embedding.embed_item(
            item.content,
            payload=getattr(item, "payload", None),
            modality=getattr(item, "modality", None),
        )
        manager.document_store.upsert(item)
        manager.vector_store.upsert(item)
        memories_done += 1
    repository.set_embedding_lock(current.model, current.dimension)
    return {
        "model": current.model,
        "dimension": current.dimension,
        "chunks": chunks_done,
        "memories": memories_done,
    }


__all__ = [
    "EmbeddingIdentity",
    "EmbeddingLockMismatch",
    "apply_embedding_lock",
    "embedding_model_name",
    "inspect_embedding_lock",
    "live_vector_dimension",
    "mismatch_from_exception",
    "rebuild_vector_projection",
    "resolve_embedding_identity",
]
