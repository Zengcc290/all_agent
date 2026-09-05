"""Base implementation shared by the four memory types."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping
from uuid import uuid4

from .config import MemoryConfig
from .embeddings import BaseEmbedding
from .models import MemoryItem, MemorySearchResult, MemoryType, utc_now
from .storage import BaseDocumentStore, BaseVectorStore, InMemoryVectorStore, SQLiteDocumentStore
from .embeddings import TFIDFEmbedding


class BaseMemory:
    """Common CRUD and semantic-search operations for one memory type."""

    memory_type: MemoryType

    def __init__(self, *, document_store: BaseDocumentStore | None = None, vector_store: BaseVectorStore | None = None, embedding: BaseEmbedding | None = None, config: MemoryConfig | None = None, memory_type: MemoryType | str | None = None) -> None:
        self.config = config if config is not None else MemoryConfig()
        self.document_store = (
            document_store
            if document_store is not None
            else SQLiteDocumentStore(self.config.sqlite_path)
        )
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()
        self.embedding = embedding if embedding is not None else TFIDFEmbedding(self.config.embedding_dimension)
        self.memory_type = MemoryType(memory_type or self.memory_type)

    def _validate_embedding_dimension(self, vector: list[float]) -> None:
        expected = getattr(self.embedding, "dimension", 0)
        if expected and len(vector) != expected:
            raise ValueError(f"embedding dimension {len(vector)} does not match expected dimension {expected}")

    def add(
        self,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.5,
        ttl_seconds: float | None = None,
        ttl: float | None = None,
        expires_at: datetime | str | None = None,
        timestamp: datetime | str | None = None,
        item_id: str | None = None,
        payload: Any = None,
        modality: str | None = None,
        relations: list[dict[str, Any]] | None = None,
    ) -> MemoryItem:
        if not isinstance(content, str):
            content = str(content)
        if ttl is not None:
            if ttl_seconds is not None:
                raise ValueError("provide either ttl or ttl_seconds, not both")
            ttl_seconds = ttl
        if expires_at is not None and ttl_seconds is not None:
            raise ValueError("provide either ttl_seconds or expires_at, not both")
        if ttl_seconds is None and self.memory_type == MemoryType.WORKING:
            ttl_seconds = self.config.default_ttl_seconds
        if ttl_seconds is not None:
            if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
                raise ValueError("ttl_seconds must be positive")
            expires_at = utc_now() + timedelta(seconds=float(ttl_seconds))
        existing = self.document_store.get(item_id) if item_id else None
        if existing is not None and existing.memory_type != self.memory_type:
            raise ValueError(f"item id already belongs to {existing.memory_type.value} memory")
        item = MemoryItem(
            id=item_id if item_id is not None else str(uuid4()),
            content=content,
            memory_type=self.memory_type,
            metadata=dict(metadata or {}),
            importance=importance,
            expires_at=expires_at,
            timestamp=timestamp,
            payload=payload,
            modality=modality,
            relations=list(relations or []),
        )
        item.embedding = self.embedding.embed(item.content)
        self._validate_embedding_dimension(item.embedding)
        # Persist first so a vector backend failure cannot create an index entry
        # for a record that does not exist in the source of truth.
        self.document_store.upsert(item)
        try:
            self.vector_store.upsert(item)
        except Exception:
            if existing is None:
                self.document_store.delete(item.id)
            else:
                self.document_store.upsert(existing)
                if existing.embedding is not None:
                    self.vector_store.upsert(existing)
                else:
                    self.vector_store.delete(existing.id)
            raise
        return item

    remember = add
    store = add

    def get(self, item_id: str) -> MemoryItem | None:
        item = self.document_store.get(item_id)
        if item is not None and item.is_expired and item.memory_type == self.memory_type:
            self.delete(item_id)
            return None
        return item if item is not None and item.memory_type == self.memory_type else None

    def delete(self, item_id: str) -> bool:
        item = self.document_store.get(item_id)
        if item is None or item.memory_type != self.memory_type:
            return False
        self.vector_store.delete(item_id)
        return self.document_store.delete(item_id)

    remove = delete
    retrieve = get

    def search(self, query: str, *, limit: int | None = None, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[MemorySearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return []
        limit = self.config.search_limit if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        threshold = self.config.similarity_threshold if threshold is None else threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
            raise ValueError("threshold must be between 0 and 1")
        vector = self.embedding.embed(query)
        # Ask for more than the requested number because metadata/expiry can
        # remove candidates after the vector index has ranked them.
        self._validate_embedding_dimension(vector)
        candidates = self.vector_store.search(vector, limit=max(limit * 4, limit), memory_type=self.memory_type)
        results: list[MemorySearchResult] = []
        for item_id, score in candidates:
            item = self.get(item_id)
            # A zero vector (or an unrelated hashed TF-IDF vector) is not a
            # meaningful match even when callers leave threshold at its
            # permissive default of 0.
            if item is None or score < threshold or (query.strip() and score <= 0):
                continue
            if metadata and any(item.metadata.get(key) != value for key, value in metadata.items()):
                continue
            results.append(MemorySearchResult(item=item, score=score))
            if len(results) >= limit:
                break
        return results

    query = search

    def list(self, *, include_expired: bool = False) -> list[MemoryItem]:
        items = self.document_store.list(memory_type=self.memory_type, include_expired=include_expired)
        if not include_expired:
            for item in items:
                self.vector_store.upsert(item)
        return items

    def clear(self) -> int:
        count = 0
        for item in self.document_store.list(memory_type=self.memory_type, include_expired=True):
            self.vector_store.delete(item.id)
            if self.document_store.delete(item.id):
                count += 1
        return count


__all__ = ["BaseMemory"]
