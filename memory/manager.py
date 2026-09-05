"""Unified coordinator for the memory layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .config import MemoryConfig
from .embeddings import BaseEmbedding, TFIDFEmbedding
from .memory_types import EpisodicMemory, PerceptualMemory, SemanticMemory, WorkingMemory
from .models import MemoryItem, MemorySearchResult, MemoryType
from .storage import BaseDocumentStore, BaseVectorStore, InMemoryVectorStore, Neo4jGraphStore, QdrantVectorStore, SQLiteDocumentStore


class MemoryManager:
    """Single entry point that coordinates all four memory implementations."""

    def __init__(self, config: MemoryConfig | None = None, *, embedding: BaseEmbedding | None = None, embedding_service: BaseEmbedding | None = None, document_store: BaseDocumentStore | None = None, vector_store: BaseVectorStore | None = None, graph_store: Neo4jGraphStore | None = None) -> None:
        self.config = config if config is not None else MemoryConfig()
        if embedding is not None and embedding_service is not None:
            raise ValueError("provide either embedding or embedding_service, not both")
        self.embedding = (
            embedding
            if embedding is not None
            else embedding_service
            if embedding_service is not None
            else TFIDFEmbedding(self.config.embedding_dimension)
        )
        self.document_store = (
            document_store
            if document_store is not None
            else SQLiteDocumentStore(self.config.sqlite_path)
        )
        if vector_store is not None:
            self.vector_store = vector_store
        elif self.config.qdrant_url:
            self.vector_store = QdrantVectorStore(
                url=self.config.qdrant_url,
                collection_name=self.config.qdrant_collection,
                dimension=self.config.embedding_dimension,
            )
        else:
            self.vector_store = InMemoryVectorStore()
        self.graph_store = (
            graph_store
            if graph_store is not None
            else Neo4jGraphStore(
                self.config.neo4j_uri,
                self.config.neo4j_username,
                self.config.neo4j_password,
            )
        )
        common = {
            "document_store": self.document_store,
            "vector_store": self.vector_store,
            "embedding": self.embedding,
            "config": self.config,
        }
        self.working = WorkingMemory(capacity=self.config.working_memory_capacity, **common)
        self.episodic = EpisodicMemory(**common)
        self.semantic = SemanticMemory(graph_store=self.graph_store, **common)
        self.perceptual = PerceptualMemory(**common)
        self.memories: dict[MemoryType, Any] = {
            MemoryType.WORKING: self.working,
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.PERCEPTUAL: self.perceptual,
        }
        # Rebuild a local index when reopening a persistent SQLite store.
        for item in self.document_store.list():
            self.vector_store.upsert(item)

    def for_type(self, memory_type: MemoryType | str) -> Any:
        try:
            return self.memories[MemoryType(memory_type)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown memory type: {memory_type}") from exc

    def add(self, content: str, *, memory_type: MemoryType | str = MemoryType.WORKING, **kwargs: Any) -> MemoryItem:
        return self.for_type(memory_type).add(content, **kwargs)

    remember = add
    store = add

    def get(self, item_id: str, *, memory_type: MemoryType | str | None = None) -> MemoryItem | None:
        if memory_type is not None:
            return self.for_type(memory_type).get(item_id)
        item = self.document_store.get(item_id)
        if item is not None and item.is_expired:
            self.delete(item_id)
            return None
        return item

    def delete(self, item_id: str, *, memory_type: MemoryType | str | None = None) -> bool:
        if memory_type is not None:
            return self.for_type(memory_type).delete(item_id)
        item = self.document_store.get(item_id)
        if item is None:
            return False
        return self.for_type(item.memory_type).delete(item_id)

    remove = delete
    retrieve = get

    def search(self, query: str, *, memory_type: MemoryType | str | None = None, limit: int | None = None, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[MemorySearchResult]:
        if memory_type is not None:
            return self.for_type(memory_type).search(query, limit=limit, threshold=threshold, metadata=metadata)
        limit = self.config.search_limit if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        # Search each namespace, then merge by score. This keeps one vector
        # collection usable while preserving the caller's type filter option.
        per_type_limit = max(limit, 1)
        found: list[MemorySearchResult] = []
        for memory in self.memories.values():
            found.extend(memory.search(query, limit=per_type_limit, threshold=threshold, metadata=metadata))
        found.sort(key=lambda result: (-result.score, result.item.created_at), reverse=False)
        return found[:limit]

    query = search

    def cleanup_expired(self) -> int:
        removed = 0
        for item in self.document_store.list(include_expired=True):
            if item.is_expired and self.delete(item.id):
                removed += 1
        return removed

    def list(self, *, memory_type: MemoryType | str | None = None, include_expired: bool = False) -> list[MemoryItem]:
        if memory_type is not None:
            return self.for_type(memory_type).list(include_expired=include_expired)
        return self.document_store.list(include_expired=include_expired)

    def clear(self, *, memory_type: MemoryType | str | None = None) -> int:
        if memory_type is not None:
            return self.for_type(memory_type).clear()
        return sum(memory.clear() for memory in self.memories.values())

    def stats(self) -> dict[str, int]:
        return {memory_type.value: len(self.document_store.list(memory_type=memory_type)) for memory_type in MemoryType}

    def close(self) -> None:
        self.document_store.close()
        close_vector = getattr(self.vector_store, "close", None)
        if callable(close_vector):
            close_vector()
        close = getattr(self.graph_store, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = ["MemoryManager"]
