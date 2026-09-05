"""Core data structures and the shared BaseMemory implementation.

The models intentionally have no dependency on a particular storage vendor: a
``MemoryItem`` can move between the SQLite, Qdrant and custom backends without
changing application code.  ``BaseMemory`` provides the common CRUD and
semantic-search operations shared by the four memory types.
"""

from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from .embedding import BaseEmbedding, TFIDFEmbedding

if TYPE_CHECKING:
    from .storage import BaseDocumentStore, BaseVectorStore


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERCEPTUAL = "perceptual"


DEFAULT_MEMORY_DB_FILENAME = "memory.sqlite3"


def default_sqlite_path() -> str:
    """Default persistent SQLite path for the agent-facing memory tools.

    The ``MEMORY_DB_PATH`` environment variable overrides the project-relative
    ``memory.sqlite3`` default.  Callers that inject their own manager are not
    affected.
    """
    return os.getenv("MEMORY_DB_PATH") or DEFAULT_MEMORY_DB_FILENAME


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("datetime values must be datetime, ISO string, or None")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


@dataclass
class MemoryItem:
    """The canonical memory record.

    ``content`` is the searchable textual representation.  ``payload`` is
    optional multimodal data (for example image bytes or a URI) and is kept
    separate so vector stores only need to index text.
    """

    content: str
    memory_type: MemoryType | str = MemoryType.WORKING
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    timestamp: datetime | None = None
    embedding: list[float] | None = None
    payload: Any = None
    modality: str | None = None
    relations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            self.content = str(self.content)
        self.memory_type = MemoryType(self.memory_type)
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("memory id must be a non-empty string")
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata or {})
        if isinstance(self.importance, bool) or not isinstance(self.importance, (int, float)):
            raise TypeError("importance must be a number")
        if not math.isfinite(float(self.importance)) or not 0 <= float(self.importance) <= 1:
            raise ValueError("importance must be between 0 and 1")
        self.importance = float(self.importance)
        self.created_at = ensure_datetime(self.created_at) or utc_now()
        self.updated_at = ensure_datetime(self.updated_at) or self.created_at
        self.expires_at = ensure_datetime(self.expires_at)
        self.timestamp = ensure_datetime(self.timestamp)
        if self.modality is not None and (not isinstance(self.modality, str) or not self.modality.strip()):
            raise ValueError("modality must be a non-empty string when provided")
        if self.embedding is not None:
            if isinstance(self.embedding, (str, bytes)):
                raise TypeError("embedding must be an iterable of numbers")
            try:
                self.embedding = [float(v) for v in self.embedding]
            except (TypeError, ValueError) as exc:
                raise TypeError("embedding must be an iterable of numbers") from exc
            if any(not math.isfinite(value) for value in self.embedding):
                raise ValueError("embedding values must be finite")
            if not self.embedding:
                raise ValueError("embedding must not be empty")
        if not isinstance(self.relations, list):
            self.relations = list(self.relations)
        if any(not isinstance(relation, Mapping) for relation in self.relations):
            raise TypeError("relations must contain mappings")

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= utc_now()

    @property
    def event_time(self) -> datetime:
        return self.timestamp or self.created_at

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": _json_safe(self.metadata),
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "embedding": self.embedding,
            "modality": self.modality,
            "relations": _json_safe(self.relations),
        }
        if include_payload:
            result["payload"] = _json_safe(self.payload)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryItem":
        data = dict(value)
        data["payload"] = _json_restore(data.get("payload"))
        return cls(**data)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _json_restore(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__bytes__"}:
        try:
            return base64.b64decode(value["__bytes__"])
        except Exception:
            return value
    if isinstance(value, dict):
        return {key: _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    return value


@dataclass(frozen=True)
class MemorySearchResult:
    item: MemoryItem
    score: float

    def to_dict(self) -> dict[str, Any]:
        data = self.item.to_dict()
        data["score"] = self.score
        return data


@dataclass
class MemoryConfig:
    """Runtime settings.

    Defaults are deliberately local and dependency-free.  Set ``sqlite_path``
    to a filename for persistence; ``":memory:"`` is useful for tests and
    short-lived agents.
    """

    sqlite_path: str | Path = ":memory:"
    default_ttl_seconds: float | None = 3600.0
    working_memory_capacity: int = 100
    search_limit: int = 10
    similarity_threshold: float = 0.0
    embedding_dimension: int = 384
    qdrant_url: str | None = None
    qdrant_collection: str = "helloagents_memory"
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.working_memory_capacity, bool) or not isinstance(self.working_memory_capacity, int) or self.working_memory_capacity < 1:
            raise ValueError("working_memory_capacity must be a positive integer")
        if isinstance(self.search_limit, bool) or not isinstance(self.search_limit, int) or self.search_limit < 1:
            raise ValueError("search_limit must be a positive integer")
        for name in ("similarity_threshold",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int) or self.embedding_dimension < 1:
            raise ValueError("embedding_dimension must be a positive integer")
        if self.default_ttl_seconds is not None:
            if isinstance(self.default_ttl_seconds, bool) or not isinstance(self.default_ttl_seconds, (int, float)) or not math.isfinite(float(self.default_ttl_seconds)) or self.default_ttl_seconds <= 0:
                raise ValueError("default_ttl_seconds must be positive or None")
        if not isinstance(self.qdrant_collection, str) or not self.qdrant_collection.strip():
            raise ValueError("qdrant_collection must be non-empty")
        if not isinstance(self.extra, dict):
            self.extra = dict(self.extra)

    @classmethod
    def from_env(cls, prefix: str = "HELLOAGENTS_MEMORY_") -> "MemoryConfig":
        """Build configuration from environment variables.

        Supported names mirror the dataclass fields, e.g.
        ``HELLOAGENTS_MEMORY_SQLITE_PATH`` and ``..._QDRANT_URL``.
        """
        values: dict[str, object] = {}
        for field_name in cls.__dataclass_fields__:
            key = f"{prefix}{field_name.upper()}"
            raw = os.getenv(key)
            if raw is None:
                continue
            if field_name in {"working_memory_capacity", "search_limit", "embedding_dimension"}:
                values[field_name] = int(raw)
            elif field_name in {"default_ttl_seconds", "similarity_threshold"}:
                values[field_name] = None if raw.casefold() == "none" else float(raw)
            elif field_name == "extra":
                continue
            else:
                values[field_name] = raw
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in self.__dict__.items()
        }


class BaseMemory:
    """Common CRUD and semantic-search operations for one memory type."""

    memory_type: MemoryType

    def __init__(self, *, document_store: BaseDocumentStore | None = None, vector_store: BaseVectorStore | None = None, embedding: BaseEmbedding | None = None, config: MemoryConfig | None = None, memory_type: MemoryType | str | None = None) -> None:
        # Storage backends import this module for the data structures, so the
        # defaults are resolved lazily to avoid an import cycle.
        from .storage import InMemoryVectorStore, SQLiteDocumentStore

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
        """List records for this memory type without touching the vector index.

        Index rebuilding is the manager's responsibility on startup; a read
        must not perform O(n) vector upserts on every call.
        """
        return self.document_store.list(memory_type=self.memory_type, include_expired=include_expired)

    def clear(self) -> int:
        count = 0
        for item in self.document_store.list(memory_type=self.memory_type, include_expired=True):
            self.vector_store.delete(item.id)
            if self.document_store.delete(item.id):
                count += 1
        return count


__all__ = [
    "DEFAULT_MEMORY_DB_FILENAME",
    "BaseMemory",
    "MemoryConfig",
    "MemoryItem",
    "MemorySearchResult",
    "MemoryType",
    "default_sqlite_path",
    "ensure_datetime",
    "utc_now",
]
