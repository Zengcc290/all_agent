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
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from constants import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMINI_EMBEDDING_BASE_URL,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_MEMORY_DB_FILENAME,
    GEMINI_API_KEY_ENV,
    GEMINI_EMBEDDING_HOST,
    MEMORY_DEFAULT_TTL_SECONDS,
    MEMORY_EMBEDDING_DIMENSION,
    MEMORY_EMBEDDING_DIMENSION_REMOTE,
    MEMORY_EMBEDDING_PROVIDER_DEFAULT,
    MEMORY_EMBEDDING_PROVIDERS,
    MEMORY_EMBEDDING_TIMEOUT,
    MEMORY_QDRANT_COLLECTION,
    MEMORY_SEARCH_LIMIT,
    MEMORY_SIMILARITY_THRESHOLD,
    MEMORY_SQLITE_DEFAULT,
    MEMORY_WORKING_CAPACITY,
    SILICONFLOW_API_KEY_ENV,
)

from .embedding import (
    APIEmbedding,
    BaseEmbedding,
    EmbedServerEmbedding,
    GeminiEmbedding,
    HashEmbedding,
    load_dotenv_once,
)

if TYPE_CHECKING:
    from .storage import BaseDocumentStore, BaseVectorStore


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERCEPTUAL = "perceptual"


def default_sqlite_path() -> str:
    """Default persistent SQLite path for the agent-facing memory tools.

    The ``MEMORY_DB_PATH`` environment variable overrides the project-relative
    ``memory.sqlite3`` default.  Callers that inject their own manager are not
    affected.
    """
    return os.getenv("MEMORY_DB_PATH") or DEFAULT_MEMORY_DB_FILENAME


def make_default_embedding(config: MemoryConfig | None = None) -> BaseEmbedding:
    """Build the default embedding service from a (possibly implicit) config.

    Uses ``MemoryConfig.from_env()`` when no config is supplied.  Selection:

    1. ``embedding_provider`` forces a backend when it is not ``"auto"``;
    2. otherwise a local embed gateway configured via ``EMBEDDING_BASE_URL`` (a
       forwarded-port service speaking the custom ``/embed`` protocol) wins;
    3. otherwise a key (``embedding_api_key`` / ``GEMINI_API_KEY`` /
       ``DASHSCOPE_API_KEY``) activates a remote API - Gemini's ``:embedContent``
       when the endpoint host or model says Gemini, the OpenAI-compatible
       ``/embeddings`` shape otherwise;
    4. without any key the deterministic offline
       :class:`~memory.embedding.HashEmbedding` is used instead of raising, so the
       memory layer, the agent tools and the web app stay usable offline.

    These vector spaces are not interchangeable; changing the provider requires
    re-indexing stored items (``scripts/reindex_embeddings.py``).
    """
    config = config if config is not None else MemoryConfig.from_env()
    load_dotenv_once()
    provider = (config.embedding_provider or MEMORY_EMBEDDING_PROVIDER_DEFAULT).strip().casefold()

    if provider == "hash":
        return HashEmbedding(dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION)

    # 转发网关与离线兜底沿用 1024 维默认值：网关那头是固定的 qwen-embed 模型。
    if provider == "gateway":
        return EmbedServerEmbedding(
            base_url=config.embedding_base_url,
            dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION,
            timeout=config.embedding_timeout,
            batch_size=config.embedding_batch_size,
        )
    server_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip()
    if provider == "auto" and server_url:
        return EmbedServerEmbedding(
            base_url=server_url,
            dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION,
            timeout=config.embedding_timeout,
            batch_size=config.embedding_batch_size,
        )

    use_gemini = provider == "gemini" or (
        provider == "auto"
        and (
            GEMINI_EMBEDDING_HOST in config.embedding_base_url
            or config.embedding_model.startswith("gemini-")
        )
    )
    if use_gemini:
        api_key = config.embedding_api_key or os.getenv(GEMINI_API_KEY_ENV)
        if not api_key or not str(api_key).strip():
            return HashEmbedding(dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION)
        # 留下的类默认值（qwen 的模型名与本机网关地址）说明调用方没填，
        # 这里补成 Gemini 的值，省得只配一个 key 还要把端点抄一遍。
        model = config.embedding_model
        if model == DEFAULT_EMBEDDING_MODEL:
            model = DEFAULT_GEMINI_EMBEDDING_MODEL
        base_url = config.embedding_base_url
        if base_url == DEFAULT_EMBEDDING_BASE_URL:
            base_url = DEFAULT_GEMINI_EMBEDDING_BASE_URL
        return GeminiEmbedding(
            api_key=api_key,
            model=model,
            base_url=base_url,
            dimension=config.embedding_dimension,
            timeout=config.embedding_timeout,
        )

    api_key = config.embedding_api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv(SILICONFLOW_API_KEY_ENV)
    if not api_key or not str(api_key).strip():
        return HashEmbedding(dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION)
    return APIEmbedding(
        api_key=api_key,
        model=config.embedding_model,
        base_url=config.embedding_base_url,
        dimension=config.embedding_dimension,
        timeout=config.embedding_timeout,
        batch_size=config.embedding_batch_size,
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        # Python 3.11+ 的 fromisoformat 直接接受末尾的 "Z"。
        result = datetime.fromisoformat(value)
    else:
        raise TypeError("datetime values must be datetime, ISO string, or None")
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


@dataclass
class MemoryItem:
    """The canonical memory record.

    ``content`` is the searchable textual representation.  ``payload`` is
    optional multimodal data (for example image bytes or a URI) and is kept
    separate so vector stores only need to index text.
    """

    content: str
    memory_type: MemoryType | str = MemoryType.WORKING
    id: str = field(default_factory=lambda: str(uuid4()))
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

    def to_dict(self) -> dict[str, Any]:
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
            "payload": _json_safe(self.payload),
        }
        return result


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
        except Exception:  # noqa: BLE001 - 无法解码的负载按原样返回，读取不应失败
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

    sqlite_path: str | Path = MEMORY_SQLITE_DEFAULT
    default_ttl_seconds: float | None = MEMORY_DEFAULT_TTL_SECONDS
    working_memory_capacity: int = MEMORY_WORKING_CAPACITY
    search_limit: int = MEMORY_SEARCH_LIMIT
    similarity_threshold: float = MEMORY_SIMILARITY_THRESHOLD
    # 远端嵌入的期望维度。留空（None）= 不预设，首次响应里自动识别；转发网关与
    # 离线兜底不受影响，仍用 MEMORY_EMBEDDING_DIMENSION（1024）。
    embedding_dimension: int | None = MEMORY_EMBEDDING_DIMENSION_REMOTE
    # 嵌入提供方；auto = 按 EMBEDDING_BASE_URL / 端点主机 / 模型名自动判定。
    embedding_provider: str = MEMORY_EMBEDDING_PROVIDER_DEFAULT
    # Embedding provider settings; an API key may also come from the
    # ``DASHSCOPE_API_KEY`` / ``GEMINI_API_KEY`` environment variables.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    embedding_api_key: str | None = None
    embedding_timeout: float = MEMORY_EMBEDDING_TIMEOUT
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    # ---- 连接开关：出厂 None = **不连接**，走内存回退（Qdrant/Neo4j 都不起也能跑）----
    # 端点/端口的单一事实来源是 constants.py 的「连接与端点」小节
    # （DEFAULT_QDRANT_URL / DEFAULT_NEO4J_URI / DEFAULT_*_PORT）。
    # 要连真服务就在 .env 里设（前缀由 from_env 的 prefix 决定）：
    #   HELLOAGENTS_MEMORY_QDRANT_URL=http://127.0.0.1:6333
    #   HELLOAGENTS_MEMORY_NEO4J_URI=bolt://127.0.0.1:7687  （+_USERNAME/_PASSWORD）
    # 选型发生在 memory/manager.py：qdrant_url 非空才建 QdrantVectorStore。
    qdrant_url: str | None = None
    qdrant_collection: str = MEMORY_QDRANT_COLLECTION
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
        if self.embedding_dimension is not None and (
            isinstance(self.embedding_dimension, bool)
            or not isinstance(self.embedding_dimension, int)
            or self.embedding_dimension < 1
        ):
            raise ValueError("embedding_dimension must be a positive integer or None (learn from the response)")
        if not isinstance(self.embedding_provider, str) or self.embedding_provider.strip().casefold() not in MEMORY_EMBEDDING_PROVIDERS:
            raise ValueError(f"embedding_provider must be one of {', '.join(MEMORY_EMBEDDING_PROVIDERS)}")
        self.embedding_provider = self.embedding_provider.strip().casefold()
        for name in ("embedding_model", "embedding_base_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.embedding_api_key, (str, type(None))) or (isinstance(self.embedding_api_key, str) and not self.embedding_api_key.strip()):
            raise ValueError("embedding_api_key must be a non-empty string or None")
        for name in ("embedding_timeout", "embedding_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.default_ttl_seconds is not None and (
            isinstance(self.default_ttl_seconds, bool)
            or not isinstance(self.default_ttl_seconds, (int, float))
            or not math.isfinite(float(self.default_ttl_seconds))
            or self.default_ttl_seconds <= 0
        ):
            raise ValueError("default_ttl_seconds must be positive or None")
        if not isinstance(self.qdrant_collection, str) or not self.qdrant_collection.strip():
            raise ValueError("qdrant_collection must be non-empty")
        if not isinstance(self.extra, dict):
            self.extra = dict(self.extra)

    @classmethod
    def from_env(cls, prefix: str = "HELLOAGENTS_MEMORY_") -> MemoryConfig:
        """Build configuration from environment variables.

        Supported names mirror the dataclass fields, e.g.
        ``HELLOAGENTS_MEMORY_SQLITE_PATH`` and ``..._QDRANT_URL``.  Numeric
        fields are coerced appropriately; ``embedding_api_key`` also falls
        back to the common ``DASHSCOPE_API_KEY`` variable.
        """
        values: dict[str, object] = {}
        load_dotenv_once()
        for field_name in cls.__dataclass_fields__:
            key = f"{prefix}{field_name.upper()}"
            raw = os.getenv(key)
            if raw is None or not raw.strip():
                # 空值等同于未设置：``VAR=`` 是 .env 里最常见的占位写法，
                # 直接透传会让 embedding_api_key / embedding_model /
                # embedding_base_url 这类「非空字符串」字段在 __post_init__
                # 里抛 ValueError，使整个 from_env 不可用。
                continue
            if field_name == "embedding_dimension":
                # auto / none / 0 = 不预设：维度由首次响应决定（换远端平台不用改这里）。
                values[field_name] = None if raw.casefold() in {"auto", "none", "0"} else int(raw)
            elif field_name in {"working_memory_capacity", "search_limit", "embedding_batch_size"}:
                values[field_name] = int(raw)
            elif field_name in {"default_ttl_seconds", "similarity_threshold", "embedding_timeout"}:
                values[field_name] = None if raw.casefold() == "none" else float(raw)
            elif field_name == "extra":
                continue
            else:
                values[field_name] = raw
        if not values.get("embedding_api_key"):
            # 同理：空的 DASHSCOPE_API_KEY 表示「没配 key」，而不是「配了空 key」。
            values["embedding_api_key"] = (os.getenv("DASHSCOPE_API_KEY") or "").strip() or None
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
        self.embedding = embedding if embedding is not None else make_default_embedding(self.config)
        self.memory_type = MemoryType(memory_type or self.memory_type)

    def _embed_item(self, content: str, *, payload: Any = None, modality: str | None = None) -> list[float]:
        """Embed one write, letting multimodal backends fold in ``payload``.

        ``BaseEmbedding.embed_item`` is text-only; ``GeminiEmbedding`` overrides it
        to embed images and fused image+text.  Injected embeddings that are not
        ``BaseEmbedding`` subclasses fall back to plain ``embed``.
        """
        embed_item = getattr(self.embedding, "embed_item", None)
        if callable(embed_item):
            return embed_item(content, payload=payload, modality=modality)
        return self.embedding.embed(content)

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
        expires_at: datetime | str | None = None,
        timestamp: datetime | str | None = None,
        item_id: str | None = None,
        payload: Any = None,
        modality: str | None = None,
        relations: list[dict[str, Any]] | None = None,
    ) -> MemoryItem:
        if not isinstance(content, str):
            content = str(content)
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
        item.embedding = self._embed_item(item.content, payload=payload, modality=modality)
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
            # A zero vector (or an unrelated embedded vector) is not a
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
    "make_default_embedding",
    "utc_now",
]
