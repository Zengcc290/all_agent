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
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MEMORY_DB_FILENAME,
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
)

# config/services.toml 是"所有外部 API 调用"的集中配置（嵌入/搜索/Qdrant/Neo4j）。
# core 不依赖 memory，此处导入无循环；只用于 from_config 的未设置字段兜底。
from core.services_config import ServicesConfig, load_services_config

from .embedding import (
    APIEmbedding,
    BaseEmbedding,
    HashEmbedding,
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
    ``memory.sqlite3`` default (这是唯一保留的路径覆盖入口，单点收拢).
    Callers that inject their own manager are not affected.
    """
    return os.getenv("MEMORY_DB_PATH") or str(Path(__file__).resolve().parent.parent / DEFAULT_MEMORY_DB_FILENAME)


def make_default_embedding(config: MemoryConfig | None = None) -> BaseEmbedding:
    """Build the default embedding service from a (possibly implicit) config.

    Uses ``MemoryConfig.from_config()`` when no config is supplied.  Selection:

    1. ``embedding_provider == "hash"`` forces the offline fallback;
    2. otherwise a configured cloud endpoint (``[embedding].base_url`` +
       ``api_key`` in config/services.toml) activates the OpenAI-compatible
       :class:`~memory.embedding.APIEmbedding` (text and image/text VL input);
    3. without a complete cloud configuration the deterministic offline
       :class:`~memory.embedding.HashEmbedding` is used instead of raising, so
       the memory layer, the agent tools and the web app stay usable offline.

    Cloud vector spaces are not interchangeable; changing the model requires
    rebuilding the projections (维度唯一开关是 ``[embedding].model``，见
    ``QdrantVectorStore`` 的维度守卫与 ``scripts/migrate_to_cloud.py``).
    """
    config = config if config is not None else MemoryConfig.from_config()
    provider = (config.embedding_provider or MEMORY_EMBEDDING_PROVIDER_DEFAULT).strip().casefold()

    if provider == "hash":
        return HashEmbedding(dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION)

    base_url = (config.embedding_base_url or "").strip()
    api_key = str(config.embedding_api_key or "").strip()
    if base_url and api_key:
        return APIEmbedding(
            api_key=api_key,
            model=config.embedding_model,
            base_url=base_url,
            dimension=config.embedding_dimension,
            timeout=config.embedding_timeout,
            batch_size=config.embedding_batch_size,
        )
    # provider=openai 显式要求云端却没配全：按约定退回离线兜底并让
    # /api/health 的 embedding_mode 如实显示 hash，而不是启动失败。
    return HashEmbedding(dimension=config.embedding_dimension or MEMORY_EMBEDDING_DIMENSION)


def _merge_services_into(values: dict[str, object], services: ServicesConfig) -> None:
    """Fill ``values`` with fields left unset, from services.toml.

    ``from_config`` 由此构建 ``values``；这个 helper 只在字段缺席时补值，
    所以显式构造参数永远优先于共享的 services 配置文件。
    """

    embedding = services.embedding
    merged = (
        ("embedding_provider", embedding.provider),
        ("embedding_base_url", embedding.base_url),
        ("embedding_model", embedding.model),
        ("embedding_api_key", embedding.api_key),
        ("embedding_dimension", embedding.dimension),
        ("embedding_batch_size", embedding.batch_size),
        ("embedding_timeout", embedding.timeout),
        ("qdrant_url", services.qdrant.url),
        ("qdrant_collection", services.qdrant.collection),
        ("qdrant_api_key", services.qdrant.api_key),
        ("neo4j_uri", services.neo4j.uri),
        ("neo4j_username", services.neo4j.username),
        ("neo4j_password", services.neo4j.password),
        ("proxy_url", services.proxy.url),
    )
    for field_name, value in merged:
        if value is None or field_name in values:
            continue
        values[field_name] = value


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
    # 远端嵌入的期望维度。留空（None）= 不预设，首次响应里自动识别；离线
    # 兜底不受影响，仍用 MEMORY_EMBEDDING_DIMENSION（1024）。
    embedding_dimension: int | None = MEMORY_EMBEDDING_DIMENSION_REMOTE
    # 嵌入提供方；auto = 配置了 [embedding] 端点+密钥就走云端，否则离线兜底。
    embedding_provider: str = MEMORY_EMBEDDING_PROVIDER_DEFAULT
    # 云端嵌入（OpenAI 兼容 /embeddings，支持文本与图文 VL 输入）。
    # 端点/密钥/模型统一来自 config/services.toml 的 [embedding] 段；
    # 留空 = 未配置云端，make_default_embedding 退回离线 HashEmbedding。
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str = ""
    embedding_api_key: str | None = None
    embedding_timeout: float = MEMORY_EMBEDDING_TIMEOUT
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    # ---- 连接开关：出厂 None = **不连接**，走内存回退（Qdrant/Neo4j 都不起也能跑）----
    # 端点/端口的单一事实来源是 constants.py 的「连接与端点」小节
    # （DEFAULT_QDRANT_URL / DEFAULT_NEO4J_URI / DEFAULT_*_PORT）。
    # 要连真服务就在 config/services.toml 集中配置：
    #   [qdrant]  url = "https://xxxx.qdrant.tech"（+ api_key / collection）
    #   [neo4j]   uri = "bolt+s://xxxx.databases.neo4j.io"（+ username/password）
    # 选型发生在 memory/manager.py：qdrant_url 非空才建 QdrantVectorStore。
    qdrant_url: str | None = None
    qdrant_collection: str = MEMORY_QDRANT_COLLECTION
    qdrant_api_key: str | None = None
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    #: 本地转发代理（http://host:port）。显式配置覆盖默认值；云端 Qdrant/Neo4j
    #: 未配置时由 MemoryManager 使用 constants.DEFAULT_PROXY_URL（7890）。
    proxy_url: str | None = None
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
        # embedding_base_url 允许空串 = 未配置云端（离线兜底）；model 必填。
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("embedding_model must be a non-empty string")
        if not isinstance(self.embedding_base_url, str):
            raise TypeError("embedding_base_url must be a string (empty = cloud not configured)")
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
        if not isinstance(self.qdrant_api_key, (str, type(None))) or (isinstance(self.qdrant_api_key, str) and not self.qdrant_api_key.strip()):
            raise ValueError("qdrant_api_key must be a non-empty string or None")
        if not isinstance(self.proxy_url, (str, type(None))) or (isinstance(self.proxy_url, str) and not self.proxy_url.strip()):
            raise ValueError("proxy_url must be a non-empty string or None")
        if not isinstance(self.extra, dict):
            self.extra = dict(self.extra)

    @classmethod
    def from_config(cls) -> MemoryConfig:
        """Build configuration from ``config/services.toml``.

        只有两层优先级：**显式构造参数 > services.toml**。历史上的
        ``HELLOAGENTS_MEMORY_*`` 环境变量层与 ``.env`` 装载已删除（配置只认
        config/ 与 constants.py）。services.toml 是"所有外部 API 调用"的集中
        配置，见 core/services_config.py；补进来的值同样要过 ``__post_init__``
        校验。
        """
        values: dict[str, object] = {}
        _merge_services_into(values, load_services_config())
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

        ``BaseEmbedding.embed_item`` is text-only; cloud VL models
        (``APIEmbedding``，模型名带 ``vl``) override it to embed images and
        fused image+text.  Injected embeddings that are not ``BaseEmbedding``
        subclasses fall back to plain ``embed``.
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
