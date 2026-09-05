"""Qdrant vector database adapter."""

from __future__ import annotations

import uuid
from typing import Any

from ..base import MemoryItem, MemoryType
from .vector import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector database adapter.

    ``client`` can be injected (a ``qdrant_client.QdrantClient`` compatible
    object) for tests.  The collection is created lazily on the first upsert,
    allowing the embedding dimension to be discovered from the item.
    """

    def __init__(self, url: str | None = None, collection_name: str = "helloagents_memory", *, api_key: str | None = None, client: Any = None, dimension: int | None = None, namespace: str = "memory") -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("QdrantVectorStore requires qdrant-client") from exc
            client = QdrantClient(url=url, api_key=api_key) if url else QdrantClient(path=":memory:")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be non-empty")
        if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
            raise ValueError("dimension must be a positive integer")
        self.client, self.collection_name, self.dimension, self.namespace = client, collection_name, dimension, namespace
        self._ready = False

    def _ensure_collection(self, dimension: int) -> None:
        if self._ready:
            return
        try:
            from qdrant_client.models import Distance, VectorParams
            exists = self.client.collection_exists(collection_name=self.collection_name)
            if not exists:
                self.client.create_collection(collection_name=self.collection_name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
            elif self.dimension is not None and self.dimension != dimension:
                raise ValueError(f"Qdrant collection dimension mismatch: expected {self.dimension}, got {dimension}")
            else:
                get_collection = getattr(self.client, "get_collection", None)
                if callable(get_collection):
                    info = get_collection(collection_name=self.collection_name)
                    configured = getattr(getattr(info, "config", None), "params", None)
                    configured_size = getattr(configured, "size", None)
                    if configured_size is not None and int(configured_size) != dimension:
                        raise ValueError(f"Qdrant collection dimension mismatch: existing {configured_size}, got {dimension}")
            self.dimension = dimension
            self._ready = True
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"unable to initialize Qdrant collection: {exc}") from exc

    def upsert(self, item: MemoryItem) -> None:
        if not item.embedding:
            return
        self._ensure_collection(len(item.embedding))
        from qdrant_client.models import PointStruct
        payload = item.to_dict()
        payload["namespace"] = self.namespace
        self.client.upsert(collection_name=self.collection_name, points=[PointStruct(id=self._point_id(item.id), vector=item.embedding, payload=payload)])

    def delete(self, item_id: str) -> bool:
        if not self._ready:
            return False
        from qdrant_client.models import PointIdsList
        self.client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=[self._point_id(item_id)]))
        return True

    def search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not self._ready:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        conditions = [FieldCondition(key="namespace", match=MatchValue(value=self.namespace))]
        if memory_type is not None:
            conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=MemoryType(memory_type).value)))
        query_filter = Filter(must=conditions)
        try:
            points = self.client.search(collection_name=self.collection_name, query_vector=vector, query_filter=query_filter, limit=limit)
        except AttributeError:
            points = self.client.query_points(collection_name=self.collection_name, query=vector, query_filter=query_filter, limit=limit).points
        return [(str((point.payload or {}).get("id", point.id)), float(point.score)) for point in points]

    @staticmethod
    def _point_id(item_id: str) -> str:
        """Qdrant accepts UUIDs/integers only; preserve arbitrary app IDs in payload."""
        try:
            uuid.UUID(item_id)
            return item_id
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"helloagents-memory:{item_id}"))

    def clear(self) -> None:
        if self._ready:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ready = False


__all__ = ["QdrantVectorStore"]
