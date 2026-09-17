"""Qdrant vector database adapter."""

from __future__ import annotations

import uuid
from typing import Any

from constants import MEMORY_QDRANT_COLLECTION

from ..base import MemoryItem, MemoryType
from .vector import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector database adapter.

    ``client`` can be injected (a ``qdrant_client.QdrantClient`` compatible
    object) for tests.  The collection is created lazily on the first upsert,
    allowing the embedding dimension to be discovered from the item.
    """

    def __init__(self, url: str | None = None, collection_name: str = MEMORY_QDRANT_COLLECTION, *, api_key: str | None = None, client: Any = None, dimension: int | None = None, namespace: str = "memory") -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("QdrantVectorStore requires qdrant-client") from exc
            if url:
                # 本机回环地址绝不走系统代理：httpx 默认信任环境/注册表代理，
                # 用户开着 Clash 等代理时 127.0.0.1 请求会被转发到代理端口，
                # 代理对回环目标返回 502。本地端点显式关掉 trust_env。
                from urllib.parse import urlparse

                parsed = urlparse(url)
                if parsed.hostname in ("127.0.0.1", "localhost", "::1"):
                    client = QdrantClient(url=url, api_key=api_key, trust_env=False)
                else:
                    client = QdrantClient(url=url, api_key=api_key)
            else:
                client = QdrantClient(path=":memory:")
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
        payload = self._light_payload(item)
        payload["namespace"] = self.namespace
        self.client.upsert(collection_name=self.collection_name, points=[PointStruct(id=self._point_id(item.id), vector=item.embedding, payload=payload)])

    def upsert_chunk(self, chunk_id: str, vector: list[float], *, document_id: str = "", chunk_index: int = 0, source: str = "", memory_type: str = "semantic") -> None:
        """Write one chunk vector with the narrow chunk payload (方案 2.2).

        Used by the projection reindex path: the ingest path still goes through
        :meth:`upsert`/``MemoryManager.add``, and both resolve to the same
        ``chunk_id`` so the SQLite look-back is identical either way.
        """
        if not vector:
            return
        self._ensure_collection(len(vector))
        from qdrant_client.models import PointStruct
        payload: dict[str, Any] = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "source": source,
            "memory_type": memory_type,
            # namespace 由存储层补，与 upsert() 同一口径：search() 强制按它过滤，
            # 缺了这个键的点会永远检索不到。
            "namespace": self.namespace,
        }
        self.client.upsert(collection_name=self.collection_name, points=[PointStruct(id=self._point_id(chunk_id), vector=vector, payload=payload)])

    @staticmethod
    def _light_payload(item: MemoryItem) -> dict[str, Any]:
        """Projection payload: only what回查 and filtering need.

        The chunk text and metadata already live in SQLite (真值源), and the
        1024-float embedding was the bulk of every point; keeping a full copy in
        Qdrant would be a second, drift-prone truth.
        """
        full = item.to_dict()
        return {key: full.get(key) for key in ("id", "memory_type", "created_at", "expires_at")}

    def delete(self, item_id: str) -> bool:
        if not self._ready:
            return False
        from qdrant_client.models import PointIdsList
        self.client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=[self._point_id(item_id)]))
        return True

    def _ensure_ready_for_read(self) -> bool:
        """Attach to an existing collection without creating one.

        ``_ready`` is only set by writes, so a process that merely reads (the
        reconcile endpoint, a UI session) would otherwise report an empty
        collection as "no vectors".
        """

        if self._ready:
            return True
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                return False
        except Exception:  # noqa: BLE001 - 探测失败按「读不到」处理，由调用方降级
            return False
        self._ready = True
        return True

    def list_ids(self, *, limit: int = 10000) -> list[str]:
        """Every app-level id in the collection (reconcile needs the full set)."""

        if not self._ensure_ready_for_read():
            return []
        scroll = getattr(self.client, "scroll", None)
        if not callable(scroll):
            return []
        points, _ = scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        ids: list[str] = []
        for point in points:
            payload = point.payload or {}
            ids.append(str(payload.get("chunk_id") or payload.get("id") or point.id))
        return ids

    def search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not self._ensure_ready_for_read():
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
        hits: list[tuple[str, float]] = []
        for point in points:
            payload = point.payload or {}
            # chunk 点优先用 chunk_id；老点沿用 id；两者皆无才退回 Qdrant 点 id。
            hits.append((str(payload.get("chunk_id") or payload.get("id") or point.id), float(point.score)))
        return hits

    @staticmethod
    def _point_id(item_id: str) -> str:
        """Qdrant accepts UUIDs/integers only; preserve arbitrary app IDs in payload."""
        try:
            uuid.UUID(item_id)
            return item_id
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"helloagents-memory:{item_id}"))


__all__ = ["QdrantVectorStore"]
