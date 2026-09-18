"""Qdrant vector database adapter."""

from __future__ import annotations

import uuid
from typing import Any

from constants import MEMORY_QDRANT_COLLECTION

from ..base import MemoryItem, MemoryType
from .vector import BaseVectorStore


def _vector_size_of(value: Any) -> int | None:
    """Pull ``size`` off VectorParams, a named-vector map, or a dict payload."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    size = getattr(value, "size", None)
    if size is not None:
        return int(size)
    if isinstance(value, dict):
        if "size" in value and value["size"] is not None:
            return int(value["size"])
        if value:
            return _vector_size_of(next(iter(value.values())))
    return None


def _collection_vector_size(info: Any) -> int | None:
    """Read the live collection vector size from Qdrant collection info.

    真实路径是 ``config.params.vectors``（``VectorParams`` 或 named-vector 字典）。
    测试替身仍可能把尺寸放在扁平的 ``config.params.size``。配置文件里的
    ``[embedding].dimension`` 不是集合现有维度，不能拿来比对。
    """

    if isinstance(info, dict):
        info = type("Info", (), info)()
    params = getattr(getattr(info, "config", None), "params", None)
    if isinstance(getattr(info, "config", None), dict):
        params = info.config.get("params")
    if params is None:
        return None
    if isinstance(params, dict):
        return _vector_size_of(params.get("vectors")) or _vector_size_of(params.get("size"))
    return _vector_size_of(getattr(params, "vectors", None)) or _vector_size_of(
        getattr(params, "size", None)
    )


def _is_dimension_error(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return "expected dim" in text or "vector dimension error" in text or "dimension mismatch" in text


def _dimension_mismatch_message(collection: str, existing: int, actual: int) -> str:
    """维度冲突的修复指引：维度只有一个开关——[embedding].model。"""

    return (
        f"Qdrant collection dimension mismatch: 集合 {collection!r} 是 {existing} 维，"
        f"但当前 embedding 模型输出 {actual} 维。维度不能单独改集合或配置："
        "要么把 config/services.toml 的 [embedding].model 换回原模型；"
        "要么确认重建向量投影（SQLite 是真值，可全量重灌）。"
        "入库/重索引接口在确认后会 recreate 集合并重灌；命令行等价于 "
        "python scripts/migrate_to_cloud.py --recreate-collection 与 "
        "python scripts/reindex_embeddings.py。"
    )


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector database adapter.

    ``client`` can be injected (a ``qdrant_client.QdrantClient`` compatible
    object) for tests.  The collection is created lazily on the first upsert,
    allowing the embedding dimension to be discovered from the item.
    """

    def __init__(self, url: str | None = None, collection_name: str = MEMORY_QDRANT_COLLECTION, *, api_key: str | None = None, client: Any = None, dimension: int | None = None, namespace: str = "memory", proxy_url: str | None = None) -> None:
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
                # 云端端点按 [proxy] 配置显式走本地转发代理（proxy kwarg 会
                # 透传到 httpx.Client）。
                from urllib.parse import urlparse

                parsed = urlparse(url)
                if parsed.hostname in ("127.0.0.1", "localhost", "::1"):
                    client = QdrantClient(url=url, api_key=api_key, trust_env=False)
                elif proxy_url:
                    client = QdrantClient(url=url, api_key=api_key, proxy=proxy_url)
                else:
                    # MemoryManager 已对云端填上默认 7890；走到这里说明调用方
                    # 明确不要代理。仍不读系统环境代理，避免畸形 NO_PROXY。
                    client = QdrantClient(url=url, api_key=api_key, trust_env=False)
            else:
                client = QdrantClient(path=":memory:")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be non-empty")
        if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
            raise ValueError("dimension must be a positive integer")
        self.client, self.collection_name, self.dimension, self.namespace = client, collection_name, dimension, namespace
        self._ready = False

    def _live_collection_size(self) -> int | None:
        get_collection = getattr(self.client, "get_collection", None)
        if not callable(get_collection):
            return None
        try:
            return _collection_vector_size(get_collection(collection_name=self.collection_name))
        except Exception:  # noqa: BLE001 - 读不到尺寸时由写入路径再核对
            return None

    def _raise_dimension_mismatch(self, existing: int | None, actual: int, cause: BaseException | None = None) -> None:
        message = _dimension_mismatch_message(self.collection_name, int(existing or 0), actual)
        if cause is None:
            raise ValueError(message)
        raise ValueError(message) from cause

    def _ensure_collection(self, dimension: int) -> None:
        existing = self._live_collection_size() if self._ready else None
        if self._ready and (existing is None or existing == dimension):
            self.dimension = dimension
            return
        try:
            from qdrant_client.models import Distance, VectorParams
            exists = self.client.collection_exists(collection_name=self.collection_name)
            if not exists:
                self.client.create_collection(collection_name=self.collection_name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
            else:
                # 只跟集合的真实尺寸比对。self.dimension 可能来自配置护栏，
                # 把它当成「集合已有维度」会在换模型后误报 1024/4096。
                if existing is None:
                    existing = self._live_collection_size()
                if existing is not None and existing != dimension:
                    self._raise_dimension_mismatch(existing, dimension)
            self.dimension = dimension
            self._ensure_payload_indexes()
            self._ready = True
        except ValueError:
            raise
        except Exception as exc:
            if _is_dimension_error(exc):
                self._raise_dimension_mismatch(self._live_collection_size(), dimension, exc)
            raise RuntimeError(f"unable to initialize Qdrant collection: {exc}") from exc

    def _ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes required by Qdrant Cloud filtered search.

        The local server scans unindexed payloads on demand; the cloud HTTP API
        rejects a filtered query (``namespace`` / ``memory_type``) on a field
        without a payload index (HTTP 400).  The calls are idempotent and
        failures are non-fatal: an index is a search requirement, not a write
        requirement, so a collection that cannot be indexed must still accept
        upserts.
        """

        create = getattr(self.client, "create_payload_index", None)
        if not callable(create):
            return
        for field in ("namespace", "memory_type"):
            try:
                create(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                    wait=False,
                )
            except Exception:  # noqa: BLE001 - 已存在 / 集群不支持 / 权限不足都按已处理
                continue

    def recreate_collection(self, dimension: int) -> None:
        """Drop and recreate the collection at ``dimension``（换嵌入模型后重建投影）。

        向量只是 SQLite 真值的投影：删除集合只丢投影不丢数据。调用方必须先
        得到确认，再全量重灌；本方法本身从不在写入路径上静默触发。
        """

        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        try:
            from qdrant_client.models import Distance, VectorParams

            if self.client.collection_exists(collection_name=self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            self.dimension = dimension
            self._ensure_payload_indexes()
            self._ready = True
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"unable to recreate Qdrant collection: {exc}") from exc

    def upsert(self, item: MemoryItem) -> None:
        if not item.embedding:
            return
        self._ensure_collection(len(item.embedding))
        from qdrant_client.models import PointStruct
        payload = self._light_payload(item)
        payload["namespace"] = self.namespace
        try:
            self.client.upsert(collection_name=self.collection_name, points=[PointStruct(id=self._point_id(item.id), vector=item.embedding, payload=payload)])
        except Exception as exc:
            if not _is_dimension_error(exc):
                raise
            self._raise_dimension_mismatch(self._live_collection_size() or self.dimension, len(item.embedding), exc)

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
        try:
            self.client.upsert(collection_name=self.collection_name, points=[PointStruct(id=self._point_id(chunk_id), vector=vector, payload=payload)])
        except Exception as exc:
            if not _is_dimension_error(exc):
                raise
            self._raise_dimension_mismatch(self._live_collection_size() or self.dimension, len(vector), exc)

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
        if not self._ensure_ready_for_read():
            return False
        from qdrant_client.models import PointIdsList
        self.client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=[self._point_id(item_id)]))
        return True

    def _ensure_ready_for_read(self) -> bool:
        """Attach to an existing collection without creating one.

        ``_ready`` is only set by writes, so a process that merely reads (the
        reconcile endpoint, a UI session) would otherwise report an empty
        collection as "no vectors". 读路径不把 ``_ready`` 置位，避免跳过写入时
        的维度核对。
        """

        if self._ready:
            return True
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                return False
        except Exception:  # noqa: BLE001 - 探测失败按「读不到」处理，由调用方降级
            return False
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
