"""Qdrant 向量库封装：建集合、upsert、相似度查询、删除。"""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app import config


class QdrantStore:
    def __init__(self):
        self.collection = config.qdrant.collection
        self._client: AsyncQdrantClient | None = None
        self._dim: int | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            kw: dict[str, Any] = {
                "prefer_grpc": config.qdrant.prefer_grpc,
                "check_compatibility": False,
            }
            if config.qdrant.local_path:
                # 嵌入式本地模式：qdrant 引擎跑在本进程内，数据落盘到 local_path，
                # 适合本机单进程开发调试（不需要单独起 qdrant 服务）。
                # 注意：同一时刻只允许一个进程打开同一个 local_path。
                kw["path"] = config.qdrant.local_path
                kw.pop("prefer_grpc")
            else:
                kw.update({"host": config.qdrant.host, "port": config.qdrant.port})
                if config.qdrant.api_key:
                    kw["api_key"] = config.qdrant.api_key
            self._client = AsyncQdrantClient(**kw)
        return self._client

    # ---------------- 集合 ----------------
    async def ensure_collection(self, dim: int | None = None) -> dict:
        d = dim or config.embedding.dim
        distance = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }.get(config.qdrant.distance, models.Distance.COSINE)
        if await self.client.collection_exists(self.collection):
            try:
                info = await self.client.get_collection(self.collection)
                existing = _vector_info(info)["dim"] or d
                if int(existing) != int(d):
                    return {"created": False, "dim": existing,
                            "warning": f"集合维度 {existing} 与配置维度 {d} 不一致，请调整 EMBEDDING_DIM"}
                self._dim = int(existing)
                return {"created": False, "dim": self._dim}
            except Exception:  # noqa: BLE001
                pass
        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=d, distance=distance),
        )
        if not config.qdrant.local_path:
            await self.client.create_payload_index(
                collection_name=self.collection, field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        self._dim = d
        return {"created": True, "dim": d}

    async def health(self) -> dict:
        info = await self.client.get_collection(self.collection)
        return {
            "ok": True,
            "collection": self.collection,
            "points": int(info.points_count or 0),
            **_vector_info(info),
            "status": str(getattr(info, "status", "")),
        }

    # ---------------- 增删改 ----------------
    async def upsert_chunk(self, chunk_id: str, vector: list[float], payload: dict) -> str:
        """以 chunk_id 作为向量点的主键（稳定可覆盖）。"""
        try:
            key = uuid.UUID(chunk_id)
            point_id = str(key)
        except (ValueError, AttributeError, TypeError):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))
        await self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=point_id, vector=vector, payload={
                "chunk_id": chunk_id, **payload, "point_id": point_id})],
        )
        return point_id

    async def upsert_many(self, chunk_ids: list[str], vectors: list[list[float]],
                          payloads: list[dict]) -> list[str]:
        pts = []
        ids = []
        for cid, vec, pl in zip(chunk_ids, vectors, payloads):
            try:
                pid = str(uuid.UUID(cid))
            except (ValueError, AttributeError, TypeError):
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(cid)))
            ids.append(pid)
            pts.append(models.PointStruct(id=pid, vector=vec, payload={"chunk_id": cid, **pl, "point_id": pid}))
        await self.client.upsert(collection_name=self.collection, points=pts)
        return ids

    async def delete_points(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(
                points=[await self._stable_id(c) for c in chunk_ids]),
        )
        return len(chunk_ids)

    async def delete_by_document(self, document_id: str) -> int:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id))])),
        )
        return await self.count()

    async def _stable_id(self, chunk_id: str) -> str:
        try:
            return str(uuid.UUID(chunk_id))
        except (ValueError, AttributeError, TypeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))

    # ---------------- 查 ----------------
    async def search(self, vector: list[float], limit: int = 10, score_threshold: float | None = None,
                     filter_doc: str | None = None) -> list[dict]:
        qfilter = None
        if filter_doc:
            qfilter = models.Filter(must=[models.FieldCondition(
                key="document_id", match=models.MatchValue(value=filter_doc))])
        res = await self.client.query_points(
            collection_name=self.collection, query=vector, limit=limit,
            score_threshold=score_threshold, query_filter=qfilter, with_payload=True)
        return [
            {
                "chunk_id": p.payload.get("chunk_id") if p.payload else None,
                "score": round(float(p.score), 6),
                "content": (p.payload or {}).get("content", ""),
                "document_id": (p.payload or {}).get("document_id"),
                "created_at": (p.payload or {}).get("created_at"),
                "point_id": str(p.id),
            }
            for p in res.points
        ]

    async def scroll(self, limit: int = 100) -> list[dict]:
        pts, _ = await self.client.scroll(
            collection_name=self.collection, limit=limit, with_payload=True)
        return [{"chunk_id": (p.payload or {}).get("chunk_id"),
                 "content": (p.payload or {}).get("content", ""),
                 "document_id": (p.payload or {}).get("document_id")} for p in pts]

    async def count(self) -> int:
        info = await self.client.get_collection(self.collection)
        return int(info.points_count or 0)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _vector_info(info) -> dict:
    """从 collection info 里安全取出维度 / 距离度量。

    qdrant-client 不同版本的模型形状不一致（有时 vectors 是 VectorParams，
    有时是 dict，偶尔连属性名都拿不到），所以这里全部用 getattr 兜底，
    取不到就返回 None，绝让「监控」把可用的库报成不可用。
    """
    dim = dist = None
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict):
        for v in vectors.values():
            dim = dim or getattr(v, "size", None)
            dist = dist or getattr(v, "distance", None)
    elif vectors is not None:
        dim = getattr(vectors, "size", None)
        dist = getattr(vectors, "distance", None)
        if dim is None:  # attributes 藏在 nested config 里的情况
            inner = getattr(vectors, "params", None)
            dim = getattr(inner, "size", None)
            dist = dist or getattr(inner, "distance", None)
    if dim is not None:
        dim = int(dim)
    if dist is not None:
        dist = str(getattr(dist, "value", dist))
    return {"dim": dim, "distance": dist or "unknown"}


# 单例（注意命名：避免与子模块名 app.db.qdrant_store 互相遮蔽）
vector_store = QdrantStore()
