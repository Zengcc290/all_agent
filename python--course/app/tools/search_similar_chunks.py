"""工具：search_similar_chunks —— qdrant 向量相似度查询。"""
from __future__ import annotations

from app.core.registry import registry
from app.core.textutil import norm_key
from app.core.validation import Tool, ToolParam
from app.db.neo4j_store import graph_store as neo4j_store
from app.db.qdrant_store import vector_store as qdrant_store
from app.db.sqlite_store import store
from app.llm import embed_texts


async def _search(query: str, top_k: int = 5, score_threshold: float = 0.0,
                  with_entities: bool = True) -> dict:
    q = (query or "").strip()
    if not q:
        raise ValueError("必填参数 query 不能为空")
    vecs = await embed_texts([q])
    if not vecs:
        raise ValueError("embedding 编码失败")
    hits = await qdrant_store.search(vecs[0], limit=top_k, score_threshold=score_threshold or None)
    if with_entities:
        import asyncio
        for h in hits:
            cid = h.get("chunk_id")
            if cid:
                h["entities"] = [e.get("name") or e.get("entity_key")
                                 for e in await asyncio.to_thread(store.get_chunk_entities, cid)]
    return {
        "ok": True,
        "query": q,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "dim": len(vecs[0]),
        "count": len(hits),
        "hits": hits,
    }


registry.register(Tool(
    name="search_similar_chunks",
    description=(
        "qdrant 向量相似度查询（向量数据库特色能力）：把输入文本用 embedding 模型编码成向量，"
        "在 qdrant 集合里做近似最近邻检索，返回最相似的 chunk 及其相似度分数，"
        "并附带每个 chunk 在 sqlite 里映射到的实体。"
    ),
    params=[
        ToolParam("query", "string", "要检索的查询文本", required=True, max_length=5000),
        ToolParam("top_k", "integer", "返回前 k 条最相似结果（默认 5）", default=5, min_value=1, max_value=100),
        ToolParam("score_threshold", "number", "相似度分数下限，0 表示不过滤（默认 0）",
                  default=0.0, min_value=0.0, max_value=1.0),
        ToolParam("with_entities", "boolean", "是否附带每个命中 chunk 关联的实体列表（默认 true）", default=True),
    ],
    handler=_search,
    tags=["qdrant", "查询", "向量"],
    timeout=60,
))
