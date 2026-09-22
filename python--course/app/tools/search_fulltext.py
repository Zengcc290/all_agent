"""工具：search_fulltext —— sqlite FTS5 全文检索（词法路），hybrid_search 的可单独调用版本。"""
from __future__ import annotations

import asyncio

from app.core.registry import registry
from app.core.validation import Tool, ToolParam
from app.db.sqlite_store import store


async def _search(query: str, top_k: int = 10) -> dict:
    q = (query or "").strip()
    if not q:
        raise ValueError("必填参数 query 不能为空")
    hits = await asyncio.to_thread(store.fts_search, q, top_k)
    for h in hits:
        cid = h.get("chunk_id")
        h["entities"] = [e.get("entity_name")
                         for e in (await asyncio.to_thread(store.get_chunk_entities, cid) if cid else [])]
    return {
        "ok": True,
        "query": q,
        "engine": "sqlite FTS5 (trigram)",
        "fts_ok": store.fts_ok,
        "count": len(hits),
        "hits": hits,
    }


registry.register(Tool(
    name="search_fulltext",
    description=(
        "sqlite FTS5 全文检索（多路混合检索的「词法路」单独版）。"
        "基于 trigram 分词器，原生支持中文 3 字以上子串匹配，适合关键词精准召回；"
        "在 hybrid_search 里它与 qdrant 向量路并行执行并由 RRF 融合。"
    ),
    params=[
        ToolParam("query", "string", "检索关键词或短语（多个词之间是 AND 关系）", required=True, max_length=2000),
        ToolParam("top_k", "integer", "返回前 k 条（默认 10）", default=10, min_value=1, max_value=100),
    ],
    handler=_search,
    tags=["检索", "sqlite", "FTS5"],
    timeout=60,
))
