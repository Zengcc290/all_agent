"""工具：list_pending_chunks —— 获取 sqlite 中未入库成功的所有 chunk，截取前 N 个字符。"""
from __future__ import annotations

import asyncio

from app.core.registry import registry
from app.core.validation import Tool, ToolParam
from app.db.sqlite_store import store


async def _list_pending(limit: int = 20, preview_chars: int = 10, status: str = "pending") -> dict:
    rows = await asyncio.to_thread(store.list_pending, limit, status)
    items = []
    for r in rows:
        content = r["content"] or ""
        items.append({
            "chunk_id": r["chunk_id"],
            "preview": content[:max(0, preview_chars)],
            "char_len": len(content),
            "document_id": r["document_id"],
            "seq": r["seq"],
            "qdrant_status": r["qdrant_status"],
            "neo4j_status": r["neo4j_status"],
            "error": r["error"] or "",
            "updated_at": r["updated_at"],
        })
    total = await asyncio.to_thread(store.count_pending)
    return {
        "ok": True,
        "total_pending": total,
        "returned": len(items),
        "preview_chars": preview_chars,
        "items": items,
    }


registry.register(Tool(
    name="list_pending_chunks",
    description=(
        "获取 sqlite 入库状态表中「未入库成功」的所有 chunk 信息，并把每个 chunk 的正文截取前 preview_chars 个字符后返回，"
        "供前端渲染成可折叠列表。默认只要 qdrant 或 neo4j 任一线未成功的 chunk（status='pending'）；"
        "status='all' 返回队列中全部行（含已成功的）。"
    ),
    params=[
        ToolParam("limit", "integer", "最多返回多少条 chunk", default=20, min_value=1, max_value=200),
        ToolParam("preview_chars", "integer", "每个 chunk 正文截取前多少个字符（默认 10）",
                  default=10, min_value=1, max_value=200),
        ToolParam("status", "string", "pending=仅未入库成功（默认）/ all=队列里全部行",
                  default="pending", enum=["pending", "all"]),
    ],
    handler=_list_pending,
    tags=["sqlite", "队列"],
    timeout=30,
))
