"""工具：ingest_chunk —— 把队列里某个未成功的 chunk 重新过一遍入库流程。

前端「重新入库」按钮点一下，就真实调用本工具：
按钮立即变为不可点击并显示「入库中」，直到工具返回 success / failed。
"""
from __future__ import annotations

import time

from app.core.ingest import ingest_chunk as _ingest_chunk
from app.core.registry import registry
from app.core.validation import Tool, ToolParam


async def _reingest(chunk_id: str, force: bool = False) -> dict:
    chunk_id = (chunk_id or "").strip()
    if not chunk_id:
        raise ValueError("参数 chunk_id 不能为空")
    t0 = time.perf_counter()
    res = await _ingest_chunk(chunk_id)
    res["ok"] = res["promoted"]
    res["elapsed_ms"] = res.get("elapsed_ms") or round((time.perf_counter() - t0) * 1000, 1)
    return res


registry.register(Tool(
    name="ingest_chunk",
    description=(
        "重新入库：对 sqlite 入库状态表中某个未成功的 chunk 重新执行 qdrant 与 neo4j 两条入库线"
        "（并行执行），两条都成功才会把该 chunk 转正进 chunks 表，否则继续留在队列中。"
        "适用于前端「重新入库」按钮点击后发起的真实入库请求。"
    ),
    params=[
        ToolParam("chunk_id", "string", "ssqlite ingest_queue.chunk_id（从前端列表或 list_pending_chunks 获得）",
                  required=True, max_length=64),
        ToolParam("force", "boolean", "是否强制重跑（已成功的 chunk 默认不再重复入库）", default=False),
    ],
    handler=_reingest,
    tags=["sqlite", "入库"],
    timeout=300,
))
