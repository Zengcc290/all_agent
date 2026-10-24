"""工具：get_stats —— 汇总 sqlite / qdrant / neo4j 三库状态，用于前端仪表盘与健康检查。"""
from __future__ import annotations

import asyncio

from app import config
from app.core.registry import registry
from app.core.validation import Tool, ToolParam
from app.db.neo4j_store import graph_store as neo4j_store
from app.db.qdrant_store import vector_store as qdrant_store
from app.db.sqlite_store import store


async def _stats() -> dict:
    sq = await asyncio.to_thread(store.stats)
    qd: dict = {"ok": False, "error": "未探测"}
    ng: dict = {"ok": False, "error": "未探测"}
    try:
        qd = await qdrant_store.ensure_collection()
        qd.update(await qdrant_store.health())
    except Exception as e:  # noqa: BLE001
        qd = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        ng = await neo4j_store.verify()
        ng.update(await neo4j_store.stats())
    except Exception as e:  # noqa: BLE001
        ng = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": True,
        "sqlite": sq,
        "qdrant": qd,
        "neo4j": ng,
        "config": {
            "llm": {"model": config.llm.model, "base_url": config.llm.base_url, "stream": config.llm.stream},
            "embedding": {"model": config.embedding.model, "base_url": config.embedding.base_url,
                          "dim": config.embedding.dim},
            "qdrant": {"host": config.qdrant.host, "port": config.qdrant.port,
                       "collection": config.qdrant.collection},
            "neo4j": {"uri": config.neo4j.uri, "user": config.neo4j.user},
        },
    }


registry.register(Tool(
    name="get_stats",
    description=(
        "获取三库运行状态统计：sqlite（文档数 / 队列数 / 未入库数 / 已转正 chunk 数 / 实体映射数）、"
        "qdrant（集合点数 / 维度 / 距离度量）、neo4j（实体数 / 关系数 / chunk 数），"
        "以及当前 LLM 与 embedding 配置。"
    ),
    params=[],
    handler=_stats,
    tags=["监控", "通用"],
    timeout=60,
))
