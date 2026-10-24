"""工具：query_graph —— neo4j 多跳查询。"""
from __future__ import annotations

from app.core.registry import registry
from app.core.validation import Tool, ToolParam
from app.db.neo4j_store import graph_store as neo4j_store


async def _query(entity: str, hops: int = 2, limit: int = 50, direction: str = "both") -> dict:
    if not (entity or "").strip():
        raise ValueError("必填参数 entity 不能为空")
    res = await neo4j_store.multi_hop(entity.strip(), hops=hops, limit=limit, direction=direction)
    res["ok"] = True
    return res


registry.register(Tool(
    name="query_graph",
    description=(
        "neo4j 多跳查询（图数据库特色能力）：给定一个起始实体，沿 REL 关系做 1~N 跳遍历，"
        "返回所有命中的路径（含沿途实体节点、关系谓词、是否有向、时间标记）。"
        "direction='both' 时按无向遍历（可发现反向关系），'out' 时只沿有向关系前进。"
    ),
    params=[
        ToolParam("entity", "string", "起始实体，可以是实体名或归一化 key", required=True, max_length=100),
        ToolParam("hops", "integer", "跳数 1~5（默认 2）", default=2, min_value=1, max_value=5),
        ToolParam("limit", "integer", "最多返回多少条路径（默认 50）", default=50, min_value=1, max_value=500),
        ToolParam("direction", "string", "both=无向双向遍历（默认）/ out=只沿有向关系前进",
                  default="both", enum=["both", "out"]),
    ],
    handler=_query,
    tags=["neo4j", "查询", "多跳"],
    timeout=60,
))
