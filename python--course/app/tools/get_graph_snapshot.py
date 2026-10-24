"""工具：get_graph_snapshot —— 给前端「实体星球」用的全量快照（节点 + 有向/无向连线）。"""
from __future__ import annotations

from app.core.registry import registry
from app.core.validation import Tool, ToolParam
from app.db.neo4j_store import graph_store as neo4j_store


async def _snapshot(limit: int = 400, relation_limit: int = 1500) -> dict:
    data = await neo4j_store.graph_snapshot(limit=limit)
    if len(data["links"]) > relation_limit:
        data["links"] = data["links"][:relation_limit]
        data["truncated"] = True
    else:
        data["truncated"] = False
    data["ok"] = True
    data["directed_links"] = sum(1 for l in data["links"] if l["directed"])
    data["undirected_links"] = sum(1 for l in data["links"] if not l["directed"])
    return data


registry.register(Tool(
    name="get_graph_snapshot",
    description=(
        "获取知识图谱全量快照：所有实体节点 + 所有实体关系连线，"
        "连线带 predicate / directed（有向画箭头，无向画直线）/ time / chunk_id，"
        "供前端渲染「实体星球」力导向图。"
    ),
    params=[
        ToolParam("limit", "integer", "最多返回多少个实体节点（默认 400）", default=400, min_value=1, max_value=5000),
        ToolParam("relation_limit", "integer", "最多返回多少条关系连线（默认 1500）",
                  default=1500, min_value=1, max_value=20000),
    ],
    handler=_snapshot,
    tags=["neo4j", "查询", "可视化"],
    timeout=60,
))
