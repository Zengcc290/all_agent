"""工具：get_all_entities / get_all_relations —— 从 neo4j 抽出所有实体（与关系）。"""
from __future__ import annotations

from app.core.registry import registry
from app.core.validation import Tool, ToolParam
from app.db.neo4j_store import graph_store as neo4j_store


async def _all_entities(limit: int = 500, keyword: str = "") -> dict:
    ents = await neo4j_store.get_all_entities(limit=limit, keyword=keyword or None)
    return {
        "ok": True,
        "count": len(ents),
        "entities": [
            {
                "key": e.get("key"),
                "name": e.get("name"),
                "type": e.get("type") or "概念",
                "time": e.get("time") or "",
                "aliases": e.get("aliases") or [],
                # out_rels / in_rels 在 store 侧就是 count() 出来的整数
                "out_count": int(e.get("out_rels") or 0),
                "in_count": int(e.get("in_rels") or 0),
            }
            for e in ents
        ],
    }


async def _all_relations(limit: int = 1000) -> dict:
    rels = await neo4j_store.get_all_relations(limit=limit)
    return {
        "ok": True,
        "count": len(rels),
        "relations": [
            {
                "source": r.get("src_label") or r.get("src_name") or r.get("src"),
                "source_key": r.get("src"),
                "target": r.get("tgt_label") or r.get("tgt_name") or r.get("tgt"),
                "target_key": r.get("tgt"),
                "predicate": r.get("predicate") or "RELATED",
                "directed": bool(r.get("directed", True)),
                "time": r.get("time") or "",
                "chunk_id": r.get("chunk_id") or "",
                "evidence": r.get("evidence") or "",
            }
            for r in rels
        ],
    }


registry.register(Tool(
    name="get_all_entities",
    description=(
        "从 neo4j 关系库中抽出当前所有的实体，返回 key / name / type / 别名 / 出入度。"
        "这是实体复用的依据：入库前先调用它，LLM 才能对已存在的实体进行复用而不是另造新实体。"
    ),
    params=[
        ToolParam("limit", "integer", "最多返回多少个实体（按实体名排序）", default=500, min_value=1, max_value=5000),
        ToolParam("keyword", "string", "可选：按实体名/ key 模糊过滤，省略返回全部", default="", max_length=100),
    ],
    handler=_all_entities,
    tags=["neo4j", "查询"],
    timeout=60,
))


registry.register(Tool(
    name="get_all_relations",
    description="从 neo4j 关系库中抽出当前所有的实体关系三元组（含谓词、是否有向、时间标记、来源 chunk）。",
    params=[
        ToolParam("limit", "integer", "最多返回多少条关系（按写入时间倒序）", default=1000, min_value=1, max_value=20000),
    ],
    handler=_all_relations,
    tags=["neo4j", "查询"],
    timeout=60,
))
