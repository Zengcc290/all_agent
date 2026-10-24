"""工具：manage_graph / manage_data —— 图库与本地库的增删改查管理面。"""
from __future__ import annotations

import asyncio

from app.core.registry import registry
from app.core.validation import Tool, ToolParam
from app.db.neo4j_store import graph_store as neo4j_store
from app.db.sqlite_store import store


async def _manage(action: str, entity_key: str = "", predicate: str = "",
                  target_key: str = "", name: str = "", type: str = "概念",
                  relation_time: str = "", directed: bool = True) -> dict:
    """对图库实体/关系做 增/删/改。"""
    act = (action or "").strip().lower()
    key = (entity_key or name or "").strip()
    tgt = (target_key or "").strip()

    if act == "add_entity":
        if not key:
            raise ValueError("add_entity 需要 entity_key 或 name")
        n = await neo4j_store.upsert_entities([
            {"key": key, "name": name or key, "type": type or "概念", "time": relation_time}])
        return {"ok": True, "action": act, "entities": n}

    if act == "add_relation":
        if not key or not tgt:
            raise ValueError("add_relation 需要 entity_key 与 target_key")
        n = await neo4j_store.upsert_relations([{
            "source": name or key, "src_key": key, "target": tgt, "tgt_key": tgt,
            "predicate": predicate or "RELATED", "directed": directed,
            "time": relation_time}])
        return {"ok": True, "action": act, "relations": n}

    if act == "update_entity":
        if not key:
            raise ValueError("update_entity 需要 entity_key 或 name")
        ent = await neo4j_store.get_entity(key)
        if not ent:
            raise ValueError(f"实体 {key!r} 不存在")
        n = await neo4j_store.upsert_entities([
            {"key": key, "name": name or ent.get("name") or key, "type": type or ent.get("type") or "概念"}])
        return {"ok": True, "action": act, "entities": n, "before": ent}

    if act == "delete_entity":
        if not key:
            raise ValueError("delete_entity 需要 entity_key")
        ent = await neo4j_store.get_entity(key)
        if not ent:
            raise ValueError(f"实体 {key!r} 不存在")
        await neo4j_store.delete_entity(key)
        return {"ok": True, "action": act, "deleted": ent}

    if act == "delete_relation":
        if not key and not predicate:
            raise ValueError("delete_relation 需要 entity_key 或 predicate")
        rels = await neo4j_store.get_all_relations(limit=2000)
        hit = [r for r in rels
               if (not key or r.get("src") == key or r.get("tgt") == key)
               and (not predicate or r.get("predicate") == predicate)
               and (not tgt or r.get("tgt") == tgt)]
        if not hit:
            raise ValueError("没有匹配到要删除的关系")
        for r in hit:
            rid = r.get("rid")
            if rid:
                await neo4j_store.delete_relation(str(rid))
        return {"ok": True, "action": act, "deleted": len(hit), "relations": hit}

    raise ValueError(f"未知 action: {act}，可用: add_entity / add_relation / update_entity / delete_entity / delete_relation")


registry.register(Tool(
    name="manage_graph",
    description=(
        "知识图谱（neo4j）的增删改管理面：新增实体、新增关系、改实体类型、删实体（级联删关系）、删关系。"
    ),
    params=[
        ToolParam("action", "string", "要执行的动作", required=True,
                  enum=["add_entity", "add_relation", "update_entity", "delete_entity", "delete_relation"]),
        ToolParam("entity_key", "string", "实体 key（缺省用 name 归一化）", default="", max_length=100),
        ToolParam("name", "string", "实体显示名（add/update 时用）", default="", max_length=100),
        ToolParam("type", "string", "实体类型（add/update 时用）", default="概念", max_length=30),
        ToolParam("target_key", "string", "关系的目标实体 key（add_relation / delete_relation 用）", default="", max_length=100),
        ToolParam("predicate", "string", "关系谓词，如「治疗」「属于」「合作」", default="", max_length=60),
        ToolParam("directed", "boolean", "该关系是否有向：true=画箭头，false=无向直线", default=True),
        ToolParam("relation_time", "string", "时间标记（可到年/月/日/时）", default="", max_length=40),
    ],
    handler=_manage,
    tags=["neo4j", "增删改"],
    timeout=60,
))
