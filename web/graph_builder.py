"""把四层记忆「压扁」成星云图需要的 nodes + edges JSON。

映射规则（与前端 web/static/index.html 的布局约定一致）：
- kind=domain  → 恒星   （level 1，前端做星系定位）
- kind=entity  → 行星   （level 2，绕所属领域公转）
- kind=fact    → 卫星   （level 3，绕主语实体公转），同时产出一条
                        实体→实体的「边」（三元组谓词就是边标签）
- kind=chunk   → 卫星   （绕所属文档实体公转）
- kind=note    → 卫星   （绕所属实体公转，不产生边）
- kind=event   → 卫星   （挂在内置「事件时间线」实体下）

分类依据（按优先级）：
1. metadata 同时含 subject/predicate/object → fact（语义三元组）
2. metadata.kind == "entity"               → 显式实体
3. metadata.kind == "note"                 → 实体备注
4. metadata 含 document_id + chunk_index    → RAG 知识块
5. 其余                                     → 事件（episodic/working 等）
"""

from __future__ import annotations

import zlib
from typing import Any

from memory import MemoryItem, MemoryManager

#: 领域配色板（与 Aetheria 深空青紫主题协调）。
PALETTE = [
    "#38bdf8",  # 青
    "#c084fc",  # 紫
    "#f43f5e",  # 玫红
    "#fbbf24",  # 金
    "#34d399",  # 绿
    "#60a5fa",  # 蓝
    "#f472b6",  # 粉
    "#a3e635",  # 黄绿
]

TIMELINE_DOMAIN = "时间线"
TIMELINE_ENTITY = "事件时间线"
TIMELINE_ID = "ent:__timeline__"
DOC_DOMAIN = "文档库"
DEFAULT_DOMAIN = "未分类"


def domain_color(name: str) -> str:
    """领域 → 稳定颜色（crc32，跨进程稳定，Python 内建 hash 不稳定）。"""
    return PALETTE[zlib.crc32((name or DEFAULT_DOMAIN).encode("utf-8")) % len(PALETTE)]


def _node(node_id: str, kind: str, title: str, *, content: str = "", domain: str = "",
          date: str = "", importance: float = 0.5, parent: str | None = None,
          source: str | None = None) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": kind,
        "title": title,
        "content": content,
        "domain": domain or DEFAULT_DOMAIN,
        "date": date,
        "importance": round(float(importance), 3),
        "color": domain_color(domain),
        "parent": parent,
        "source": source,
        "meta": {},
    }


def build_graph(manager: MemoryManager) -> dict[str, Any]:
    items: list[MemoryItem] = manager.list()
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    domains: dict[str, str] = {}
    entity_ids: dict[str, str] = {}

    def domain_node(name: str) -> str:
        name = (name or DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN
        if name not in domains:
            node_id = f"dom:{name}"
            nodes[node_id] = _node(node_id, "domain", name, domain=name, importance=0.9)
            domains[name] = node_id
        return domains[name]

    def entity_node(name: str, *, domain: str = "", item: MemoryItem | None = None,
                    title: str | None = None) -> str:
        key = (name or "").strip()
        if not key:
            key = DEFAULT_DOMAIN
        if key in entity_ids:
            return entity_ids[key]
        if item is not None:
            node_id = item.id
            node_title = title or item.metadata.get("title") or key
            content, date, importance = item.content, _date(item), item.importance
            source = item.metadata.get("filename") or item.metadata.get("source")
        else:
            node_id = f"ent:{key}"
            node_title, content, date, importance, source = key, "", "", 0.6, None
        nodes[node_id] = _node(
            node_id, "entity", node_title, content=content, domain=domain or DEFAULT_DOMAIN,
            date=date, importance=importance, parent=domain_node(domain or DEFAULT_DOMAIN),
            source=source,
        )
        entity_ids[key] = node_id
        return node_id

    # 内置「事件时间线」实体：聊天/上传等事件都挂在这里。
    nodes[TIMELINE_ID] = _node(
        TIMELINE_ID, "entity", TIMELINE_ENTITY, domain=TIMELINE_DOMAIN,
        importance=0.4, parent=domain_node(TIMELINE_DOMAIN),
    )
    entity_ids[TIMELINE_ENTITY] = TIMELINE_ID

    # --- 第一遍：显式实体（种子数据里的 kind=entity 项） ---
    explicit_entities = [item for item in items if item.metadata.get("kind") == "entity"]
    for item in explicit_entities:
        entity_node(
            item.metadata.get("title") or item.content,
            domain=item.metadata.get("domain") or DEFAULT_DOMAIN,
            item=item,
        )

    # --- 事实（三元组）：月亮 + 实体间的边 ---
    for item in items:
        md = item.metadata
        subject, predicate, obj = md.get("subject"), md.get("predicate"), md.get("object")
        if not (subject and predicate and obj):
            continue
        domain = md.get("domain") or DEFAULT_DOMAIN
        source_id = entity_node(subject, domain=domain)
        target_id = entity_node(obj, domain=domain)
        node = _node(
            item.id, "fact", f"{subject} —{predicate}→ {obj}",
            content=md.get("note") or item.content, domain=domain, date=_date(item),
            importance=item.importance, parent=source_id,
            source=md.get("filename") or md.get("source"),
        )
        node["meta"] = {"subject": subject, "predicate": predicate, "object": obj,
                       "confidence": md.get("confidence", item.importance)}
        nodes[item.id] = node
        edges.append({
            "id": f"edge:{item.id}",
            "source": source_id,
            "target": target_id,
            "relation": predicate,
            "confidence": md.get("confidence", item.importance),
            "evidence": md.get("evidence", ""),
            "source_document": md.get("source_document") or md.get("source") or "",
            "chunk_id": md.get("chunk_id") or "",
        })

    # --- 实体备注（kind=note）：挂到所属实体的卫星 ---
    for item in items:
        md = item.metadata
        if md.get("kind") != "note":
            continue
        parent = entity_ids.get((md.get("entity") or "").strip(), TIMELINE_ID)
        domain = md.get("domain") or nodes[parent]["domain"]
        nodes[item.id] = _node(
            item.id, "note", md.get("title") or "档案", content=item.content,
            domain=domain, date=_date(item), importance=item.importance,
            parent=parent, source=md.get("filename") or md.get("source"),
        )

    # --- RAG 知识块：按 document_id 聚成「文档实体」的卫星 ---
    docs: dict[str, dict[str, Any]] = {}
    for item in items:
        md = item.metadata
        if md.get("document_id") is None or "chunk_index" not in md:
            continue
        document_id = str(md["document_id"])
        if document_id not in docs:
            filename = md.get("filename") or md.get("source") or document_id
            doc_id = f"doc:{document_id}"
            nodes[doc_id] = _node(
                doc_id, "entity", f"文档：{filename}", content=str(md.get("source") or ""),
                domain=DOC_DOMAIN, date=_date(item), importance=0.5,
                parent=domain_node(DOC_DOMAIN), source=filename,
            )
            docs[document_id] = nodes[doc_id]
        nodes[item.id] = _node(
            item.id, "chunk", f"{md.get('filename', '片段')} #{md.get('chunk_index')}",
            content=item.content[:400], domain=DOC_DOMAIN, date=_date(item),
            importance=item.importance, parent=docs[document_id]["id"],
            source=md.get("filename") or md.get("source"),
        )
        nodes[item.id]["meta"]["document_id"] = document_id

    # --- 其余：事件卫星（episodic/working/perceptual 等） ---
    for item in items:
        md = item.metadata
        if (md.get("subject") and md.get("predicate") and md.get("object")) \
                or md.get("kind") in {"entity", "note"} \
                or (md.get("document_id") is not None and "chunk_index" in md):
            continue
        nodes[item.id] = _node(
            item.id, "event", md.get("title") or item.content[:24], content=item.content,
            domain=TIMELINE_DOMAIN, date=_date(item), importance=item.importance,
            parent=TIMELINE_ID,
        )

    node_list = list(nodes.values())
    kinds = {"domain": 0, "entity": 0, "fact": 0, "chunk": 0, "note": 0, "event": 0}
    for node in node_list:
        kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
    return {
        "stats": {
            "domains": kinds["domain"],
            "entities": kinds["entity"],
            "facts": kinds["fact"],
            "chunks": kinds["chunk"],
            "notes": kinds["note"],
            "events": kinds["event"],
            "edges": len(edges),
            "total": len(node_list),
        },
        "nodes": node_list,
        "edges": edges,
    }


def _date(item: MemoryItem) -> str:
    created = item.created_at
    try:
        return created.date().isoformat()
    except AttributeError:
        return str(created)[:10]


__all__ = ["build_graph", "domain_color", "PALETTE"]
