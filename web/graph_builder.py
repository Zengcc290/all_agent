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

from constants import (
    DEFAULT_DOMAIN,
    NEBULA_CONTENT_PREVIEW_CHARS,
    NEBULA_DATE_CHARS,
    NEBULA_EVENT_TITLE_CHARS,
    NEBULA_PALETTE,
    TIMELINE_DOMAIN,
    TIMELINE_ENTITY,
    TIMELINE_ID,
)
from memory import MemoryItem, MemoryManager
from web.domain_classifier import classify_domain, majority_domain


def domain_color(name: str) -> str:
    """领域 → 稳定颜色（crc32，跨进程稳定，Python 内建 hash 不稳定）。"""
    return NEBULA_PALETTE[zlib.crc32((name or DEFAULT_DOMAIN).encode("utf-8")) % len(NEBULA_PALETTE)]


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


def _entity_aliases(manager: MemoryManager) -> dict[str, list[str]]:
    """实体名 → 别名（一次取回）。图库读不到就退回空表，不能因此整图失败。"""

    getter = getattr(getattr(manager, "graph_store", None), "entity_aliases", None)
    if not callable(getter):
        return {}
    try:
        return {
            str(name): [str(alias) for alias in (aliases or [])]
            for name, aliases in getter().items()
        }
    except Exception:  # noqa: BLE001 - 图库抖动时退化为「无别名」
        return {}


def _graph_snapshot(
    manager: MemoryManager, *, at: str | None = None
) -> dict[str, Any] | None:
    """Read topology from the graph backend; ``None`` means compatibility fallback."""

    getter = getattr(getattr(manager, "graph_store", None), "graph_snapshot", None)
    if not callable(getter):
        return None
    try:
        return dict(getter(at=at))
    except (TypeError, ValueError):
        raise
    except Exception:  # noqa: BLE001 - unavailable graph falls back to SQLite satellites
        return None


def build_graph(manager: MemoryManager, *, at: str | None = None) -> dict[str, Any]:
    items: list[MemoryItem] = manager.list()
    snapshot = _graph_snapshot(manager, at=at)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    domains: dict[str, str] = {}
    entity_ids: dict[str, str] = {}
    aliases_by_entity = _entity_aliases(manager)

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
        node = _node(
            node_id, "entity", node_title, content=content, domain=domain or DEFAULT_DOMAIN,
            date=date, importance=importance, parent=domain_node(domain or DEFAULT_DOMAIN),
            source=source,
        )
        # U7 实体侧栏：别名来自图库（P4 已把别名写进实体属性，这里只读不写）。
        aliases = aliases_by_entity.get(key) or aliases_by_entity.get(node_title) or []
        if aliases:
            node["meta"]["aliases"] = aliases
        nodes[node_id] = node
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
        active = md.get("active", True) is not False
        label = f"{subject} —{predicate}→ {obj}" if active else f"{subject} —{predicate}→ {obj}（历史）"
        node = _node(
            item.id, "fact", label,
            content=md.get("note") or item.content, domain=domain, date=_date(item),
            importance=item.importance if active else min(item.importance, 0.3),
            parent=source_id,
            source=md.get("filename") or md.get("source"),
        )
        node["meta"] = {"subject": subject, "predicate": predicate, "object": obj,
                       "confidence": md.get("confidence", item.importance),
                       "active": active,
                       "cardinality": md.get("cardinality", "multi"),
                       "action": md.get("action", "assert"),
                       "superseded_by": md.get("superseded_by", []),
                       "supersedes": md.get("supersedes", []),
                       "superseded_at": md.get("superseded_at", ""),
                       "roles": md.get("roles", []),
                       "event_at": md.get("event_at", ""),
                       "valid_from": md.get("valid_from", ""),
                       "valid_to": md.get("valid_to", ""),
                       "status": md.get("status", "fact"),
                       "modality": md.get("modality", "text"),
                       "captured_at": md.get("captured_at", "")}
        nodes[item.id] = node
        # Superseded and retracted facts stay visible as history satellites but
        # produce no edge, so graph traversal only walks current values.
        if not active or snapshot is not None:
            continue
        edges.append({
            "id": f"edge:{item.id}",
            "source": source_id,
            "target": target_id,
            "relation": predicate,
            "confidence": md.get("confidence", item.importance),
            "evidence": md.get("evidence", ""),
            "source_document": md.get("source_document") or md.get("source") or "",
            "chunk_id": md.get("chunk_id") or "",
            "active": active,
            "cardinality": md.get("cardinality", "multi"),
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

    # --- RAG 知识块：按 document_id 聚成「文档实体」的卫星，并按内容自动分类 ---
    docs: dict[str, dict[str, Any]] = {}
    for item in items:
        md = item.metadata
        if md.get("document_id") is None or "chunk_index" not in md:
            continue
        document_id = str(md["document_id"])
        # 自动领域分类：每块按正文内容归到对应恒星系，不再统一堆进「文档库」
        chunk_domain = classify_domain(
            item.content, title=md.get("filename") or md.get("source") or ""
        )
        if document_id not in docs:
            filename = md.get("filename") or md.get("source") or document_id
            doc_id = f"doc:{document_id}"
            nodes[doc_id] = _node(
                doc_id, "entity", f"文档：{filename}", content=str(md.get("source") or ""),
                domain=chunk_domain, date=_date(item), importance=0.5,
                parent=domain_node(chunk_domain), source=filename,
            )
            # 文档实体最终挂到本文档多数知识块的主题恒星系下（而非固定「文档库」）
            docs[document_id] = {"node": nodes[doc_id], "domains": [chunk_domain]}
        else:
            docs[document_id]["domains"].append(chunk_domain)
        nodes[item.id] = _node(
            item.id, "chunk", f"{md.get('filename', '片段')} #{md.get('chunk_index')}",
            content=item.content[:NEBULA_CONTENT_PREVIEW_CHARS], domain=chunk_domain, date=_date(item),
            importance=item.importance, parent=docs[document_id]["node"]["id"],
            source=md.get("filename") or md.get("source"),
        )
        nodes[item.id]["meta"]["document_id"] = document_id

    # 文档实体按众数领域重挂恒星系（众数才决定位置；单个块跨类不受影响）
    for doc in docs.values():
        node = doc["node"]
        node["domain"] = majority_domain(doc["domains"])
        node["parent"] = domain_node(node["domain"])
        node["color"] = domain_color(node["domain"])

    # --- 其余：事件卫星（episodic/working/perceptual 等） ---
    for item in items:
        md = item.metadata
        if (md.get("subject") and md.get("predicate") and md.get("object")) \
                or md.get("kind") in {"entity", "note"} \
                or (md.get("document_id") is not None and "chunk_index" in md):
            continue
        nodes[item.id] = _node(
            item.id, "event", md.get("title") or item.content[:NEBULA_EVENT_TITLE_CHARS], content=item.content,
            domain=TIMELINE_DOMAIN, date=_date(item), importance=item.importance,
            parent=TIMELINE_ID,
        )

    # Relationship topology comes from Neo4j/in-memory graph, not from SQLite
    # inference. SQLite still supplies source text, document previews and event
    # satellites. Old backends without graph_snapshot retain the legacy path.
    if snapshot is not None:
        for entity in snapshot.get("entities") or []:
            name = str(entity.get("name") or "").strip()
            if not name:
                continue
            properties = dict(entity.get("properties") or {})
            entity_id = entity_node(
                name, domain=str(properties.get("domain") or DEFAULT_DOMAIN)
            )
            aliases = [str(value) for value in properties.get("aliases") or []]
            if aliases:
                nodes[entity_id]["meta"]["aliases"] = aliases
            if properties.get("entity_type"):
                nodes[entity_id]["meta"]["entity_type"] = properties["entity_type"]

        observation_ids = {
            str(observation.get("id") or "")
            for observation in snapshot.get("observations") or []
        }
        for observation in snapshot.get("observations") or []:
            observation_id = str(observation.get("id") or "").strip()
            if not observation_id:
                continue
            predicate = str(observation.get("predicate") or "关联")
            properties = dict(observation.get("properties") or {})
            participants = sorted(
                [dict(value) for value in observation.get("participants") or []],
                key=lambda value: int(value.get("ordinal") or 0),
            )
            subject = next(
                (str(value.get("name")) for value in participants if value.get("role") == "subject"),
                "",
            )
            object_name = next(
                (str(value.get("name")) for value in participants if value.get("role") == "object"),
                "",
            )
            if not subject or not object_name:
                continue
            domain = str(properties.get("domain") or DEFAULT_DOMAIN)
            source_id = entity_node(subject, domain=domain)
            target_id = entity_node(object_name, domain=domain)
            active = properties.get("active", True) is not False
            if observation_id not in nodes:
                title = f"{subject} —{predicate}→ {object_name}"
                nodes[observation_id] = _node(
                    observation_id,
                    "fact",
                    title if active else f"{title}（历史）",
                    content=str(properties.get("evidence") or ""),
                    domain=domain,
                    date=str(properties.get("event_at") or properties.get("created_at") or "")[:NEBULA_DATE_CHARS],
                    importance=float(properties.get("confidence", 0.75) or 0.75),
                    parent=source_id,
                    source=str(properties.get("source") or "") or None,
                )
            nodes[observation_id]["meta"].update(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_name,
                    "participants": participants,
                    **properties,
                }
            )
            if not active:
                continue
            extra = [
                value
                for value in participants
                if value.get("role") not in {"subject", "object"}
            ]
            common = {
                "confidence": properties.get("confidence", 0.75),
                "evidence": properties.get("evidence", ""),
                "event_at": properties.get("event_at", ""),
                "valid_from": properties.get("valid_from", ""),
                "valid_to": properties.get("valid_to", ""),
                "active": active,
                "observation_id": observation_id,
            }
            if not extra:
                edges.append(
                    {
                        "id": f"edge:{observation_id}",
                        "source": source_id,
                        "target": target_id,
                        "relation": predicate,
                        **common,
                    }
                )
                continue
            edges.append(
                {
                    "id": f"edge:{observation_id}:subject",
                    "source": source_id,
                    "target": observation_id,
                    "relation": predicate,
                    **common,
                }
            )
            for participant in [
                value for value in participants if value.get("role") != "subject"
            ]:
                name = str(participant.get("name") or "")
                participant_id = entity_node(
                    name, domain=str(participant.get("domain") or domain)
                )
                role = str(participant.get("role") or "参与")
                edges.append(
                    {
                        "id": f"edge:{observation_id}:{int(participant.get('ordinal') or 0)}",
                        "source": observation_id,
                        "target": participant_id,
                        "relation": "宾语" if role == "object" else role,
                        "role": role,
                        **common,
                    }
                )

        for index, relation in enumerate(snapshot.get("relations") or []):
            properties = dict(relation.get("properties") or {})
            if properties.get("active", True) is False:
                continue
            if str(properties.get("memory_id") or "") in observation_ids:
                continue
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            predicate = str(relation.get("relation") or "关联")
            if not source or not target:
                continue
            edges.append(
                {
                    "id": f"edge:neo4j:{properties.get('memory_id') or index}",
                    "source": entity_node(source),
                    "target": entity_node(target),
                    "relation": predicate,
                    **properties,
                }
            )

    node_list = list(nodes.values())
    kinds = {"domain": 0, "entity": 0, "fact": 0, "chunk": 0, "note": 0, "event": 0}
    for node in node_list:
        kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
    return {
        "graph_source": (
            str(snapshot.get("mode") or "graph")
            if snapshot is not None
            else "sqlite-fallback"
        ),
        "as_of": at or "",
        "stats": {
            "domains": kinds["domain"],
            "entities": kinds["entity"],
            "facts": kinds["fact"],
            "chunks": kinds["chunk"],
            "notes": kinds["note"],
            "events": kinds["event"],
            "edges": len(edges),
            "historical_facts": sum(
                1
                for node in node_list
                if node["kind"] == "fact" and not node["meta"].get("active", True)
            ),
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
        return str(created)[:NEBULA_DATE_CHARS]


__all__ = ["NEBULA_PALETTE", "build_graph", "domain_color"]
