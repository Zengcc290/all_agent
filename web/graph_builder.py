"""把四层记忆「压扁」成星云图需要的 nodes + edges JSON。

映射规则（与前端 web/static/index.html 的布局约定一致）：
- kind=domain  → 恒星   （level 1，前端做星系定位）
- kind=entity  → 行星   （level 2，绕所属领域公转；全局同名同一个）
- kind=chunk   → 行星   （原句：一个文档凝聚为一颗行星，挂领域下，含有向「提及」边连到全部相关实体）
- kind=fact    → 卫星   （多元观察保留，挂主语实体下）
- kind=note    → 卫星   （绕所属实体公转，不产生边）
- kind=event   → 卫星   （绕所属领域公转）

关系不再作为节点：二元关系是一条有向边 source→target，relation 字段是谓词；
时序观察把 event_at/cardinality=temporal 附在边上，同一实体的多条时序边并列保留。

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
)
from memory import MemoryItem, MemoryManager
from web.cleanup import find_orphan_entities
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
    historical_facts = 0
    fact_keys: set[tuple[str, str, str]] = set()
    chunk_to_doc: dict[str, str] = {}
    doc_meta: dict[str, dict[str, Any]] = {}
    linked_pairs: set[tuple[str, str]] = set()

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

    def link_triple(
        subject: str,
        predicate: str,
        obj: str,
        *,
        domain: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        triple = (str(subject), str(predicate), str(obj))
        if triple in fact_keys:
            return
        fact_keys.add(triple)
        source_id = entity_node(subject, domain=domain)
        target_id = entity_node(obj, domain=domain)
        pair = (source_id, predicate, target_id)
        if pair in linked_pairs:
            return
        linked_pairs.add(pair)
        edges.append(
            {
                "id": f"edge:{source_id}:{predicate}:{target_id}",
                "source": source_id,
                "target": target_id,
                "relation": predicate,
                **dict(extra or {}),
            }
        )

    def attach_doc_entities(doc_id: str | None, *names: object) -> None:
        if not doc_id or doc_id not in doc_meta:
            return
        info = doc_meta[doc_id]
        for name in names:
            key_name = str(name or "").strip()
            if key_name and key_name not in info["entity_set"]:
                info["entity_set"].add(key_name)
                info["entities"].append(key_name)


    # --- 第一遍：显式实体（种子数据里的 kind=entity 项） ---
    explicit_entities = [item for item in items if item.metadata.get("kind") == "entity"]
    for item in explicit_entities:
        entity_node(
            item.metadata.get("title") or item.content,
            domain=item.metadata.get("domain") or DEFAULT_DOMAIN,
            item=item,
        )

    # --- RAG chunks: unique document satellites (filled after entities exist) ---
    for item in items:
        md = item.metadata
        if md.get("document_id") is None or "chunk_index" not in md:
            continue
        document_id = str(md["document_id"])
        chunk_to_doc[item.id] = document_id
        chunk_domain = classify_domain(
            item.content, title=md.get("filename") or md.get("source") or ""
        )
        if document_id not in doc_meta:
            filename = md.get("filename") or md.get("source") or document_id
            doc_meta[document_id] = {
                "filename": filename,
                "preview": item.content[:NEBULA_CONTENT_PREVIEW_CHARS],
                "date": _date(item),
                "importance": item.importance,
                "domains": [chunk_domain],
                "entities": [],
                "entity_set": set(),
                "source": filename,
            }
        else:
            doc_meta[document_id]["domains"].append(chunk_domain)

    # --- facts: attach documents now; edges use directed predicates later ---
    sqlite_facts: list[MemoryItem] = []
    for item in items:
        md = item.metadata
        subject, predicate, obj = md.get("subject"), md.get("predicate"), md.get("object")
        if not (subject and predicate and obj):
            continue
        doc_id = md.get("document_id") or chunk_to_doc.get(str(md.get("chunk_id") or ""))
        attach_doc_entities(str(doc_id) if doc_id else None, subject, obj)
        sqlite_facts.append(item)


    # --- 实体→文档：被提取的实体进入其所属文档的 entity_set（即便还没有任何关系边）---
    # 这样原句行星才能用「提及」边连到全部被提取实体，而不只是连到有二元关系的实体。
    for item in items:
        md = item.metadata
        if md.get("kind") != "entity":
            continue
        doc_id = md.get("document_id") or chunk_to_doc.get(str(md.get("chunk_id") or ""))
        if not doc_id:
            doc_id = next(
                (
                    chunk_to_doc[source_id]
                    for source_id in md.get("source_ids") or []
                    if source_id in chunk_to_doc
                ),
                None,
            )
        name = md.get("canonical_name") or md.get("title") or item.content
        attach_doc_entities(str(doc_id) if doc_id else None, name)

    # --- 实体备注（kind=note）：挂到所属实体的卫星 ---
    for item in items:
        md = item.metadata
        if md.get("kind") != "note":
            continue
        parent = entity_ids.get((md.get("entity") or "").strip())
        domain = md.get("domain") or (nodes[parent]["domain"] if parent else DEFAULT_DOMAIN)
        if parent is None:
            parent = domain_node(domain)
        nodes[item.id] = _node(
            item.id, "note", md.get("title") or "档案", content=item.content,
            domain=domain, date=_date(item), importance=item.importance,
            parent=parent, source=md.get("filename") or md.get("source"),
        )

    # --- 其余：事件卫星（episodic/working/perceptual 等） ---
    for item in items:
        md = item.metadata
        if (md.get("subject") and md.get("predicate") and md.get("object")) \
                or md.get("kind") in {"entity", "note"} \
                or (md.get("document_id") is not None and "chunk_index" in md):
            continue
        title = str(md.get("title") or "")
        source = str(md.get("source") or "")
        if md.get("ingest_job_id") or title == "一句话入库" or source == "一句话入库":
            continue
        if str(item.content or "").startswith("添加了一条知识"):
            continue
        nodes[item.id] = _node(
            item.id, "event", md.get("title") or item.content[:NEBULA_EVENT_TITLE_CHARS], content=item.content,
            domain=md.get("domain") or DEFAULT_DOMAIN, date=_date(item), importance=item.importance,
            parent=domain_node(md.get("domain") or DEFAULT_DOMAIN),
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
            active = properties.get("active", True) is not False
            extra = [
                value
                for value in participants
                if value.get("role") not in {"subject", "object"}
            ]
            if not active:
                historical_facts += 1
                continue
            chunk_id = str(properties.get("chunk_id") or "")
            source = str(properties.get("source") or properties.get("source_document") or "")
            doc_id = properties.get("document_id") or chunk_to_doc.get(chunk_id)
            if not doc_id:
                for did, info in doc_meta.items():
                    if info["source"] == source or info["filename"] == source:
                        doc_id = did
                        break
            attach_doc_entities(str(doc_id) if doc_id else None, subject, object_name)
            domain = str(properties.get("domain") or DEFAULT_DOMAIN)
            common = {
                "confidence": properties.get("confidence", 0.75),
                "evidence": properties.get("evidence", ""),
                "event_at": properties.get("event_at", ""),
                "valid_from": properties.get("valid_from", ""),
                "valid_to": properties.get("valid_to", ""),
                "active": active,
                "observation_id": observation_id,
            }
            if extra:
                source_id = entity_node(subject, domain=domain)
                if observation_id not in nodes:
                    title = f"{subject} -{predicate}-> {object_name}"
                    nodes[observation_id] = _node(
                        observation_id,
                        "fact",
                        title,
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
                fact_keys.add((subject, predicate, object_name))
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
            else:
                link_triple(
                    subject,
                    predicate,
                    object_name,
                    domain=domain,
                    extra=common,
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
            link_triple(source, predicate, target, extra=dict(properties))

    for item in sqlite_facts:
        md = item.metadata
        subject, predicate, obj = md.get("subject"), md.get("predicate"), md.get("object")
        active = md.get("active", True) is not False
        triple = (str(subject), str(predicate), str(obj))
        if not active:
            if snapshot is None and triple not in fact_keys:
                historical_facts += 1
                fact_keys.add(triple)
            continue
        link_triple(
            str(subject),
            str(predicate),
            str(obj),
            domain=str(md.get("domain") or DEFAULT_DOMAIN),
            extra={
                "confidence": md.get("confidence", item.importance),
                "evidence": md.get("evidence", ""),
                "source_document": md.get("source_document") or md.get("source") or "",
                "chunk_id": md.get("chunk_id") or "",
                "active": True,
                "cardinality": md.get("cardinality", "multi"),
            },
        )

    unique_docs: dict[str, tuple[str, dict[str, Any]]] = {}
    for document_id, info in doc_meta.items():
        preview = " ".join(str(info.get("preview") or "").split())
        filename = str(info.get("filename") or "").strip()
        key = preview or filename or document_id
        if key in unique_docs:
            _did, dest = unique_docs[key]
            dest["domains"].extend(info["domains"])
            for name in info["entities"]:
                if name not in dest["entity_set"]:
                    dest["entity_set"].add(name)
                    dest["entities"].append(name)
            continue
        unique_docs[key] = (document_id, info)

    for document_id, info in unique_docs.values():
        domain = majority_domain(info["domains"]) or DEFAULT_DOMAIN
        node_id = f"doc:{document_id}"
        if node_id in nodes:
            continue
        title = str(info["filename"] or document_id)
        if title in {"一句话入库", "问答抽取"} and info.get("preview"):
            title = str(info["preview"]).strip().splitlines()[0][:40]
        node = _node(
            node_id,
            "chunk",
            title,
            content=str(info["preview"]),
            domain=domain,
            date=str(info["date"]),
            importance=float(info["importance"] or 0.5),
            parent=domain_node(domain),
            source=str(info["source"] or "") or None,
        )
        node["meta"]["document_id"] = document_id
        node["meta"]["related_entities"] = list(info["entities"])
        nodes[node_id] = node
        # 原句行星必须和被提取的全部实体有边：即便某个实体还没有任何关系边，
        # 也要通过「提及」连到原句，避免实体在图上变成孤儿。
        for name in info["entities"]:
            endpoint = entity_ids.get(name)
            if endpoint is None:
                continue
            pair = (node_id, endpoint)
            if pair in linked_pairs:
                continue
            linked_pairs.add(pair)
            edges.append(
                {
                    "id": f"edge:{node_id}:{endpoint}",
                    "source": node_id,
                    "target": endpoint,
                    "relation": "提及",
                }
            )

    # --- 恒星 ↔ 行星连线：每个实体/原句行星都有一条指向所属领域恒星的有向边 ---
    # 让星图把「恒星系」和它辖下的行星用引力桥连起来，而不是只在布局上相邻。
    for node in nodes.values():
        if node["kind"] not in {"entity", "chunk"}:
            continue
        star_id = f"dom:{node['domain']}"
        if star_id not in nodes:
            continue
        pair = (star_id, "属于", node["id"])
        if pair in linked_pairs:
            continue
        linked_pairs.add(pair)
        edges.append(
            {
                "id": f"edge:{star_id}:属于:{node['id']}",
                "source": star_id,
                "target": node["id"],
                "relation": "属于",
                "confidence": 1.0,
                "structural": True,
            }
        )

    node_list = list(nodes.values())
    kinds = {"domain": 0, "entity": 0, "fact": 0, "chunk": 0, "note": 0, "event": 0}
    for node in node_list:
        kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
    orphan_count = len(find_orphan_entities(manager, items=items))
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
            "relations": len(edges),
            "facts": kinds["fact"],
            "chunks": kinds["chunk"],
            "notes": kinds["note"],
            "events": kinds["event"],
            "edges": len(edges),
            "historical_facts": historical_facts,
            "orphan_entities": orphan_count,
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
