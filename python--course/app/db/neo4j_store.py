"""Neo4j 关系库封装：实体/关系/chunk 节点的增删改查 + 多跳查询。"""
from __future__ import annotations

import uuid
from typing import Any

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app import config


def _rel_id() -> str:
    return uuid.uuid4().hex[:16]


class Neo4jStore:
    def __init__(self):
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.user, config.neo4j.password),
                connection_timeout=config.neo4j.timeout,
                max_connection_pool_size=20,
            )
        return self._driver

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def verify(self) -> dict:
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run("RETURN 1 AS ok")
            val = await rec.single()
            return {"ok": True, "uri": config.neo4j.uri, "database": config.neo4j.database,
                    "probe": val["ok"] if val else None}

    def _session(self):
        return self.driver.session(database=config.neo4j.database, default_access_mode="WRITE")

    # =================================================================
    #  实体
    # =================================================================
    async def upsert_entities(self, entities: list[dict]) -> int:
        """MERGE 实体：已存在的复用（更新属性），不存在的创建。"""
        if not entities:
            return 0
        rows = []
        for e in entities:
            name = (e.get("name") or "").strip()
            # key 一律由实体名归一化而来，与关系表的 src_key/tgt_key 保持一致，
            # 避免 LLM 给的拼音/自定义 key 与 norm_key(name) 不一致导致重复节点。
            key = norm_key(name or e.get("key") or "")
            if not name and not key:
                continue
            alias = e.get("aliases") or e.get("alias") or []
            if isinstance(alias, str):
                alias = [a.strip() for a in alias.replace("，", ",").split(",") if a.strip()]
            rows.append({
                "key": key, "name": name or key,
                "type": (e.get("type") or "概念").strip(),
                "alias": [a for a in (alias or []) if a],
                "time": e.get("time") or "",
            })
        if not rows:
            return 0
        async with self._session() as s:
            await s.run(
                """
                UNWIND $rows AS r
                MERGE (e:Entity {key: r.key})
                SET e.name        = coalesce(nullif(r.name,''), e.name),
                    e.type        = CASE WHEN r.type = '' THEN e.type ELSE r.type END,
                    e.time        = CASE WHEN r.time = '' THEN e.time ELSE r.time END,
                    e.aliases     = CASE WHEN size(r.alias) = 0 THEN e.aliases
                                         WHEN e.aliases IS NULL THEN r.alias
                                         ELSE e.aliases + [a IN r.alias WHERE NOT a IN e.aliases] END,
                    e.updated_at  = datetime()
                """,
                rows=rows,
            )
        return len(rows)

    async def get_all_entities(self, limit: int = 500, keyword: str | None = None) -> list[dict]:
        where, params = "", {"limit": limit}
        if keyword:
            where = "WHERE e.name CONTAINS $kw OR e.key CONTAINS $kw"
            params["kw"] = keyword
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                f"""MATCH (e:Entity) {where}
                    OPTIONAL MATCH (e)-[ro:REL]->(yo:Entity)
                    WITH e, count(ro) AS out_cnt
                    OPTIONAL MATCH (yi:Entity)-[ri:REL]->(e)
                    WITH e, out_cnt, count(ri) AS in_cnt
                    RETURN e.key AS key, e.name AS name, e.type AS type,
                           e.time AS time, e.aliases AS aliases,
                           out_cnt AS out_rels, in_cnt AS in_rels
                    ORDER BY e.name LIMIT $limit""",
                **params,
            )
            return [dict(r) async for r in rec]

    async def get_entity(self, key: str) -> dict | None:
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                "MATCH (e:Entity {key: $key}) RETURN e.key AS key, e.name AS name, e.type AS type",
                key=key,
            )
            r = await rec.single()
            return dict(r) if r else None

    async def delete_entity(self, key: str) -> bool:
        async with self._session() as s:
            await s.run("MATCH (e:Entity {key: $key}) DETACH DELETE e", key=key)
        return True

    # =================================================================
    #  关系
    # =================================================================
    async def upsert_relations(self, triples: list[dict]) -> int:
        """写入三元组；关系按 (source, predicate, target, time) 去重。"""
        if not triples:
            return 0
        rows = []
        for t in triples:
            src = (t.get("source") or "").strip()
            tgt = (t.get("target") or "").strip()
            pred = (t.get("predicate") or t.get("relation") or "RELATED").strip()
            if not src or not tgt:
                continue
            directed = bool(t.get("directed", True))
            rows.append({
                "rid": _rel_id(),
                "src_key": norm_key(t.get("src_key") or src), "src": src,
                "tgt_key": norm_key(t.get("tgt_key") or tgt), "tgt": tgt,
                "predicate": pred,
                "directed": directed,
                "time": t.get("time") or "",
                "chunk_id": t.get("chunk_id") or "",
                "evidence": (t.get("evidence") or "")[:300],
            })
        if not rows:
            return 0
        async with self._session() as s:
            await s.run(
                """
                UNWIND $rows AS r
                MERGE (a:Entity {key: r.src_key})
                  ON CREATE SET a.name = r.src, a.type = '概念',
                                a.time = CASE WHEN r.time = '' THEN '' ELSE r.time END,
                                a.created_at = datetime()
                MERGE (b:Entity {key: r.tgt_key})
                  ON CREATE SET b.name = r.tgt, b.type = '概念',
                                b.time = CASE WHEN r.time = '' THEN '' ELSE r.time END,
                                b.created_at = datetime()
                MERGE (a)-[rel:REL {predicate: r.predicate, tgt_key: r.tgt_key, time: r.time}]->(b)
                SET rel.src_key   = r.src_key,
                    rel.directed  = r.directed,
                    rel.src_name  = r.src,
                    rel.tgt_name  = r.tgt,
                    rel.chunk_id  = CASE WHEN r.chunk_id = '' THEN rel.chunk_id ELSE r.chunk_id END,
                    rel.evidence  = CASE WHEN r.evidence = '' THEN rel.evidence ELSE r.evidence END,
                    rel.updated_at = datetime()
                """,
                rows=rows,
            )
        return len(rows)

    async def get_all_relations(self, limit: int = 1000) -> list[dict]:
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                f"""MATCH (a:Entity)-[r:REL]->(b:Entity)
                    RETURN {rel_proj("r", "a", "b")}, elementId(r) AS rid
                    ORDER BY r.updated_at DESC LIMIT $limit""",
                limit=limit,
            )
            return [dict(x) async for x in rec]

    async def get_relation(self, rid: str) -> dict | None:
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                f"MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE elementId(r) = $rid OR toString(id(r)) = $rid "
                f"RETURN {rel_proj('r', 'a', 'b')}",
                rid=rid,
            )
            r = await rec.single()
            return dict(r) if r else None

    async def delete_relation(self, rid: str) -> bool:
        async with self._session() as s:
            await s.run("MATCH ()-[r:REL]->() WHERE elementId(r) = $rid OR toString(id(r)) = $rid DELETE r", rid=rid)
        return True

    # =================================================================
    #  chunk 节点
    # =================================================================
    async def link_chunk(self, chunk_id: str, entity_keys: list[str], content: str = "") -> int:
        if not chunk_id or not entity_keys:
            return 0
        async with self._session() as s:
            await s.run(
                """
                MERGE (c:Chunk {id: $cid})
                SET c.content = $content, c.updated_at = datetime()
                WITH c
                UNWIND $keys AS k
                MATCH (e:Entity {key: k})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                cid=chunk_id, keys=list(dict.fromkeys(entity_keys)), content=content[:500],
            )
        return len(set(entity_keys))

    async def get_chunk(self, chunk_id: str) -> dict | None:
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                "MATCH (c:Chunk {id: $cid}) RETURN c.id AS id, c.content AS content", cid=chunk_id)
            r = await rec.single()
            return dict(r) if r else None

    async def delete_chunk(self, chunk_id: str) -> bool:
        async with self._session() as s:
            await s.run("MATCH (c:Chunk {id: $cid}) DETACH DELETE c", cid=chunk_id)
        return True

    async def cleanup_chunk(self, chunk_id: str) -> dict:
        """重新入库前清理该 chunk 的旧图产物。

        规则（与「重新入库」语义一致）：
        1. 断开本 chunk 与所有实体的 MENTIONS（相当于从实体上摘掉这个 chunk 编号）；
        2. 删除本 chunk 抽取的关系 REL（rel.chunk_id = chunk_id）；
        3. 逐个实体检查：若还被其他 chunk 引用则保留，否则 DETACH DELETE；
        4. 删除 Chunk 节点本身（link_chunk 会用同一 chunk_id 重新建）。
        """
        if not chunk_id:
            return {"linked": 0, "deleted_entities": 0, "deleted_relations": 0}
        async with self._session() as s:
            rec = await s.run(
                "MATCH (c:Chunk {id: $cid})-[m:MENTIONS]->(e:Entity) "
                "RETURN DISTINCT e.key AS key", cid=chunk_id)
            keys = [r["key"] async for r in rec]

            rel = await s.run(
                "MATCH ()-[r:REL]->() WHERE r.chunk_id = $cid DELETE r "
                "RETURN count(r) AS n", cid=chunk_id)
            rel_row = await rel.single()
            deleted_rels = int(rel_row["n"]) if rel_row else 0

            await s.run("MATCH (c:Chunk {id: $cid})-[m:MENTIONS]->() DELETE m", cid=chunk_id)

            deleted = 0
            for k in keys:
                rec2 = await s.run(
                    "MATCH (e:Entity {key: $k}) "
                    "OPTIONAL MATCH (e)<-[:MENTIONS]-(c2:Chunk) WHERE c2.id <> $cid "
                    "RETURN count(c2) AS other", k=k, cid=chunk_id)
                row = await rec2.single()
                if row and int(row["other"] or 0) == 0:
                    await s.run("MATCH (e:Entity {key: $k}) DETACH DELETE e", k=k)
                    deleted += 1

            await s.run("MATCH (c:Chunk {id: $cid}) DETACH DELETE c", cid=chunk_id)
            return {"linked": len(keys), "deleted_entities": deleted,
                    "deleted_relations": deleted_rels}

    async def get_chunks_of_entity(self, entity_key: str) -> list[str]:
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {key: $k}) RETURN c.id AS id", k=entity_key)
            return [r["id"] async for r in rec]

    # =================================================================
    #  查询
    # =================================================================
    async def multi_hop(self, start: str, hops: int = 2, limit: int = 50,
                        direction: str = "both") -> dict:
        """多跳查询（Neo4j 特色）。start 可以是实体 key 或 name。"""
        hops = max(1, min(int(hops), 5))
        limit = max(1, min(int(limit), 500))
        pat = "-[:REL*1..{}]-".format(hops) if direction == "both" else "-[:REL*1..{}]->".format(hops)
        cypher = f"""
            MATCH p = (a:Entity) {pat} (b:Entity)
            WHERE a.key = $w OR a.name = $w
            RETURN p AS path, [n IN nodes(p) | {{key: n.key, name: n.name, type: n.type, time: n.time}}] AS nodes,
                   [r IN relationships(p) | {{predicate: r.predicate, directed: r.directed,
                                             time: r.time, src: r.src_key, tgt: r.tgt_key,
                                             evidence: r.evidence}}] AS rels
            LIMIT $limit
        """
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(cypher, w=start, limit=limit)
            paths = []
            async for r in rec:
                paths.append({
                    "nodes": _nlist(r["nodes"]),
                    "rels": _rlist(r["rels"]),
                    "hops": len(r["rels"] or []),
                })
        return {"start": start, "hops_requested": hops, "direction": direction,
                "path_count": len(paths), "paths": paths,
                "nodes": _dedupe([n for p in paths for n in p["nodes"]], "key"),
                "rels": _dedupe([x for p in paths for x in p["rels"]], "tgt")}

    async def neighbors(self, key: str, depth: int = 1, limit: int = 100) -> dict:
        """一跳邻域查询（direction=both 时无向）。"""
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                f"""MATCH (a:Entity)-[r:REL]-(b:Entity)
                    WHERE a.key = $k OR a.name = $k
                    RETURN {rel_proj('r', 'a', 'b')} LIMIT $limit""",
                k=key, limit=max(1, min(int(limit), 500)),
            )
            rels = [dict(x) async for x in rec]
        return {"center": key, "relations": rels, "count": len(rels)}

    async def graph_snapshot(self, limit: int = 400) -> dict:
        ents = await self.get_all_entities(limit=limit)
        rels = await self.get_all_relations(limit=limit * 3)
        nodes = [
            {"id": e["key"], "key": e["key"], "name": e.get("name") or e["key"],
             "type": e.get("type") or "概念", "time": e.get("time") or "",
             "group": _group_of(e.get("type") or "")}
            for e in ents
        ]
        links = [
            {
                "id": r.get("rid") or f"{r.get('src')}->{r.get('tgt')}:{r.get('predicate')}",
                "source": r.get("src"), "target": r.get("tgt"),
                "predicate": r.get("predicate") or "RELATED",
                "directed": bool(r.get("directed", True)),
                "time": r.get("time") or "",
                "chunk_id": r.get("chunk_id") or "",
                "evidence": r.get("evidence") or "",
            }
            for r in rels
        ]
        return {"nodes": nodes, "links": links,
                "stats": {"entities": len(nodes), "relations": len(links)}}

    async def stats(self) -> dict:
        async with self.driver.session(database=config.neo4j.database) as s:
            rec = await s.run(
                "OPTIONAL MATCH (e:Entity) WITH count(e) AS n_e "
                "OPTIONAL MATCH ()-[r:REL]->() WITH n_e, count(r) AS n_r "
                "OPTIONAL MATCH (c:Chunk) WITH n_e, n_r, count(c) AS n_c "
                "RETURN n_e, n_r, n_c"
            )
            r = await rec.single()
            return {"entities": r["n_e"] if r else 0, "relations": r["n_r"] if r else 0,
                    "chunks": r["n_c"] if r else 0}

    async def clear_graph(self) -> dict:
        async with self._session() as s:
            await s.run("MATCH (n) DETACH DELETE n")
        return {"ok": True}


# ---------------- 工具函数 ----------------
def norm_key(s: str) -> str:
    """统一走 core.textutil.norm_key，保证 parser 与图库的实体归一化完全一致。"""
    from app.core.textutil import norm_key as _nk
    return _nk(s)


def rel_proj(alias: str = "r", a: str = "a", b: str = "b") -> str:
    return (
        f"{alias}.predicate AS predicate, {alias}.directed AS directed, {alias}.time AS time, "
        f"{alias}.src_key AS src, {alias}.src_name AS src_name, {alias}.tgt_key AS tgt, "
        f"{alias}.tgt_name AS tgt_name, {alias}.chunk_id AS chunk_id, {alias}.evidence AS evidence, "
        f"{a}.name AS src_label, {b}.name AS tgt_label"
    )


_TYPES_COLORS = {
    "人物": "#f472b6", "人员": "#f472b6", "公司": "#60a5fa", "组织": "#a78bfa", "机构": "#a78bfa",
    "地点": "#34d399", "产品": "#fbbf24", "时间": "#22d3ee", "事件": "#fb7185", "疾病": "#f87171",
    "药物": "#4ade80", "法律": "#c084fc", "金融": "#facc15", "科技": "#38bdf8", "概念": "#94a3b8",
}


def _group_of(t: str) -> int:
    keys = list(_TYPES_COLORS)
    if t in keys:
        return keys.index(t) % 10
    return len(keys) % 10


def _nlist(nodes) -> list[dict]:
    out = []
    for n in nodes or []:
        if isinstance(n, dict):
            out.append(n)
    return out


def _rlist(rels) -> list[dict]:
    out = []
    for r in rels or []:
        if isinstance(r, dict):
            out.append(r)
    return out


def _dedupe(items: list[dict], key: str) -> list[dict]:
    seen, out = set(), []
    for it in items:
        k = it.get(key) if isinstance(it, dict) else None
        if k is None:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# 单例（注意命名：避免与子模块名 app.db.neo4j_store 互相遮蔽）
graph_store = Neo4jStore()
