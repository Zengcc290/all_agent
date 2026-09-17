"""Neo4j graph relation store with an in-memory fallback."""

from __future__ import annotations

from collections.abc import Mapping
from operator import itemgetter
from typing import Any

from constants import MEMORY_EDGE_WEIGHT_GROWTH, MEMORY_EDGE_WEIGHT_MAX


class Neo4jGraphStore:
    """Neo4j relation store with an in-memory fallback for local development."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        *,
        driver: Any = None,
        database: str | None = None,
    ) -> None:
        self.database = database
        self.driver = driver
        self._local: dict[str, list[dict[str, Any]]] = {}
        self._reverse: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        #: 内存回退下的实体属性（Neo4j 侧由 ON CREATE/ON MATCH 维护同样的三项）。
        self._entities: dict[str, dict[str, Any]] = {}
        if self.driver is None and uri:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError("Neo4jGraphStore requires neo4j") from exc
            if not username or password is None:
                raise ValueError("username and password are required for Neo4j")
            self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        properties: Mapping[str, Any] | None = None,
        source_domain: str = "",
        target_domain: str = "",
        source_aliases: list[str] | None = None,
        target_aliases: list[str] | None = None,
        source_importance: float = 0.5,
        target_importance: float = 0.5,
        bump: bool = False,
    ) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (source, relation, target)
        ):
            raise ValueError("source, relation and target must be non-empty strings")
        if bump:
            # F1 回忆强化：只动已存在边的计数/权重，不写实体、不造边。
            self._bump_relation(source, relation, target)
            return
        props = dict(properties or {})
        if self.driver is None:
            edges = self._local.setdefault(source, [])
            existing = next(
                (
                    edge
                    for edge in edges
                    if edge["relation"] == relation and edge["target"] == target
                ),
                None,
            )
            if existing is None:
                edge = {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "properties": {
                        **props,
                        # F1：计数/权重由存储层在创建时给默认值，写入方（抽取
                        # 管道）不传它们——否则幂等重写会把计数清零。
                        "weight": 1.0,
                        "recall_count": 0,
                        "last_accessed_at": "",
                    },
                }
                edges.append(edge)
                self._reverse.setdefault(target, []).append((source, edge))
            else:
                existing["properties"].update(props)
            self._merge_entity(source, source_domain, source_aliases, source_importance)
            self._merge_entity(target, target_domain, target_aliases, target_importance)
            return
        # ON MATCH 只更新别名：实体的 domain/importance 由首次创建它的那次抽取决定，
        # 后续边写入不该把它们覆盖成空值。weight/recall_count 只在 ON CREATE 给
        # 默认值（F1）：SET r += $properties 是覆盖语义，常规写入碰它们会把
        # 已累计的回忆计数清零。
        query = (
            "MERGE (a:MemoryEntity {name: $source}) "
            "ON CREATE SET a.domain = $source_domain, a.aliases = $source_aliases, a.importance = $source_importance "
            "ON MATCH SET a.aliases = CASE WHEN size($source_aliases) = 0 THEN a.aliases ELSE $source_aliases END "
            "MERGE (b:MemoryEntity {name: $target}) "
            "ON CREATE SET b.domain = $target_domain, b.aliases = $target_aliases, b.importance = $target_importance "
            "ON MATCH SET b.aliases = CASE WHEN size($target_aliases) = 0 THEN b.aliases ELSE $target_aliases END "
            "MERGE (a)-[r:RELATED {kind: $relation}]->(b) "
            "ON CREATE SET r += $properties, r.weight = 1.0, r.recall_count = 0, r.last_accessed_at = '' "
            "ON MATCH SET r += $properties"
        )
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                source=source,
                target=target,
                relation=relation,
                properties=props,
                source_domain=source_domain,
                target_domain=target_domain,
                source_aliases=list(source_aliases or []),
                target_aliases=list(target_aliases or []),
                source_importance=source_importance,
                target_importance=target_importance,
            ).consume()

    def _merge_entity(
        self, name: str, domain: str, aliases: list[str] | None, importance: float
    ) -> None:
        """In-memory twin of the Cypher ON CREATE/ON MATCH entity clauses."""

        known = list(aliases or [])
        entity = self._entities.get(name)
        if entity is None:
            self._entities[name] = {
                "domain": domain,
                "aliases": known,
                "importance": importance,
            }
        elif known:
            entity["aliases"] = known

    def _bump_relation(self, source: str, relation: str, target: str) -> bool:
        """F1 回忆强化：给已存在的边 +1 次回忆、权重×增长倍数（有上限）。

        边不存在时什么都不做——强化不能凭空造边。返回是否真的强化了一条边。
        """

        from ..base import utc_now

        accessed_at = utc_now().isoformat()
        if self.driver is None:
            for edge in self._local.get(source, []):
                if edge["relation"] == relation and edge["target"] == target:
                    self._apply_bump(edge["properties"], accessed_at)
                    return True
            return False
        query = (
            "MATCH (a:MemoryEntity {name: $source})-[r:RELATED {kind: $relation}]->(b:MemoryEntity {name: $target}) "
            "SET r.recall_count = coalesce(r.recall_count, 0) + 1, "
            "r.last_accessed_at = $accessed_at, "
            "r.weight = CASE "
            "WHEN coalesce(r.weight, 1.0) * $grow_factor > $weight_max THEN $weight_max "
            "ELSE coalesce(r.weight, 1.0) * $grow_factor END "
            "RETURN 1 AS bumped"
        )
        with self.driver.session(database=self.database) as session:
            record = session.run(
                query,
                source=source,
                relation=relation,
                target=target,
                accessed_at=accessed_at,
                grow_factor=MEMORY_EDGE_WEIGHT_GROWTH,
                weight_max=MEMORY_EDGE_WEIGHT_MAX,
            ).single()
        return record is not None

    @staticmethod
    def _apply_bump(properties: dict[str, Any], accessed_at: str) -> None:
        """In-memory twin of the Cypher bump clauses."""

        properties["recall_count"] = int(properties.get("recall_count", 0) or 0) + 1
        properties["last_accessed_at"] = accessed_at
        grown = float(properties.get("weight", 1.0) or 1.0) * MEMORY_EDGE_WEIGHT_GROWTH
        properties["weight"] = min(grown, MEMORY_EDGE_WEIGHT_MAX)

    def entity(self, name: str) -> dict[str, Any]:
        """Entity attributes: 别名 / 领域 / 重要度。

        启用 Neo4j 时以它为准——``_entities`` 只是本进程写入过的内存镜像，
        新进程里它是空的，直接读它会静默返回「没有别名」。
        """

        if self.driver is None:
            return dict(self._entities.get(name, {}))
        query = (
            "MATCH (e:MemoryEntity {name: $name}) "
            "RETURN e.domain AS domain, e.aliases AS aliases, e.importance AS importance"
        )
        with self.driver.session(database=self.database) as session:
            record = session.run(query, name=name).single()
        if record is None:
            return {}
        importance = record["importance"]
        return {
            "domain": record["domain"] or "",
            "aliases": list(record["aliases"] or []),
            "importance": float(importance) if importance is not None else 0.5,
        }

    def entity_aliases(self) -> dict[str, list[str]]:
        """``{实体名: 别名}`` 一次取回；星云图要给每个实体节点带别名，不能逐个查。

        无 driver 时读内存镜像；有 driver 时一条 Cypher 查完全部实体（避免 N+1）。
        """

        if self.driver is None:
            return {
                name: list(attributes.get("aliases") or [])
                for name, attributes in self._entities.items()
            }
        query = "MATCH (e:MemoryEntity) RETURN e.name AS name, e.aliases AS aliases"
        with self.driver.session(database=self.database) as session:
            return {
                str(record["name"]): [str(alias) for alias in (record["aliases"] or [])]
                for record in session.run(query)
            }

    def relation_memory_ids(self) -> list[str]:
        """``memory_id`` of every edge (reconcile compares this against facts)."""

        if self.driver is None:
            return [
                str(edge["properties"].get("memory_id") or "")
                for edges in self._local.values()
                for edge in edges
            ]
        query = "MATCH ()-[r:RELATED]->() RETURN r.memory_id AS memory_id"
        with self.driver.session(database=self.database) as session:
            return [str(record["memory_id"]) for record in session.run(query)]

    def get_relations(
        self, entity: str, *, relation: str | None = None, direction: str = "both"
    ) -> list[dict[str, Any]]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be in, out, or both")
        if self.driver is None:
            values = (
                list(self._local.get(entity, []))
                if direction in ("out", "both")
                else []
            )
            if direction in ("in", "both"):
                values += [
                    {
                        "source": source,
                        "relation": edge["relation"],
                        "target": entity,
                        "properties": edge["properties"],
                    }
                    for source, edge in self._reverse.get(entity, [])
                ]
            if relation is not None:
                values = [edge for edge in values if edge["relation"] == relation]
            return values
        if direction == "out":
            match, condition = "(a)-[r:RELATED]->(b)", "a.name = $entity"
        elif direction == "in":
            match, condition = "(a)-[r:RELATED]->(b)", "b.name = $entity"
        else:
            match, condition = (
                "(a)-[r:RELATED]->(b)",
                "a.name = $entity OR b.name = $entity",
            )
        clauses = [condition]
        if relation is not None:
            clauses.append("r.kind = $relation")
        query = f"MATCH {match} WHERE {' AND '.join(clauses)} RETURN a.name AS source, r.kind AS relation, b.name AS target, properties(r) AS properties"
        with self.driver.session(database=self.database) as session:
            return [
                dict(record)
                for record in session.run(query, entity=entity, relation=relation)
            ]

    related = get_relations

    def path_query(
        self, start: str, target: str, *, max_depth: int = 3, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Simple paths between two entities, up to ``max_depth`` hops.

        Each result is ``{"entities": [...], "relations": [{source, relation, target, weight}, ...]}``;
        the in-memory fallback mirrors what the Cypher returns. Results are
        ordered by total path weight (F1) — reinforced paths come first.
        """

        if not all(
            isinstance(value, str) and value.strip() for value in (start, target)
        ):
            raise ValueError("start and target must be non-empty strings")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if self.driver is None:
            return self._local_paths(start, target, max_depth=max_depth, limit=limit)
        # 变长区间的上界不能参数化，只能拼进语句；已用 int() 收敛为整数。
        # path_weight 用 reduce 累加边权重（F1），让 LIMIT 截断发生在权重排序之后。
        query = (
            f"MATCH p=(a:MemoryEntity {{name: $start}})-[*1..{int(max_depth)}]-(b:MemoryEntity {{name: $target}}) "
            "RETURN [n IN nodes(p) | n.name] AS entities, "
            "[r IN relationships(p) | {source: startNode(r).name, relation: r.kind, target: endNode(r).name, weight: coalesce(r.weight, 1.0)}] AS relations, "
            "reduce(w = 0.0, r IN relationships(p) | w + coalesce(r.weight, 1.0)) AS path_weight "
            "ORDER BY path_weight DESC, length(p) ASC "
            "LIMIT $limit"
        )
        with self.driver.session(database=self.database) as session:
            return [
                {
                    "entities": list(record["entities"]),
                    "relations": [dict(relation) for relation in record["relations"]],
                }
                for record in session.run(query, start=start, target=target, limit=limit)
            ]

    def _neighbours(self, entity: str) -> list[tuple[str, str, bool, float]]:
        """``(other_end, relation, is_outgoing, weight)`` for every edge at ``entity``."""

        values = [
            (edge["target"], edge["relation"], True, float(edge["properties"].get("weight", 1.0) or 1.0))
            for edge in self._local.get(entity, [])
        ]
        values += [
            (source, edge["relation"], False, float(edge["properties"].get("weight", 1.0) or 1.0))
            for source, edge in self._reverse.get(entity, [])
        ]
        # 权重优先：强边先入队，limit 截断时留下的是更强的路径（F1）。
        values.sort(key=itemgetter(3), reverse=True)
        return values

    def _local_paths(
        self, start: str, target: str, *, max_depth: int, limit: int
    ) -> list[dict[str, Any]]:
        """BFS over simple paths; shortest paths come first because it is a queue."""

        paths: list[dict[str, Any]] = []
        queue: list[tuple[str, list[str], list[dict[str, Any]]]] = [(start, [start], [])]
        while queue and len(paths) < limit:
            node, entities, relations = queue.pop(0)
            if len(relations) >= max_depth:
                continue
            for other, relation, outgoing, weight in self._neighbours(node):
                if other in entities:  # 简单路径：不重复经过同一实体，天然无环
                    continue
                step = {
                    "source": node if outgoing else other,
                    "relation": relation,
                    "target": other if outgoing else node,
                    "weight": weight,
                }
                if other == target:
                    paths.append(
                        {"entities": [*entities, other], "relations": [*relations, step]}
                    )
                    if len(paths) >= limit:
                        break
                    continue
                queue.append((other, [*entities, other], [*relations, step]))
        return paths

    def delete_memory_relation(self, memory_id: str) -> bool:
        """Remove a relation created for a specific semantic memory item."""
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("memory_id must be a non-empty string")
        if self.driver is None:
            removed = False
            for source, edges in list(self._local.items()):
                kept = [
                    edge
                    for edge in edges
                    if edge.get("properties", {}).get("memory_id") != memory_id
                ]
                for edge in edges:
                    if edge in kept:
                        continue
                    self._reverse[edge["target"]] = [
                        (edge_source, indexed_edge)
                        for edge_source, indexed_edge in self._reverse.get(
                            edge["target"], []
                        )
                        if indexed_edge is not edge
                    ]
                removed = removed or len(kept) != len(edges)
                if kept:
                    self._local[source] = kept
                else:
                    self._local.pop(source, None)
            return removed
        query = "MATCH ()-[r:RELATED {memory_id: $memory_id}]->() DELETE r"
        with self.driver.session(database=self.database) as session:
            result = session.run(query, memory_id=memory_id).consume()
            return bool(getattr(result.counters, "relationships_deleted", 0))

    def close(self) -> None:
        if self.driver is not None and callable(getattr(self.driver, "close", None)):
            self.driver.close()


__all__ = ["Neo4jGraphStore"]
