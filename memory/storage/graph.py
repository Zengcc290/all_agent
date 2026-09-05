"""Neo4j graph relation store with an in-memory fallback."""

from __future__ import annotations

from typing import Any, Mapping


class Neo4jGraphStore:
    """Neo4j relation store with an in-memory fallback for local development."""

    def __init__(self, uri: str | None = None, username: str | None = None, password: str | None = None, *, driver: Any = None, database: str | None = None) -> None:
        self.database = database
        self.driver = driver
        self._local: dict[str, list[dict[str, Any]]] = {}
        if self.driver is None and uri:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError("Neo4jGraphStore requires neo4j") from exc
            if not username or password is None:
                raise ValueError("username and password are required for Neo4j")
            self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def add_relation(self, source: str, relation: str, target: str, *, properties: Mapping[str, Any] | None = None) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (source, relation, target)):
            raise ValueError("source, relation and target must be non-empty strings")
        props = dict(properties or {})
        if self.driver is None:
            edge = {"source": source, "relation": relation, "target": target, "properties": props}
            edges = self._local.setdefault(source, [])
            if edge not in edges:
                edges.append(edge)
            return
        query = "MERGE (a:MemoryEntity {name: $source}) MERGE (b:MemoryEntity {name: $target}) MERGE (a)-[r:RELATED {kind: $relation}]->(b) SET r += $properties"
        with self.driver.session(database=self.database) as session:
            session.run(query, source=source, target=target, relation=relation, properties=props).consume()

    # Common aliases used by graph-oriented clients.
    upsert_relation = add_relation

    def get_relations(self, entity: str, *, relation: str | None = None, direction: str = "both") -> list[dict[str, Any]]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be in, out, or both")
        if self.driver is None:
            values = list(self._local.get(entity, [])) if direction in ("out", "both") else []
            if direction in ("in", "both"):
                values += [edge for edges in self._local.values() for edge in edges if edge["target"] == entity]
            if relation is not None:
                values = [edge for edge in values if edge["relation"] == relation]
            return values
        if direction == "out":
            match, condition = "(a)-[r:RELATED]->(b)", "a.name = $entity"
        elif direction == "in":
            match, condition = "(a)-[r:RELATED]->(b)", "b.name = $entity"
        else:
            match, condition = "(a)-[r:RELATED]->(b)", "a.name = $entity OR b.name = $entity"
        clauses = [condition]
        if relation is not None:
            clauses.append("r.kind = $relation")
        query = f"MATCH {match} WHERE {' AND '.join(clauses)} RETURN a.name AS source, r.kind AS relation, b.name AS target, properties(r) AS properties"
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(query, entity=entity, relation=relation)]

    related = get_relations

    def delete_relation(self, source: str, relation: str, target: str) -> bool:
        if self.driver is None:
            before = len(self._local.get(source, []))
            self._local[source] = [e for e in self._local.get(source, []) if not (e["relation"] == relation and e["target"] == target)]
            return len(self._local[source]) < before
        query = "MATCH (a:MemoryEntity {name: $source})-[r:RELATED {kind: $relation}]->(b:MemoryEntity {name: $target}) DELETE r"
        with self.driver.session(database=self.database) as session:
            result = session.run(query, source=source, relation=relation, target=target).consume()
            return bool(getattr(result.counters, "relationships_deleted", 0))

    def delete_memory_relation(self, memory_id: str) -> bool:
        """Remove a relation created for a specific semantic memory item."""
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("memory_id must be a non-empty string")
        if self.driver is None:
            removed = False
            for source, edges in list(self._local.items()):
                kept = [edge for edge in edges if edge.get("properties", {}).get("memory_id") != memory_id]
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
