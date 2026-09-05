"""Storage backends for memory records, vectors and graph relations."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .models import MemoryItem, MemorySearchResult, MemoryType, _json_restore


class BaseDocumentStore(ABC):
    @abstractmethod
    def upsert(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def get(self, item_id: str) -> MemoryItem | None: ...

    @abstractmethod
    def delete(self, item_id: str) -> bool: ...

    @abstractmethod
    def list(self, *, memory_type: MemoryType | str | None = None, include_expired: bool = False) -> list[MemoryItem]: ...

    def save(self, item: MemoryItem) -> None:
        self.upsert(item)

    def clear(self, memory_type: MemoryType | str | None = None) -> int:
        """Delete records, with a portable fallback for custom stores."""
        items = self.list(memory_type=memory_type, include_expired=True)
        return sum(1 for item in items if self.delete(item.id))

    def close(self) -> None:
        pass


class SQLiteDocumentStore(BaseDocumentStore):
    """SQLite document persistence with safe JSON serialization."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        if self.path == ":memory:":
            self._connection = sqlite3.connect(self.path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection_scope(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    yield connection
            finally:
                if connection is not self._connection:
                    connection.close()

    def _initialize(self) -> None:
        with self._connection_scope() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    timestamp TEXT,
                    embedding TEXT,
                    payload TEXT,
                    modality TEXT,
                    relations TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)")

    def upsert(self, item: MemoryItem) -> None:
        if not isinstance(item, MemoryItem):
            raise TypeError("item must be a MemoryItem")
        data = item.to_dict()
        with self._connection_scope() as connection:
            connection.execute(
                """
                INSERT INTO memories
                (id, content, memory_type, metadata, importance, created_at, updated_at,
                 expires_at, timestamp, embedding, payload, modality, relations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  content=excluded.content, memory_type=excluded.memory_type,
                  metadata=excluded.metadata, importance=excluded.importance,
                  created_at=excluded.created_at, updated_at=excluded.updated_at,
                  expires_at=excluded.expires_at, timestamp=excluded.timestamp,
                  embedding=excluded.embedding, payload=excluded.payload,
                  modality=excluded.modality, relations=excluded.relations
                """,
                (
                    item.id, item.content, item.memory_type.value,
                    json.dumps(data["metadata"], ensure_ascii=False), item.importance,
                    data["created_at"], data["updated_at"], data["expires_at"],
                    data["timestamp"], json.dumps(item.embedding),
                    json.dumps(data.get("payload"), ensure_ascii=False), item.modality,
                    json.dumps(data["relations"], ensure_ascii=False),
                ),
            )

    def get(self, item_id: str) -> MemoryItem | None:
        with self._connection_scope() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (item_id,)).fetchone()
        return self._decode(row) if row else None

    def delete(self, item_id: str) -> bool:
        with self._connection_scope() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (item_id,))
            return cursor.rowcount > 0

    def list(self, *, memory_type: MemoryType | str | None = None, include_expired: bool = False) -> list[MemoryItem]:
        query = "SELECT * FROM memories"
        params: list[Any] = []
        if memory_type is not None:
            query += " WHERE memory_type = ?"
            params.append(MemoryType(memory_type).value)
        query += " ORDER BY COALESCE(timestamp, created_at) DESC"
        with self._connection_scope() as connection:
            rows = connection.execute(query, params).fetchall()
        items = [self._decode(row) for row in rows]
        return items if include_expired else [item for item in items if not item.is_expired]

    def clear(self, memory_type: MemoryType | str | None = None) -> int:
        with self._connection_scope() as connection:
            if memory_type is None:
                cursor = connection.execute("DELETE FROM memories")
            else:
                cursor = connection.execute("DELETE FROM memories WHERE memory_type = ?", (MemoryType(memory_type).value,))
            return cursor.rowcount

    @staticmethod
    def _decode(row: sqlite3.Row) -> MemoryItem:
        payload = _json_restore(json.loads(row["payload"])) if row["payload"] is not None else None
        return MemoryItem(
            id=row["id"], content=row["content"], memory_type=row["memory_type"],
            metadata=json.loads(row["metadata"]), importance=row["importance"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            expires_at=row["expires_at"], timestamp=row["timestamp"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            payload=payload, modality=row["modality"], relations=_json_restore(json.loads(row["relations"])),
        )

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def delete(self, item_id: str) -> bool: ...

    @abstractmethod
    def search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]: ...

    def clear(self) -> None:
        pass


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    if any(not math.isfinite(float(value)) for value in (*left, *right)):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class InMemoryVectorStore(BaseVectorStore):
    """Fast local vector index, useful as the default and for unit tests."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], MemoryType]] = {}
        self._lock = threading.RLock()

    def upsert(self, item: MemoryItem) -> None:
        if item.embedding is not None:
            if not item.embedding or any(not math.isfinite(float(value)) for value in item.embedding):
                raise ValueError("item embedding must contain finite values")
            with self._lock:
                self._vectors[item.id] = (list(item.embedding), item.memory_type)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            return self._vectors.pop(item_id, None) is not None

    def search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        wanted = MemoryType(memory_type) if memory_type is not None else None
        with self._lock:
            scores = [
                (item_id, cosine_similarity(vector, stored), stored_type)
                for item_id, (stored, stored_type) in self._vectors.items()
                if wanted is None or stored_type == wanted
            ]
        scores.sort(key=lambda row: (-row[1], row[0]))
        return [(item_id, score) for item_id, score, _ in scores[:limit]]

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector database adapter.

    ``client`` can be injected (a ``qdrant_client.QdrantClient`` compatible
    object) for tests.  The collection is created lazily on the first upsert,
    allowing the embedding dimension to be discovered from the item.
    """

    def __init__(self, url: str | None = None, collection_name: str = "helloagents_memory", *, api_key: str | None = None, client: Any = None, dimension: int | None = None, namespace: str = "memory") -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("QdrantVectorStore requires qdrant-client") from exc
            client = QdrantClient(url=url, api_key=api_key) if url else QdrantClient(path=":memory:")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be non-empty")
        if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
            raise ValueError("dimension must be a positive integer")
        self.client, self.collection_name, self.dimension, self.namespace = client, collection_name, dimension, namespace
        self._ready = False

    def _ensure_collection(self, dimension: int) -> None:
        if self._ready:
            return
        try:
            from qdrant_client.models import Distance, VectorParams
            exists = self.client.collection_exists(collection_name=self.collection_name)
            if not exists:
                self.client.create_collection(collection_name=self.collection_name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
            elif self.dimension is not None and self.dimension != dimension:
                raise ValueError(f"Qdrant collection dimension mismatch: expected {self.dimension}, got {dimension}")
            else:
                get_collection = getattr(self.client, "get_collection", None)
                if callable(get_collection):
                    info = get_collection(collection_name=self.collection_name)
                    configured = getattr(getattr(info, "config", None), "params", None)
                    configured_size = getattr(configured, "size", None)
                    if configured_size is not None and int(configured_size) != dimension:
                        raise ValueError(f"Qdrant collection dimension mismatch: existing {configured_size}, got {dimension}")
            self.dimension = dimension
            self._ready = True
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"unable to initialize Qdrant collection: {exc}") from exc

    def upsert(self, item: MemoryItem) -> None:
        if not item.embedding:
            return
        self._ensure_collection(len(item.embedding))
        from qdrant_client.models import PointStruct
        payload = item.to_dict()
        payload["namespace"] = self.namespace
        self.client.upsert(collection_name=self.collection_name, points=[PointStruct(id=self._point_id(item.id), vector=item.embedding, payload=payload)])

    def delete(self, item_id: str) -> bool:
        if not self._ready:
            return False
        from qdrant_client.models import PointIdsList
        self.client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=[self._point_id(item_id)]))
        return True

    def search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not self._ready:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        conditions = [FieldCondition(key="namespace", match=MatchValue(value=self.namespace))]
        if memory_type is not None:
            conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=MemoryType(memory_type).value)))
        query_filter = Filter(must=conditions)
        try:
            points = self.client.search(collection_name=self.collection_name, query_vector=vector, query_filter=query_filter, limit=limit)
        except AttributeError:
            points = self.client.query_points(collection_name=self.collection_name, query=vector, query_filter=query_filter, limit=limit).points
        return [(str((point.payload or {}).get("id", point.id)), float(point.score)) for point in points]

    @staticmethod
    def _point_id(item_id: str) -> str:
        """Qdrant accepts UUIDs/integers only; preserve arbitrary app IDs in payload."""
        try:
            uuid.UUID(item_id)
            return item_id
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"helloagents-memory:{item_id}"))

    def clear(self) -> None:
        if self._ready:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ready = False


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
        condition = "a.name = $entity" if direction == "out" else "b.name = $entity" if direction == "in" else "a.name = $entity OR b.name = $entity"
        query = f"MATCH (a:MemoryEntity)-[r:RELATED]->(b:MemoryEntity) WHERE {condition} RETURN a.name AS source, r.kind AS relation, b.name AS target, properties(r) AS properties"
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(query, entity=entity)]

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


__all__ = [
    "BaseDocumentStore", "BaseVectorStore", "InMemoryVectorStore", "Neo4jGraphStore",
    "QdrantVectorStore", "SQLiteDocumentStore", "cosine_similarity",
]
