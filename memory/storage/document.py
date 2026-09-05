"""SQLite document persistence for memory records."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..base import MemoryItem, MemoryType, _json_restore


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
        self.path = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
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


__all__ = ["BaseDocumentStore", "SQLiteDocumentStore"]
