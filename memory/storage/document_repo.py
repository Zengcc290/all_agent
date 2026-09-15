"""Document/chunk persistence: the local source of truth for ingested text.

``documents`` and ``chunks`` live in the same SQLite file as ``memories`` but are
a separate concern: ``memories`` holds the four memory types, while these two
tables hold the corpus itself (原文 + 分块边界).  The vector store and the graph
are projections that can be rebuilt from here, so this module never talks to
them - keeping raw text on disk is what makes re-indexing possible after an
embedding space change.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import utc_now

#: ``documents.status`` state machine (方案 2.1).
DOCUMENT_STATUSES = ("uploaded", "parsed", "vectorized", "extracted", "failed")
#: ``chunks.vector_status`` state machine.
CHUNK_VECTOR_STATUSES = ("pending", "indexed", "failed")
#: ``documents.permission`` - 决定是否允许离开本机（D9）。
PERMISSIONS = ("private", "shared", "public")
#: FTS5 分词器优先级：``trigram`` 支持中文子串，退化时用 ``unicode61``。
FTS_TOKENIZERS = ("trigram", "unicode61")


@dataclass
class DocumentRecord:
    """One ingested document; ``raw_text`` is the normalized full text."""

    document_id: str
    title: str = ""
    raw_text: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)
    permission: str = "private"
    status: str = "uploaded"
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ChunkRecord:
    """One chunk of a document, with its character range inside ``raw_text``."""

    chunk_id: str
    document_id: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    vector_status: str = "pending"


class DocumentRepository:
    """CRUD for ``documents``/``chunks`` on the shared memory SQLite file.

    Mirrors :class:`~memory.storage.document.SQLiteDocumentStore`'s connection
    and locking model: one connection per call for file-backed databases, an
    ``RLock`` around every scope, and ``:memory:`` pinned to a single
    connection so tests can use a scratch database.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        #: ``trigram`` / ``unicode61``，FTS5 不可用时为 ``None``（检索退化为纯向量）。
        self.fts_tokenizer: str | None = None
        if self.path == ":memory:":
            self._connection = sqlite3.connect(self.path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        self._initialize()

    # -- plumbing ------------------------------------------------------
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
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT '',
                    raw_text    TEXT NOT NULL,
                    source      TEXT NOT NULL DEFAULT '',
                    tags        TEXT NOT NULL DEFAULT '[]',
                    permission  TEXT NOT NULL DEFAULT 'private',
                    status      TEXT NOT NULL DEFAULT 'uploaded',
                    error       TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id      TEXT PRIMARY KEY,
                    document_id   TEXT NOT NULL,
                    chunk_index   INTEGER NOT NULL,
                    char_start    INTEGER NOT NULL,
                    char_end      INTEGER NOT NULL,
                    text          TEXT NOT NULL,
                    vector_status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
            self._initialize_fts(connection)

    def _initialize_fts(self, connection: sqlite3.Connection) -> None:
        """Build the FTS5 index over ``chunks.text`` (external content + triggers).

        FTS5 is what keeps retrieval alive when the embedding tunnel is down
        (D8), so a build without FTS5 is recorded in ``fts_tokenizer`` instead
        of preventing the repository from opening at all.
        """

        existing = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
        ).fetchone()
        rebuild = existing is None
        if existing is not None:
            # 表已存在：沿用当初建成的分词器，避免把 unicode61 误报为 trigram。
            self.fts_tokenizer = "trigram" if "trigram" in (existing["sql"] or "") else "unicode61"
        else:
            for tokenizer in FTS_TOKENIZERS:
                try:
                    connection.execute(
                        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                        "text, content='chunks', content_rowid='rowid', "
                        f"tokenize='{tokenizer}')"
                    )
                except sqlite3.OperationalError:
                    connection.execute("DROP TABLE IF EXISTS chunks_fts")
                    continue
                self.fts_tokenizer = tokenizer
                break
        if self.fts_tokenizer is None:
            return
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN"
            "  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);"
            " END"
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN"
            "  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);"
            " END"
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN"
            "  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);"
            "  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);"
            " END"
        )
        if rebuild:
            # 外部内容表不会自己回填：为 FTS 之前就存在的 chunks 行建一次索引。
            connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

    # -- documents -----------------------------------------------------
    def upsert_document(self, doc: DocumentRecord) -> None:
        if not isinstance(doc, DocumentRecord):
            raise TypeError("doc must be a DocumentRecord")
        self._check_choice("permission", doc.permission, PERMISSIONS)
        self._check_choice("status", doc.status, DOCUMENT_STATUSES)
        if not isinstance(doc.document_id, str) or not doc.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        now = utc_now().isoformat()
        with self._connection_scope() as connection:
            # created_at 只在首次插入时落库，重跑 ingest 不会把它刷掉。
            connection.execute(
                """
                INSERT INTO documents
                (document_id, title, raw_text, source, tags, permission, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                  title=excluded.title, raw_text=excluded.raw_text, source=excluded.source,
                  tags=excluded.tags, permission=excluded.permission, status=excluded.status,
                  error=excluded.error, updated_at=excluded.updated_at
                """,
                (
                    doc.document_id,
                    doc.title,
                    doc.raw_text,
                    doc.source,
                    json.dumps(list(doc.tags), ensure_ascii=False),
                    doc.permission,
                    doc.status,
                    doc.error,
                    doc.created_at or now,
                    doc.updated_at or now,
                ),
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connection_scope() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._decode_document(row) if row else None

    def list_documents(
        self, *, tag: str = "", status: str = "", page: int = 1, page_size: int = 20
    ) -> tuple[list[DocumentRecord], int]:
        """Return ``(items, total)``; ``total`` ignores paging but keeps filters."""
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("page must be a positive integer")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
            raise ValueError("page_size must be a positive integer")
        where = (
            "WHERE (? = '' OR EXISTS (SELECT 1 FROM json_each(documents.tags) WHERE json_each.value = ?))"
            "  AND (? = '' OR status = ?)"
        )
        params = (tag, tag, status, status)
        with self._connection_scope() as connection:
            total = connection.execute(
                f"SELECT count(*) AS n FROM documents {where}", params
            ).fetchone()["n"]
            rows = connection.execute(
                f"SELECT * FROM documents {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()
        return [self._decode_document(row) for row in rows], int(total)

    def count_documents(self) -> int:
        with self._connection_scope() as connection:
            return int(connection.execute("SELECT count(*) AS n FROM documents").fetchone()["n"])

    def set_status(self, document_id: str, status: str, *, error: str | None = None) -> None:
        self._check_choice("status", status, DOCUMENT_STATUSES)
        with self._connection_scope() as connection:
            connection.execute(
                "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE document_id = ?",
                (status, error, utc_now().isoformat(), document_id),
            )

    def delete_document(self, document_id: str) -> int:
        """Delete a document and its chunks; returns the number of chunks removed."""
        with self._connection_scope() as connection:
            cursor = connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            removed = cursor.rowcount
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return int(removed)

    # -- chunks --------------------------------------------------------
    def upsert_chunk(self, chunk: ChunkRecord) -> None:
        if not isinstance(chunk, ChunkRecord):
            raise TypeError("chunk must be a ChunkRecord")
        self._check_choice("vector_status", chunk.vector_status, CHUNK_VECTOR_STATUSES)
        with self._connection_scope() as connection:
            self._write_chunk(connection, chunk)

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        for chunk in chunks:
            if not isinstance(chunk, ChunkRecord):
                raise TypeError("chunks must contain ChunkRecord instances")
            self._check_choice("vector_status", chunk.vector_status, CHUNK_VECTOR_STATUSES)
        with self._connection_scope() as connection:
            for chunk in chunks:
                self._write_chunk(connection, chunk)

    @staticmethod
    def _write_chunk(connection: sqlite3.Connection, chunk: ChunkRecord) -> None:
        connection.execute(
            """
            INSERT INTO chunks
            (chunk_id, document_id, chunk_index, char_start, char_end, text, vector_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              document_id=excluded.document_id, chunk_index=excluded.chunk_index,
              char_start=excluded.char_start, char_end=excluded.char_end,
              text=excluded.text, vector_status=excluded.vector_status
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.char_start,
                chunk.char_end,
                chunk.text,
                chunk.vector_status,
            ),
        )

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        with self._connection_scope() as connection:
            row = connection.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._decode_chunk(row) if row else None

    def list_chunks(self, document_id: str) -> list[ChunkRecord]:
        with self._connection_scope() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
            ).fetchall()
        return [self._decode_chunk(row) for row in rows]

    def set_chunk_vector_status(self, chunk_id: str, status: str) -> None:
        self._check_choice("vector_status", status, CHUNK_VECTOR_STATUSES)
        with self._connection_scope() as connection:
            connection.execute(
                "UPDATE chunks SET vector_status = ? WHERE chunk_id = ?", (status, chunk_id)
            )

    # -- keyword search (FTS5) -----------------------------------------
    @staticmethod
    def _fts_query(query: str) -> str:
        """Turn a raw user query into a safe FTS5 phrase: quote it, double inner quotes.

        Mandatory, not theoretical: bare ``abc-123`` makes FTS5 read ``-`` as
        column syntax and raises ``no such column: 123`` (same for ``型号: X200``),
        which surfaced as a 500 before this existed.
        """

        return '"' + query.replace('"', '""') + '"'

    def search_keywords(self, query: str, *, limit: int = 10) -> list[tuple[str, float]]:
        """BM25 keyword search over chunks; higher score means more relevant."""
        if self.fts_tokenizer is None or not isinstance(query, str) or not query.strip():
            return []
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._connection_scope() as connection:
            rows = connection.execute(
                "SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS rank "
                "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (self._fts_query(query), limit),
            ).fetchall()
        return [(row["chunk_id"], -float(row["rank"])) for row in rows]

    # -- reporting -----------------------------------------------------
    def chunk_counts(self) -> dict[str, int]:
        """``{document_id: chunk_count}`` in one query (list view needs it per row)."""

        with self._connection_scope() as connection:
            rows = connection.execute(
                "SELECT document_id, count(*) AS n FROM chunks GROUP BY document_id"
            ).fetchall()
        return {row["document_id"]: int(row["n"]) for row in rows}

    def chunk_ids(self, *, vector_status: str = "") -> list[str]:
        """Chunk ids, optionally filtered by ``vector_status`` (reconcile input)."""

        if vector_status:
            self._check_choice("vector_status", vector_status, CHUNK_VECTOR_STATUSES)
        with self._connection_scope() as connection:
            if vector_status:
                rows = connection.execute(
                    "SELECT chunk_id FROM chunks WHERE vector_status = ? ORDER BY chunk_id",
                    (vector_status,),
                ).fetchall()
            else:
                rows = connection.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()
        return [row["chunk_id"] for row in rows]

    def stats(self) -> dict[str, int]:
        with self._connection_scope() as connection:
            documents = connection.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
            chunks = connection.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
            indexed = connection.execute(
                "SELECT count(*) AS n FROM chunks WHERE vector_status = 'indexed'"
            ).fetchone()["n"]
        return {"documents": int(documents), "chunks": int(chunks), "chunks_indexed": int(indexed)}

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _check_choice(field_name: str, value: Any, allowed: tuple[str, ...]) -> None:
        if value not in allowed:
            raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}")

    @staticmethod
    def _decode_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            title=row["title"],
            raw_text=row["raw_text"],
            source=row["source"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            permission=row["permission"],
            status=row["status"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _decode_chunk(row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            text=row["text"],
            vector_status=row["vector_status"],
        )


__all__ = [
    "CHUNK_VECTOR_STATUSES",
    "DOCUMENT_STATUSES",
    "FTS_TOKENIZERS",
    "PERMISSIONS",
    "ChunkRecord",
    "DocumentRecord",
    "DocumentRepository",
]
