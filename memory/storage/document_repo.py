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
from uuid import uuid4

from ..base import utc_now

#: ``documents.status`` state machine (方案 2.1).
DOCUMENT_STATUSES = ("uploaded", "parsed", "vectorized", "extracted", "failed")
#: ``ingest_jobs.status`` 状态机（一句话后台入库队列，重启续跑）。
INGEST_JOB_STATUSES = ("pending", "running", "done", "failed")
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


@dataclass(frozen=True)
class EmbeddingLockRecord:
    """Single-row lock of the embedding space this SQLite file belongs to."""

    model: str
    dimension: int
    updated_at: str = ""


@dataclass
class IngestJobRecord:
    """One background one-sentence ingestion job, durable across restarts.

    ``status`` 在 SQLite 里随时可查（排队中/正在入库/成功/失败）；``result``
    存完成后的 JSON 摘要（块数 + 抽取报告），历史记录页直接展示。
    """

    job_id: str
    text: str
    event_at: str = ""
    kind: str = "sentence"
    status: str = "pending"
    attempts: int = 0
    error: str = ""
    result: str = ""
    created_at: str = ""
    updated_at: str = ""


class DocumentRepository:
    """CRUD for ``documents``/``chunks`` on the shared memory SQLite file.

    Mirrors :class:`~memory.storage.document.SQLiteDocumentStore`'s connection
    and locking model: one connection per call for file-backed databases, an
    ``RLock`` around every scope, and ``:memory:`` pinned to a single
    connection so tests can use a scratch database.
    """

    def __init__(self, path: str | Path = ":memory:", *, connection: sqlite3.Connection | None = None) -> None:
        self.path = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        #: ``trigram`` / ``unicode61``，FTS5 不可用时为 ``None``（检索退化为纯向量）。
        self.fts_tokenizer: str | None = None
        if connection is not None:
            # F2：注入的 ``:memory:`` 连接（与 ``memories`` 同库复用），仅测试用。
            self._connection = connection
            self._connection.row_factory = sqlite3.Row
        elif self.path == ":memory:":
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
            # F2：待确认删除提议（LLM 先提议、用户确认后才真删）。
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS delete_proposals (
                    proposal_id   TEXT PRIMARY KEY,
                    created_at    TEXT NOT NULL,
                    expires_at    TEXT NOT NULL,
                    requested_by  TEXT NOT NULL,
                    reason        TEXT NOT NULL,
                    item_ids      TEXT NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    confirmed_at  TEXT,
                    confirm_token TEXT
                )
                """
            )
            # 一句话后台入库队列：提交即返回，worker 落库推进状态，重启续跑。
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id     TEXT PRIMARY KEY,
                    kind       TEXT NOT NULL DEFAULT 'sentence',
                    text       TEXT NOT NULL,
                    event_at   TEXT NOT NULL DEFAULT '',
                    status     TEXT NOT NULL DEFAULT 'pending',
                    attempts   INTEGER NOT NULL DEFAULT 0,
                    error      TEXT NOT NULL DEFAULT '',
                    result     TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status ON ingest_jobs(status)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_lock (
                    id         INTEGER PRIMARY KEY CHECK (id = 1),
                    model      TEXT NOT NULL,
                    dimension  INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
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
        params = (tag, tag, status, status)
        with self._connection_scope() as connection:
            total = connection.execute(
                "SELECT count(*) AS n FROM documents "
                "WHERE (? = '' OR EXISTS (SELECT 1 FROM json_each(documents.tags) "
                "WHERE json_each.value = ?))"
                "  AND (? = '' OR status = ?)",
                params,
            ).fetchone()["n"]
            rows = connection.execute(
                "SELECT * FROM documents "
                "WHERE (? = '' OR EXISTS (SELECT 1 FROM json_each(documents.tags) "
                "WHERE json_each.value = ?))"
                "  AND (? = '' OR status = ?) "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
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

    def list_all_chunks(self) -> list[ChunkRecord]:
        """Every chunk in document order (full projection rebuild)."""

        with self._connection_scope() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks ORDER BY document_id, chunk_index"
            ).fetchall()
        return [self._decode_chunk(row) for row in rows]

    def get_embedding_lock(self) -> EmbeddingLockRecord | None:
        with self._connection_scope() as connection:
            row = connection.execute(
                "SELECT model, dimension, updated_at FROM embedding_lock WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        return EmbeddingLockRecord(
            model=str(row["model"]),
            dimension=int(row["dimension"]),
            updated_at=str(row["updated_at"] or ""),
        )

    def set_embedding_lock(self, model: str, dimension: int) -> EmbeddingLockRecord:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("embedding lock model must be a non-empty string")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("embedding lock dimension must be a positive integer")
        now = utc_now().isoformat()
        with self._connection_scope() as connection:
            connection.execute(
                """
                INSERT INTO embedding_lock (id, model, dimension, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  model=excluded.model, dimension=excluded.dimension, updated_at=excluded.updated_at
                """,
                (model.strip(), dimension, now),
            )
        return EmbeddingLockRecord(model=model.strip(), dimension=dimension, updated_at=now)

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

    # -- 一句话后台入库队列 ---------------------------------------------
    def create_ingest_job(self, text: str, *, kind: str = "sentence", event_at: str = "") -> IngestJobRecord:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ingest job text must be a non-empty string")
        now = utc_now().isoformat()
        record = IngestJobRecord(
            job_id=f"job_{uuid4().hex}",
            text=text,
            event_at=event_at,
            kind=kind,
            created_at=now,
            updated_at=now,
        )
        with self._connection_scope() as connection:
            connection.execute(
                """
                INSERT INTO ingest_jobs
                (job_id, kind, text, event_at, status, attempts, error, result, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', 0, '', '', ?, ?)
                """,
                (record.job_id, record.kind, record.text, record.event_at, now, now),
            )
        return record

    def get_ingest_job(self, job_id: str) -> IngestJobRecord | None:
        with self._connection_scope() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _ingest_job_from_row(row) if row is not None else None

    def set_ingest_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error: str = "",
        result: str = "",
    ) -> bool:
        """Advance a job; moving to ``running`` also counts one attempt."""

        self._check_choice("status", status, INGEST_JOB_STATUSES)
        now = utc_now().isoformat()
        with self._connection_scope() as connection:
            cursor = connection.execute(
                """
                UPDATE ingest_jobs SET
                  status = :status,
                  error = CASE WHEN :error != '' THEN :error ELSE error END,
                  result = CASE WHEN :result != '' THEN :result ELSE result END,
                  attempts = attempts + :attempt,
                  updated_at = :now
                WHERE job_id = :job_id
                """,
                {
                    "status": status,
                    "error": error,
                    "result": result,
                    "attempt": 1 if status == "running" else 0,
                    "now": now,
                    "job_id": job_id,
                },
            )
        return cursor.rowcount > 0

    def list_ingest_jobs(self, *, status: str | None = None, limit: int = 50) -> list[IngestJobRecord]:
        if status is not None:
            self._check_choice("status", status, INGEST_JOB_STATUSES)
        query = "SELECT * FROM ingest_jobs"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._connection_scope() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_ingest_job_from_row(row) for row in rows]

    def restart_stale_ingest_jobs(self) -> int:
        """Reset ``running`` jobs to ``pending``（进程上次异常退出）。"""

        with self._connection_scope() as connection:
            cursor = connection.execute(
                "UPDATE ingest_jobs SET status = 'pending', updated_at = ? WHERE status = 'running'",
                (utc_now().isoformat(),),
            )
        return cursor.rowcount

    def reset_failed_ingest_job(self, job_id: str) -> IngestJobRecord | None:
        """User retry: only ``failed`` jobs go back to ``pending`` with a clean slate."""

        now = utc_now().isoformat()
        with self._connection_scope() as connection:
            cursor = connection.execute(
                """
                UPDATE ingest_jobs SET
                  status = 'pending',
                  error = '',
                  result = '',
                  attempts = 0,
                  updated_at = ?
                WHERE job_id = ? AND status = 'failed'
                """,
                (now, job_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_ingest_job(job_id)

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


#: F2 ``delete_proposals.status`` 状态机。
DELETION_PROPOSAL_STATUSES = ("pending", "confirmed", "rejected", "expired")
#: F2 提议默认有效期（分钟）；过期后确认被拒绝。
DELETION_PROPOSAL_TTL_MINUTES = 15


@dataclass
class DeletionProposal:
    """One pending deletion: what to remove, who asked, until when."""

    proposal_id: str
    created_at: str
    expires_at: str
    requested_by: str
    reason: str
    item_ids: list[str]
    status: str = "pending"
    confirmed_at: str = ""
    confirm_token: str = ""


class DeletionProposalStore:
    """CRUD for ``delete_proposals`` on the shared memory SQLite file.

    Mirrors :class:`DocumentRepository`'s connection/locking model. The store
    only records proposals — it never deletes memories itself; execution goes
    through :meth:`execute_deletion` after the user confirms.
    """

    def __init__(self, path: str | Path = ":memory:", *, connection: sqlite3.Connection | None = None) -> None:
        self._repo = DocumentRepository(path, connection=connection)

    @property
    def path(self) -> str:
        return self._repo.path

    def create(self, *, requested_by: str, reason: str, item_ids: list[str], ttl_minutes: int = DELETION_PROPOSAL_TTL_MINUTES) -> DeletionProposal:
        if not isinstance(requested_by, str) or not requested_by.strip():
            raise ValueError("requested_by must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not item_ids or not all(isinstance(value, str) and value.strip() for value in item_ids):
            raise ValueError("item_ids must be a non-empty list of item ids")
        if isinstance(ttl_minutes, bool) or not isinstance(ttl_minutes, int) or ttl_minutes < 1:
            raise ValueError("ttl_minutes must be a positive integer")
        now = utc_now()
        from datetime import timedelta
        from uuid import uuid4

        token = uuid4().hex[:8].upper()
        proposal = DeletionProposal(
            proposal_id=str(uuid4()),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=ttl_minutes)).isoformat(),
            requested_by=requested_by,
            reason=reason,
            item_ids=list(dict.fromkeys(item_ids)),
            confirm_token=token,
        )
        with self._repo._connection_scope() as connection:
            connection.execute(
                """
                INSERT INTO delete_proposals
                (proposal_id, created_at, expires_at, requested_by, reason, item_ids, status, confirmed_at, confirm_token)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, ?)
                """,
                (
                    proposal.proposal_id,
                    proposal.created_at,
                    proposal.expires_at,
                    proposal.requested_by,
                    proposal.reason,
                    json.dumps(proposal.item_ids, ensure_ascii=False),
                    proposal.confirm_token,
                ),
            )
        return proposal

    def get(self, proposal_id: str) -> DeletionProposal | None:
        """One proposal; a pending one past ``expires_at`` is marked expired."""

        with self._repo._connection_scope() as connection:
            row = connection.execute(
                "SELECT * FROM delete_proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        if row is None:
            return None
        proposal = self._decode(row)
        if proposal.status == "pending" and utc_now().isoformat() > proposal.expires_at:
            self._set_status(proposal_id, "expired")
            proposal.status = "expired"
        return proposal

    def _set_status(self, proposal_id: str, status: str, confirmed_at: str = "") -> None:
        with self._repo._connection_scope() as connection:
            connection.execute(
                "UPDATE delete_proposals SET status = ?, confirmed_at = ? WHERE proposal_id = ?",
                (status, confirmed_at or None, proposal_id),
            )

    def confirm(self, proposal_id: str, token: str) -> DeletionProposal:
        """Atomically mark a pending, unexpired proposal confirmed.

        The state flip runs as a single guarded UPDATE so concurrent callers
        cannot both observe ``pending`` and confirm twice: only one row change
        wins. Raises ``ValueError`` for a missing, expired, already-confirmed
        or wrongly-tokened proposal — fail closed.
        """

        if not token:
            raise ValueError("confirmation token does not match")
        confirmed_at = utc_now().isoformat()
        with self._repo._connection_scope() as connection:
            cursor = connection.execute(
                "UPDATE delete_proposals SET status = 'confirmed', confirmed_at = ? "
                "WHERE proposal_id = ? AND status = 'pending' AND confirm_token = ?",
                (confirmed_at, proposal_id, token),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT * FROM delete_proposals WHERE proposal_id = ?", (proposal_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"unknown proposal: {proposal_id}")
                current = self._decode(row)
                if current.status == "expired":
                    raise ValueError("proposal has expired")
                if current.status != "pending" or token != current.confirm_token:
                    raise ValueError("confirmation token does not match")
                raise ValueError(f"proposal is not pending (status={current.status})")
            proposal = self._decode(
                connection.execute(
                    "SELECT * FROM delete_proposals WHERE proposal_id = ?", (proposal_id,)
                ).fetchone()
            )
            proposal.status = "confirmed"
            proposal.confirmed_at = confirmed_at
            return proposal

    @staticmethod
    def _decode(row: sqlite3.Row) -> DeletionProposal:
        return DeletionProposal(
            proposal_id=row["proposal_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            requested_by=row["requested_by"],
            reason=row["reason"],
            item_ids=json.loads(row["item_ids"]),
            status=row["status"],
            confirmed_at=row["confirmed_at"] or "",
            confirm_token=row["confirm_token"] or "",
        )

    def close(self) -> None:
        self._repo.close()


def execute_deletion(proposal_id: str, token: str, manager: Any) -> dict[str, Any]:
    """Execute one confirmed deletion proposal against the memory manager.

    The gate is the store: only a pending, unexpired proposal with a matching
    token executes, and repeating the same proposal never deletes twice
    (idempotent). Semantic facts are hard-deleted through
    ``SemanticMemory.delete`` (which also removes the Neo4j edge); an episodic
    audit record preserves who deleted what and why.
    """

    from ..manager import MemoryManager

    if not isinstance(manager, MemoryManager):
        raise TypeError("manager must be a MemoryManager")
    # ``:memory:`` 时与 memories 同库复用同一条连接（否则提议写进去、执行端读不到）；
    # 文件库传 None，走普通路径连接。
    connection = getattr(manager.document_store, "connection", None)
    store = DeletionProposalStore(manager.document_store.path, connection=connection)
    proposal = store.get(proposal_id)
    if proposal is None:
        raise ValueError(f"unknown proposal: {proposal_id}")
    if proposal.status == "confirmed":
        # 幂等：重复确认同一 proposal 直接返回既有结果，不重复删。
        return {"deleted": [], "already_confirmed": True, "proposal_id": proposal_id}
    store.confirm(proposal_id, token)
    deleted: list[str] = []
    for item_id in proposal.item_ids:
        item = manager.get(item_id)
        memory_type = item.memory_type if item is not None else None
        if manager.delete(item_id, memory_type=memory_type):
            deleted.append(item_id)
    manager.add(
        f"已删除 {len(deleted)} 条记忆（提议 {proposal_id[:8]}）：{proposal.reason}",
        memory_type="episodic",
        metadata={
            "kind": "deletion_audit",
            "proposal_id": proposal_id,
            "deleted_ids": deleted,
            "requested_by": proposal.requested_by,
            "reason": proposal.reason,
        },
    )
    return {"deleted": deleted, "already_confirmed": False, "proposal_id": proposal_id}


def _ingest_job_from_row(row: sqlite3.Row) -> IngestJobRecord:
    return IngestJobRecord(
        job_id=row["job_id"],
        text=row["text"],
        event_at=row["event_at"],
        kind=row["kind"],
        status=row["status"],
        attempts=row["attempts"],
        error=row["error"],
        result=row["result"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


__all__ = [
    "CHUNK_VECTOR_STATUSES",
    "DELETION_PROPOSAL_STATUSES",
    "DELETION_PROPOSAL_TTL_MINUTES",
    "DOCUMENT_STATUSES",
    "FTS_TOKENIZERS",
    "INGEST_JOB_STATUSES",
    "PERMISSIONS",
    "ChunkRecord",
    "DeletionProposal",
    "DeletionProposalStore",
    "DocumentRecord",
    "DocumentRepository",
    "EmbeddingLockRecord",
    "IngestJobRecord",
    "execute_deletion",
]
