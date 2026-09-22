"""SQLite 原始文档库：documents / ingest_queue / chunks / chunk_entities 四张表。

- documents      : 原始文档数据表（用户输入的原文）
- ingest_queue   : 入库状态表（每个 chunk 一行，记录 qdrant / neo4j 两条线的入库状态）
- chunks         : chunk 表（只有当 qdrant 与 neo4j 双双成功，才从队列「转正」到这里）
- chunk_entities : chunk <-> 实体 的多对多映射表
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app import config

TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    source      TEXT DEFAULT 'manual',
    meta        TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_queue (
    chunk_id        TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    seq             INTEGER DEFAULT 0,
    content         TEXT NOT NULL,
    qdrant_status   TEXT NOT NULL DEFAULT 'pending',   -- pending|success|failed
    neo4j_status    TEXT NOT NULL DEFAULT 'pending',
    chunker_status  TEXT NOT NULL DEFAULT 'pending',
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_queue_doc ON ingest_queue(document_id);
CREATE INDEX IF NOT EXISTS idx_queue_state ON ingest_queue(qdrant_status, neo4j_status);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content     TEXT NOT NULL,
    char_len    INTEGER DEFAULT 0,
    qdrant_status TEXT,
    neo4j_status  TEXT,
    created_at  TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id    TEXT NOT NULL,
    entity_key  TEXT NOT NULL,
    entity_name TEXT,
    entity_type TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (chunk_id, entity_key)
);
CREATE INDEX IF NOT EXISTS idx_ce_entity ON chunk_entities(entity_key);

-- FTS5 全文索引：多路混合检索里的「词法路」。
-- 用 trigram 分词器，原生支持中文（3 字以上子串即可命中），
-- 对英文则退化为更细粒度的匹配（影响小而可接受）。
"""


def _create_fts_schema(c: sqlite3.Connection) -> None:
    """单独建 FTS5，失败（比如 sqlite 未编译 FTS5）也不应阻塞主库。"""
    try:
        c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "chunk_id UNINDEXED, content, tokenize='trigram')"
        )
        # 把 chunks 表内容同步进 FTS 索引（幂等）
        c.execute("INSERT INTO chunks_fts(chunk_id, content) "
                  " SELECT c.chunk_id, c.content FROM chunks c "
                  " WHERE NOT EXISTS (SELECT 1 FROM chunks_fts f WHERE f.chunk_id = c.chunk_id)")
        c.commit()
    except sqlite3.Error as e:
        print(f"[sqlite] FTS5 不可用，多路混合检索将只走向量路: {e}")


class SQLiteStore:
    """同步实现，异步调用统一走 asyncio.to_thread（SQLite 写入很快，线程开销可接受）。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else config.app.sqlite_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.fts_ok = False
        with self._conn() as c:
            c.executescript(SCHEMA)
            try:
                _create_fts_schema(c)
                self.fts_ok = True
            except Exception as e:  # noqa: BLE001
                print(f"[sqlite] FTS5 初始化失败: {e}")

    # ---------------- 连接 ----------------
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.c = c
        return c

    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        if args is None:
            args = ()
        elif isinstance(args, str):
            args = (args,)          # 防御：禁止把裸字符串当参数序列透出
        with self._conn() as c:
            cur = c.execute(sql, tuple(args))
            return [dict(r) for r in cur.fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> dict | None:
        rs = self._rows(sql, args)
        return rs[0] if rs else None

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._conn() as c:
            cur = c.execute(sql, args)
            c.commit()
            return cur.rowcount

    @staticmethod
    def _rid(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    # ---------------- documents ----------------
    def add_document(self, content: str, source: str = "manual", meta: dict | None = None) -> dict:
        did = self._rid("doc")
        ts = now_iso()
        self._exec(
            "INSERT INTO documents(id, content, source, meta, created_at) VALUES(?,?,?,?,?)",
            (did, content, source, json.dumps(meta or {}, ensure_ascii=False), ts),
        )
        return {"id": did, "content": content, "source": source, "created_at": ts}

    def get_document(self, doc_id: str) -> dict | None:
        return self._one("SELECT * FROM documents WHERE id=?", (doc_id,))

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[dict]:
        return self._rows(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
        )

    def count_documents(self) -> int:
        return self._one("SELECT COUNT(*) AS n FROM documents")["n"]

    # ---------------- ingest_queue ----------------
    def enqueue_chunk(self, document_id: str, content: str, seq: int = 0) -> str:
        cid = self._rid("ck")
        ts = now_iso()
        self._exec(
            """INSERT INTO ingest_queue(chunk_id, document_id, seq, content,
                 qdrant_status, neo4j_status, created_at, updated_at)
               VALUES(?,?,?,?, 'pending','pending',?,?)""",
            (cid, document_id, seq, content, ts, ts),
        )
        return cid

    def get_queue_item(self, chunk_id: str) -> dict | None:
        return self._one("SELECT * FROM ingest_queue WHERE chunk_id=?", (chunk_id,))

    def list_pending(self, limit: int = 20, status: str | None = None) -> list[dict]:
        """未入库成功的所有 chunk（默认：qdrant 或 neo4j 任一未完成的都算）。"""
        if status == "pending":
            sql = ("SELECT * FROM ingest_queue WHERE qdrant_status!='success' OR neo4j_status!='success' "
                   "ORDER BY created_at ASC LIMIT ?")
            return self._rows(sql, (limit,))
        if status == "success":
            sql = ("SELECT * FROM ingest_queue WHERE qdrant_status='success' AND neo4j_status='success' "
                   "ORDER BY created_at ASC LIMIT ?")
            return self._rows(sql, (limit,))
        sql = ("SELECT * FROM ingest_queue WHERE qdrant_status!='success' OR neo4j_status!='success' "
               "ORDER BY created_at ASC LIMIT ?")
        return self._rows(sql, (limit,))

    def count_pending(self) -> int:
        return self._one(
            "SELECT COUNT(*) AS n FROM ingest_queue WHERE qdrant_status!='success' OR neo4j_status!='success'"
        )["n"]

    def count_queue(self) -> int:
        return self._one("SELECT COUNT(*) AS n FROM ingest_queue")["n"]

    def update_chunk_status(self, chunk_id: str, qdrant_status: str | None = None,
                            neo4j_status: str | None = None, error: str | None = None) -> None:
        sets, args = [], []
        if qdrant_status is not None:
            sets.append("qdrant_status=?")
            args.append(qdrant_status)
        if neo4j_status is not None:
            sets.append("neo4j_status=?")
            args.append(neo4j_status)
        if error is not None:
            sets.append("error=?")
            args.append(error)
        if not sets:
            return
        sets.append("updated_at=?")
        args.append(now_iso())
        args.append(chunk_id)
        self._exec(f"UPDATE ingest_queue SET {', '.join(sets)} WHERE chunk_id=?", tuple(args))

    def is_complete(self, chunk_id: str) -> bool:
        r = self._one(
            "SELECT qdrant_status, neo4j_status FROM ingest_queue WHERE chunk_id=?", (chunk_id,)
        )
        return bool(r and r["qdrant_status"] == "success" and r["neo4j_status"] == "success")

    # ---------------- chunks（转正） ----------------
    def promote_chunk(self, chunk_id: str) -> dict | None:
        """两条入库线都成功后，从 ingest_queue 转移到 chunks 表。"""
        row = self.get_queue_item(chunk_id)
        if not row or not self.is_complete(chunk_id):
            return None
        ts = now_iso()
        self._exec(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, document_id, content, char_len, qdrant_status, neo4j_status, created_at, ingested_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (chunk_id, row["document_id"], row["content"], len(row["content"]),
             row["qdrant_status"], row["neo4j_status"], row["created_at"], ts),
        )
        self._exec("DELETE FROM ingest_queue WHERE chunk_id=?", (chunk_id,))
        self.index_chunk_content(chunk_id, row["content"])
        return {"chunk_id": chunk_id, "document_id": row["document_id"],
                "content": row["content"], "ingested_at": ts}

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._one("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))

    def list_chunks(self, limit: int = 50, offset: int = 0) -> list[dict]:
        return self._rows(
            "SELECT * FROM chunks ORDER BY ingested_at DESC LIMIT ? OFFSET ?", (limit, offset)
        )

    def count_chunks(self) -> int:
        return self._one("SELECT COUNT(*) AS n FROM chunks")["n"]

    # ---------------- chunk_entities 映射 ----------------
    def set_chunk_entities(self, chunk_id: str, entities: list[dict]) -> int:
        ts = now_iso()
        with self._conn() as c:
            c.execute("DELETE FROM chunk_entities WHERE chunk_id=?", (chunk_id,))
            for e in entities:
                key = (e.get("key") or e.get("name") or "").strip()
                if not key:
                    continue
                c.execute(
                    "INSERT OR REPLACE INTO chunk_entities(chunk_id, entity_key, entity_name, entity_type, created_at)"
                    " VALUES(?,?,?,?,?)",
                    (chunk_id, key, e.get("name") or key, e.get("type") or "", ts),
                )
            c.commit()
        return len(entities)

    def get_chunk_entities(self, chunk_id: str) -> list[dict]:
        return self._rows("SELECT * FROM chunk_entities WHERE chunk_id=?", (chunk_id,))

    def get_entities_of_chunk(self, chunk_id: str) -> list[dict]:
        return self.get_chunk_entities(chunk_id)

    def get_chunks_of_entity(self, entity_key: str) -> list[dict]:
        """一个实体可以来自多个 chunk —— 反向映射。"""
        return self._rows(
            """SELECT ce.chunk_id, c.content, c.ingested_at
               FROM chunk_entities ce JOIN chunks c ON c.chunk_id = ce.chunk_id
               WHERE ce.entity_key=? ORDER BY c.ingested_at DESC""",
            (entity_key,),
        )

    def recent_entity_keys(self, limit: int = 100) -> list[str]:
        return [r["entity_key"] for r in self._rows(
            "SELECT DISTINCT entity_key FROM chunk_entities ORDER BY created_at DESC LIMIT ?", (limit,)
        )]

    # ---------------- 统计 ----------------
    def fts_search(self, query: str, limit: int = 20) -> list[dict]:
        """FTS5 全文检索（多路混合检索的词法路）。

        中文检索的关键：trigram 分词器只认 3 字子串，直接拿整句去 MATCH 必然 0 命中。
        所以这里把查询切成「短词条 + 长词条的 n-gram 窗口」以后，用 OR 组合提高召回，
        再按 FTS5 内置 bm25() 排序（越相关越靠前，返回 rank 1..k）。
        """
        if not self.fts_ok:
            return []
        q = (query or "").strip()
        if not q:
            return []

        runs = [r for r in re.findall(r"[0-9A-Za-z\u4e00-\u9fff]+", q)]
        terms: list[str] = []
        for run in runs:
            if len(run) <= 6:
                terms.append(run)
            else:
                # 长词条切 n-gram 窗口，保证中文子串一定能在 trigram 索引里命中
                terms.extend(run[i:i + 3] for i in range(0, len(run) - 2))
        # 去重保序，限制规模
        seen, uniq = set(), []
        for t in terms:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        if not uniq:
            return []

        match = " OR ".join(f'"{t}"' for t in uniq[:24])
        try:
            with self._conn() as c:
                cur = c.execute(
                    """SELECT chunk_id, content, bm25(chunks_fts) AS bm
                       FROM chunks_fts WHERE chunks_fts MATCH ?
                       ORDER BY bm ASC LIMIT ?""",
                    (match, limit),
                )
                rows = [dict(r) for r in cur.fetchall()]
        except sqlite3.Error:
            return []
        return [
            {"chunk_id": r["chunk_id"], "content": r["content"],
             "rank": i + 1, "engine": "fts5", "score": round(1.0 / (i + 1), 6),
             "bm25": round(float(r["bm"] or 0), 5)}
            for i, r in enumerate(rows)
        ]

    def index_chunk_content(self, chunk_id: str, content: str) -> None:
        """chunk 转正时同步写入 FTS 索引。"""
        if not self.fts_ok:
            return
        try:
            with self._conn() as c:
                c.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
                c.execute("INSERT INTO chunks_fts(chunk_id, content) VALUES(?,?)", (chunk_id, content))
                c.commit()
        except sqlite3.Error:
            pass

    def fts_rebuild(self) -> dict:
        if not self.fts_ok:
            return {"ok": False, "error": "FTS5 不可用"}
        n = 0
        with self._conn() as c:
            c.execute("DELETE FROM chunks_fts")
            for r in c.execute("SELECT chunk_id, content FROM chunks").fetchall():
                c.execute("INSERT INTO chunks_fts(chunk_id, content) VALUES(?,?)",
                          (r["chunk_id"], r["content"]))
                n += 1
            c.commit()
        return {"ok": True, "indexed": n}

    def stats(self) -> dict:
        return {
            "documents": self.count_documents(),
            "queue": self.count_queue(),
            "pending": self.count_pending(),
            "chunks": self.count_chunks(),
            "entity_links": self._one("SELECT COUNT(*) AS n FROM chunk_entities")["n"],
            "fts_ok": self.fts_ok,
            "fts_rows": (self._one("SELECT COUNT(*) AS n FROM chunks_fts")["n"]
                         if self.fts_ok else 0),
            "db_path": str(self.path),
        }

    def clear_all(self) -> None:
        for t in ("chunk_entities", "chunks", "ingest_queue", "documents"):
            self._exec(f"DELETE FROM {t}")
        if self.fts_ok:
            # FTS5 是虚拟表，上面的循环不会碰它，必须单独清
            self._exec("DELETE FROM chunks_fts")


# 单例
store = SQLiteStore()
