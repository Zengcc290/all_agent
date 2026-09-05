"""SQLite persistence for compact, append-only project update logs.

The repository intentionally returns only the row needed by a writer (the new
ID and timestamp). Historical entries stay in SQLite instead of being loaded
into an AI context window.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_UPDATE_LOG_FILENAME = "update_log.sqlite3"


class UpdateLogRepository:
    """Thread-safe SQLite store for one immutable row per project change."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = os.getenv("UPDATE_LOG_DB_PATH") or DEFAULT_UPDATE_LOG_FILENAME
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string, pathlib.Path, or None")
        # Normalize the path before opening sqlite so ``~/...`` behaves the
        # same way for directory creation and subsequent connections.
        self.path = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        self._lock = threading.RLock()
        self._shared_connection: sqlite3.Connection | None = None
        if self.path == ":memory:":
            self._shared_connection = sqlite3.connect(
                self.path, check_same_thread=False, timeout=10
            )
            self._shared_connection.row_factory = sqlite3.Row
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._shared_connection is not None:
            return self._shared_connection
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA busy_timeout = 10000")
                connection.execute("PRAGMA foreign_keys = ON")
                with connection:
                    yield connection
            finally:
                if connection is not self._shared_connection:
                    connection.close()

    def close(self) -> None:
        """Close an in-memory connection; file-backed repositories are per-call."""
        with self._lock:
            if self._shared_connection is not None:
                self._shared_connection.close()
                self._shared_connection = None

    def _initialize(self) -> None:
        if self.path != ":memory:":
            parent = Path(self.path).parent
            if str(parent) not in {"", "."}:
                parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS update_logs (
                    update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    system_name TEXT NOT NULL,
                    executor TEXT NOT NULL,
                    update_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    task_background TEXT NOT NULL,
                    update_details TEXT NOT NULL,
                    added_features TEXT NOT NULL,
                    files_json TEXT NOT NULL,
                    behavior_impact TEXT NOT NULL,
                    validation TEXT NOT NULL,
                    risks TEXT NOT NULL,
                    follow_up TEXT NOT NULL
                )
                """
            )

    def append(
        self,
        *,
        executor: str,
        update_type: str,
        title: str,
        task_background: str,
        update_details: str,
        added_features: str,
        files: Sequence[Mapping[str, str]],
        behavior_impact: str,
        validation: str,
        risks: str,
        follow_up: str,
        system_name: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Append one row atomically and return a compact acknowledgement.

        ``timestamp`` and ``system_name`` are optional only for controlled data
        migration. Normal callers should omit them so the repository captures
        the real write time and host platform.
        """
        values = {
            "executor": executor,
            "update_type": update_type,
            "title": title,
            "task_background": task_background,
            "update_details": update_details,
            "added_features": added_features,
            "behavior_impact": behavior_impact,
            "validation": validation,
            "risks": risks,
            "follow_up": follow_up,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
            raise TypeError("files must be a sequence of mappings")
        normalized_files: list[dict[str, str]] = []
        for item in files:
            if not isinstance(item, Mapping):
                raise TypeError("each file change must be a mapping")
            path = item.get("path")
            action = item.get("action")
            description = item.get("description")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (path, action, description)
            ):
                raise ValueError(
                    "each file change requires non-empty path, action and description"
                )
            normalized_files.append(
                {"path": path.strip(), "action": action.strip(), "description": description.strip()}
            )
        if not normalized_files:
            raise ValueError("files must contain at least one file change")

        normalized_system = (system_name or platform.system() or "Unknown").strip()
        if not normalized_system:
            normalized_system = "Unknown"
        normalized_timestamp = timestamp or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        if not isinstance(normalized_timestamp, str) or not normalized_timestamp.strip():
            raise ValueError("timestamp must be a non-empty string")

        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO update_logs (
                    timestamp, system_name, executor, update_type, title,
                    task_background, update_details, added_features, files_json,
                    behavior_impact, validation, risks, follow_up
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_timestamp,
                    normalized_system,
                    values["executor"].strip(),
                    values["update_type"].strip(),
                    values["title"].strip(),
                    values["task_background"].strip(),
                    values["update_details"].strip(),
                    values["added_features"].strip(),
                    json.dumps(normalized_files, ensure_ascii=False, sort_keys=True),
                    values["behavior_impact"].strip(),
                    values["validation"].strip(),
                    values["risks"].strip(),
                    values["follow_up"].strip(),
                ),
            )
            update_id = int(cursor.lastrowid)
        return {
            "update_id": update_id,
            "timestamp": normalized_timestamp,
            "system_name": normalized_system,
            "next_update_id": update_id + 1,
        }

    def get(self, update_id: int) -> dict[str, Any] | None:
        """Return one decoded entry for audits and tests without bulk loading."""
        if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id < 1:
            raise ValueError("update_id must be a positive integer")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM update_logs WHERE update_id = ?", (update_id,)
            ).fetchone()
        return self._decode(row) if row is not None else None

    def latest_id(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT MAX(update_id) AS value FROM update_logs").fetchone()
        return int(row["value"] or 0)

    def get_range(self, start_id: int, end_id: int) -> list[dict[str, Any]]:
        """Return a contiguous range using one SQLite connection/query.

        The old caller pattern opened a connection for every ID.  That made a
        full audit progressively slow and could look stuck when a model asked
        for the same range repeatedly.
        """
        if (
            isinstance(start_id, bool)
            or not isinstance(start_id, int)
            or start_id < 1
            or isinstance(end_id, bool)
            or not isinstance(end_id, int)
            or end_id < start_id
        ):
            raise ValueError("start_id and end_id must be positive integers")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM update_logs WHERE update_id BETWEEN ? AND ? "
                "ORDER BY update_id ASC",
                (start_id, end_id),
            ).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["files"] = json.loads(item.pop("files_json"))
        return item


__all__ = ["DEFAULT_UPDATE_LOG_FILENAME", "UpdateLogRepository"]
