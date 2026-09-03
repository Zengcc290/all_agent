from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import ToolSpec


class ToolSpecRepository:
    """SQLite persistence for discoverable tool metadata, never executable code."""

    def __init__(self, path: str | Path = "tools.sqlite3") -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")
        self.path = str(path)
        self._lock = threading.RLock()
        self._shared_connection: sqlite3.Connection | None = None
        if self.path == ":memory:":
            self._shared_connection = sqlite3.connect(
                self.path, check_same_thread=False
            )
            self._shared_connection.row_factory = sqlite3.Row
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._shared_connection is not None:
            return self._shared_connection
        if self.path == ":memory:":
            raise RuntimeError("repository is closed")
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a serialized shared connection or a short-lived file connection."""
        with self._lock:
            connection = self._connect()
            try:
                with connection:
                    yield connection
            finally:
                if connection is not self._shared_connection:
                    connection.close()

    def close(self) -> None:
        """Close a shared in-memory connection, if one is in use."""
        with self._lock:
            if self._shared_connection is not None:
                self._shared_connection.close()
                self._shared_connection = None

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_specs (
                    tool_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT NOT NULL,
                    schema_hash TEXT NOT NULL,
                    input_schema TEXT NOT NULL,
                    output_schema TEXT NOT NULL,
                    side_effect TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    timeout_seconds REAL NOT NULL,
                    idempotent INTEGER NOT NULL,
                    parallel_safe INTEGER NOT NULL,
                    max_concurrency INTEGER,
                    tags TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    implementation_ref TEXT,
                    PRIMARY KEY (tool_name, version)
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(tool_specs)")
            }
            if "max_concurrency" not in columns:
                connection.execute(
                    "ALTER TABLE tool_specs ADD COLUMN max_concurrency INTEGER"
                )

    def save(
        self,
        spec: ToolSpec,
        *,
        implementation_ref: str | None = None,
        replace: bool = False,
    ) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("spec must be a ToolSpec instance")
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean")
        if implementation_ref is not None and not isinstance(implementation_ref, str):
            raise TypeError("implementation_ref must be a string or None")
        statement = "INSERT OR REPLACE" if replace else "INSERT"
        with self._connection() as connection:
            connection.execute(
                f"""
                {statement} INTO tool_specs (
                    tool_name, version, description, schema_hash, input_schema, output_schema,
                    side_effect, permissions, timeout_seconds, idempotent, parallel_safe,
                    max_concurrency, tags, enabled, implementation_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    spec.name,
                    spec.version,
                    spec.description,
                    spec.schema_hash,
                    json.dumps(spec.input_schema, sort_keys=True),
                    json.dumps(spec.output_schema, sort_keys=True),
                    spec.side_effect,
                    json.dumps(list(spec.permissions)),
                    spec.timeout_seconds,
                    int(spec.idempotent),
                    int(spec.parallel_safe),
                    spec.max_concurrency,
                    json.dumps(list(spec.tags)),
                    implementation_ref,
                ),
            )

    def get(self, tool_name: str, version: str | None = None) -> dict[str, Any] | None:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if len(tool_name) > 200:
            raise ValueError("tool_name must be at most 200 characters")
        if version is not None and (
            not isinstance(version, str) or not version.strip()
        ):
            raise ValueError("version must be a non-empty string or None")
        if isinstance(version, str) and len(version) > 32:
            raise ValueError("version must be at most 32 characters")
        query = "SELECT * FROM tool_specs WHERE tool_name = ? AND enabled = 1"
        params: list[Any] = [tool_name]
        if version is not None:
            query += " AND version = ?"
            params.append(version)
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        if version is not None:
            return self._decode(rows[0]) if rows else None
        decoded = [self._decode(row) for row in rows]
        return (
            max(decoded, key=lambda item: self._version_key(item["version"]))
            if decoded
            else None
        )

    def search(self, intent: str, limit: int = 5) -> list[dict[str, Any]]:
        if not isinstance(intent, str):
            raise TypeError("intent must be a string")
        if len(intent) > 500:
            raise ValueError("intent must be at most 500 characters")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 20
        ):
            raise ValueError("limit must be an integer between 1 and 20")
        terms = [term for term in intent.casefold().split() if term]
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tool_specs WHERE enabled = 1"
            ).fetchall()
        decoded = [self._decode(row) for row in rows]
        latest: dict[str, dict[str, Any]] = {}
        for item in decoded:
            current = latest.get(item["tool_name"])
            if current is None or self._version_key(
                item["version"]
            ) > self._version_key(current["version"]):
                latest[item["tool_name"]] = item
        decoded = sorted(latest.values(), key=lambda item: item["tool_name"])
        if not terms:
            return decoded[:limit]
        ranked = []
        for item in decoded:
            haystack = f"{item['tool_name']} {item['description']} {' '.join(item['tags'])}".casefold()
            ranked.append((sum(term in haystack for term in terms), item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["tool_name"]))
        return [item for score, item in ranked if score > 0][:limit]

    @staticmethod
    def _version_key(version: str) -> tuple[object, object]:
        """Sort numeric versions naturally while keeping pre-releases older.

        The first component compares the version core (``10`` after ``2``),
        then a release marker makes ``1.0`` newer than ``1.0-alpha``. Labels
        that are not semantic-version components remain sortable strings.
        """
        normalized = version.strip()
        core, separator, suffix = normalized.partition("-")
        parts = re.findall(r"\d+|\D+", core)
        core_key = tuple(
            (0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts
        )
        release_key = 1 if not separator else 0
        suffix_key = tuple(
            (0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.findall(r"\d+|\D+", suffix)
        )
        return core_key, (release_key, suffix_key)

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("input_schema", "output_schema", "permissions", "tags"):
            item[key] = json.loads(item[key])
        item["enabled"] = bool(item["enabled"])
        item["idempotent"] = bool(item["idempotent"])
        item["parallel_safe"] = bool(item["parallel_safe"])
        return item
