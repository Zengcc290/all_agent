from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import ToolSpec


class ToolSpecRepository:
    """SQLite persistence for discoverable tool metadata, never executable code."""

    def __init__(self, path: str | Path = "tools.sqlite3") -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
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
                    tags TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    implementation_ref TEXT,
                    PRIMARY KEY (tool_name, version)
                )
                """
            )

    def save(self, spec: ToolSpec, *, implementation_ref: str | None = None, replace: bool = False) -> None:
        statement = "INSERT OR REPLACE" if replace else "INSERT"
        with self._connect() as connection:
            connection.execute(
                f"""
                {statement} INTO tool_specs (
                    tool_name, version, description, schema_hash, input_schema, output_schema,
                    side_effect, permissions, timeout_seconds, idempotent, parallel_safe, tags,
                    enabled, implementation_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
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
                    json.dumps(list(spec.tags)),
                    implementation_ref,
                ),
            )

    def get(self, tool_name: str, version: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM tool_specs WHERE tool_name = ? AND enabled = 1"
        params: list[Any] = [tool_name]
        if version is not None:
            query += " AND version = ?"
            params.append(version)
        else:
            query += " ORDER BY version DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._decode(row) if row else None

    def search(self, intent: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = [term for term in intent.casefold().split() if term]
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tool_specs WHERE enabled = 1 ORDER BY tool_name, version DESC"
            ).fetchall()
        decoded = [self._decode(row) for row in rows]
        if not terms:
            return decoded[:limit]
        ranked = []
        for item in decoded:
            haystack = f"{item['tool_name']} {item['description']} {' '.join(item['tags'])}".casefold()
            ranked.append((sum(term in haystack for term in terms), item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["tool_name"]))
        return [item for score, item in ranked if score > 0][:limit]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("input_schema", "output_schema", "permissions", "tags"):
            item[key] = json.loads(item[key])
        item["enabled"] = bool(item["enabled"])
        item["idempotent"] = bool(item["idempotent"])
        item["parallel_safe"] = bool(item["parallel_safe"])
        return item
