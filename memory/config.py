"""Configuration for the memory layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class MemoryConfig:
    """Runtime settings.

    Defaults are deliberately local and dependency-free.  Set ``sqlite_path``
    to a filename for persistence; ``":memory:"`` is useful for tests and
    short-lived agents.
    """

    sqlite_path: str | Path = ":memory:"
    default_ttl_seconds: float | None = 3600.0
    working_memory_capacity: int = 100
    search_limit: int = 10
    similarity_threshold: float = 0.0
    embedding_dimension: int = 384
    qdrant_url: str | None = None
    qdrant_collection: str = "helloagents_memory"
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.working_memory_capacity, bool) or not isinstance(self.working_memory_capacity, int) or self.working_memory_capacity < 1:
            raise ValueError("working_memory_capacity must be a positive integer")
        if isinstance(self.search_limit, bool) or not isinstance(self.search_limit, int) or self.search_limit < 1:
            raise ValueError("search_limit must be a positive integer")
        for name in ("similarity_threshold",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int) or self.embedding_dimension < 1:
            raise ValueError("embedding_dimension must be a positive integer")
        if self.default_ttl_seconds is not None:
            if isinstance(self.default_ttl_seconds, bool) or not isinstance(self.default_ttl_seconds, (int, float)) or self.default_ttl_seconds <= 0:
                raise ValueError("default_ttl_seconds must be positive or None")
        if not isinstance(self.qdrant_collection, str) or not self.qdrant_collection.strip():
            raise ValueError("qdrant_collection must be non-empty")
        if not isinstance(self.extra, dict):
            self.extra = dict(self.extra)

    @classmethod
    def from_env(cls, prefix: str = "HELLOAGENTS_MEMORY_") -> "MemoryConfig":
        """Build configuration from environment variables.

        Supported names mirror the dataclass fields, e.g.
        ``HELLOAGENTS_MEMORY_SQLITE_PATH`` and ``..._QDRANT_URL``.
        """
        values: dict[str, object] = {}
        for field_name in cls.__dataclass_fields__:
            key = f"{prefix}{field_name.upper()}"
            raw = os.getenv(key)
            if raw is None:
                continue
            if field_name in {"working_memory_capacity", "search_limit", "embedding_dimension"}:
                values[field_name] = int(raw)
            elif field_name in {"default_ttl_seconds", "similarity_threshold"}:
                values[field_name] = None if raw.casefold() == "none" else float(raw)
            elif field_name == "extra":
                continue
            else:
                values[field_name] = raw
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in self.__dict__.items()
        }


__all__ = ["MemoryConfig"]
