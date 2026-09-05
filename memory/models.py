"""Data models shared by the standalone memory system.

The models intentionally have no dependency on a particular storage vendor.  A
``MemoryItem`` can therefore be moved between the SQLite, Qdrant and custom
backends without changing application code.
"""

from __future__ import annotations

import base64
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERCEPTUAL = "perceptual"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("datetime values must be datetime, ISO string, or None")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


@dataclass
class MemoryItem:
    """The canonical memory record.

    ``content`` is the searchable textual representation.  ``payload`` is
    optional multimodal data (for example image bytes or a URI) and is kept
    separate so vector stores only need to index text.
    """

    content: str
    memory_type: MemoryType | str = MemoryType.WORKING
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    timestamp: datetime | None = None
    embedding: list[float] | None = None
    payload: Any = None
    modality: str | None = None
    relations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            self.content = str(self.content)
        self.memory_type = MemoryType(self.memory_type)
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("memory id must be a non-empty string")
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata or {})
        if isinstance(self.importance, bool) or not isinstance(self.importance, (int, float)):
            raise TypeError("importance must be a number")
        if not math.isfinite(float(self.importance)) or not 0 <= float(self.importance) <= 1:
            raise ValueError("importance must be between 0 and 1")
        self.importance = float(self.importance)
        self.created_at = ensure_datetime(self.created_at) or utc_now()
        self.updated_at = ensure_datetime(self.updated_at) or self.created_at
        self.expires_at = ensure_datetime(self.expires_at)
        self.timestamp = ensure_datetime(self.timestamp)
        if self.embedding is not None:
            self.embedding = [float(v) for v in self.embedding]
        if not isinstance(self.relations, list):
            self.relations = list(self.relations)

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= utc_now()

    @property
    def event_time(self) -> datetime:
        return self.timestamp or self.created_at

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": _json_safe(self.metadata),
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "embedding": self.embedding,
            "modality": self.modality,
            "relations": _json_safe(self.relations),
        }
        if include_payload:
            result["payload"] = _json_safe(self.payload)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryItem":
        data = dict(value)
        data["payload"] = _json_restore(data.get("payload"))
        return cls(**data)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _json_restore(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__bytes__"}:
        try:
            return base64.b64decode(value["__bytes__"])
        except Exception:
            return value
    if isinstance(value, dict):
        return {key: _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    return value


@dataclass(frozen=True)
class MemorySearchResult:
    item: MemoryItem
    score: float

    def to_dict(self) -> dict[str, Any]:
        data = self.item.to_dict()
        data["score"] = self.score
        return data


__all__ = ["MemoryItem", "MemorySearchResult", "MemoryType", "ensure_datetime", "utc_now"]
