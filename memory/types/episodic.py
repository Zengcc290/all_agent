"""Episodic memory for timestamped event sequences."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from ..base import BaseMemory, MemoryItem, MemoryType, ensure_datetime


class EpisodicMemory(BaseMemory):
    memory_type = MemoryType.EPISODIC

    def record(self, event: str, *, timestamp: datetime | str | None = None, metadata: Mapping[str, Any] | None = None, importance: float = 0.5) -> MemoryItem:
        return self.add(event, timestamp=timestamp, metadata=metadata, importance=importance)

    def timeline(self, *, start: datetime | str | None = None, end: datetime | str | None = None, limit: int | None = None) -> list[MemoryItem]:
        start_dt, end_dt = ensure_datetime(start), ensure_datetime(end)
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("start must not be later than end")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer")
        items = [item for item in self.list() if (start_dt is None or item.event_time >= start_dt) and (end_dt is None or item.event_time <= end_dt)]
        items.sort(key=lambda item: item.event_time)
        return items[:limit] if limit is not None else items


__all__ = ["EpisodicMemory"]
