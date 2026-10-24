"""Episodic memory for timestamped event sequences."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..base import BaseMemory, MemoryItem, MemoryType


class EpisodicMemory(BaseMemory):
    memory_type = MemoryType.EPISODIC

    def record(self, event: str, *, timestamp: datetime | str | None = None, metadata: Mapping[str, Any] | None = None, importance: float = 0.5) -> MemoryItem:
        return self.add(event, timestamp=timestamp, metadata=metadata, importance=importance)


__all__ = ["EpisodicMemory"]
