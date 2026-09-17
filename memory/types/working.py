"""Working memory with TTL and capacity-based eviction."""

from __future__ import annotations

from typing import Any

from ..base import BaseMemory, MemoryItem, MemoryType


class WorkingMemory(BaseMemory):
    memory_type = MemoryType.WORKING

    def __init__(self, *, capacity: int | None = None, **kwargs: Any) -> None:
        super().__init__(memory_type=self.memory_type, **kwargs)
        self.capacity = self.config.working_memory_capacity if capacity is None else capacity
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 1:
            raise ValueError("capacity must be a positive integer")

    def add(self, content: str, **kwargs: Any) -> MemoryItem:
        item = super().add(content, **kwargs)
        self._evict_if_needed()
        return item

    def _evict_if_needed(self) -> None:
        items = self.list()
        if len(items) <= self.capacity:
            return
        # Lower importance first; ties prefer the oldest update.
        victims = sorted(items, key=lambda item: (item.importance, item.updated_at))[: len(items) - self.capacity]
        for item in victims:
            self.delete(item.id)


__all__ = ["WorkingMemory"]
