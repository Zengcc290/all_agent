"""Working memory with TTL and capacity-based eviction."""

from __future__ import annotations

from typing import Any, Mapping

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

    def set(self, key: str, value: Any, *, ttl_seconds: float | None = None, importance: float = 0.5, metadata: Mapping[str, Any] | None = None) -> MemoryItem:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        existing = self.get_by_key(key)
        if existing is not None:
            self.delete(existing.id)
        merged = dict(metadata or {})
        merged["key"] = key
        return self.add(str(value), metadata=merged, payload=value, ttl_seconds=ttl_seconds, importance=importance, item_id=key)

    def get_by_key(self, key: str) -> MemoryItem | None:
        item = self.document_store.get(key)
        if item is not None and item.memory_type == self.memory_type:
            return None if item.is_expired else item
        for candidate in self.list():
            if candidate.metadata.get("key") == key:
                return candidate
        return None

    def get_value(self, key: str, default: Any = None) -> Any:
        item = self.get_by_key(key)
        return item.payload if item is not None else default

    def _evict_if_needed(self) -> None:
        items = self.list()
        if len(items) <= self.capacity:
            return
        # Lower importance first; ties prefer the oldest update.
        victims = sorted(items, key=lambda item: (item.importance, item.updated_at))[: len(items) - self.capacity]
        for item in victims:
            self.delete(item.id)


__all__ = ["WorkingMemory"]
