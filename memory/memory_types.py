"""Concrete working, episodic, semantic and perceptual memories."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .base import BaseMemory
from .models import MemoryItem, MemoryType
from .storage import Neo4jGraphStore


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
        # Lower importance is evicted first; ties prefer the oldest update.
        victims = sorted(items, key=lambda item: (item.importance, item.updated_at))[: len(items) - self.capacity]
        for item in victims:
            self.delete(item.id)


class EpisodicMemory(BaseMemory):
    memory_type = MemoryType.EPISODIC

    def record(self, event: str, *, timestamp: datetime | str | None = None, metadata: Mapping[str, Any] | None = None, importance: float = 0.5) -> MemoryItem:
        return self.add(event, timestamp=timestamp, metadata=metadata, importance=importance)

    def timeline(self, *, start: datetime | str | None = None, end: datetime | str | None = None, limit: int | None = None) -> list[MemoryItem]:
        from .models import ensure_datetime
        start_dt, end_dt = ensure_datetime(start), ensure_datetime(end)
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("start must not be later than end")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer")
        items = [item for item in self.list() if (start_dt is None or item.event_time >= start_dt) and (end_dt is None or item.event_time <= end_dt)]
        items.sort(key=lambda item: item.event_time)
        return items[:limit] if limit is not None else items


class SemanticMemory(BaseMemory):
    memory_type = MemoryType.SEMANTIC

    def __init__(self, *, graph_store: Neo4jGraphStore | None = None, **kwargs: Any) -> None:
        super().__init__(memory_type=self.memory_type, **kwargs)
        self.graph_store = graph_store if graph_store is not None else Neo4jGraphStore()

    def add_fact(self, subject: str, predicate: str, object: str, *, metadata: Mapping[str, Any] | None = None, confidence: float = 1.0) -> MemoryItem:
        if not all(isinstance(value, str) and value.strip() for value in (subject, predicate, object)):
            raise ValueError("subject, predicate and object must be non-empty strings")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        item_metadata = dict(metadata or {})
        item_metadata.update({"subject": subject, "predicate": predicate, "object": object, "confidence": confidence})
        item = self.add(f"{subject} {predicate} {object}", metadata=item_metadata, importance=confidence)
        self.graph_store.add_relation(subject, predicate, object, properties={"memory_id": item.id, "confidence": confidence})
        return item

    def add_relation(self, source: str, relation: str, target: str, *, metadata: Mapping[str, Any] | None = None) -> MemoryItem:
        return self.add_fact(source, relation, target, metadata=metadata)

    def delete(self, item_id: str) -> bool:
        item = self.document_store.get(item_id)
        if item is None or item.memory_type != self.memory_type:
            return False
        removed = super().delete(item_id)
        if removed:
            remove_relation = getattr(self.graph_store, "delete_memory_relation", None)
            if callable(remove_relation):
                remove_relation(item_id)
        return removed

    def related(self, entity: str, *, relation: str | None = None) -> list[dict[str, Any]]:
        return self.graph_store.get_relations(entity, relation=relation)

    def facts(self, entity: str | None = None) -> list[MemoryItem]:
        items = self.list()
        if entity is None:
            return items
        return [item for item in items if entity in (item.metadata.get("subject"), item.metadata.get("object"))]


class PerceptualMemory(BaseMemory):
    memory_type = MemoryType.PERCEPTUAL

    def store(self, data: Any, *, modality: str, content: str | None = None, metadata: Mapping[str, Any] | None = None, importance: float = 0.5, item_id: str | None = None) -> MemoryItem:
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError("modality must be a non-empty string")
        text = content if content is not None else (data if isinstance(data, str) else f"{modality} perceptual memory")
        return self.add(str(text), metadata=metadata, importance=importance, payload=data, modality=modality, item_id=item_id)

    def get_data(self, item_id: str, default: Any = None) -> Any:
        item = self.get(item_id)
        return item.payload if item is not None else default

    remember = store


__all__ = ["WorkingMemory", "EpisodicMemory", "SemanticMemory", "PerceptualMemory"]
