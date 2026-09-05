"""Semantic memory with graph-backed entity relations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ..base import BaseMemory, MemoryItem, MemoryType

if TYPE_CHECKING:
    from ..storage import Neo4jGraphStore


class SemanticMemory(BaseMemory):
    memory_type = MemoryType.SEMANTIC

    def __init__(self, *, graph_store: "Neo4jGraphStore | None" = None, **kwargs: Any) -> None:
        super().__init__(memory_type=self.memory_type, **kwargs)
        if graph_store is None:
            from ..storage import Neo4jGraphStore

            graph_store = Neo4jGraphStore()
        self.graph_store = graph_store

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


__all__ = ["SemanticMemory"]
