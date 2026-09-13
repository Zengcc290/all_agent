"""Semantic memory with graph-backed entity relations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..base import BaseMemory, MemoryItem, MemoryType
from ..ids import legacy_fact_id_for, relation_id_for

if TYPE_CHECKING:
    from ..storage import Neo4jGraphStore


class SemanticMemory(BaseMemory):
    memory_type = MemoryType.SEMANTIC

    def __init__(
        self, *, graph_store: Neo4jGraphStore | None = None, **kwargs: Any
    ) -> None:
        super().__init__(memory_type=self.memory_type, **kwargs)
        if graph_store is None:
            from ..storage import Neo4jGraphStore

            graph_store = Neo4jGraphStore()
        self.graph_store = graph_store

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        confidence: float = 1.0,
        item_id: str | None = None,
    ) -> MemoryItem:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (subject, predicate, object)
        ):
            raise ValueError("subject, predicate and object must be non-empty strings")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1")
        # One canonical id scheme for every writer: the extraction pipeline uses
        # ``relation_id_for`` directly, so deriving the same id here stops the
        # same triple from being stored twice (once as ``fact:...``, once as
        # ``relation:...``). Rows written before the unification are still
        # updated in place through the legacy-id fallback below.
        fact_id = item_id or relation_id_for(subject, predicate, object)
        existing = self.document_store.get(fact_id)
        if existing is None and item_id is None:
            legacy_id = legacy_fact_id_for(subject, predicate, object)
            legacy = self.document_store.get(legacy_id)
            if legacy is not None and legacy.memory_type == self.memory_type:
                fact_id = legacy_id
                existing = legacy
        if existing is not None and existing.memory_type == self.memory_type:
            merged_metadata = dict(existing.metadata)
            merged_metadata.update(dict(metadata or {}))
            # Only reactivation clears the retired marker; an explicit
            # active=False must survive the merge so retract is idempotent.
            if metadata is None or "active" not in metadata:
                merged_metadata["active"] = True
                merged_metadata["superseded_by"] = []
                merged_metadata["superseded_at"] = ""
            return self.add(
                f"{subject} {predicate} {object}",
                metadata=merged_metadata,
                importance=max(existing.importance, confidence),
                item_id=fact_id,
            )
        item_metadata = dict(metadata or {})
        item_metadata.update(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "confidence": confidence,
            }
        )
        item = self.add(
            f"{subject} {predicate} {object}",
            metadata=item_metadata,
            importance=confidence,
            item_id=fact_id,
        )
        graph_properties = {
            "memory_id": item.id,
            "confidence": confidence,
        }
        for key in (
            "evidence",
            "source",
            "source_document",
            "chunk_id",
            "predicate_key",
            "action",
            "cardinality",
            "active",
            "superseded_by",
            "superseded_at",
            "supersedes",
        ):
            if key in item_metadata:
                graph_properties[key] = item_metadata[key]
        self.graph_store.add_relation(
            subject, predicate, object, properties=graph_properties
        )
        return item

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
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

    def related(
        self, entity: str, *, relation: str | None = None
    ) -> list[dict[str, Any]]:
        return self.graph_store.get_relations(entity, relation=relation)

    def facts(self, entity: str | None = None) -> list[MemoryItem]:
        items = self.list()
        if entity is None:
            return items
        return [
            item
            for item in items
            if entity in (item.metadata.get("subject"), item.metadata.get("object"))
        ]


__all__ = ["SemanticMemory"]
