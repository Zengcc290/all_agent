"""Semantic memory with graph-backed entity relations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..base import BaseMemory, MemoryItem, MemoryType
from ..ids import entity_id_for, legacy_fact_id_for, relation_id_for

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
            updated = self.add(
                f"{subject} {predicate} {object}",
                metadata=merged_metadata,
                importance=max(existing.importance, confidence),
                item_id=fact_id,
            )
            # 合并/撤回也要刷边：否则图里那条边会一直保留旧的 active/memory_id，
            # 与 SQLite 事实条目（active 真值）不一致（F6）。
            self._write_edge(subject, predicate, object, updated, merged_metadata)
            return updated
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
        self._write_edge(subject, predicate, object, item, item_metadata)
        return item

    def _graph_properties(
        self, item: MemoryItem, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Edge properties for one fact; the same key set is written on every path."""

        properties: dict[str, Any] = {
            "memory_id": item.id,
            "confidence": metadata.get("confidence", 1.0),
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
            # F4：时间/状态分类随边落地 Neo4j（SET r += $properties 自动带上）。
            "valid_from",
            "valid_to",
            "status",
            "event_at",
            "captured_at",
            "modality",
            "observation_id",
        ):
            if key in metadata:
                properties[key] = metadata[key]
        return properties

    def _endpoint_attributes(self, name: str) -> dict[str, Any]:
        """``domain``/``aliases``/``importance`` of an endpoint's entity item.

        The extraction pipeline writes entities before relations, so the item is
        normally present; a miss simply leaves the graph node at its defaults.
        """

        item = self.document_store.get(entity_id_for(name))
        if item is None or item.metadata.get("kind") != "entity":
            return {}
        return {
            "domain": str(item.metadata.get("domain", "")),
            "aliases": list(item.metadata.get("aliases") or []),
            "importance": float(item.importance),
            "entity_type": str(item.metadata.get("entity_type", "概念")),
        }

    def _write_edge(
        self,
        subject: str,
        predicate: str,
        object: str,
        item: MemoryItem,
        metadata: Mapping[str, Any],
    ) -> None:
        source_entity = self._endpoint_attributes(subject)
        target_entity = self._endpoint_attributes(object)
        properties = self._graph_properties(item, metadata)
        self.graph_store.add_relation(
            subject,
            predicate,
            object,
            properties=properties,
            source_domain=str(source_entity.get("domain", "")),
            target_domain=str(target_entity.get("domain", "")),
            source_aliases=list(source_entity.get("aliases") or []),
            target_aliases=list(target_entity.get("aliases") or []),
            source_importance=float(source_entity.get("importance", 0.5)),
            target_importance=float(target_entity.get("importance", 0.5)),
        )

        participants: list[dict[str, Any]] = [
            {
                "name": subject,
                "role": "subject",
                "ordinal": 0,
                **source_entity,
            },
            {
                "name": object,
                "role": "object",
                "ordinal": 1,
                **target_entity,
            },
        ]
        for ordinal, role in enumerate(metadata.get("roles") or [], start=2):
            if not isinstance(role, Mapping):
                continue
            role_name = str(role.get("role") or "").strip()
            value = str(role.get("value") or "").strip()
            if not role_name or not value:
                continue
            attributes = self._endpoint_attributes(value)
            participant = {
                "name": value,
                "role": role_name,
                "ordinal": ordinal,
                **attributes,
            }
            if role.get("entity_type"):
                participant["entity_type"] = str(role["entity_type"])
            participants.append(participant)
            # Compatibility projection: existing GraphRAG traverses RELATED
            # edges, while the authoritative n-ary shape remains the
            # MemoryObservation + HAS_PARTICIPANT topology below.
            self.graph_store.add_relation(
                subject,
                role_name,
                value,
                properties={**properties, "observation_id": item.id},
                source_domain=str(source_entity.get("domain", "")),
                target_domain=str(attributes.get("domain", "")),
                source_aliases=list(source_entity.get("aliases") or []),
                target_aliases=list(attributes.get("aliases") or []),
                source_importance=float(source_entity.get("importance", 0.5)),
                target_importance=float(attributes.get("importance", 0.5)),
            )

        add_observation = getattr(self.graph_store, "add_observation", None)
        if callable(add_observation):
            add_observation(
                item.id,
                predicate,
                participants,
                properties={
                    **properties,
                    "domain": str(metadata.get("domain") or ""),
                    "created_at": item.created_at.isoformat(),
                },
            )

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
        self, entity: str, *, relation: str | None = None, at: str | None = None
    ) -> list[dict[str, Any]]:
        return self.graph_store.get_relations(entity, relation=relation, at=at)

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
