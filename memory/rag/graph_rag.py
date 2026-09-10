"""Hybrid vector and graph retrieval for the knowledge nebula."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..base import MemorySearchResult, MemoryType
from ..manager import MemoryManager


@dataclass(frozen=True)
class GraphPath:
    source: str
    target: str
    relations: tuple[str, ...]
    entities: tuple[str, ...]
    confidence: float = 0.0
    evidence: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relations": list(self.relations),
            "entities": list(self.entities),
            "confidence": self.confidence,
            "evidence": [dict(item) for item in self.evidence],
        }


@dataclass
class GraphRAGResult:
    query: str
    evidence: list[MemorySearchResult] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "evidence": [item.to_dict() for item in self.evidence],
            "paths": [path.to_dict() for path in self.paths],
            "entities": list(self.entities),
        }

    def build_context(self, *, max_chars: int = 12000) -> str:
        parts: list[str] = []
        for result in self.evidence:
            item = result.item
            source = item.metadata.get("filename") or item.metadata.get("source") or "记忆库"
            parts.append(f"[证据|来源={source}|相似度={result.score:.3f}]\n{item.content}")
        for path in self.paths:
            relation = " -".join(path.relations)
            parts.append(
                f"[关系路径|置信度={path.confidence:.3f}] "
                f"{' -> '.join(path.entities)}（{relation}）"
            )
            for evidence in path.evidence:
                text = evidence.get("evidence") or evidence.get("source") or ""
                if text:
                    parts.append(f"[关系证据] {text}")
        return "\n\n".join(parts)[:max_chars]


class GraphRAGPipeline:
    """Retrieve semantic evidence, then expand one or more graph hops."""

    def __init__(self, manager: MemoryManager) -> None:
        self.manager = manager

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        hops: int = 1,
        threshold: float | None = None,
        path_limit: int = 20,
    ) -> GraphRAGResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if isinstance(hops, bool) or not isinstance(hops, int) or not 0 <= hops <= 3:
            raise ValueError("hops must be an integer between 0 and 3")
        if isinstance(path_limit, bool) or not isinstance(path_limit, int) or path_limit < 1:
            raise ValueError("path_limit must be a positive integer")

        evidence = self.manager.search(
            query,
            memory_type=MemoryType.SEMANTIC,
            limit=max(limit * 3, limit),
            threshold=threshold,
        )
        seeds = self._find_seed_entities(query, evidence)
        paths = self._expand(seeds, hops=hops, path_limit=path_limit)
        return GraphRAGResult(
            query=query,
            evidence=evidence[:limit],
            paths=paths,
            entities=seeds,
        )

    def build_context(self, query: str, *, limit: int = 5, hops: int = 1, max_chars: int = 12000) -> str:
        return self.retrieve(query, limit=limit, hops=hops).build_context(max_chars=max_chars)

    def _find_seed_entities(self, query: str, evidence: list[MemorySearchResult]) -> list[str]:
        names: list[str] = []
        known = self.manager.semantic.list()
        query_folded = query.casefold()
        for result in evidence:
            item = result.item
            metadata = item.metadata
            if metadata.get("kind") == "entity":
                names.append(str(metadata.get("canonical_name") or metadata.get("title") or item.content))
            for key in ("subject", "object"):
                value = metadata.get(key)
                if value and str(value).casefold() in query_folded:
                    names.append(str(value))
        for item in known:
            if item.metadata.get("kind") != "entity":
                continue
            name = str(item.metadata.get("canonical_name") or item.content)
            aliases = [str(value) for value in item.metadata.get("aliases", [])]
            if any(value.casefold() in query_folded for value in [name, *aliases] if value):
                names.append(name)
        return list(dict.fromkeys(names))

    def _expand(self, seeds: list[str], *, hops: int, path_limit: int) -> list[GraphPath]:
        if hops == 0 or not seeds:
            return []
        paths: list[GraphPath] = []
        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], tuple[dict[str, Any], ...], int, float]] = deque(
            (seed, (seed,), (), (), 0, 1.0) for seed in seeds
        )
        visited: set[tuple[str, tuple[str, ...]]] = set()
        while queue and len(paths) < path_limit:
            current, entities, relations, evidence, depth, confidence = queue.popleft()
            if depth >= hops:
                continue
            for edge in self.manager.semantic.related(current):
                source = str(edge.get("source") or current)
                target = str(edge.get("target") or current)
                neighbor = target if source == current else source
                relation = str(edge.get("relation") or "关联")
                props = dict(edge.get("properties") or {})
                edge_confidence = float(props.get("confidence", 0.0) or 0.0)
                next_entities = (*entities, neighbor)
                next_relations = (*relations, relation)
                marker = (neighbor, next_relations)
                if marker in visited:
                    continue
                visited.add(marker)
                next_confidence = min(confidence, edge_confidence or confidence)
                next_evidence = (*evidence, props)
                paths.append(
                    GraphPath(
                        source=entities[0],
                        target=neighbor,
                        relations=next_relations,
                        entities=next_entities,
                        confidence=next_confidence,
                        evidence=next_evidence,
                    )
                )
                queue.append((neighbor, next_entities, next_relations, next_evidence, depth + 1, next_confidence))
        paths.sort(key=lambda item: (-item.confidence, len(item.relations), item.target))
        return paths[:path_limit]


__all__ = ["GraphPath", "GraphRAGPipeline", "GraphRAGResult"]
