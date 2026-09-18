"""Hybrid vector and graph retrieval for the knowledge nebula."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from constants import (
    MEMORY_EDGE_REINFORCE,
    RAG_CONTEXT_MAX_CHARS,
    RAG_GRAPH_HOPS,
    RAG_GRAPH_MAX_HOPS,
    RAG_GRAPH_PATH_LIMIT,
    RAG_RETRIEVE_LIMIT,
)

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
    #: F1 权重感知排序分（confidence × 边权重沿路径取最小）；不影响 to_dict 负载。
    effective: float = 0.0
    #: 路径的真实边 ``{source, relation, target}``，供回忆强化按边递增；不进 to_dict。
    steps: tuple[dict[str, Any], ...] = ()

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

    def build_context(self, *, max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str:
        parts: list[str] = []
        for result in self.evidence:
            item = result.item
            source = (
                item.metadata.get("filename") or item.metadata.get("source") or "记忆库"
            )
            current = item.metadata.get("active", True) is not False
            marker = "" if current else "|历史记录，已被更新"
            parts.append(
                f"[证据|来源={source}|相似度={result.score:.3f}{marker}]\n{item.content}"
            )
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
        limit: int = RAG_RETRIEVE_LIMIT,
        hops: int = RAG_GRAPH_HOPS,
        threshold: float | None = None,
        path_limit: int = RAG_GRAPH_PATH_LIMIT,
        at: str | None = None,
    ) -> GraphRAGResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if isinstance(hops, bool) or not isinstance(hops, int) or not 0 <= hops <= RAG_GRAPH_MAX_HOPS:
            raise ValueError(f"hops must be an integer between 0 and {RAG_GRAPH_MAX_HOPS}")
        if (
            isinstance(path_limit, bool)
            or not isinstance(path_limit, int)
            or path_limit < 1
        ):
            raise ValueError("path_limit must be a positive integer")

        evidence = self.manager.search(
            query,
            memory_type=MemoryType.SEMANTIC,
            limit=max(limit * 3, limit),
            threshold=threshold,
        )
        try:
            seeds = self._find_seed_entities(query, evidence)
            paths = self._expand(seeds, hops=hops, path_limit=path_limit, at=at)
        except Exception:  # noqa: BLE001 - Aura/local graph flaps must not drop vector evidence
            # Graph backends (Aura via proxy, local Neo4j) can flap without
            # taking the whole chat answer down.  Vector/keyword evidence still
            # returns; the nebula just has no multi-hop paths this round.
            return GraphRAGResult(
                query=query,
                evidence=evidence[:limit],
                paths=[],
                entities=[],
            )
        return GraphRAGResult(
            query=query,
            evidence=evidence[:limit],
            paths=paths,
            entities=seeds,
        )

    def build_context(
        self, query: str, *, limit: int = 5, hops: int = 1, max_chars: int = 12000
    ) -> str:
        return self.retrieve(query, limit=limit, hops=hops).build_context(
            max_chars=max_chars
        )

    def retrieve_multi(
        self,
        queries: list[str],
        *,
        limit: int = RAG_RETRIEVE_LIMIT,
        hops: int = RAG_GRAPH_HOPS,
        threshold: float | None = None,
        path_limit: int = RAG_GRAPH_PATH_LIMIT,
    ) -> GraphRAGResult:
        """F3：每条子查询各做 seed+expand，路径按 effective 合并去重。"""

        queries = [query for query in queries if isinstance(query, str) and query.strip()]
        if not queries:
            raise ValueError("queries must be a non-empty list of strings")
        if len(queries) == 1:
            return self.retrieve(queries[0], limit=limit, hops=hops, threshold=threshold, path_limit=path_limit)
        evidence: dict[str, MemorySearchResult] = {}
        paths: dict[tuple[tuple[str, ...], tuple[str, ...]], GraphPath] = {}
        seeds: list[str] = []
        for query in queries:
            result = self.retrieve(query, limit=limit, hops=hops, threshold=threshold, path_limit=path_limit)
            for item in result.evidence:
                evidence.setdefault(item.item.id, item)
            seeds.extend(result.entities)
            for path in result.paths:
                key = (path.entities, path.relations)
                existing = paths.get(key)
                if existing is None or path.effective > existing.effective:
                    paths[key] = path
        merged = sorted(paths.values(), key=lambda path: (-path.effective, len(path.relations), path.target))[:path_limit]
        return GraphRAGResult(
            query=" | ".join(queries),
            evidence=list(evidence.values())[:limit],
            paths=merged,
            entities=list(dict.fromkeys(seeds)),
        )

    def _find_seed_entities(
        self, query: str, evidence: list[MemorySearchResult]
    ) -> list[str]:
        names: list[str] = []
        known = self.manager.semantic.list()
        query_folded = query.casefold()
        for result in evidence:
            item = result.item
            metadata = item.metadata
            if metadata.get("kind") == "entity":
                names.append(
                    str(
                        metadata.get("canonical_name")
                        or metadata.get("title")
                        or item.content
                    )
                )
            for key in ("subject", "object"):
                value = metadata.get(key)
                if value and str(value).casefold() in query_folded:
                    names.append(str(value))
        for item in known:
            if item.metadata.get("kind") != "entity":
                continue
            name = str(item.metadata.get("canonical_name") or item.content)
            aliases = [str(value) for value in item.metadata.get("aliases", [])]
            if any(
                value.casefold() in query_folded for value in [name, *aliases] if value
            ):
                names.append(name)
        return list(dict.fromkeys(names))

    def _expand(
        self, seeds: list[str], *, hops: int, path_limit: int, at: str | None = None
    ) -> list[GraphPath]:
        if hops == 0 or not seeds:
            return []
        reinforce = self._reinforce_enabled()
        paths: list[GraphPath] = []
        queue: deque[
            tuple[
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[dict[str, Any], ...],
                int,
                float,
                float,
                tuple[dict[str, Any], ...],
            ]
        ] = deque(
            (seed, (seed,), (), (), 0, 1.0, 1.0, ()) for seed in seeds
        )
        visited: set[tuple[str, tuple[str, ...]]] = set()
        while queue and len(paths) < path_limit:
            current, entities, relations, evidence, depth, confidence, effective, steps = queue.popleft()
            if depth >= hops:
                continue
            for edge in self._weighted_edges(current, at=at):
                source = str(edge.get("source") or current)
                target = str(edge.get("target") or current)
                neighbor = target if source == current else source
                relation = str(edge.get("relation") or "关联")
                props = dict(edge.get("properties") or {})
                # Superseded or retracted edges stay in the store for audit,
                # but they must never carry a retrieval hop. SQLite is the
                # source of truth: the in-process/Neo4j edge copy can still
                # hold the pre-retirement flag, so check the memory record too.
                if props.get("active") is False or not self._edge_is_active(props):
                    continue
                # F4：时间/状态过滤。时间段空值视为无界；status 默认只要
                # fact/plan（uncertain 降权而非排除，expired 不命中）。
                if not self._edge_in_window(props, at=at):
                    continue
                edge_confidence = float(props.get("confidence", 0.0) or 0.0)
                edge_status = str(props.get("status") or "fact")
                if edge_status == "expired":
                    continue
                if edge_status == "uncertain":
                    edge_confidence *= 0.5
                # F1：边权重让被反复回忆的边在排序中靠前；全部权重为 1.0 时
                # effective 退化为原来的 confidence 语义。
                edge_weight = float(props.get("weight", 1.0) or 1.0)
                edge_effective = edge_confidence * edge_weight
                next_entities = (*entities, neighbor)
                next_relations = (*relations, relation)
                marker = (neighbor, next_relations)
                if marker in visited:
                    continue
                if neighbor in entities:
                    continue
                visited.add(marker)
                next_confidence = min(confidence, edge_confidence or confidence)
                next_effective = min(effective, edge_effective or effective)
                next_evidence = (*evidence, props)
                next_steps = (
                    *steps,
                    {"source": source, "relation": relation, "target": target},
                )
                paths.append(
                    GraphPath(
                        source=entities[0],
                        target=neighbor,
                        relations=next_relations,
                        entities=next_entities,
                        confidence=next_confidence,
                        evidence=next_evidence,
                        effective=next_effective,
                        steps=next_steps,
                    )
                )
                queue.append(
                    (
                        neighbor,
                        next_entities,
                        next_relations,
                        next_evidence,
                        depth + 1,
                        next_confidence,
                        next_effective,
                        next_steps,
                    )
                )
        paths.sort(
            key=lambda item: (-item.effective, len(item.relations), item.target)
        )
        adopted = paths[:path_limit]
        if reinforce:
            self._reinforce(adopted)
        return adopted

    def _weighted_edges(
        self, entity: str, *, at: str | None = None
    ) -> list[dict[str, Any]]:
        """Edges around ``entity`` with the strongest first (F1 权重优先遍历)."""

        edges = self.manager.semantic.related(entity, at=at)
        edges.sort(
            key=lambda edge: float(
                (edge.get("properties") or {}).get("weight", 1.0) or 1.0
            ),
            reverse=True,
        )
        return edges

    def _reinforce_enabled(self) -> bool:
        """F1 总开关（constants.MEMORY_EDGE_REINFORCE，默认开）。"""

        return MEMORY_EDGE_REINFORCE

    def _reinforce(self, paths: list[GraphPath]) -> None:
        """"回忆即强化"：对本次真正返回的路径边各 +1 次回忆（每条边每轮一次）。"""

        bumped: set[tuple[str, str, str]] = set()
        add_relation = self.manager.semantic.graph_store.add_relation
        for path in paths:
            for step in path.steps:
                key = (step["source"], step["relation"], step["target"])
                if key in bumped:
                    continue
                bumped.add(key)
                add_relation(*key, bump=True)

    def _edge_is_active(self, properties: dict[str, Any]) -> bool:
        """Confirm an edge against its memory record before it carries a hop."""

        memory_id = properties.get("memory_id")
        if not memory_id:
            return True
        item = self.manager.get(str(memory_id), memory_type=MemoryType.SEMANTIC)
        if item is None:
            return False
        return item.metadata.get("active", True) is not False

    def _edge_in_window(
        self, properties: dict[str, Any], *, at: str | None = None
    ) -> bool:
        """Time-window filter at ``at`` (or now); empty bounds are unbounded."""

        from ..base import ensure_datetime, utc_now

        try:
            moment = ensure_datetime(at) if at else utc_now()
        except (TypeError, ValueError):
            return False
        if moment is None:
            moment = utc_now()
        event_raw = str(properties.get("event_at") or "").strip()
        if event_raw and at:
            try:
                event_at = ensure_datetime(event_raw)
            except (TypeError, ValueError):
                event_at = None
            if event_at is not None and event_at > moment:
                return False
        for key in ("valid_from", "valid_to"):
            raw = str(properties.get(key) or "").strip()
            if not raw:
                continue
            try:
                bound = ensure_datetime(raw)
            except (TypeError, ValueError):
                return True  # 畸形时间不静默丢边，交给 status/active 判断
            if bound is None:
                continue
            if key == "valid_from" and moment < bound:
                return False
            if key == "valid_to" and moment > bound:
                return False
        return True


__all__ = ["GraphPath", "GraphRAGPipeline", "GraphRAGResult"]
