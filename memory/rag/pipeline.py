"""Memory-backed retrieval and prompt-context assembly."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from constants import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_CONTEXT_MAX_CHARS,
    RAG_GRAPH_HOPS,
    RAG_RETRIEVE_LIMIT,
)

from ..base import MemoryItem, MemorySearchResult, MemoryType
from ..manager import MemoryManager
from .document import Document, DocumentProcessor
from .graph_rag import GraphRAGPipeline, GraphRAGResult
from .knowledge import (
    EntityResolver,
    KnowledgeExtractor,
    NullKnowledgeExtractor,
    build_graph_context,
    materialize_extraction,
)


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    score: float
    memory_id: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_result(cls, result: MemorySearchResult) -> "RetrievedChunk":
        return cls(result.item.content, result.score, result.item.id, result.item.metadata)


def _accepts_graph_context(extractor: KnowledgeExtractor) -> bool:
    """True when the extractor can consume the pre-extraction subgraph.

    Custom extractors written against the older two-argument contract keep
    working; only implementations that opt in receive ``graph_context``.
    """

    try:
        parameters = inspect.signature(extractor.extract).parameters
    except (TypeError, ValueError):
        return False
    return "graph_context" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class RAGPipeline:
    def __init__(
        self,
        manager: MemoryManager | None = None,
        *,
        processor: DocumentProcessor | None = None,
        extractor: KnowledgeExtractor | None = None,
        auto_extract: bool = True,
    ) -> None:
        self.manager = manager if manager is not None else MemoryManager()
        self.processor = processor if processor is not None else DocumentProcessor()
        self.extractor = extractor if extractor is not None else NullKnowledgeExtractor()
        self.auto_extract = auto_extract
        self.graph = GraphRAGPipeline(self.manager)
        self.last_ingest_report: dict[str, Any] = {}

    def ingest(self, documents: Document | Iterable[Document], *, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[MemoryItem]:
        values = [documents] if isinstance(documents, Document) else list(documents)
        items: list[MemoryItem] = []
        report = {
            "chunks": 0,
            "domains": [],
            "entities": 0,
            "relations": 0,
            "superseded": 0,
            "retracted": 0,
            "skipped_relations": 0,
            "errors": [],
        }
        # One resolver per ingest call: entities created by chunk 1 must be
        # reusable and aliasable by chunk 2 without a full reload each time.
        resolver = EntityResolver(self.manager)
        accepts_context = _accepts_graph_context(self.extractor)
        for document in values:
            for chunk in self.processor.chunks(document, chunk_size=chunk_size, overlap=overlap):
                metadata = dict(chunk.metadata)
                metadata.setdefault("source", document.metadata.get("source", document.id))
                item = self.manager.add(chunk.content, memory_type=MemoryType.SEMANTIC, metadata=metadata, item_id=chunk.id)
                items.append(item)
                report["chunks"] += 1
                if not self.auto_extract:
                    continue
                try:
                    # Feed the relevant subgraph to the extractor first, so the
                    # model reuses canonical entity names and retire the right
                    # old value instead of inventing a second entity.
                    graph_context = build_graph_context(self.manager, chunk.content)
                    if accepts_context:
                        extraction = self.extractor.extract(
                            chunk.content,
                            metadata=metadata,
                            graph_context=graph_context,
                        )
                    else:
                        extraction = self.extractor.extract(
                            chunk.content, metadata=metadata
                        )
                    materialized = materialize_extraction(
                        self.manager,
                        extraction,
                        source_item=item,
                        source_metadata=metadata,
                        resolver=resolver,
                    )
                    report["domains"].append(materialized["domain"])
                    report["entities"] += materialized["entities"]
                    report["relations"] += materialized["relations"]
                    report["superseded"] += materialized["superseded"]
                    report["retracted"] += materialized["retracted"]
                    report["skipped_relations"] += materialized["skipped_relations"]
                except Exception as exc:  # extraction failure must not lose source text
                    report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["domains"] = list(dict.fromkeys(report["domains"]))
        self.last_ingest_report = report
        return items

    def ingest_source(self, source: Any, **kwargs: Any) -> list[MemoryItem]:
        return self.ingest(self.processor.parse(source, metadata=kwargs.pop("metadata", None)), **kwargs)

    def retrieve(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        return [RetrievedChunk.from_result(result) for result in self.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=limit, threshold=threshold, metadata=metadata)]

    def build_context(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, separator: str = "\n\n") -> str:
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")
        return separator.join(chunk.content for chunk in self.retrieve(query, limit=limit))

    def graph_retrieve(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, hops: int = RAG_GRAPH_HOPS) -> GraphRAGResult:
        return self.graph.retrieve(query, limit=limit, hops=hops)

    def graph_context(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, hops: int = RAG_GRAPH_HOPS, max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str:
        return self.graph.build_context(query, limit=limit, hops=hops, max_chars=max_chars)

    def answer(self, query: str, generator: Callable[[str], str], *, limit: int = RAG_RETRIEVE_LIMIT) -> str:
        if not callable(generator):
            raise TypeError("generator must be callable")
        context = self.build_context(query, limit=limit)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return str(generator(prompt))

    def delete_document(self, document_id: str) -> int:
        items = self.manager.list(memory_type=MemoryType.SEMANTIC, include_expired=True)
        removed = 0
        for item in items:
            if item.metadata.get("document_id") == document_id and self.manager.delete(item.id):
                removed += 1
        return removed

    def close(self) -> None:
        self.manager.close()


__all__ = ["RAGPipeline", "RetrievedChunk"]
