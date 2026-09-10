"""Memory-backed retrieval and prompt-context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from ..base import MemoryItem, MemorySearchResult, MemoryType
from ..manager import MemoryManager
from .document import Document, DocumentProcessor
from .graph_rag import GraphRAGPipeline, GraphRAGResult
from .knowledge import KnowledgeExtractor, NullKnowledgeExtractor, materialize_extraction


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    score: float
    memory_id: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_result(cls, result: MemorySearchResult) -> "RetrievedChunk":
        return cls(result.item.content, result.score, result.item.id, result.item.metadata)


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

    def ingest(self, documents: Document | Iterable[Document], *, chunk_size: int = 1000, overlap: int = 100) -> list[MemoryItem]:
        values = [documents] if isinstance(documents, Document) else list(documents)
        items: list[MemoryItem] = []
        report = {"chunks": 0, "domains": [], "entities": 0, "relations": 0, "skipped_relations": 0, "errors": []}
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
                    extraction = self.extractor.extract(chunk.content, metadata=metadata)
                    materialized = materialize_extraction(
                        self.manager,
                        extraction,
                        source_item=item,
                        source_metadata=metadata,
                    )
                    report["domains"].append(materialized["domain"])
                    report["entities"] += materialized["entities"]
                    report["relations"] += materialized["relations"]
                    report["skipped_relations"] += materialized["skipped_relations"]
                except Exception as exc:  # extraction failure must not lose source text
                    report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["domains"] = list(dict.fromkeys(report["domains"]))
        self.last_ingest_report = report
        return items

    def ingest_source(self, source: Any, **kwargs: Any) -> list[MemoryItem]:
        return self.ingest(self.processor.parse(source, metadata=kwargs.pop("metadata", None)), **kwargs)

    def retrieve(self, query: str, *, limit: int = 5, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        return [RetrievedChunk.from_result(result) for result in self.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=limit, threshold=threshold, metadata=metadata)]

    def build_context(self, query: str, *, limit: int = 5, separator: str = "\n\n") -> str:
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")
        return separator.join(chunk.content for chunk in self.retrieve(query, limit=limit))

    def graph_retrieve(self, query: str, *, limit: int = 5, hops: int = 1) -> GraphRAGResult:
        return self.graph.retrieve(query, limit=limit, hops=hops)

    def graph_context(self, query: str, *, limit: int = 5, hops: int = 1, max_chars: int = 12000) -> str:
        return self.graph.build_context(query, limit=limit, hops=hops, max_chars=max_chars)

    def answer(self, query: str, generator: Callable[[str], str], *, limit: int = 5) -> str:
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
