"""Memory-backed retrieval and prompt-context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from ..manager import MemoryManager
from ..models import MemoryItem, MemorySearchResult, MemoryType
from .document import Document, DocumentProcessor


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
    def __init__(self, manager: MemoryManager | None = None, *, processor: DocumentProcessor | None = None) -> None:
        self.manager = manager or MemoryManager()
        self.processor = processor or DocumentProcessor()

    def ingest(self, documents: Document | Iterable[Document], *, chunk_size: int = 1000, overlap: int = 100) -> list[MemoryItem]:
        values = [documents] if isinstance(documents, Document) else list(documents)
        items: list[MemoryItem] = []
        for document in values:
            for chunk in self.processor.chunks(document, chunk_size=chunk_size, overlap=overlap):
                metadata = dict(chunk.metadata)
                metadata.setdefault("source", document.metadata.get("source", document.id))
                items.append(self.manager.add(chunk.content, memory_type=MemoryType.SEMANTIC, metadata=metadata, item_id=chunk.id))
        return items

    def ingest_source(self, source: Any, **kwargs: Any) -> list[MemoryItem]:
        return self.ingest(self.processor.parse(source, metadata=kwargs.pop("metadata", None)), **kwargs)

    def retrieve(self, query: str, *, limit: int = 5, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        return [RetrievedChunk.from_result(result) for result in self.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=limit, threshold=threshold, metadata=metadata)]

    def build_context(self, query: str, *, limit: int = 5, separator: str = "\n\n") -> str:
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")
        return separator.join(chunk.content for chunk in self.retrieve(query, limit=limit))

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
