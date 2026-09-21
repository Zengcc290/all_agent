"""Memory-backed retrieval and prompt-context assembly."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from constants import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_CONTEXT_MAX_CHARS,
    RAG_GRAPH_HOPS,
    RAG_RETRIEVE_LIMIT,
)

from ..base import MemoryItem, MemorySearchResult, MemoryType
from ..embedding_lock import apply_embedding_lock
from ..manager import MemoryManager
from ..storage.document_repo import (
    PERMISSIONS,
    DocumentRecord,
    DocumentRepository,
)
from .document import Document, DocumentProcessor, resolve_within
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
    #: 混合检索的分数明细（U4 溯源面板）：rrf_score / vector_score / keyword_score。
    detail: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: MemorySearchResult) -> RetrievedChunk:
        return cls(result.item.content, result.score, result.item.id, result.item.metadata)


def accepts_parameter(extractor: KnowledgeExtractor, name: str) -> bool:
    """True when ``extractor.extract`` accepts ``name`` (or ``**kwargs``).

    抽取器协议自省：图片入库（``tool/ingest_image.py``）与文本入库共用这一处判定，
    所以它是公开的——不写第二份 ``inspect.signature`` 逻辑。
    """

    try:
        parameters = inspect.signature(extractor.extract).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _accepts_graph_context(extractor: KnowledgeExtractor) -> bool:
    """True when the extractor can consume the pre-extraction subgraph.

    Custom extractors written against the older two-argument contract keep
    working; only implementations that opt in receive ``graph_context``.
    """

    return accepts_parameter(extractor, "graph_context")


def _document_tags(metadata: Mapping[str, Any]) -> list[str]:
    """Read ``tags`` from caller-supplied metadata (list, single string, or none)."""

    tags = metadata.get("tags") or []
    if isinstance(tags, str):
        return [tags]
    return [str(tag) for tag in tags]


def _document_permission(metadata: Mapping[str, Any]) -> str:
    """Fail closed: anything unrecognized is ``private``, i.e. stays on this machine."""

    permission = str(metadata.get("permission", "private"))
    return permission if permission in PERMISSIONS else "private"


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
        self._repository: DocumentRepository | None = None

    def document_repo(self) -> DocumentRepository | None:
        """The ``documents``/``chunks`` source of truth, or ``None`` if unavailable.

        Two cases return ``None`` instead of failing ingest: an injected document
        store that is not SQLite-backed (no ``path``), and ``:memory:`` - a
        second in-memory database would be a private connection that shares no
        data with the store it is supposed to mirror.
        """

        path = getattr(self.manager.document_store, "path", None)
        if not path or str(path) == ":memory:":
            return None
        if self._repository is None or self._repository.path != str(path):
            self._repository = DocumentRepository(path)
        return self._repository

    def ingest(self, documents: Document | Iterable[Document], *, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP, granularity: str = "chunk") -> list[MemoryItem]:
        values = [documents] if isinstance(documents, Document) else list(documents)
        if granularity not in {"chunk", "sentences"}:
            raise ValueError("granularity must be 'chunk' or 'sentences'")
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
        repository = self.document_repo()
        apply_embedding_lock(self.manager, repository)
        for document in values:
            source = str(document.metadata.get("source", document.id))
            if repository is not None:
                repository.upsert_document(
                    DocumentRecord(
                        document_id=document.id,
                        title=str(document.metadata.get("title", "")),
                        raw_text=self.processor.normalized_text(document),
                        source=source,
                        tags=_document_tags(document.metadata),
                        permission=_document_permission(document.metadata),
                        status="parsed",
                    )
                )
            document_error: str | None = None
            # F4：granularity="sentences" 逐句切块（每句一条记录）；默认仍是字符
            # 窗口块，保持向后兼容。句级也写真值源 chunks，不新建表。
            spans = (
                self.processor.sentences_with_spans(document)
                if granularity == "sentences"
                else self.processor.chunks_with_spans(document, chunk_size=chunk_size, overlap=overlap)
            )
            try:
                for span in spans:
                    chunk = span.chunk
                    metadata = dict(chunk.metadata)
                    metadata.setdefault("source", source)
                    # 混合索引（先写真值源再写向量、再置 vector_status）的唯一实现
                    # 在 tool/hybrid_index.py；函数内导入避免 memory.rag -> tool 的
                    # 模块级初始化环。
                    from tool.hybrid_index import index_chunk

                    item = index_chunk(
                        self.manager,
                        repository,
                        chunk_id=chunk.id,
                        document_id=document.id,
                        chunk_index=int(chunk.metadata["chunk_index"]),
                        char_start=span.char_start,
                        char_end=span.char_end,
                        text=chunk.content,
                        metadata=metadata,
                    )
                    items.append(item)
                    report["chunks"] += 1
                    if not self.auto_extract:
                        continue
                    try:
                        # Feed the relevant subgraph to the extractor first, so the
                        # model reuses canonical entity names and retire the right
                        # old value instead of inventing a second entity.
                        graph_context = build_graph_context(
                            self.manager, chunk.content, resolver=resolver
                        )
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
                    except Exception as exc:  # noqa: BLE001 - extraction failure must not lose source text
                        message = f"{type(exc).__name__}: {exc}"
                        report["errors"].append(message)
                        document_error = document_error or message
            except Exception:
                # 嵌入/真值写入失败（如云端端点不可达）不能留下半吊子记录：
                # 回滚该文档的真值行后原样抛出，由调用方转成明确的错误响应。
                if repository is not None:
                    repository.delete_document(document.id)
                raise
            if repository is not None:
                if document_error is None:
                    repository.set_status(document.id, "extracted" if self.auto_extract else "vectorized")
                else:
                    repository.set_status(document.id, "failed", error=document_error)
        report["domains"] = list(dict.fromkeys(report["domains"]))
        report["extractor"] = type(self.extractor).__name__
        report["extraction_skipped"] = isinstance(self.extractor, NullKnowledgeExtractor)
        self.last_ingest_report = report
        return items

    def ingest_media(
        self,
        image: bytes,
        *,
        text: str = "",
        mime_type: str = "image/jpeg",
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
        """Index one image and materialize vision-extracted n-ary observations.

        实现已工具化（``tool/ingest_image.py``，工具名 ``knowledge.ingest_image``）；
        这里保留薄委托，因为「图片入库」在本类上是一个公开入口（多模态测试与
        Web 端点都用它），而规则细节（锁闸门顺序、抽取失败不回滚）只有一处实现。
        """

        from tool.ingest_image import ingest_image

        result = ingest_image(
            self, image=image, text=text, mime_type=mime_type, metadata=dict(metadata or {})
        )
        item = self.manager.get(result["item_id"])
        if item is None:  # pragma: no cover - add() 刚写入，取不到说明库被外部改动
            raise RuntimeError(f"图片已入库但读不回：{result['item_id']}")
        return item

    def ingest_source(
        self,
        source: str | Path,
        *,
        base_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> list[MemoryItem]:
        """Ingest a file by path.

        ``source`` is a path here (a string is converted to ``Path`` so short
        text is never mistaken for a file name). When ``base_dir`` is given the
        resolved path must stay inside it, which is what lets a model-driven
        caller pass an explicit containment boundary.
        """

        path = Path(source)
        if base_dir is not None:
            path = resolve_within(base_dir, path)
        return self.ingest(
            self.processor.parse(path, metadata=kwargs.pop("metadata", None)), **kwargs
        )

    def retrieve(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        repository = self.document_repo()
        results: list[RetrievedChunk] = []
        for result in self.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=limit, threshold=threshold, metadata=metadata):
            # 正文以 chunks 真值源为准（同一 id 的内容可能已被修正）；查不到再退回 memories。
            chunk = repository.get_chunk(result.item.id) if repository is not None else None
            if chunk is None:
                results.append(RetrievedChunk.from_result(result))
            else:
                results.append(
                    RetrievedChunk(chunk.text, float(result.score), result.item.id, result.item.metadata)
                )
        return results

    def build_context(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, separator: str = "\n\n") -> str:
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")
        return separator.join(chunk.content for chunk in self.retrieve(query, limit=limit))

    def graph_retrieve(
        self,
        query: str,
        *,
        limit: int = RAG_RETRIEVE_LIMIT,
        hops: int = RAG_GRAPH_HOPS,
        at: str | None = None,
    ) -> GraphRAGResult:
        return self.graph.retrieve(query, limit=limit, hops=hops, at=at)

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
        # 真值源同步删除，否则 documents/chunks 会留下永远查不到来源的孤儿行。
        repository = self.document_repo()
        if repository is not None:
            repository.delete_document(document_id)
        return removed

    def close(self) -> None:
        if self._repository is not None:
            self._repository.close()
            self._repository = None
        self.manager.close()


__all__ = ["RAGPipeline", "RetrievedChunk"]
