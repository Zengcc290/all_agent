"""Memory-backed retrieval and prompt-context assembly."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from constants import (
    MEMORY_HYBRID,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_CONTEXT_MAX_CHARS,
    RAG_GRAPH_HOPS,
    RAG_RETRIEVE_LIMIT,
)

from ..base import MemoryItem, MemorySearchResult, MemoryType
from ..manager import MemoryManager
from ..storage.document_repo import (
    PERMISSIONS,
    ChunkRecord,
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


def _accepts_parameter(extractor: KnowledgeExtractor, name: str) -> bool:
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

    return _accepts_parameter(extractor, "graph_context")


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


def _hybrid_enabled() -> bool:
    """混合检索开关（constants.MEMORY_HYBRID，默认开；关闭即回到纯向量）。"""

    return MEMORY_HYBRID


def _rrf_fuse(rank_lists: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: 只按名次计分，避免余弦与 bm25 两套量纲混算。"""

    scores: dict[str, float] = {}
    for hits in rank_lists:
        for rank, chunk_id in enumerate(hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _chunk_metadata(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


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
        #: 最近一次混合检索的降级说明（空串表示向量路正常），供 UI/健康检查展示。
        self.last_retrieval_note = ""

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
                    # 先写真值源（原文与分块边界）再写向量：反过来的话，向量写成功而
                    # 真值行失败就会留下无法解释的孤立向量。
                    if repository is not None:
                        repository.upsert_chunk(
                            ChunkRecord(
                                chunk_id=chunk.id,
                                document_id=document.id,
                                chunk_index=int(chunk.metadata["chunk_index"]),
                                char_start=span.char_start,
                                char_end=span.char_end,
                                text=chunk.content,
                            )
                        )
                    item = self.manager.add(chunk.content, memory_type=MemoryType.SEMANTIC, metadata=metadata, item_id=chunk.id)
                    items.append(item)
                    report["chunks"] += 1
                    if repository is not None:
                        repository.set_chunk_vector_status(chunk.id, "indexed")
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

        The image bytes are the canonical perceptual payload and are embedded by
        a VL-capable embedding backend. Knowledge graph edges come only from the
        structured vision extractor, never from vector similarity.
        """

        if not isinstance(image, bytes) or not image:
            raise ValueError("image must be non-empty bytes")
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise ValueError("mime_type must be an image media type")
        details = dict(metadata or {})
        details.setdefault("source", details.get("filename") or "图片入库")
        details["modality"] = "image"
        content = text.strip() if isinstance(text, str) else ""
        if not content:
            content = str(details.get("filename") or "图片观察")
        item = self.manager.add(
            content,
            memory_type=MemoryType.PERCEPTUAL,
            metadata=details,
            payload=image,
            modality="image",
            timestamp=details.get("captured_at") or None,
        )
        report = {
            "chunks": 1,
            "domains": [],
            "entities": 0,
            "relations": 0,
            "superseded": 0,
            "retracted": 0,
            "skipped_relations": 0,
            "errors": [],
            "modality": "image",
            "multimodal_embedding": bool(
                getattr(self.manager.embedding, "multimodal", False)
            ),
        }
        if self.auto_extract:
            try:
                resolver = EntityResolver(self.manager)
                graph_context = build_graph_context(
                    self.manager, content, resolver=resolver
                )
                kwargs: dict[str, Any] = {"metadata": details}
                if _accepts_parameter(self.extractor, "graph_context"):
                    kwargs["graph_context"] = graph_context
                if _accepts_parameter(self.extractor, "image"):
                    kwargs.update({"image": image, "mime_type": mime_type})
                extraction = self.extractor.extract(content, **kwargs)
                materialized = materialize_extraction(
                    self.manager,
                    extraction,
                    source_item=item,
                    source_metadata=details,
                    resolver=resolver,
                )
                for key in (
                    "entities",
                    "relations",
                    "superseded",
                    "retracted",
                    "skipped_relations",
                ):
                    report[key] = materialized[key]
                report["domains"] = [materialized["domain"]]
            except Exception as exc:  # noqa: BLE001 - keep indexed source on extraction failure
                report["errors"].append(f"{type(exc).__name__}: {exc}")
        self.last_ingest_report = report
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

    def _vector_hits(self, query: str, *, limit: int, threshold: float | None, metadata: Mapping[str, Any] | None) -> list[tuple[str, float]]:
        """向量路 ``(chunk_id, 相似度)``；不可用时返回空表（并写下降级原因）。

        不再做请求前的 TCP 可达性探测（历史隧道网关的产物）：云端端点一次
        urlopen 的代价与探测相同，失败时异常本身就是降级信号。
        """

        try:
            results = self.manager.search(query, memory_type=MemoryType.SEMANTIC, limit=limit, threshold=threshold, metadata=metadata)
        except (ConnectionError, OSError, RuntimeError) as exc:
            self.last_retrieval_note = f"向量检索失败，本次检索降级为纯关键词（FTS5）：{type(exc).__name__}: {exc}"
            return []
        return [(result.item.id, float(result.score)) for result in results]

    def hybrid_retrieve(self, query: str, *, limit: int = RAG_RETRIEVE_LIMIT, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        """向量路 × FTS5 关键词路，RRF 融合；向量不可用时退化为纯关键词（D8）。

        精确词（型号、编号、代码标识符）向量区分度差，转述又只有向量能召回，
        两路互补；任一投影不可用都不能让检索整体失败。
        """

        repository = self.document_repo()
        if not _hybrid_enabled() or repository is None:
            return self.retrieve(query, limit=limit, threshold=threshold, metadata=metadata)
        self.last_retrieval_note = ""
        vector_hits = self._vector_hits(query, limit=limit * 2, threshold=threshold, metadata=metadata)
        keyword_hits = [
            (chunk_id, float(score))
            for chunk_id, score in repository.search_keywords(query, limit=limit * 2)
        ]
        # U4：两路的原始分数在融合前留一份，否则 RRF 只留下名次、贡献不可见。
        vector_scores = dict(vector_hits)
        keyword_scores = dict(keyword_hits)
        results: list[RetrievedChunk] = []
        fused = _rrf_fuse([[chunk_id for chunk_id, _ in vector_hits],
                           [chunk_id for chunk_id, _ in keyword_hits]])[:limit]
        for chunk_id, score in fused:
            chunk = repository.get_chunk(chunk_id)
            if chunk is not None:  # 真值源没有的分块不返回（孤立向量不外泄）
                results.append(
                    RetrievedChunk(
                        chunk.text, score, chunk.chunk_id, _chunk_metadata(chunk),
                        detail={
                            # 与对外 score 同源同值（不在这里四舍五入，展示精度交给前端）
                            "rrf_score": float(score),
                            "vector_score": vector_scores.get(chunk_id),
                            "keyword_score": keyword_scores.get(chunk_id),
                        },
                    )
                )
        return results

    def hybrid_retrieve_multi(self, queries: list[str], *, limit: int = RAG_RETRIEVE_LIMIT, threshold: float | None = None, metadata: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        """F3：每条子查询各跑向量+关键词路，N 路 rank 列表一起丢给 RRF 融合。"""

        queries = [query for query in queries if isinstance(query, str) and query.strip()]
        if not queries:
            return []
        if len(queries) == 1:
            return self.hybrid_retrieve(queries[0], limit=limit, threshold=threshold, metadata=metadata)
        repository = self.document_repo()
        if not _hybrid_enabled() or repository is None:
            results: list[RetrievedChunk] = []
            seen: set[str] = set()
            for query in queries:
                for chunk in self.retrieve(query, limit=limit, threshold=threshold, metadata=metadata):
                    if chunk.memory_id in seen:
                        continue
                    seen.add(chunk.memory_id)
                    results.append(chunk)
            return results[:limit]
        self.last_retrieval_note = ""
        vector_rank_lists: list[list[str]] = []
        keyword_rank_lists: list[list[str]] = []
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}
        for query in queries:
            vector_hits = self._vector_hits(query, limit=limit * 2, threshold=threshold, metadata=metadata)
            keyword_hits = [
                (chunk_id, float(score))
                for chunk_id, score in repository.search_keywords(query, limit=limit * 2)
            ]
            vector_rank_lists.append([chunk_id for chunk_id, _ in vector_hits])
            keyword_rank_lists.append([chunk_id for chunk_id, _ in keyword_hits])
            vector_scores.update(vector_hits)
            keyword_scores.update(keyword_hits)
        fused = _rrf_fuse([*(vector_rank_lists), *(keyword_rank_lists)])[:limit]
        results = []
        for chunk_id, score in fused:
            chunk = repository.get_chunk(chunk_id)
            if chunk is not None:  # 真值源没有的分块不返回（孤立向量不外泄）
                results.append(
                    RetrievedChunk(
                        chunk.text, score, chunk.chunk_id, _chunk_metadata(chunk),
                        detail={
                            "rrf_score": float(score),
                            "vector_score": vector_scores.get(chunk_id),
                            "keyword_score": keyword_scores.get(chunk_id),
                        },
                    )
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

    def graph_retrieve_multi(self, queries: list[str], *, limit: int = RAG_RETRIEVE_LIMIT, hops: int = RAG_GRAPH_HOPS) -> GraphRAGResult:
        """F3：多路图检索（分解后的子查询分别查，路径融合）。"""

        return self.graph.retrieve_multi(queries, limit=limit, hops=hops)

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
