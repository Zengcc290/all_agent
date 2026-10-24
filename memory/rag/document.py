"""Document normalization and chunking for the memory-backed RAG pipeline."""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from constants import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


@dataclass(frozen=True)
class Document:
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("document content must be a non-empty string")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("document id must be a non-empty string")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ChunkSpan:
    """A chunk plus its character range inside the document's normalized text.

    ``char_start``/``char_end`` index into
    :meth:`DocumentProcessor.normalized_text`, which is also what
    ``documents.raw_text`` stores - that shared origin is what makes the offsets
    usable for re-slicing and for highlighting in the UI (方案 2.3).
    """

    chunk: Document
    char_start: int
    char_end: int


class DocumentProcessor:
    """Parse common local formats and split text into searchable chunks."""

    def parse(self, source: str | Path | io.TextIOBase | bytes, *, metadata: Mapping[str, Any] | None = None) -> Document:
        """Normalize one source into a document.

        ``str`` always means literal text: probing the filesystem for a
        string-shaped path made short text that happened to match a file name
        (and any absolute path) readable, which let a model-supplied
        ``memory.rag`` source escape the workspace sandbox. File input is
        explicit: pass ``Path``/``os.PathLike`` (or use
        :meth:`RAGPipeline.ingest_source`, which converts a string path and can
        enforce containment).
        """

        if isinstance(source, Path) or hasattr(source, "__fspath__"):
            path = Path(source)
            raw = path.read_bytes()
            base = {"source": str(path), "filename": path.name, "extension": path.suffix.lower()}
            base.update(metadata or {})
            return Document(self._parse_bytes(raw, path.suffix.lower()), metadata=base)
        if isinstance(source, bytes):
            return Document(source.decode("utf-8", errors="replace"), metadata=metadata or {})
        if isinstance(source, io.TextIOBase):
            return Document(source.read(), metadata=metadata or {})
        if isinstance(source, str):
            return Document(source, metadata=metadata or {})
        raise TypeError("source must be text, bytes, a path, or a text stream")

    def normalized_text(self, document: Document) -> str:
        """The single normalized form that both ``raw_text`` and chunking consume.

        Whitespace is collapsed here and nowhere else: if ``documents.raw_text``
        and the chunk offsets were derived from two different normalizations,
        ``char_start``/``char_end`` would point at the wrong text.
        """

        return re.sub(r"\s+", " ", document.content).strip()

    def chunks_with_spans(self, document: Document, *, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[ChunkSpan]:
        """Split ``document`` and keep each chunk's range in the normalized text."""
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        text = self.normalized_text(document)
        if not text:
            return []
        step = chunk_size - overlap
        result: list[ChunkSpan] = []
        for index, start in enumerate(range(0, len(text), step)):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if not chunk:
                break
            metadata = dict(document.metadata)
            metadata.update({"document_id": document.id, "chunk_index": index})
            result.append(
                ChunkSpan(Document(chunk, id=f"{document.id}:{index}", metadata=metadata), start, end)
            )
            if end >= len(text):
                break
        return result

    def sentences_with_spans(self, document: Document) -> list[ChunkSpan]:
        """Split ``document`` into sentences and keep each one's character range.

        F4：逐句导入用。句读号是中英文常用的 ``。！？!?.;`` 与换行；句边界
        索引进 :meth:`normalized_text`，与 ``chunks_with_spans`` 同源，偏移
        因此可直接用于重切与高亮。chunk_index 连续编号，句级块用
        ``metadata["granularity"] = "sentences"`` 与字符块区分。
        """

        text = self.normalized_text(document)
        if not text:
            return []
        result: list[ChunkSpan] = []
        start = 0
        for index, char in enumerate(text):
            if char not in "。！？!?.;\n":
                continue
            chunk = text[start : index + 1].strip()
            if chunk:
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "document_id": document.id,
                        "chunk_index": len(result),
                        "granularity": "sentences",
                    }
                )
                end = start + len(chunk)
                result.append(
                    ChunkSpan(
                        Document(chunk, id=f"{document.id}:{len(result)}", metadata=metadata),
                        start,
                        end,
                    )
                )
            start = index + 1
        tail = text[start:].strip()
        if tail:  # 结尾没有句读号的残句也要落库，否则最后一句丢失
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "document_id": document.id,
                    "chunk_index": len(result),
                    "granularity": "sentences",
                }
            )
            begin = start + (len(text[start:]) - len(text[start:].lstrip()))
            result.append(
                ChunkSpan(
                    Document(tail, id=f"{document.id}:{len(result)}", metadata=metadata),
                    begin,
                    begin + len(tail),
                )
            )
        return result

    @staticmethod
    def _parse_bytes(raw: bytes, extension: str) -> str:
        if extension == ".jsonl":
            return "\n".join(json.dumps(json.loads(line), ensure_ascii=False) for line in raw.decode("utf-8").splitlines() if line.strip())
        if extension == ".json":
            value = json.loads(raw.decode("utf-8"))
            return json.dumps(value, ensure_ascii=False, indent=2)
        if extension == ".csv":
            rows = csv.reader(io.StringIO(raw.decode("utf-8", errors="replace")))
            return "\n".join(" | ".join(row) for row in rows)
        if extension in {".html", ".htm"}:
            return re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="replace"))
        if extension == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF parsing requires pypdf") from exc
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return raw.decode("utf-8", errors="replace")


def resolve_within(base_dir: str | Path, path: str | Path) -> Path:
    """Resolve ``path`` and reject anything outside ``base_dir``.

    Used by the ingest entry points that accept a caller-supplied path, so a
    model-driven tool call cannot read arbitrary files. The comparison is done
    on resolved paths with case normalization, mirroring the ``fs.*`` tools'
    containment rule.
    """

    if not isinstance(base_dir, (str, Path)):
        raise TypeError("base_dir must be a string or pathlib.Path")
    base = Path(base_dir).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    base_text = os.path.normcase(str(base))
    resolved_text = os.path.normcase(str(resolved))
    if resolved_text != base_text and not resolved_text.startswith(base_text + os.sep):
        raise ValueError(
            f"path '{path}' resolves outside the allowed directory '{base}'"
        )
    return resolved


__all__ = ["ChunkSpan", "Document", "DocumentProcessor", "resolve_within"]
