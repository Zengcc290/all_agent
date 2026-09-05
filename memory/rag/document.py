"""Document normalization and chunking for the memory-backed RAG pipeline."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


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


class DocumentProcessor:
    """Parse common local formats and split text into searchable chunks."""

    def parse(self, source: str | Path | io.TextIOBase | bytes, *, metadata: Mapping[str, Any] | None = None) -> Document:
        is_path = isinstance(source, Path)
        if isinstance(source, str) and "\n" not in source:
            try:
                is_path = Path(source).exists()
            except OSError:
                is_path = False
        if is_path:
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

    def chunks(self, document: Document, *, chunk_size: int = 1000, overlap: int = 100) -> list[Document]:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        text = re.sub(r"\s+", " ", document.content).strip()
        if not text:
            return []
        step = chunk_size - overlap
        result: list[Document] = []
        for index, start in enumerate(range(0, len(text), step)):
            chunk = text[start : start + chunk_size]
            if not chunk:
                break
            metadata = dict(document.metadata)
            metadata.update({"document_id": document.id, "chunk_index": index})
            result.append(Document(chunk, id=f"{document.id}:{index}", metadata=metadata))
            if start + chunk_size >= len(text):
                break
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


__all__ = ["Document", "DocumentProcessor"]
