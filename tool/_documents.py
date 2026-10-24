"""文档真值源（``documents``/``chunks``）的共享序列化助手。

这是**不可发现**模块：``core.discovery`` 会忽略以 ``_`` 开头的模块，本文件既没有
``TOOL_ENABLED`` 也没有 ``create_tool()``。三个文档类工具
（``knowledge.document_list`` / ``knowledge.document_get`` /
``knowledge.document_revectorize``）共用这里的记录序列化逻辑，避免三份拷贝各自漂移。

仓储本身不在这里新建：``repository_for(manager)`` 已在 ``tool/hybrid_index.py`` 里
按「:memory: → None」的口径实现，这里直接复用那一份，绝不写第二份。
"""

from __future__ import annotations

from typing import Any

from memory.storage.document_repo import ChunkRecord, DocumentRecord


def chunk_summary(chunk: ChunkRecord) -> dict[str, Any]:
    """One chunk row as the JSON shape the API and the tools both expose."""

    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "text": chunk.text,
        "vector_status": chunk.vector_status,
    }


def document_summary(document: DocumentRecord, *, chunk_count: int) -> dict[str, Any]:
    """One document row plus its chunk count (list view: no raw text)."""

    return {
        "document_id": document.document_id,
        "title": document.title,
        "source": document.source,
        "tags": document.tags,
        "status": document.status,
        "chunk_count": chunk_count,
        "created_at": document.created_at,
    }


def document_detail(document: DocumentRecord, chunks: list[ChunkRecord]) -> dict[str, Any]:
    """Full document view: metadata, raw text and every chunk (detail view)."""

    return {
        "document_id": document.document_id,
        "title": document.title,
        "raw_text": document.raw_text,
        "source": document.source,
        "tags": document.tags,
        "permission": document.permission,
        "status": document.status,
        "error": document.error,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "chunks": [chunk_summary(chunk) for chunk in chunks],
    }


__all__ = ["chunk_summary", "document_detail", "document_summary"]
