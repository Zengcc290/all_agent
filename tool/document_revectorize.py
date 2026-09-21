"""单文档重嵌入工具：只重建这一篇文档的向量投影（写工具）。

为什么这是一个独立能力
======================

投影落后于真值源时（云端嵌入当时不可达、进程被 kill），最省事的修复不是全量重建，
而是"只重灌这一篇"。这是有边界的写操作：只动这篇文档的 chunk 向量与状态，
不碰其它文档、不碰图。

闸门顺序与旧实现逐字一致（重要）
================================

先过嵌入锁闸门 ``apply_embedding_lock``，再看文档是否存在、是否有分块。
顺序不能换：锁不一致时必须先报 409（嵌入空间不一致比"文档不存在"更严重），
这是原来的行为，也是唯一安全的行为——否则会往错误的向量空间里写。

``confirm_rebuild`` 默认 false：锁不一致时抛 ``EmbeddingLockMismatch``（明确失败），
只有调用方显式确认才会"重建投影并全量重灌"。绝不静默切换向量空间。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 里内联的重嵌入循环已删除，
端点只保留 409/404/422/502 的错误码映射。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory.base import MemoryType
from memory.embedding_lock import apply_embedding_lock
from memory.manager import MemoryManager
from memory.storage.document_repo import DocumentRepository

from .hybrid_index import repository_for

TOOL_ENABLED = True


def revectorize_document(
    manager: MemoryManager,
    repository: DocumentRepository,
    document_id: str,
    *,
    confirm_rebuild: bool = False,
) -> dict[str, Any]:
    """Re-embed one document's chunks and set its status.

    Order matters: the embedding-lock gate runs first (a mismatched embedding space
    must fail with the lock error, not with a 404), then existence, then chunks.
    """

    apply_embedding_lock(manager, repository, confirm_rebuild=confirm_rebuild)
    document = repository.get_document(document_id)
    if document is None:
        raise LookupError(f"文档不存在：{document_id}")
    chunks = repository.list_chunks(document_id)
    if not chunks:
        raise ValueError("该文档没有分块，无法重嵌入")
    vectors = manager.embedding.embed_batch([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        manager.vector_store.upsert_chunk(
            chunk.chunk_id,
            vector,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            source=document.source,
            memory_type=MemoryType.SEMANTIC.value,
        )
        repository.set_chunk_vector_status(chunk.chunk_id, "indexed")
    status = "extracted" if document.status == "extracted" else "vectorized"
    repository.set_status(document_id, status)
    return {"document_id": document_id, "chunks_reindexed": len(chunks), "status": status}


class DocumentRevectorizeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str = Field(min_length=1, max_length=200, description="要重嵌入的文档 id。")
    confirm_rebuild: bool = Field(
        default=False,
        description=(
            "嵌入锁不一致时是否确认重建整个向量投影并全量重灌（破坏性）。"
            "默认 false 表示明确失败，绝不静默切换向量空间。"
        ),
    )


class DocumentRevectorizeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    chunks_reindexed: int = Field(description="本次重新嵌入的分块数。")
    status: str = Field(description="写入后的文档状态：vectorized，或保持 extracted。")


class DocumentRevectorizeTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.document_revectorize",
        description=(
            "Re-embed one document's chunks into the vector projection and mark them "
            "indexed. Fails loudly on an embedding-space mismatch unless "
            "confirm_rebuild is set, because silently mixing embedding spaces "
            "corrupts retrieval."
        ),
        version="1.0.0",
        input_model=DocumentRevectorizeInput,
        output_model=DocumentRevectorizeOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=600.0,
        idempotent=True,
        parallel_safe=False,
        tags=("knowledge", "document", "embedding", "write"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: DocumentRevectorizeInput) -> DocumentRevectorizeOutput:
        repository = repository_for(self.manager)
        if repository is None:
            raise LookupError(
                "当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源"
            )
        result = revectorize_document(
            self.manager,
            repository,
            arguments.document_id,
            confirm_rebuild=arguments.confirm_rebuild,
        )
        return DocumentRevectorizeOutput(**result)


def create_tool() -> BaseTool:
    return DocumentRevectorizeTool()


__all__ = [
    "DocumentRevectorizeInput",
    "DocumentRevectorizeOutput",
    "DocumentRevectorizeTool",
    "create_tool",
    "revectorize_document",
]
