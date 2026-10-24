"""文档详情工具：读一篇文档的元数据、原文与全部分块（只读）。

为什么这是一个独立能力
======================

详情是"从投影回到真值源"的动作：向量命中了某个 ``chunk_id``，要核对原文、
看分块边界、看它是否已投影（``vector_status``），都必须读这里。
它只读，所以可以随时调用；缺失文档用 ``LookupError`` 表达，
由调用方决定是 404 还是别的错误码。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 里内联的详情构造已删除。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory.manager import MemoryManager
from memory.storage.document_repo import DocumentRepository

from ._documents import document_detail
from .hybrid_index import repository_for

TOOL_ENABLED = True


def get_document(repository: DocumentRepository, document_id: str) -> dict[str, Any]:
    """Full document view (metadata + raw text + every chunk).

    Raises ``LookupError`` when the document does not exist.
    """

    document = repository.get_document(document_id)
    if document is None:
        raise LookupError(f"文档不存在：{document_id}")
    return document_detail(document, repository.list_chunks(document_id))


class DocumentGetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str = Field(min_length=1, max_length=200, description="文档 id（来自 knowledge.document_list）。")
    max_chunks: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="最多返回多少个分块（chunk_count 始终是完整数量）。",
    )


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    text: str = ""
    vector_status: str = ""


class DocumentGetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    title: str = ""
    raw_text: str = ""
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    permission: str = ""
    status: str = ""
    error: str | None = Field(default=None, description="入库/抽取失败原因；成功时为 null。")
    created_at: str = ""
    updated_at: str = ""
    chunk_count: int = Field(description="分块总数（不受 max_chunks 影响）。")
    chunks: list[DocumentChunk] = Field(default_factory=list)
    truncated: bool = Field(description="true 表示分块被 max_chunks 截断。")


class DocumentGetTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.document_get",
        description=(
            "Read one document from the truth source: metadata, raw text and its "
            "chunks (each with vector_status). Use it to verify what a retrieval "
            "hit actually says before quoting it."
        ),
        version="1.0.0",
        input_model=DocumentGetInput,
        output_model=DocumentGetOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=60.0,
        idempotent=True,
        parallel_safe=True,
        tags=("knowledge", "document", "detail", "read"),
        guidance=(
            "引用或核对某篇文档的原文时使用（先用 knowledge.document_list 拿 id）。"
            "max_chunks 只裁剪返回的分块列表，chunk_count 仍是全量，判断是否读完要看 truncated。文档不存在会明确报错，不要反复重试同一个 id。"
        ),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: DocumentGetInput) -> DocumentGetOutput:
        repository = repository_for(self.manager)
        if repository is None:
            raise LookupError(
                "当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源"
            )
        payload = get_document(repository, arguments.document_id)
        chunks = [DocumentChunk(**chunk) for chunk in payload["chunks"]]
        return DocumentGetOutput(
            document_id=payload["document_id"],
            title=payload["title"],
            raw_text=payload["raw_text"],
            source=payload["source"],
            tags=list(payload["tags"]),
            permission=payload["permission"],
            status=payload["status"],
            error=payload["error"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            chunk_count=len(chunks),
            chunks=chunks[: arguments.max_chunks],
            truncated=len(chunks) > arguments.max_chunks,
        )


def create_tool() -> BaseTool:
    return DocumentGetTool()


__all__ = [
    "DocumentChunk",
    "DocumentGetInput",
    "DocumentGetOutput",
    "DocumentGetTool",
    "create_tool",
    "get_document",
]
