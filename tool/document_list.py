"""文档列表工具：按标签/状态分页浏览真值源里的文档（只读）。

为什么这是一个独立能力
======================

``documents``/``chunks`` 是**真值源**：向量与图都只是它的投影。所以"库里有哪几篇文档、
各自什么状态、有多少分块"是任何检索之前最该先问的问题——Agent 过去问不了，
因为它只能通过 ``GET /api/documents`` 这个 Web 端点看到。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 里内联的列表构造与分页校验已删除，
端点只保留 422 错误码映射。分页上限口径来自 ``constants.WEB_DOCUMENTS_PAGE_SIZE_MAX``
（与仓储层的下界校验互补：仓储只管 ``page_size >= 1``，上限是 API 契约）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from constants import WEB_DOCUMENTS_PAGE_SIZE_MAX
from core import BaseTool, ToolSpec
from memory.manager import MemoryManager
from memory.storage.document_repo import DocumentRepository

from ._documents import document_summary
from .hybrid_index import repository_for

TOOL_ENABLED = True


def list_documents(
    repository: DocumentRepository,
    *,
    tag: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """One page of documents plus the total count and per-document chunk counts.

    Raises ``ValueError`` for a bad ``page_size``/``page`` so the caller can map it
    to its own 4xx without this function knowing about HTTP.
    """

    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= WEB_DOCUMENTS_PAGE_SIZE_MAX
    ):
        raise ValueError(
            f"page_size 必须是 1 到 {WEB_DOCUMENTS_PAGE_SIZE_MAX} 之间的整数"
        )
    items, total = repository.list_documents(
        tag=tag, status=status, page=page, page_size=page_size
    )
    counts = repository.chunk_counts()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            document_summary(item, chunk_count=counts.get(item.document_id, 0))
            for item in items
        ],
    }


class DocumentListInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tag: str = Field(default="", max_length=100, description="按标签过滤；空串表示不过滤。")
    status: str = Field(
        default="",
        max_length=40,
        description="按状态过滤（parsed/vectorized/extracted/failed 等）；空串表示不过滤。",
    )
    page: int = Field(default=1, ge=1, description="页码，从 1 开始。")
    page_size: int = Field(
        default=20,
        ge=1,
        le=WEB_DOCUMENTS_PAGE_SIZE_MAX,
        description=f"每页条目数，上限 {WEB_DOCUMENTS_PAGE_SIZE_MAX}。",
    )


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    title: str = ""
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    status: str = ""
    chunk_count: int = 0
    created_at: str = ""


class DocumentListOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    total: int = Field(description="符合过滤条件的文档总数（不是本页条数）。")
    page: int
    page_size: int
    items: list[DocumentSummary] = Field(default_factory=list)


class DocumentListTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.document_list",
        description=(
            "Browse the documents in the truth source (documents/chunks) with "
            "pagination and optional tag/status filters. Returns each document's "
            "status and chunk count, so it is the right first call before "
            "retrieving or revectorizing anything."
        ),
        version="1.0.0",
        input_model=DocumentListInput,
        output_model=DocumentListOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=60.0,
        idempotent=True,
        parallel_safe=True,
        tags=("knowledge", "document", "list", "read"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: DocumentListInput) -> DocumentListOutput:
        repository = repository_for(self.manager)
        if repository is None:
            raise LookupError(
                "当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源"
            )
        page = list_documents(
            repository,
            tag=arguments.tag,
            status=arguments.status,
            page=arguments.page,
            page_size=arguments.page_size,
        )
        return DocumentListOutput(
            total=page["total"],
            page=page["page"],
            page_size=page["page_size"],
            items=[DocumentSummary(**item) for item in page["items"]],
        )


def create_tool() -> BaseTool:
    return DocumentListTool()


__all__ = [
    "DocumentListInput",
    "DocumentListOutput",
    "DocumentListTool",
    "DocumentSummary",
    "create_tool",
    "list_documents",
]
