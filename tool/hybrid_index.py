"""混合索引工具：一段文本同时写入「关键词真值源」与「向量投影」。

职责边界（单一职责）
====================

本工具只做一件事：把**一个**分块写进两套投影，并保证顺序与状态一致。

1. 真值源：SQLite ``documents``/``chunks``（``DocumentRepository.upsert_chunk``）。
   ``chunks`` 上的 FTS5 触发器顺带维护关键词索引，所以写进去就等于关键词可召回。
2. 向量投影：``MemoryManager.add``（``memories`` 行 + 向量库 upsert）。

顺序不可颠倒：先写真值源再写向量。反过来的话，向量写成功而真值行失败就会留下
无法解释的孤立向量。写入成功后把分块的 ``vector_status`` 置为 ``indexed``，让
``/api/reconcile`` 与重嵌入脚本能分辨「已投影 / 待投影」。

本模块是这条双写路径的**唯一实现**：``memory.rag.pipeline.RAGPipeline.ingest``
不再内联这三步，而是调用这里的 :func:`index_chunk`。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryItem, MemoryManager, MemoryType
from memory.storage.document_repo import ChunkRecord, DocumentRepository

TOOL_ENABLED = True

#: 单次索引允许的最大字符数（与 WEB_KNOWLEDGE_MAX_CHARS 同量级，防止无界输入）。
MAX_INDEX_CHARS = 200_000


class IndexMetadataEntry(BaseModel):
    """One caller-supplied metadata pair; keeps the Input free of open dicts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    key: str = Field(min_length=1, max_length=100, description="元数据键。")
    value: str = Field(max_length=2000, description="元数据值。")


def _metadata_dict(entries: list[IndexMetadataEntry] | None) -> dict[str, str]:
    return {entry.key: entry.value for entry in (entries or [])}


def _default_manager() -> MemoryManager:
    """Build the shared on-disk manager (imported lazily: ``_memory`` pulls ``memory.rag``)."""

    from ._memory import build_default_manager

    return build_default_manager()


def repository_for(manager: MemoryManager) -> DocumentRepository | None:
    """Return the ``documents``/``chunks`` truth source for ``manager``.

    Mirrors ``RAGPipeline.document_repo``: a non-SQLite document store and
    ``:memory:`` both return ``None`` — a second in-memory database would be a
    private connection that shares no data with the store it mirrors.
    """

    path = getattr(manager.document_store, "path", None)
    if not path or str(path) == ":memory:":
        return None
    return DocumentRepository(str(path))


def index_chunk(
    manager: MemoryManager,
    repository: DocumentRepository | None,
    *,
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    char_start: int,
    char_end: int,
    text: str,
    metadata: Mapping[str, Any] | None = None,
) -> MemoryItem:
    """Write one chunk into both projections and return the vector-side item.

    ``repository=None`` means the keyword side is unavailable (non-SQLite or
    in-memory document store); the vector side is still written so callers keep
    working, and the returned item is the single source of truth for the caller.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if repository is not None:
        repository.upsert_chunk(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=int(chunk_index),
                char_start=int(char_start),
                char_end=int(char_end),
                text=text,
            )
        )
    item = manager.add(
        text,
        memory_type=MemoryType.SEMANTIC,
        metadata=dict(metadata or {}),
        item_id=chunk_id,
    )
    if repository is not None:
        repository.set_chunk_vector_status(chunk_id, "indexed")
    return item


class HybridIndexInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(
        min_length=1,
        max_length=MAX_INDEX_CHARS,
        description="要入库的原文分块；同时成为关键词索引与向量投影的内容。",
    )
    document_id: str | None = Field(
        default=None,
        description="所属文档 id；为空时用 source 或 'manual' 生成一个稳定 id。",
    )
    chunk_index: int = Field(
        default=0, ge=0, le=1_000_000, description="分块在文档内的序号，从 0 开始。"
    )
    char_start: int = Field(
        default=0, ge=0, description="分块在文档归一化正文中的起始字符偏移。"
    )
    char_end: int = Field(
        default=0, ge=0, description="分块在文档归一化正文中的结束字符偏移。"
    )
    source: str = Field(
        default="", max_length=500, description="来源标识（文件名或 URL），可为空串。"
    )
    filename: str = Field(default="", max_length=500, description="原始文件名，可为空串。")
    metadata: list[IndexMetadataEntry] | None = Field(
        default=None, description="附加元数据键值对；没有就传 null。"
    )


class HybridIndexOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    document_id: str
    memory_id: str
    keyword_indexed: bool = Field(description="真值源（FTS5 关键词索引）是否写入成功。")
    vector_indexed: bool = Field(description="向量投影是否写入成功。")
    vector_status: Literal["indexed", "pending"]
    character_count: int


class HybridIndexTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.hybrid_index",
        description=(
            "Index one text chunk into BOTH the keyword truth source (SQLite "
            "chunks + FTS5) and the vector projection. Use it when raw text must "
            "become retrievable by exact terms and by meaning at the same time. "
            "It does not run LLM extraction and does not create graph facts."
        ),
        version="1.0.0",
        input_model=HybridIndexInput,
        output_model=HybridIndexOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=60.0,
        idempotent=True,
        parallel_safe=False,
        tags=("memory", "index", "hybrid", "fts5", "vector", "write"),
        guidance=(
            "需要让一段原文同时可被关键词与语义检索时使用。"
            "不要用它写记忆条目或抽知识（那走 memory.rag 与 knowledge.add_fact），也不要用它改图节点属性（走 knowledge.graph_node_update）。"
            "同一 chunk_id 重复索引是覆盖写、幂等；换 document_id 会产生新的分块行。写操作需要人工确认钥匙。"
        ),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        # Lazily built so importing/discovering the tool never opens SQLite.
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = _default_manager()
        return self._manager

    def execute(self, arguments: HybridIndexInput) -> HybridIndexOutput:
        document_id = (
            arguments.document_id
            or arguments.source
            or arguments.filename
            or "manual"
        )
        chunk_id = f"{document_id}:{arguments.chunk_index}"
        metadata: dict[str, Any] = dict(_metadata_dict(arguments.metadata))
        metadata.setdefault("source", arguments.source or document_id)
        if arguments.filename:
            metadata.setdefault("filename", arguments.filename)
        metadata.setdefault("document_id", document_id)
        metadata.setdefault("chunk_index", arguments.chunk_index)
        metadata.setdefault("char_start", arguments.char_start)
        metadata.setdefault("char_end", arguments.char_end)
        manager = self.manager
        repository = repository_for(manager)
        item = index_chunk(
            manager,
            repository,
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=arguments.chunk_index,
            char_start=arguments.char_start,
            char_end=arguments.char_end,
            text=arguments.text,
            metadata=metadata,
        )
        return HybridIndexOutput(
            chunk_id=chunk_id,
            document_id=document_id,
            memory_id=item.id,
            keyword_indexed=repository is not None,
            vector_indexed=True,
            vector_status="indexed" if repository is not None else "pending",
            character_count=len(arguments.text),
        )


def create_tool() -> BaseTool:
    return HybridIndexTool()


__all__ = [
    "HybridIndexInput",
    "HybridIndexOutput",
    "HybridIndexTool",
    "IndexMetadataEntry",
    "create_tool",
    "index_chunk",
    "repository_for",
]
