"""Built-in Agent tool for retrieval augmented answers.

The default pipeline persists ingested documents to ``MEMORY_DB_PATH`` (or
``memory.sqlite3`` next to the project); inject a custom ``RAGPipeline`` for
different backends.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryConfig, MemoryManager, default_sqlite_path
from memory.rag import RAGPipeline


TOOL_ENABLED = True


class RAGToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["ingest", "retrieve", "context"]
    text: str | None = None
    query: str | None = None
    source: str | None = None
    document_id: str | None = None
    limit: int = Field(default=5, ge=1, le=50)
    chunk_size: int = Field(default=1000, ge=1, le=100000)
    overlap: int = Field(default=100, ge=0)


class RAGToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    context: str = ""
    items: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class RAGTool(BaseTool):
    spec = ToolSpec(
        name="memory.rag",
        description="Ingest local text and retrieve relevant context from agent memory.",
        version="1.0.0",
        input_model=RAGToolInput,
        output_model=RAGToolOutput,
        side_effect="write",
        permissions=("memory.write",),
        timeout_seconds=30.0,
        idempotent=False,
        parallel_safe=False,
        tags=("memory", "rag", "retrieval"),
    )

    def __init__(self, pipeline: RAGPipeline | None = None) -> None:
        # Created lazily so importing/discovering the tool never opens SQLite.
        self._pipeline = pipeline

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = RAGPipeline(MemoryManager(MemoryConfig(sqlite_path=default_sqlite_path())))
        return self._pipeline

    def execute(self, arguments: RAGToolInput) -> RAGToolOutput:
        if arguments.action == "ingest":
            if arguments.text is None and arguments.source is None:
                raise ValueError("text or source is required for ingest")
            if arguments.text is not None and arguments.source is not None:
                raise ValueError("provide either text or source, not both")
            if arguments.text is not None:
                from memory.rag import Document
                values = self.pipeline.ingest(Document(arguments.text), chunk_size=arguments.chunk_size, overlap=arguments.overlap)
            else:
                values = self.pipeline.ingest_source(arguments.source, chunk_size=arguments.chunk_size, overlap=arguments.overlap)
            return RAGToolOutput(action="ingest", count=len(values), items=[item.to_dict() for item in values])
        if arguments.action == "retrieve":
            if arguments.query is None:
                raise ValueError("query is required for retrieve")
            values = self.pipeline.retrieve(arguments.query, limit=arguments.limit)
            return RAGToolOutput(action="retrieve", count=len(values), items=[{"content": item.content, "score": item.score, "memory_id": item.memory_id, "metadata": dict(item.metadata)} for item in values])
        if arguments.query is None:
            raise ValueError("query is required for context")
        context = self.pipeline.build_context(arguments.query, limit=arguments.limit)
        return RAGToolOutput(action="context", count=1 if context else 0, context=context)


def create_tool() -> BaseTool:
    return RAGTool()


__all__ = ["RAGTool", "RAGToolInput", "RAGToolOutput", "create_tool"]
