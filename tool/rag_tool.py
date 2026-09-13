"""Built-in Agent tool for ingesting text or files into agent memory.

Split out of the former combined ``memory.rag`` tool so that read-only retrieval
(``memory.rag_search``) never carries the ingest write confirmation. Ingestion
stays ``side_effect="write"``.

A model-supplied ``source`` is resolved through the same workspace sandbox as
the ``fs.*`` tools, so the tool cannot be talked into reading an arbitrary path
outside the workspace. The default pipeline persists to ``MEMORY_DB_PATH`` (or
``memory.sqlite3`` next to the project); inject a custom ``RAGPipeline`` for
different backends.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory.rag import Document, RAGPipeline

from ._memory import build_default_pipeline
from ._shared import resolve_path, workspace_root


TOOL_ENABLED = True


class RAGToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["ingest"] = "ingest"
    text: str | None = Field(default=None, description="Literal text to ingest.")
    source: str | None = Field(
        default=None,
        description=(
            "Workspace-relative path of a file to ingest. Provide either text or "
            "source, never both; paths outside the workspace are rejected."
        ),
    )
    chunk_size: int = Field(default=1000, ge=1, le=100000)
    overlap: int = Field(default=100, ge=0)


class RAGToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    count: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)


class RAGTool(BaseTool):
    spec = ToolSpec(
        name="memory.rag",
        description=(
            "Ingest text or a workspace file into agent memory, extracting "
            "knowledge for later retrieval."
        ),
        version="2.0.0",
        input_model=RAGToolInput,
        output_model=RAGToolOutput,
        side_effect="write",
        permissions=("memory.write",),
        timeout_seconds=30.0,
        idempotent=False,
        parallel_safe=False,
        tags=("memory", "rag", "ingest", "write"),
    )

    def __init__(self, pipeline: RAGPipeline | None = None) -> None:
        # Created lazily so importing/discovering the tool never opens SQLite.
        self._pipeline = pipeline

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = build_default_pipeline()
        return self._pipeline

    def execute(self, arguments: RAGToolInput) -> RAGToolOutput:
        if arguments.text is None and arguments.source is None:
            raise ValueError("text or source is required for ingest")
        if arguments.text is not None and arguments.source is not None:
            raise ValueError("provide either text or source, not both")
        if arguments.text is not None:
            values = self.pipeline.ingest(
                Document(arguments.text),
                chunk_size=arguments.chunk_size,
                overlap=arguments.overlap,
            )
        else:
            assert arguments.source is not None  # narrowed by the check above
            resolved = resolve_path(workspace_root(), arguments.source)
            values = self.pipeline.ingest_source(
                resolved,
                chunk_size=arguments.chunk_size,
                overlap=arguments.overlap,
            )
        return RAGToolOutput(
            action="ingest",
            count=len(values),
            items=[item.to_dict() for item in values],
            report=self.pipeline.last_ingest_report,
        )


def create_tool() -> BaseTool:
    return RAGTool()


__all__ = ["RAGTool", "RAGToolInput", "RAGToolOutput", "create_tool"]
