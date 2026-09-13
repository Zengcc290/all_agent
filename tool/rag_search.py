"""Built-in Agent tool for vector and graph retrieval over agent memory.

Read-only companion to ``memory.rag`` (which ingests). Declaring
``side_effect="read"`` keeps the runtime's write-confirmation gate out of the
retrieval path, so an agent can ground an answer in stored knowledge without
user approval.

The default pipeline reads ``MEMORY_DB_PATH`` (or ``memory.sqlite3`` next to the
project); inject a custom ``RAGPipeline`` for different backends.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory.rag import RAGPipeline

from ._memory import build_default_pipeline

TOOL_ENABLED = True


class RAGSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["retrieve", "context", "graph_retrieve", "graph_context"]
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    hops: int = Field(default=1, ge=0, le=3)


class RAGSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    context: str = ""
    items: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    entities: list[str] = Field(default_factory=list)
    paths: list[dict[str, Any]] = Field(default_factory=list)


class RAGSearchTool(BaseTool):
    spec = ToolSpec(
        name="memory.rag_search",
        description=(
            "Retrieve vector matches, graph facts, or a ready-made context block "
            "from stored knowledge. Read-only: never ingests."
        ),
        version="1.0.0",
        input_model=RAGSearchInput,
        output_model=RAGSearchOutput,
        side_effect="read",
        permissions=("memory.read",),
        timeout_seconds=30.0,
        idempotent=True,
        parallel_safe=True,
        tags=("memory", "rag", "retrieval", "read"),
    )

    def __init__(self, pipeline: RAGPipeline | None = None) -> None:
        # Created lazily so importing/discovering the tool never opens SQLite.
        self._pipeline = pipeline

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = build_default_pipeline()
        return self._pipeline

    def execute(self, arguments: RAGSearchInput) -> RAGSearchOutput:
        query = arguments.query
        if arguments.action == "retrieve":
            values = self.pipeline.retrieve(query, limit=arguments.limit)
            return RAGSearchOutput(
                action="retrieve",
                count=len(values),
                items=[
                    {
                        "content": item.content,
                        "score": item.score,
                        "memory_id": item.memory_id,
                        "metadata": dict(item.metadata),
                    }
                    for item in values
                ],
            )
        if arguments.action == "graph_retrieve":
            result = self.pipeline.graph_retrieve(
                query, limit=arguments.limit, hops=arguments.hops
            )
            return RAGSearchOutput(
                action=arguments.action,
                count=len(result.evidence),
                items=[item.to_dict() for item in result.evidence],
                entities=result.entities,
                paths=[path.to_dict() for path in result.paths],
                context=result.build_context(),
            )
        if arguments.action == "graph_context":
            context = self.pipeline.graph_context(
                query, limit=arguments.limit, hops=arguments.hops
            )
            return RAGSearchOutput(
                action=arguments.action, count=1 if context else 0, context=context
            )
        context = self.pipeline.build_context(query, limit=arguments.limit)
        return RAGSearchOutput(
            action="context", count=1 if context else 0, context=context
        )


def create_tool() -> BaseTool:
    return RAGSearchTool()


__all__ = [
    "RAGSearchInput",
    "RAGSearchOutput",
    "RAGSearchTool",
    "create_tool",
]
