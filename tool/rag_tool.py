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
from memory.rag import LLMKnowledgeExtractor, NullKnowledgeExtractor, RAGPipeline


TOOL_ENABLED = True


class RAGToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["ingest", "retrieve", "context", "graph_retrieve", "graph_context"]
    text: str | None = None
    query: str | None = None
    source: str | None = None
    document_id: str | None = None
    limit: int = Field(default=5, ge=1, le=50)
    chunk_size: int = Field(default=1000, ge=1, le=100000)
    overlap: int = Field(default=100, ge=0)
    hops: int = Field(default=1, ge=0, le=3)


class RAGToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    context: str = ""
    items: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    entities: list[str] = Field(default_factory=list)
    paths: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)


class RAGTool(BaseTool):
    spec = ToolSpec(
        name="memory.rag",
        description="Ingest text, automatically extract knowledge, and retrieve vector plus graph context from agent memory.",
        version="1.1.0",
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
            extractor = NullKnowledgeExtractor()
            try:
                from agents.llm import LLM
                from agents.providers import ProviderRegistry

                registry = ProviderRegistry()
                profile = registry.get(registry.active_profile)
                key = registry.resolve_api_key(profile.name)
                if key and not key.startswith("replace-with"):
                    client = LLM(api_key=key, base_url=profile.base_url, model=profile.default_model)
                    extractor = LLMKnowledgeExtractor(client.complete, model=profile.default_model)
            except Exception:
                pass
            self._pipeline = RAGPipeline(
                MemoryManager(MemoryConfig(sqlite_path=default_sqlite_path())),
                extractor=extractor,
            )
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
            return RAGToolOutput(
                action="ingest",
                count=len(values),
                items=[item.to_dict() for item in values],
                report=self.pipeline.last_ingest_report,
            )
        if arguments.action == "retrieve":
            if arguments.query is None:
                raise ValueError("query is required for retrieve")
            values = self.pipeline.retrieve(arguments.query, limit=arguments.limit)
            return RAGToolOutput(action="retrieve", count=len(values), items=[{"content": item.content, "score": item.score, "memory_id": item.memory_id, "metadata": dict(item.metadata)} for item in values])
        if arguments.query is None:
            raise ValueError("query is required for context or graph retrieval")
        if arguments.action == "graph_retrieve":
            result = self.pipeline.graph_retrieve(arguments.query, limit=arguments.limit, hops=arguments.hops)
            return RAGToolOutput(
                action=arguments.action,
                count=len(result.evidence),
                items=[item.to_dict() for item in result.evidence],
                entities=result.entities,
                paths=[path.to_dict() for path in result.paths],
                context=result.build_context(),
            )
        if arguments.action == "graph_context":
            context = self.pipeline.graph_context(arguments.query, limit=arguments.limit, hops=arguments.hops)
            return RAGToolOutput(action=arguments.action, count=1 if context else 0, context=context)
        context = self.pipeline.build_context(arguments.query, limit=arguments.limit)
        return RAGToolOutput(action="context", count=1 if context else 0, context=context)


def create_tool() -> BaseTool:
    return RAGTool()


__all__ = ["RAGTool", "RAGToolInput", "RAGToolOutput", "create_tool"]
