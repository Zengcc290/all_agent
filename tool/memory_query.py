"""Built-in Agent tool for reading agent memory.

Read-only companion to ``memory.add``/``memory.manage``. Declaring
``side_effect="read"`` keeps the runtime's write-confirmation gate out of the
retrieval path, so an agent can search its own memory without user approval.

The default manager reads ``MEMORY_DB_PATH`` (or ``memory.sqlite3`` next to the
project); applications that need a different backend inject their own
``MemoryManager``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core import BaseTool, ToolSpec
from memory import MemoryManager, MemoryType

from ._memory import (
    MemoryMetadata,
    MemoryScope,
    build_default_manager,
    metadata_dict,
    normalize_metadata_payload,
)

TOOL_ENABLED = True


class MemoryQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search", "get", "list"]
    memory_type: MemoryScope | None = Field(
        default=None,
        description=(
            "Which memory layer to read. For 'search', leaving it null searches all "
            "four layers (past Q&A and experiences live in 'episodic'); for the "
            "other actions it defaults to 'working'."
        ),
    )
    item_id: str | None = Field(default=None, description="Item id for 'get'.")
    query: str | None = Field(default=None, description="Search text for 'search'.")
    metadata: list[MemoryMetadata] | None = None
    limit: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> Any:
        return normalize_metadata_payload(value)


class MemoryQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    count: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)


class MemoryQueryTool(BaseTool):
    spec = ToolSpec(
        name="memory.query",
        description=(
            "Search, inspect, or list what is stored in the agent memory system. "
            "Read-only: never modifies memory."
        ),
        version="1.0.0",
        input_model=MemoryQueryInput,
        output_model=MemoryQueryOutput,
        side_effect="read",
        permissions=("memory.read",),
        timeout_seconds=10.0,
        idempotent=True,
        parallel_safe=True,
        tags=("memory", "search", "read"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        # Created lazily so importing/discovering the tool never opens SQLite.
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: MemoryQueryInput) -> MemoryQueryOutput:
        action = arguments.action
        scope = arguments.memory_type
        memory_type = MemoryType(scope or "working")
        items: list[dict[str, Any]] = []
        if action == "search":
            if arguments.query is None:
                raise ValueError("query is required for search")
            results = self.manager.search(
                arguments.query,
                memory_type=scope,
                limit=arguments.limit,
                metadata=metadata_dict(arguments.metadata),
            )
            items = [result.to_dict() for result in results]
        elif action == "get":
            if arguments.item_id is None:
                raise ValueError("item_id is required for get")
            item = self.manager.get(arguments.item_id, memory_type=memory_type)
            items = [item.to_dict()] if item is not None else []
        else:
            values = self.manager.list(memory_type=memory_type)
            items = [item.to_dict() for item in values[: arguments.limit]]
        return MemoryQueryOutput(action=action, count=len(items), items=items)


def create_tool() -> BaseTool:
    return MemoryQueryTool()


__all__ = [
    "MemoryMetadata",
    "MemoryQueryInput",
    "MemoryQueryOutput",
    "MemoryQueryTool",
    "create_tool",
]
