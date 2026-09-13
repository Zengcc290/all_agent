"""Built-in Agent tool for storing one new memory item.

Split out of the former combined ``memory.manage`` tool so that a caller can
grant the narrow "remember this" write without also granting delete/clear. The
tool stays ``side_effect="write"``: the runtime still requires an explicit
confirmation key before it runs.

The default manager writes to ``MEMORY_DB_PATH`` (or ``memory.sqlite3`` next to
the project); applications that need a different backend inject their own
``MemoryManager``.
"""

from __future__ import annotations

from typing import Any

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


class MemoryAddInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    content: str = Field(min_length=1, description="Text to remember.")
    memory_type: MemoryScope = Field(
        default="working",
        description=(
            "Which memory layer to write to. Durable user facts and preferences "
            "belong in 'episodic' or 'semantic'; 'working' expires with the session."
        ),
    )
    item_id: str | None = None
    metadata: list[MemoryMetadata] | None = None
    importance: float = Field(default=0.5, ge=0, le=1)
    ttl_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> Any:
        return normalize_metadata_payload(value)


class MemoryAddOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str = "add"
    count: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)


class MemoryAddTool(BaseTool):
    spec = ToolSpec(
        name="memory.add",
        description="Store one new item in the agent memory system.",
        version="1.0.0",
        input_model=MemoryAddInput,
        output_model=MemoryAddOutput,
        side_effect="write",
        permissions=("memory.write",),
        timeout_seconds=10.0,
        idempotent=False,
        parallel_safe=False,
        tags=("memory", "storage", "write"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        # Created lazily so importing/discovering the tool never opens SQLite.
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: MemoryAddInput) -> MemoryAddOutput:
        item = self.manager.add(
            arguments.content,
            memory_type=MemoryType(arguments.memory_type),
            metadata=metadata_dict(arguments.metadata),
            importance=arguments.importance,
            ttl_seconds=arguments.ttl_seconds,
            item_id=arguments.item_id,
        )
        return MemoryAddOutput(action="add", count=1, items=[item.to_dict()])


def create_tool() -> BaseTool:
    return MemoryAddTool()


__all__ = [
    "MemoryAddInput",
    "MemoryAddOutput",
    "MemoryAddTool",
    "MemoryMetadata",
    "create_tool",
]
