"""Built-in Agent tool for storing and retrieving HelloAgents memories."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core import BaseTool, ToolSpec
from memory import MemoryConfig, MemoryManager, MemoryType


TOOL_ENABLED = True


class MemoryToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["add", "search", "get", "delete", "list", "clear"]
    content: str | None = Field(default=None, description="Text to store for add.")
    memory_type: Literal["working", "episodic", "semantic", "perceptual"] = "working"
    item_id: str | None = None
    query: str | None = None
    metadata: list["MemoryMetadata"] | None = None
    importance: float = Field(default=0.5, ge=0, le=1)
    ttl_seconds: float | None = Field(default=None, gt=0)
    limit: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("metadata"), dict):
            value = dict(value)
            value["metadata"] = [{"key": key, "value": str(item)} for key, item in value["metadata"].items()]
        return value


class MemoryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: str = Field(min_length=1)
    value: str


class MemoryToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    count: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)


class MemoryTool(BaseTool):
    spec = ToolSpec(
        name="memory.manage",
        description="Store, search, inspect, list, or delete information in the agent memory system.",
        version="1.0.0",
        input_model=MemoryToolInput,
        output_model=MemoryToolOutput,
        side_effect="write",
        permissions=("memory.write",),
        timeout_seconds=10.0,
        idempotent=False,
        parallel_safe=False,
        tags=("memory", "storage", "search"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self.manager = manager or MemoryManager(MemoryConfig(sqlite_path=":memory:"))

    def execute(self, arguments: MemoryToolInput) -> MemoryToolOutput:
        action = arguments.action
        memory_type = MemoryType(arguments.memory_type)
        items: list[dict[str, Any]] = []
        count = 0
        if action == "add":
            if arguments.content is None:
                raise ValueError("content is required for add")
            metadata = {entry.key: entry.value for entry in (arguments.metadata or [])}
            item = self.manager.add(arguments.content, memory_type=memory_type, metadata=metadata, importance=arguments.importance, ttl_seconds=arguments.ttl_seconds, item_id=arguments.item_id)
            items = [item.to_dict()]
            count = 1
        elif action == "search":
            if arguments.query is None:
                raise ValueError("query is required for search")
            metadata = {entry.key: entry.value for entry in (arguments.metadata or [])}
            results = self.manager.search(arguments.query, memory_type=memory_type, limit=arguments.limit, metadata=metadata)
            items = [result.to_dict() for result in results]
            count = len(items)
        elif action == "get":
            if arguments.item_id is None:
                raise ValueError("item_id is required for get")
            item = self.manager.get(arguments.item_id, memory_type=memory_type)
            items = [item.to_dict()] if item is not None else []
            count = len(items)
        elif action == "delete":
            if arguments.item_id is None:
                raise ValueError("item_id is required for delete")
            count = int(self.manager.delete(arguments.item_id, memory_type=memory_type))
        elif action == "list":
            values = self.manager.list(memory_type=memory_type)
            items = [item.to_dict() for item in values[: arguments.limit]]
            count = len(items)
        else:
            count = self.manager.clear(memory_type=memory_type)
        return MemoryToolOutput(action=action, count=count, items=items)


def create_tool() -> BaseTool:
    return MemoryTool()


__all__ = ["MemoryMetadata", "MemoryTool", "MemoryToolInput", "MemoryToolOutput", "create_tool"]
