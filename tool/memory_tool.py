"""Built-in Agent tool for deleting or clearing agent memory.

Administrative counterpart to ``memory.query``/``memory.add``. Both actions are
destructive, so the tool is ``side_effect="write"`` and the runtime requires an
explicit confirmation key; nothing in the web chat path grants one, which means
a model cannot erase memory on its own.

The default manager targets ``MEMORY_DB_PATH`` (or ``memory.sqlite3`` next to the
project); applications that need a different backend inject their own
``MemoryManager``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryManager, MemoryType

from ._memory import MemoryScope, build_default_manager

TOOL_ENABLED = True


class MemoryManageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["delete", "clear"]
    memory_type: MemoryScope | None = Field(
        default=None,
        description=(
            "Which memory layer to target; defaults to 'working', so 'clear' "
            "never wipes the whole store by accident."
        ),
    )
    item_id: str | None = Field(default=None, description="Item id for 'delete'.")


class MemoryManageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    count: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)


class MemoryManageTool(BaseTool):
    spec = ToolSpec(
        name="memory.manage",
        description=(
            "Delete one memory item or clear an entire memory layer. Destructive; "
            "requires explicit user confirmation."
        ),
        version="3.0.0",
        input_model=MemoryManageInput,
        output_model=MemoryManageOutput,
        side_effect="destructive",
        permissions=("memory.write",),
        timeout_seconds=10.0,
        idempotent=False,
        parallel_safe=False,
        tags=("memory", "storage", "admin"),
        guidance=(
            "删除单条记忆或清空整层记忆，破坏性操作，必须已获得用户确认。清空整层前必须让用户明确说出层名；不确定时先用 memory.query 列出候选。不要在无人确认的情况下批量删除。"
        ),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        # Created lazily so importing/discovering the tool never opens SQLite.
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: MemoryManageInput) -> MemoryManageOutput:
        action = arguments.action
        memory_type = MemoryType(arguments.memory_type or "working")
        count = 0
        if action == "delete":
            if arguments.item_id is None:
                raise ValueError("item_id is required for delete")
            count = int(self.manager.delete(arguments.item_id, memory_type=memory_type))
        else:
            count = self.manager.clear(memory_type=memory_type)
        return MemoryManageOutput(action=action, count=count, items=[])


def create_tool() -> BaseTool:
    return MemoryManageTool()


__all__ = [
    "MemoryManageInput",
    "MemoryManageOutput",
    "MemoryManageTool",
    "create_tool",
]
