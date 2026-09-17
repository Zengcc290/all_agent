"""Built-in Agent tool for proposing memory deletions (F2).

The LLM must first retrieve relations before it may propose a deletion, and a
proposal never deletes anything itself: it only persists a pending record that
waits for explicit user confirmation. Execution goes through
``memory.storage.document_repo.execute_deletion`` after the user confirms, so
this tool is ``side_effect="read"`` — the chat path needs no write confirmation
for the proposal step, and no human can delete through it either.

The default manager targets ``MEMORY_DB_PATH`` (or ``memory.sqlite3`` next to
the project); applications that need a different backend inject their own
``MemoryManager``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryManager
from memory.storage.document_repo import DeletionProposalStore

from ._memory import build_default_manager

TOOL_ENABLED = True


class MemoryProposeDeleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["by_query", "by_ids", "by_relation"]
    #: by_query：检索串；by_ids：item_id 列表；by_relation：三元组文本（subject predicate object）。
    target: str = Field(min_length=1, max_length=2000)
    #: LLM 必须给理由；没有理由的提议直接被拒绝。
    reason: str = Field(min_length=1, max_length=600)


class MemoryProposeDeleteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    proposal_id: str
    confirm_token: str
    expires_at: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class MemoryProposeDeleteTool(BaseTool):
    spec = ToolSpec(
        name="memory.propose_delete",
        description=(
            "Propose deleting memory items after retrieving them. Persists a "
            "pending proposal that waits for explicit user confirmation; it "
            "never deletes anything itself."
        ),
        version="1.0.0",
        input_model=MemoryProposeDeleteInput,
        output_model=MemoryProposeDeleteOutput,
        side_effect="read",
        permissions=("memory.read",),
        timeout_seconds=10.0,
        idempotent=True,
        parallel_safe=True,
        tags=("memory", "admin"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        # Created lazily so importing/discovering the tool never opens SQLite.
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: MemoryProposeDeleteInput) -> MemoryProposeDeleteOutput:
        items = self._candidates(arguments)
        if not items:
            return MemoryProposeDeleteOutput(
                proposal_id="",
                confirm_token="",
                expires_at="",
                items=[],
                note="没有检索到任何匹配的记忆，未创建提议。",
            )
        # ``:memory:`` 时与 memories 同库复用同一条连接；文件库传 None。
        connection = getattr(self.manager.document_store, "connection", None)
        store = DeletionProposalStore(self.manager.document_store.path, connection=connection)
        proposal = store.create(
            requested_by="llm",
            reason=arguments.reason,
            item_ids=[item["item_id"] for item in items],
        )
        return MemoryProposeDeleteOutput(
            proposal_id=proposal.proposal_id,
            confirm_token=proposal.confirm_token,
            expires_at=proposal.expires_at,
            items=items,
        )

    def _candidates(self, arguments: MemoryProposeDeleteInput) -> list[dict[str, Any]]:
        """Retrieve candidates; a proposal without retrieved items is refused."""

        if arguments.action == "by_query":
            results = self.manager.search(arguments.target, limit=10)
            return [
                {
                    "item_id": result.item.id,
                    "content_preview": result.item.content[:120],
                    "memory_type": str(result.item.memory_type),
                    "why": arguments.reason,
                }
                for result in results
            ]
        if arguments.action == "by_ids":
            items = []
            for item_id in arguments.target.split():
                item = self.manager.get(item_id.strip())
                if item is not None:
                    items.append(
                        {
                            "item_id": item.id,
                            "content_preview": item.content[:120],
                            "memory_type": str(item.memory_type),
                            "why": arguments.reason,
                        }
                    )
            return items
        # by_relation：三元组文本（subject predicate object，空格分隔）。
        parts = arguments.target.split()
        if len(parts) < 3:
            raise ValueError("by_relation target must be 'subject predicate object'")
        subject, predicate = parts[0], parts[1]
        object_name = " ".join(parts[2:])
        facts = self.manager.semantic.facts(subject)
        return [
            {
                "item_id": item.id,
                "content_preview": item.content[:120],
                "memory_type": str(item.memory_type),
                "why": arguments.reason,
            }
            for item in facts
            if item.metadata.get("predicate") == predicate
            and str(item.metadata.get("object") or "") == object_name
        ]


def create_tool() -> BaseTool:
    return MemoryProposeDeleteTool()


__all__ = [
    "MemoryProposeDeleteInput",
    "MemoryProposeDeleteOutput",
    "MemoryProposeDeleteTool",
    "create_tool",
]
