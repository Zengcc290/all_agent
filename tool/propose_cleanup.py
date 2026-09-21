"""孤儿实体清理提案工具：为完全孤立的实体生成**待确认**的删除提案（写工具）。

为什么这是一个独立能力
======================

它把"发现垃圾"变成"可审核的提案"：只写一条待确认提案，绝不删数据；调用方拿到
``proposal_id`` 后，由用户走确认流程调用 ``execute_deletion`` 才真正删除。
这是维护能力（Agent 可以主动体检并提交提案），但**不是**自动化删除能力。

安全决策：工具刻意**不返回** ``confirm_token``
=============================================

底层函数 ``propose_orphan_cleanup`` 会把确认令牌一起返回，那是给确认闸门用的。
工具是给 LLM 调用的，把令牌写进模型上下文等于把"人工确认"降级成"模型自己就能确认"，
因此工具输出只给 ``proposal_id`` 与计数；令牌仍留在函数返回值里（人类/脚本通道），
两条通道的权限边界因此不同。

本模块是这段逻辑的**唯一实现**：``web/cleanup.py`` 里的 ``propose_orphan_cleanup`` 已删除。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryManager
from memory.storage.document_repo import DeletionProposalStore

from .orphan_entities import find_orphan_entities

TOOL_ENABLED = True


def propose_orphan_cleanup(
    manager: MemoryManager,
    *,
    requested_by: str = "maintenance",
    reason: str = "完全孤立实体：无关系边、无原句提及、无备注挂靠",
) -> dict[str, object]:
    """为所有完全孤立实体创建一条待确认的删除提案。

    只写提案、不删数据；调用方拿到 proposal_id + confirm_token 后，由用户
    （或显式确认流程）调用 ``execute_deletion`` 才会真正删除。
    """

    orphans = find_orphan_entities(manager)
    item_ids = [item.id for item in orphans]
    if not item_ids:
        return {
            "proposal_id": "",
            "confirm_token": "",
            "count": 0,
            "item_ids": [],
            "note": "没有发现完全孤立的实体",
        }
    connection = getattr(manager.document_store, "connection", None)
    store = DeletionProposalStore(manager.document_store.path, connection=connection)
    proposal = store.create(requested_by=requested_by, reason=reason, item_ids=item_ids)
    return {
        "proposal_id": proposal.proposal_id,
        "confirm_token": proposal.confirm_token,
        "count": len(item_ids),
        "item_ids": item_ids,
        "note": f"已生成删除提案 {proposal.proposal_id[:8]}，等待确认",
    }


class ProposeCleanupInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requested_by: str = Field(
        default="maintenance", min_length=1, max_length=120, description="提案发起人标识（审计用）。"
    )
    reason: str = Field(
        default="完全孤立实体：无关系边、无原句提及、无备注挂靠",
        min_length=1,
        max_length=500,
        description="提案原因（会写进待确认提案，供审核者判断）。",
    )
    max_ids: int = Field(
        default=100, ge=1, le=5000, description="最多回报多少个待删 id（count 始终是完整数量）。"
    )


class ProposeCleanupOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    proposal_id: str = Field(description="待确认提案 id；空串表示没有候选，未创建提案。")
    count: int = Field(description="提案包含的待删实体数。")
    item_ids: list[str] = Field(default_factory=list, description="待删 id 的前 max_ids 个。")
    note: str = Field(description="人类可读的结果说明。")
    requires_human_confirmation: bool = Field(
        description="恒为 true：真正的删除必须由确认流程完成，工具无权删除。"
    )


class ProposeCleanupTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.propose_cleanup",
        description=(
            "Open a PENDING deletion proposal for every completely isolated entity. "
            "Writes only a proposal — never deletes data; the actual deletion needs "
            "an explicit human confirmation step. Returns the proposal id and count, "
            "but deliberately NOT the confirm token."
        ),
        version="1.0.0",
        input_model=ProposeCleanupInput,
        output_model=ProposeCleanupOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=120.0,
        idempotent=False,
        parallel_safe=False,
        tags=("knowledge", "graph", "cleanup", "proposal", "write"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: ProposeCleanupInput) -> ProposeCleanupOutput:
        result = propose_orphan_cleanup(
            self.manager,
            requested_by=arguments.requested_by,
            reason=arguments.reason,
        )
        return ProposeCleanupOutput(
            proposal_id=str(result["proposal_id"]),
            count=int(result["count"]),
            item_ids=[str(value) for value in result["item_ids"]][: arguments.max_ids],
            note=str(result["note"]),
            requires_human_confirmation=True,
        )


def create_tool() -> BaseTool:
    return ProposeCleanupTool()


__all__ = [
    "ProposeCleanupInput",
    "ProposeCleanupOutput",
    "ProposeCleanupTool",
    "create_tool",
    "propose_orphan_cleanup",
]
