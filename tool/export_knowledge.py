"""知识库导出工具：把记忆库序列化成可移植的 JSON 载荷（只读）。

为什么这是一个独立能力
======================

导出要保证三件事，缺一不可：**可移植**（纯 JSON、无 Python 特有类型）、**可核对**
（带 format 版本号与 counts，导入方能先验后写）、**可界定**（哪些条目算语义、哪些算
情景，计数口径写死在载荷里，不靠导入方猜）。

它只读、不改库，因此可以被 Agent 随时调用（"把知识库导出来看看"），
而 HTTP 层的 ``/api/export`` 只负责加 ``Content-Disposition`` 文件名。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 里原先内联的导出构造已删除。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory.manager import MemoryManager

TOOL_ENABLED = True

#: 导出载荷的格式版本；导入方据此判断兼容性。
EXPORT_FORMAT = "knowledge-nebula-export/v1"


def export_payload(
    manager: MemoryManager,
    *,
    include_expired: bool = False,
    limit: int = 0,
    exported_at: str | None = None,
) -> dict[str, Any]:
    """Serialize the memory library into a portable, self-describing payload.

    ``limit=0`` means "everything"; the tool defaults to a bounded number so a
    tool result cannot blow up a model context, while the HTTP export keeps
    exporting the whole library.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    items = [
        item.to_dict() for item in manager.list(include_expired=include_expired)
    ]
    if limit:
        items = items[:limit]
    return {
        "format": EXPORT_FORMAT,
        "exported_at": exported_at or datetime.now(UTC).isoformat(),
        "counts": {
            "total": len(items),
            "semantic": sum(1 for item in items if item["memory_type"] == "semantic"),
            "episodic": sum(1 for item in items if item["memory_type"] == "episodic"),
        },
        "items": items,
    }


def export_filename(*, stamp: str | None = None) -> str:
    """Export file name with a local timestamp (user-facing, not a time semantic)."""

    stamp = stamp or datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return f"knowledge_export_{stamp}.json"


class ExportKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    include_expired: bool = Field(
        default=False, description="是否包含已过期条目（默认不包含，与导出文件一致）。"
    )
    limit: int = Field(
        default=100,
        ge=0,
        le=10_000,
        description="最多返回多少条；0 表示不限制（HTTP 导出用 0，工具默认 100 以防上下文爆炸）。",
    )


class ExportCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    total: int
    semantic: int = Field(description="其中语义层条目数。")
    episodic: int = Field(description="其中情景层条目数。")


class ExportKnowledgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    format: str
    exported_at: str
    counts: ExportCounts
    items: list[dict[str, Any]] = Field(default_factory=list)
    filename: str = Field(description="建议的导出文件名（HTTP 层用作下载名）。")


class ExportKnowledgeTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.export",
        description=(
            "Export the whole knowledge library as a portable, self-describing "
            "JSON payload (format version + counts + items). Read-only; use "
            "knowledge.import to load such a payload back."
        ),
        version="1.0.0",
        input_model=ExportKnowledgeInput,
        output_model=ExportKnowledgeOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=120.0,
        idempotent=True,
        parallel_safe=True,
        tags=("knowledge", "export", "backup", "read"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: ExportKnowledgeInput) -> ExportKnowledgeOutput:
        payload = export_payload(
            self.manager,
            include_expired=arguments.include_expired,
            limit=arguments.limit,
        )
        return ExportKnowledgeOutput(
            format=payload["format"],
            exported_at=payload["exported_at"],
            counts=ExportCounts(**payload["counts"]),
            items=payload["items"],
            filename=export_filename(),
        )


def create_tool() -> BaseTool:
    return ExportKnowledgeTool()


__all__ = [
    "EXPORT_FORMAT",
    "ExportCounts",
    "ExportKnowledgeInput",
    "ExportKnowledgeOutput",
    "ExportKnowledgeTool",
    "create_tool",
    "export_filename",
    "export_payload",
]
