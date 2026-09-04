"""Append-only project update log tool backed by SQLite.

The tool deliberately has no read operation: AI callers receive only the new
numeric ID, so historical logs do not consume model context.
"""

from __future__ import annotations

import platform
from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from core.update_log import UpdateLogRepository

TOOL_ENABLED = True


class UpdateLogFileChange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(min_length=1, max_length=500, description="Changed project-relative file path")
    action: str = Field(
        min_length=1,
        max_length=32,
        description="One of added, modified, deleted, renamed, or generated",
    )
    description: str = Field(
        min_length=1,
        max_length=2000,
        description="What changed in this file",
    )


class UpdateLogInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    executor: str = Field(
        min_length=1,
        max_length=200,
        description="AI/model or person that performed the change",
    )
    update_type: str = Field(
        min_length=1,
        max_length=64,
        description="Change category, such as feature, fix, refactor, config, or docs",
    )
    title: str = Field(min_length=1, max_length=300, description="Short update title")
    task_background: str = Field(
        min_length=1,
        max_length=4000,
        description="Why the change was requested and what it targets",
    )
    update_details: str = Field(
        min_length=1,
        max_length=12000,
        description="Concrete implementation details and important decisions",
    )
    added_features: str = Field(
        min_length=1,
        max_length=6000,
        description="New capabilities or 'none' when no feature was added",
    )
    files: list[UpdateLogFileChange] = Field(
        min_length=1,
        max_length=100,
        description="Every added, modified, deleted, or renamed project file",
    )
    behavior_impact: str = Field(
        min_length=1,
        max_length=6000,
        description="Compatibility, API, configuration, data, deployment, or user-impact notes",
    )
    validation: str = Field(
        min_length=1,
        max_length=6000,
        description="Tests/checks actually run and their real results",
    )
    risks: str = Field(
        min_length=1,
        max_length=4000,
        description="Known risks and rollback notes, or 'none'",
    )
    follow_up: str = Field(
        min_length=1,
        max_length=4000,
        description="Remaining work or 'none'",
    )


class UpdateLogOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    update_id: int = Field(ge=1, description="ID assigned to this update")
    next_update_id: int = Field(ge=1, description="ID to show in the guide for the next write")
    timestamp: str = Field(min_length=1, description="Actual UTC write timestamp")
    system_name: str = Field(min_length=1, description="Detected operating-system name")
    recorded: bool = True


class UpdateLogTool(BaseTool):
    """Write exactly one structured update and return only its compact ID."""

    spec = ToolSpec(
        name="system.update_log",
        description=(
            "MANDATORY after every project modification: append one complete, factual "
            "structured update record to the SQLite audit log. This tool writes only "
            "the record and returns its ID; it never reads historical entries."
        ),
        version="1.0",
        input_model=UpdateLogInput,
        output_model=UpdateLogOutput,
        side_effect="write",
        # The write confirmation below protects side effects; no permission
        # gate is applied to this public project tool.
        permissions=(),
        timeout_seconds=10.0,
        idempotent=False,
        parallel_safe=False,
        max_concurrency=1,
        tags=("update-log", "audit", "project", "mandatory"),
    )

    def __init__(self, repository: UpdateLogRepository | None = None) -> None:
        self.repository = repository or UpdateLogRepository()

    def execute(self, arguments: UpdateLogInput) -> UpdateLogOutput:
        if not isinstance(arguments, UpdateLogInput):
            raise TypeError("arguments must be an UpdateLogInput instance")
        result = self.repository.append(
            executor=arguments.executor,
            update_type=arguments.update_type,
            title=arguments.title,
            task_background=arguments.task_background,
            update_details=arguments.update_details,
            added_features=arguments.added_features,
            files=[item.model_dump() for item in arguments.files],
            behavior_impact=arguments.behavior_impact,
            validation=arguments.validation,
            risks=arguments.risks,
            follow_up=arguments.follow_up,
            system_name=platform.system(),
        )
        return UpdateLogOutput(**result)


def create_tool() -> BaseTool:
    return UpdateLogTool()


__all__ = [
    "TOOL_ENABLED",
    "UpdateLogFileChange",
    "UpdateLogInput",
    "UpdateLogOutput",
    "UpdateLogTool",
    "create_tool",
]
