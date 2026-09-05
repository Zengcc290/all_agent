"""Read one structured project update-log entry by its numeric ID.

The viewer intentionally performs a single-record lookup.  It never lists or
loads the complete append-only history into the caller's context.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from core.update_log import UpdateLogRepository
from tool.update_log import UpdateLogFileChange

TOOL_ENABLED = True


class ReadUpdateLogInput(BaseModel):
    """The one identifier accepted by the log viewer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    update_id: int = Field(
        ge=1,
        description="Positive update ID of the single log entry to retrieve",
    )


class ReadUpdateLogOutput(BaseModel):
    """A complete, decoded update-log row with stable field names."""

    model_config = ConfigDict(extra="forbid", strict=True)

    update_id: int = Field(ge=1, description="Stored update ID")
    timestamp: str = Field(min_length=1, max_length=64, description="UTC write timestamp")
    system_name: str = Field(
        min_length=1, max_length=200, description="Operating-system name recorded for the update"
    )
    executor: str = Field(min_length=1, max_length=200, description="AI/model or person that made the change")
    update_type: str = Field(min_length=1, max_length=64, description="Change category")
    title: str = Field(min_length=1, max_length=300, description="Short update title")
    task_background: str = Field(
        min_length=1, max_length=4000, description="Reason and target of the change"
    )
    update_details: str = Field(
        min_length=1, max_length=12000, description="Concrete implementation details and decisions"
    )
    added_features: str = Field(
        min_length=1, max_length=6000, description="New capabilities, or none"
    )
    files: list[UpdateLogFileChange] = Field(
        min_length=1, max_length=100, description="Files affected by the update"
    )
    behavior_impact: str = Field(
        min_length=1, max_length=6000, description="Compatibility and user-impact notes"
    )
    validation: str = Field(
        min_length=1, max_length=6000, description="Checks actually run and their results"
    )
    risks: str = Field(min_length=1, max_length=4000, description="Known risks and rollback notes")
    follow_up: str = Field(min_length=1, max_length=4000, description="Remaining work, or none")


class ReadUpdateLogTool(BaseTool):
    """Read exactly one immutable update-log entry by ID."""

    spec = ToolSpec(
        name="system.read_update_log",
        description=(
            "Retrieve one complete project update-log record by its positive update ID "
            "for targeted audits or troubleshooting. It is read-only and never lists "
            "or returns the historical log in bulk."
        ),
        version="1.0",
        input_model=ReadUpdateLogInput,
        output_model=ReadUpdateLogOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=5.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=8,
        tags=("update-log", "audit", "read", "project"),
    )

    def __init__(self, repository: UpdateLogRepository | None = None) -> None:
        self.repository = repository or UpdateLogRepository()

    def execute(self, arguments: ReadUpdateLogInput) -> ReadUpdateLogOutput:
        if not isinstance(arguments, ReadUpdateLogInput):
            raise TypeError("arguments must be a ReadUpdateLogInput instance")
        record = self.repository.get(arguments.update_id)
        if record is None:
            raise LookupError(f"update log entry {arguments.update_id} was not found")
        return ReadUpdateLogOutput(**record)


def create_tool() -> BaseTool:
    """Return the zero-argument instance used by automatic discovery."""

    return ReadUpdateLogTool()


__all__ = [
    "TOOL_ENABLED",
    "ReadUpdateLogInput",
    "ReadUpdateLogOutput",
    "ReadUpdateLogTool",
    "create_tool",
]
