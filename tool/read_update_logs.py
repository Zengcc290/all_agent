"""Read a contiguous update-log range with one model tool call.

The repository is still queried one immutable row at a time.  The loop lives
inside the tool so an audit of the complete history does not require one LLM
round per row.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core import BaseTool, ToolSpec
from core.update_log import UpdateLogRepository
from tool.read_update_log import ReadUpdateLogOutput

TOOL_ENABLED = True


class ReadUpdateLogsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_id: int = Field(
        default=1,
        ge=1,
        description="First positive update ID to read, inclusive",
    )
    end_id: int | None = Field(
        default=None,
        ge=1,
        description="Last update ID to read, inclusive; defaults to the current latest ID",
    )

    @model_validator(mode="after")
    def validate_range(self) -> ReadUpdateLogsInput:
        if self.end_id is not None and self.end_id < self.start_id:
            raise ValueError("end_id must be greater than or equal to start_id")
        return self


class ReadUpdateLogsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    records: list[ReadUpdateLogOutput] = Field(
        min_length=1,
        description="Update records read in ascending ID order",
    )
    start_id: int = Field(ge=1)
    end_id: int = Field(ge=1)
    latest_update_id: int = Field(
        ge=0,
        description="Current highest stored update ID",
    )


class ReadUpdateLogsTool(BaseTool):
    """Read each row in a requested range inside one tool execution."""

    spec = ToolSpec(
        name="system.read_update_logs",
        description=(
            "Read a contiguous range of project update-log records. The tool loops "
            "over SQLite one ID at a time internally, then returns the records in "
            "ascending order so a complete audit needs only one model tool call."
        ),
        version="1.0",
        input_model=ReadUpdateLogsInput,
        output_model=ReadUpdateLogsOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=10.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=4,
        tags=("update-log", "audit", "read", "batch", "project"),
    )

    def __init__(self, repository: UpdateLogRepository | None = None) -> None:
        self.repository = repository if repository is not None else UpdateLogRepository()

    def execute(self, arguments: ReadUpdateLogsInput) -> ReadUpdateLogsOutput:
        if not isinstance(arguments, ReadUpdateLogsInput):
            raise TypeError("arguments must be a ReadUpdateLogsInput instance")
        latest_update_id = self.repository.latest_id()
        if latest_update_id < arguments.start_id:
            raise LookupError(
                f"update log entry {arguments.start_id} was not found; "
                f"latest update ID is {latest_update_id}"
            )
        end_id = arguments.end_id if arguments.end_id is not None else latest_update_id
        if end_id > latest_update_id:
            raise LookupError(
                f"update log range ends at {end_id}, but latest update ID is {latest_update_id}"
            )
        if hasattr(self.repository, "get_range"):
            raw_records = self.repository.get_range(arguments.start_id, end_id)
        else:  # compatibility with lightweight repository doubles
            raw_records = [
                self.repository.get(update_id)
                for update_id in range(arguments.start_id, end_id + 1)
            ]
            raw_records = [record for record in raw_records if record is not None]
        expected_count = end_id - arguments.start_id + 1
        if len(raw_records) != expected_count:
            present = {record["update_id"] for record in raw_records}
            missing = next(
                update_id
                for update_id in range(arguments.start_id, end_id + 1)
                if update_id not in present
            )
            raise LookupError(f"update log entry {missing} was not found")
        records: list[ReadUpdateLogOutput] = []
        for record in raw_records:
            record["latest_update_id"] = latest_update_id
            records.append(ReadUpdateLogOutput(**record))
        return ReadUpdateLogsOutput(
            records=records,
            start_id=arguments.start_id,
            end_id=end_id,
            latest_update_id=latest_update_id,
        )


def create_tool() -> BaseTool:
    return ReadUpdateLogsTool()


__all__ = [
    "TOOL_ENABLED",
    "ReadUpdateLogsInput",
    "ReadUpdateLogsOutput",
    "ReadUpdateLogsTool",
    "create_tool",
]
