"""获取运行 Agent 的当前系统本地时间。"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec


TOOL_ENABLED = True


class CurrentTimeInput(BaseModel):
    """该工具不需要调用方提供参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class CurrentTimeOutput(BaseModel):
    """当前系统时间的稳定、可序列化表示。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    local_time: str = Field(
        min_length=1,
        description="系统本地时间，ISO 8601 格式，并包含 UTC 偏移量。",
    )
    timezone_name: str = Field(
        min_length=1,
        description="系统本地时区名称；无法获取名称时为 UTC。",
    )
    unix_timestamp: int = Field(
        ge=0,
        description="当前时刻的 Unix 时间戳（自 1970-01-01 UTC 起的整数秒）。",
    )


class CurrentTimeTool(BaseTool):
    """读取当前系统本地时间，不执行写操作。"""

    spec = ToolSpec(
        name="system.current_time",
        description=(
            "Get the current system local date and time. Use this when the agent "
            "needs the actual current time; it accepts no arguments and performs "
            "no writes."
        ),
        version="1.0.0",
        input_model=CurrentTimeInput,
        output_model=CurrentTimeOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=5.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=None,
        tags=("system", "time", "clock", "datetime"),
    )

    def execute(self, arguments: CurrentTimeInput) -> CurrentTimeOutput:
        """读取并返回一次当前系统时间。"""
        del arguments
        local_now = datetime.now().astimezone()
        timezone_name = local_now.tzname() or "UTC"
        unix_timestamp = int(local_now.astimezone(timezone.utc).timestamp())
        return CurrentTimeOutput(
            local_time=local_now.isoformat(timespec="seconds"),
            timezone_name=timezone_name,
            unix_timestamp=unix_timestamp,
        )


def create_tool() -> BaseTool:
    """自动发现器调用的唯一零参数工厂。"""
    return CurrentTimeTool()
