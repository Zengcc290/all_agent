import asyncio

import pytest
from pydantic import BaseModel, ConfigDict, Field

from core import ExecutionContext, ToolCall, ToolExecutionManager, ToolRegistry, ToolSpec
from core.registry import BaseTool


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int = Field(ge=0)


class EchoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class EchoTool(BaseTool):
    spec = ToolSpec(
        name="test.echo",
        description="echo",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
    )

    def execute(self, arguments: EchoInput) -> EchoOutput:
        return EchoOutput(value=arguments.value)


class DelayTool(EchoTool):
    spec = ToolSpec(
        name="test.delay",
        description="delay",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
    )

    async def execute(self, arguments: EchoInput) -> EchoOutput:
        await asyncio.sleep(arguments.value / 1000)
        return EchoOutput(value=arguments.value)


class TimeoutTool(DelayTool):
    spec = ToolSpec(
        name="test.timeout",
        description="timeout",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
        timeout_seconds=0.01,
    )


def call_for(tool, call_id, value, depends_on=None):
    return ToolCall(
        call_id=call_id,
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        arguments={"value": value},
        depends_on=depends_on or [],
    )


@pytest.fixture
def registry():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(DelayTool())
    registry.register(TimeoutTool())
    return registry


@pytest.mark.asyncio
async def test_independent_calls_run_in_parallel(registry):
    manager = ToolExecutionManager(registry, max_concurrency=2)
    delay = registry.get("test.delay")
    calls = [call_for(delay, "slow", 80), call_for(delay, "fast", 80)]
    result = await manager.execute_batch(calls)
    assert [item.ok for item in result.results] == [True, True]
    assert [item.data["value"] for item in result.results] == [80, 80]


@pytest.mark.asyncio
async def test_dependencies_are_executed_in_layers(registry):
    manager = ToolExecutionManager(registry)
    echo = registry.get("test.echo")
    first = call_for(echo, "first", 1)
    second = call_for(echo, "second", 2, depends_on=["first"])
    result = await manager.execute_batch([second, first])
    assert all(item.ok for item in result.results)
    assert [item.call_id for item in result.results] == ["second", "first"]


@pytest.mark.asyncio
async def test_invalid_arguments_never_execute(registry):
    manager = ToolExecutionManager(registry)
    echo = registry.get("test.echo")
    call = call_for(echo, "bad", 1)
    call.arguments["unexpected"] = True
    result = await manager.execute_batch([call])
    assert result.results[0].error.code == "INVALID_ARGUMENTS"


@pytest.mark.asyncio
async def test_schema_mismatch_is_rejected(registry):
    manager = ToolExecutionManager(registry)
    echo = registry.get("test.echo")
    call = call_for(echo, "stale", 1)
    call.schema_hash = "stale"
    result = await manager.execute_batch([call])
    assert result.results[0].error.code == "SCHEMA_MISMATCH"


@pytest.mark.asyncio
async def test_failed_dependency_skips_dependent_call(registry):
    manager = ToolExecutionManager(registry)
    echo = registry.get("test.echo")
    bad = call_for(echo, "bad", 1)
    bad.arguments["unexpected"] = True
    dependent = call_for(echo, "dependent", 2, depends_on=["bad"])
    result = await manager.execute_batch([bad, dependent])
    assert result.results[0].error.code == "INVALID_ARGUMENTS"
    assert result.results[1].error.code == "DEPENDENCY_FAILED"


@pytest.mark.asyncio
async def test_timeout_is_returned_per_call(registry):
    manager = ToolExecutionManager(registry)
    timeout = registry.get("test.timeout")
    result = await manager.execute_batch([call_for(timeout, "late", 100)])
    assert result.results[0].error.code == "TIMEOUT"
