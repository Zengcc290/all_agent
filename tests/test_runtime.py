import asyncio
import threading
import time

import pytest
from pydantic import BaseModel, ConfigDict, Field

from core import (
    ExecutionContext,
    ToolCall,
    ToolExecutionManager,
    ToolRegistry,
    ToolSpec,
)
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


class PermissionedEchoTool(EchoTool):
    spec = ToolSpec(
        name="test.permissioned_echo",
        description="echo with informational permission metadata",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
        permissions=("test.restricted",),
    )


def test_successful_registration_logs_all_active_tool_names(capsys):
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(PermissionedEchoTool())

    output = capsys.readouterr().out

    assert "Tool registered: test.echo" in output
    assert "Tool registered: test.permissioned_echo" in output
    assert "Current registered tools: test.echo, test.permissioned_echo" in output


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


class BlockingTimeoutTool(EchoTool):
    spec = ToolSpec(
        name="test.blocking_timeout",
        description="blocking timeout",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
        timeout_seconds=0.01,
        max_concurrency=1,
    )
    completed = threading.Event()

    def execute(self, arguments: EchoInput) -> EchoOutput:
        time.sleep(0.05)
        type(self).completed.set()
        return EchoOutput(value=arguments.value)


class NonIdempotentBlockingTimeoutTool(BlockingTimeoutTool):
    completed = threading.Event()
    spec = ToolSpec(
        name="test.non_idempotent_timeout",
        description="non-idempotent timeout",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
        side_effect="write",
        idempotent=False,
        timeout_seconds=0.01,
        max_concurrency=1,
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
    registry.register(PermissionedEchoTool())
    registry.register(DelayTool())
    registry.register(TimeoutTool())
    registry.register(BlockingTimeoutTool())
    registry.register(NonIdempotentBlockingTimeoutTool())
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
async def test_permissioned_tool_executes_without_context_permissions(registry):
    manager = ToolExecutionManager(registry)
    tool = registry.get("test.permissioned_echo")

    result = await manager.execute_batch([call_for(tool, "unrestricted", 1)])

    assert result.results[0].ok
    assert result.results[0].data == {"value": 1}


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


@pytest.mark.asyncio
async def test_duplicate_call_ids_are_rejected_without_result_overwrite(registry):
    manager = ToolExecutionManager(registry)
    echo = registry.get("test.echo")
    calls = [call_for(echo, "same", 1), call_for(echo, "same", 2)]

    result = await manager.execute_batch(calls)

    assert [item.error.code for item in result.results] == [
        "DUPLICATE_CALL_ID",
        "DUPLICATE_CALL_ID",
    ]
    assert [item.tool_name for item in result.results] == ["test.echo", "test.echo"]


@pytest.mark.asyncio
async def test_fail_fast_aborts_after_pre_execution_validation_failure(registry):
    manager = ToolExecutionManager(registry)
    echo = registry.get("test.echo")
    invalid = call_for(echo, "invalid", 1)
    invalid.arguments = {"value": "not-an-integer"}
    pending = call_for(echo, "pending", 2)

    result = await manager.execute_batch([invalid, pending], failure_policy="fail_fast")

    assert result.results[0].error.code == "INVALID_ARGUMENTS"
    assert result.results[1].error.code == "ABORTED"


@pytest.mark.asyncio
async def test_invalid_failure_policy_is_rejected(registry):
    manager = ToolExecutionManager(registry)
    echo = registry.get("test.echo")

    with pytest.raises(ValueError, match="failure_policy"):
        await manager.execute_batch(
            [call_for(echo, "one", 1)], failure_policy="invalid"
        )


@pytest.mark.asyncio
async def test_timed_out_sync_work_keeps_its_concurrency_slot_until_it_finishes(
    registry,
):
    BlockingTimeoutTool.completed.clear()
    manager = ToolExecutionManager(registry, max_concurrency=2)
    blocking = registry.get("test.blocking_timeout")
    first = await manager.execute_batch([call_for(blocking, "first", 1)])
    second = await manager.execute_batch([call_for(blocking, "second", 2)])

    assert first.results[0].error.code == "TIMEOUT"
    assert second.results[0].error.code == "TIMEOUT"
    assert await asyncio.to_thread(BlockingTimeoutTool.completed.wait, 0.5)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_manager_concurrency_limit_is_shared_by_concurrent_batches(registry):
    manager = ToolExecutionManager(registry, max_concurrency=1)
    delay = registry.get("test.delay")

    started = time.monotonic()
    first, second = await asyncio.gather(
        manager.execute_batch([call_for(delay, "one", 40)]),
        manager.execute_batch([call_for(delay, "two", 40)]),
    )

    assert all(item.ok for item in [first.results[0], second.results[0]])
    assert time.monotonic() - started >= 0.07


def test_manager_can_be_reused_across_event_loops(registry):
    manager = ToolExecutionManager(registry, max_concurrency=1)
    delay = registry.get("test.delay")

    async def run_pair(prefix):
        result = await manager.execute_batch(
            [
                call_for(delay, f"{prefix}-one", 10),
                call_for(delay, f"{prefix}-two", 10),
            ]
        )
        assert all(item.ok for item in result.results)

    asyncio.run(run_pair("first"))
    asyncio.run(run_pair("second"))


@pytest.mark.asyncio
async def test_non_idempotent_timeout_is_not_retryable(registry):
    manager = ToolExecutionManager(registry)
    tool = registry.get("test.non_idempotent_timeout")
    tool.completed.clear()
    context = ExecutionContext(
        confirmed_side_effects=frozenset({registry.confirmation_key(tool.spec.name)})
    )

    result = await manager.execute_batch([call_for(tool, "write-timeout", 1)], context)

    assert result.results[0].error.code == "TIMEOUT"
    assert result.results[0].error.retryable is False
    assert await asyncio.to_thread(tool.completed.wait, 0.5)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_registration_generation_rejects_replaced_implementation(registry):
    manager = ToolExecutionManager(registry)
    tool, generation = registry.resolve("test.echo")
    call = call_for(tool, "stale-generation", 1)
    call.registry_generation = generation
    registry.register(EchoTool(), replace=True)

    result = await manager.execute_batch([call])

    assert result.results[0].error.code == "SCHEMA_MISMATCH"


@pytest.mark.asyncio
async def test_side_effect_confirmation_is_bound_to_registered_generation(registry):
    class SideEffectTool(EchoTool):
        spec = ToolSpec(
            name="test.side_effect",
            description="side effect",
            version="1.0",
            input_model=EchoInput,
            output_model=EchoOutput,
            side_effect="write",
        )

    manager = ToolExecutionManager(registry)
    side_effect = SideEffectTool()
    registry.register(side_effect)
    call = call_for(side_effect, "write", 1)
    old_key = side_effect.spec.confirmation_key

    denied = await manager.execute_batch(
        [call], ExecutionContext(confirmed_side_effects=frozenset({old_key}))
    )
    assert denied.results[0].error.code == "CONFIRMATION_REQUIRED"

    current_key = registry.confirmation_key(side_effect.spec.name)
    allowed = await manager.execute_batch(
        [call], ExecutionContext(confirmed_side_effects=frozenset({current_key}))
    )
    assert allowed.results[0].ok


def test_tool_spec_rejects_invalid_metadata():
    with pytest.raises(ValueError):
        ToolSpec(
            name=".",
            description="bad",
            version="1",
            input_model=EchoInput,
            output_model=EchoOutput,
        )
    with pytest.raises(ValueError):
        ToolSpec(
            name="test.bad",
            description="bad",
            version="",
            input_model=EchoInput,
            output_model=EchoOutput,
        )
    with pytest.raises(ValueError):
        ToolSpec(
            name="test.bad",
            description="bad",
            version="1",
            input_model=EchoInput,
            output_model=EchoOutput,
            timeout_seconds=True,
        )

    class PermissiveInput(BaseModel):
        value: int

    with pytest.raises(ValueError, match="extra='forbid'"):
        ToolSpec(
            name="test.permissive",
            description="permissive",
            version="1",
            input_model=PermissiveInput,
            output_model=EchoOutput,
        )
