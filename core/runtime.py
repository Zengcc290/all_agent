from __future__ import annotations

import asyncio
import inspect
from typing import Literal

from pydantic import ValidationError

from .models import BatchToolResult, ExecutionContext, ToolCall, ToolError, ToolResult
from .registry import ToolRegistry


class ToolExecutionManager:
    """Validate, schedule, execute and aggregate one batch of tool calls."""

    def __init__(self, registry: ToolRegistry, *, max_concurrency: int = 8) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.registry = registry
        self.max_concurrency = max_concurrency

    async def execute_batch(
        self,
        calls: list[ToolCall],
        context: ExecutionContext | None = None,
        *,
        failure_policy: Literal["continue", "fail_fast"] = "continue",
    ) -> BatchToolResult:
        context = context or ExecutionContext()
        if not calls:
            return BatchToolResult(results=[])

        positions: dict[str, int] = {}
        tools: dict[str, object] = {}
        errors: dict[str, ToolError] = {}
        for index, call in enumerate(calls):
            if call.call_id in positions:
                errors[call.call_id] = ToolError(code="DUPLICATE_CALL_ID", message="call_id must be unique")
            positions[call.call_id] = index
            tool = self.registry.maybe_get(call.tool_name)
            if tool is None:
                errors[call.call_id] = ToolError(code="UNKNOWN_TOOL", message=f"tool '{call.tool_name}' is not registered")
                continue
            tools[call.call_id] = tool
            spec = tool.spec
            if call.schema_version != spec.version or call.schema_hash != spec.schema_hash:
                errors[call.call_id] = ToolError(code="SCHEMA_MISMATCH", message="tool schema version or hash is stale")
                continue
            try:
                call.arguments = spec.input_model.model_validate(call.arguments, strict=True).model_dump()
            except ValidationError as exc:
                errors[call.call_id] = ToolError(code="INVALID_ARGUMENTS", message=str(exc))
                continue
            if not set(spec.permissions).issubset(context.permissions):
                errors[call.call_id] = ToolError(code="FORBIDDEN", message="required tool permission is missing")
                continue
            if spec.side_effect != "read" and spec.name not in context.confirmed_side_effects:
                errors[call.call_id] = ToolError(code="CONFIRMATION_REQUIRED", message="side-effecting tool requires confirmation")

        for call in calls:
            for dependency in call.depends_on:
                if dependency not in positions:
                    errors[call.call_id] = ToolError(code="UNKNOWN_DEPENDENCY", message=f"unknown dependency '{dependency}'")
            if call.call_id in call.depends_on:
                errors[call.call_id] = ToolError(code="DEPENDENCY_CYCLE", message="a call cannot depend on itself")

        results: dict[str, ToolResult] = {
            call.call_id: ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=False, error=error)
            for call in calls
            if (error := errors.get(call.call_id)) is not None
        }
        remaining = {call.call_id for call in calls if call.call_id not in errors}
        completed: set[str] = set(results)
        global_sem = asyncio.Semaphore(self.max_concurrency)
        tool_sems: dict[str, asyncio.Semaphore] = {}

        while remaining:
            ready = [
                call for call in calls
                if call.call_id in remaining and all(dep in completed for dep in call.depends_on)
            ]
            if not ready:
                for call_id in remaining:
                    results[call_id] = ToolResult(
                        call_id=call_id,
                        tool_name=next(call.tool_name for call in calls if call.call_id == call_id),
                        ok=False,
                        error=ToolError(code="DEPENDENCY_CYCLE", message="tool call dependency graph contains a cycle"),
                    )
                completed.update(remaining)
                remaining.clear()
                break

            # Unsafe tools are isolated into one-call layers; independent read tools share a layer.
            unsafe_ready = [call for call in ready if not tools[call.call_id].spec.parallel_safe]
            batch = unsafe_ready[:1] if unsafe_ready else ready

            async def run_one(call: ToolCall) -> ToolResult:
                tool = tools[call.call_id]
                spec = tool.spec
                if any(not results[dep].ok for dep in call.depends_on if dep in results):
                    return ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        ok=False,
                        error=ToolError(code="DEPENDENCY_FAILED", message="a dependency failed"),
                    )
                sem = tool_sems.setdefault(spec.name, asyncio.Semaphore(spec.max_concurrency or self.max_concurrency))
                try:
                    async with global_sem, sem:
                        arguments = spec.input_model.model_validate(call.arguments, strict=True)
                        if inspect.iscoroutinefunction(tool.execute):
                            value = await asyncio.wait_for(tool.execute(arguments), timeout=spec.timeout_seconds)
                        else:
                            value = await asyncio.wait_for(asyncio.to_thread(tool.execute, arguments), timeout=spec.timeout_seconds)
                        if inspect.isawaitable(value):
                            value = await asyncio.wait_for(value, timeout=spec.timeout_seconds)
                        data = spec.output_model.model_validate(value, strict=True).model_dump()
                        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=True, data=data)
                except asyncio.TimeoutError:
                    return ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=False, error=ToolError(code="TIMEOUT", message="tool execution timed out", retryable=True))
                except ValidationError as exc:
                    return ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=False, error=ToolError(code="INVALID_OUTPUT", message=str(exc)))
                except Exception as exc:
                    return ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=False, error=ToolError(code="EXECUTION_ERROR", message=str(exc), retryable=spec.idempotent))

            batch_results = await asyncio.gather(*(run_one(call) for call in batch))
            for result in batch_results:
                results[result.call_id] = result
                remaining.remove(result.call_id)
                completed.add(result.call_id)
            if failure_policy == "fail_fast" and any(not result.ok for result in batch_results):
                for call in calls:
                    if call.call_id in remaining:
                        results[call.call_id] = ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=False, error=ToolError(code="ABORTED", message="batch aborted after a tool failure"))
                remaining.clear()

        return BatchToolResult(results=[results[call.call_id] for call in calls])
