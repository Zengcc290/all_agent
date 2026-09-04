from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from .models import BatchToolResult, ExecutionContext, ToolCall, ToolError, ToolResult
from .registry import BaseTool, ToolRegistry

LOGGER = logging.getLogger(__name__)


class _ExecutionTimedOut(Exception):
    """Internal marker for a runtime-enforced execution deadline."""


@dataclass
class _LoopExecutionState:
    """Async primitives owned by exactly one event loop."""

    global_semaphore: asyncio.Semaphore
    tool_semaphores: dict[tuple[object, ...], asyncio.Semaphore] = field(
        default_factory=dict
    )


class ToolExecutionManager:
    """Validate, schedule, execute and aggregate one batch of tool calls."""

    def __init__(self, registry: ToolRegistry, *, max_concurrency: int = 8) -> None:
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")
        self.registry = registry
        self.max_concurrency = max_concurrency
        # Asyncio synchronization primitives are bound to the event loop on
        # which they contend. Keep shared batch limits per loop so a manager can
        # also be reused safely by sequential ``asyncio.run`` calls.
        self._loop_states: dict[asyncio.AbstractEventLoop, _LoopExecutionState] = {}
        self._loop_states_lock = threading.Lock()

    async def execute_batch(
        self,
        calls: list[ToolCall],
        context: ExecutionContext | None = None,
        *,
        failure_policy: Literal["continue", "fail_fast"] = "continue",
    ) -> BatchToolResult:
        if failure_policy not in {"continue", "fail_fast"}:
            raise ValueError("failure_policy must be 'continue' or 'fail_fast'")
        if not isinstance(calls, (list, tuple)):
            raise TypeError("calls must be a list or tuple of ToolCall instances")
        if not all(isinstance(call, ToolCall) for call in calls):
            raise TypeError("calls must contain ToolCall instances")
        if context is None:
            context = ExecutionContext()
        elif not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext instance")
        if not calls:
            return BatchToolResult(results=[])
        loop_state = self._loop_state()
        indices_by_call_id: dict[str, list[int]] = {}
        for index, call in enumerate(calls):
            indices_by_call_id.setdefault(call.call_id, []).append(index)
        duplicate_indices = {
            index
            for indices in indices_by_call_id.values()
            if len(indices) > 1
            for index in indices
        }

        tools: list[BaseTool | None] = [None] * len(calls)
        normalized_arguments: list[BaseModel | None] = [None] * len(calls)
        errors: list[ToolError | None] = [None] * len(calls)

        for index, call in enumerate(calls):
            if index in duplicate_indices:
                errors[index] = ToolError(
                    code="DUPLICATE_CALL_ID",
                    message="call_id must be unique within a batch",
                )
                continue

            registration = self.registry.maybe_resolve(call.tool_name)
            if registration is None:
                errors[index] = ToolError(
                    code="UNKNOWN_TOOL",
                    message=f"tool '{call.tool_name}' is not registered",
                )
                continue
            tool, generation = registration
            tools[index] = tool
            spec = tool.spec
            if (
                call.schema_version != spec.version
                or call.schema_hash != spec.schema_hash
                or (
                    call.registry_generation is not None
                    and call.registry_generation != generation
                )
            ):
                errors[index] = ToolError(
                    code="SCHEMA_MISMATCH",
                    message="tool registration, schema version or hash is stale",
                )
                continue
            # Tool permissions are currently informational metadata.  Runtime
            # execution is deliberately unrestricted so every caller can use
            # every registered tool; keep the metadata for cataloging and a
            # possible future authorization policy.
            confirmation_key = self.registry.confirmation_key(spec.name)
            if (
                spec.side_effect != "read"
                and confirmation_key not in context.confirmed_side_effects
            ):
                errors[index] = ToolError(
                    code="CONFIRMATION_REQUIRED",
                    message="side-effecting tool requires confirmation",
                )
                continue
            try:
                normalized_arguments[index] = spec.input_model.model_validate(
                    call.arguments, strict=True
                )
            except ValidationError as exc:
                errors[index] = ToolError(
                    code="INVALID_ARGUMENTS", message=self._limited_message(exc)
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Unexpected argument validation failure for tool %s (%s)",
                    spec.name,
                    type(exc).__name__,
                )
                errors[index] = ToolError(
                    code="INVALID_ARGUMENTS", message="argument validation failed"
                )

        dependencies: list[list[int]] = [[] for _ in calls]
        for index, call in enumerate(calls):
            for dependency_id in call.depends_on:
                matching_indices = indices_by_call_id.get(dependency_id, [])
                if not matching_indices:
                    if errors[index] is None:
                        errors[index] = ToolError(
                            code="UNKNOWN_DEPENDENCY",
                            message=f"unknown dependency '{dependency_id}'",
                        )
                    continue
                if len(matching_indices) != 1:
                    if errors[index] is None:
                        errors[index] = ToolError(
                            code="AMBIGUOUS_DEPENDENCY",
                            message=f"dependency '{dependency_id}' is not unique within the batch",
                        )
                    continue
                dependency_index = matching_indices[0]
                if dependency_index == index:
                    if errors[index] is None:
                        errors[index] = ToolError(
                            code="DEPENDENCY_CYCLE",
                            message="a call cannot depend on itself",
                        )
                    continue
                dependencies[index].append(dependency_index)

        results: list[ToolResult | None] = [None] * len(calls)
        for index, error in enumerate(errors):
            if error is not None:
                results[index] = self._error_result(calls[index], error)

        remaining = {index for index, result in enumerate(results) if result is None}
        completed = {
            index for index, result in enumerate(results) if result is not None
        }
        if failure_policy == "fail_fast" and completed:
            self._abort_remaining(calls, results, remaining)

        while remaining:
            ready = [
                index
                for index in remaining
                if all(
                    dependency_index in completed
                    for dependency_index in dependencies[index]
                )
            ]
            if not ready:
                for index in remaining:
                    results[index] = self._error_result(
                        calls[index],
                        ToolError(
                            code="DEPENDENCY_CYCLE",
                            message="tool call dependency graph contains a cycle",
                        ),
                    )
                remaining.clear()
                break

            unsafe_ready = [
                index for index in ready if not tools[index].spec.parallel_safe
            ]
            batch = unsafe_ready[:1] if unsafe_ready else ready
            batch_results = await asyncio.gather(
                *(
                    self._run_one(
                        index,
                        calls[index],
                        tools[index],
                        normalized_arguments[index],
                        dependencies[index],
                        results,
                        context,
                        loop_state,
                    )
                    for index in batch
                )
            )
            for index, result in batch_results:
                results[index] = result
                remaining.remove(index)
                completed.add(index)

            if failure_policy == "fail_fast" and any(
                not result.ok for _, result in batch_results
            ):
                self._abort_remaining(calls, results, remaining)

        if any(result is None for result in results):
            raise RuntimeError("tool scheduler finished with incomplete results")
        return BatchToolResult(
            results=[result for result in results if result is not None]
        )

    def _abort_remaining(
        self,
        calls: list[ToolCall],
        results: list[ToolResult | None],
        remaining: set[int],
    ) -> None:
        for index in remaining:
            results[index] = self._error_result(
                calls[index],
                ToolError(code="ABORTED", message="batch aborted after a tool failure"),
            )
        remaining.clear()

    @staticmethod
    def _error_result(call: ToolCall, error: ToolError) -> ToolResult:
        return ToolResult(
            call_id=call.call_id, tool_name=call.tool_name, ok=False, error=error
        )

    def _loop_state(self) -> _LoopExecutionState:
        loop = asyncio.get_running_loop()
        with self._loop_states_lock:
            closed_loops = [
                known_loop
                for known_loop in self._loop_states
                if known_loop is not loop and known_loop.is_closed()
            ]
            for closed_loop in closed_loops:
                del self._loop_states[closed_loop]
            state = self._loop_states.get(loop)
            if state is None:
                state = _LoopExecutionState(
                    global_semaphore=asyncio.Semaphore(self.max_concurrency)
                )
                self._loop_states[loop] = state
            return state

    def _tool_semaphore(
        self, loop_state: _LoopExecutionState, tool: BaseTool
    ) -> asyncio.Semaphore:
        spec = tool.spec
        capacity = (
            1
            if not spec.parallel_safe
            else (spec.max_concurrency or self.max_concurrency)
        )
        key = (spec.name, spec.version, spec.schema_hash, spec.parallel_safe, capacity)
        semaphore = loop_state.tool_semaphores.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(capacity)
            loop_state.tool_semaphores[key] = semaphore
        return semaphore

    async def _run_one(
        self,
        index: int,
        call: ToolCall,
        tool: BaseTool | None,
        arguments: BaseModel | None,
        dependency_indices: list[int],
        results: list[ToolResult | None],
        context: ExecutionContext,
        loop_state: _LoopExecutionState,
    ) -> tuple[int, ToolResult]:
        if tool is None or arguments is None:
            raise RuntimeError("scheduler received an unvalidated tool call")
        if any(
            results[dependency_index] is not None and not results[dependency_index].ok
            for dependency_index in dependency_indices
        ):
            return index, self._error_result(
                call,
                ToolError(code="DEPENDENCY_FAILED", message="a dependency failed"),
            )

        spec = tool.spec
        tool_semaphore = self._tool_semaphore(loop_state, tool)
        global_acquired = False
        tool_acquired = False
        task: asyncio.Task[Any] | None = None
        defer_release = False
        try:
            # Waiting for a tool-specific slot does not consume a global slot.
            await tool_semaphore.acquire()
            tool_acquired = True
            await loop_state.global_semaphore.acquire()
            global_acquired = True

            loop = asyncio.get_running_loop()
            deadline = loop.time() + spec.timeout_seconds
            validated_arguments = arguments
            execute_arguments = [validated_arguments]
            execute_keywords: dict[str, Any] = {}
            context_parameter = inspect.signature(tool.execute).parameters.get(
                "context"
            )
            if context_parameter is not None:
                if context_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                    execute_keywords["context"] = context
                elif context_parameter.kind in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }:
                    execute_arguments.append(context)
            if inspect.iscoroutinefunction(tool.execute):
                task = asyncio.create_task(
                    tool.execute(*execute_arguments, **execute_keywords)
                )
                value = await self._await_until(task, deadline, cancel_on_timeout=True)
            else:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        tool.execute, *execute_arguments, **execute_keywords
                    )
                )
                value = await self._await_until(task, deadline, cancel_on_timeout=False)

            if inspect.isawaitable(value):
                task = asyncio.ensure_future(value)
                value = await self._await_until(task, deadline, cancel_on_timeout=True)
            data = spec.output_model.model_validate(value, strict=True).model_dump()
            return index, ToolResult(
                call_id=call.call_id, tool_name=call.tool_name, ok=True, data=data
            )
        except _ExecutionTimedOut:
            return index, self._error_result(
                call,
                ToolError(
                    code="TIMEOUT",
                    message="tool execution timed out",
                    retryable=spec.idempotent,
                ),
            )
        except ValidationError as exc:
            return index, self._error_result(
                call,
                ToolError(code="INVALID_OUTPUT", message=self._limited_message(exc)),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Tool execution failed for %s (%s)", spec.name, type(exc).__name__
            )
            return index, self._error_result(
                call,
                ToolError(
                    code="EXECUTION_ERROR",
                    message="tool execution failed",
                    retryable=spec.idempotent,
                ),
            )
        finally:
            if (
                task is not None
                and not task.done()
                and global_acquired
                and tool_acquired
            ):
                # asyncio cannot stop an already-running thread. Keep both slots
                # reserved until it exits so timeout never creates hidden overlap.
                defer_release = True
                task.add_done_callback(
                    self._release_after_background_work(loop_state, tool_semaphore)
                )
            elif task is not None and task.done() and not task.cancelled():
                task.exception()
            if not defer_release:
                if global_acquired:
                    loop_state.global_semaphore.release()
                if tool_acquired:
                    tool_semaphore.release()

    async def _await_until(
        self,
        task: asyncio.Task[Any],
        deadline: float,
        *,
        cancel_on_timeout: bool,
    ) -> Any:
        timeout = max(0.0, deadline - asyncio.get_running_loop().time())
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            if cancel_on_timeout:
                task.cancel()
            raise _ExecutionTimedOut
        return task.result()

    def _release_after_background_work(
        self,
        loop_state: _LoopExecutionState,
        tool_semaphore: asyncio.Semaphore,
    ):
        def release(task: asyncio.Task[Any]) -> None:
            # Observe late failures so asyncio does not report an unhandled task.
            if not task.cancelled():
                task.exception()
            loop_state.global_semaphore.release()
            tool_semaphore.release()

        return release

    @staticmethod
    def _limited_message(error: Exception, limit: int = 2000) -> str:
        if isinstance(error, ValidationError):
            # Pydantic's default string includes ``input_value``. Do not echo
            # untrusted arguments or secrets back to a model or caller.
            details = []
            for item in error.errors(include_url=False, include_context=False):
                location = ".".join(str(part) for part in item.get("loc", ()))
                message = item.get("msg", "validation failed")
                details.append(f"{location}: {message}" if location else message)
            message = "; ".join(details)
        else:
            message = str(error)
        message = message or type(error).__name__
        return message[:limit]
