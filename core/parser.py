from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .models import ToolCall
from .registry import BaseTool, ToolRegistry


def parse_tool_calls(
    payload: str | dict[str, Any] | list[dict[str, Any]],
) -> list[ToolCall]:
    """Parse native or fallback JSON calls without executing anything."""
    if isinstance(payload, str):
        payload = _load_json(payload)
    if isinstance(payload, dict):
        payload = payload.get("tool_calls", payload)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise TypeError("tool call payload must be an object or list")
    return [ToolCall.model_validate(item, strict=True) for item in payload]


def parse_openai_tool_calls(
    tool_calls: list[Any],
    registry: ToolRegistry,
    name_map: dict[str, str] | None = None,
    registrations: Mapping[str, tuple[BaseTool, int]] | None = None,
) -> list[ToolCall]:
    """Convert OpenAI-compatible native calls into the internal versioned envelope."""
    if not isinstance(tool_calls, (list, tuple)):
        raise TypeError("native tool calls must be a list")
    parsed: list[ToolCall] = []
    for item in tool_calls:
        call_id = _get(item, "id")
        function = _get(item, "function")
        name = _get(function, "name")
        raw_arguments = _get(function, "arguments", "{}")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("native tool call is missing a valid id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("native tool call is missing id or function name")
        canonical_name = (name_map or {}).get(name, name)
        try:
            if registrations is None:
                tool, generation = registry.resolve(canonical_name)
            else:
                tool, generation = registrations[canonical_name]
        except KeyError as exc:
            raise ValueError(f"unknown tool '{name}'") from exc
        if isinstance(raw_arguments, str):
            try:
                arguments = _load_json(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON arguments for tool '{name}'") from exc
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            raise TypeError(f"arguments for tool '{name}' must be a JSON object")
        parsed.append(
            ToolCall(
                call_id=call_id,
                tool_name=canonical_name,
                schema_version=tool.spec.version,
                schema_hash=tool.spec.schema_hash,
                registry_generation=generation,
                arguments=arguments,
            )
        )
    return parsed


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _load_json(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
