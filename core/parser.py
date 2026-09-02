from __future__ import annotations

import json
from typing import Any

from .models import ToolCall
from .registry import ToolRegistry


def parse_tool_calls(payload: str | dict[str, Any] | list[dict[str, Any]]) -> list[ToolCall]:
    """Parse native or fallback JSON calls without executing anything."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        payload = payload.get("tool_calls", payload)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("tool call payload must be an object or list")
    return [ToolCall.model_validate(item, strict=True) for item in payload]


def parse_openai_tool_calls(tool_calls: list[Any], registry: ToolRegistry) -> list[ToolCall]:
    """Convert OpenAI-compatible native calls into the internal versioned envelope."""
    parsed: list[ToolCall] = []
    for item in tool_calls:
        call_id = getattr(item, "id", None)
        function = getattr(item, "function", None)
        name = getattr(function, "name", None)
        raw_arguments = getattr(function, "arguments", "{}")
        if not call_id or not name:
            raise ValueError("native tool call is missing id or function name")
        tool = registry.get(name)
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        parsed.append(ToolCall(
            call_id=call_id,
            tool_name=name,
            schema_version=tool.spec.version,
            schema_hash=tool.spec.schema_hash,
            arguments=arguments,
        ))
    return parsed
