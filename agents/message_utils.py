"""JSON-compatible conversion helpers shared by the agent modules.

Provider SDKs return a mix of Pydantic models, dataclasses and plain mappings.
``agent.py`` (function-calling loop), ``react.py`` (text protocol) and ``llm.py``
(streaming adapters) all need the same normalization, and before this module
existed ``_field`` and the message/tool-call converters were copied per module,
so a provider quirk fixed in one place stayed broken in the others.

Nothing here imports the agent runtime: these helpers only translate shapes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

DEFAULT_TOOL_NAME = "unknown.tool"
MAX_TOOL_NAME_CHARS = 200
MAX_TOOL_ERROR_CHARS = 1000


def field(value: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an object, falling back to ``default``."""

    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def safe_tool_name(value: Any) -> str:
    """Bounded tool name for logs and error placeholders."""

    name = value.strip() if isinstance(value, str) else DEFAULT_TOOL_NAME
    return (name or DEFAULT_TOOL_NAME)[:MAX_TOOL_NAME_CHARS]


def safe_tool_call_error(error: Exception) -> str:
    """Bounded error text; a blank message falls back to the exception class."""

    message = str(error) or type(error).__name__
    return message[:MAX_TOOL_ERROR_CHARS]


def tool_call_dict(item: Any) -> dict[str, Any]:
    """Convert one SDK tool-call object into a JSON-compatible message part."""

    function = field(item, "function")
    result: dict[str, Any] = {
        "id": field(item, "id"),
        "type": field(item, "type", "function"),
        "function": {
            "name": field(function, "name"),
            "arguments": field(function, "arguments", "{}"),
        },
    }
    return {key: value for key, value in result.items() if value is not None}


def message_dict(message: Any) -> dict[str, Any]:
    """Normalize an SDK assistant message into a plain dict for the transcript."""

    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {
            key: value
            for key in ("role", "content", "tool_calls")
            if (value := getattr(message, key, None)) is not None
        }
    data.setdefault("role", "assistant")
    if "tool_calls" in data and data["tool_calls"] is not None:
        data["tool_calls"] = [tool_call_dict(item) for item in data["tool_calls"]]
    return data


def result_json(result: Any) -> str:
    """Serialize tool results for providers, including permissive Any fields."""

    try:
        payload = result.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        payload = result.model_dump()
    return json.dumps(payload, ensure_ascii=False, default=str)


__all__ = [
    "DEFAULT_TOOL_NAME",
    "MAX_TOOL_ERROR_CHARS",
    "MAX_TOOL_NAME_CHARS",
    "field",
    "message_dict",
    "result_json",
    "safe_tool_call_error",
    "safe_tool_name",
    "tool_call_dict",
]
