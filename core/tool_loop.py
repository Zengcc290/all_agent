"""Shared round control for model/tool conversations.

The loop deliberately knows nothing about ReAct text or provider-native
function calls.  Adapters parse one model response and then use this class for
the common lifecycle concerns: round limits, a safety limit for ``None`` and
repeated-call detection.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from constants import TOOL_LOOP_SAFETY_LIMIT


class ToolLoop:
    """Generate bounded conversation rounds and detect stuck tool calls."""

    # 单一来源是 constants.TOOL_LOOP_SAFETY_LIMIT；类属性保留为兼容别名。
    DEFAULT_SAFETY_LIMIT = TOOL_LOOP_SAFETY_LIMIT

    def __init__(self, max_rounds: int | None = None, *, safety_limit: int = DEFAULT_SAFETY_LIMIT) -> None:
        if max_rounds is not None and (
            isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1
        ):
            raise ValueError("max_rounds must be None or a positive integer")
        if isinstance(safety_limit, bool) or not isinstance(safety_limit, int) or safety_limit < 1:
            raise ValueError("safety_limit must be a positive integer")
        self.max_rounds = max_rounds
        self.safety_limit = safety_limit
        self._calls: dict[str, int] = {}

    def rounds(self) -> Iterator[int]:
        """Yield one-based round numbers until the configured bound."""
        round_number = 0
        while (
            (self.max_rounds is None and round_number < self.safety_limit)
            or (self.max_rounds is not None and round_number < self.max_rounds)
        ):
            round_number += 1
            yield round_number

    def record_call(self, name: str, arguments: Mapping[str, Any] | Any) -> None:
        """Record a tool call and fail fast when a provider repeats it forever."""
        signature = json.dumps(
            [name, arguments],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        count = self._calls.get(signature, 0) + 1
        self._calls[signature] = count
        if count > 3:
            raise RuntimeError(
                "tool loop stopped after the same tool call was repeated more than three times"
            )


__all__ = ["ToolLoop"]
