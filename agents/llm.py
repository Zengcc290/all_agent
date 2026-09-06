from __future__ import annotations

import logging
import math
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)

LOGGER = logging.getLogger(__name__)

# Line-start final-answer markers shared by the ReAct text protocol. The live
# echo stays silent until one of these appears, then streams the remainder.
_FINAL_ANSWER_MARKER_RE = re.compile(
    r"(?im)^(?:\*\*)?(?:final\s+answer|最终答案|最终回答|最终回复)(?:\*\*)?\s*[：:]"
)

EchoMode = Literal["react_final", "content"]


def _default_echo_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class _FinalAnswerEchoer:
    """Terminal echo controller for one streamed completion.

    ``content`` mode forwards every delta immediately (native tool rounds
    normally carry no content, so this effectively streams only answer text).
    ``react_final`` mode buffers silently until a final-answer marker shows
    up, then streams everything after the marker; rounds that end in a tool
    call never echo anything.
    """

    def __init__(
        self,
        mode: EchoMode,
        write: Callable[[str], None] | None,
        prefix: str = "\nAI：",
    ) -> None:
        self._mode = mode
        self._write = write or _default_echo_write
        self._prefix = prefix
        self._buffer = ""
        self._live = mode == "content"
        self.echoed = False

    def feed(self, piece: str) -> None:
        if not piece:
            return
        if self._live:
            self._emit(piece)
            return
        self._buffer += piece
        match = _FINAL_ANSWER_MARKER_RE.search(self._buffer)
        if match is not None:
            tail = self._buffer[match.end() :]
            self._buffer = ""
            self._live = True
            if tail:
                self._emit(tail)

    def flush(self) -> None:
        """Close the echo line once anything was written to the terminal."""

        if self.echoed:
            self._write("\n")

    def _emit(self, text: str) -> None:
        if not text:
            return
        if not self.echoed and self._prefix:
            try:
                self._write(self._prefix)
            except Exception:  # noqa: BLE001 - echo must never break assembly
                LOGGER.debug("stream echo write failed", exc_info=True)
        self.echoed = True
        try:
            self._write(text)
        except Exception:  # noqa: BLE001 - echo must never break assembly
            LOGGER.debug("stream echo write failed", exc_info=True)


def _assemble_streaming_response(
    stream: Iterable[Any],
    *,
    on_first_chunk: Callable[[], None] | None = None,
    echo_mode: EchoMode | None = None,
    echo_write: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Consume one streamed completion and rebuild the response envelope.

    Assembles ``content`` deltas plus streaming ``tool_calls`` fragments (the
    OpenAI chunks deliver ``index``-keyed pieces with partial ``id``,
    ``function.name`` and ``function.arguments`` strings). The returned
    mapping matches the non-streaming response shape, so message extraction
    helpers need no branching.

    ``echo_mode`` optionally mirrors content pieces to the terminal while the
    stream is still running; see :class:`_FinalAnswerEchoer`.
    """

    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    first_chunk_seen = False
    echoer = _FinalAnswerEchoer(echo_mode, echo_write) if echo_mode else None
    try:
        for chunk in stream:
            choices = _field(chunk, "choices")
            if not choices:
                continue
            choice = choices[0]
            if not first_chunk_seen:
                first_chunk_seen = True
                if on_first_chunk is not None:
                    try:
                        on_first_chunk()
                    except Exception:  # noqa: BLE001 - callback must not break the loop
                        LOGGER.debug("on_first_chunk callback failed", exc_info=True)
            if (reason := _field(choice, "finish_reason")) is not None:
                finish_reason = reason
            delta = _field(choice, "delta")
            if delta is None:
                continue
            if (piece := _field(delta, "content")):
                content_parts.append(piece)
                if echoer is not None:
                    echoer.feed(piece)
            raw_fragments = _field(delta, "tool_calls")
            if not raw_fragments:
                continue
            for fragment in raw_fragments:
                index = _field(fragment, "index") or 0
                entry = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if (fragment_id := _field(fragment, "id")):
                    entry["id"] = (
                        fragment_id if not entry["id"] else entry["id"] + fragment_id
                    )
                function = _field(fragment, "function")
                if function is None:
                    continue
                if (name := _field(function, "name")):
                    entry["function"]["name"] += name
                if (arguments := _field(function, "arguments")):
                    entry["function"]["arguments"] += arguments
    finally:
        if echoer is not None:
            echoer.flush()
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                LOGGER.debug("failed to close LLM stream", exc_info=True)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = [
            tool_calls[index] for index in sorted(tool_calls)
        ]
    response: dict[str, Any] = {
        "choices": [
            {
                "message": message,
                "finish_reason": finish_reason,
            }
        ]
    }
    if echoer is not None:
        # Callers use this to detect answers that never reached the terminal
        # (no final-answer marker) and need a one-shot fallback echo.
        response["stream_echoed"] = echoer.echoed
    return response


class LLM:
    """Small OpenAI-compatible client wrapper for one resolved profile.

    ``complete`` forwards OpenAI's prompt-cache routing fields when supplied.
    Callers can therefore keep a stable cache key across turns while older
    compatible gateways continue to work when the fields are omitted.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        if not isinstance(base_url, str):
            raise TypeError("base_url must be a string")
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        self.api_key = api_key.strip()
        self.base_url = base_url.strip()
        self.model = model.strip()
        self.max_retries = max_retries
        missing = [
            name
            for name, value in (
                ("api_key", self.api_key),
                ("base_url", self.base_url),
                ("model", self.model),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Configuration Error: {', '.join(missing)} is not configured."
            )
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=self.max_retries,
            )
        except ImportError as exc:
            raise RuntimeError("The 'openai' package is required to use LLM.") from exc

    @staticmethod
    def get_query() -> str:
        return input("请输入你的问题：")

    def complete_streaming(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        on_first_chunk: Callable[[], None] | None = None,
        echo_mode: EchoMode | None = None,
        echo_write: Callable[[str], None] | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one streaming completion and return an assembled response.

        Tool loops must not use non-streaming requests: a gateway generating a
        long completion sends no bytes until the whole response exists, so the
        read timeout fires long before the gateway finishes and the OpenAI SDK
        retries the identical request in the background. Streaming keeps the
        connection alive from the first token, making the same timeout measure
        the gap *between* tokens instead of the total generation time.

        The assembled mapping mimics the non-streaming response shape so the
        existing message extraction helpers keep working unchanged.
        """

        reserved = {"messages", "model", "stream"} & kwargs.keys()
        if reserved:
            raise TypeError(
                f"reserved streaming arguments cannot be overridden: {', '.join(sorted(reserved))}"
            )
        options: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "timeout": timeout,
        }
        if prompt_cache_key is not None:
            options["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            options["prompt_cache_retention"] = prompt_cache_retention
        stream = self.client.chat.completions.create(**options, **kwargs)
        return _assemble_streaming_response(
            stream,
            on_first_chunk=on_first_chunk,
            echo_mode=echo_mode,
            echo_write=echo_write,
        )

    def think(
        self,
        messages: list[dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        stream_response_bool: bool = True,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        **kwargs: Any,
    ) -> str:
        response = self.complete(
            messages,
            temperature=temperature,
            timeout=timeout,
            stream=stream_response_bool,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            **kwargs,
        )
        if stream_response_bool:
            return "".join(self.stream_response(response))
        return self._message_content(response)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        stream: bool = False,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
        ):
            raise ValueError("temperature must be a finite number")
        if not isinstance(stream, bool):
            raise TypeError("stream must be a boolean")
        if prompt_cache_key is not None:
            if (
                not isinstance(prompt_cache_key, str)
                or not prompt_cache_key.strip()
                or len(prompt_cache_key) > 64
            ):
                raise ValueError(
                    "prompt_cache_key must be a non-empty string of at most 64 characters"
                )
            prompt_cache_key = prompt_cache_key.strip()
        if prompt_cache_retention is not None:
            if prompt_cache_retention not in {"in_memory", "24h"}:
                raise ValueError(
                    "prompt_cache_retention must be 'in_memory', '24h', or None"
                )
        reserved = {
            "messages",
            "model",
            "temperature",
            "stream",
            "timeout",
            "prompt_cache_key",
            "prompt_cache_retention",
        } & kwargs.keys()
        if reserved:
            raise TypeError(
                f"reserved completion arguments cannot be overridden: {', '.join(sorted(reserved))}"
            )
        options: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "timeout": timeout,
        }
        # These are first-class OpenAI Chat Completions parameters.  Omitting
        # them when unset keeps older OpenAI-compatible gateways working.
        if prompt_cache_key is not None:
            options["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            options["prompt_cache_retention"] = prompt_cache_retention
        return self.client.chat.completions.create(
            **options,
            **kwargs,
        )

    @staticmethod
    def _message_content(response: Any) -> str:
        choices = (
            response.get("choices")
            if isinstance(response, Mapping)
            else getattr(response, "choices", None)
        )
        if not choices:
            raise ValueError("LLM response contained no choices")
        message = (
            choices[0].get("message")
            if isinstance(choices[0], Mapping)
            else getattr(choices[0], "message", None)
        )
        if message is None:
            raise ValueError("LLM response contained no message")
        content = (
            message.get("content")
            if isinstance(message, Mapping)
            else getattr(message, "content", None)
        )
        return content or ""

    def stream_response(self, response: Iterable[Any]):
        """Yield text deltas from SDK or mapping-based streaming chunks.

        A ``finally`` block closes synchronous streams when a caller stops
        iteration early (including generator cancellation), preventing an
        open HTTP connection from being leaked.
        """
        try:
            for chunk in response:
                choices = _field(chunk, "choices")
                if not choices:
                    continue
                choice = choices[0]
                delta = _field(choice, "delta")
                content = _field(delta, "content")
                if content:
                    print(content, end="", flush=True)
                    yield content
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Closing is best effort; never hide a useful stream error.
                    LOGGER.debug("failed to close LLM stream", exc_info=True)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)
