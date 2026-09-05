from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

LOGGER = logging.getLogger(__name__)


def _assemble_streaming_response(
    stream: Iterable[Any],
    *,
    on_first_chunk: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Consume one streamed completion and rebuild the response envelope.

    Assembles ``content`` deltas plus streaming ``tool_calls`` fragments (the
    OpenAI chunks deliver ``index``-keyed pieces with partial ``id``,
    ``function.name`` and ``function.arguments`` strings). The returned
    mapping matches the non-streaming response shape, so message extraction
    helpers need no branching.
    """

    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    first_chunk_seen = False
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
        max_retries: int = 3,
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
        temperature: float = 0.7,
        timeout: float = 60,
        on_first_chunk: Callable[[], None] | None = None,
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
        return _assemble_streaming_response(stream, on_first_chunk=on_first_chunk)

    def think(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        timeout: float = 60,
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
        temperature: float = 0.7,
        timeout: float = 60,
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
