from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Mapping
from typing import Any

LOGGER = logging.getLogger(__name__)


class LLM:
    """Small OpenAI-compatible client wrapper with lazy SDK import."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        from dotenv import load_dotenv

        load_dotenv(override=False)
        self.api_key = os.getenv("LLM_API_KEY") if api_key is None else api_key
        self.base_url = os.getenv("LLM_BASE_URL") if base_url is None else base_url
        self.model = os.getenv("LLM_MODEL") if model is None else model
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise TypeError("base_url must be a string")
        if self.model is not None and not isinstance(self.model, str):
            raise TypeError("model must be a string")
        self.max_retries = max_retries
        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", self.api_key),
                ("LLM_BASE_URL", self.base_url),
                ("LLM_MODEL", self.model),
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

    def think(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        timeout: float = 60,
        stream_response_bool: bool = True,
        **kwargs: Any,
    ) -> str:
        response = self.complete(
            messages,
            temperature=temperature,
            timeout=timeout,
            stream=stream_response_bool,
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
        reserved = {
            "messages",
            "model",
            "temperature",
            "stream",
            "timeout",
        } & kwargs.keys()
        if reserved:
            raise TypeError(
                f"reserved completion arguments cannot be overridden: {', '.join(sorted(reserved))}"
            )
        return self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            stream=stream,
            timeout=timeout,
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
