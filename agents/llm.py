from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class LLM:
    """Small OpenAI-compatible client wrapper with lazy SDK import."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL")
        self.max_retries = max_retries
        missing = [
            name for name, value in (
                ("LLM_API_KEY", self.api_key),
                ("LLM_BASE_URL", self.base_url),
                ("LLM_MODEL", self.model),
            ) if not value
        ]
        if missing:
            raise ValueError(f"Configuration Error: {', '.join(missing)} is not configured.")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=self.max_retries)
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=stream_response_bool,
            timeout=timeout,
            **kwargs,
        )
        if stream_response_bool:
            return "".join(self.stream_response(response))
        return self._message_content(response)

    @staticmethod
    def _message_content(response: Any) -> str:
        message = response.choices[0].message
        return message.content or ""

    def stream_response(self, response: Iterable[Any]):
        for chunk in response:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            content = getattr(getattr(choices[0], "delta", None), "content", None)
            if content:
                print(content, end="", flush=True)
                yield content
