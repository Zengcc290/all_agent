from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .models import ToolSpec


class BaseTool(ABC):
    spec: ToolSpec

    @abstractmethod
    def execute(self, arguments: BaseModel) -> BaseModel | Any:
        """Execute validated arguments and return data matching output_model."""
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        name = tool.spec.name
        if name in self._tools and not replace:
            raise ValueError(f"tool '{name}' is already registered")
        if not inspect.isclass(tool.spec.input_model) or not inspect.isclass(tool.spec.output_model):
            raise TypeError("tool input_model and output_model must be Pydantic models")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool '{name}' is not registered") from exc

    def maybe_get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def summaries(self) -> list[dict[str, Any]]:
        return [tool.spec.summary() for tool in self._tools.values()]

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
