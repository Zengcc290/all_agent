from __future__ import annotations

import inspect
import threading
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .activity_log import log_tool_registration
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
        self._generations: dict[str, int] = {}
        self._lock = threading.RLock()

    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be a BaseTool instance")
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean")
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, ToolSpec):
            raise TypeError("tool.spec must be a ToolSpec instance")
        if not inspect.isclass(spec.input_model) or not inspect.isclass(
            spec.output_model
        ):
            raise TypeError("tool input_model and output_model must be Pydantic models")
        name = spec.name
        with self._lock:
            if name in self._tools and not replace:
                raise ValueError(f"tool '{name}' is already registered")
            self._tools[name] = tool
            self._generations[name] = self._generations.get(name, 0) + 1
            generation = self._generations[name]
            registered_names = tuple(self._tools)
        log_tool_registration(name, generation, registered_names)

    def unregister(self, name: str) -> BaseTool:
        """Remove one executable from the runtime registry.

        The per-name generation counter is preserved so the generation
        sequence stays monotonic across unregister/re-register cycles. The
        removed tool instance is returned to the caller.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        with self._lock:
            tool = self._tools.pop(name, None)
            if tool is None:
                raise KeyError(f"tool '{name}' is not registered")
        return tool

    def confirmation_key(self, name: str) -> str:
        """Return a key invalidated whenever the registered implementation changes."""
        tool, generation = self.resolve(name)
        return f"{tool.spec.confirmation_key}:{generation}"

    def resolve(self, name: str) -> tuple[BaseTool, int]:
        """Atomically return an executable and its current registration generation."""
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        with self._lock:
            try:
                return self._tools[name], self._generations[name]
            except KeyError as exc:
                raise KeyError(f"tool '{name}' is not registered") from exc

    def maybe_resolve(self, name: str) -> tuple[BaseTool, int] | None:
        if not isinstance(name, str) or not name:
            return None
        with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                return None
            return tool, self._generations[name]

    def snapshot(
        self, names: list[str] | tuple[str, ...] | None = None
    ) -> dict[str, tuple[BaseTool, int]]:
        """Capture a stable name-to-executable view for one provider request."""
        if names is not None and (
            not isinstance(names, (list, tuple))
            or not all(isinstance(name, str) and name for name in names)
        ):
            raise TypeError("names must be a list of non-empty strings")
        with self._lock:
            if names is None:
                return {
                    name: (tool, self._generations[name])
                    for name, tool in self._tools.items()
                }
            missing = [name for name in names if name not in self._tools]
            if missing:
                raise KeyError(f"tool '{missing[0]}' is not registered")
            return {
                name: (self._tools[name], self._generations[name]) for name in names
            }

    def get(self, name: str) -> BaseTool:
        return self.resolve(name)[0]

    def maybe_get(self, name: str) -> BaseTool | None:
        registration = self.maybe_resolve(name)
        return registration[0] if registration is not None else None

    def is_registered(
        self,
        name: str,
        *,
        version: str | None = None,
        schema_hash: str | None = None,
    ) -> bool:
        """Return whether an active registration matches the requested contract."""
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if version is not None and (
            not isinstance(version, str) or not version.strip()
        ):
            raise ValueError("version must be a non-empty string or None")
        if schema_hash is not None and (
            not isinstance(schema_hash, str) or not schema_hash.strip()
        ):
            raise ValueError("schema_hash must be a non-empty string or None")
        registration = self.maybe_resolve(name)
        if registration is None:
            return False
        tool, _ = registration
        return (version is None or tool.spec.version == version) and (
            schema_hash is None or tool.spec.schema_hash == schema_hash
        )

    def registration_status(self, name: str) -> dict[str, Any]:
        """Return JSON-friendly details for one active tool registration."""
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        registration = self.maybe_resolve(name)
        if registration is None:
            return {
                "name": name,
                "registered": False,
                "version": None,
                "schema_hash": None,
                "generation": None,
                "implementation": None,
            }
        tool, generation = registration
        implementation = f"{type(tool).__module__}:{type(tool).__qualname__}"
        return {
            "name": name,
            "registered": True,
            "version": tool.spec.version,
            "schema_hash": tool.spec.schema_hash,
            "generation": generation,
            "implementation": implementation,
        }

    def summaries(self) -> list[dict[str, Any]]:
        return [tool.spec.summary() for tool, _ in self.snapshot().values()]

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool, _ in self.snapshot().values()]

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)
