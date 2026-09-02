from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import ToolSpec
from .registry import BaseTool, ToolRegistry


class CatalogInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search", "get_spec", "resolve"]
    intent: str | None = Field(default=None, max_length=500)
    tool_name: str | None = Field(default=None, max_length=200)
    version: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=5, ge=1, le=20)


class CatalogOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidates: list[dict[str, Any]] = Field(default_factory=list)
    spec: dict[str, Any] | None = None


class ToolCatalogTool(BaseTool):
    """A constrained, read-only catalog facade; it never accepts raw SQL."""

    spec = ToolSpec(
        name="system.tool_catalog",
        description="Find available tools or load one tool's complete versioned schema.",
        version="1.0",
        input_model=CatalogInput,
        output_model=CatalogOutput,
        side_effect="read",
        tags=("catalog", "discovery"),
    )

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, arguments: CatalogInput) -> CatalogOutput:
        if arguments.action == "get_spec":
            if not arguments.tool_name:
                raise ValueError("tool_name is required for get_spec")
            tool = self.registry.maybe_get(arguments.tool_name)
            if tool is None:
                raise ValueError(f"tool '{arguments.tool_name}' is not registered")
            if arguments.version and arguments.version != tool.spec.version:
                raise ValueError("requested tool version is not active")
            return CatalogOutput(spec=self._full_spec(tool.spec))

        intent = (arguments.intent or "").casefold()
        terms = set(intent.replace(".", " ").split())
        ranked = []
        for spec in self.registry.specs():
            haystack = " ".join((spec.name, spec.description, *spec.tags)).casefold()
            score = sum(term in haystack for term in terms) if terms else 0
            ranked.append((score, spec))
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        selected = [spec for score, spec in ranked if score > 0] or [spec for _, spec in ranked]
        selected = selected[: arguments.limit]
        if arguments.action == "resolve" and selected:
            return CatalogOutput(spec=self._full_spec(selected[0]))
        return CatalogOutput(candidates=[spec.summary() for spec in selected])

    @staticmethod
    def _full_spec(spec: ToolSpec) -> dict[str, Any]:
        return {
            **spec.summary(),
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "permissions": list(spec.permissions),
            "timeout_seconds": spec.timeout_seconds,
            "idempotent": spec.idempotent,
            "parallel_safe": spec.parallel_safe,
        }
