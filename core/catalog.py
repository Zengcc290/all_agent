from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import ExecutionContext, ToolSpec
from .registry import BaseTool, ToolRegistry
from .repository import ToolSpecRepository


class CatalogInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search", "get_spec", "resolve"]
    intent: str | None = Field(default=None, max_length=500)
    tool_name: str | None = Field(default=None, max_length=200)
    version: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_action_arguments(self) -> CatalogInput:
        if self.action == "get_spec" and not (self.tool_name or "").strip():
            raise ValueError("tool_name is required for get_spec")
        if self.action == "resolve" and not (self.intent or "").strip():
            raise ValueError("intent is required for resolve")
        return self


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

    def __init__(
        self,
        registry: ToolRegistry,
        repository: ToolSpecRepository | None = None,
    ) -> None:
        self.registry = registry
        self.repository = repository

    def execute(
        self,
        arguments: CatalogInput,
        context: ExecutionContext | None = None,
    ) -> CatalogOutput:
        if not isinstance(arguments, CatalogInput):
            raise TypeError("arguments must be a CatalogInput instance")
        if context is None:
            context = ExecutionContext()
        elif not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext instance")
        if arguments.action == "get_spec":
            if not arguments.tool_name:
                raise ValueError("tool_name is required for get_spec")
            registration = self.registry.maybe_resolve(arguments.tool_name)
            if registration is None:
                raise ValueError(f"tool '{arguments.tool_name}' is not registered")
            tool, generation = registration
            if not set(tool.spec.permissions).issubset(context.permissions):
                raise ValueError(f"tool '{arguments.tool_name}' is not available")
            if arguments.version and arguments.version != tool.spec.version:
                raise ValueError("requested tool version is not active")
            if self.repository is not None:
                stored = self.repository.get(tool.spec.name, tool.spec.version)
                if stored is None or stored["schema_hash"] != tool.spec.schema_hash:
                    raise ValueError("active tool metadata is not synchronized")
            return CatalogOutput(spec=self._full_spec(tool.spec, generation))

        selected = self._search_specs(arguments.intent or "", arguments.limit, context)
        if arguments.action == "resolve" and selected:
            spec = selected[0]
            generation = self._active_generation(spec)
            return CatalogOutput(spec=self._full_spec(spec, generation))
        if arguments.action == "resolve":
            raise ValueError("no matching tool found")
        candidates = []
        for spec in selected:
            generation = self._active_generation(spec)
            candidates.append({**spec.summary(), "registry_generation": generation})
        return CatalogOutput(candidates=candidates)

    def _active_generation(self, spec: ToolSpec) -> int:
        tool, generation = self.registry.resolve(spec.name)
        if tool.spec != spec:
            raise ValueError("tool catalog changed while the request was running")
        return generation

    def _search_specs(
        self,
        intent: str,
        limit: int,
        context: ExecutionContext,
    ) -> list[ToolSpec]:
        active = {spec.name: spec for spec in self.registry.specs()}
        if self.repository is not None:
            active = {
                name: spec
                for name, spec in active.items()
                if (stored := self.repository.get(name, spec.version)) is not None
                and stored["schema_hash"] == spec.schema_hash
            }

        normalized_intent = intent.casefold()
        terms = set(normalized_intent.replace(".", " ").split())
        ranked = []
        for spec in active.values():
            if not set(spec.permissions).issubset(context.permissions):
                continue
            haystack = " ".join((spec.name, spec.description, *spec.tags)).casefold()
            score = sum(term in haystack for term in terms) if terms else 0
            ranked.append((score, spec))
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        return [spec for score, spec in ranked if not terms or score > 0][:limit]

    @staticmethod
    def _full_spec(spec: ToolSpec, generation: int) -> dict[str, Any]:
        return {
            **spec.summary(),
            "registry_generation": generation,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "permissions": list(spec.permissions),
            "timeout_seconds": spec.timeout_seconds,
            "idempotent": spec.idempotent,
            "parallel_safe": spec.parallel_safe,
            "max_concurrency": spec.max_concurrency,
        }
