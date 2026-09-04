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
    # Resolve/search can return a complete set of matching capabilities in one
    # call.  Callers may still lower this explicitly when they need a smaller
    # result set.
    limit: int = Field(default=20, ge=1, le=20)

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
    # ``spec`` remains for callers that expect one resolved tool.  ``specs``
    # contains every matching complete contract for a resolve/get_spec call.
    specs: list[dict[str, Any]] = Field(default_factory=list)


class ToolCatalogTool(BaseTool):
    """A constrained, read-only catalog facade; it never accepts raw SQL.

    ``resolve`` returns all matching complete tool contracts in ``specs`` so a
    caller can discover and load several capabilities in one catalog request.
    """

    spec = ToolSpec(
        name="system.tool_catalog",
        description="Find available tools or load complete versioned schemas for one or more matching tools.",
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
        *,
        repository_only: bool = False,
    ) -> None:
        self.registry = registry
        self.repository = repository
        if not isinstance(repository_only, bool):
            raise TypeError("repository_only must be a boolean")
        self.repository_only = repository_only

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
                # A persistent catalog may intentionally contain metadata for
                # tools whose executable implementation has not been loaded
                # into this process yet.  Returning that metadata is what
                # enables a catalog-first/lazy execution workflow.
                if self.repository is None:
                    raise ValueError(
                        f"tool '{arguments.tool_name}' is not registered"
                    )
                stored = self.repository.get(arguments.tool_name, arguments.version)
                if stored is None:
                    raise ValueError(
                        f"tool '{arguments.tool_name}' is not registered"
                    )
                full_spec = self._full_stored_spec(stored, 0)
                return CatalogOutput(spec=full_spec, specs=[full_spec])
            tool, generation = registration
            if arguments.version and arguments.version != tool.spec.version:
                raise ValueError("requested tool version is not active")
            if self.repository is not None:
                stored = self.repository.get(tool.spec.name, tool.spec.version)
                if stored is None or stored["schema_hash"] != tool.spec.schema_hash:
                    raise ValueError("active tool metadata is not synchronized")
            full_spec = self._full_spec(tool.spec, generation)
            return CatalogOutput(spec=full_spec, specs=[full_spec])

        if self.repository is not None and (
            self.repository_only or not self._has_loaded_tools()
        ):
            # Permission declarations remain catalog metadata, but must not
            # hide tools while authorization is disabled project-wide.
            selected_records = self.repository.search(
                arguments.intent or "", arguments.limit
            )
            if arguments.action == "resolve" and selected_records:
                specs = [
                    self._full_stored_spec(record, self._stored_generation(record))
                    for record in selected_records
                ]
                return CatalogOutput(spec=specs[0], specs=specs)
            if arguments.action == "resolve":
                raise ValueError("no matching tool found")
            return CatalogOutput(
                candidates=[
                    {
                        "tool_name": item["tool_name"],
                        "description": item["description"],
                        "version": item["version"],
                        "schema_hash": item["schema_hash"],
                        "side_effect": item["side_effect"],
                        "max_concurrency": item["max_concurrency"],
                        "tags": list(item["tags"]),
                        "recommended_before_tools": list(
                            item["recommended_before_tools"]
                        ),
                        "registry_generation": self._stored_generation(item),
                    }
                    for item in selected_records
                ]
            )

        selected = self._search_specs(arguments.intent or "", arguments.limit, context)
        if arguments.action == "resolve" and selected:
            specs = [
                self._full_spec(spec, self._active_generation(spec))
                for spec in selected
            ]
            return CatalogOutput(spec=specs[0], specs=specs)
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

    def _has_loaded_tools(self) -> bool:
        """Whether the registry contains an executable tool besides catalog."""

        return any(
            name != self.spec.name for name in self.registry.snapshot()
        )

    def _stored_generation(self, stored: dict[str, Any]) -> int:
        """Return the active registry generation when metadata is loaded.

        ``0`` explicitly means "catalog-only".  Once a lazy agent loads the
        implementation, the next catalog call reports its real generation.
        """

        registration = self.registry.maybe_resolve(stored["tool_name"])
        if registration is None:
            return 0
        tool, generation = registration
        if tool.spec.version != stored["version"] or tool.spec.schema_hash != stored[
            "schema_hash"
        ]:
            return 0
        return generation

    def _search_specs(
        self,
        intent: str,
        limit: int,
        _context: ExecutionContext,
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
            haystack = " ".join((spec.name, spec.description, *spec.tags)).casefold()
            score = sum(term in haystack for term in terms) if terms else 0
            ranked.append((score, spec))
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        return [spec for score, spec in ranked if not terms or score > 0][:limit]

    @staticmethod
    def _full_stored_spec(stored: dict[str, Any], generation: int) -> dict[str, Any]:
        """Convert a repository row to the same shape as ``_full_spec``."""

        return {
            "tool_name": stored["tool_name"],
            "description": stored["description"],
            "version": stored["version"],
            "schema_hash": stored["schema_hash"],
            "side_effect": stored["side_effect"],
            "max_concurrency": stored["max_concurrency"],
            "tags": list(stored["tags"]),
            "recommended_before_tools": list(stored["recommended_before_tools"]),
            "registry_generation": generation,
            "input_schema": stored["input_schema"],
            "output_schema": stored["output_schema"],
            "permissions": list(stored["permissions"]),
            "timeout_seconds": stored["timeout_seconds"],
            "idempotent": stored["idempotent"],
            "parallel_safe": stored["parallel_safe"],
        }

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
