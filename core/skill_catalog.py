"""The ``system.skill_catalog`` tool: on-demand skill viewing for the model.

Following the same catalog-first philosophy as ``system.tool_catalog``, only
skill names/descriptions/triggers are kept in the persistent system message;
the full ``<name>.md`` content is returned here, per ``view`` call, and
therefore stays after the reusable prompt prefix (KV-cache friendly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import ExecutionContext, ToolSpec
from .registry import BaseTool
from .skill_discovery import read_skill_content
from .skill_registry import SkillRegistry


class SkillCatalogInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["list", "view"]
    skill_name: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_action_arguments(self) -> SkillCatalogInput:
        if self.action == "view" and not (self.skill_name or "").strip():
            raise ValueError("skill_name is required for view")
        return self


class SkillCatalogOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    # Populated by ``list``: one entry per skill, names/descriptions only.
    skills: list[dict[str, Any]] = Field(default_factory=list)
    # Populated by ``view``: the full current skill file content.
    content: str | None = None
    # Populated by ``view``: version + content hash for change detection.
    version: str | None = None
    content_hash: str | None = None


class SkillCatalogTool(BaseTool):
    """Read-only facade over the skill registry and skill files."""

    spec = ToolSpec(
        name="system.skill_catalog",
        description=(
            "List available skills or load the full instruction content of one "
            "skill. Use action=view when the current task matches a skill's "
            "description or triggers; then follow the returned content. "
            "Skills are read-only playbooks, not callable tools."
        ),
        version="1.1",
        input_model=SkillCatalogInput,
        output_model=SkillCatalogOutput,
        side_effect="read",
        tags=("catalog", "skills", "instructions"),
    )

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        root: str | Path = "skills",
    ) -> None:
        if not isinstance(registry, SkillRegistry):
            raise TypeError("registry must be a SkillRegistry")
        self.registry = registry
        self.root = Path(root).resolve()

    def execute(
        self,
        arguments: SkillCatalogInput,
        context: ExecutionContext | None = None,
    ) -> SkillCatalogOutput:
        if not isinstance(arguments, SkillCatalogInput):
            raise TypeError("arguments must be a SkillCatalogInput instance")
        del context  # read-only tool; context is accepted for signature parity
        if arguments.action == "list":
            skills = [spec.summary() for spec, _ in self.registry.snapshot().values()]
            return SkillCatalogOutput(skills=skills)
        # action == "view"
        registration = self.registry.maybe_resolve(
            (arguments.skill_name or "").strip()
        )
        if registration is None:
            raise ValueError(
                f"skill '{arguments.skill_name}' is not registered; call "
                "action=list to see available skills"
            )
        spec, _ = registration
        content = read_skill_content(spec)
        return SkillCatalogOutput(
            content=content,
            version=spec.version,
            content_hash=spec.content_hash,
        )
