"""The ``system.skill_catalog`` tool: on-demand skill viewing for the model.

Following the same catalog-first philosophy as ``system.tool_catalog``, only
skill names/descriptions/triggers are kept in the persistent system message;
full ``SKILL.md`` content is returned here, per ``view`` call, and therefore
stays after the reusable prompt prefix (KV-cache friendly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import ExecutionContext, ToolSpec
from .registry import BaseTool
from .skill_discovery import read_skill_content
from .skill_models import SkillSpec
from .skill_registry import SkillRegistry


class SkillCatalogInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["list", "view", "read_reference"]
    skill_name: str | None = Field(default=None, max_length=64)
    # Relative to the skill directory; must stay inside it.
    reference_path: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_action_arguments(self) -> SkillCatalogInput:
        if self.action == "view" and not (self.skill_name or "").strip():
            raise ValueError("skill_name is required for view")
        if self.action == "read_reference":
            if not (self.skill_name or "").strip():
                raise ValueError("skill_name is required for read_reference")
            if not (self.reference_path or "").strip():
                raise ValueError("reference_path is required for read_reference")
        return self


class SkillCatalogOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    # Populated by ``list``: one entry per skill, names/descriptions only.
    skills: list[dict[str, Any]] = Field(default_factory=list)
    # Populated by ``view``: the full current SKILL.md content.
    content: str | None = None
    # Populated by ``view``: version + content hash for change detection.
    version: str | None = None
    content_hash: str | None = None
    # Populated by ``read_reference``.
    reference_path: str | None = None
    reference_content: str | None = None


class SkillCatalogTool(BaseTool):
    """Read-only facade over the skill registry and skill directories."""

    spec = ToolSpec(
        name="system.skill_catalog",
        description=(
            "List available skills or load the full instruction content of one "
            "skill. Use action=view when the current task matches a skill's "
            "description or triggers; then follow the returned content. "
            "Skills are read-only playbooks, not callable tools."
        ),
        version="1.0",
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
        if arguments.action == "view":
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
        # action == "read_reference"
        registration = self.registry.maybe_resolve(
            (arguments.skill_name or "").strip()
        )
        if registration is None:
            raise ValueError(f"skill '{arguments.skill_name}' is not registered")
        spec, _ = registration
        resolved = self._resolve_reference(spec, arguments.reference_path or "")
        relative = resolved.relative_to(Path(spec.directory or "."))
        return SkillCatalogOutput(
            reference_path=relative.as_posix(),
            reference_content=resolved.read_text(encoding="utf-8"),
        )

    def _resolve_reference(self, spec: SkillSpec, reference_path: str) -> Path:
        """Resolve a reference path strictly inside the skill directory."""

        if spec.directory is None:
            raise ValueError(f"skill '{spec.name}' has no source directory")
        base = Path(spec.directory).resolve()
        candidate = (base / reference_path).resolve()
        if candidate == base or base not in candidate.parents:
            raise ValueError(
                "reference_path must resolve to a file inside the skill "
                "directory (path traversal is blocked)"
            )
        if not candidate.is_file():
            raise FileNotFoundError(
                f"reference file not found inside skill '{spec.name}': "
                f"{reference_path}"
            )
        return candidate
