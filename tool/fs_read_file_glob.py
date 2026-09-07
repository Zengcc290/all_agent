"""Glob files and directories under a workspace root with a bounded result set.

Matches are returned as absolute resolved paths, deterministically sorted, so
they can be passed directly to ``fs.read_text`` or edited afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import _is_within, resolve_path, workspace_root

TOOL_ENABLED = True

MAX_MATCHES = 10_000


class ReadFileGlobInput(BaseModel):
    """Parameters for one bounded glob search."""

    model_config = ConfigDict(extra="forbid", strict=True)

    root: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Directory to search. Relative paths resolve against the workspace "
            "root; absolute paths are allowed only inside the workspace."
        ),
    )
    pattern: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Glob pattern relative to root, for example '*.py', 'tests/**/*.py' "
            "or '**/*.md'. A '**' segment recurses into subdirectories; a "
            "single '*' does not cross directory boundaries."
        ),
    )
    include_hidden: bool = Field(
        default=False,
        description=(
            "When false, matches under any hidden path segment (a segment "
            "starting with a dot) are omitted, for example .git or .venv."
        ),
    )
    max_matches: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum number of matches to return (1-10000).",
    )


class ReadFileGlobOutput(BaseModel):
    """Stable glob result returned to the model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    resolved_root: str = Field(
        min_length=1, description="Absolute normalized directory that was searched."
    )
    pattern: str = Field(
        min_length=1, description="The glob pattern that was applied."
    )
    matches: list[str] = Field(
        max_length=10000,
        description=(
            "Absolute resolved paths of matched files and directories, sorted "
            "case-insensitively."
        ),
    )
    match_count: int = Field(ge=0, description="Number of matches returned.")
    truncated: bool = Field(
        description="True when more matches existed than max_matches."
    )


class ReadFileGlobTool(BaseTool):
    """Search for paths inside one workspace directory using a glob pattern."""

    spec = ToolSpec(
        name="fs.read_file_glob",
        description=(
            "Search a directory recursively for paths matching a glob pattern "
            "and return them as absolute, sorted paths. Use it to discover "
            "files by name or extension (for example '**/*.py') before reading "
            "them. Matches are confined to the search root and the workspace."
        ),
        version="1.0.0",
        input_model=ReadFileGlobInput,
        output_model=ReadFileGlobOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=15.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=8,
        tags=("fs", "glob", "find", "search"),
        recommended_before_tools=("fs.read_dir",),
    )

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        allow_outside: bool | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else workspace_root()
        self.allow_outside = allow_outside

    def execute(self, arguments: ReadFileGlobInput) -> ReadFileGlobOutput:
        if not isinstance(arguments, ReadFileGlobInput):
            raise TypeError("arguments must be a ReadFileGlobInput instance")
        root = resolve_path(
            self.base_dir, arguments.root, allow_outside=self.allow_outside
        )
        if not root.exists():
            raise FileNotFoundError(f"root '{arguments.root}' does not exist")
        if not root.is_dir():
            raise NotADirectoryError(f"'{arguments.root}' is not a directory")
        try:
            raw_matches = list(root.glob(arguments.pattern))
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(
                f"invalid glob pattern '{arguments.pattern}': {exc}"
            ) from exc

        matches: list[Path] = []
        escaped = False
        for match in raw_matches:
            resolved = match.resolve()
            if not _is_within(root, resolved):
                escaped = True
                continue
            if not arguments.include_hidden and _has_hidden_segment(
                resolved.relative_to(root)
            ):
                continue
            matches.append(resolved)
        if escaped:
            raise ValueError(
                "glob pattern escapes the search root; use a pattern confined "
                "to the root directory"
            )
        matches.sort(key=lambda item: os.path.normcase(str(item)))
        truncated = len(matches) > arguments.max_matches
        bounded = [str(item) for item in matches[: arguments.max_matches]]
        return ReadFileGlobOutput(
            resolved_root=str(root),
            pattern=arguments.pattern,
            matches=bounded,
            match_count=len(bounded),
            truncated=truncated,
        )


def _has_hidden_segment(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def create_tool() -> BaseTool:
    return ReadFileGlobTool()


__all__ = [
    "TOOL_ENABLED",
    "ReadFileGlobInput",
    "ReadFileGlobOutput",
    "ReadFileGlobTool",
    "create_tool",
]