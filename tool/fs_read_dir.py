"""List one directory in the workspace with deterministic ordering.

Entries are sorted directories-first, then by case-insensitive name, so the
model sees a stable inventory it can feed to ``fs.read_text`` or
``fs.read_file_glob``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import resolve_path, workspace_root

TOOL_ENABLED = True

MAX_ENTRIES = 5_000


class ReadDirInput(BaseModel):
    """Parameters for one directory listing."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Directory to list. Relative paths resolve against the workspace "
            "root; absolute paths are allowed only inside the workspace."
        ),
    )
    include_hidden: bool = Field(
        default=False,
        description=(
            "When false, entries whose names start with a dot are omitted. "
            "Enable to inspect hidden folders such as .git or .venv."
        ),
    )
    max_entries: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum number of entries to return (1-5000).",
    )


class DirEntry(BaseModel):
    """One directory entry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=2000, description="Entry name.")
    path: str = Field(
        min_length=1, description="Absolute normalized path of the entry."
    )
    is_dir: bool = Field(description="True when the entry is a directory.")
    is_file: bool = Field(description="True when the entry is a regular file.")
    is_symlink: bool = Field(description="True when the entry is a symbolic link.")
    size: int = Field(ge=0, description="Byte size of the entry content.")
    modified: str = Field(
        min_length=1,
        description="Last modification time as ISO 8601 with UTC offset.",
    )


class ReadDirOutput(BaseModel):
    """Stable directory listing returned to the model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    resolved_path: str = Field(
        min_length=1, description="Absolute normalized directory that was listed."
    )
    entries: list[DirEntry] = Field(
        max_length=5000, description="Sorted entries, directories first."
    )
    entry_count: int = Field(ge=0, description="Number of entries in the result.")
    truncated: bool = Field(
        description="True when more entries existed than max_entries."
    )


class ReadDirTool(BaseTool):
    """List the immediate children of one directory."""

    spec = ToolSpec(
        name="fs.read_dir",
        description=(
            "List the immediate children of a directory in the workspace and "
            "return each entry's name, absolute path, type, size and modified "
            "time. Entries are sorted directories-first then by name. Use it to "
            "explore the project layout before reading or editing files."
        ),
        version="1.0.0",
        input_model=ReadDirInput,
        output_model=ReadDirOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=15.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=8,
        tags=("fs", "directory", "list", "read"),
        recommended_before_tools=(),
    )

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        allow_outside: bool | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else workspace_root()
        self.allow_outside = allow_outside

    def execute(self, arguments: ReadDirInput) -> ReadDirOutput:
        if not isinstance(arguments, ReadDirInput):
            raise TypeError("arguments must be a ReadDirInput instance")
        path = resolve_path(
            self.base_dir, arguments.path, allow_outside=self.allow_outside
        )
        if not path.exists():
            raise FileNotFoundError(f"directory '{arguments.path}' does not exist")
        if not path.is_dir():
            raise NotADirectoryError(f"'{arguments.path}' is not a directory")

        entries: list[DirEntry] = []
        with os.scandir(path) as iterator:
            for item in iterator:
                if not arguments.include_hidden and item.name.startswith("."):
                    continue
                try:
                    stat_result = item.stat(follow_symlinks=True)
                except OSError:
                    # A broken symlink has no target to stat; fall back to the
                    # link itself so the listing never aborts mid-directory.
                    stat_result = item.stat(follow_symlinks=False)
                entries.append(
                    DirEntry(
                        name=item.name,
                        path=str(Path(item.path).resolve()),
                        is_dir=item.is_dir(follow_symlinks=True),
                        is_file=item.is_file(follow_symlinks=True),
                        is_symlink=item.is_symlink(),
                        size=stat_result.st_size,
                        modified=datetime.fromtimestamp(
                            stat_result.st_mtime
                        )
                        .astimezone()
                        .isoformat(timespec="seconds"),
                    )
                )
        entries.sort(
            key=lambda entry: (not entry.is_dir, entry.name.casefold())
        )
        truncated = len(entries) > arguments.max_entries
        bounded = entries[: arguments.max_entries]
        return ReadDirOutput(
            resolved_path=str(path),
            entries=bounded,
            entry_count=len(bounded),
            truncated=truncated,
        )


def create_tool() -> BaseTool:
    return ReadDirTool()


__all__ = [
    "TOOL_ENABLED",
    "DirEntry",
    "ReadDirInput",
    "ReadDirOutput",
    "ReadDirTool",
    "create_tool",
]