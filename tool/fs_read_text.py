"""Read a text file from the workspace with bounded, deterministic output.

The tool enforces a workspace sandbox by default: relative paths resolve
against the workspace root and absolute paths are rejected when they escape it,
so a prompt-injected model can never exfiltrate files outside the project.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import (
    DEFAULT_MAX_OUTPUT_CHARS,
    cap_text,
    read_text_file,
    resolve_path,
    workspace_root,
)

TOOL_ENABLED = True


class ReadTextInput(BaseModel):
    """Parameters for one bounded text-file read."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Path of the text file to read. Relative paths resolve against the "
            "workspace root; absolute paths are allowed only inside the workspace."
        ),
    )
    encoding: str = Field(
        default="utf-8",
        min_length=1,
        max_length=32,
        description=(
            "Text encoding of the file, for example 'utf-8', 'utf-16', 'gbk' "
            "or 'latin-1'. Must match the file content."
        ),
    )
    max_chars: int | None = Field(
        default=None,
        ge=1,
        le=2_000_000,
        description=(
            "Maximum number of characters to return. Null means the tool "
            "default (200,000). Larger files are truncated at this boundary."
        ),
    )


class ReadTextOutput(BaseModel):
    """Stable, bounded text content returned to the runtime and model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    resolved_path: str = Field(
        min_length=1, description="Absolute normalized path that was read."
    )
    content: str = Field(
        max_length=2_001_000,
        description="Decoded text, truncated to max_chars when the file is larger.",
    )
    character_count: int = Field(
        ge=0, description="Character count of the full decoded file."
    )
    truncated: bool = Field(
        description="True when content was cut to fit the requested max_chars."
    )
    encoding: str = Field(
        min_length=1, description="Canonical codec name used to decode the file."
    )


class ReadTextTool(BaseTool):
    """Read one text file without modifying anything."""

    spec = ToolSpec(
        name="fs.read_text",
        description=(
            "Read a UTF-8 (or other explicitly named encoding) text file from the "
            "workspace and return its content with a bounded character count. Use "
            "it to inspect source files, configuration, or logs before editing. "
            "It cannot read binary files and never writes anything."
        ),
        version="1.0.0",
        input_model=ReadTextInput,
        output_model=ReadTextOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=15.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=8,
        tags=("fs", "file", "read", "text"),
        recommended_before_tools=("fs.read_dir", "fs.read_file_glob"),
    )

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        allow_outside: bool | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else workspace_root()
        self.allow_outside = allow_outside

    def execute(self, arguments: ReadTextInput) -> ReadTextOutput:
        if not isinstance(arguments, ReadTextInput):
            raise TypeError("arguments must be a ReadTextInput instance")
        path = resolve_path(
            self.base_dir, arguments.path, allow_outside=self.allow_outside
        )
        if not path.exists():
            raise FileNotFoundError(f"file '{arguments.path}' does not exist")
        if not path.is_file():
            raise IsADirectoryError(f"'{arguments.path}' is not a file")
        text, canonical = read_text_file(path, arguments.encoding)
        max_chars = (
            arguments.max_chars if arguments.max_chars is not None else DEFAULT_MAX_OUTPUT_CHARS
        )
        content, truncated = cap_text(text, max_chars)
        return ReadTextOutput(
            resolved_path=str(path),
            content=content,
            character_count=len(text),
            truncated=truncated,
            encoding=canonical,
        )


def create_tool() -> BaseTool:
    return ReadTextTool()


__all__ = [
    "TOOL_ENABLED",
    "ReadTextInput",
    "ReadTextOutput",
    "ReadTextTool",
    "create_tool",
]