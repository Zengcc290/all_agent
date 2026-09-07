"""Apply a literal text replacement inside one workspace file.

Reading, replacement and the atomic write happen inside a single execute call,
so the file cannot drift between a prior read and the edit. The old string must
be unique unless ``replace_all`` is enabled, which prevents accidental multiple
rewrites from a too-short search term.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import atomic_write_text, read_text_file, resolve_path, workspace_root

TOOL_ENABLED = True

MAX_OLD_CHARS = 100_000
MAX_NEW_CHARS = 1_000_000


class EditTextInput(BaseModel):
    """Parameters for one literal text replacement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Path of the file to edit. Relative paths resolve against the "
            "workspace root; absolute paths are allowed only inside the workspace."
        ),
    )
    old_string: str = Field(
        min_length=1,
        max_length=100_000,
        description=(
            "Exact literal text to find in the file. It must occur exactly once "
            "unless replace_all is enabled."
        ),
    )
    new_string: str = Field(
        max_length=1_000_000,
        description=(
            "Replacement text. An empty string deletes the old_string; newlines "
            "and indentation are preserved literally."
        ),
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "When true, every occurrence of old_string is replaced. When false, "
            "a non-unique old_string is an error."
        ),
    )
    encoding: str = Field(
        default="utf-8",
        min_length=1,
        max_length=32,
        description="Text encoding of the file, for example 'utf-8'.",
    )


class EditTextOutput(BaseModel):
    """Confirmation of one completed text replacement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    resolved_path: str = Field(
        min_length=1, description="Absolute normalized path that was edited."
    )
    replacements: int = Field(
        ge=0, description="Number of old_string occurrences that were replaced."
    )
    changed: bool = Field(description="Always true for a successful edit.")
    character_count: int = Field(
        ge=0, description="Character count of the file after the edit."
    )
    byte_count: int = Field(
        ge=0, description="Encoded byte count of the file after the edit."
    )


class EditTextTool(BaseTool):
    """Replace one literal block inside an existing workspace file."""

    spec = ToolSpec(
        name="fs.edit_text",
        description=(
            "Replace one exact literal string inside an existing UTF-8 text file "
            "in the workspace and write the result back atomically. Use it for "
            "targeted changes such as fixing a constant, a line or a block; use "
            "fs.write_text to create files or replace whole content. The old "
            "string must match exactly once unless replace_all is true."
        ),
        version="1.0.0",
        input_model=EditTextInput,
        output_model=EditTextOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=15.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=4,
        tags=("fs", "file", "edit", "replace", "text"),
        recommended_before_tools=("fs.read_text",),
    )

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        allow_outside: bool | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else workspace_root()
        self.allow_outside = allow_outside

    def execute(self, arguments: EditTextInput) -> EditTextOutput:
        if not isinstance(arguments, EditTextInput):
            raise TypeError("arguments must be an EditTextInput instance")
        path = resolve_path(
            self.base_dir, arguments.path, allow_outside=self.allow_outside
        )
        if not path.exists():
            raise FileNotFoundError(f"file '{arguments.path}' does not exist")
        if not path.is_file():
            raise IsADirectoryError(f"'{arguments.path}' is not a file")
        text, canonical = read_text_file(path, arguments.encoding)

        occurrences = text.count(arguments.old_string)
        if occurrences == 0:
            raise ValueError(
                "old_string was not found in the file; read the file first and "
                "match it exactly"
            )
        if occurrences > 1 and not arguments.replace_all:
            raise ValueError(
                f"old_string occurs {occurrences} times; enable replace_all or "
                "make old_string unique"
            )
        if arguments.replace_all:
            replacement = text.replace(arguments.old_string, arguments.new_string)
            replacements = occurrences
        else:
            replacement = text.replace(arguments.old_string, arguments.new_string, 1)
            replacements = 1

        byte_count = atomic_write_text(
            path,
            replacement,
            canonical,
            create_parents=False,
        )
        return EditTextOutput(
            resolved_path=str(path),
            replacements=replacements,
            changed=True,
            character_count=len(replacement),
            byte_count=byte_count,
        )


def create_tool() -> BaseTool:
    return EditTextTool()


__all__ = [
    "TOOL_ENABLED",
    "EditTextInput",
    "EditTextOutput",
    "EditTextTool",
    "create_tool",
]