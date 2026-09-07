"""Write text to one workspace file using an atomic replace.

The write is staged in a temporary file next to the target and moved into
place with ``os.replace``, so a crash or concurrent read never observes a
half-written file. This tool is a registered side-effecting write: the runtime
requires a generation-bound side-effect confirmation before it will execute.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import atomic_write_text, normalize_encoding, resolve_path, workspace_root

TOOL_ENABLED = True

MAX_CONTENT_CHARS = 1_000_000


class WriteTextInput(BaseModel):
    """Parameters for one atomic text-file write."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Path of the file to write. Relative paths resolve against the "
            "workspace root; absolute paths are allowed only inside the workspace."
        ),
    )
    content: str = Field(
        max_length=1_000_000,
        min_length=0,
        description="Full text content to write, replacing any existing content.",
    )
    encoding: str = Field(
        default="utf-8",
        min_length=1,
        max_length=32,
        description="Text encoding used to encode the content, for example 'utf-8'.",
    )
    create_parents: bool = Field(
        default=True,
        description=(
            "When true, missing parent directories are created automatically. "
            "When false, a missing parent directory is an error."
        ),
    )
    overwrite: bool = Field(
        default=True,
        description=(
            "When true, an existing file is overwritten. When false, writing to "
            "an existing file is an error (create-only semantics)."
        ),
    )


class WriteTextOutput(BaseModel):
    """Confirmation of one completed file write."""

    model_config = ConfigDict(extra="forbid", strict=True)

    resolved_path: str = Field(
        min_length=1, description="Absolute normalized path that was written."
    )
    existed: bool = Field(
        description="True when the target file already existed before the write."
    )
    created: bool = Field(
        description="True when the target file did not exist before the write."
    )
    bytes_written: int = Field(ge=0, description="Number of bytes written.")
    character_count: int = Field(ge=0, description="Number of characters written.")
    encoding: str = Field(
        min_length=1, description="Canonical codec name used to encode the content."
    )


class WriteTextTool(BaseTool):
    """Create or replace one text file inside the workspace."""

    spec = ToolSpec(
        name="fs.write_text",
        description=(
            "Write full text content to one file in the workspace, creating the "
            "file or replacing its existing content. The write is atomic and the "
            "target path must stay inside the workspace. Use it to create new "
            "files or to rewrite files whose content is known; use fs.edit_text "
            "for a targeted replacement inside an existing file."
        ),
        version="1.0.0",
        input_model=WriteTextInput,
        output_model=WriteTextOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=15.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=4,
        tags=("fs", "file", "write", "text", "create"),
        recommended_before_tools=("fs.read_text", "fs.read_dir"),
    )

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        allow_outside: bool | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else workspace_root()
        self.allow_outside = allow_outside

    def execute(self, arguments: WriteTextInput) -> WriteTextOutput:
        if not isinstance(arguments, WriteTextInput):
            raise TypeError("arguments must be a WriteTextInput instance")
        path = resolve_path(
            self.base_dir, arguments.path, allow_outside=self.allow_outside
        )
        existed = path.exists()
        if existed and not arguments.overwrite:
            raise FileExistsError(f"file '{arguments.path}' already exists")
        bytes_written = atomic_write_text(
            path,
            arguments.content,
            arguments.encoding,
            create_parents=arguments.create_parents,
        )
        return WriteTextOutput(
            resolved_path=str(path),
            existed=existed,
            created=not existed,
            bytes_written=bytes_written,
            character_count=len(arguments.content),
            encoding=normalize_encoding(arguments.encoding),
        )


def create_tool() -> BaseTool:
    return WriteTextTool()


__all__ = [
    "TOOL_ENABLED",
    "WriteTextInput",
    "WriteTextOutput",
    "WriteTextTool",
    "create_tool",
]