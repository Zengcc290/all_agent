"""Run one shell command with a hard deadline and bounded captured output.

The command runs through the platform shell (``cmd.exe`` on Windows, ``/bin/sh``
on POSIX) inside an optional working directory anchored to the workspace. On
timeout the whole process tree is terminated and the partial output is still
returned. Because this tool can execute arbitrary code, it is registered with
``side_effect="execute"`` and serialized to one concurrent call.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import resolve_path, run_process, workspace_root

TOOL_ENABLED = True

MAX_COMMAND_CHARS = 20_000
MAX_TIMEOUT_SECONDS = 300


class ShellRunInput(BaseModel):
    """Parameters for one bounded shell command."""

    model_config = ConfigDict(extra="forbid", strict=True)

    command: str = Field(
        min_length=1,
        max_length=20_000,
        description=(
            "Complete shell command line, for example 'pytest -q' or "
            "'git status'. Runs in the platform default shell: cmd.exe on "
            "Windows, /bin/sh on POSIX. Do not include secrets in the command; "
            "use environment variables or configuration files instead."
        ),
    )
    cwd: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description=(
            "Working directory for the command. Relative paths resolve against "
            "the workspace root; null uses the workspace root itself."
        ),
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        description=(
            "Hard deadline in whole seconds (1-300). On expiry the process tree "
            "is terminated and partial output is returned with timed_out=true."
        ),
    )


class ShellRunOutput(BaseModel):
    """Normalized result of one shell command."""

    model_config = ConfigDict(extra="forbid", strict=True)

    exit_code: int = Field(
        ge=-1,
        description=(
            "Process exit code; -1 means the command was terminated by the "
            "timeout, otherwise the real exit code (0 = success)."
        ),
    )
    stdout: str = Field(
        max_length=250_000,
        description="Captured standard output, truncated when very large.",
    )
    stderr: str = Field(
        max_length=250_000,
        description="Captured standard error, truncated when very large.",
    )
    timed_out: bool = Field(
        description="True when the deadline expired and the process was killed."
    )
    duration_seconds: float = Field(
        ge=0.0, description="Wall-clock duration of the command in seconds."
    )
    stdout_truncated: bool = Field(
        description="True when stdout was cut to fit the output bound."
    )
    stderr_truncated: bool = Field(
        description="True when stderr was cut to fit the output bound."
    )
    resolved_cwd: str = Field(
        min_length=1, description="Absolute working directory that was used."
    )


class ShellRunTool(BaseTool):
    """Execute one shell command and capture its output."""

    spec = ToolSpec(
        name="shell.run",
        description=(
            "Run one shell command and return its exit code plus captured stdout "
            "and stderr. Use it for anything outside the file tools: running "
            "tests, invoking git, inspecting the environment, or driving local "
            "tools. The command runs with the workspace root as its default "
            "working directory and is terminated if it exceeds the timeout. "
            "Only use this for commands that do not require interactive input."
        ),
        version="1.0.0",
        input_model=ShellRunInput,
        output_model=ShellRunOutput,
        side_effect="execute",
        permissions=(),
        timeout_seconds=330.0,
        idempotent=False,
        parallel_safe=False,
        max_concurrency=1,
        tags=("shell", "command", "process", "run", "execute"),
        recommended_before_tools=("git.status",),
    )

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        allow_outside: bool | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else workspace_root()
        self.allow_outside = allow_outside

    def execute(self, arguments: ShellRunInput) -> ShellRunOutput:
        if not isinstance(arguments, ShellRunInput):
            raise TypeError("arguments must be a ShellRunInput instance")
        if arguments.cwd is not None:
            cwd = resolve_path(
                self.base_dir, arguments.cwd, allow_outside=self.allow_outside
            )
            if not cwd.exists():
                raise FileNotFoundError(f"cwd '{arguments.cwd}' does not exist")
            if not cwd.is_dir():
                raise NotADirectoryError(f"'{arguments.cwd}' is not a directory")
        else:
            cwd = self.base_dir
        outcome = run_process(
            arguments.command,
            cwd=cwd,
            timeout_seconds=float(arguments.timeout_seconds),
            shell=True,
        )
        return ShellRunOutput(
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            timed_out=outcome.timed_out,
            duration_seconds=outcome.duration_seconds,
            stdout_truncated=outcome.stdout_truncated,
            stderr_truncated=outcome.stderr_truncated,
            resolved_cwd=str(cwd),
        )


def create_tool() -> BaseTool:
    return ShellRunTool()


__all__ = [
    "TOOL_ENABLED",
    "ShellRunInput",
    "ShellRunOutput",
    "ShellRunTool",
    "create_tool",
]