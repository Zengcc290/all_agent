"""Read the working-tree status of a git repository without modifying it.

Parses ``git status --porcelain=v1 -b`` into staged, unstaged, untracked and
conflicted path lists so the model can decide what to commit next. Terminal
prompting is disabled so the tool can never hang on credentials.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import resolve_path, run_git, workspace_root

TOOL_ENABLED = True

MAX_STATUS_FILES = 2_000


class GitStatusInput(BaseModel):
    """Parameters for one git status read."""

    model_config = ConfigDict(extra="forbid", strict=True)

    repo_path: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description=(
            "Path of the git repository. Relative paths resolve against the "
            "workspace root; null uses the workspace root itself."
        ),
    )


class GitStatusOutput(BaseModel):
    """Normalized working-tree status."""

    model_config = ConfigDict(extra="forbid", strict=True)

    repo_path: str = Field(
        min_length=1, description="Absolute repository root reported by git."
    )
    branch: str | None = Field(
        description="Current branch name, or null when HEAD is detached."
    )
    head_short: str | None = Field(
        description="Short HEAD commit hash, or null in a repository with no commits."
    )
    clean: bool = Field(description="True when there is nothing to commit.")
    staged: list[str] = Field(
        max_length=2000,
        description="Paths already added to the index, in git status order.",
    )
    unstaged: list[str] = Field(
        max_length=2000,
        description="Paths modified or deleted in the working tree, not staged.",
    )
    untracked: list[str] = Field(
        max_length=2000,
        description="Paths not tracked by git (for example new files).",
    )
    conflicted: list[str] = Field(
        max_length=2000,
        description="Paths with unresolved merge/rebase conflicts.",
    )
    short_status: list[str] = Field(
        max_length=2000,
        description="Raw porcelain v1 status lines, including the branch header.",
    )
    truncated: bool = Field(
        description="True when a path list was cut because the change set was large."
    )


class GitStatusTool(BaseTool):
    """Report the current git working-tree status."""

    spec = ToolSpec(
        name="git.status",
        description=(
            "Read the git working-tree status of the workspace repository and "
            "return the current branch, head commit and the staged, unstaged, "
            "untracked and conflicted paths. Use it before git.commit_push to "
            "decide what to commit, and after changes to confirm the result."
        ),
        version="1.0.0",
        input_model=GitStatusInput,
        output_model=GitStatusOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=90.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=4,
        tags=("git", "status", "vcs", "read"),
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

    def execute(self, arguments: GitStatusInput) -> GitStatusOutput:
        if not isinstance(arguments, GitStatusInput):
            raise TypeError("arguments must be a GitStatusInput instance")
        if arguments.repo_path is not None:
            path = resolve_path(
                self.base_dir, arguments.repo_path, allow_outside=self.allow_outside
            )
        else:
            path = self.base_dir
        if not path.exists():
            raise FileNotFoundError(f"repo_path '{path}' does not exist")
        if not path.is_dir():
            raise NotADirectoryError(f"'{path}' is not a directory")

        toplevel = _toplevel(path)
        status = run_git(
            ["status", "--porcelain=v1", "-b"],
            cwd=toplevel,
            timeout_seconds=30.0,
        )
        if status.exit_code != 0:
            raise RuntimeError(f"git status failed: {status.stderr.strip()}")
        lines = status.stdout.splitlines()

        branch: str | None = None
        depends = None
        if lines and lines[0].startswith("## "):
            header = lines[0][3:]
            if not header.startswith("HEAD (no branch)"):
                branch, _, depends = header.partition("...")
                if depends:
                    depends = depends.split(" ", 1)[0]
            lines = lines[1:]

        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        conflicted: list[str] = []
        for line in lines:
            if len(line) < 4 or line[0] not in " MADRCTU?" or line[1] not in " MADRCTU?":
                continue
            index_code, worktree_code = line[0], line[1]
            file_path = line[3:]
            code = index_code + worktree_code
            if "U" in code or code in {"AA", "DD"}:
                conflicted.append(file_path)
            elif code == "??":
                untracked.append(file_path)
            elif index_code != " " and index_code != "?":
                staged.append(file_path)
            elif worktree_code != " " and worktree_code != "?":
                unstaged.append(file_path)

        bounded, truncated = _bound_lists(
            staged, unstaged, untracked, conflicted, lines
        )
        head_short = _head_short(toplevel)
        clean = not any(
            (
                staged,
                unstaged,
                untracked,
                conflicted,
            )
        )
        return GitStatusOutput(
            repo_path=toplevel,
            branch=branch,
            head_short=head_short,
            clean=clean,
            staged=bounded[0],
            unstaged=bounded[1],
            untracked=bounded[2],
            conflicted=bounded[3],
            short_status=bounded[4],
            truncated=truncated,
        )


def _toplevel(path: Path) -> str:
    result = run_git(
        ["rev-parse", "--show-toplevel"],
        cwd=path,
        timeout_seconds=15.0,
    )
    if result.exit_code != 0:
        raise RuntimeError(
            f"'{path}' is not inside a git repository: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _head_short(toplevel: str) -> str | None:
    result = run_git(["rev-parse", "--short", "HEAD"], cwd=toplevel, timeout_seconds=15.0)
    if result.exit_code != 0:
        return None
    return result.stdout.strip() or None


def _bound_lists(
    staged: list[str],
    unstaged: list[str],
    untracked: list[str],
    conflicted: list[str],
    raw_lines: list[str],
) -> tuple[tuple[list[str], list[str], list[str], list[str], list[str]], bool]:
    truncated = any(
        len(items) > MAX_STATUS_FILES
        for items in (staged, unstaged, untracked, conflicted)
    ) or len(raw_lines) > MAX_STATUS_FILES
    return (
        (
            staged[:MAX_STATUS_FILES],
            unstaged[:MAX_STATUS_FILES],
            untracked[:MAX_STATUS_FILES],
            conflicted[:MAX_STATUS_FILES],
            raw_lines[:MAX_STATUS_FILES],
        ),
        truncated,
    )


def create_tool() -> BaseTool:
    return GitStatusTool()


__all__ = [
    "TOOL_ENABLED",
    "GitStatusInput",
    "GitStatusOutput",
    "GitStatusTool",
    "create_tool",
]