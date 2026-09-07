"""Stage, commit and push changes in the workspace git repository.

This is the delivery half of the code-change loop: it stages either an explicit
path list or all changes, creates one commit, and pushes it to the configured
remote (setting the upstream with ``-u`` when the branch has none yet). It never
attempts interactive authentication: terminal prompting is disabled, so a push
that needs credentials fails fast with an explicit error instead of hanging.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec

from ._shared import _is_within, cap_text, resolve_path, run_git, workspace_root

TOOL_ENABLED = True

MAX_PUSH_OUTPUT_CHARS = 200_000


class GitCommitPushInput(BaseModel):
    """Parameters for one stage/commit/push operation."""

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
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Commit message. Follow the repository's commit conventions.",
    )
    paths: list[str] | None = Field(
        default=None,
        max_length=1000,
        description=(
            "Optional repository-relative paths to stage. Null stages every "
            "change in the repository. Each path must stay inside the repository."
        ),
    )
    remote: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Remote name to push to; null uses 'origin'.",
    )
    branch: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Branch name to commit onto and push; null uses the current branch.",
    )
    allow_empty: bool = Field(
        default=False,
        description=(
            "When true, an empty commit is created when there is nothing to "
            "stage. When false, a no-change request returns committed=false."
        ),
    )
    push: bool = Field(
        default=True,
        description=(
            "When true, the commit is pushed to the remote after it is created. "
            "When false, only the local commit is created."
        ),
    )


class GitCommitPushOutput(BaseModel):
    """Result of one stage/commit/push operation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    repo_path: str = Field(
        min_length=1, description="Absolute repository root reported by git."
    )
    committed: bool = Field(
        description="True when a commit was created; false when nothing was staged."
    )
    empty_commit: bool = Field(
        description="True when the commit was created empty via allow_empty."
    )
    commit_hash: str | None = Field(
        description="Short hash of the created commit, or null when not committed."
    )
    changed_files: list[str] = Field(
        max_length=5000,
        description=(
            "Name-status lines of the created commit (for example 'M path.py'), "
            "or an empty list when nothing was committed."
        ),
    )
    staged_count: int = Field(
        ge=0, description="Number of files included in the commit."
    )
    push_attempted: bool = Field(
        description="True when a push was attempted after the commit."
    )
    push_succeeded: bool | None = Field(
        description="Push outcome: true, false, or null when no push was attempted."
    )
    push_output: str = Field(
        max_length=250_000,
        description="Capped combined push stdout/stderr for diagnosing failures.",
    )


class GitCommitPushTool(BaseTool):
    """Stage, commit and push one change set in the workspace repository."""

    spec = ToolSpec(
        name="git.commit_push",
        description=(
            "Stage changes (an explicit repository-relative path list or every "
            "change), create one commit with the given message, and push it to "
            "the configured remote, setting the upstream when the branch has "
            "none. Use it after fs.write_text / fs.edit_text changes are ready; "
            "check git.status first. Never include secrets in the message, and "
            "note that a push requiring credentials fails with an error instead "
            "of prompting."
        ),
        version="1.0.0",
        input_model=GitCommitPushInput,
        output_model=GitCommitPushOutput,
        side_effect="execute",
        permissions=(),
        timeout_seconds=360.0,
        idempotent=False,
        parallel_safe=False,
        max_concurrency=1,
        tags=("git", "commit", "push", "vcs", "write"),
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

    def execute(self, arguments: GitCommitPushInput) -> GitCommitPushOutput:
        if not isinstance(arguments, GitCommitPushInput):
            raise TypeError("arguments must be a GitCommitPushInput instance")
        path = (
            resolve_path(
                self.base_dir, arguments.repo_path, allow_outside=self.allow_outside
            )
            if arguments.repo_path is not None
            else self.base_dir
        )
        if not path.exists():
            raise FileNotFoundError(f"repo_path '{path}' does not exist")
        if not path.is_dir():
            raise NotADirectoryError(f"'{path}' is not a directory")

        toplevel = _toplevel(path)
        self._stage(toplevel, arguments.paths)

        staged = run_git(["diff", "--cached", "--quiet"], cwd=toplevel, timeout_seconds=30.0)
        if staged.exit_code not in (0, 1):
            raise RuntimeError(f"git diff failed: {staged.stderr.strip()}")

        if staged.exit_code == 0:
            if not arguments.allow_empty:
                return GitCommitPushOutput(
                    repo_path=toplevel,
                    committed=False,
                    empty_commit=False,
                    commit_hash=None,
                    changed_files=[],
                    staged_count=0,
                    push_attempted=False,
                    push_succeeded=None,
                    push_output="",
                )
            empty_commit = True
            commit_command = ["commit", "--allow-empty", "-m", arguments.message]
        else:
            empty_commit = False
            commit_command = ["commit", "-m", arguments.message]

        commit = run_git(commit_command, cwd=toplevel, timeout_seconds=60.0)
        if commit.exit_code != 0:
            raise RuntimeError(f"git commit failed: {commit.stderr.strip()}")
        commit_hash = _head_short(toplevel)
        changed_files = _commit_files(toplevel)

        push_attempted = False
        push_succeeded: bool | None = None
        push_output = ""
        if arguments.push:
            push_attempted = True
            push_succeeded, push_output = self._push(toplevel, arguments)

        return GitCommitPushOutput(
            repo_path=toplevel,
            committed=True,
            empty_commit=empty_commit,
            commit_hash=commit_hash,
            changed_files=changed_files,
            staged_count=len(changed_files),
            push_attempted=push_attempted,
            push_succeeded=push_succeeded,
            push_output=push_output,
        )

    @staticmethod
    def _stage(toplevel: str, paths: list[str] | None) -> None:
        if paths is None:
            result = run_git(["add", "-A"], cwd=toplevel, timeout_seconds=60.0)
            if result.exit_code != 0:
                raise RuntimeError(f"git add failed: {result.stderr.strip()}")
            return
        repository_root = Path(toplevel)
        relative_paths: list[str] = []
        for item in paths:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("each path must be a non-empty string")
            absolute = (repository_root / item).resolve()
            if not _is_within(repository_root, absolute):
                raise ValueError(f"path '{item}' escapes the repository root")
            relative = os.path.relpath(absolute, repository_root).replace(os.sep, "/")
            relative_paths.append(relative)
        result = run_git(
            ["add", "--", *relative_paths],
            cwd=toplevel,
            timeout_seconds=60.0,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"git add failed: {result.stderr.strip()}")

    @staticmethod
    def _push(toplevel: str, arguments: GitCommitPushInput) -> tuple[bool, str]:
        remote = arguments.remote or "origin"
        branch = arguments.branch
        if branch is None:
            current = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=toplevel, timeout_seconds=15.0)
            if current.exit_code != 0:
                raise RuntimeError(f"cannot resolve branch: {current.stderr.strip()}")
            branch = current.stdout.strip()
        if branch == "HEAD":
            raise RuntimeError("cannot push from a detached HEAD")

        upstream = run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=toplevel,
            timeout_seconds=15.0,
        )
        has_upstream = upstream.exit_code == 0 and upstream.stdout.strip()
        command = ["push", "-u", remote, branch] if not has_upstream else ["push", remote, branch]
        result = run_git(command, cwd=toplevel, timeout_seconds=120.0)
        combined = f"{result.stdout}\n{result.stderr}".strip()
        capped, _ = cap_text(combined, MAX_PUSH_OUTPUT_CHARS)
        return result.exit_code == 0, capped


def _toplevel(path: Path) -> str:
    result = run_git(["rev-parse", "--show-toplevel"], cwd=path, timeout_seconds=15.0)
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


def _commit_files(toplevel: str, *, limit: int = 5_000) -> list[str]:
    result = run_git(
        ["show", "--name-status", "--format=", "HEAD"],
        cwd=toplevel,
        timeout_seconds=30.0,
    )
    if result.exit_code != 0:
        return []
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and line[0] in "MADRCUT"
    ][:limit]


def create_tool() -> BaseTool:
    return GitCommitPushTool()


__all__ = [
    "TOOL_ENABLED",
    "GitCommitPushInput",
    "GitCommitPushOutput",
    "GitCommitPushTool",
    "create_tool",
]