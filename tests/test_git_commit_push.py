from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from core import ExecutionContext, ToolCall, ToolExecutionManager, ToolRegistry
from tool.git_commit_push import (
    TOOL_ENABLED,
    GitCommitPushInput,
    GitCommitPushTool,
    create_tool,
)


def run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={"GIT_TERMINAL_PROMPT": "0", "PATH": os.environ.get("PATH", "")},
    )


def init_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    assert run_git(["init", "-b", "main"], cwd=repo).returncode == 0
    assert run_git(["config", "user.name", "Test Agent"], cwd=repo).returncode == 0
    assert run_git(["config", "user.email", "test@example.com"], cwd=repo).returncode == 0
    assert run_git(["config", "core.autocrlf", "false"], cwd=repo).returncode == 0
    return repo


def commit_file(repo: Path, name: str, content: str, message: str) -> None:
    (repo / name).write_text(content, encoding="utf-8")
    assert run_git(["add", name], cwd=repo).returncode == 0
    assert run_git(["commit", "-m", message], cwd=repo).returncode == 0


def init_bare_remote(tmp_path: Path, name: str = "remote.git") -> Path:
    remote = tmp_path / name
    # -b main keeps the bare repo's HEAD on the branch that tests push.
    assert run_git(["init", "--bare", "-b", "main", str(remote)]).returncode == 0
    return remote


def test_commit_push_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, GitCommitPushTool)


def test_commit_push_commits_and_pushes(tmp_path):
    repo = init_repo(tmp_path)
    remote = init_bare_remote(tmp_path)
    assert run_git(["remote", "add", "origin", str(remote)], cwd=repo).returncode == 0
    commit_file(repo, "a.txt", "a", "initial")
    assert (
        run_git(["push", "-u", "origin", "main"], cwd=repo).returncode == 0
    )

    (repo / "b.txt").write_text("b", encoding="utf-8")
    tool = GitCommitPushTool(base_dir=tmp_path)
    output = tool.execute(
        GitCommitPushInput(repo_path="repo", message="add b", remote="origin", branch="main")
    )

    assert output.committed is True
    assert output.empty_commit is False
    assert output.commit_hash
    assert output.push_attempted is True
    assert output.push_succeeded is True
    assert any("b.txt" in line for line in output.changed_files)

    local_head = run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    remote_head = run_git(["-C", str(remote), "rev-parse", "HEAD"]).stdout.strip()
    assert local_head == remote_head


def test_commit_push_sets_upstream_when_missing(tmp_path):
    repo = init_repo(tmp_path)
    remote = init_bare_remote(tmp_path)
    assert run_git(["remote", "add", "origin", str(remote)], cwd=repo).returncode == 0
    commit_file(repo, "a.txt", "a", "initial")
    (repo / "b.txt").write_text("b", encoding="utf-8")

    tool = GitCommitPushTool(base_dir=tmp_path)
    output = tool.execute(GitCommitPushInput(repo_path="repo", message="first push"))

    assert output.committed is True
    assert output.push_attempted is True
    assert output.push_succeeded is True
    upstream = run_git(["rev-parse", "--abbrev-ref", "@{u}"], cwd=repo)
    assert upstream.returncode == 0
    assert upstream.stdout.strip() == "origin/main"


def test_commit_push_nothing_to_commit_returns_not_committed(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "a.txt", "a", "initial")
    tool = GitCommitPushTool(base_dir=tmp_path)

    output = tool.execute(GitCommitPushInput(repo_path="repo", message="noop"))

    assert output.committed is False
    assert output.commit_hash is None
    assert output.push_attempted is False
    assert output.push_succeeded is None
    assert output.changed_files == []


def test_commit_push_allow_empty_creates_commit(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "a.txt", "a", "initial")
    tool = GitCommitPushTool(base_dir=tmp_path)

    output = tool.execute(
        GitCommitPushInput(
            repo_path="repo", message="empty", allow_empty=True, push=False
        )
    )

    assert output.committed is True
    assert output.empty_commit is True
    assert output.commit_hash
    assert output.push_attempted is False


def test_commit_push_stages_only_specified_paths(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "base.txt", "b", "initial")
    (repo / "one.txt").write_text("1", encoding="utf-8")
    (repo / "two.txt").write_text("2", encoding="utf-8")
    tool = GitCommitPushTool(base_dir=tmp_path)

    output = tool.execute(
        GitCommitPushInput(
            repo_path="repo", message="only one", paths=["one.txt"], push=False
        )
    )

    assert output.committed is True
    files = " ".join(output.changed_files)
    assert "one.txt" in files
    assert "two.txt" not in files
    status = run_git(["status", "--porcelain"], cwd=repo).stdout
    assert "two.txt" in status


def test_commit_push_rejects_paths_escaping_repository(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "a.txt", "a", "initial")
    tool = GitCommitPushTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="escapes the repository"):
        tool.execute(
            GitCommitPushInput(
                repo_path="repo", message="x", paths=["../outside.txt"], push=False
            )
        )


def test_commit_push_reports_not_a_repository(tmp_path):
    tool = GitCommitPushTool(base_dir=tmp_path)

    with pytest.raises(RuntimeError, match="not inside a git repository"):
        tool.execute(GitCommitPushInput(repo_path=".", message="x", push=False))


def test_commit_push_validation_is_strict():
    with pytest.raises(ValidationError):
        GitCommitPushInput.model_validate({"message": ""}, strict=True)
    with pytest.raises(ValidationError):
        GitCommitPushInput.model_validate(
            {"message": "x", "paths": ["ok", 7]}, strict=True
        )
    with pytest.raises(ValidationError):
        GitCommitPushInput.model_validate(
            {"message": "x", "allow_empty": "no"}, strict=True
        )


@pytest.mark.asyncio
async def test_commit_push_requires_side_effect_confirmation(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "a.txt", "a", "initial")
    (repo / "b.txt").write_text("b", encoding="utf-8")
    tool = GitCommitPushTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    arguments = {
        "repo_path": "repo",
        "message": "runtime commit",
        "paths": None,
        "remote": None,
        "branch": None,
        "allow_empty": False,
        "push": False,
    }
    call = ToolCall(
        call_id="commit-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments=arguments,
    )
    manager = ToolExecutionManager(registry)

    denied = await manager.execute_batch([call])
    assert denied.results[0].error.code == "CONFIRMATION_REQUIRED"
    assert run_git(["log", "--oneline"], cwd=repo).stdout.count("\n") == 1

    context = ExecutionContext(
        confirmed_side_effects=frozenset(
            {registry.confirmation_key(tool.spec.name)}
        ),
    )
    allowed = await manager.execute_batch([call], context)
    assert allowed.results[0].ok
    assert allowed.results[0].data["committed"] is True
    assert run_git(["log", "--oneline"], cwd=repo).stdout.count("\n") == 2


def test_commit_push_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("git.commit_push")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("git.commit_push", version="1.0.0")