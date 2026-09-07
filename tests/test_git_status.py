from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from core import ToolCall, ToolExecutionManager, ToolRegistry
from tool.git_status import TOOL_ENABLED, GitStatusInput, GitStatusTool, create_tool


def run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={"GIT_TERMINAL_PROMPT": "0", "PATH": _path_env()},
    )


def _path_env() -> str:
    import os

    return os.environ.get("PATH", "")


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


def test_git_status_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, GitStatusTool)


def test_git_status_reports_clean_repo(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "tracked.txt", "v1", "initial")
    tool = GitStatusTool(base_dir=tmp_path)

    output = tool.execute(GitStatusInput(repo_path="repo"))

    assert output.clean is True
    assert output.branch == "main"
    assert output.head_short
    assert output.staged == []
    assert output.unstaged == []
    assert output.untracked == []
    assert output.conflicted == []
    assert os.path.normcase(output.repo_path) == os.path.normcase(str(repo.resolve()))


def test_git_status_classifies_changes(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "tracked.txt", "v1", "initial")
    (repo / "tracked.txt").write_text("v2", encoding="utf-8")
    (repo / "staged.txt").write_text("s", encoding="utf-8")
    assert run_git(["add", "staged.txt"], cwd=repo).returncode == 0
    (repo / "new.txt").write_text("n", encoding="utf-8")
    tool = GitStatusTool(base_dir=tmp_path)

    output = tool.execute(GitStatusInput(repo_path="repo"))

    assert output.clean is False
    assert output.unstaged == ["tracked.txt"]
    assert output.staged == ["staged.txt"]
    assert output.untracked == ["new.txt"]
    assert output.conflicted == []


def test_git_status_detached_head(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "tracked.txt", "v1", "initial")
    assert run_git(["checkout", "--detach"], cwd=repo).returncode == 0
    tool = GitStatusTool(base_dir=tmp_path)

    output = tool.execute(GitStatusInput(repo_path="repo"))

    assert output.branch is None
    assert output.head_short


def test_git_status_reports_not_a_repository(tmp_path):
    tool = GitStatusTool(base_dir=tmp_path)

    with pytest.raises(RuntimeError, match="not inside a git repository"):
        tool.execute(GitStatusInput(repo_path="."))


def test_git_status_rejects_paths_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside_repo"
    outside.mkdir()
    assert run_git(["init", "-b", "main"], cwd=outside).returncode == 0
    tool = GitStatusTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tool.execute(GitStatusInput(repo_path=str(outside)))


def test_git_status_validation_is_strict():
    with pytest.raises(ValidationError):
        GitStatusInput.model_validate({"repo_path": 7}, strict=True)
    with pytest.raises(ValidationError):
        GitStatusInput.model_validate({"repo_path": "x", "extra": 1}, strict=True)


@pytest.mark.asyncio
async def test_git_status_runs_without_side_effect_confirmation(tmp_path):
    repo = init_repo(tmp_path)
    commit_file(repo, "tracked.txt", "v1", "initial")
    tool = GitStatusTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    call = ToolCall(
        call_id="status-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments={"repo_path": "repo"},
    )
    manager = ToolExecutionManager(registry)

    batch = await manager.execute_batch([call])

    assert batch.results[0].ok
    assert batch.results[0].data["clean"] is True
    assert batch.results[0].data["branch"] == "main"


def test_git_env_preserves_credential_helpers():
    """GIT_ASKPASS / GIT_CONFIG_NOSYSTEM must stay unset so credential
    managers (registered at the system level on Windows) can authenticate."""

    from tool._shared import git_env

    env = git_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_ASKPASS" not in env
    assert "GIT_CONFIG_NOSYSTEM" not in env


def test_git_status_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("git.status")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("git.status", version="1.0.0")