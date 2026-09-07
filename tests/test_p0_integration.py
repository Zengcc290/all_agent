"""End-to-end integration test for the P0 code-change loop.

Binds the P0 tools to one scratch workspace like a real agent deployment,
executes them through the real runtime chain (ToolCall -> confirmation gate ->
execute) and walks the complete closed loop: write -> edit -> read verify ->
git status -> commit+push -> clean status. Also covers depends_on batching and
the side-effect confirmation gate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core import (
    ExecutionContext,
    ToolCall,
    ToolExecutionManager,
    ToolRegistry,
)
from tool.fs_edit_text import EditTextTool
from tool.fs_read_text import ReadTextTool
from tool.fs_write_text import WriteTextTool
from tool.git_commit_push import GitCommitPushTool
from tool.git_status import GitStatusTool


def run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={"GIT_TERMINAL_PROMPT": "0", "PATH": os.environ.get("PATH", "")},
    )


def init_repo(tmp_path: Path, name: str = "repo", do_configure: bool = True) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    assert run_git(["init", "-b", "main"], cwd=repo).returncode == 0
    if do_configure:
        assert run_git(["config", "user.name", "Test Agent"], cwd=repo).returncode == 0
        assert run_git(["config", "user.email", "test@example.com"], cwd=repo).returncode == 0
        assert run_git(["config", "core.autocrlf", "false"], cwd=repo).returncode == 0
    return repo


def init_bare_remote(tmp_path: Path, name: str = "remote.git") -> Path:
    remote = tmp_path / name
    assert run_git(["init", "--bare", "-b", "main", str(remote)]).returncode == 0
    return remote


class P0Loop:
    """One workspace's P0 tool instances bound to the real runtime chain."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.registry = ToolRegistry()
        for tool in (
            ReadTextTool(base_dir=base_dir),
            WriteTextTool(base_dir=base_dir),
            EditTextTool(base_dir=base_dir),
            GitStatusTool(base_dir=base_dir),
            GitCommitPushTool(base_dir=base_dir),
        ):
            self.registry.register(tool)
        self.manager = ToolExecutionManager(self.registry)
        self._counter = 0

    def _call(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        confirm: bool,
        depends_on: list[str] | None = None,
    ) -> ToolCall:
        tool, generation = self.registry.resolve(name)
        return ToolCall(
            call_id=call_id,
            tool_name=name,
            schema_version=tool.spec.version,
            schema_hash=tool.spec.schema_hash,
            registry_generation=generation,
            arguments=arguments,
            depends_on=depends_on or [],
        )

    async def call(self, name: str, arguments: dict, *, confirm: bool = False):
        self._counter += 1
        context = ExecutionContext(
            confirmed_side_effects=frozenset(
                {self.registry.confirmation_key(name)}
            )
            if confirm
            else frozenset()
        )
        batch = await self.manager.execute_batch(
            [self._call(f"loop-{self._counter}", name, arguments, confirm)], context
        )
        return batch.results[0]

    async def batch(self, calls: list[tuple[str, str, dict, bool, list[str]]]):
        confirmed: set[str] = set()
        built = []
        for call_id, name, arguments, needs_confirmation, depends_on in calls:
            built.append(
                self._call(call_id, name, arguments, needs_confirmation, depends_on)
            )
            if needs_confirmation:
                confirmed.add(self.registry.confirmation_key(name))
        context = ExecutionContext(confirmed_side_effects=frozenset(confirmed))
        outcomes = await self.manager.execute_batch(built, context)
        return list(outcomes.results)


@pytest.mark.asyncio
async def test_complete_code_change_loop_with_push(tmp_path):
    repo = init_repo(tmp_path)
    remote = init_bare_remote(tmp_path)
    assert run_git(["remote", "add", "origin", str(remote)], cwd=repo).returncode == 0
    loop = P0Loop(repo)

    created = await loop.call(
        "fs.write_text",
        {
            "path": "greet.py",
            "content": "def greet(name):\n    return f'hello {name}'\n",
            "encoding": "utf-8",
            "create_parents": True,
            "overwrite": True,
        },
        confirm=True,
    )
    assert created.ok and created.data["created"] is True
    assert (repo / "greet.py").exists()

    edited = await loop.call(
        "fs.edit_text",
        {
            "path": "greet.py",
            "old_string": "f'hello {name}'",
            "new_string": "f'Hi {name}!'",
            "replace_all": False,
            "encoding": "utf-8",
        },
        confirm=True,
    )
    assert edited.ok and edited.data["replacements"] == 1

    read_back = await loop.call(
        "fs.read_text", {"path": "greet.py", "encoding": "utf-8", "max_chars": None}
    )
    assert read_back.ok and "Hi {name}!" in read_back.data["content"]

    before = await loop.call("git.status", {"repo_path": None})
    assert before.ok and before.data["clean"] is False
    assert "greet.py" in before.data["untracked"]

    pushed = await loop.call(
        "git.commit_push",
        {
            "repo_path": None,
            "message": "feat: add greet",
            "paths": None,
            "remote": None,
            "branch": None,
            "allow_empty": False,
            "push": True,
        },
        confirm=True,
    )
    assert pushed.ok and pushed.data["committed"] is True
    assert pushed.data["push_succeeded"] is True

    after = await loop.call("git.status", {"repo_path": None})
    assert after.ok and after.data["clean"] is True
    local_head = run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    remote_head = run_git(["-C", str(remote), "rev-parse", "HEAD"]).stdout.strip()
    assert local_head == remote_head


@pytest.mark.asyncio
async def test_depends_on_read_after_write_in_one_batch(tmp_path):
    repo = init_repo(tmp_path)
    loop = P0Loop(repo)

    results = await loop.batch(
        [
            (
                "write-notes",
                "fs.write_text",
                {
                    "path": "notes.txt",
                    "content": "batch data\n",
                    "encoding": "utf-8",
                    "create_parents": True,
                    "overwrite": True,
                },
                True,
                [],
            ),
            (
                "read-notes",
                "fs.read_text",
                {"path": "notes.txt", "encoding": "utf-8", "max_chars": None},
                False,
                ["write-notes"],
            ),
        ]
    )

    assert results[0].ok
    assert results[1].ok
    assert results[1].data["content"] == "batch data\n"


@pytest.mark.asyncio
async def test_write_without_confirmation_is_rejected(tmp_path):
    repo = init_repo(tmp_path)
    loop = P0Loop(repo)

    result = await loop.call(
        "fs.write_text",
        {
            "path": "blocked.txt",
            "content": "x",
            "encoding": "utf-8",
            "create_parents": True,
            "overwrite": True,
        },
        confirm=False,
    )

    assert not result.ok
    assert result.error is not None and result.error.code == "CONFIRMATION_REQUIRED"
    assert not (repo / "blocked.txt").exists()