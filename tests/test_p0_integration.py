"""End-to-end integration test for the P0 code-change loop.

Binds the P0 tools to one scratch workspace like a real agent deployment,
executes them through the real runtime chain (ToolCall -> confirmation gate ->
execute) and covers fs write -> edit -> read verify, depends_on batching, and
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


class P0Loop:
    """One workspace's P0 tool instances bound to the real runtime chain."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.registry = ToolRegistry()
        for tool in (
            ReadTextTool(base_dir=base_dir),
            WriteTextTool(base_dir=base_dir),
            EditTextTool(base_dir=base_dir),
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