from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from core import ExecutionContext, ToolCall, ToolExecutionManager, ToolRegistry
from tool.shell_run import TOOL_ENABLED, ShellRunInput, ShellRunTool, create_tool


def test_shell_run_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, ShellRunTool)


def test_shell_run_echo_success(tmp_path):
    tool = ShellRunTool(base_dir=tmp_path)

    output = tool.execute(ShellRunInput(command="echo hello", timeout_seconds=30))

    assert output.exit_code == 0
    assert output.stdout.strip() == "hello"
    assert output.timed_out is False
    assert output.resolved_cwd == str(tmp_path.resolve())


def test_shell_run_reports_nonzero_exit(tmp_path):
    tool = ShellRunTool(base_dir=tmp_path)

    output = tool.execute(ShellRunInput(command="exit 7", timeout_seconds=30))

    assert output.exit_code == 7


def test_shell_run_captures_stderr(tmp_path):
    tool = ShellRunTool(base_dir=tmp_path)
    command = f'"{sys.executable}" -c "import sys; print(\'boom\', file=sys.stderr)"'

    output = tool.execute(ShellRunInput(command=command, timeout_seconds=60))

    assert output.exit_code == 0
    assert "boom" in output.stderr


def test_shell_run_uses_working_directory(tmp_path):
    tool = ShellRunTool(base_dir=tmp_path)
    command = f'"{sys.executable}" -c "import os; print(os.getcwd())"'

    output = tool.execute(
        ShellRunInput(command=command, cwd=".", timeout_seconds=60)
    )

    assert str(tmp_path.resolve()) in output.stdout


def test_shell_run_times_out_and_kills_process(tmp_path):
    tool = ShellRunTool(base_dir=tmp_path)
    command = f'"{sys.executable}" -c "import time; time.sleep(5)"'

    output = tool.execute(ShellRunInput(command=command, timeout_seconds=1))

    assert output.timed_out is True
    assert output.exit_code == -1
    assert output.duration_seconds < 5


def test_shell_run_rejects_bad_cwd(tmp_path):
    tool = ShellRunTool(base_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        tool.execute(ShellRunInput(command="echo x", cwd="missing", timeout_seconds=30))
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        tool.execute(ShellRunInput(command="echo x", cwd="file.txt", timeout_seconds=30))


def test_shell_run_rejects_cwd_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside_cwd"
    outside.mkdir()
    tool = ShellRunTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tool.execute(
            ShellRunInput(command="echo x", cwd=str(outside), timeout_seconds=30)
        )


def test_shell_run_validation_is_strict():
    with pytest.raises(ValidationError):
        ShellRunInput.model_validate({"command": ""}, strict=True)
    with pytest.raises(ValidationError):
        ShellRunInput.model_validate(
            {"command": "echo x", "timeout_seconds": 0.5}, strict=True
        )
    with pytest.raises(ValidationError):
        ShellRunInput.model_validate(
            {"command": "echo x", "timeout_seconds": 30, "surprise": 1}, strict=True
        )


@pytest.mark.asyncio
async def test_shell_run_requires_side_effect_confirmation(tmp_path):
    tool = ShellRunTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    arguments = {"command": "echo hi", "cwd": None, "timeout_seconds": 30}
    call = ToolCall(
        call_id="shell-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments=arguments,
    )
    manager = ToolExecutionManager(registry)

    denied = await manager.execute_batch([call])
    assert denied.results[0].error.code == "CONFIRMATION_REQUIRED"

    context = ExecutionContext(
        confirmed_side_effects=frozenset(
            {registry.confirmation_key(tool.spec.name)}
        ),
    )
    allowed = await manager.execute_batch([call], context)
    assert allowed.results[0].ok
    assert allowed.results[0].data["exit_code"] == 0


def test_shell_run_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("shell.run")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("shell.run", version="1.0.0")