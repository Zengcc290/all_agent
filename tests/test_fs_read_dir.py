from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import ToolCall, ToolExecutionManager, ToolRegistry
from tool.fs_read_dir import TOOL_ENABLED, ReadDirInput, ReadDirTool, create_tool


def test_read_dir_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, ReadDirTool)


def test_read_dir_lists_entries_directories_first_then_by_name(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    tool = ReadDirTool(base_dir=tmp_path)

    output = tool.execute(ReadDirInput(path=".", max_entries=100))

    assert [entry.name for entry in output.entries] == ["sub", "a.txt", "b.txt"]
    assert output.entry_count == 3
    assert output.truncated is False
    by_name = {entry.name: entry for entry in output.entries}
    assert by_name["a.txt"].is_file is True
    assert by_name["a.txt"].is_dir is False
    assert by_name["a.txt"].size == 1
    assert by_name["sub"].is_file is False
    assert by_name["sub"].is_dir is True
    assert by_name["a.txt"].modified


def test_read_dir_hides_dot_entries_by_default(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "visible.txt").write_text("v", encoding="utf-8")
    tool = ReadDirTool(base_dir=tmp_path)

    plain = tool.execute(ReadDirInput(path="."))
    with_hidden = tool.execute(ReadDirInput(path=".", include_hidden=True))

    plain_names = {entry.name for entry in plain.entries}
    hidden_names = {entry.name for entry in with_hidden.entries}
    assert ".hidden" not in plain_names
    assert "visible.txt" in plain_names
    assert ".hidden" in hidden_names


def test_read_dir_truncates_and_reports(tmp_path):
    for index in range(5):
        (tmp_path / f"f{index}.txt").write_text("x", encoding="utf-8")
    tool = ReadDirTool(base_dir=tmp_path)

    output = tool.execute(ReadDirInput(path=".", max_entries=2))

    assert output.truncated is True
    assert len(output.entries) == 2
    assert output.entry_count == 2


def test_read_dir_rejects_bad_paths(tmp_path):
    tool = ReadDirTool(base_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        tool.execute(ReadDirInput(path="missing"))
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        tool.execute(ReadDirInput(path="file.txt"))


def test_read_dir_survives_broken_symlinks(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "broken_link"
    try:
        link.symlink_to(target.with_name("no_such_target.txt"))
    except OSError:
        pytest.skip("symlink creation is not permitted on this system")
    target.unlink()
    tool = ReadDirTool(base_dir=tmp_path)

    output = tool.execute(ReadDirInput(path="."))

    by_name = {entry.name: entry for entry in output.entries}
    assert "broken_link" in by_name
    assert by_name["broken_link"].is_symlink is True


def test_read_dir_rejects_paths_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir()
    tool = ReadDirTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tool.execute(ReadDirInput(path=str(outside)))


def test_read_dir_validation_is_strict():
    with pytest.raises(ValidationError):
        ReadDirInput.model_validate({"path": "x", "include_hidden": "yes"}, strict=True)
    with pytest.raises(ValidationError):
        ReadDirInput.model_validate(
            {"path": ".", "include_hidden": False, "max_entries": 99999}, strict=True
        )
    with pytest.raises(ValidationError):
        ReadDirInput.model_validate({}, strict=True)


@pytest.mark.asyncio
async def test_read_dir_runs_without_side_effect_confirmation(tmp_path):
    (tmp_path / "r.txt").write_text("r", encoding="utf-8")
    tool = ReadDirTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    call = ToolCall(
        call_id="read-dir-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments={"path": ".", "include_hidden": False, "max_entries": 500},
    )
    manager = ToolExecutionManager(registry)

    batch = await manager.execute_batch([call])

    assert batch.results[0].ok
    assert [entry["name"] for entry in batch.results[0].data["entries"]] == ["r.txt"]


def test_read_dir_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("fs.read_dir")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("fs.read_dir", version="1.0.0")