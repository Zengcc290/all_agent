from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core import ToolCall, ToolExecutionManager, ToolRegistry
from tool.fs_read_file_glob import (
    TOOL_ENABLED,
    ReadFileGlobInput,
    ReadFileGlobTool,
    create_tool,
)


def test_read_file_glob_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, ReadFileGlobTool)


def test_read_file_glob_finds_files_recursively(tmp_path):
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "c.py").write_text("c", encoding="utf-8")
    tool = ReadFileGlobTool(base_dir=tmp_path)

    output = tool.execute(ReadFileGlobInput(root=".", pattern="**/*.py"))

    names = sorted(Path(item).name for item in output.matches)
    assert names == ["a.py", "c.py"]
    assert output.match_count == 2
    assert output.truncated is False
    assert output.resolved_root == str(tmp_path.resolve())


def test_read_file_glob_single_star_stays_in_one_level(tmp_path):
    (tmp_path / "top.py").write_text("t", encoding="utf-8")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "nested.py").write_text("n", encoding="utf-8")
    tool = ReadFileGlobTool(base_dir=tmp_path)

    output = tool.execute(ReadFileGlobInput(root=".", pattern="*.py"))

    assert [Path(item).name for item in output.matches] == ["top.py"]


def test_read_file_glob_excludes_hidden_segments_by_default(tmp_path):
    (tmp_path / "keep.py").write_text("k", encoding="utf-8")
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "lib.py").write_text("l", encoding="utf-8")
    tool = ReadFileGlobTool(base_dir=tmp_path)

    plain = tool.execute(ReadFileGlobInput(root=".", pattern="**/*.py"))
    with_hidden = tool.execute(
        ReadFileGlobInput(root=".", pattern="**/*.py", include_hidden=True)
    )

    assert all(".venv" not in item for item in plain.matches)
    assert any(".venv" in item for item in with_hidden.matches)


def test_read_file_glob_rejects_patterns_escaping_root(tmp_path):
    outside = tmp_path.parent / "escape_target.py"
    outside.write_text("x", encoding="utf-8")
    tool = ReadFileGlobTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="escape"):
        tool.execute(ReadFileGlobInput(root=".", pattern="../escape_target.py"))


def test_read_file_glob_truncates_matches(tmp_path):
    for index in range(5):
        (tmp_path / f"f{index}.py").write_text("x", encoding="utf-8")
    tool = ReadFileGlobTool(base_dir=tmp_path)

    output = tool.execute(ReadFileGlobInput(root=".", pattern="*.py", max_matches=2))

    assert output.truncated is True
    assert len(output.matches) == 2
    assert output.match_count == 2


def test_read_file_glob_validation_is_strict(tmp_path):
    with pytest.raises(ValidationError):
        ReadFileGlobInput.model_validate({"root": ".", "pattern": 7}, strict=True)
    with pytest.raises(ValidationError):
        ReadFileGlobInput.model_validate({"root": "."}, strict=True)
    with pytest.raises(ValidationError):
        ReadFileGlobInput.model_validate(
            {"root": ".", "pattern": "*.py", "surprise": True}, strict=True
        )


@pytest.mark.asyncio
async def test_read_file_glob_runs_without_side_effect_confirmation(tmp_path):
    (tmp_path / "one.py").write_text("1", encoding="utf-8")
    tool = ReadFileGlobTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    call = ToolCall(
        call_id="glob-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments={
            "root": ".",
            "pattern": "*.py",
            "include_hidden": False,
            "max_matches": 500,
        },
    )
    manager = ToolExecutionManager(registry)

    batch = await manager.execute_batch([call])

    assert batch.results[0].ok
    assert "one.py" in batch.results[0].data["matches"][0]


def test_read_file_glob_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("fs.read_file_glob")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("fs.read_file_glob", version="1.0.0")