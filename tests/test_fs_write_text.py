from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import ExecutionContext, ToolCall, ToolExecutionManager, ToolRegistry
from tool.fs_write_text import TOOL_ENABLED, WriteTextInput, WriteTextTool, create_tool


def test_write_text_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, WriteTextTool)


def test_write_text_creates_file_and_parents(tmp_path):
    tool = WriteTextTool(base_dir=tmp_path)

    output = tool.execute(WriteTextInput(path="nested/deep/file.txt", content="hello"))

    target = tmp_path / "nested" / "deep" / "file.txt"
    assert target.read_text(encoding="utf-8") == "hello"
    assert output.created is True
    assert output.existed is False
    assert output.resolved_path == str(target.resolve())
    assert output.bytes_written == 5
    assert output.character_count == 5


def test_write_text_overwrites_existing_file(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    tool = WriteTextTool(base_dir=tmp_path)

    output = tool.execute(WriteTextInput(path="file.txt", content="new"))

    assert target.read_text(encoding="utf-8") == "new"
    assert output.existed is True
    assert output.created is False


def test_write_text_refuses_overwrite_when_disabled(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    tool = WriteTextTool(base_dir=tmp_path)

    with pytest.raises(FileExistsError):
        tool.execute(WriteTextInput(path="file.txt", content="new", overwrite=False))
    assert target.read_text(encoding="utf-8") == "old"


def test_write_text_refuses_missing_parents_when_disabled(tmp_path):
    tool = WriteTextTool(base_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        tool.execute(
            WriteTextInput(
                path="missing/deep/file.txt", content="x", create_parents=False
            )
        )
    assert not (tmp_path / "missing").exists()


def test_write_text_rejects_paths_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside_write.txt"
    tool = WriteTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tool.execute(WriteTextInput(path=str(outside), content="x"))
    assert not outside.exists()


def test_write_text_leaves_no_temp_files(tmp_path):
    tool = WriteTextTool(base_dir=tmp_path)

    tool.execute(WriteTextInput(path="a.txt", content="x"))
    tool.execute(WriteTextInput(path="b.txt", content="y"))

    leftovers = [item.name for item in tmp_path.iterdir() if item.name.endswith(".tmp")]
    assert leftovers == []


def test_write_text_rejects_non_encodable_content(tmp_path):
    tool = WriteTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="cannot be encoded"):
        tool.execute(WriteTextInput(path="a.txt", content="你好", encoding="ascii"))
    assert not (tmp_path / "a.txt").exists()


def test_write_text_validation_is_strict():
    with pytest.raises(ValidationError):
        WriteTextInput.model_validate({"path": "a.txt"}, strict=True)
    with pytest.raises(ValidationError):
        WriteTextInput.model_validate(
            {"path": "a.txt", "content": 42}, strict=True
        )
    with pytest.raises(ValidationError):
        WriteTextInput.model_validate(
            {
                "path": "a.txt",
                "content": "x",
                "encoding": "utf-8",
                "create_parents": True,
                "overwrite": True,
                "surprise": 1,
            },
            strict=True,
        )


@pytest.mark.asyncio
async def test_write_text_requires_side_effect_confirmation(tmp_path):
    tool = WriteTextTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    arguments = {
        "path": "runtime.txt",
        "content": "data",
        "encoding": "utf-8",
        "create_parents": True,
        "overwrite": True,
    }
    call = ToolCall(
        call_id="write-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments=arguments,
    )
    manager = ToolExecutionManager(registry)

    denied = await manager.execute_batch([call])
    assert denied.results[0].error.code == "CONFIRMATION_REQUIRED"
    assert not (tmp_path / "runtime.txt").exists()

    context = ExecutionContext(
        confirmed_side_effects=frozenset(
            {registry.confirmation_key(tool.spec.name)}
        ),
    )
    allowed = await manager.execute_batch([call], context)
    assert allowed.results[0].ok
    assert allowed.results[0].data["created"] is True
    assert (tmp_path / "runtime.txt").read_text(encoding="utf-8") == "data"


def test_write_text_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("fs.write_text")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("fs.write_text", version="1.0.0")