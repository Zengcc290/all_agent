from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import ExecutionContext, ToolCall, ToolExecutionManager, ToolRegistry
from tool.fs_edit_text import TOOL_ENABLED, EditTextInput, EditTextTool, create_tool


def test_edit_text_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, EditTextTool)


def test_edit_text_replaces_unique_old_string(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("alpha beta alpha", encoding="utf-8")
    tool = EditTextTool(base_dir=tmp_path)

    output = tool.execute(
        EditTextInput(path="file.txt", old_string="beta", new_string="GAMMA")
    )

    assert target.read_text(encoding="utf-8") == "alpha GAMMA alpha"
    assert output.replacements == 1
    assert output.changed is True


def test_edit_text_replaces_all_when_requested(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("alpha beta alpha", encoding="utf-8")
    tool = EditTextTool(base_dir=tmp_path)

    output = tool.execute(
        EditTextInput(
            path="file.txt", old_string="alpha", new_string="X", replace_all=True
        )
    )

    assert target.read_text(encoding="utf-8") == "X beta X"
    assert output.replacements == 2


def test_edit_text_rejects_ambiguous_old_string(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("alpha beta alpha", encoding="utf-8")
    tool = EditTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="occurs 2 times"):
        tool.execute(EditTextInput(path="file.txt", old_string="alpha", new_string="X"))
    assert target.read_text(encoding="utf-8") == "alpha beta alpha"


def test_edit_text_rejects_missing_old_string(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("hello", encoding="utf-8")
    tool = EditTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="not found"):
        tool.execute(EditTextInput(path="file.txt", old_string="nope", new_string="X"))


def test_edit_text_deletes_with_empty_new_string(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("keep [drop] keep", encoding="utf-8")
    tool = EditTextTool(base_dir=tmp_path)

    tool.execute(EditTextInput(path="file.txt", old_string="[drop] ", new_string=""))

    assert target.read_text(encoding="utf-8") == "keep keep"


def test_edit_text_rejects_missing_file(tmp_path):
    tool = EditTextTool(base_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        tool.execute(EditTextInput(path="nope.txt", old_string="a", new_string="b"))


def test_edit_text_rejects_paths_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside_edit.txt"
    outside.write_text("x", encoding="utf-8")
    tool = EditTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tool.execute(
            EditTextInput(path=str(outside), old_string="x", new_string="y")
        )
    assert outside.read_text(encoding="utf-8") == "x"


def test_edit_text_validation_is_strict():
    with pytest.raises(ValidationError):
        EditTextInput.model_validate({"path": "a.txt", "new_string": "b"}, strict=True)
    with pytest.raises(ValidationError):
        EditTextInput.model_validate(
            {"path": "a.txt", "old_string": "a", "new_string": "b", "extra": 1},
            strict=True,
        )
    with pytest.raises(ValidationError):
        EditTextInput.model_validate(
            {"path": "a.txt", "old_string": "", "new_string": "b"}, strict=True
        )


@pytest.mark.asyncio
async def test_edit_text_requires_side_effect_confirmation(tmp_path):
    target = tmp_path / "runtime.txt"
    target.write_text("before", encoding="utf-8")
    tool = EditTextTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    arguments = {
        "path": "runtime.txt",
        "old_string": "before",
        "new_string": "after",
        "replace_all": False,
        "encoding": "utf-8",
    }
    call = ToolCall(
        call_id="edit-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments=arguments,
    )
    manager = ToolExecutionManager(registry)

    denied = await manager.execute_batch([call])
    assert denied.results[0].error.code == "CONFIRMATION_REQUIRED"
    assert target.read_text(encoding="utf-8") == "before"

    context = ExecutionContext(
        confirmed_side_effects=frozenset(
            {registry.confirmation_key(tool.spec.name)}
        ),
    )
    allowed = await manager.execute_batch([call], context)
    assert allowed.results[0].ok
    assert allowed.results[0].data["replacements"] == 1
    assert target.read_text(encoding="utf-8") == "after"


def test_edit_text_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("fs.edit_text")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("fs.edit_text", version="1.0.0")