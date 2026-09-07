from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import ToolCall, ToolExecutionManager, ToolRegistry
from tool.fs_read_text import TOOL_ENABLED, ReadTextInput, ReadTextTool, create_tool


def test_read_text_implements_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    tool = create_tool()
    assert isinstance(tool, ReadTextTool)


def test_read_text_returns_content_and_metadata(tmp_path):
    target = tmp_path / "hello.txt"
    # write_bytes avoids Windows text-mode newline translation.
    target.write_bytes(b"hello world\n")
    tool = ReadTextTool(base_dir=tmp_path)

    output = tool.execute(ReadTextInput(path="hello.txt"))

    assert output.resolved_path == str(target.resolve())
    assert output.content == "hello world\n"
    assert output.character_count == 12
    assert output.truncated is False
    assert output.encoding == "utf-8"


def test_read_text_honors_other_encodings(tmp_path):
    target = tmp_path / "uni.txt"
    target.write_text("你好世界", encoding="utf-16")
    tool = ReadTextTool(base_dir=tmp_path)

    output = tool.execute(ReadTextInput(path="uni.txt", encoding="utf-16"))

    assert output.content == "你好世界"
    assert output.encoding == "utf-16"
    assert output.character_count == 4


def test_read_text_rejects_unknown_encoding(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    tool = ReadTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="unknown text encoding"):
        tool.execute(ReadTextInput(path="a.txt", encoding="not-a-codec"))


def test_read_text_rejects_paths_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = ReadTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tool.execute(ReadTextInput(path=str(outside)))


def test_read_text_rejects_missing_and_directory_paths(tmp_path):
    tool = ReadTextTool(base_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        tool.execute(ReadTextInput(path="missing.txt"))
    with pytest.raises(IsADirectoryError):
        tool.execute(ReadTextInput(path="."))


def test_read_text_rejects_binary_files(tmp_path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"\x00\x01\x02")
    tool = ReadTextTool(base_dir=tmp_path)

    with pytest.raises(ValueError, match="binary"):
        tool.execute(ReadTextInput(path="blob.bin"))


def test_read_text_truncates_large_content(tmp_path):
    target = tmp_path / "big.txt"
    target.write_text("x" * 100, encoding="utf-8")
    tool = ReadTextTool(base_dir=tmp_path)

    output = tool.execute(ReadTextInput(path="big.txt", max_chars=10))

    assert output.truncated is True
    assert output.content.startswith("x" * 10)
    assert output.character_count == 100


def test_read_text_validation_is_strict():
    with pytest.raises(ValidationError):
        ReadTextInput.model_validate({"path": 123}, strict=True)
    with pytest.raises(ValidationError):
        ReadTextInput.model_validate({"path": "a.txt", "surprise": 1}, strict=True)
    with pytest.raises(ValidationError):
        ReadTextInput.model_validate({}, strict=True)


@pytest.mark.asyncio
async def test_read_text_runs_without_side_effect_confirmation(tmp_path):
    target = tmp_path / "runtime.txt"
    target.write_text("runtime", encoding="utf-8")
    tool = ReadTextTool(base_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    call = ToolCall(
        call_id="read-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments={"path": "runtime.txt", "encoding": "utf-8", "max_chars": None},
    )
    manager = ToolExecutionManager(registry)

    batch = await manager.execute_batch([call])

    assert batch.results[0].ok
    assert batch.results[0].data["content"] == "runtime"
    assert batch.results[0].data["truncated"] is False


def test_read_text_is_auto_discoverable():
    from core import discover_tools

    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("fs.read_text")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("fs.read_text", version="1.0.0")