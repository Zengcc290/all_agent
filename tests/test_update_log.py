from __future__ import annotations

import pytest

from core import (
    ExecutionContext,
    ToolCall,
    ToolExecutionManager,
    ToolRegistry,
    UpdateLogRepository,
    discover_tools,
)
from tool.update_log import UpdateLogInput, UpdateLogTool


def _payload() -> dict:
    return {
        "executor": "test-ai",
        "update_type": "feature",
        "title": "Add compact update log",
        "task_background": "Keep historical logs out of the model context.",
        "update_details": "Store one structured row per update in SQLite.",
        "added_features": "A durable append-only log writer.",
        "files": [
            {
                "path": "core/update_log.py",
                "action": "added",
                "description": "SQLite repository and schema",
            }
        ],
        "behavior_impact": "No runtime behavior changes outside the new tool.",
        "validation": "Repository unit tests pass.",
        "risks": "none",
        "follow_up": "none",
    }


def test_update_log_repository_assigns_monotonic_ids_and_decodes_files(tmp_path):
    repository = UpdateLogRepository(tmp_path / "updates.sqlite3")
    first = repository.append(**_payload(), system_name="TestOS", timestamp="2026-01-01T00:00:00+00:00")
    second = repository.append(**_payload(), system_name="TestOS", timestamp="2026-01-01T00:00:01+00:00")

    assert first["update_id"] == 1
    assert first["next_update_id"] == 2
    assert second["update_id"] == 2
    assert repository.latest_id() == 2
    stored = repository.get(1)
    assert stored is not None
    assert stored["system_name"] == "TestOS"
    assert stored["files"][0]["path"] == "core/update_log.py"


def test_update_log_tool_returns_compact_acknowledgement():
    repository = UpdateLogRepository(":memory:")
    tool = UpdateLogTool(repository)
    output = tool.execute(UpdateLogInput(**_payload()))

    assert output.update_id == 1
    assert output.next_update_id == 2
    assert output.recorded is True
    assert repository.get(output.update_id)["executor"] == "test-ai"
    repository.close()


def test_update_log_tool_is_auto_discoverable(tmp_path, monkeypatch):
    monkeypatch.setenv("UPDATE_LOG_DB_PATH", str(tmp_path / "discovered.sqlite3"))
    registry = ToolRegistry()
    report = discover_tools(registry, package="tool")

    record = report.for_tool("system.update_log")
    assert record is not None
    assert record.status == "registered"
    assert registry.is_registered("system.update_log", version="1.0")


@pytest.mark.asyncio
async def test_update_log_tool_requires_write_permission_and_generation_confirmation():
    repository = UpdateLogRepository(":memory:")
    tool = UpdateLogTool(repository)
    registry = ToolRegistry()
    registry.register(tool)
    _, generation = registry.resolve(tool.spec.name)
    call = ToolCall(
        call_id="log-1",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        registry_generation=generation,
        arguments=_payload(),
    )
    manager = ToolExecutionManager(registry)
    denied = await manager.execute_batch(
        [call], ExecutionContext(permissions=frozenset({"database.write"}))
    )
    assert denied.results[0].error.code == "CONFIRMATION_REQUIRED"

    context = ExecutionContext(
        permissions=frozenset({"database.write"}),
        confirmed_side_effects=frozenset({registry.confirmation_key(tool.spec.name)}),
    )
    allowed = await manager.execute_batch([call], context)
    assert allowed.results[0].ok
    assert allowed.results[0].data["update_id"] == 1
    repository.close()
