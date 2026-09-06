"""Hot-reload prompt-cache zone tests: frozen prefix, trailing hot zone."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from agents.react import ReActAgent
from core.registry import BaseTool, ToolRegistry
from core import ToolSpec


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


class EchoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


class EchoTool(BaseTool):
    spec = ToolSpec(
        name="test.react_echo",
        description="Echo one integer.",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
    )

    def execute(self, arguments: EchoInput) -> EchoOutput:
        return EchoOutput(value=arguments.value)


class LoudEchoTool(EchoTool):
    spec = ToolSpec(
        name="test.react_echo",
        description="Echo one integer (loud description).",
        version="1.1",
        input_model=EchoInput,
        output_model=EchoOutput,
    )

    def execute(self, arguments: EchoInput) -> EchoOutput:
        return EchoOutput(value=arguments.value * 11)


class QuietEchoTool(EchoTool):
    spec = ToolSpec(
        name="test.react_echo",
        description="Echo one integer (quiet description).",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
    )

    def execute(self, arguments: EchoInput) -> EchoOutput:
        return EchoOutput(value=arguments.value)


def _build_agent(name: str) -> ReActAgent:
    agent = ReActAgent(name, auto_discover_tools=False)
    agent.register_tool(EchoTool())
    return agent


def _named_echo_tool(name: str, description: str) -> QuietEchoTool:
    tool = QuietEchoTool()
    tool.spec = ToolSpec(
        name=name,
        description=description,
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
    )
    return tool


def test_initial_request_builds_frozen_manifest_and_empty_hot_zone():
    agent = _build_agent("froze-once")
    assert agent._frozen_manifest is None
    agent._sync_frozen_manifest()
    assert agent._frozen_manifest == {
        "test.react_echo": agent._prompt_fingerprint(EchoTool().spec),
        "system.skill_catalog": agent._prompt_fingerprint(
            agent.tools.get("system.skill_catalog").spec
        ),
        "system.tool_catalog": agent._prompt_fingerprint(
            agent.tools.get("system.tool_catalog").spec
        ),
    }
    assert agent._hot_tools == {}
    # A second sync must not rebuild the manifest.
    agent._frozen_manifest["probe.tool"] = "deadbeef"
    agent._sync_frozen_manifest()
    assert "probe.tool" in agent._frozen_manifest


def test_hot_tool_registers_into_hot_zone_not_frozen_zone():
    agent = _build_agent("hot-add")
    agent._sync_frozen_manifest()
    assert agent.cache_epoch == 0

    agent.register_hot_tool(LoudEchoTool(), replace=True)

    assert agent._hot_tools == {
        "test.react_echo": agent._prompt_fingerprint(LoudEchoTool().spec)
    }
    assert agent._frozen_manifest["test.react_echo"] == agent._prompt_fingerprint(
        EchoTool().spec
    )
    assert agent.cache_epoch == 0
    assert LoudEchoTool().spec.schema_hash in agent.tools.confirmation_key("test.react_echo")


def test_hot_zone_lines_render_schema_only_for_recent_tools():
    agent = _build_agent("hot-render")
    agent._sync_frozen_manifest()
    for index in range(4):
        agent.register_hot_tool(
            _named_echo_tool(f"test.extra{index}", f"Extra tool {index}."),
        )
    agent.register_hot_tool(LoudEchoTool(), replace=True)

    lines = agent._hot_zone_lines({})
    rendered = "\n".join(lines)

    assert lines[0].startswith("Hot-loaded tools")
    # The four most recent hot tools keep full schemas; the oldest degrades.
    assert "- test.react_echo: Echo one integer (loud description)." in rendered
    roster = [
        line
        for line in rendered.splitlines()
        if line.startswith("- test.extra0:")
    ]
    assert roster == ["- test.extra0: (see schema above or via catalog)"]
    assert "- test.extra1: Extra tool 1." in rendered
    assert QuietEchoTool().spec.schema_hash in agent.tools.confirmation_key("test.extra1")


def test_unregister_hot_tool_is_free_and_unregister_frozen_tool_bumps_epoch():
    agent = _build_agent("epoch-behavior")
    agent._sync_frozen_manifest()
    assert agent.cache_epoch == 0

    agent.unregister_tool("test.react_echo")
    assert agent.cache_epoch == 1
    assert "test.react_echo" not in agent._frozen_manifest
    assert not agent.tools.is_registered("test.react_echo")

    agent.register_hot_tool(QuietEchoTool())
    assert agent.cache_epoch == 1
    assert agent._hot_tools == {
        "test.react_echo": agent._prompt_fingerprint(QuietEchoTool().spec)
    }

    agent.unregister_tool("test.react_echo")
    assert agent.cache_epoch == 1
    assert agent._hot_tools == {}


def test_prompt_cache_key_ignores_hot_zone_and_respects_epoch():
    agent = _build_agent("cache-key")
    agent._sync_frozen_manifest()
    key_before = agent._default_prompt_cache_key("p", "m", mode="react")
    agent.register_hot_tool(LoudEchoTool(), replace=True)
    key_after_hot = agent._default_prompt_cache_key("p", "m", mode="react")
    assert key_before == key_after_hot

    agent.unregister_tool("test.react_echo")
    key_after_epoch = agent._default_prompt_cache_key("p", "m", mode="react")
    assert key_before != key_after_epoch


def test_registry_unregister_removes_and_preserves_generation():
    registry = ToolRegistry()
    registry.register(EchoTool())
    first_generation = registry.confirmation_key("test.react_echo")

    registry.unregister("test.react_echo")
    assert not registry.is_registered("test.react_echo")

    registry.register(QuietEchoTool(), replace=True)
    second_generation = registry.confirmation_key("test.react_echo")
    first_number = int(first_generation.rsplit(":", 1)[1])
    second_number = int(second_generation.rsplit(":", 1)[1])
    assert second_number == first_number + 1


@pytest.mark.asyncio
async def test_hot_tool_is_callable_after_hot_reload():
    class HotLLM:
        model = "test-model"

        def __init__(self) -> None:
            self.calls = 0
            self.requests: list[list[dict]] = []

        def complete(self, messages, **_options):
            self.requests.append([dict(message) for message in messages])
            self.calls += 1
            if self.calls == 1:
                return (
                    "Thought: use the loud echo tool\n"
                    "Action: test.react_echo\n"
                    'Action Input: {"value": 7}'
                )
            return "Final Answer: done"

    agent = ReActAgent("hot-exec-test", llm=HotLLM(), auto_discover_tools=False)
    agent.register_tool(EchoTool())
    agent._sync_frozen_manifest()
    agent.register_hot_tool(LoudEchoTool(), replace=True)

    answer = await agent.run_with_react("echo 7", max_rounds=3)

    assert answer == "done"
    # The request tail must carry the hot-zone block after the user message.
    last_message = agent.llm.requests[0][-1]
    assert last_message["role"] == "system"
    assert "Hot-loaded tools" in last_message["content"]
    assert "test.react_echo: Echo one integer (loud description)." in last_message["content"]


@pytest.mark.asyncio
async def test_frozen_inventory_stays_stable_after_hot_reload():
    class FinalLLM:
        model = "test-model"

        def __init__(self) -> None:
            self.requests: list[list[dict]] = []

        def complete(self, messages, **_options):
            self.requests.append([dict(message) for message in messages])
            return "Final Answer: done"

    agent = ReActAgent("frozen-inventory-test", llm=FinalLLM(), auto_discover_tools=False)
    agent.register_tool(EchoTool())
    await agent.run_with_react("echo 7")

    agent.register_hot_tool(LoudEchoTool(), replace=True)
    await agent.run_with_react("echo 8")

    first_inventory = agent.llm.requests[0][0]["content"]
    second_inventory = agent.llm.requests[1][0]["content"]
    assert first_inventory == second_inventory
    assert "test.react_echo: Echo one integer (loud description)." not in first_inventory
    assert "Hot-loaded tools" in agent.llm.requests[1][-1]["content"]


