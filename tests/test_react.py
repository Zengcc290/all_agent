from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agents.react import ReActAgent
from core import ToolSpec, ToolSpecRepository
from core.registry import BaseTool
from tool.current_time import CurrentTimeTool
from tool.search import SearchTool


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


class EchoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


class EchoTool(BaseTool):
    spec = ToolSpec(
        name="test.react_echo",
        description="Echo one integer for ReAct logging tests.",
        version="1.0",
        input_model=EchoInput,
        output_model=EchoOutput,
    )

    def execute(self, arguments: EchoInput) -> EchoOutput:
        return EchoOutput(value=arguments.value)


class ReActLoggingLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def complete(self, messages, **_options):
        self.requests.append([dict(message) for message in messages])
        self.calls += 1
        if self.calls == 1:
            return (
                "Thought: use the echo tool\n"
                "Action: test.react_echo\n"
                'Action Input: {"value": 7}'
            )
        return "Final Answer: done"


class NativeReActLoggingLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _messages, **_options):
        self.calls += 1
        if self.calls == 1:
            tool_call = SimpleNamespace(
                id="native-echo",
                function=SimpleNamespace(
                    name="test__react_echo", arguments='{"value": 7}'
                ),
            )
            message = SimpleNamespace(
                role="assistant", content=None, tool_calls=[tool_call]
            )
        else:
            message = SimpleNamespace(
                role="assistant", content="Final Answer: done", tool_calls=[]
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class MultiCatalogLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def complete(self, messages, **_options):
        self.requests.append([dict(message) for message in messages])
        self.calls += 1
        if self.calls == 1:
            return (
                "Thought: resolve both capabilities\n"
                "Action: system.tool_catalog\n"
                'Action Input: {"action":"resolve",'
                '"intent":"current time and web search","limit":20}'
            )
        if self.calls == 2:
            return (
                "Thought: read the current time\n"
                "Action: system.current_time\n"
                'Action Input: {}'
            )
        return "Final Answer: done"


class DirectCurrentTimeLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _messages, **_options):
        self.calls += 1
        if self.calls == 1:
            return (
                "Thought: use the registered clock\n"
                "Action: system.current_time\n"
                'Action Input: {}'
            )
        return "Final Answer: done"


class UnmarkedToolIntentLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _messages, **_options):
        self.calls += 1
        if self.calls == 1:
            return "我无法读取当前时间，也无法查询今日运势。"
        if self.calls == 2:
            return (
                "Thought: use the clock tool\n"
                "Action: system.current_time\n"
                'Action Input: {}'
            )
        return "Final Answer: done"


def test_lazy_registration_logs_all_persisted_tool_names(capsys):
    repository = ToolSpecRepository(":memory:")
    agent = ReActAgent(
        "lazy-logging-test",
        llm=ReActLoggingLLM(),
        repository=repository,
        auto_discover_tools=False,
        lazy_tools=True,
    )
    capsys.readouterr()

    agent.register_tool(EchoTool())

    output = capsys.readouterr().out
    assert (
        "Current registered tools: system.tool_catalog, test.react_echo" in output
    )
    instruction = agent._with_tool_instructions([], agent.tools.snapshot())
    assert (
        "All registered tool names: system.tool_catalog, test.react_echo"
        in instruction[0]["content"]
    )
    repository.close()


@pytest.mark.asyncio
async def test_lazy_catalog_loads_all_resolved_specs_and_sends_schemas():
    repository = ToolSpecRepository(":memory:")
    llm = MultiCatalogLLM()
    agent = ReActAgent(
        "multi-catalog-test",
        llm=llm,
        repository=repository,
        auto_discover_tools=False,
        lazy_tools=True,
    )
    agent.register_tool(CurrentTimeTool())
    agent.register_tool(SearchTool(base_url="https://example.invalid"))

    answer = await agent.run_with_react("what time is it", max_rounds=4)

    assert answer == "done"
    assert agent.is_tool_registered("system.current_time")
    assert agent.is_tool_registered("web.search")
    second_system_prompt = llm.requests[1][0]["content"]
    assert "Loaded tool input schemas:" in second_system_prompt
    assert "system.current_time:" in second_system_prompt
    assert "web.search:" in second_system_prompt
    repository.close()


@pytest.mark.asyncio
async def test_lazy_react_direct_call_loads_cataloged_current_time():
    repository = ToolSpecRepository(":memory:")
    agent = ReActAgent(
        "direct-current-time-test",
        llm=DirectCurrentTimeLLM(),
        repository=repository,
        auto_discover_tools=False,
        lazy_tools=True,
    )
    agent.register_tool(CurrentTimeTool())

    answer = await agent.run_with_react("现在是什么时候", max_rounds=3)

    assert answer == "done"
    assert "system.current_time" in agent.tools
    repository.close()


@pytest.mark.asyncio
async def test_unmarked_tool_intent_is_retried_instead_of_being_final_answer():
    agent = ReActAgent(
        "unmarked-tool-intent-test",
        llm=UnmarkedToolIntentLLM(),
        auto_discover_tools=False,
        lazy_tools=False,
    )
    agent.register_tool(CurrentTimeTool())

    answer = await agent.run_with_react("现在是什么时候", max_rounds=4, defer_tool_loading=False)

    assert answer == "done"


@pytest.mark.asyncio
async def test_react_logs_progress_and_each_round_result(capsys):
    llm = ReActLoggingLLM()
    agent = ReActAgent(
        "logging-test",
        llm=llm,
        auto_discover_tools=False,
        lazy_tools=False,
    )
    agent.register_tool(EchoTool())
    capsys.readouterr()

    answer = await agent.run_with_react(
        "echo 7", max_rounds=3, defer_tool_loading=False
    )

    output = capsys.readouterr().out
    assert answer == "done"
    assert "ReAct round 1/3 started." in output
    assert "ReAct round 1 model completed in" in output
    assert "ReAct round 1 thought: use the echo tool" in output
    assert "ReAct round 1 calling tools: test.react_echo." in output
    assert "ReAct round 1 tools completed in" in output
    assert "ReAct round 2/3 started." in output
    assert "ReAct round 2 final answer: done" in output
    assert "Action Input" not in output
    assert '"value": 7' not in output
    assert all(
        "All registered tool names: system.tool_catalog, test.react_echo"
        in request[0]["content"]
        for request in llm.requests
    )


@pytest.mark.asyncio
async def test_react_eagerly_exposes_registered_tools_by_default():
    llm = ReActLoggingLLM()
    agent = ReActAgent(
        "eager-default-test",
        llm=llm,
        auto_discover_tools=False,
    )
    agent.register_tool(EchoTool())

    answer = await agent.run_with_react("echo 7", max_rounds=3)

    assert answer == "done"
    first_instruction = llm.requests[0][0]["content"]
    assert "- test.react_echo:" in first_instruction
    assert "Action Input schema:" in first_instruction


@pytest.mark.asyncio
async def test_react_logs_native_tool_names_and_duration(capsys):
    agent = ReActAgent(
        "native-logging-test",
        llm=NativeReActLoggingLLM(),
        auto_discover_tools=False,
        lazy_tools=False,
    )
    agent.register_tool(EchoTool())
    capsys.readouterr()

    answer = await agent.run_with_react(
        "echo 7", max_rounds=3, defer_tool_loading=False
    )

    output = capsys.readouterr().out
    assert answer == "done"
    assert "ReAct round 1 calling tools: test.react_echo." in output
    assert "ReAct round 1 tools completed in" in output
    assert '"value": 7' not in output
