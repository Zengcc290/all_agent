from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agents.react import ReActAgent, parse_react_response
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


class SearchProbeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str


class SearchProbeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str


class SearchProbeTool(BaseTool):
    spec = ToolSpec(
        name="web.search",
        description="Record a web search query for ReAct ordering tests.",
        version="1.0",
        input_model=SearchProbeInput,
        output_model=SearchProbeOutput,
        recommended_before_tools=("system.current_time",),
    )

    def __init__(self) -> None:
        self.queries = []

    def execute(self, arguments: SearchProbeInput) -> SearchProbeOutput:
        self.queries.append(arguments.query)
        return SearchProbeOutput(query=arguments.query)


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


class CatalogFirstEchoLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def complete(self, messages, **_options):
        self.requests.append([dict(message) for message in messages])
        self.calls += 1
        if self.calls == 1:
            return (
                "Thought: find the echo input contract\n"
                "Action: system.tool_catalog\n"
                'Action Input: {"action":"resolve","intent":"echo"}'
            )
        if self.calls == 2:
            return (
                "Thought: use the resolved echo tool\n"
                "Action: test.react_echo\n"
                'Action Input: {"value":7}'
            )
        return "Final Answer: done"


class DirectThenCatalogEchoLLM(CatalogFirstEchoLLM):
    def complete(self, messages, **_options):
        self.requests.append([dict(message) for message in messages])
        self.calls += 1
        if self.calls == 1:
            return (
                "Thought: try the echo tool\n"
                "Action: test.react_echo\n"
                'Action Input: {"value":7}'
            )
        if self.calls == 2:
            return (
                "Thought: retrieve the echo input contract\n"
                "Action: system.tool_catalog\n"
                'Action Input: {"action":"resolve","intent":"echo"}'
            )
        if self.calls == 3:
            return (
                "Thought: use the resolved echo tool\n"
                "Action: test.react_echo\n"
                'Action Input: {"value":7}'
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


class AlwaysUnmarkedAnswerLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _messages, **_options):
        self.calls += 1
        return "我直接根据记忆回答：日志共有 40 条。"


class AlwaysMalformedActionLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _messages, **_options):
        self.calls += 1
        return "Action: system.current_time\nAction Input: {not json"


class SearchBeforeTimeLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def complete(self, messages, **_options):
        self.requests.append([dict(message) for message in messages])
        self.calls += 1
        if self.calls == 1:
            return (
                "Thought: search immediately\n"
                "Action: web.search\n"
                'Action Input: {"query":"latest model news"}'
            )
        if self.calls == 2:
            return (
                "Thought: establish the current date first\n"
                "Action: system.current_time\n"
                "Action Input: {}"
            )
        if self.calls == 3:
            return (
                "Thought: search using the observed date\n"
                "Action: web.search\n"
                'Action Input: {"query":"model news in the last two days"}'
            )
        return "Final Answer: done"


class TemporalCatalogOmitsTimeLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def complete(self, messages, **_options):
        self.requests.append([dict(message) for message in messages])
        self.calls += 1
        if self.calls == 1:
            return (
                "Thought: resolve web search\n"
                "Action: system.tool_catalog\n"
                'Action Input: {"action":"resolve","intent":"web search"}'
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
    assert output == ""
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


def test_parse_react_response_accepts_full_width_colons_and_chinese_markers():
    parsed = parse_react_response(
        "思考：需要读取时间\n行动：system.current_time\n行动输入：{}"
    )
    assert parsed.action == "system.current_time"
    assert parsed.arguments == {}
    assert parsed.thought == "需要读取时间"
    assert not parsed.is_final

    final = parse_react_response("一些推理。\n最终答案：现在是 12:00。")
    assert final.is_final
    assert final.final_answer == "现在是 12:00。"


def test_parse_react_response_extracts_json_surrounded_by_prose():
    parsed = parse_react_response(
        'Thought: search\nAction: web.search\n'
        'Action Input: The JSON is {"query": "python"} as requested.'
    )
    assert parsed.arguments == {"query": "python"}
    assert parsed.error is None


def test_parse_react_response_repairs_single_quoted_python_literals():
    parsed = parse_react_response(
        "Thought: search\nAction: web.search\nAction Input: {'query': 'python'}"
    )
    assert parsed.arguments == {"query": "python"}
    assert parsed.error is None


def test_parse_react_response_reports_unclosed_json_as_action_error():
    parsed = parse_react_response(
        "Thought: search\nAction: web.search\nAction Input: {not json"
    )
    assert parsed.action == "web.search"
    assert parsed.arguments is None
    assert parsed.error is not None
    assert "not valid JSON" in parsed.error


@pytest.mark.asyncio
async def test_unmarked_answer_retry_limit_accepts_answer_instead_of_looping():
    llm = AlwaysUnmarkedAnswerLLM()
    agent = ReActAgent(
        "unmarked-limit-test",
        llm=llm,
        auto_discover_tools=False,
        lazy_tools=False,
    )
    agent.register_tool(CurrentTimeTool())

    answer = await agent.run_with_react(
        "请查询一下更新日志",
        max_rounds=None,
        defer_tool_loading=False,
    )

    assert answer.startswith("我直接根据记忆回答")
    # 3 protocol corrections, then the answer is accepted on the next round.
    assert llm.calls == 4


@pytest.mark.asyncio
async def test_malformed_action_repetition_stops_with_repeat_call_guard():
    agent = ReActAgent(
        "malformed-limit-test",
        llm=AlwaysMalformedActionLLM(),
        auto_discover_tools=False,
        lazy_tools=False,
    )
    agent.register_tool(CurrentTimeTool())

    with pytest.raises(RuntimeError, match="repeated more than three times"):
        await agent.run_with_react(
            "现在是什么时候",
            max_rounds=None,
            defer_tool_loading=False,
        )


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
    assert "模型思考：use the echo tool" in output
    assert "调用工具：test.react_echo" in output
    assert "ReAct round" not in output
    assert "model completed" not in output
    assert "Action Input" not in output
    assert '"value": 7' not in output
    assert all(
        "All registered tool names: system.tool_catalog, test.react_echo"
        in request[0]["content"]
        for request in llm.requests
    )


@pytest.mark.asyncio
async def test_react_exposes_registered_names_and_catalog_schema_by_default():
    llm = CatalogFirstEchoLLM()
    agent = ReActAgent(
        "catalog-first-default-test",
        llm=llm,
        auto_discover_tools=False,
    )
    agent.register_tool(EchoTool())
    assert "test.react_echo" in agent.tools

    answer = await agent.run_with_react("echo 7", max_rounds=4)

    assert answer == "done"
    first_instruction = llm.requests[0][0]["content"]
    assert "All registered tool names: system.tool_catalog, test.react_echo" in first_instruction
    assert "- system.tool_catalog:" in first_instruction
    assert "Action Input schema:" in first_instruction
    assert "- test.react_echo:" not in first_instruction
    assert "already registered and usable" in first_instruction
    second_instruction = llm.requests[1][0]["content"]
    assert "test.react_echo:" in second_instruction
    assert "test.react_echo" in agent.tools


@pytest.mark.asyncio
async def test_catalog_first_direct_call_requests_schema_without_denying_tool():
    llm = DirectThenCatalogEchoLLM()
    agent = ReActAgent(
        "catalog-first-recovery-test",
        llm=llm,
        auto_discover_tools=False,
    )
    agent.register_tool(EchoTool())

    answer = await agent.run_with_react("echo 7", max_rounds=5)

    assert answer == "done"
    first_observation = llm.requests[1][-1]["content"]
    assert "TOOL_SCHEMA_REQUIRED" in first_observation
    assert "registered and usable" in first_observation


@pytest.mark.asyncio
async def test_recommended_preceding_tool_is_advisory_only():
    llm = SearchBeforeTimeLLM()
    search = SearchProbeTool()
    agent = ReActAgent(
        "time-before-search-test",
        llm=llm,
        auto_discover_tools=False,
    )
    agent.register_tool(CurrentTimeTool())
    agent.register_tool(search)

    answer = await agent.run_with_react(
        "Give me the latest model news from the last two days",
        max_rounds=5,
        defer_tool_loading=False,
    )

    assert answer == "done"
    assert search.queries == [
        "latest model news",
        "model news in the last two days",
    ]
    first_instruction = llm.requests[0][0]["content"]
    assert "Recommended preceding tools (advisory only; not enforced): system.current_time." in first_instruction
    first_observation = llm.requests[1][-1]["content"]
    assert '"ok": true' in first_observation
    assert "TEMPORAL_CONTEXT_REQUIRED" not in first_observation


@pytest.mark.asyncio
async def test_catalog_lookup_does_not_inject_tools_from_user_wording():
    llm = TemporalCatalogOmitsTimeLLM()
    agent = ReActAgent(
        "time-aware-catalog-test",
        llm=llm,
        auto_discover_tools=False,
    )
    agent.register_tool(CurrentTimeTool())
    agent.register_tool(SearchProbeTool())

    answer = await agent.run_with_react("latest model news", max_rounds=3)

    assert answer == "done"
    second_instruction = llm.requests[1][0]["content"]
    assert "web.search:" in second_instruction
    assert "Recommended preceding tools (advisory only; not enforced): system.current_time." in second_instruction
    assert "system.current_time:" not in second_instruction


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
    assert "调用工具：test.react_echo" in output
    assert "ReAct round" not in output
    assert '"value": 7' not in output


@pytest.mark.asyncio
async def test_react_accepts_unlimited_rounds_when_max_rounds_is_none():
    llm = ReActLoggingLLM()
    agent = ReActAgent(
        "unlimited-rounds-test",
        llm=llm,
        auto_discover_tools=False,
        lazy_tools=False,
    )
    agent.register_tool(EchoTool())

    answer = await agent.run_with_react(
        "echo 7", max_rounds=None, defer_tool_loading=False
    )

    assert answer == "done"
