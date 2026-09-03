import importlib
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agents.agent import Agent
from core import ToolSpec
from core.registry import BaseTool


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class EchoTool(BaseTool):
    spec = ToolSpec(
        name="test.agent_echo",
        description="Echo a value",
        version="1.0",
        input_model=Input,
        output_model=Output,
    )

    def execute(self, arguments: Input) -> Output:
        return Output(value=arguments.value)


class ReplacementEchoTool(EchoTool):
    spec = ToolSpec(
        name="test.agent_echo",
        description="Replacement echo implementation",
        version="2.0",
        input_model=Input,
        output_model=Output,
    )

    def execute(self, arguments: Input) -> Output:
        return Output(value=arguments.value + 1)


class FakeLLM:
    model = "fake-model"

    def __init__(self, tool_name: str, schema_hash: str) -> None:
        self.requests: list[list[dict]] = []
        self.options: list[dict] = []
        self.tool_name = tool_name
        self.schema_hash = schema_hash
        self.count = 0

    def complete(self, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        self.options.append(dict(kwargs))
        self.count += 1
        if self.count == 1:
            call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name=self.tool_name.replace(".", "__"), arguments='{"value": 7}'
                ),
            )
            message = SimpleNamespace(role="assistant", content=None, tool_calls=[call])
        else:
            message = SimpleNamespace(role="assistant", content="done", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DemoAgent(Agent):
    def run(self, query: str) -> str:
        return query


@pytest.mark.asyncio
async def test_run_with_tools_executes_and_returns_final_answer():
    tool = EchoTool()
    llm = FakeLLM(tool.spec.name, tool.spec.schema_hash)
    agent = DemoAgent("test", llm=llm)
    agent.register_tool(tool)

    answer = await agent.run_with_tools([{"role": "user", "content": "echo 7"}])

    assert answer == "done"
    assert len(llm.requests) == 2
    assert llm.requests[1][-1]["role"] == "tool"
    assert '"value": 7' in llm.requests[1][-1]["content"]
    assert llm.options[0]["tools"][0]["function"]["name"] == "system__tool_catalog"


def test_tool_definitions_are_schema_driven():
    agent = DemoAgent("test", llm=object())
    tool = EchoTool()
    agent.register_tool(tool)

    definition = next(
        item
        for item in agent.tool_definitions()
        if item["function"]["name"] == tool.spec.name.replace(".", "__")
    )
    assert definition["function"]["parameters"] == tool.spec.input_schema
    assert definition["function"]["strict"] is True
    assert "." not in definition["function"]["name"]

    catalog = next(
        item
        for item in agent.tool_definitions()
        if item["function"]["name"] == "system__tool_catalog"
    )
    parameters = catalog["function"]["parameters"]
    assert set(parameters["required"]) == set(parameters["properties"])
    assert parameters["additionalProperties"] is False


@pytest.mark.asyncio
async def test_tool_call_is_bound_to_the_definition_snapshot():
    tool = EchoTool()
    agent = DemoAgent("test", llm=None)
    agent.register_tool(tool)

    class SwappingLLM(FakeLLM):
        def complete(self, messages, **kwargs):
            if self.count == 0:
                agent.register_tool(ReplacementEchoTool(), replace=True)
            return super().complete(messages, **kwargs)

    llm = SwappingLLM(tool.spec.name, tool.spec.schema_hash)
    agent.llm = llm

    await agent.run_with_tools([{"role": "user", "content": "echo 7"}])

    result = json.loads(agent.history[-2]["content"])
    assert result["ok"] is False
    assert result["error"]["code"] == "SCHEMA_MISMATCH"


@pytest.mark.asyncio
async def test_malformed_arguments_are_returned_to_the_model():
    tool = EchoTool()

    class MalformedLLM(FakeLLM):
        def complete(self, messages, **kwargs):
            self.requests.append([dict(message) for message in messages])
            self.options.append(dict(kwargs))
            self.count += 1
            if self.count == 1:
                function = SimpleNamespace(
                    name=self.tool_name.replace(".", "__"), arguments="{"
                )
                call = SimpleNamespace(id="bad-call", function=function)
                message = SimpleNamespace(
                    role="assistant", content=None, tool_calls=[call]
                )
            else:
                message = SimpleNamespace(
                    role="assistant", content="recovered", tool_calls=[]
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    llm = MalformedLLM(tool.spec.name, tool.spec.schema_hash)
    agent = DemoAgent("test", llm=llm)
    agent.register_tool(tool)

    answer = await agent.run_with_tools([{"role": "user", "content": "echo"}])

    assert answer == "recovered"
    result = json.loads(llm.requests[1][-1]["content"])
    assert result["error"]["code"] == "INVALID_TOOL_CALL"


@pytest.mark.asyncio
async def test_deferred_catalog_loads_a_resolved_tool_for_the_next_round():
    tool = EchoTool()

    class DiscoveryLLM:
        model = "fake"

        def __init__(self):
            self.count = 0
            self.exposed_names = []

        def complete(self, messages, **kwargs):
            self.exposed_names.append(
                [item["function"]["name"] for item in kwargs["tools"]]
            )
            self.count += 1
            if self.count == 1:
                arguments = json.dumps(
                    {
                        "action": "resolve",
                        "intent": "echo value",
                        "tool_name": None,
                        "version": None,
                        "limit": 5,
                    }
                )
                function = SimpleNamespace(
                    name="system__tool_catalog", arguments=arguments
                )
                call = SimpleNamespace(id="catalog-call", function=function)
                message = SimpleNamespace(
                    role="assistant", content=None, tool_calls=[call]
                )
            elif self.count == 2:
                function = SimpleNamespace(
                    name="test__agent_echo", arguments='{"value": 9}'
                )
                call = SimpleNamespace(id="echo-call", function=function)
                message = SimpleNamespace(
                    role="assistant", content=None, tool_calls=[call]
                )
            else:
                message = SimpleNamespace(
                    role="assistant", content="done", tool_calls=[]
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    llm = DiscoveryLLM()
    agent = DemoAgent("test", llm=llm)
    agent.register_tool(tool)

    answer = await agent.run_with_tools(
        [{"role": "user", "content": "echo 9"}], defer_tool_loading=True
    )

    assert answer == "done"
    assert llm.exposed_names[0] == ["system__tool_catalog"]
    assert "test__agent_echo" in llm.exposed_names[1]


@pytest.mark.asyncio
async def test_configured_prompts_and_history_are_used():
    class FinalLLM:
        model = "fake"

        def __init__(self):
            self.requests = []

        def complete(self, messages, **kwargs):
            self.requests.append([dict(message) for message in messages])
            message = SimpleNamespace(role="assistant", content="done", tool_calls=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    llm = FinalLLM()
    agent = DemoAgent("test", llm=llm)
    agent.set_system_prompt("Follow the configured system prompt.")

    await agent.run_with_tools([{"role": "user", "content": "first"}])
    await agent.run_with_tools(
        [{"role": "user", "content": "second"}], use_history=True
    )

    assert llm.requests[0][0]["role"] == "system"
    assert llm.requests[1][-2]["content"] == "done"
    assert llm.requests[1][-1]["content"] == "second"


@pytest.mark.asyncio
async def test_registered_provider_is_used_for_completion(monkeypatch):
    created = []

    class ProviderLLM:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def complete(self, messages, **kwargs):
            message = SimpleNamespace(
                role="assistant", content="provider answer", tool_calls=[]
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    agent_module = importlib.import_module("agents.agent")
    monkeypatch.setattr(agent_module, "LLM", ProviderLLM)
    agent = DemoAgent("test")
    monkeypatch.setattr(agent, "detect_models", lambda *_: ["model-a"])
    agent.add_provider("provider-a", "api-key", "https://example.invalid/v1", "model-a")

    answer = await agent.run_with_tools(
        [{"role": "user", "content": "hello"}], provider_name="provider-a"
    )

    assert answer == "provider answer"
    assert created[0]["base_url"] == "https://example.invalid/v1"
    assert created[0]["model"] == "model-a"
