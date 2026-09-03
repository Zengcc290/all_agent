import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from agents.agent import Agent
from core import (
    BaseTool,
    ToolDiscoveryError,
    ToolRegistry,
    ToolSpec,
    ToolSpecRepository,
    discover_tools,
)

PLUGIN_SOURCE = """
from pydantic import BaseModel, ConfigDict
from core import BaseTool, ToolSpec

TOOL_ENABLED = True

class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int

class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int

class PluginTool(BaseTool):
    spec = ToolSpec(
        name="plugin.echo",
        description="Echo a value from a discovered plugin.",
        version="1.2.3",
        input_model=Input,
        output_model=Output,
    )

    def execute(self, arguments: Input) -> Output:
        return Output(value=arguments.value)

def create_tool() -> BaseTool:
    return PluginTool()
"""


def make_package(tmp_path: Path, monkeypatch, name: str, files: dict[str, str]) -> str:
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for filename, source in files.items():
        (package / filename).write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


@pytest.fixture(autouse=True)
def remove_test_packages():
    prefixes = ("sample_tools", "agent_tools", "duplicate_tools")
    yield
    for module_name in list(sys.modules):
        if module_name.startswith(prefixes):
            sys.modules.pop(module_name, None)


def test_discovery_registers_enabled_and_skips_disabled_factory(tmp_path, monkeypatch):
    package = make_package(
        tmp_path,
        monkeypatch,
        "sample_tools",
        {
            "echo.py": PLUGIN_SOURCE,
            "off.py": """
TOOL_ENABLED = False
def create_tool():
    raise AssertionError("disabled factories must not run")
""",
            "broken.py": "TOOL_ENABLED = True\n",
            "base.py": "raise AssertionError('base.py must not be imported')\n",
            "_private.py": "raise AssertionError('private modules must not import')\n",
        },
    )
    registry = ToolRegistry()
    repository = ToolSpecRepository(":memory:")

    report = discover_tools(registry, package=package, repository=repository)

    assert registry.is_registered("plugin.echo", version="1.2.3")
    assert registry.registration_status("plugin.echo")["generation"] == 1
    assert report.for_tool("plugin.echo").status == "registered"
    assert (
        next(
            record for record in report.records if record.module.endswith(".off")
        ).status
        == "disabled"
    )
    assert (
        next(
            record for record in report.records if record.module.endswith(".base")
        ).status
        == "ignored"
    )
    assert len(report.errors) == 1
    assert "create_tool" in report.errors[0].error
    stored = repository.get("plugin.echo", "1.2.3")
    assert stored["implementation_ref"].endswith(":PluginTool")

    second = discover_tools(registry, package=package, repository=repository)
    assert second.for_tool("plugin.echo").status == "already_registered"
    assert registry.registration_status("plugin.echo")["generation"] == 1

    with pytest.raises(ToolDiscoveryError) as caught:
        discover_tools(registry, package=package, strict=True)
    assert caught.value.report.errors
    repository.close()


class LocalInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class LocalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class ConflictingTool(BaseTool):
    spec = ToolSpec(
        name="plugin.echo",
        description="A different implementation with the same registered name.",
        version="9.0",
        input_model=LocalInput,
        output_model=LocalOutput,
    )

    def execute(self, arguments: LocalInput) -> LocalOutput:
        return LocalOutput(value=arguments.value + 1)


def test_discovery_detects_duplicates_and_can_replace(tmp_path, monkeypatch):
    package = make_package(
        tmp_path,
        monkeypatch,
        "duplicate_tools",
        {"echo.py": PLUGIN_SOURCE},
    )
    registry = ToolRegistry()
    registry.register(ConflictingTool())

    conflict = discover_tools(registry, package=package)

    assert conflict.errors[0].tool_name == "plugin.echo"
    assert "different active implementation" in conflict.errors[0].error
    assert registry.get("plugin.echo").spec.version == "9.0"

    replaced = discover_tools(registry, package=package, replace=True)

    assert replaced.for_tool("plugin.echo").status == "registered"
    assert registry.get("plugin.echo").spec.version == "1.2.3"
    assert registry.registration_status("plugin.echo")["generation"] == 2


class DemoAgent(Agent):
    def run(self, query: str) -> str:
        return query


def test_agent_auto_discovers_and_exposes_registration_queries(tmp_path, monkeypatch):
    package = make_package(
        tmp_path,
        monkeypatch,
        "agent_tools",
        {"echo.py": PLUGIN_SOURCE},
    )

    repository = ToolSpecRepository(":memory:")
    agent = DemoAgent(
        "discovery",
        tool_package=package,
        repository=repository,
    )

    assert agent.is_tool_registered("plugin.echo")
    assert not agent.is_tool_registered("plugin.missing")
    status = agent.tool_registration_status("plugin.echo")
    assert status["implementation"].endswith(":PluginTool")
    assert agent.tool_discovery_report.for_tool("plugin.echo").status == "registered"
    assert repository.get("plugin.echo", "1.2.3") is not None

    disabled = DemoAgent(
        "manual-only",
        tool_package=package,
        auto_discover_tools=False,
    )
    assert disabled.tool_discovery_report is None
    assert not disabled.is_tool_registered("plugin.echo")
    repository.close()
