"""Unit tests for the shared agent message/tool-call conversion helpers."""

from __future__ import annotations

import json

from agents import message_utils


class FakeModel:
    """Stand-in for a provider SDK message object."""

    def __init__(self, payload: dict, *, mode_json_ok: bool = True) -> None:
        self.payload = payload
        self.mode_json_ok = mode_json_ok
        self.mode_json_calls = 0

    def model_dump(self, *, mode: str | None = None, exclude_none: bool = False):
        if mode == "json":
            self.mode_json_calls += 1
            if not self.mode_json_ok:
                raise TypeError("not JSON serializable")
        data = dict(self.payload)
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data


class BareObject:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_field_reads_mappings_and_objects():
    assert message_utils.field({"a": 1}, "a") == 1
    assert message_utils.field(BareObject(a=2), "a") == 2
    assert message_utils.field({"a": 1}, "missing", "fallback") == "fallback"
    assert message_utils.field(BareObject(), "missing") is None


def test_safe_tool_name_is_bounded_and_never_blank():
    assert message_utils.safe_tool_name("  fs.read_text  ") == "fs.read_text"
    assert message_utils.safe_tool_name("") == "unknown.tool"
    assert message_utils.safe_tool_name("   ") == "unknown.tool"
    assert message_utils.safe_tool_name(None) == "unknown.tool"
    assert message_utils.safe_tool_name(42) == "unknown.tool"
    assert len(message_utils.safe_tool_name("x" * 500)) == 200


def test_safe_tool_call_error_falls_back_to_the_exception_class():
    assert message_utils.safe_tool_call_error(ValueError("boom")) == "boom"
    assert message_utils.safe_tool_call_error(ValueError("")) == "ValueError"
    assert len(message_utils.safe_tool_call_error(ValueError("x" * 5000))) == 1000


def test_tool_call_dict_normalizes_and_drops_missing_ids():
    converted = message_utils.tool_call_dict(
        BareObject(
            id="call-1",
            type="function",
            function=BareObject(name="fs.read_text", arguments='{"path": "a"}'),
        )
    )
    assert converted == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "fs.read_text", "arguments": '{"path": "a"}'},
    }

    without_id = message_utils.tool_call_dict(
        BareObject(function={"name": "fs.read_dir", "arguments": "{}"})
    )
    assert "id" not in without_id
    assert without_id["type"] == "function"


def test_tool_call_dict_defaults_missing_arguments_to_empty_object():
    converted = message_utils.tool_call_dict(
        {"function": {"name": "fs.read_dir"}}
    )
    assert converted["function"]["arguments"] == "{}"


def test_message_dict_from_model_dump_normalizes_tool_calls():
    message = FakeModel(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "fs.read_text", "arguments": "{}"},
                }
            ],
        }
    )

    data = message_utils.message_dict(message)

    assert data["role"] == "assistant"
    # exclude_none=True 语义：content 为 None 时不写入。
    assert "content" not in data
    assert data["tool_calls"][0]["id"] == "call-1"
    assert data["tool_calls"][0]["type"] == "function"


def test_message_dict_from_mapping_copies_and_defaults_role():
    original = {"content": "hello"}
    data = message_utils.message_dict(original)

    assert data == {"content": "hello", "role": "assistant"}
    # 返回副本：修改结果不得污染原 dict。
    data["content"] = "changed"
    assert original["content"] == "hello"


def test_message_dict_from_bare_object_keeps_only_known_fields():
    data = message_utils.message_dict(BareObject(role="user", content="hi", extra="x"))

    assert data == {"role": "user", "content": "hi"}


def test_result_json_prefers_json_mode_and_falls_back():
    model = FakeModel({"value": 1})
    assert json.loads(message_utils.result_json(model)) == {"value": 1}
    assert model.mode_json_calls == 1

    fallback = FakeModel({"value": 2}, mode_json_ok=False)
    assert json.loads(message_utils.result_json(fallback)) == {"value": 2}
    # 先走 mode="json"，失败后回退到普通 model_dump()。
    assert fallback.mode_json_calls == 1


def test_agent_reexports_the_shared_helpers():
    """agent.py 保留私有别名，react.py 的既有导入路径不能断。"""

    import importlib

    # agents.__init__ 也导出了一个名为 ``agent`` 的工厂，必须按模块路径取。
    agent_module = importlib.import_module("agents.agent")

    assert agent_module._field is message_utils.field
    assert agent_module._message_dict is message_utils.message_dict
    assert agent_module._tool_call_dict is message_utils.tool_call_dict
    assert agent_module._result_json is message_utils.result_json
    assert agent_module._safe_tool_name is message_utils.safe_tool_name
    assert agent_module._safe_tool_call_error is message_utils.safe_tool_call_error


def test_llm_field_alias_matches_the_shared_helper():
    import importlib

    llm = importlib.import_module("agents.llm")

    assert llm._field({"a": 1}, "a") == 1
    assert llm._field(BareObject(a=2), "a") == 2
    assert llm._field(None, "a", "x") == "x"
