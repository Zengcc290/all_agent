import json

import pytest

from core import ToolRegistry, parse_openai_tool_calls, parse_tool_calls
from tool.search import SearchTool


def test_fallback_json_parser_accepts_a_batch():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    payload = json.dumps(
        {
            "tool_calls": [
                {
                    "call_id": "one",
                    "tool_name": search.spec.name,
                    "schema_version": search.spec.version,
                    "schema_hash": search.spec.schema_hash,
                    "arguments": {"query": "typed tools"},
                }
            ]
        }
    )
    calls = parse_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0].arguments["query"] == "typed tools"


def test_openai_parser_accepts_mapping_objects_and_name_aliases():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    calls = parse_openai_tool_calls(
        [
            {
                "id": "one",
                "function": {
                    "name": "web__search",
                    "arguments": '{"query": "typed tools"}',
                },
            }
        ],
        registry,
        {"web__search": "web.search"},
    )

    assert calls[0].tool_name == "web.search"
    assert calls[0].registry_generation == registry.resolve("web.search")[1]


def test_openai_parser_generates_call_id_when_gateway_omits_it():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    calls = parse_openai_tool_calls(
        [
            {
                "function": {
                    "name": "web__search",
                    "arguments": '{"query": "typed tools"}',
                }
            }
        ],
        registry,
        {"web__search": "web.search"},
    )

    assert calls[0].call_id == "native-call-1"


def test_openai_parser_extracts_json_object_surrounded_by_prose():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    calls = parse_openai_tool_calls(
        [
            {
                "id": "one",
                "function": {
                    "name": "web__search",
                    "arguments": '{"query": "python"} trailing text',
                },
            }
        ],
        registry,
        {"web__search": "web.search"},
    )

    assert calls[0].arguments == {"query": "python"}


def test_openai_parser_repairs_single_quoted_arguments():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    calls = parse_openai_tool_calls(
        [
            {
                "id": "one",
                "function": {
                    "name": "web__search",
                    "arguments": "{'query': 'python'}",
                },
            }
        ],
        registry,
        {"web__search": "web.search"},
    )

    assert calls[0].arguments == {"query": "python"}


def test_load_json_rejects_non_finite_constants_as_json_errors():
    """回归：`parse_constant` 拒绝 NaN/Infinity 时抛裸 ValueError，会绕过
    调用方的 JSONDecodeError 处理，最终变成难以理解的内部异常。"""

    from core.parser import _load_json

    for payload in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
        with pytest.raises(json.JSONDecodeError) as excinfo:
            _load_json(payload)
        # 原始原因仍保留在消息里，便于排查。
        assert "invalid JSON constant" in str(excinfo.value)


def test_fallback_parser_reports_non_finite_arguments_as_json_errors():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_calls('{"tool_name": "web.search", "arguments": {"limit": NaN}}')


def test_openai_parser_reports_non_finite_arguments_as_invalid_json():
    """原生调用路径把非法常量归一为与其它坏 JSON 相同的 ValueError。"""

    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)

    with pytest.raises(ValueError, match="invalid JSON arguments for tool"):
        parse_openai_tool_calls(
            [
                {
                    "id": "one",
                    "function": {
                        "name": "web.search",
                        "arguments": '{"query": "x", "limit": NaN}',
                    },
                }
            ],
            registry,
        )


def test_load_json_still_repairs_wrapped_and_single_quoted_payloads():
    """B5 的重构不得破坏原有的三段容错顺序。"""

    from core.parser import _load_json

    assert _load_json('  {"query": "python"}  ') == {"query": "python"}
    assert _load_json('preamble {"query": "python"} trailing') == {"query": "python"}
    assert _load_json("{'query': 'python'}") == {"query": "python"}
    assert _load_json("[1, 2]") == [1, 2]
