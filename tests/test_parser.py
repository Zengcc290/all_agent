import json

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
