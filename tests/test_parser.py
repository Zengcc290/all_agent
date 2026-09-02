import json

from core import ToolRegistry, parse_tool_calls
from tool.search import SearchTool


def test_fallback_json_parser_accepts_a_batch():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    payload = json.dumps({
        "tool_calls": [{
            "call_id": "one",
            "tool_name": search.spec.name,
            "schema_version": search.spec.version,
            "schema_hash": search.spec.schema_hash,
            "arguments": {"query": "typed tools"},
        }]
    })
    calls = parse_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0].arguments["query"] == "typed tools"
