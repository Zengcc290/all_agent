import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from core import ExecutionContext, ToolCall, ToolExecutionManager, ToolRegistry
from tool.search import TOOL_ENABLED, SearchInput, SearchTool, create_tool


def search_call(tool):
    return ToolCall(
        call_id="search",
        tool_name=tool.spec.name,
        schema_version=tool.spec.version,
        schema_hash=tool.spec.schema_hash,
        arguments={"query": "typed tools", "limit": 1},
    )


def test_search_implements_the_discovery_protocol():
    assert isinstance(TOOL_ENABLED, bool)
    assert isinstance(create_tool(), SearchTool)


def test_search_timeout_updates_the_runtime_contract():
    tool = SearchTool(base_url="https://example.invalid", timeout=1.25)
    assert tool.timeout == 1.25
    assert tool.spec.timeout_seconds == 1.25


def test_search_input_normalizes_nullable_values_from_compatible_models():
    arguments = SearchInput.model_validate(
        {
            "query": "typed tools",
            "tag": "web.search",
            "params": "{}",
            "language": "None",
        },
        strict=True,
    )

    assert arguments.tag is None
    assert arguments.params == {}
    assert arguments.language is None


def test_search_input_keeps_rejecting_unparseable_provider_params():
    with pytest.raises(ValueError, match="params"):
        SearchInput.model_validate(
            {"query": "typed tools", "params": "not-json"}, strict=True
        )


def test_search_normalizes_a_local_http_response():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["query"] = parse_qs(urlsplit(self.path).query)
            seen["authorization"] = self.headers.get("Authorization")
            body = json.dumps(
                {
                    "results": [
                        {
                            "title": "Result",
                            "link": "https://example.test/result",
                            "description": "Snippet",
                        }
                    ]
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tool = SearchTool(
            base_url=(f"http://127.0.0.1:{server.server_port}/search?lang=zh#ignored"),
            api_key="test-key",
        )
        registry = ToolRegistry()
        registry.register(tool)
        result = asyncio.run(
            ToolExecutionManager(registry).execute_batch(
                [search_call(tool)],
                ExecutionContext(permissions=frozenset({"network.read"})),
            )
        ).results[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result.ok
    assert result.data == {
        "items": [
            {
                "title": "Result",
                "url": "https://example.test/result",
                "snippet": "Snippet",
            }
        ]
    }
    assert seen["query"] == {
        "lang": ["zh"],
        "q": ["typed tools"],
        "limit": ["1"],
    }
    assert seen["authorization"] == "Bearer test-key"
