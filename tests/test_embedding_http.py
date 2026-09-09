"""End-to-end HTTP tests for APIEmbedding against a local mock /embeddings server.

The other APIEmbedding tests inject a Python client, which skips the real
urllib request path (headers, JSON body, HTTP status handling).  These tests
stand up a throwaway HTTP server on 127.0.0.1 so the transport layer is
actually exercised without needing a vendor API key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import pytest

from memory import APIEmbedding


class _EmbeddingsHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /embeddings endpoint."""

    requests: list[dict[str, Any]] = []
    responder: Callable[[dict[str, Any]], tuple[int, Any]] = staticmethod(
        lambda body: (200, {"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]})
    )

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            body = {"__unparsable__": raw.decode("utf-8", errors="replace")}
        type(self).requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        status, payload = type(self).responder(body)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:  # silence test output
        pass


@pytest.fixture()
def base_url() -> Any:
    _EmbeddingsHandler.requests = []
    _EmbeddingsHandler.responder = staticmethod(
        lambda body: (200, {"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]})
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_http_request_shape_and_parsing(base_url: str):
    instance = APIEmbedding(api_key="test-key", base_url=base_url, model="qwen3-embedding-0.6b")
    vectors = instance.embed_batch(["你好", "世界"])

    assert vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert instance.dimension == 2

    assert len(_EmbeddingsHandler.requests) == 1
    sent = _EmbeddingsHandler.requests[0]
    assert sent["path"] == "/v1/embeddings"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["headers"]["Content-Type"] == "application/json"
    assert sent["body"] == {"model": "qwen3-embedding-0.6b", "input": ["你好", "世界"]}


def test_real_http_batching_sends_multiple_requests(base_url: str):
    instance = APIEmbedding(api_key="test-key", base_url=base_url, batch_size=2)
    vectors = instance.embed_batch(["a", "b", "c", "d", "e"])

    assert len(_EmbeddingsHandler.requests) == 3
    assert [len(req["body"]["input"]) for req in _EmbeddingsHandler.requests] == [2, 2, 1]
    assert len(vectors) == 5


def test_real_http_error_status_is_surfaced(base_url: str):
    _EmbeddingsHandler.responder = staticmethod(
        lambda body: (401, {"error": {"message": "Invalid API-key provided."}})
    )
    instance = APIEmbedding(api_key="bad-key", base_url=base_url)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        instance.embed("text")


def test_real_http_invalid_json_is_surfaced(base_url: str):
    _EmbeddingsHandler.responder = staticmethod(lambda body: (200, b"<html>not json</html>"))
    instance = APIEmbedding(api_key="test-key", base_url=base_url)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        instance.embed("text")


def test_real_http_connection_refused_is_surfaced():
    # Port 1 is reserved/unbound: the request must fail with a clear message.
    instance = APIEmbedding(api_key="test-key", base_url="http://127.0.0.1:1/v1", timeout=2.0)

    with pytest.raises(RuntimeError, match="request failed"):
        instance.embed("text")