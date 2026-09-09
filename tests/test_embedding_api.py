"""Unit tests for the vendor-neutral APIEmbedding (OpenAI-compatible /embeddings)."""

from __future__ import annotations

import pytest

from memory import APIEmbedding, DEFAULT_EMBEDDING_BASE_URL, DEFAULT_EMBEDDING_MODEL


@pytest.fixture()
def embedding() -> APIEmbedding:
    return APIEmbedding(api_key="test-key")


def test_requires_api_key():
    # Use an env name that no .env file can provide, so the assertion holds even
    # when the developer's real .env contains DASHSCOPE_API_KEY.
    missing_env = "DASHSCOPE_API_KEY_MISSING_FOR_TEST"
    with pytest.raises(RuntimeError, match="API key"):
        APIEmbedding(api_key=None, api_key_env=missing_env)
    with pytest.raises(RuntimeError, match="API key"):
        APIEmbedding(api_key="  ", api_key_env=missing_env)


def test_accepts_api_key_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    instance = APIEmbedding()
    assert instance.api_key == "env-key"


def test_validation_rejects_bad_arguments():
    with pytest.raises(ValueError, match="model"):
        APIEmbedding(api_key="k", model="  ")
    with pytest.raises(ValueError, match="base_url"):
        APIEmbedding(api_key="k", base_url="")
    with pytest.raises(ValueError, match="dimension"):
        APIEmbedding(api_key="k", dimension=0)
    with pytest.raises(ValueError, match="timeout"):
        APIEmbedding(api_key="k", timeout=-1)
    with pytest.raises(ValueError, match="batch_size"):
        APIEmbedding(api_key="k", batch_size=0)


def test_defaults_point_at_qwen3_embedding():
    instance = APIEmbedding(api_key="k")
    assert instance.model == DEFAULT_EMBEDDING_MODEL == "qwen3-embedding-0.6b"
    assert instance.base_url == DEFAULT_EMBEDDING_BASE_URL
    assert instance.dimension == 0  # learned from the first response


def test_embed_batch_uses_openai_shape_and_learns_dimension():
    calls: list[dict] = []

    def fake_client(payload, **kwargs):
        calls.append(payload)
        return {
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }

    instance = APIEmbedding(api_key="k", client=fake_client)
    vectors = instance.embed_batch(["hello", "world"])

    assert calls == [{"model": "qwen3-embedding-0.6b", "input": ["hello", "world"]}]
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert instance.dimension == 3


def test_embed_single_text():
    instance = APIEmbedding(api_key="k", client=lambda payload, **kw: {"data": [{"embedding": [1.0, 2.0]}]})
    assert instance.embed("solo") == [1.0, 2.0]


def test_dashscope_native_output_layout_is_accepted():
    instance = APIEmbedding(
        api_key="k",
        client=lambda payload, **kw: {"output": {"embeddings": [{"embedding": [7.0], "text_index": 0}]}},
    )
    assert instance.embed("x") == [7.0]


def test_batch_splits_by_batch_size():
    received: list[list[str]] = []
    counter = {"n": 0}

    def fake_client(payload, **kwargs):
        received.append(list(payload["input"]))
        start = counter["n"]
        counter["n"] += len(payload["input"])
        return {"data": [{"embedding": [float(start + i)]} for i in range(len(payload["input"]))]}

    instance = APIEmbedding(api_key="k", client=fake_client, batch_size=2)
    vectors = instance.embed_batch(["a", "b", "c", "d", "e"])

    assert received == [["a", "b"], ["c", "d"], ["e"]]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_index_out_of_order_is_reordered():
    instance = APIEmbedding(
        api_key="k",
        client=lambda payload, **kw: {
            "data": [
                {"index": 1, "embedding": [2.0]},
                {"index": 0, "embedding": [1.0]},
            ]
        },
    )
    assert instance.embed_batch(["a", "b"]) == [[1.0], [2.0]]


def test_incomplete_indices_raise():
    # Two vectors both labelled index 0 -> positions [0, 0] can't cover [0, 1].
    instance = APIEmbedding(
        api_key="k",
        client=lambda payload, **kw: {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},
    )
    with pytest.raises(RuntimeError, match="indices"):
        instance.embed_batch(["a", "b"])


def test_count_mismatch_raises():
    instance = APIEmbedding(api_key="k", client=lambda payload, **kw: {"data": [{"embedding": [1.0]}]})
    with pytest.raises(RuntimeError, match="count"):
        instance.embed_batch(["a", "b"])


def test_dimension_mismatch_raises():
    instance = APIEmbedding(api_key="k", dimension=4, client=lambda payload, **kw: {"data": [{"embedding": [1.0, 2.0, 3.0]}]})
    with pytest.raises(RuntimeError, match="dimension"):
        instance.embed("text")


def test_invalid_vector_values_raise():
    instance = APIEmbedding(api_key="k", client=lambda payload, **kw: {"data": [{"embedding": [float("nan")]}]})
    with pytest.raises(RuntimeError, match="non-finite"):
        instance.embed("text")


def test_rejects_non_string_input():
    instance = APIEmbedding(api_key="k", client=lambda payload, **kw: {"data": []})
    with pytest.raises(TypeError, match="string"):
        instance.embed(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string"):
        instance.embed_batch(["ok", 7])  # type: ignore[list-item]


def test_repr_and_to_dict_are_safe():
    instance = APIEmbedding(api_key="k")
    assert instance.model in repr(instance)
    data = instance.to_dict()
    assert data["type"] == "APIEmbedding"
    assert data["model"] == DEFAULT_EMBEDDING_MODEL
    assert "api_key" not in data