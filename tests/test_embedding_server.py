"""Unit tests for EmbedServerEmbedding (custom /embed gateway over a forwarded port)."""

from __future__ import annotations

import pytest

from memory import EmbedServerEmbedding


def test_validation_rejects_bad_arguments():
    with pytest.raises(ValueError, match="base_url"):
        EmbedServerEmbedding(base_url="")
    with pytest.raises(ValueError, match="dimension"):
        EmbedServerEmbedding(dimension=0)
    with pytest.raises(ValueError, match="timeout"):
        EmbedServerEmbedding(timeout=-1)
    with pytest.raises(ValueError, match="batch_size"):
        EmbedServerEmbedding(batch_size=0)


def test_defaults_point_at_local_gateway():
    instance = EmbedServerEmbedding()
    assert instance.base_url == "http://127.0.0.1:10800"
    assert instance.dimension == 0  # learned from the first response
    assert instance.api_key == ""


def test_embed_single_text():
    instance = EmbedServerEmbedding(
        client=lambda payload: {"embeddings": [[1.0, 2.0]]}
    )
    assert instance.embed("solo") == [1.0, 2.0]
    assert instance.dimension == 2


def test_embed_batch_uses_custom_shape_and_learns_dimension():
    calls: list[dict] = []

    def fake_client(payload):
        calls.append(payload)
        return {"dim": 3, "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "loaded": True}

    instance = EmbedServerEmbedding(client=fake_client)
    vectors = instance.embed_batch(["hello", "world"])

    assert calls == [{"texts": ["hello", "world"]}]
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert instance.dimension == 3


def test_batch_splits_by_batch_size():
    received: list[list[str]] = []
    counter = {"n": 0}

    def fake_client(payload):
        received.append(list(payload["texts"]))
        start = counter["n"]
        counter["n"] += len(payload["texts"])
        return {"embeddings": [[float(start + i)] for i in range(len(payload["texts"]))]}

    instance = EmbedServerEmbedding(client=fake_client, batch_size=2)
    vectors = instance.embed_batch(["a", "b", "c", "d", "e"])

    assert received == [["a", "b"], ["c", "d"], ["e"]]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_response_count_mismatch_raises():
    instance = EmbedServerEmbedding(
        client=lambda payload: {"embeddings": [[1.0]]}
    )
    with pytest.raises(RuntimeError, match="count 1 did not match input count 2"):
        instance.embed_batch(["a", "b"])


def test_expected_dimension_mismatch_raises():
    instance = EmbedServerEmbedding(
        dimension=4,
        client=lambda payload: {"embeddings": [[1.0, 2.0, 3.0]]},
    )
    with pytest.raises(RuntimeError, match="dimension 3 does not match expected dimension 4"):
        instance.embed("x")


def test_empty_batch_returns_empty():
    instance = EmbedServerEmbedding(client=lambda payload: {"embeddings": []})
    assert instance.embed_batch([]) == []


def test_missing_embeddings_key_raises():
    instance = EmbedServerEmbedding(client=lambda payload: {"data": []})
    with pytest.raises(RuntimeError, match="no embeddings list"):
        instance.embed("x")


def test_non_finite_values_raise():
    instance = EmbedServerEmbedding(
        client=lambda payload: {"embeddings": [[float("nan"), 1.0]]}
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        instance.embed("x")


def test_inconsistent_dimensions_raise():
    instance = EmbedServerEmbedding(
        client=lambda payload: {"embeddings": [[1.0, 2.0], [3.0]]}
    )
    with pytest.raises(RuntimeError, match="inconsistent dimensions"):
        instance.embed_batch(["a", "b"])


def test_to_dict_reports_config():
    instance = EmbedServerEmbedding(base_url="http://127.0.0.1:10800", dimension=1024)
    assert instance.to_dict() == {
        "type": "EmbedServerEmbedding",
        "base_url": "http://127.0.0.1:10800",
        "dimension": 1024,
        "batch_size": 10,
    }