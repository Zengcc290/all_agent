"""Gemini ``:embedContent`` embedding: 文本 / 图片 / 图文混合 三种输入。

协议要点（与 OpenAI 兼容的 ``/embeddings`` 不同，必须逐项锁死）：
``{base_url}/models/{model}:embedContent``、``x-goog-api-key`` 头、
``{"content": {"parts": [...]}}`` 请求体、``embedding.values`` 响应、
可选 ``config.outputDimensionality`` 降维。
"""

from __future__ import annotations

import json
from typing import Self

import pytest

from memory import (
    DEFAULT_GEMINI_EMBEDDING_BASE_URL,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    GeminiEmbedding,
    HashEmbedding,
    MemoryConfig,
    PerceptualMemory,
)
from memory import base as memory_base
from memory import embedding as embedding_module

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"fake-jpeg-body"


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _capture_http(monkeypatch: pytest.MonkeyPatch, payload: dict, captured: list) -> None:
    """Record the outgoing Request and reply with ``payload``."""

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        return _FakeResponse(payload)

    monkeypatch.setattr(embedding_module.urllib.request, "urlopen", fake_urlopen)


def _request_body(request) -> dict:
    return json.loads(request.data.decode("utf-8"))


def _request_headers(request) -> dict:
    return {key.lower(): value for key, value in request.headers.items()}


def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 .env / 真实 key 挡在外面，让选型结果确定。"""
    # 两个命名空间都要挡：base 里的是给 MemoryConfig 用的，embedding 里的是给
    # 脚本直接 ``from memory.embedding import load_dotenv_once`` 用的。
    monkeypatch.setattr(memory_base, "load_dotenv_once", lambda: None)
    monkeypatch.setattr(embedding_module, "load_dotenv_once", lambda: None)
    for name in ("DASHSCOPE_API_KEY", "GEMINI_API_KEY", "EMBEDDING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- 基础


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        GeminiEmbedding(api_key=None, api_key_env="GEMINI_API_KEY_MISSING_FOR_TEST")
    with pytest.raises(RuntimeError, match="API key"):
        GeminiEmbedding(api_key="   ", api_key_env="GEMINI_API_KEY_MISSING_FOR_TEST")


def test_accepts_api_key_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert GeminiEmbedding().api_key == "env-key"


def test_validation_rejects_bad_arguments():
    with pytest.raises(ValueError, match="model"):
        GeminiEmbedding(api_key="k", model="  ")
    with pytest.raises(ValueError, match="base_url"):
        GeminiEmbedding(api_key="k", base_url="")
    with pytest.raises(ValueError, match="dimension"):
        GeminiEmbedding(api_key="k", dimension=0)
    with pytest.raises(ValueError, match="timeout"):
        GeminiEmbedding(api_key="k", timeout=-1)


def test_defaults_point_at_gemini_embedding_2():
    instance = GeminiEmbedding(api_key="k")
    assert instance.model == DEFAULT_GEMINI_EMBEDDING_MODEL == "gemini-embedding-2"
    assert instance.base_url == DEFAULT_GEMINI_EMBEDDING_BASE_URL
    assert instance.dimension == 0  # learned from the first response


# ----------------------------------------------------------------- 三种输入形状


def test_text_request_matches_the_documented_wire_format(monkeypatch: pytest.MonkeyPatch):
    captured: list = []
    _capture_http(monkeypatch, {"embedding": {"values": [0.1, 0.2, 0.3]}}, captured)

    embedding = GeminiEmbedding(api_key="secret")
    vector = embedding.embed("这是一段需要向量化的中文文本")

    assert vector == [0.1, 0.2, 0.3]
    assert embedding.dimension == 3
    request = captured[0]
    assert request.full_url == f"{DEFAULT_GEMINI_EMBEDDING_BASE_URL}/models/gemini-embedding-2:embedContent"
    assert request.get_method() == "POST"
    assert _request_headers(request)["x-goog-api-key"] == "secret"
    assert "authorization" not in _request_headers(request)
    # 没配维度就不下发 outputDimensionality，用服务端默认 3072。
    assert _request_body(request) == {"content": {"parts": [{"text": "这是一段需要向量化的中文文本"}]}}


def test_image_request_uses_inline_data_base64(monkeypatch: pytest.MonkeyPatch):
    captured: list = []
    _capture_http(monkeypatch, {"embedding": {"values": [1.0]}}, captured)

    GeminiEmbedding(api_key="k").embed_image(PNG_BYTES, mime_type="image/png")

    parts = _request_body(captured[0])["content"]["parts"]
    assert len(parts) == 1
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    import base64

    assert base64.b64decode(parts[0]["inline_data"]["data"]) == PNG_BYTES


def test_mixed_request_carries_both_parts(monkeypatch: pytest.MonkeyPatch):
    captured: list = []
    _capture_http(monkeypatch, {"embedding": {"values": [2.0]}}, captured)

    GeminiEmbedding(api_key="k").embed_multimodal("商品描述：白色运动鞋", PNG_BYTES, mime_type="image/png")

    parts = _request_body(captured[0])["content"]["parts"]
    assert parts[0] == {"text": "商品描述：白色运动鞋"}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"


def test_accepts_pre_encoded_base64_image():
    embedding = GeminiEmbedding(api_key="k")
    part = embedding.image_part("QUJD", mime_type="image/jpeg")
    assert part == {"inline_data": {"mime_type": "image/jpeg", "data": "QUJD"}}


def test_rejects_empty_parts():
    embedding = GeminiEmbedding(api_key="k")
    with pytest.raises(ValueError, match="text/image"):
        embedding.parts_for("   ")
    with pytest.raises(TypeError, match="bytes or a non-empty base64"):
        embedding.image_part("")
    with pytest.raises(TypeError, match="bytes or a non-empty base64"):
        embedding.image_part(12345)
    with pytest.raises(TypeError, match="string"):
        embedding.embed(123)  # type: ignore[arg-type]


# ------------------------------------------------------------------ 维度与响应


def test_output_dimensionality_sent_only_when_dimension_configured(monkeypatch: pytest.MonkeyPatch):
    captured: list = []
    # 服务端按 outputDimensionality 截断，所以回一个 768 维的向量。
    _capture_http(monkeypatch, {"embedding": {"values": [0.5] * 768}}, captured)

    vector = GeminiEmbedding(api_key="k", dimension=768).embed("x")

    assert len(vector) == 768
    assert _request_body(captured[0])["config"] == {"outputDimensionality": 768}


def test_learns_dimension_then_enforces_it(monkeypatch: pytest.MonkeyPatch):
    captured: list = []
    _capture_http(monkeypatch, {"embedding": {"values": [0.5, 0.6]}}, captured)
    embedding = GeminiEmbedding(api_key="k")
    embedding.embed("first")
    assert embedding.dimension == 2
    # 学到维度后，下一次请求会把 outputDimensionality 带上，保持同一向量空间。
    embedding.embed("second")
    assert _request_body(captured[1])["config"] == {"outputDimensionality": 2}


def test_dimension_mismatch_raises():
    instance = GeminiEmbedding(api_key="k", dimension=4, client=lambda payload, **kw: {"embedding": {"values": [1.0, 2.0]}})
    with pytest.raises(RuntimeError, match="dimension"):
        instance.embed("text")


def test_batch_embeddings_endpoint_layout_is_accepted():
    instance = GeminiEmbedding(api_key="k", client=lambda payload, **kw: {"embeddings": [{"values": [7.0, 8.0]}]})
    assert instance.embed("x") == [7.0, 8.0]


def test_response_without_values_raises():
    instance = GeminiEmbedding(api_key="k", client=lambda payload, **kw: {"embedding": {}})
    with pytest.raises(RuntimeError, match="embedding.values"):
        instance.embed("x")


def test_non_finite_values_raise():
    instance = GeminiEmbedding(api_key="k", client=lambda payload, **kw: {"embedding": {"values": [float("nan")]}})
    with pytest.raises(RuntimeError, match="non-finite"):
        instance.embed("x")
    instance = GeminiEmbedding(api_key="k", client=lambda payload, **kw: {"embedding": {"values": ["x"]}})
    with pytest.raises(RuntimeError, match="non-finite"):
        instance.embed("x")


def test_batch_rejects_non_string_input():
    instance = GeminiEmbedding(api_key="k", client=lambda payload, **kw: {"embedding": {"values": [1.0]}})
    with pytest.raises(TypeError, match="strings"):
        instance.embed_batch(["ok", 7])  # type: ignore[list-item]
    assert instance.embed_batch(["a", "b"]) == [[1.0], [1.0]]


def test_repr_and_to_dict_are_safe():
    instance = GeminiEmbedding(api_key="k")
    assert instance.model in repr(instance)
    data = instance.to_dict()
    assert data["type"] == "GeminiEmbedding"
    assert data["model"] == DEFAULT_GEMINI_EMBEDDING_MODEL
    assert "api_key" not in data


# ------------------------------------------------------------------ 记忆写入路径


def test_embed_item_routes_the_three_shapes():
    seen: list[dict] = []

    def client(payload, **kwargs):
        seen.append(payload)
        return {"embedding": {"values": [1.0]}}

    embedding = GeminiEmbedding(api_key="k", client=client)
    embedding.embed_item("只有文字")
    embedding.embed_item("", payload=PNG_BYTES, modality="image")
    embedding.embed_item("会议室照片", payload=PNG_BYTES, modality="image")

    assert seen[0]["content"]["parts"] == [{"text": "只有文字"}]
    assert seen[1]["content"]["parts"][0]["inline_data"]["mime_type"] == "image/png"
    assert seen[2]["content"]["parts"][0] == {"text": "会议室照片"}
    assert "inline_data" in seen[2]["content"]["parts"][1]


def test_image_mime_is_sniffed_from_magic_bytes():
    seen: list[dict] = []

    def client(payload, **kwargs):
        seen.append(payload)
        return {"embedding": {"values": [1.0]}}

    embedding = GeminiEmbedding(api_key="k", client=client)
    embedding.embed_item("", payload=JPEG_BYTES, modality="image")
    embedding.embed_item("", payload=b"RIFF\x00\x00\x00\x00WEBPVP8 ", modality="image")
    embedding.embed_item("", payload=b"unknown-bytes", modality="image")

    assert seen[0]["content"]["parts"][0]["inline_data"]["mime_type"] == "image/jpeg"
    assert seen[1]["content"]["parts"][0]["inline_data"]["mime_type"] == "image/webp"
    assert seen[2]["content"]["parts"][0]["inline_data"]["mime_type"] == "image/png"


def test_memory_add_embeds_image_payload_multimodally():
    seen: list[dict] = []

    def client(payload, **kwargs):
        seen.append(payload)
        return {"embedding": {"values": [0.25, 0.5]}}

    memory = PerceptualMemory(
        embedding=GeminiEmbedding(api_key="k", client=client),
        config=MemoryConfig(sqlite_path=":memory:"),
    )
    item = memory.add("会议室照片", payload=PNG_BYTES, modality="image")

    assert item.embedding == [0.25, 0.5]
    assert item.payload == PNG_BYTES
    assert seen[0]["content"]["parts"][0] == {"text": "会议室照片"}
    assert seen[0]["content"]["parts"][1]["inline_data"]["mime_type"] == "image/png"


def test_text_only_backend_ignores_payload():
    """没有多模态能力的后端必须照旧只嵌文字，不能因为带了 payload 就崩。"""

    memory = PerceptualMemory(
        embedding=HashEmbedding(8),
        config=MemoryConfig(sqlite_path=":memory:"),
    )
    item = memory.add("会议室照片", payload=PNG_BYTES, modality="image")

    assert len(item.embedding) == 8
    assert item.payload == PNG_BYTES


# -------------------------------------------------------------------- 提供方选型


def test_provider_gemini_explicitly(monkeypatch: pytest.MonkeyPatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_API_KEY", "k")

    embedding = memory_base.make_default_embedding(MemoryConfig.from_env())

    assert isinstance(embedding, GeminiEmbedding)
    # 没填 base_url/model 时补成 Gemini 的值，而不是留下 qwen 的默认值。
    assert embedding.base_url == DEFAULT_GEMINI_EMBEDDING_BASE_URL
    assert embedding.model == DEFAULT_GEMINI_EMBEDDING_MODEL


def test_auto_detects_gemini_from_model_name(monkeypatch: pytest.MonkeyPatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_MODEL", "gemini-embedding-2")

    embedding = memory_base.make_default_embedding(MemoryConfig.from_env())

    assert isinstance(embedding, GeminiEmbedding)
    assert embedding.base_url == DEFAULT_GEMINI_EMBEDDING_BASE_URL


def test_auto_detects_gemini_from_base_url_host(monkeypatch: pytest.MonkeyPatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv(
        "HELLOAGENTS_MEMORY_EMBEDDING_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )

    embedding = memory_base.make_default_embedding(MemoryConfig.from_env())

    assert isinstance(embedding, GeminiEmbedding)
    # 模型名仍是 qwen 的类默认值时也要换成 Gemini 的模型，否则请求打错模型。
    assert embedding.model == DEFAULT_GEMINI_EMBEDDING_MODEL


def test_tunnel_gateway_still_wins_over_a_gemini_key(monkeypatch: pytest.MonkeyPatch):
    """既有优先级不许回归：EMBEDDING_BASE_URL 仍是 auto 下的最高优先级。"""

    from memory import EmbedServerEmbedding

    _isolate_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:10800")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_PROVIDER", "auto")

    embedding = memory_base.make_default_embedding(MemoryConfig.from_env())

    assert isinstance(embedding, EmbedServerEmbedding)
    assert embedding.dimension == 1024  # 网关维度默认值不受远端留空影响


def test_no_key_falls_back_to_hash(monkeypatch: pytest.MonkeyPatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_PROVIDER", "gemini")

    embedding = memory_base.make_default_embedding(MemoryConfig.from_env())

    assert isinstance(embedding, HashEmbedding)


def test_from_env_blank_dimension_means_auto(monkeypatch: pytest.MonkeyPatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_API_KEY", "k")

    # 留空 / auto / 0 都表示「不预设」，维度由首次响应决定。
    assert MemoryConfig.from_env().embedding_dimension is None
    assert MemoryConfig().embedding_dimension is None
    for raw in ("auto", "none", "0", "AUTO"):
        monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_DIMENSION", raw)
        assert MemoryConfig.from_env().embedding_dimension is None
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_DIMENSION", "768")
    assert MemoryConfig.from_env().embedding_dimension == 768


def test_from_env_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_PROVIDER", "cohere")
    with pytest.raises(ValueError, match="embedding_provider"):
        MemoryConfig.from_env()


def test_reindex_script_refuses_the_offline_space(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """离线 HashEmbedding 不是远端向量空间的替代品，重索引必须拒绝而不是污染。"""

    import importlib.util
    from pathlib import Path

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.chdir(tmp_path)

    script = Path(__file__).resolve().parent.parent / "scripts" / "reindex_embeddings.py"
    spec = importlib.util.spec_from_file_location("reindex_embeddings_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main() == 1
