"""Unit tests for the vendor-neutral APIEmbedding (OpenAI-compatible /embeddings)."""

from __future__ import annotations

import base64

import pytest

from memory import DEFAULT_EMBEDDING_MODEL, APIEmbedding

#: 测试注入假 client（不出网）；base_url 只需是一个语法合法的端点。
TEST_BASE_URL = "https://api.example.com/v1"


@pytest.fixture()
def embedding() -> APIEmbedding:
    return APIEmbedding(api_key="test-key", base_url=TEST_BASE_URL)


def test_requires_api_key():
    # 密钥只来自 config/services.toml [embedding].api_key（经 make_default_embedding
    # 传入）；环境变量不再是来源，None/空白都必须直接报错。
    with pytest.raises(RuntimeError, match="API key"):
        APIEmbedding(api_key=None, base_url=TEST_BASE_URL)
    with pytest.raises(RuntimeError, match="API key"):
        APIEmbedding(api_key="  ", base_url=TEST_BASE_URL)


def test_ignores_api_key_environment(monkeypatch: pytest.MonkeyPatch):
    """配置只认 services.toml：设了 DASHSCOPE_API_KEY 也不能凭空构造。"""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    with pytest.raises(RuntimeError, match="API key"):
        APIEmbedding(base_url=TEST_BASE_URL)


def test_validation_rejects_bad_arguments():
    with pytest.raises(ValueError, match="model"):
        APIEmbedding(api_key="k", base_url=TEST_BASE_URL, model="  ")
    with pytest.raises(ValueError, match="base_url"):
        APIEmbedding(api_key="k", base_url="")
    with pytest.raises(ValueError, match="dimension"):
        APIEmbedding(api_key="k", base_url=TEST_BASE_URL, dimension=0)
    with pytest.raises(ValueError, match="timeout"):
        APIEmbedding(api_key="k", base_url=TEST_BASE_URL, timeout=-1)
    with pytest.raises(ValueError, match="batch_size"):
        APIEmbedding(api_key="k", base_url=TEST_BASE_URL, batch_size=0)


def test_default_model_and_learned_dimension():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL)
    assert instance.model == DEFAULT_EMBEDDING_MODEL == "qwen3-embedding-0.6b"
    assert instance.dimension == 0  # learned from the first response


def test_no_factory_endpoint_cloud_must_be_configured():
    """云端端点没有出厂值：base_url 只能来自 services.toml [embedding]。

    历史 DEFAULT_EMBEDDING_BASE_URL 指向本机隧道网关（10800），随网关实现一并
    删除；缺配置时 make_default_embedding 回落离线 HashEmbedding，而不是猜一个
    端点静默切换向量空间。
    """

    from memory import MemoryConfig

    assert MemoryConfig().embedding_base_url == ""
    with pytest.raises(ValueError, match="base_url"):
        APIEmbedding(api_key="k")


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

    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, client=fake_client)
    vectors = instance.embed_batch(["hello", "world"])

    assert calls == [{"model": "qwen3-embedding-0.6b", "input": ["hello", "world"]}]
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert instance.dimension == 3


def test_embed_single_text():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, client=lambda payload, **kw: {"data": [{"embedding": [1.0, 2.0]}]})
    assert instance.embed("solo") == [1.0, 2.0]


def test_dashscope_native_output_layout_is_accepted():
    instance = APIEmbedding(
        api_key="k",
        base_url=TEST_BASE_URL,
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

    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, client=fake_client, batch_size=2)
    vectors = instance.embed_batch(["a", "b", "c", "d", "e"])

    assert received == [["a", "b"], ["c", "d"], ["e"]]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_index_out_of_order_is_reordered():
    instance = APIEmbedding(
        api_key="k",
        base_url=TEST_BASE_URL,
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
        base_url=TEST_BASE_URL,
        client=lambda payload, **kw: {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},
    )
    with pytest.raises(RuntimeError, match="indices"):
        instance.embed_batch(["a", "b"])


def test_count_mismatch_raises():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, client=lambda payload, **kw: {"data": [{"embedding": [1.0]}]})
    with pytest.raises(RuntimeError, match="count"):
        instance.embed_batch(["a", "b"])


def test_dimension_mismatch_raises():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, dimension=4, client=lambda payload, **kw: {"data": [{"embedding": [1.0, 2.0, 3.0]}]})
    with pytest.raises(RuntimeError, match="dimension"):
        instance.embed("text")


def test_invalid_vector_values_raise():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, client=lambda payload, **kw: {"data": [{"embedding": [float("nan")]}]})
    with pytest.raises(RuntimeError, match="non-finite"):
        instance.embed("text")


def test_rejects_non_string_input():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, client=lambda payload, **kw: {"data": []})
    with pytest.raises(TypeError, match="string"):
        instance.embed(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string"):
        instance.embed_batch(["ok", 7])  # type: ignore[list-item]


def test_repr_and_to_dict_are_safe():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL)
    assert instance.model in repr(instance)
    data = instance.to_dict()
    assert data["type"] == "APIEmbedding"
    assert data["model"] == DEFAULT_EMBEDDING_MODEL
    assert "api_key" not in data


# ---------------------------------------------------------------------------
# SiliconFlow 视觉语言嵌入：同一个 /embeddings 端点，input 换成内容对象
# （https://api-docs.siliconflow.cn/docs/api/embeddings-post 的 EmbeddingsVLRequest）
# ---------------------------------------------------------------------------

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
VL_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"fake-jpeg-body"


def _vl_embedding(calls: list[dict], **kwargs) -> APIEmbedding:
    """假客户端按文档的两种形态返回：字符串数组 → 逐条向量；内容对象列表 → 一个融合向量。"""

    def fake_client(payload, **inner):
        calls.append(payload)
        items = payload["input"]
        count = len(items) if all(isinstance(item, str) for item in items) else 1
        return {
            "object": "list",
            "model": payload["model"],
            "data": [{"index": index, "embedding": [0.1 + index, 0.2]} for index in range(count)],
            "usage": {},
        }

    return APIEmbedding(api_key="k", model=VL_MODEL, base_url=SILICONFLOW_BASE_URL, client=fake_client, **kwargs)


def test_vl_model_is_detected_from_the_model_name():
    assert _vl_embedding([]).multimodal is True
    assert APIEmbedding(api_key="k", base_url=TEST_BASE_URL).multimodal is False
    assert APIEmbedding(api_key="k", base_url=TEST_BASE_URL, model="Qwen/Qwen3-Embedding-8B").multimodal is False
    assert _vl_embedding([]).to_dict()["multimodal"] is True


def test_text_only_model_still_ignores_a_payload():
    """关键回归：非 VL 模型带着 payload 写记忆时，行为必须与改动前完全一致。"""

    calls: list[dict] = []
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, client=lambda payload, **kw: calls.append(payload) or {"data": [{"embedding": [1.0]}]})

    instance.embed_item("会议室照片", payload=PNG_BYTES, modality="image")

    assert calls == [{"model": DEFAULT_EMBEDDING_MODEL, "input": ["会议室照片"]}]


def test_vl_text_request_keeps_the_classic_string_shape():
    """纯文本对 VL 模型仍走经典字符串 input（文档：支持单个字符串/字符串数组）。"""

    calls: list[dict] = []
    instance = _vl_embedding(calls)

    assert instance.embed("Hello, world!") == [0.1, 0.2]
    assert calls == [{"model": VL_MODEL, "input": ["Hello, world!"]}]


def test_vl_image_bytes_become_a_data_uri_content_object():
    calls: list[dict] = []
    instance = _vl_embedding(calls)

    instance.embed_image(PNG_BYTES)

    assert calls[0]["input"] == [{"image": "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")}]


def test_vl_image_mime_is_sniffed_for_the_data_uri():
    calls: list[dict] = []
    instance = _vl_embedding(calls)

    instance.embed_image(JPEG_BYTES)
    instance.embed_image(PNG_BYTES, mime_type="image/png")

    assert calls[0]["input"][0]["image"].startswith("data:image/jpeg;base64,")
    assert calls[1]["input"][0]["image"].startswith("data:image/png;base64,")


def test_vl_url_and_base64_strings_pass_through_untouched():
    """URL 与原样 base64 由调用方决定编码，代码不擅自加工。"""

    calls: list[dict] = []
    instance = _vl_embedding(calls)

    instance.embed_image("https://example.com/image.jpg")
    instance.embed_image("aGVsbG8=")
    instance.embed_image("data:image/webp;base64,aGVsbG8=")

    assert [call["input"][0]["image"] for call in calls] == [
        "https://example.com/image.jpg",
        "aGVsbG8=",
        "data:image/webp;base64,aGVsbG8=",
    ]


def test_vl_mixed_request_fuses_text_and_image_in_one_call():
    """混合列表一次请求 → 一个融合向量（不是两次请求拼起来）。"""

    calls: list[dict] = []
    instance = _vl_embedding(calls)

    vector = instance.embed_multimodal("商品描述：白色运动鞋", PNG_BYTES)

    assert vector == [0.1, 0.2]
    assert len(calls) == 1
    assert calls[0]["input"] == [
        {"text": "商品描述：白色运动鞋"},
        {"image": "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")},
    ]


def test_vl_embedding_of_multiple_items_still_batches_strings():
    calls: list[dict] = []
    instance = _vl_embedding(calls)

    vectors = instance.embed_batch(["a", "b"])

    assert calls == [{"model": VL_MODEL, "input": ["a", "b"]}]
    assert vectors == [[0.1, 0.2], [1.1, 0.2]]


def test_vl_fused_list_rejects_per_item_vectors():
    """文档对「混合列表」的返回条数没有明说：若服务端按条返回 N 个向量，
    必须报错而不是静默只取第一个（那会把图片整个丢掉，索引出来的向量是错的）。"""

    instance = APIEmbedding(
        api_key="k",
        base_url=TEST_BASE_URL,
        model=VL_MODEL,
        client=lambda payload, **kw: {"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]},
    )

    with pytest.raises(RuntimeError, match="single fused vector"):
        instance.embed_multimodal("商品描述", PNG_BYTES)


def test_vl_embed_item_routes_the_three_shapes():
    calls: list[dict] = []
    instance = _vl_embedding(calls)

    instance.embed_item("只有文字")
    instance.embed_item("", payload=PNG_BYTES, modality="image")
    instance.embed_item("会议室照片", payload=PNG_BYTES, modality="image")

    assert calls[0]["input"] == ["只有文字"]
    assert calls[1]["input"][0]["image"].startswith("data:image/png;base64,")
    assert calls[2]["input"][0] == {"text": "会议室照片"}
    assert "image" in calls[2]["input"][1]


def test_vl_rejects_empty_inputs():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL, model=VL_MODEL)
    with pytest.raises(ValueError, match="text/image"):
        instance.inputs_for("   ")
    with pytest.raises(ValueError, match="non-empty list"):
        instance.embed_inputs([])
    with pytest.raises(TypeError, match="bytes, or a non-empty URL"):
        instance.image_input(12345)
    with pytest.raises(TypeError, match="bytes, or a non-empty URL"):
        instance.image_input("  ")


def test_vl_memory_write_fuses_the_image_through_the_memory_layer():
    """端到端：VL 后端下，带图片 payload 的写入真的把图片送进了嵌入请求。"""

    from memory import MemoryConfig, PerceptualMemory

    calls: list[dict] = []
    memory = PerceptualMemory(
        embedding=_vl_embedding(calls),
        config=MemoryConfig(sqlite_path=":memory:"),
    )

    item = memory.add("会议室照片", payload=PNG_BYTES, modality="image")

    assert item.embedding == [0.1, 0.2]
    assert item.payload == PNG_BYTES
    assert calls[0]["input"][0] == {"text": "会议室照片"}
    assert calls[0]["input"][1]["image"].startswith("data:image/png;base64,")


def test_vl_real_http_request_uses_the_documented_wire_format(monkeypatch: pytest.MonkeyPatch):
    """真实出网路径：POST {base}/embeddings + Bearer 认证 + model/input 体。"""

    import json as _json
    from typing import Self

    from memory import embedding as embedding_module

    captured: list = []

    class _Response:
        def read(self) -> bytes:
            return _json.dumps({"object": "list", "data": [{"index": 0, "embedding": [3.0]}]}).encode("utf-8")

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        return _Response()

    monkeypatch.setattr(embedding_module.urllib.request, "urlopen", fake_urlopen)

    vector = APIEmbedding(api_key="secret", model=VL_MODEL, base_url=SILICONFLOW_BASE_URL).embed_image(PNG_BYTES)

    assert vector == [3.0]
    request = captured[0]
    assert request.full_url == "https://api.siliconflow.cn/v1/embeddings"
    assert request.get_method() == "POST"
    headers = {key.lower(): value for key, value in request.headers.items()}
    assert headers["authorization"] == "Bearer secret"
    body = _json.loads(request.data.decode("utf-8"))
    assert body["model"] == VL_MODEL
    assert body["input"][0]["image"].startswith("data:image/png;base64,")
    # 文档标注 dimensions 仅 Qwen/Qwen3 文本系列支持，不能替 VL 模型擅自下发。
    assert "dimensions" not in body


def test_services_toml_configuration_selects_the_openai_path(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """[embedding] 配齐端点+密钥+模型 -> 云端 APIEmbedding（含 VL 自动识别）。"""

    from core import services_config
    from memory import MemoryConfig
    from memory import base as memory_base

    monkeypatch.setattr(services_config, "default_config_path", lambda: tmp_path / "services.toml")
    (tmp_path / "services.toml").write_text(
        "[embedding]\n"
        f'base_url = "{SILICONFLOW_BASE_URL}"\n'
        f'model = "{VL_MODEL}"\n'
        'api_key = "sf-key"\n',
        encoding="utf-8",
    )

    embedding = memory_base.make_default_embedding(MemoryConfig.from_config())

    assert isinstance(embedding, APIEmbedding)
    assert embedding.base_url == SILICONFLOW_BASE_URL
    assert embedding.model == VL_MODEL
    assert embedding.api_key == "sf-key"
    assert embedding.multimodal is True
    assert embedding.dimension == 0      # 留空 = 首次响应自动识别


def test_rejects_non_http_base_url():
    for bad_url in ("file:///etc/passwd", "ftp://example.com/v1", "not-a-url", "//example.com/v1"):
        with pytest.raises(ValueError, match="HTTP\\(S\\)"):
            APIEmbedding(api_key="k", base_url=bad_url)


def test_request_time_rejects_non_http_base_url():
    instance = APIEmbedding(api_key="k", base_url=TEST_BASE_URL)
    instance.base_url = "file:///etc/passwd"
    with pytest.raises(ValueError, match="HTTP\\(S\\)"):
        instance.embed("x")
