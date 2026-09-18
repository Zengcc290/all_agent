"""Unified embedding service: cloud OpenAI-compatible API + offline fallback.

运行时嵌入只保留两种实现：

- :class:`APIEmbedding` - 任何 OpenAI 兼容 ``/embeddings`` 端点（SiliconFlow、
  DashScope、OpenAI、Zhipu、本地 vLLM）。端点/密钥/模型统一来自
  ``config/services.toml`` 的 ``[embedding]`` 段。
- :class:`HashEmbedding` - 确定性离线兜底：未配置云端嵌入时保持记忆层、
  Web 与 RAG 可用（向量空间与云端互不兼容，仅测试/离线场景使用）。

历史实现（Gemini ``:embedContent``、本机转发网关 ``/embed``、SSH 隧道自动
拉起、``.env`` 装载）已按"配置只认 config/ 与 constants.py"的约定删除。
云端向量空间互不兼容：切换模型后必须重建投影（维度唯一开关是
``[embedding].model``，见 ``QdrantVectorStore`` 的维度守卫与迁移脚本）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from typing import Any

from constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    MEMORY_EMBEDDING_DIMENSION,
)

#: 模型名里带 ``vl`` 的按视觉语言嵌入处理（``Qwen/Qwen3-VL-Embedding-*`` 等）。
#: 这类模型仍是 OpenAI 兼容 ``/embeddings`` 端点，只是 ``input`` 额外接受
#: ``{"text": ...}`` / ``{"image": ...}`` 内容对象与它们的混合列表。
VL_EMBEDDING_MODEL_MARKER = "vl"


class BaseEmbedding(ABC):
    """Tiny interface every embedding provider implements."""

    dimension: int = 0

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one text into a finite numeric vector."""
        raise NotImplementedError

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        return [self.embed(text) for text in values]

    def embed_item(self, text: str, *, payload: Any = None, modality: str | None = None) -> list[float]:
        """Embed one stored memory item.

        Text-only by default.  ``BaseMemory.add`` routes every write through this
        hook, so a multimodal backend overrides it to fold ``payload``/
        ``modality`` into the same vector - that is what makes a stored image
        retrievable by its own content.  Providers that cannot embed
        ``payload`` ignore it and keep embedding ``text``.
        """
        del payload, modality
        return self.embed(text)


class HashEmbedding(BaseEmbedding):
    """Deterministic offline embedding (bag-of-words hashed into fixed dims).

    Not compatible with :class:`APIEmbedding` vector spaces: it exists so the
    memory layer, agent tools and the web app stay usable without any cloud
    configuration, and as the injected test double for the whole suite.
    """

    def __init__(self, dimension: int = MEMORY_EMBEDDING_DIMENSION) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        vector = [0.0] * self.dimension
        for token, count in Counter(self.tokenize(text)).items():
            vector[self._index(token)] += 1.0 + math.log(float(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def to_dict(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "dimension": self.dimension}

    def __repr__(self) -> str:
        return f"HashEmbedding(dimension={self.dimension})"


class APIEmbedding(BaseEmbedding):
    """OpenAI-compatible ``/embeddings`` client for any cloud provider.

    ``base_url`` is the provider root without the trailing ``/embeddings``
    path (e.g. ``https://api.siliconflow.cn/v1``).  The request body and the
    response parser follow the OpenAI shape, and the parser additionally
    accepts DashScope's native ``output.embeddings`` layout so either gateway
    works.  The API key comes from ``config/services.toml [embedding].api_key``
    (resolved by ``memory.base.make_default_embedding``), never from the
    environment.

    Visual-language embedding models (SiliconFlow's
    ``Qwen/Qwen3-VL-Embedding-*``, whose model id contains ``vl``) keep the same
    endpoint, auth header and ``data[].embedding`` response, but their ``input``
    additionally accepts content objects - ``{"text": "..."}``,
    ``{"image": "<url|base64>"}`` - and mixed lists of them, where one request
    fuses the list into a single vector.  :attr:`multimodal` is derived from the
    model name and gates the automatic routing in :meth:`embed_item`;
    ``embed_image``/``embed_multimodal`` always work regardless of the name.

    ``client`` may be injected for tests: a callable ``client(payload, *)``
    returning a parsed JSON object, or an object exposing
    ``embeddings.create(input=..., model=...)``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = "",
        dimension: int | None = None,
        timeout: float = 30.0,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        client: Any = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string (config/services.toml [embedding].base_url)")
        if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
            raise ValueError("dimension must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not api_key or not str(api_key).strip():
            raise RuntimeError(
                "APIEmbedding requires an API key: pass api_key= (config/services.toml [embedding].api_key)"
            )
        self.api_key = str(api_key).strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.batch_size = batch_size
        self.client = client
        self.dimension = dimension or 0
        #: 是否按视觉语言嵌入发送内容对象；可显式改写以支持名字里没有 vl 的多模态模型。
        self.multimodal = VL_EMBEDDING_MODEL_MARKER in self.model.casefold()

    # ------------------------------------------------------------------ embed

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        if not values:
            return []
        result: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            result.extend(_embed_batch_once(self, values[start : start + self.batch_size]))
        return result

    # -------------------------------------------------------- VL input objects

    @staticmethod
    def text_input(text: str) -> dict[str, str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return {"text": text}

    @staticmethod
    def image_input(data: Any, *, mime_type: str | None = None) -> dict[str, str]:
        """Build the ``{"image": ...}`` content object.

        A URL or an already-encoded base64 string (raw or data URI) is passed
        through untouched, so callers control the encoding the provider wants.
        ``bytes`` become a ``data:<mime>;base64,...`` URI, since a bare base64
        blob is not self-describing.
        """
        if isinstance(data, (bytes, bytearray, memoryview)):
            mime = mime_type or _image_mime_type(data)
            encoded = base64.b64encode(bytes(data)).decode("ascii")
            return {"image": f"data:{mime};base64,{encoded}"}
        if isinstance(data, str) and data.strip():
            return {"image": data.strip()}
        raise TypeError("image data must be bytes, or a non-empty URL/base64 string")

    def inputs_for(self, text: str = "", *, image: Any = None, mime_type: str | None = None) -> list[Any]:
        """Assemble the VL ``input`` list: text object, image object, or both."""
        items: list[Any] = []
        if isinstance(text, str) and text.strip():
            items.append(self.text_input(text))
        if image is not None:
            items.append(self.image_input(image, mime_type=mime_type))
        if not items:
            raise ValueError("at least one of text/image must be provided")
        return items

    def embed_inputs(self, items: list[Any]) -> list[float]:
        """Send one VL ``input`` list, fused by the model into a single vector.

        A mixed list (``[{"text": ...}, {"image": ...}]``) is one request that
        yields exactly one vector, so it must not go through the per-item batch
        count check.
        """
        if not isinstance(items, list) or not items:
            raise ValueError("items must be a non-empty list")
        vectors = _extract_embedding_vectors(_request(self, items))
        if len(vectors) != 1:
            raise RuntimeError(
                f"embedding response count {len(vectors)} did not match the single fused vector "
                "expected for a VL input list"
            )
        _learn_dimension(self, vectors)
        return vectors[0]

    def embed_image(self, data: Any, *, mime_type: str | None = None) -> list[float]:
        """Embed one image on its own (no accompanying text)."""
        return self.embed_inputs([self.image_input(data, mime_type=mime_type)])

    def embed_multimodal(self, text: str, data: Any, *, mime_type: str | None = None) -> list[float]:
        """Embed text and image together into one fused vector."""
        return self.embed_inputs(self.inputs_for(text, image=data, mime_type=mime_type))

    def embed_item(self, text: str, *, payload: Any = None, modality: str | None = None) -> list[float]:
        """Route one stored item; text-only models ignore ``payload`` as before."""
        if payload is None or not self.multimodal:
            return self.embed(text)
        if isinstance(text, str) and text.strip():
            return self.embed_multimodal(text, payload)
        return self.embed_image(payload)

    # ------------------------------------------------------------------ config

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "model": self.model,
            "base_url": self.base_url,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            # 便于 /api/health 直接看出当前是不是按 VL 内容对象在发请求。
            "multimodal": self.multimodal,
        }

    def __repr__(self) -> str:
        return f"APIEmbedding(model={self.model!r}, dimension={self.dimension}, base_url={self.base_url!r})"


def _embed_batch_once(embedding: APIEmbedding, values: list[Any]) -> list[list[float]]:
    """Send one batch and normalize the response into a list of vectors.

    ``values`` are plain strings for text models, or VL content objects
    (``{"text": ...}`` / ``{"image": ...}``) for visual-language models - both
    travel in the same OpenAI-compatible ``input`` field.  One input item yields
    one vector here; a *fused* VL list is handled by :meth:`APIEmbedding.embed_inputs`.
    """
    response = _request(embedding, values)
    vectors = _extract_embedding_vectors(response)
    if len(vectors) != len(values):
        raise RuntimeError(
            f"embedding response count {len(vectors)} did not match input count {len(values)}"
        )
    _learn_dimension(embedding, vectors)
    return vectors


def _learn_dimension(embedding: APIEmbedding, vectors: list[list[float]]) -> None:
    """Validate vector shapes and record the dimension learned from a response."""
    dimension = len(vectors[0]) if vectors else 0
    if dimension == 0:
        raise RuntimeError("embedding response contained empty vectors")
    if any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("embedding response contained inconsistent dimensions")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise RuntimeError("embedding response contained non-finite values")
    if embedding.dimension and embedding.dimension != dimension:
        raise RuntimeError(
            f"embedding dimension {dimension} does not match expected dimension {embedding.dimension}"
        )
    embedding.dimension = dimension


def _request(embedding: APIEmbedding, values: list[Any]) -> Any:
    """Perform the HTTP call (or the injected test client) and parse JSON."""
    payload = {"model": embedding.model, "input": values}
    client = embedding.client
    if client is not None:
        if callable(client):
            return client(payload, model=embedding.model)
        if callable(getattr(client, "embed", None)):
            return client.embed(values, model=embedding.model)
        if callable(getattr(getattr(client, "embeddings", None), "create", None)):
            return client.embeddings.create(input=values, model=embedding.model)
        raise TypeError("client must be callable or expose embed()/embeddings.create()")
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{embedding.base_url}/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {embedding.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=embedding.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"embedding API HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"embedding API request failed: {exc.reason}") from exc
    except ValueError as exc:
        raise RuntimeError(f"embedding API returned invalid JSON: {exc}") from exc


def _extract_embedding_vectors(response: Any) -> list[list[float]]:
    """Accept OpenAI ``data[].embedding`` and DashScope ``output.embeddings``."""
    if isinstance(response, dict):
        data = response.get("data")
        if data is None:
            output = response.get("output", {})
            data = output.get("embeddings") if isinstance(output, dict) else None
    else:
        data = getattr(response, "data", None)
        if data is None:
            output = getattr(response, "output", None)
            data = getattr(output, "embeddings", None) if output is not None else None
    if data is None:
        raise RuntimeError("embedding response contained no data/embeddings list")

    indexed: list[tuple[int, list[float]]] = []
    for index, item in enumerate(data):
        if isinstance(item, dict):
            values = item.get("embedding")
            explicit = item.get("index", item.get("text_index"))
        else:
            values = getattr(item, "embedding", item)
            explicit = getattr(item, "index", None)
            if explicit is None:
                explicit = getattr(item, "text_index", None)
        if values is None:
            raise RuntimeError("embedding response item contained no embedding vector")
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("embedding response item contained an invalid vector") from exc
        if not vector:
            raise RuntimeError("embedding response item contained an empty vector")
        # Reorder explicitly when providers label each vector with a position.
        if explicit is not None:
            indexed.append((int(explicit), vector))
        else:
            indexed.append((index, vector))

    indexed.sort(key=lambda pair: pair[0])
    positions = [position for position, _ in indexed]
    if positions != list(range(len(indexed))):
        raise RuntimeError("embedding response indices were incomplete")
    return [vector for _, vector in indexed]


#: 图片魔数 → MIME。声明错 MIME 会被服务端拒收，所以按魔数猜而不是一律写 PNG。
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _image_mime_type(data: Any, fallback: str = "image/png") -> str:
    """按魔数猜图片 MIME，猜不出就用 ``fallback``。"""
    if isinstance(data, (bytes, bytearray, memoryview)):
        head = bytes(data)[:12]
        for magic, mime in _IMAGE_MAGIC:
            if head.startswith(magic):
                return mime
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "image/webp"
    return fallback


__all__ = [
    "VL_EMBEDDING_MODEL_MARKER",
    "APIEmbedding",
    "BaseEmbedding",
    "HashEmbedding",
]
