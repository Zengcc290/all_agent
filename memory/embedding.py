"""Unified embedding service backed by remote HTTP APIs.

Three remote implementations ship, each speaking a different wire protocol, plus
the offline fallback:

- :class:`APIEmbedding` - any provider exposing the OpenAI ``/embeddings`` shape
  (DashScope for qwen3-embedding-0.6b, OpenAI, SiliconFlow, Zhipu, local vLLM).
- :class:`GeminiEmbedding` - Google's ``:embedContent`` shape, which additionally
  accepts images and mixed image+text in one request.
- :class:`EmbedServerEmbedding` - the local port-forwarded ``qwen-embed`` gateway
  (custom ``/embed`` protocol).
- :class:`HashEmbedding` - deterministic offline fallback used when no provider is
  configured, so the memory layer, web app and RAG pipeline keep working air-gapped.

The default model is ``qwen3-embedding-0.6b`` (1024 dimensions) served by
DashScope's OpenAI-compatible endpoint.  The API key is read from
``DASHSCOPE_API_KEY`` (or ``MemoryConfig.embedding_api_key`` / the
``HELLOAGENTS_MEMORY_EMBEDDING_API_KEY`` environment variable); Gemini falls back
to ``GEMINI_API_KEY``.  Without a key the hash fallback keeps local search working
offline; every remote provider produces a different vector space, so switching
providers requires re-indexing stored items.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from typing import Any

from constants import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMINI_EMBEDDING_BASE_URL,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    GEMINI_API_KEY_ENV,
    LOCALHOST,
    MEMORY_EMBEDDING_DIMENSION,
)

#: Gemini ``embedContent`` 用这个头传 key（不是 ``Authorization: Bearer``）。
GEMINI_EMBEDDING_API_KEY_HEADER = "x-goog-api-key"


def load_dotenv_once() -> None:
    """Load the repository's ``.env`` when python-dotenv is available.

    Mirrors ``tool/search.py`` so a key written to ``.env`` is picked up by the
    memory layer without requiring the caller to export it first.  Existing
    environment variables always win (``override=False``).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Environment variables remain usable without the optional loader.
        return
    load_dotenv(override=False)


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
        hook, so a multimodal backend (:class:`GeminiEmbedding`) overrides it to
        fold ``payload``/``modality`` into the same vector - that is what makes a
        stored image retrievable by its own content.  Providers that cannot embed
        ``payload`` ignore it and keep embedding ``text``.
        """
        return self.embed(text)


class HashEmbedding(BaseEmbedding):
    """Deterministic offline embedding used when no API key is configured.

    Tokens are hashed into fixed-size buckets, summed and normalized. Retrieval
    quality is limited (it is bag-of-words, not semantic), but it is completely
    local, reproducible and dependency-free, so the memory layer, the web app
    and the RAG pipeline all keep working without network access. Vectors from
    this class are NOT compatible with :class:`APIEmbedding` vectors.
    """

    def __init__(self, dimension: int = MEMORY_EMBEDDING_DIMENSION) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension

    @staticmethod
    def tokenize(text: str) -> list[str]:
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
    """OpenAI-compatible ``/embeddings`` client for any provider.

    ``base_url`` is the provider root without the trailing ``/embeddings``
    path (e.g. ``https://dashscope.aliyuncs.com/compatible-mode/v1``).  The
    request body and the response parser follow the OpenAI shape, and the
    parser additionally accepts DashScope's native ``output.embeddings``
    layout so either gateway works.

    ``client`` may be injected for tests: a callable ``client(payload, *)``
    returning a parsed JSON object, or an object exposing
    ``embeddings.create(input=..., model=...)``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        dimension: int | None = None,
        timeout: float = 30.0,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        client: Any = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
            raise ValueError("dimension must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError("api_key_env must be a non-empty string")

        if api_key is None:
            load_dotenv_once()
        resolved_key = api_key if api_key is not None else os.getenv(api_key_env)
        if not resolved_key or not str(resolved_key).strip():
            raise RuntimeError(
                f"APIEmbedding requires an API key: pass api_key=... or set the "
                f"{api_key_env} environment variable (a .env file in the project "
                f"root is loaded automatically)"
            )
        self.api_key = str(resolved_key).strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.batch_size = batch_size
        self.client = client
        self.api_key_env = api_key_env
        self.dimension = dimension or 0

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

    # ------------------------------------------------------------------ config

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "model": self.model,
            "base_url": self.base_url,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
        }

    def __repr__(self) -> str:
        return f"APIEmbedding(model={self.model!r}, dimension={self.dimension}, base_url={self.base_url!r})"


#: Shown when the gateway port refuses connections.  Deliberately address-free:
#: the tunnel command belongs to deployment docs, not to the source tree.
EMBED_GATEWAY_HINT = (
    "embedding gateway unreachable - start the SSH port-forward that backs "
    "EMBEDDING_BASE_URL, then retry"
)


def gateway_reachable(base_url: str, *, timeout: float = 1.0) -> bool:
    """Return True when the ``/embed`` gateway port accepts a TCP connection now.

    Used to *report* degraded state (``/api/health``) and, from Phase 3 on, to
    pick a retrieval path.  It must never be used to swap embedding providers:
    the gateway, the public API and ``HashEmbedding`` produce mutually
    incompatible vector spaces, so a configured-but-down gateway stays
    configured and callers degrade to keyword search instead - substituting the
    hash space would mix incompatible vectors into an existing index.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    parsed = urllib.parse.urlsplit(base_url if "//" in base_url else f"//{base_url}")
    host = parsed.hostname or LOCALHOST
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class EmbedServerEmbedding(BaseEmbedding):
    """Client for a custom embed gateway exposed at ``POST {base_url}/embed``.

    This is the client for the local ``qwen-embed`` service reachable through
    a forwarded port (e.g. ``ssh -L10800:127.0.0.1:18000 ...``).  The request
    is ``{"texts": [...]}`` and the response is ``{"embeddings": [[...], ...]}``
    plus an optional ``dim`` field; the vector count, dimension and finiteness
    are validated exactly like :class:`APIEmbedding`.  No API key is required
    by the bundled gateway, but an optional ``Authorization: Bearer`` header is
    sent when one is supplied.  Use :class:`APIEmbedding` instead when the
    gateway speaks the standard OpenAI ``/embeddings`` shape.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        dimension: int | None = None,
        timeout: float = 60.0,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        client: Any = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
            raise ValueError("dimension must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.api_key = str(api_key).strip() if api_key is not None else ""
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.batch_size = batch_size
        self.client = client
        self.dimension = dimension or 0

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
            result.extend(_embed_server_batch_once(self, values[start : start + self.batch_size]))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "base_url": self.base_url,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
        }

    def __repr__(self) -> str:
        return f"EmbedServerEmbedding(dimension={self.dimension}, base_url={self.base_url!r})"


def _embed_server_batch_once(embedding: EmbedServerEmbedding, values: list[str]) -> list[list[float]]:
    """Send one batch through the custom ``/embed`` gateway and validate it."""
    response = _embed_server_request(embedding, values)
    vectors = _extract_embed_server_vectors(response)
    if len(vectors) != len(values):
        raise RuntimeError(
            f"embedding response count {len(vectors)} did not match input count {len(values)}"
        )
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
    return vectors


def _embed_server_request(embedding: EmbedServerEmbedding, values: list[str]) -> Any:
    """POST ``{"texts": [...]}`` to ``{base_url}/embed`` and parse the JSON."""
    payload = {"texts": values}
    client = embedding.client
    if client is not None:
        if callable(client):
            return client(payload)
        if callable(getattr(client, "embed", None)):
            return client.embed(values)
        if callable(getattr(getattr(client, "embeddings", None), "create", None)):
            return client.embeddings.create(input=values)
        raise TypeError("client must be callable or expose embed()/embeddings.create()")
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if embedding.api_key:
        headers["Authorization"] = f"Bearer {embedding.api_key}"
    request = urllib.request.Request(
        f"{embedding.base_url}/embed",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=embedding.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"embedding API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"embedding API request failed: {exc.reason}. {EMBED_GATEWAY_HINT}") from exc
    except ValueError as exc:
        raise RuntimeError(f"embedding API returned invalid JSON: {exc}") from exc


def _extract_embed_server_vectors(response: Any) -> list[list[float]]:
    """Extract and reorder the bare ``embeddings`` list from the gateway."""
    if isinstance(response, dict):
        data = response.get("embeddings")
    else:
        data = getattr(response, "embeddings", None)
    if data is None:
        raise RuntimeError("embedding response contained no embeddings list")
    vectors: list[list[float]] = []
    for item in data:
        try:
            vector = [float(value) for value in item]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("embedding response item contained an invalid vector") from exc
        if not vector:
            raise RuntimeError("embedding response item contained an empty vector")
        vectors.append(vector)
    return vectors


def _embed_batch_once(embedding: APIEmbedding, values: list[str]) -> list[list[float]]:
    """Send one batch and normalize the response into a list of vectors."""
    response = _request(embedding, values)
    vectors = _extract_embedding_vectors(response)
    if len(vectors) != len(values):
        raise RuntimeError(
            f"embedding response count {len(vectors)} did not match input count {len(values)}"
        )
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
    return vectors


def _request(embedding: APIEmbedding, values: list[str]) -> Any:
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


class GeminiEmbedding(BaseEmbedding):
    """Google Gemini ``:embedContent`` client: text, image and mixed input.

    All three input shapes hit the same endpoint (``{base_url}/models/{model}:embedContent``)::

        text   parts=[{"text": "..."}]
        image  parts=[{"inline_data": {"mime_type": "image/png", "data": "<base64>"}}]
        mixed  both parts in one request -> one fused multimodal vector

    Differences from :class:`APIEmbedding`: the key travels in the
    ``x-goog-api-key`` header instead of ``Authorization: Bearer``, the body is
    ``{"content": {"parts": [...]}}`` rather than ``{"model", "input"}``, and the
    vector comes back as ``embedding.values``.  When ``dimension`` is set it is
    also sent as ``config.outputDimensionality`` so the server truncates (default
    3072; 1536/768/128 are supported).  Vectors from this class are NOT
    compatible with the other vector spaces.

    ``client`` may be injected for tests: a callable ``client(payload, model=...)``
    returning the parsed JSON object.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
        base_url: str = DEFAULT_GEMINI_EMBEDDING_BASE_URL,
        dimension: int | None = None,
        timeout: float = 30.0,
        client: Any = None,
        api_key_env: str = GEMINI_API_KEY_ENV,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if dimension is not None and (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1):
            raise ValueError("dimension must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError("api_key_env must be a non-empty string")

        if api_key is None:
            load_dotenv_once()
        resolved_key = api_key if api_key is not None else os.getenv(api_key_env)
        if not resolved_key or not str(resolved_key).strip():
            raise RuntimeError(
                f"GeminiEmbedding requires an API key: pass api_key=... or set the "
                f"{api_key_env} environment variable (a .env file in the project "
                f"root is loaded automatically)"
            )
        self.api_key = str(resolved_key).strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.client = client
        self.api_key_env = api_key_env
        #: 0 = 尚未知：首次响应里学到；非 0 时同时作为 ``outputDimensionality`` 下发。
        self.dimension = dimension or 0

    # ------------------------------------------------------------------- parts

    @staticmethod
    def text_part(text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return {"text": text}

    @staticmethod
    def image_part(data: Any, *, mime_type: str = "image/png") -> dict[str, Any]:
        if not isinstance(mime_type, str) or not mime_type.strip():
            raise ValueError("mime_type must be a non-empty string")
        if isinstance(data, (bytes, bytearray, memoryview)):
            encoded = base64.b64encode(bytes(data)).decode("ascii")
        elif isinstance(data, str) and data.strip():
            encoded = data.strip()  # 调用方已经给的是 base64
        else:
            raise TypeError("image data must be bytes or a non-empty base64 string")
        return {"inline_data": {"mime_type": mime_type.strip(), "data": encoded}}

    def parts_for(self, text: str = "", *, image: Any = None, mime_type: str = "image/png") -> list[dict[str, Any]]:
        """Assemble ``content.parts`` for one of the three input shapes."""
        parts: list[dict[str, Any]] = []
        if isinstance(text, str) and text.strip():
            parts.append(self.text_part(text))
        if image is not None:
            parts.append(self.image_part(image, mime_type=mime_type))
        if not parts:
            raise ValueError("at least one of text/image must be provided")
        return parts

    # ------------------------------------------------------------------- embed

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.embed_parts([self.text_part(text)])

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        # embedContent takes exactly one content per call, so batches are issued
        # one request per text.
        # ponytail: per-item requests; switch to :batchEmbedContents if the
        # free tier's 100 requests/minute becomes the bottleneck.
        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        return [self.embed(text) for text in values]

    def embed_image(self, data: Any, *, mime_type: str = "image/png") -> list[float]:
        """Embed one image on its own (no accompanying text)."""
        return self.embed_parts([self.image_part(data, mime_type=mime_type)])

    def embed_multimodal(self, text: str, data: Any, *, mime_type: str = "image/png") -> list[float]:
        """Embed text and image together into one fused vector."""
        return self.embed_parts(self.parts_for(text, image=data, mime_type=mime_type))

    def embed_item(self, text: str, *, payload: Any = None, modality: str | None = None) -> list[float]:
        """Route one stored item: image -> image, image+text -> mixed, else text."""
        if payload is None:
            return self.embed(text)
        mime_type = _image_mime_type(payload)
        if isinstance(text, str) and text.strip():
            return self.embed_multimodal(text, payload, mime_type=mime_type)
        return self.embed_image(payload, mime_type=mime_type)

    def embed_parts(self, parts: list[dict[str, Any]]) -> list[float]:
        """Raw escape hatch: send a caller-built ``parts`` list."""
        if not isinstance(parts, list) or not parts:
            raise ValueError("parts must be a non-empty list")
        return _finish_gemini_vector(self, _gemini_request(self, parts))

    # ------------------------------------------------------------------ config

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "model": self.model,
            "base_url": self.base_url,
            "dimension": self.dimension,
        }

    def __repr__(self) -> str:
        return f"GeminiEmbedding(model={self.model!r}, base_url={self.base_url!r}, dimension={self.dimension})"


def _gemini_request(embedding: GeminiEmbedding, parts: list[dict[str, Any]]) -> Any:
    """POST one ``:embedContent`` call (or the injected test client) and parse JSON."""
    payload: dict[str, Any] = {"content": {"parts": parts}}
    if embedding.dimension:
        # 只有显式配了维度才降维；留空即用服务端默认（3072）。
        payload["config"] = {"outputDimensionality": embedding.dimension}
    client = embedding.client
    if client is not None:
        if not callable(client):
            raise TypeError("client must be callable")
        return client(payload, model=embedding.model)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{embedding.base_url}/models/{embedding.model}:embedContent",
        data=body,
        headers={
            GEMINI_EMBEDDING_API_KEY_HEADER: embedding.api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=embedding.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"embedding API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"embedding API request failed: {exc.reason}") from exc
    except ValueError as exc:
        raise RuntimeError(f"embedding API returned invalid JSON: {exc}") from exc


def _extract_gemini_vector(response: Any) -> list[float]:
    """Read ``embedding.values``; also accept a ``embeddings[0].values`` list."""
    if isinstance(response, dict):
        embedding = response.get("embedding")
        if embedding is None:
            embeddings = response.get("embeddings")
            embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
    else:
        embedding = getattr(response, "embedding", None)
        if embedding is None:
            embeddings = getattr(response, "embeddings", None)
            embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
    values = embedding.get("values") if isinstance(embedding, dict) else getattr(embedding, "values", None)
    try:
        # 归一成 list；None / 标量这类形状直接在这里失败，交给下面统一报错。
        return list(values)
    except TypeError as exc:
        raise RuntimeError("embedding response did not contain embedding.values") from exc


def _finish_gemini_vector(embedding: GeminiEmbedding, response: Any) -> list[float]:
    """Validate one vector, learn the dimension, and enforce a configured one."""
    raw = _extract_gemini_vector(response)
    if not raw:
        raise RuntimeError("embedding response contained empty vectors")
    vector: list[float] = []
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("embedding response contained non-finite values") from exc
        if not math.isfinite(number):
            raise RuntimeError("embedding response contained non-finite values")
        vector.append(number)
    if embedding.dimension and embedding.dimension != len(vector):
        raise RuntimeError(
            f"embedding dimension {len(vector)} does not match expected dimension {embedding.dimension}"
        )
    embedding.dimension = len(vector)
    return vector


__all__ = [
    "DEFAULT_EMBEDDING_BASE_URL",
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GEMINI_EMBEDDING_BASE_URL",
    "DEFAULT_GEMINI_EMBEDDING_MODEL",
    "APIEmbedding",
    "BaseEmbedding",
    "EmbedServerEmbedding",
    "GeminiEmbedding",
    "load_dotenv_once",
]
