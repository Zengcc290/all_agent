"""Unified embedding service backed by an OpenAI-compatible HTTP API.

Only one concrete implementation ships: :class:`APIEmbedding`, a vendor-neutral
client for any provider that exposes the OpenAI ``/embeddings`` shape (DashScope
for qwen3-embedding-0.6b, OpenAI, SiliconFlow, Zhipu, local vLLM, ...).  The
abstract :class:`BaseEmbedding` interface stays so applications can inject
their own model or callable without touching the rest of the system.

The default model is ``qwen3-embedding-0.6b`` (1024 dimensions) served by
DashScope's OpenAI-compatible endpoint.  The API key is read from
``DASHSCOPE_API_KEY`` (or ``MemoryConfig.embedding_api_key`` / the
``HELLOAGENTS_MEMORY_EMBEDDING_API_KEY`` environment variable).
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

#: Default vendor endpoint and model used when nothing else is configured.
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"

#: DashScope style batch ceiling; other providers tolerate different sizes and
#: can lower/raise ``batch_size`` at construction time.
DEFAULT_BATCH_SIZE = 10


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
        batch_size: int = DEFAULT_BATCH_SIZE,
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

        resolved_key = api_key if api_key is not None else os.getenv(api_key_env)
        if not resolved_key or not str(resolved_key).strip():
            raise RuntimeError(
                f"APIEmbedding requires an API key: pass api_key=... or set the "
                f"{api_key_env} environment variable"
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


# A friendly alias used by integrations that treat this layer as a service.
EmbeddingService = BaseEmbedding

__all__ = [
    "APIEmbedding",
    "BaseEmbedding",
    "DEFAULT_EMBEDDING_BASE_URL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_BATCH_SIZE",
    "EmbeddingService",
]