"""Embedding service implementations.

All providers expose the same tiny interface, making them interchangeable in
the manager and in applications that want to supply their own model.
"""

from __future__ import annotations

import hashlib
import math
import re
import urllib.error
import urllib.request
import json
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Iterable


class BaseEmbedding(ABC):
    dimension: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        return [self.embed(text) for text in values]


class TFIDFEmbedding(BaseEmbedding):
    """A deterministic, dependency-free TF-IDF style embedding.

    Tokens are hashed into a fixed-size vector, so adding documents never
    changes vector dimensionality (important for persistent vector stores).
    ``fit`` may be called with a corpus to improve IDF weighting.
    """

    def __init__(self, dimension: int = 384) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self._idf: dict[int, float] = {}
        self._documents = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # Keep Unicode words (including Chinese runs) and latin/numeric terms.
        return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)

    def fit(self, texts: Iterable[str]) -> "TFIDFEmbedding":
        documents = list(texts)
        df: Counter[int] = Counter()
        for text in documents:
            seen = set()
            for token in self.tokenize(text):
                seen.add(self._index(token))
            df.update(seen)
        self._documents = len(documents)
        self._idf = {
            index: math.log((1 + self._documents) / (1 + count)) + 1.0
            for index, count in df.items()
        }
        return self

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def embed(self, text: str) -> list[float]:
        tokens = self.tokenize(text)
        vector = [0.0] * self.dimension
        if not tokens:
            return vector
        counts = Counter(tokens)
        for token, count in counts.items():
            index = self._index(token)
            # Sublinear TF reduces the impact of repeated boilerplate words.
            tf = 1.0 + math.log(float(count))
            vector[index] += tf * self._idf.get(index, 1.0)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class LocalTransformerEmbedding(BaseEmbedding):
    """Sentence-transformers adapter loaded lazily.

    A model instance or callable can be injected in tests and in applications;
    the optional ``sentence-transformers`` package is only imported when needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model: Any = None) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "LocalTransformerEmbedding requires sentence-transformers; "
                    "install it or inject a model instance"
                ) from exc
            model = SentenceTransformer(model_name)
        if not callable(getattr(model, "encode", None)) and not callable(model):
            raise TypeError("model must provide encode() or be callable")
        self.model_name = model_name
        self.model = model
        dimensions = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        self.dimension = int(dimensions) if dimensions else 0

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if callable(getattr(self.model, "encode", None)):
            try:
                value = self.model.encode(text, convert_to_numpy=False)
            except TypeError:
                value = self.model.encode(text)
        else:
            value = self.model(text)
        # Some model wrappers return a one-row matrix for a single input.
        try:
            first = value[0]
        except (IndexError, KeyError, TypeError):
            first = None
        if isinstance(first, (list, tuple)) or getattr(first, "ndim", 0) > 0:
            value = first
        result = [float(item) for item in value]
        if not result or any(not math.isfinite(item) for item in result):
            raise ValueError("embedding model returned an invalid vector")
        if not self.dimension:
            self.dimension = len(result)
        return result


class DashScopeEmbedding(BaseEmbedding):
    """DashScope compatible embedding API adapter using the standard library."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        *,
        client: Any = None,
        timeout: float = 30.0,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        self.api_key, self.model, self.base_url = api_key.strip(), model.strip(), base_url.strip()
        if not self.base_url:
            raise ValueError("base_url must be non-empty")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.client = client
        self.timeout = timeout
        self.dimension = 0

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not all(isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        if not values:
            return []
        if self.client is not None:
            if callable(self.client):
                response = self.client(values, model=self.model)
            elif callable(getattr(self.client, "embed", None)):
                response = self.client.embed(values, model=self.model)
            elif callable(getattr(getattr(self.client, "embeddings", None), "create", None)):
                response = self.client.embeddings.create(input=values, model=self.model)
            else:
                raise TypeError("client must be callable or expose embed()/embeddings.create()")
            result = _extract_embeddings(response)
        else:
            body = json.dumps({"model": self.model, "input": {"texts": values}}).encode("utf-8")
            request = urllib.request.Request(
                self.base_url,
                data=body,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = _extract_embeddings(json.loads(response.read().decode("utf-8")))
            except (urllib.error.URLError, ValueError) as exc:
                raise RuntimeError(f"DashScope embedding request failed: {exc}") from exc
        if result:
            self.dimension = len(result[0])
            if any(len(vector) != self.dimension for vector in result) or len(result) != len(values):
                raise RuntimeError("embedding response count or dimensions did not match input")
        return result


def _extract_embeddings(response: Any) -> list[list[float]]:
    if isinstance(response, dict):
        data = response.get("output", response)
    else:
        data = getattr(response, "output", None) or getattr(response, "data", response)
    if isinstance(data, dict):
        data = data.get("embeddings", data.get("data", []))
    if not isinstance(data, (list, tuple)):
        try:
            data = list(data)
        except TypeError:
            data = []
    result = []
    for item in data or []:
        values = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", item)
        if values is None:
            continue
        vector = [float(value) for value in values]
        if not vector or any(not math.isfinite(value) for value in vector):
            raise RuntimeError("embedding response contained an invalid vector")
        result.append(vector)
    if not result:
        raise RuntimeError("embedding response contained no vectors")
    return result


# Friendly alias used by integrations that call this layer an embedding service.
EmbeddingService = BaseEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingService",
    "DashScopeEmbedding",
    "LocalTransformerEmbedding",
    "TFIDFEmbedding",
]
