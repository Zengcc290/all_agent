"""Shared pytest fixtures and test-only embedding for the memory suite.

The production embedding is :class:`memory.APIEmbedding` (network + API key),
so tests inject this deterministic, offline ``HashEmbedding`` instead.  It is
deliberately dependency-free and reproduces the old TF-IDF behaviour (space
delimited tokens hashed into a fixed-size vector) so the CRUD/search tests keep
their original meaning without requiring an HTTP call.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import pytest

from memory import BaseEmbedding, MemoryConfig, MemoryManager


class HashEmbedding(BaseEmbedding):
    """Deterministic offline embedding used only by tests."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        vector = [0.0] * self.dimension
        counts = Counter(self.tokenize(text))
        for token, count in counts.items():
            vector[self._index(token)] += 1.0 + math.log(float(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


@pytest.fixture()
def memory_config() -> MemoryConfig:
    return MemoryConfig(sqlite_path=":memory:")


@pytest.fixture()
def manager(memory_config: MemoryConfig) -> MemoryManager:
    return MemoryManager(memory_config, embedding=HashEmbedding())