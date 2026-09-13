"""Shared pytest fixtures and test-only embedding for the memory suite.

The production embedding is :class:`memory.APIEmbedding` (network + API key);
without a key the library now falls back to the deterministic offline
:class:`memory.HashEmbedding`. Tests inject a 128-dimension variant so vectors
stay small and the CRUD/search assertions keep their original meaning without
any HTTP call.
"""

from __future__ import annotations

import pytest

from memory import HashEmbedding as _LibraryHashEmbedding
from memory import MemoryConfig, MemoryManager


class HashEmbedding(_LibraryHashEmbedding):
    """Deterministic offline embedding used only by tests (128 dimensions)."""

    def __init__(self, dimension: int = 128) -> None:
        super().__init__(dimension=dimension)


@pytest.fixture()
def memory_config() -> MemoryConfig:
    return MemoryConfig(sqlite_path=":memory:")


@pytest.fixture()
def manager(memory_config: MemoryConfig) -> MemoryManager:
    return MemoryManager(memory_config, embedding=HashEmbedding())