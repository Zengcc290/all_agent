"""Shared pytest fixtures and test-only embedding for the memory suite.

The production embedding is :class:`memory.APIEmbedding` (network + API key);
without a key the library now falls back to the deterministic offline
:class:`memory.HashEmbedding`. Tests inject a 128-dimension variant so vectors
stay small and the CRUD/search assertions keep their original meaning without
any HTTP call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import core.services_config as _services_config
from memory import HashEmbedding as _LibraryHashEmbedding
from memory import MemoryConfig, MemoryManager


@pytest.fixture(autouse=True)
def _neutralize_local_services_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``MemoryConfig.from_env()`` hermetic across the test session.

    ``config/services.toml`` is the developer's local runtime file; unit tests
    must not depend on its contents (a fresh checkout has none, a local machine
    may carry real cloud credentials).  Pointing ``default_config_path`` at a
    non-existent path makes ``load_services_config()`` return an empty config.
    Tests that exercise the services file re-point the path explicitly
    (``test_services_config.py``).
    """

    monkeypatch.setattr(
        _services_config,
        "default_config_path",
        lambda: Path("__services_toml_absent_for_tests__"),
    )


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