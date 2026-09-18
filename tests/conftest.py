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
def _strip_ambient_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """让测试对本机代理环境变量免疫。

    开着 Clash 等代理软件时，``NO_PROXY``/``ALL_PROXY`` 常带畸形值（如
    ``[::1]`` 的括号写法），httpx 构造客户端解析 URLPattern 直接抛
    ``InvalidURL``——与被测代码无关，却在任何真实客户端构造处炸。测试
    会话统一剥离代理变量；生产端代理只认 config/services.toml [proxy]。
    """

    for name in (
        "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "all_proxy", "http_proxy", "https_proxy", "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _web_autoseed_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Web 启动播种默认关闭（原 WEB_AUTOSEED 环境变量已收拢为 constants 常量）。

    测试用注入的内存库显式控制数据；需要播种的用例自行调用 /api/seed 或
    把 ``web.app.WEB_AUTOSEED`` monkeypatch 回 True。
    """

    import web.app as web_app

    monkeypatch.setattr(web_app, "WEB_AUTOSEED", False)


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


@pytest.fixture(autouse=True)
def _neutralize_local_provider_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep chat/extractor construction hermetic across the test session.

    A local ``config/provider.toml`` with a real key would make ``chat_ready()``
    return True and ``build_knowledge_extractor()`` call the live LLM during
    ingest.  Pointing the default path at the published example (placeholder
    key) restores the fresh-checkout behaviour: chat is disabled, extraction
    is a no-op, Agent construction still succeeds.
    """

    from agents.providers import ProviderRegistry

    example = Path(__file__).resolve().parent.parent / "config" / "provider.example.toml"
    monkeypatch.setattr(
        ProviderRegistry,
        "default_config_path",
        staticmethod(lambda: example),
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