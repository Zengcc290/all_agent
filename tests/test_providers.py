from pathlib import Path

import pytest

from agents import ProviderRegistry


def write_config(path: Path, active: str = "local") -> None:
    path.write_text(
        f"""[defaults]
active_profile = "{active}"

[profiles.local]
base_url = "http://localhost:8000/v1"
api_key_env = "TEST_LOCAL_KEY"
default_model = "local-model"
models = ["local-model", "local-fast"]
tool_mode = "text_react"

[profiles.remote]
base_url = "https://example.test/v1"
api_key_env = "TEST_REMOTE_KEY"
default_model = "remote-model"
models = ["remote-model"]
""",
        encoding="utf-8",
    )


def test_registry_loads_profiles_without_resolving_secrets(tmp_path, monkeypatch):
    config = tmp_path / "providers.toml"
    write_config(config)
    registry = ProviderRegistry(config)

    assert registry.active_profile == "local"
    assert registry.get("local").tool_mode == "text_react"
    assert registry.get("remote").public_info()["api_key_env"] == "TEST_REMOTE_KEY"
    with pytest.raises(ValueError, match="TEST_LOCAL_KEY"):
        registry.resolve_api_key("local")

    monkeypatch.setenv("TEST_LOCAL_KEY", "local-secret")
    assert registry.resolve_api_key("local") == "local-secret"


def test_registry_rejects_unknown_active_profile(tmp_path):
    config = tmp_path / "providers.toml"
    write_config(config, active="missing")
    with pytest.raises(ValueError, match="does not name a configured profile"):
        ProviderRegistry(config)


def test_registry_uses_legacy_single_profile_environment(tmp_path, monkeypatch):
    config = tmp_path / "providers.toml"
    write_config(config, active="local")
    # The compatibility bridge is intentionally limited to the conventional
    # openai profile, so this custom profile must continue using its own key.
    monkeypatch.setenv("LLM_API_KEY", "legacy-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://legacy.test/v1")
    monkeypatch.setenv("LLM_MODEL", "legacy-model")
    registry = ProviderRegistry(config)
    assert registry.get("local").base_url == "http://localhost:8000/v1"
