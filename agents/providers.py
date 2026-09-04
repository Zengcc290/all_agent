"""Named provider-profile configuration for OpenAI-compatible LLMs.

Each profile is self-contained: its TOML entry contains the API URL and key,
so switching profiles does not require changing process environment variables.
The real ``provider.toml`` is ignored by Git; ``provider.example.toml`` is the
safe template to publish.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOOL_MODES = frozenset({"native_strict", "native_loose", "text_react", "none"})


@dataclass(frozen=True)
class ProviderProfile:
    """One named OpenAI-compatible provider configuration."""

    name: str
    base_url: str
    api_key: str
    default_model: str
    models: tuple[str, ...]
    api_key_env: str | None = None
    adapter: str = "openai_compatible"
    tool_mode: str = "native_strict"

    @property
    def api_url(self) -> str:
        """Public spelling for the normalized URL used by the client."""

        return self.base_url

    def public_info(self) -> dict[str, Any]:
        """Return inspectable profile metadata without resolving its secret."""

        return {
            "name": self.name,
            "adapter": self.adapter,
            "base_url": self.base_url,
            "api_url": self.base_url,
            "api_key": "***" if self.api_key else "",
            "api_key_env": self.api_key_env,
            "default_model": self.default_model,
            "models": list(self.models),
            "tool_mode": self.tool_mode,
        }


class ProviderRegistry:
    """Load, validate, and resolve named provider profiles from TOML."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else self.default_config_path()
        )
        self._profiles: Mapping[str, ProviderProfile] = MappingProxyType({})
        self.active_profile = ""
        self.reload()

    @staticmethod
    def default_config_path() -> Path:
        config_dir = Path(__file__).resolve().parent.parent / "config"
        # Prefer the private runtime file. The example fallback keeps a fresh
        # checkout usable for injected/fake LLMs until the user copies it.
        for filename in ("provider.toml", "providers.toml", "provider.example.toml"):
            candidate = config_dir / filename
            if candidate.is_file():
                return candidate
        return config_dir / "provider.toml"

    @property
    def profiles(self) -> Mapping[str, ProviderProfile]:
        return self._profiles

    def reload(self) -> None:
        """Reload the TOML configuration without exposing keys."""

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"provider profile configuration was not found: {self.config_path}"
            )
        try:
            with self.config_path.open("rb") as handle:
                document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"invalid provider profile TOML: {self.config_path}"
            ) from exc
        profiles, active_profile = _parse_document(document, self.config_path)
        self._profiles = MappingProxyType(profiles)
        self.active_profile = active_profile

    def get(self, name: str) -> ProviderProfile:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("profile_name must be a non-empty string")
        try:
            return self._profiles[name]
        except KeyError as exc:
            available = ", ".join(self._profiles) or "(none)"
            raise ValueError(
                f"provider profile '{name}' was not found; available profiles: {available}"
            ) from exc

    def resolve_api_key(self, profile_name: str) -> str:
        """Return the key stored in the selected profile."""

        profile = self.get(profile_name)
        if profile.api_key:
            return profile.api_key
        if profile.api_key_env:
            api_key = os.getenv(profile.api_key_env)
            if isinstance(api_key, str) and api_key.strip():
                return api_key.strip()
        if profile.api_key_env:
            raise ValueError(
                f"provider profile '{profile.name}' requires environment variable "
                f"{profile.api_key_env}"
            )
        raise ValueError(f"provider profile '{profile.name}' has an empty api_key")

    def register_ephemeral(
        self,
        name: str,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
    ) -> None:
        """Register an in-memory profile for legacy integrations.

        New applications should put metadata and credentials in TOML. This
        escape hatch exists only for callers that inject
        credentials programmatically (for example test doubles).
        """

        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        profile = _parse_profile(
            name,
            {
                "base_url": base_url,
                "api_key": api_key,
                "default_model": default_model,
                "models": [default_model],
            },
        )
        profiles = dict(self._profiles)
        if name in profiles:
            raise ValueError(f"provider profile '{name}' already exists")
        profiles[name] = profile
        self._profiles = MappingProxyType(profiles)


def _parse_document(
    document: Mapping[str, Any], config_path: Path
) -> tuple[dict[str, ProviderProfile], str]:
    if not isinstance(document, Mapping):
        raise ValueError(f"provider profile configuration must be a TOML table: {config_path}")
    defaults = document.get("defaults")
    raw_profiles = document.get("profiles")
    if not isinstance(defaults, Mapping):
        raise ValueError(f"provider profile configuration needs a [defaults] table: {config_path}")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ValueError(f"provider profile configuration needs one [profiles.<name>] table: {config_path}")
    active_profile = _required_string(defaults, "active_profile", "[defaults]")
    profiles: dict[str, ProviderProfile] = {}
    for name, raw_profile in raw_profiles.items():
        if not isinstance(name, str) or not _PROFILE_NAME_RE.fullmatch(name):
            raise ValueError(
                "profile names must use letters, numbers, underscores, or hyphens "
                "and start with a letter or number"
            )
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"[profiles.{name}] must be a TOML table")
        profiles[name] = _parse_profile(name, raw_profile)
    if active_profile not in profiles:
        raise ValueError(
            f"[defaults].active_profile '{active_profile}' does not name a configured profile"
        )
    return profiles, active_profile


def _parse_profile(name: str, raw_profile: Mapping[str, Any]) -> ProviderProfile:
    adapter = _optional_string(raw_profile, "adapter", "openai_compatible")
    if adapter != "openai_compatible":
        raise ValueError(
            f"profile '{name}' has unsupported adapter '{adapter}'; "
            "only 'openai_compatible' is currently implemented"
        )
    base_url = _required_string(
        raw_profile,
        "api_url" if "api_url" in raw_profile else "base_url",
        f"profile '{name}'",
    )
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"profile '{name}' base_url must be an absolute HTTP(S) URL")
    raw_api_key = raw_profile.get("api_key")
    api_key_env: str | None = None
    if raw_api_key is None:
        # Accept the previous field for one migration cycle. A value that is
        # not a valid variable name is treated as the old, accidentally
        # inlined key so existing local files continue to work.
        legacy_key = raw_profile.get("api_key_env")
        if isinstance(legacy_key, str) and legacy_key.strip():
            if _ENV_NAME_RE.fullmatch(legacy_key.strip()):
                api_key_env = legacy_key.strip()
                raw_api_key = ""
            else:
                raw_api_key = legacy_key
        else:
            raise ValueError(f"profile '{name}' requires non-empty 'api_key'")
    if not isinstance(raw_api_key, str):
        raise ValueError(f"profile '{name}' requires non-empty 'api_key'")
    if not raw_api_key.strip() and api_key_env is None:
        raise ValueError(f"profile '{name}' requires non-empty 'api_key'")
    default_model = _required_string(raw_profile, "default_model", f"profile '{name}'")
    raw_models = raw_profile.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"profile '{name}' models must be a non-empty TOML array")
    models = tuple(_validate_model(name, model) for model in raw_models)
    if len(set(models)) != len(models):
        raise ValueError(f"profile '{name}' models must not contain duplicates")
    if default_model not in models:
        raise ValueError(f"profile '{name}' default_model must appear in models")
    tool_mode = _optional_string(raw_profile, "tool_mode", "native_strict")
    if tool_mode not in _TOOL_MODES:
        options = ", ".join(sorted(_TOOL_MODES))
        raise ValueError(f"profile '{name}' tool_mode must be one of: {options}")
    return ProviderProfile(
        name=name,
        adapter=adapter,
        base_url=base_url.rstrip("/"),
        api_key=raw_api_key.strip(),
        api_key_env=api_key_env,
        default_model=default_model,
        models=models,
        tool_mode=tool_mode,
    )


def _required_string(source: Mapping[str, Any], key: str, location: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} requires non-empty string '{key}'")
    return value.strip()


def _optional_string(source: Mapping[str, Any], key: str, default: str) -> str:
    value = source.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string when configured")
    return value.strip()


def _validate_model(profile_name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"profile '{profile_name}' models must contain non-empty strings")
    return value.strip()


__all__ = ["ProviderProfile", "ProviderRegistry"]
