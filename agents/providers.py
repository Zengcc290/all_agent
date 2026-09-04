"""Named provider-profile configuration for OpenAI-compatible LLMs.

Provider metadata is kept in a TOML file while credentials remain environment
variables.  A profile therefore contains no secret material and is safe to
commit or share with the project.
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

from dotenv import load_dotenv

_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOOL_MODES = frozenset({"native_strict", "native_loose", "text_react", "none"})


@dataclass(frozen=True)
class ProviderProfile:
    """One named, secret-free OpenAI-compatible provider configuration."""

    name: str
    base_url: str
    api_key_env: str
    default_model: str
    models: tuple[str, ...]
    adapter: str = "openai_compatible"
    tool_mode: str = "native_strict"

    def public_info(self) -> dict[str, Any]:
        """Return inspectable profile metadata without resolving its secret."""

        return {
            "name": self.name,
            "adapter": self.adapter,
            "base_url": self.base_url,
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
        self._ephemeral_keys: dict[str, str] = {}
        self.active_profile = ""
        self.reload()

    @staticmethod
    def default_config_path() -> Path:
        return Path(__file__).resolve().parent.parent / "config" / "providers.toml"

    @property
    def profiles(self) -> Mapping[str, ProviderProfile]:
        return self._profiles

    def reload(self) -> None:
        """Reload configuration and environment variables without exposing keys."""

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"provider profile configuration was not found: {self.config_path}"
            )
        dotenv_path = self.config_path.parent.parent / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=False)
        try:
            with self.config_path.open("rb") as handle:
                document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"invalid provider profile TOML: {self.config_path}"
            ) from exc
        profiles, active_profile = _parse_document(document, self.config_path)
        self._ephemeral_keys.clear()
        # One-release migration bridge for projects that still have the old
        # single-profile variables. New profile variables always take priority.
        if (
            active_profile == "openai"
            and os.getenv("LLM_OPENAI_API_KEY") is None
            and os.getenv("LLM_API_KEY")
        ):
            legacy_url = os.getenv("LLM_BASE_URL")
            legacy_model = os.getenv("LLM_MODEL")
            current = profiles[active_profile]
            if legacy_url and legacy_model:
                profiles[active_profile] = ProviderProfile(
                    name=current.name,
                    adapter=current.adapter,
                    base_url=legacy_url.rstrip("/"),
                    api_key_env="LLM_API_KEY",
                    default_model=legacy_model,
                    models=tuple(dict.fromkeys((*current.models, legacy_model))),
                    tool_mode=current.tool_mode,
                )
            else:
                profiles[active_profile] = ProviderProfile(
                    name=current.name,
                    adapter=current.adapter,
                    base_url=current.base_url,
                    api_key_env="LLM_API_KEY",
                    default_model=current.default_model,
                    models=current.models,
                    tool_mode=current.tool_mode,
                )
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
        """Resolve a profile credential only at client creation time."""

        profile = self.get(profile_name)
        api_key = self._ephemeral_keys.get(profile_name) or os.getenv(profile.api_key_env)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                f"provider profile '{profile.name}' requires environment variable "
                f"{profile.api_key_env}"
            )
        return api_key.strip()

    def register_ephemeral(
        self,
        name: str,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
    ) -> None:
        """Register an in-memory profile for legacy integrations.

        New applications should put metadata in TOML and credentials in the
        environment. This escape hatch exists only for callers that inject
        credentials programmatically (for example test doubles).
        """

        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        profile = _parse_profile(
            name,
            {
                "base_url": base_url,
                "api_key_env": f"__ALL_AGENT_EPHEMERAL_{re.sub(r'[^A-Z0-9_]', '_', name.upper())}_KEY",
                "default_model": default_model,
                "models": [default_model],
            },
        )
        profiles = dict(self._profiles)
        if name in profiles:
            raise ValueError(f"provider profile '{name}' already exists")
        profiles[name] = profile
        self._profiles = MappingProxyType(profiles)
        self._ephemeral_keys[name] = api_key.strip()


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
    base_url = _required_string(raw_profile, "base_url", f"profile '{name}'")
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"profile '{name}' base_url must be an absolute HTTP(S) URL")
    api_key_env = _required_string(raw_profile, "api_key_env", f"profile '{name}'")
    if not _ENV_NAME_RE.fullmatch(api_key_env):
        raise ValueError(f"profile '{name}' api_key_env must be an environment variable name")
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
