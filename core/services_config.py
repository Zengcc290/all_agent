"""Central loader for every external API / cloud-service call.

``config/services.toml`` (gitignored; the publishable template is
``config/services.example.toml``) is the single place that describes the
endpoints, models and credentials for the external services the project calls:
embedding, vision extraction, web search, Qdrant Cloud and Neo4j Aura.

Secrets follow the ``provider.toml`` convention: a section either carries the
plaintext value (the file is gitignored) or names an environment variable via
``*_env``, resolved at load time.

Precedence is deliberately **explicit argument > services.toml** (only these
two layers; the historical ``.env`` / ``HELLOAGENTS_MEMORY_*`` environment
priority layer has been removed — configuration lives in ``config/`` and
``constants.py``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

#: Schemes each section accepts.  Neo4j Aura uses ``bolt+s``/``neo4j+s``, which
#: the LLM provider parser deliberately rejects — one more reason these live in
#: a separate file with its own validation.
_HTTP_SCHEMES = frozenset({"http", "https"})
_NEO4J_SCHEMES = frozenset({"bolt", "bolt+s", "neo4j", "neo4j+s", "http", "https"})
_EMBEDDING_PROVIDERS = frozenset({"auto", "openai", "hash"})


@dataclass(frozen=True)
class EmbeddingService:
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    dimension: int | None = None
    batch_size: int | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class VisionService:
    """Vision-language model for image → entity/relation extraction.

    Only the model name lives here. The endpoint and credentials come from the
    active ``config/provider.toml`` profile (OpenAI-compatible chat API), so a
    vision-capable model on the same provider is all this section selects.
    """

    model: str | None = None


@dataclass(frozen=True)
class SearchService:
    base_url: str | None = None
    api_key: str | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class QdrantService:
    url: str | None = None
    api_key: str | None = None
    collection: str | None = None


@dataclass(frozen=True)
class Neo4jService:
    uri: str | None = None
    username: str | None = None
    password: str | None = None


@dataclass(frozen=True)
class ProxyService:
    """Local forward proxy (e.g. Clash on port 7890) cloud calls route through.

    ``url`` is an ``http://host:port`` CONNECT proxy; Neo4j Aura is tunneled
    through it because the Neo4j driver has no native proxy support, and
    Qdrant Cloud passes it to the underlying HTTP client.
    """

    url: str | None = None


@dataclass(frozen=True)
class ServicesConfig:
    """Parsed view of ``config/services.toml``; missing sections are all-None."""

    embedding: EmbeddingService = EmbeddingService()
    vision: VisionService = VisionService()
    search: SearchService = SearchService()
    qdrant: QdrantService = QdrantService()
    neo4j: Neo4jService = Neo4jService()
    proxy: ProxyService = ProxyService()

    @property
    def configured(self) -> bool:
        """True when any section carries a value (file exists and is used)."""

        return any(
            value is not None
            for section in (self.embedding, self.vision, self.search, self.qdrant, self.neo4j, self.proxy)
            for value in section.__dict__.values()
        )


def default_config_path() -> Path:
    """Prefer the private runtime file; fall back to the publishable template."""

    config_dir = Path(__file__).resolve().parent.parent / "config"
    for filename in ("services.toml", "services.example.toml"):
        candidate = config_dir / filename
        if candidate.is_file():
            return candidate
    return config_dir / "services.toml"


def load_services_config(path: str | Path | None = None) -> ServicesConfig:
    """Parse the services TOML; a missing file yields an empty configuration.

    Never raises for a missing file: the project must stay usable (offline
    memory fallbacks, no search) when no services file exists.  A malformed
    TOML *does* raise, because that signals a broken local edit.
    """

    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    if not config_path.is_file():
        return ServicesConfig()
    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid services TOML: {config_path}") from exc
    if not isinstance(document, dict):
        raise TypeError(f"services configuration must be a TOML table: {config_path}")
    return ServicesConfig(
        embedding=_parse_embedding(_table(document, "embedding")),
        vision=_parse_vision(_table(document, "vision")),
        search=_parse_search(_table(document, "search")),
        qdrant=_parse_qdrant(_table(document, "qdrant")),
        neo4j=_parse_neo4j(_table(document, "neo4j")),
        proxy=_parse_proxy(_table(document, "proxy")),
    )


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    table = document.get(name, {})
    if not isinstance(table, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return table


def _parse_embedding(table: dict[str, Any]) -> EmbeddingService:
    provider = _optional_string(table, "provider")
    if provider is not None and provider.casefold() not in _EMBEDDING_PROVIDERS:
        raise ValueError(
            f"[embedding].provider must be one of: {', '.join(sorted(_EMBEDDING_PROVIDERS))}"
        )
    base_url = _optional_string(table, "base_url")
    if base_url is not None:
        _require_scheme("embedding", "base_url", base_url, _HTTP_SCHEMES)
    model = _optional_string(table, "model")
    api_key = _resolve_secret(table)
    dimension = _optional_positive_int(table, "dimension", "[embedding]")
    batch_size = _optional_positive_int(table, "batch_size", "[embedding]")
    timeout = _optional_positive_float(table, "timeout", "[embedding]")
    return EmbeddingService(
        provider=provider.casefold() if provider else None,
        base_url=base_url,
        model=model,
        api_key=api_key,
        dimension=dimension,
        batch_size=batch_size,
        timeout=timeout,
    )


def _parse_vision(table: dict[str, Any]) -> VisionService:
    return VisionService(model=_optional_string(table, "model"))


def _parse_search(table: dict[str, Any]) -> SearchService:
    base_url = _optional_string(table, "base_url")
    if base_url is not None:
        _require_scheme("search", "base_url", base_url, _HTTP_SCHEMES)
    return SearchService(
        base_url=base_url,
        api_key=_resolve_secret(table),
        timeout=_optional_positive_float(table, "timeout", "[search]"),
    )


def _parse_qdrant(table: dict[str, Any]) -> QdrantService:
    url = _optional_string(table, "url")
    if url is not None:
        _require_scheme("qdrant", "url", url, _HTTP_SCHEMES)
    return QdrantService(
        url=url,
        api_key=_resolve_secret(table),
        collection=_optional_string(table, "collection"),
    )


def _parse_neo4j(table: dict[str, Any]) -> Neo4jService:
    uri = _optional_string(table, "uri")
    if uri is not None:
        _require_scheme("neo4j", "uri", uri, _NEO4J_SCHEMES)
    return Neo4jService(
        uri=uri,
        username=_optional_string(table, "username"),
        password=_resolve_secret(table, key="password", env_key="password_env"),
    )


def _parse_proxy(table: dict[str, Any]) -> ProxyService:
    url = _optional_string(table, "url")
    if url is not None:
        _require_scheme("proxy", "url", url, _HTTP_SCHEMES)
    return ProxyService(url=url)


def _resolve_secret(
    table: dict[str, Any], *, key: str = "api_key", env_key: str = "api_key_env"
) -> str | None:
    """Plaintext value wins; otherwise resolve the named environment variable."""

    value = _optional_string(table, key)
    if value is not None:
        return value
    env_name = _optional_string(table, env_key)
    if env_name is None:
        return None
    resolved = os.getenv(env_name)
    if resolved is None or not resolved.strip():
        return None
    return resolved.strip()


def _optional_string(table: dict[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        # Blank values mean "not configured" (the .env convention) so a template
        # file with empty placeholders never crashes consumers.
        return None
    return value.strip()


def _optional_positive_int(table: dict[str, Any], key: str, location: str) -> int | None:
    value = table.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{location}.{key} must be a positive integer")
    return value


def _optional_positive_float(table: dict[str, Any], key: str, location: str) -> float | None:
    value = table.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{location}.{key} must be a positive number")
    return float(value)


def _require_scheme(location: str, key: str, url: str, schemes: frozenset[str]) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in schemes or not parsed.netloc:
        allowed = ", ".join(sorted(schemes))
        raise ValueError(f"[{location}].{key} must be an absolute URL with scheme in: {allowed}")


__all__ = [
    "EmbeddingService",
    "Neo4jService",
    "ProxyService",
    "QdrantService",
    "SearchService",
    "ServicesConfig",
    "VisionService",
    "default_config_path",
    "load_services_config",
]
