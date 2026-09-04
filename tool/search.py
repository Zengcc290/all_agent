"""AnySearch-backed web search tool.

The public AnySearch API exposes ``POST /v1/search`` and returns a response
envelope whose search entries live under ``data.results``. This module keeps
the provider-specific request and response details behind the normal single
file tool protocol used by the project.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from typing import Annotated, Any, Literal
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
)

from core import BaseTool, ToolSpec

# Discovery uses this literal switch before constructing the tool.
TOOL_ENABLED = True


class SearchInput(BaseModel):
    """Arguments accepted by AnySearch's ``/v1/search`` endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        # ``limit`` was used by the original local tool. It remains a
        # validation-only alias so existing callers can migrate without
        # changing the provider-facing ``max_results`` request field.
        populate_by_name=True,
    )

    query: str = Field(min_length=1, max_length=500, description="Search query")
    max_results: int = Field(
        default=10,
        ge=1,
        le=10,
        validation_alias=AliasChoices("max_results", "limit"),
        serialization_alias="max_results",
        description="Number of results to return (1-10)",
    )
    tag: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "Optional AnySearch capability tag. Leave null for an ordinary web "
            "search; never use the tool name (web.search/web) as a tag."
        ),
    )
    zone: Literal["cn", "intl"] | None = Field(
        default=None, description="Optional search region"
    )
    language: str | None = Field(
        default=None, min_length=1, max_length=32, description="Preferred language"
    )
    # AnySearch forwards this object to AnyMix and therefore accepts provider
    # specific keys. The runtime's strict function-schema dialect does not
    # permit ``additionalProperties: true``; WithJsonSchema keeps the public
    # schema valid while Pydantic still validates the value as a dictionary.
    params: Annotated[
        dict[str, Any] | None,
        WithJsonSchema(
            {
                "anyOf": [
                    {"type": "object", "additionalProperties": False},
                    {"type": "null"},
                ],
                "description": (
                    "Optional provider-specific parameters passed to AnyMix as a "
                    "JSON object. Leave null unless the selected capability tag "
                    "requires parameters; do not encode the object as a string."
                ),
            }
        ),
    ] = None
    format: Literal["json", "markdown"] | None = Field(
        default=None,
        description="Optional result format; the outer API response remains JSON",
    )

    @field_validator("tag", mode="before")
    @classmethod
    def normalize_tag(cls, value: Any) -> Any:
        """Normalize common nullable values emitted by OpenAI-compatible models.

        Some compatible models serialize a nullable string as ``""`` or
        ``"None"``.  Treating those values as absent keeps the provider request
        valid while preserving strict validation for all other non-string types.
        The tool-name aliases are not AnySearch capability tags and are omitted
        when a model mistakenly copies the function name into ``tag``.
        """

        normalized = _normalize_nullable_text(value)
        if normalized is None:
            return None
        if normalized.casefold() in {"web", "web.search", "web__search"}:
            return None
        return normalized

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, value: Any) -> Any:
        """Normalize empty/null sentinels without accepting tool-name aliases."""

        return _normalize_nullable_text(value)

    @field_validator("params", mode="before")
    @classmethod
    def normalize_provider_params(cls, value: Any) -> Any:
        """Accept JSON-encoded nullable objects from weaker tool-call clients.

        The public contract remains ``object | null``.  A small compatibility
        exception is made for providers that return the JSON object as a string
        (for example ``"{}"``) or serialize null as ``"None"``.  Arbitrary
        strings are left untouched and are rejected by strict Pydantic
        validation.
        """

        if value is None or not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized or normalized.casefold() in {"none", "null"}:
            return None
        try:
            parsed = json.loads(normalized)
        except (TypeError, ValueError):
            return value
        return parsed if isinstance(parsed, dict) else value

    @field_validator("query")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text values must contain a non-whitespace character")
        return value

    @property
    def limit(self) -> int:
        """Backward-compatible view of the former ``limit`` argument."""

        return self.max_results


class SearchItem(BaseModel):
    """Stable subset of an AnySearch result exposed to the agent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(default="", max_length=1000)
    url: str = Field(default="", max_length=2000)
    snippet: str = Field(default="", max_length=5000)


class SearchOutput(BaseModel):
    """Normalized result contract returned by ``web.search``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[SearchItem]

    @property
    def results(self) -> list[SearchItem]:
        """Alias matching AnySearch's provider response terminology."""

        return self.items


class SearchTool(BaseTool):
    """Execute authenticated searches against AnySearch."""

    spec = ToolSpec(
        name="web.search",
        description=(
            "Search public web content with AnySearch and return normalized title, "
            "URL and snippet items. Use it when current public web information is "
            "needed."
        ),
        version="1.1",
        input_model=SearchInput,
        output_model=SearchOutput,
        side_effect="read",
        # Permission metadata is intentionally empty; access is not filtered
        # by permissions in the current public-tool deployment.
        permissions=(),
        timeout_seconds=30.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=8,
        tags=("web", "search"),
        recommended_before_tools=("system.current_time",),
    )

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        # Loading happens in the constructor (rather than module import) so
        # discovery remains free of configuration and I/O side effects.
        if base_url is None or api_key is None:
            _load_dotenv()
        self.base_url = (
            base_url
            if base_url is not None
            else _first_env("SEARCH_BASE_URL", "ANYSEARCH_BASE_URL")
        )
        self.api_key = (
            api_key
            if api_key is not None
            else _first_env("SEARCH_API", "SEARCH_API_KEY", "ANYSEARCH_API_KEY")
        )
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise TypeError("base_url must be a string")
        if self.base_url is not None:
            self.base_url = self.base_url.strip()
            if not self.base_url:
                raise ValueError("base_url must be a non-empty string")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        if self.api_key is not None:
            self.api_key = self.api_key.strip()
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        self.timeout = float(timeout)
        # The execution manager enforces the same deadline as urllib.
        self.spec = replace(type(self).spec, timeout_seconds=self.timeout)

    def execute(self, arguments: SearchInput) -> SearchOutput:
        if not isinstance(arguments, SearchInput):
            raise TypeError("arguments must be a SearchInput instance")
        if not self.base_url:
            raise RuntimeError("SEARCH_BASE_URL is not configured")
        if not self.api_key:
            raise RuntimeError("SEARCH_API is not configured")

        endpoint = _search_endpoint(self.base_url)
        request_body: dict[str, Any] = {
            "query": arguments.query,
            "max_results": arguments.max_results,
        }
        for name in ("tag", "zone", "language", "params", "format"):
            value = getattr(arguments, name)
            if value is not None:
                request_body[name] = value
        try:
            encoded_body = json.dumps(
                request_body,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("search parameters must be JSON-serializable") from exc

        request = Request(
            endpoint,
            data=encoded_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_body = _read_response_body(response)
        except HTTPError as exc:
            # Very old AnySearch-compatible gateways exposed a GET endpoint.
            # Keep a narrow 501 fallback for those deployments; the documented
            # POST request above remains the normal and preferred path.
            if exc.code != 501:
                raise
            exc.close()
            legacy_endpoint = _legacy_search_endpoint(endpoint, arguments)
            legacy_request = Request(
                legacy_endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                method="GET",
            )
            with urlopen(legacy_request, timeout=self.timeout) as response:
                raw_body = _read_response_body(response)
        try:
            payload = json.loads(
                raw_body.decode("utf-8"), parse_constant=_reject_json_constant
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("search response was not valid UTF-8 JSON") from exc
        return _normalize_response(payload, arguments.max_results)


def _load_dotenv() -> None:
    """Load the repository's .env when python-dotenv is available."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        # Environment variables are still usable when the optional loader is
        # unavailable; the project declares python-dotenv as a dependency.
        return
    load_dotenv(override=False)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value
    return None


def _normalize_nullable_text(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    normalized = value.strip()
    if not normalized or normalized.casefold() in {"none", "null"}:
        return None
    return normalized


def _search_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("SEARCH_BASE_URL must be an absolute HTTP(S) URL")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1/search"
    elif path.endswith("/v1"):
        path += "/search"
    return urlunsplit(parsed._replace(path=path, fragment=""))


def _legacy_search_endpoint(endpoint: str, arguments: SearchInput) -> str:
    parsed = urlsplit(endpoint)
    query = urlencode({"q": arguments.query, "limit": arguments.max_results})
    merged_query = f"{parsed.query}&{query}" if parsed.query else query
    return urlunsplit(parsed._replace(query=merged_query, fragment=""))


def _read_response_body(response: Any) -> bytes:
    headers = getattr(response, "headers", None)
    content_length = None
    if headers is not None:
        content_length = headers.get("Content-Length") or headers.get("content-length")
    if content_length is not None:
        try:
            content_length_value = int(content_length)
        except (TypeError, ValueError) as exc:
            raise TypeError("invalid Content-Length header") from exc
        if content_length_value < 0:
            raise TypeError("invalid Content-Length header")
        if content_length_value > 2_000_000:
            raise ValueError("search response is too large")
    try:
        body = response.read(2_000_001)
    except TypeError:
        # Small test doubles and a few urllib-compatible adapters expose the
        # zero-argument ``read()`` form only. They are still bounded by the
        # Content-Length check above when that header is available.
        body = response.read()
    if len(body) > 2_000_000:
        raise ValueError("search response is too large")
    return body


def _normalize_response(payload: Any, max_results: int) -> SearchOutput:
    if not isinstance(payload, dict):
        raise TypeError("search response must be a JSON object")

    # The documented envelope is code/message/request_id/data. A top-level
    # results/items fallback keeps the tool tolerant of older AnySearch-style
    # gateways without weakening validation of the documented response.
    code = payload.get("code")
    if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
        raise TypeError("search response code must be an integer")
    if code not in (None, 0):
        raise RuntimeError("AnySearch search request failed")

    data = payload.get("data")
    if data is not None and not isinstance(data, dict):
        raise TypeError("search response data must be an object")
    if data is None and "items" not in payload and "results" not in payload:
        raise TypeError("search response is missing data.results")
    raw_items = (
        data.get("results", [])
        if isinstance(data, dict)
        else payload.get("items", payload.get("results", []))
    )
    if not isinstance(raw_items, list):
        raise TypeError("search response results must be a list")

    items = []
    for item in raw_items[:max_results]:
        if not isinstance(item, dict):
            continue
        items.append(
            SearchItem(
                title=_text(item.get("title"))[:1000],
                url=_text(item.get("url", item.get("link")))[:2000],
                snippet=_text(item.get("snippet", item.get("description")))[:5000],
            )
        )
    return SearchOutput(items=items)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def create_tool() -> BaseTool:
    """Build the configured instance used by automatic tool discovery."""

    return SearchTool()
