from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from core.models import ToolSpec
from core.registry import BaseTool


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class SearchItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = ""
    url: str = ""
    snippet: str = ""


class SearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[SearchItem]


class SearchTool(BaseTool):
    spec = ToolSpec(
        name="web.search",
        description="Search public web content and return normalized title, URL and snippet items.",
        version="1.0",
        input_model=SearchInput,
        output_model=SearchOutput,
        side_effect="read",
        parallel_safe=True,
        tags=("web", "search"),
    )

    def __init__(self, base_url: str | None = None, api_key: str | None = None, *, timeout: float = 30.0) -> None:
        self.base_url = base_url or os.getenv("SEARCH_BASE_URL")
        self.api_key = api_key or os.getenv("SEARCH_API")
        self.timeout = timeout

    def execute(self, arguments: SearchInput) -> SearchOutput:
        if not self.base_url:
            raise RuntimeError("SEARCH_BASE_URL is not configured")
        query = urlencode({"q": arguments.query, "limit": arguments.limit})
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        request = Request(f"{self.base_url}?{query}", headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw_items = payload.get("items", payload.get("results", []))
        if not isinstance(raw_items, list):
            raise ValueError("search response items must be a list")
        items = [
            SearchItem(
                title=str(item.get("title", "")),
                url=str(item.get("url", item.get("link", ""))),
                snippet=str(item.get("snippet", item.get("description", ""))),
            )
            for item in raw_items[: arguments.limit]
            if isinstance(item, dict)
        ]
        return SearchOutput(items=items)

