import pytest
from pydantic import BaseModel, ConfigDict

from core import ToolSpec, ToolSpecRepository
from tool.search import SearchTool


class RepoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class RepoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


def test_repository_persists_versioned_metadata(tmp_path):
    repository = ToolSpecRepository(tmp_path / "tools.sqlite3")
    search = SearchTool(base_url="https://example.invalid")
    repository.save(search.spec)
    stored = repository.get(search.spec.name, search.spec.version)
    assert stored["schema_hash"] == search.spec.schema_hash
    assert stored["input_schema"]["properties"]["query"]["type"] == "string"
    assert repository.search("web search")[0]["tool_name"] == "web.search"
    assert repository.active_tool_names() == ["web.search"]


def test_repository_returns_natural_latest_version_and_one_entry_per_tool(tmp_path):
    repository = ToolSpecRepository(tmp_path / "tools.sqlite3")
    for version in ("2.0", "10.0"):
        repository.save(
            ToolSpec(
                name="test.versioned",
                description="versioned",
                version=version,
                input_model=RepoInput,
                output_model=RepoOutput,
                max_concurrency=3,
            )
        )

    assert repository.get("test.versioned")["version"] == "10.0"
    matches = repository.search("versioned")
    assert [
        (item["tool_name"], item["version"], item["max_concurrency"])
        for item in matches
    ] == [
        ("test.versioned", "10.0", 3),
    ]
    assert repository.active_tool_names() == ["test.versioned"]

    with pytest.raises(ValueError, match="between 1 and 20"):
        repository.search("versioned", 0)


def test_repository_sorts_numeric_prerelease_components_naturally(tmp_path):
    repository = ToolSpecRepository(tmp_path / "tools.sqlite3")
    for version in ("1.0-alpha.10", "1.0-alpha.2"):
        repository.save(
            ToolSpec(
                name="test.prerelease",
                description="prerelease",
                version=version,
                input_model=RepoInput,
                output_model=RepoOutput,
            )
        )

    assert repository.get("test.prerelease")["version"] == "1.0-alpha.10"
