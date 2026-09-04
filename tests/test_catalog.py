import pytest

from core import (
    CatalogInput,
    ExecutionContext,
    ToolCall,
    ToolCatalogTool,
    ToolExecutionManager,
    ToolRegistry,
    ToolSpecRepository,
)
from tool.search import SearchTool


def test_catalog_exposes_permissioned_tools_without_context_permissions():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    catalog = ToolCatalogTool(registry)
    candidates = catalog.execute(
        CatalogInput(action="search", intent="search web", limit=5)
    )
    assert candidates.candidates[0]["tool_name"] == "web.search"

    spec = catalog.execute(
        CatalogInput(action="get_spec", tool_name="web.search")
    )
    assert spec.spec["input_schema"]["properties"]["query"]["type"] == "string"
    assert spec.spec["schema_hash"] == search.spec.schema_hash
    assert spec.spec["recommended_before_tools"] == ["system.current_time"]
    assert spec.spec["registry_generation"] == registry.resolve("web.search")[1]


def test_catalog_resolve_requires_a_matching_intent():
    registry = ToolRegistry()
    registry.register(SearchTool(base_url="https://example.invalid"))
    catalog = ToolCatalogTool(registry)

    with pytest.raises(ValueError, match="no matching tool"):
        catalog.execute(CatalogInput(action="resolve", intent="completely unrelated"))

    with pytest.raises(ValueError, match="intent is required"):
        catalog.execute(CatalogInput(action="resolve"))


def test_catalog_resolve_returns_all_matching_specs():
    registry = ToolRegistry()
    first = SearchTool(base_url="https://example.invalid")

    # Reuse the registry with two contracts whose descriptions share distinct
    # capability terms; resolve should return both in one response.
    registry.register(first)
    from tool.current_time import CurrentTimeTool

    current_time = CurrentTimeTool()
    registry.register(current_time)
    catalog = ToolCatalogTool(registry)

    result = catalog.execute(
        CatalogInput(action="resolve", intent="web search current time", limit=20)
    )

    assert result.spec is not None
    assert {item["tool_name"] for item in result.specs} == {
        "web.search",
        "system.current_time",
    }
    assert all(item["input_schema"] for item in result.specs)


@pytest.mark.asyncio
async def test_catalog_uses_an_in_memory_repository_from_runtime_thread():
    repository = ToolSpecRepository(":memory:")
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    repository.save(search.spec)
    catalog = ToolCatalogTool(registry, repository)
    registry.register(catalog)
    repository.save(catalog.spec)
    _, generation = registry.resolve(catalog.spec.name)
    call = ToolCall(
        call_id="catalog",
        tool_name=catalog.spec.name,
        schema_version=catalog.spec.version,
        schema_hash=catalog.spec.schema_hash,
        registry_generation=generation,
        arguments={"action": "search", "intent": "web search", "limit": 5},
    )

    result = await ToolExecutionManager(registry).execute_batch(
        [call], ExecutionContext(permissions=frozenset({"network.read"}))
    )

    assert result.results[0].ok
    assert result.results[0].data["candidates"][0]["tool_name"] == "web.search"
    repository.close()
