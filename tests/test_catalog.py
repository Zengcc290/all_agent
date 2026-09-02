from core import CatalogInput, ToolCatalogTool, ToolRegistry
from tool.search import SearchTool


def test_catalog_returns_summary_and_full_schema():
    registry = ToolRegistry()
    search = SearchTool(base_url="https://example.invalid")
    registry.register(search)
    catalog = ToolCatalogTool(registry)

    candidates = catalog.execute(CatalogInput(action="search", intent="search web", limit=5))
    assert candidates.candidates[0]["tool_name"] == "web.search"

    spec = catalog.execute(CatalogInput(action="get_spec", tool_name="web.search"))
    assert spec.spec["input_schema"]["properties"]["query"]["type"] == "string"
    assert spec.spec["schema_hash"] == search.spec.schema_hash
