from core import ToolSpecRepository
from tool.search import SearchTool


def test_repository_persists_versioned_metadata(tmp_path):
    repository = ToolSpecRepository(tmp_path / "tools.sqlite3")
    search = SearchTool(base_url="https://example.invalid")
    repository.save(search.spec)
    stored = repository.get(search.spec.name, search.spec.version)
    assert stored["schema_hash"] == search.spec.schema_hash
    assert stored["input_schema"]["properties"]["query"]["type"] == "string"
    assert repository.search("web search")[0]["tool_name"] == "web.search"
