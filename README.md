# all-agent

This project provides a small, typed runtime for OpenAI-compatible agents and tools.

## Tool execution

Tools declare their input and output Pydantic models in `ToolSpec`. The runtime validates every call before execution, checks the schema version/hash, applies permissions and confirmation rules, and returns one result per call.

Independent calls can run concurrently. Calls with `depends_on` are executed in dependency layers. Tools marked `parallel_safe=False` are isolated into single-call layers.

```python
import asyncio

from core import ToolCall, ToolExecutionManager, ToolRegistry
from tool.search import SearchTool


async def main():
    registry = ToolRegistry()
    search = SearchTool()
    registry.register(search)
    manager = ToolExecutionManager(registry, max_concurrency=4)

    def call(call_id, query):
        return ToolCall(
            call_id=call_id,
            tool_name=search.spec.name,
            schema_version=search.spec.version,
            schema_hash=search.spec.schema_hash,
            arguments={"query": query, "limit": 5},
        )

    result = await manager.execute_batch([
        call("search-python", "Python 3.12"),
        call("search-pydantic", "Pydantic 2"),
    ])
    print(result.model_dump())


asyncio.run(main())
```

`ToolCatalogTool` is available as `system.tool_catalog` for summary search and versioned schema loading. It exposes fixed operations and does not execute arbitrary SQL.

Run tests with:

```text
.venv\\Scripts\\python.exe -m pytest -q
```
