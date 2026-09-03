# all-agent

This project provides a small, typed runtime for OpenAI-compatible agents and tools.

## Single-file tool plugins

`Agent` automatically discovers enabled Python modules directly inside the trusted
`tool` package. Copy `tool/tool_template.py`, implement its typed contract, and
set `TOOL_ENABLED = True`; no central registration list needs to be edited.
Discovery outcomes and registration state are queryable from the Agent. See the
[Chinese tool guide](tool/README.md) for the complete template rules and the
function-level LLM/tool call chain.

## Tool execution

Tools declare their input and output Pydantic models in `ToolSpec`. Both models must use `ConfigDict(extra="forbid")`; this prevents unknown fields from being silently discarded. The runtime validates every call before execution, checks the registration generation and schema version/hash, applies permissions and confirmation rules, and returns one result per call.

Independent calls can run concurrently. Calls with `depends_on` are executed in dependency layers. Tools marked `parallel_safe=False` are isolated into single-call layers.

```python
import asyncio

from core import ExecutionContext, ToolCall, ToolExecutionManager, ToolRegistry
from tool.search import SearchTool


async def main():
    registry = ToolRegistry()
    search = SearchTool()
    registry.register(search)
    _, generation = registry.resolve(search.spec.name)
    manager = ToolExecutionManager(registry, max_concurrency=4)

    def call(call_id, query):
        return ToolCall(
            call_id=call_id,
            tool_name=search.spec.name,
            schema_version=search.spec.version,
            schema_hash=search.spec.schema_hash,
            registry_generation=generation,
            arguments={"query": query, "limit": 5},
        )

    result = await manager.execute_batch(
        [
            call("search-python", "Python 3.12"),
            call("search-pydantic", "Pydantic 2"),
        ],
        ExecutionContext(permissions=frozenset({"network.read"})),
    )
    print(result.model_dump())


asyncio.run(main())
```

`ToolCatalogTool` is available as `system.tool_catalog` for summary search and versioned schema loading. It exposes fixed operations and does not execute arbitrary SQL. Pass a `ToolSpecRepository` to `Agent` to persist the active catalog, and use `run_with_tools(..., defer_tool_loading=True)` to expose only the catalog initially and load a resolved tool on the next model round.

Project changes are recorded compactly in SQLite by the auto-discovered
`system.update_log` tool. Read [update_log_readme_first.md](update_log_readme_first.md)
before making changes. After every actual modification, grant the tool
`database.write` permission plus its generation-bound confirmation and call it
once; the guide stores only the last/next numeric ID, while full structured
entries remain in `update_log.sqlite3`.

OpenAI function definitions are generated in strict mode. Malformed, forbidden, or stale native calls are returned to the model as structured tool errors so it can correct the request without executing unvalidated input. Configured Agent prompts are applied automatically; previous `history` can be included explicitly with `use_history=True`. Registered providers can be selected with `provider_name=` or a `provider:model` global model value.

Install development dependencies and run tests with:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Copy `.env.example` to `.env` and provide credentials locally. Never commit `.env`.
