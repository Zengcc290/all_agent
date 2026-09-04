# all-agent

This project provides a small, typed runtime for OpenAI-compatible agents and tools.

## Single-file tool plugins

`Agent` automatically discovers enabled Python modules directly inside the trusted
`tool` package. Copy `tool/tool_template.py`, implement its typed contract, and
set `TOOL_ENABLED = True`; no central registration list needs to be edited.
Discovery outcomes and registration state are queryable from the Agent. See the
[Chinese tool guide](tool/README.md) for the complete template rules and the
function-level LLM/tool call chain.

## Runtime logs

The runtime prints timestamped activity logs to the console. Each successful
tool registration includes the tool name and the complete current registry;
each ReAct round prints its number, model and tool durations, parsed Thought,
tool names being called, and the final answer. Raw tool arguments and return
values are omitted to keep the output readable.

Every model request also includes a system message listing all registered tool
names. In the explicit lazy-loading mode this can include repository-only tools
whose schemas have not yet been loaded.

## Tool execution

Tools declare their input and output Pydantic models in `ToolSpec`. Both models must use `ConfigDict(extra="forbid")`; this prevents unknown fields from being silently discarded. The runtime validates every call before execution, checks the registration generation and schema version/hash, applies side-effect confirmation rules, and returns one result per call. Permission declarations are optional compatibility metadata and are not enforced, so every caller can access every registered tool.

Independent calls can run concurrently. Calls with `depends_on` are executed in dependency layers. Tools marked `parallel_safe=False` are isolated into single-call layers.

```python
import asyncio

from core import ToolCall, ToolExecutionManager, ToolRegistry
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
    )
    print(result.model_dump())


asyncio.run(main())
```

`ToolCatalogTool` is available as `system.tool_catalog` for summary search and versioned schema loading. It exposes fixed operations and does not execute arbitrary SQL. Tools are eagerly discovered, instantiated, and registered before the first model request by default. Pass a `ToolSpecRepository` to persist the active catalog. Use `lazy_tools=True` on `ReActAgent` or `run_with_tools(..., defer_tool_loading=True)` only when the catalog-first/lazy-loading tradeoff is intentional.

Project changes are recorded compactly in SQLite by the auto-discovered
`system.update_log` tool. Read [update_log_readme_first.md](update_log_readme_first.md)
before making changes. After every actual modification, provide the tool's
generation-bound side-effect confirmation and call it once. Permission fields
are retained only as compatibility/audit metadata; they do not grant or deny
access in this public-tool deployment. The guide
stores only the last/next numeric ID, while full structured
entries remain in `update_log.sqlite3`.

OpenAI function definitions are generated in strict mode. Malformed, unknown, or stale native calls are returned to the model as structured tool errors so it can correct the request without executing unvalidated input. Configured Agent prompts are applied automatically; previous `history` can be included explicitly with `use_history=True` (history is isolated per provider profile). Copy [`config/provider.example.toml`](config/provider.example.toml) to the ignored `config/provider.toml`, then fill each profile's `api_key` and `api_url`. Select one request with `profile_name="deepseek"`, or change the default with `agent.set_active_profile("deepseek")`; call `agent.reload_provider_profiles()` after editing configuration. `provider_name=` remains a deprecated alias during migration.

Each profile currently uses the `openai_compatible` adapter. The `models` array is declarative, so registration does not depend on a provider exposing `/models`. `tool_mode` documents whether the endpoint supports strict native tools or should use the textual ReAct protocol. The real `provider.toml` contains plaintext keys and is intentionally excluded from Git.

Install development dependencies and run tests with:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Copy `.env.example` to `.env` and provide credentials locally. Never commit `.env`.
