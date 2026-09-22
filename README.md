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

Console output is deliberately narrow (`core/activity_log.py`). The activity
stream shows the parsed `Thought` of each ReAct round, the names of the tools
about to be called, the model round latency, and the tool-call latency; tool
arguments and return values are never printed, so a large tool result cannot
flood the terminal. Protocol problems are surfaced as warnings
(`工具调用格式问题`). Tool registration, discovery summaries, round starts and
final answers are intentionally silent, so a normal run prints only the round
lines above. The logger is `all_agent.activity` and prints through `print`
(one `[HH:MM:SS]` prefixed line per event); attach your own handler or set the
level if you need the rest.

Every model request also includes a system message listing all registered tool
names. In the explicit lazy-loading mode this can include repository-only tools
whose schemas have not yet been loaded.

## Tool execution

Tools declare their input and output Pydantic models in `ToolSpec`. Both models must use `ConfigDict(extra="forbid")`; this prevents unknown fields from being silently discarded. `recommended_before_tools` can list namespaced tools that are useful before this tool; it is advisory metadata included in the model-facing description and catalog responses, never an execution dependency or runtime gate. The runtime validates every call before execution, checks the registration generation and schema version/hash, applies side-effect confirmation rules, and returns one result per call. Permission declarations are optional compatibility metadata and are not enforced, so every caller can access every registered tool.

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

`ToolCatalogTool` is available as `system.tool_catalog` for summary search and versioned schema loading. It exposes fixed operations and does not execute arbitrary SQL. Tools are eagerly discovered, instantiated, and registered before the first model request by default. ReAct runs use catalog-first schema exposure by default: the first request contains every registered tool name and only the complete schema for `system.tool_catalog`; the model resolves a needed capability before invoking it. This is schema retrieval, not registration or permission checking. Pass `defer_tool_loading=False` to `run_with_react(...)` for a one-round full-schema ReAct prompt. Pass a `ToolSpecRepository` to persist the active catalog; `lazy_tools=True` remains a separate implementation-loading optimization.

When a tool declares `recommended_before_tools`, ReAct and native function-calling expose that list as a non-binding hint. The model may follow the recommendations when they improve accuracy, but the runtime never requires a preceding call and does not reject a direct call.

Project changes are recorded compactly in SQLite by the auto-discovered
`system.update_log` tool. Read [update_log_readme_first.md](update_log_readme_first.md)
before making changes. After every actual modification, provide the tool's
generation-bound side-effect confirmation and call it once. Permission fields
are retained only as compatibility/audit metadata; they do not grant or deny
access in this public-tool deployment. The guide
stores only the last/next numeric ID, while full structured
entries remain in `update_log.sqlite3`.

OpenAI function definitions are generated in strict mode. Malformed, unknown, or stale native calls are returned to the model as structured tool errors so it can correct the request without executing unvalidated input. Configured Agent prompts are applied automatically; conversation history is reused by default (pass `use_history=False` for a stateless request, and use `clear_history()` to reset it). History is isolated per provider profile. Requests use canonical tool ordering, compact schema JSON, and a deterministic `prompt_cache_key` derived only from the reusable prompt prefix. This keeps user text, tool results, and lazy-loaded schemas after the stable prefix so OpenAI-compatible gateways can reuse their KV/prefix cache. Pass `prompt_cache_retention="24h"` to retain the cache longer, or `enable_prompt_cache=False` for a gateway that does not implement the OpenAI cache fields. Copy [`config/provider.example.toml`](config/provider.example.toml) to the ignored `config/provider.toml`, then fill each profile's `api_key` and `api_url`. Select one request with `profile_name="deepseek"`, or change the default with `agent.set_active_profile("deepseek")`; call `agent.reload_provider_profiles()` after editing configuration. `provider_name=` remains a deprecated alias during migration.

Each profile currently uses the `openai_compatible` adapter. The `models` array is declarative, so registration does not depend on a provider exposing `/models`. `tool_mode` documents whether the endpoint supports strict native tools or should use the textual ReAct protocol. The real `provider.toml` contains plaintext keys and is intentionally excluded from Git.

Install development dependencies and run the quality gate with:

```bash
python -m pip install -e '.[dev]'
python scripts/check.py     # ruff + full pytest, stops at the first failure
```

`python -m pytest -q` and `python -m ruff check .` can still be run separately.
The same gate is wired as a local `pre-commit` hook in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml) (run `pre-commit install`
once); it uses the hooks' own environment, so the tools must be installed
there.

## Standalone memory system

The project now includes an opt-in four-layer memory package under
[`memory/`](memory/). It provides
`MemoryManager`, working/episodic/semantic/perceptual memories, SQLite document
persistence, Qdrant vector search, Neo4j graph relations, and two interchangeable
embedding services: `APIEmbedding` (any OpenAI-compatible `/embeddings` endpoint,
default qwen3-embedding-0.6b on DashScope) and the deterministic offline
`HashEmbedding` used automatically when no API key is configured. The built-in
memory tools (`memory.query`, `memory.add`, `memory.manage` and the RAG pair
`memory.rag_search` / `memory.rag`) expose the same APIs to the
current `Agent` tool runtime. See [`memory/README.md`](memory/README.md) for
usage and backend configuration.

Copy `.env.example` to `.env` and provide credentials locally. Never commit `.env`.

## 知识星云 Web application

The repository also ships an optional FastAPI application that exposes the
memory system as an HTTP API and serves a React single-page front end
("知识星云"): chat with the knowledge agent, upload documents for RAG chunking
plus LLM entity/relation extraction, add facts or single sentences, inspect the
graph, run hybrid graph+vector retrieval, and export/import the whole memory
store as JSON.

```powershell
# run from the project root
.venv\Scripts\python.exe -m web.app
# open http://127.0.0.1:8765
```

It starts without any API key: retrieval falls back to the offline
`HashEmbedding`, and `/api/chat` returns 503 with a configuration hint until a
chat provider is configured. The port comes from `constants.DEFAULT_WEB_PORT` (8765). See
[`web/README.md`](web/README.md) for the endpoint table, the data flow and the
front end layout (Vite sources in `web/frontend`, built into `web/static`).

The `config/` directory is resolved relative to the project root and is
therefore meant to run from a checkout (`python -m web.app`); it is not embedded
in the built wheel.
