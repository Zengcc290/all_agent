"""End-to-end integration test for the real runtime chain.

The old P0 filesystem/shell loop is gone, but two runtime behaviours were only
ever covered here, so they are re-bound to the surviving memory tools:

* a ``side_effect="write"`` tool without a confirmation key is refused with
  ``CONFIRMATION_REQUIRED`` and leaves no trace behind;
* ``depends_on`` inside one batch really orders the dependent read after the
  write it waits for.

Both go through the production chain (ToolCall -> permission gate ->
confirmation gate -> execute) against an isolated in-memory memory.
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from core import (
    ExecutionContext,
    ToolCall,
    ToolExecutionManager,
    ToolRegistry,
)
from memory import MemoryConfig, MemoryManager
from tool.memory_add import MemoryAddTool
from tool.memory_query import MemoryQueryTool

MEMORY_CONTENT = "batch dependency probe: qdrant holds vectors, neo4j holds relations"


class RuntimeLoop:
    """One isolated memory instance bound to the real runtime chain."""

    def __init__(self) -> None:
        self.manager = MemoryManager(
            MemoryConfig(sqlite_path=":memory:"), embedding=HashEmbedding()
        )
        self.registry = ToolRegistry()
        for tool in (
            MemoryAddTool(manager=self.manager),
            MemoryQueryTool(manager=self.manager),
        ):
            self.registry.register(tool)
        self.execution = ToolExecutionManager(self.registry)
        self._counter = 0

    def close(self) -> None:
        self.manager.close()

    def _call(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        depends_on: list[str] | None = None,
    ) -> ToolCall:
        tool, generation = self.registry.resolve(name)
        return ToolCall(
            call_id=call_id,
            tool_name=name,
            schema_version=tool.spec.version,
            schema_hash=tool.spec.schema_hash,
            registry_generation=generation,
            arguments=arguments,
            depends_on=depends_on or [],
        )

    def _permissions(self, names: list[str]) -> frozenset[str]:
        """Grant exactly what each spec declares, so the test isolates the gate
        under examination (confirmation / ordering) from the permission gate."""

        granted: set[str] = set()
        for name in names:
            tool, _ = self.registry.resolve(name)
            granted.update(tool.spec.permissions)
        return frozenset(granted)

    async def call(self, name: str, arguments: dict, *, confirm: bool = False):
        self._counter += 1
        context = ExecutionContext(
            permissions=self._permissions([name]),
            confirmed_side_effects=(
                frozenset({self.registry.confirmation_key(name)})
                if confirm
                else frozenset()
            ),
        )
        batch = await self.execution.execute_batch(
            [self._call(f"loop-{self._counter}", name, arguments)], context
        )
        return batch.results[0]

    async def batch(self, calls: list[tuple[str, str, dict, bool, list[str]]]):
        confirmed: set[str] = set()
        built = []
        for call_id, name, arguments, needs_confirmation, depends_on in calls:
            built.append(self._call(call_id, name, arguments, depends_on))
            if needs_confirmation:
                confirmed.add(self.registry.confirmation_key(name))
        context = ExecutionContext(
            permissions=self._permissions([name for _, name, *_ in calls]),
            confirmed_side_effects=frozenset(confirmed),
        )
        outcomes = await self.execution.execute_batch(built, context)
        return list(outcomes.results)


@pytest.fixture()
def loop():
    instance = RuntimeLoop()
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.asyncio
async def test_depends_on_read_after_write_in_one_batch(loop):
    results = await loop.batch(
        [
            (
                "add-memory",
                "memory.add",
                {
                    "content": MEMORY_CONTENT,
                    "memory_type": "episodic",
                    "importance": 0.9,
                },
                True,
                [],
            ),
            (
                "search-memory",
                "memory.query",
                {"action": "search", "query": MEMORY_CONTENT, "limit": 5},
                False,
                ["add-memory"],
            ),
        ]
    )

    assert results[0].ok, results[0].error
    assert results[1].ok, results[1].error
    # 读依赖写：同批内 search 必须已经能看到刚写入的内容。
    contents = [item["content"] for item in results[1].data["items"]]
    assert MEMORY_CONTENT in contents


@pytest.mark.asyncio
async def test_write_without_confirmation_is_rejected(loop):
    result = await loop.call(
        "memory.add",
        {"content": "must never be persisted", "memory_type": "episodic"},
        confirm=False,
    )

    assert not result.ok
    assert result.error is not None and result.error.code == "CONFIRMATION_REQUIRED"
    # 被拒的写不能留下任何痕迹。
    assert loop.manager.list(memory_type="episodic") == []
