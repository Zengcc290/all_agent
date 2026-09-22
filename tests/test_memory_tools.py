"""记忆工具的读/写副作用契约。

回归 P0：``memory.manage``/``memory.rag`` 曾经把只读动作（search/get/list、
retrieve/context/graph_*）也标成 ``side_effect="write"``，运行时因此要求写确认，
而 Web/Agent 层从不提供确认，导致「先检索知识库再回答」实际失效。工具按读/写
拆分后：只读工具无需确认即可执行，写工具仍然必须拿到确认钥匙。
"""

from __future__ import annotations

import pytest

from core import (
    ExecutionContext,
    ToolCall,
    ToolExecutionManager,
    ToolRegistry,
)
from memory.rag import NullKnowledgeExtractor, RAGPipeline
from tool.memory_add import MemoryAddTool
from tool.memory_query import MemoryQueryTool
from tool.memory_tool import MemoryManageTool
from tool.rag_search import RAGSearchTool
from tool.rag_tool import RAGTool


@pytest.fixture()
def runtime(manager):
    """五个记忆工具共用一个测试 manager，绑定真实运行时链。"""

    pipeline = RAGPipeline(manager, extractor=NullKnowledgeExtractor())
    registry = ToolRegistry()
    for tool in (
        MemoryQueryTool(manager=manager),
        MemoryAddTool(manager=manager),
        MemoryManageTool(manager=manager),
        RAGSearchTool(pipeline=pipeline),
        RAGTool(pipeline=pipeline),
    ):
        registry.register(tool)
    return registry, ToolExecutionManager(registry)


async def _call(runtime, name: str, arguments: dict, *, confirm: bool):
    registry, executor = runtime
    tool, generation = registry.resolve(name)
    confirmation = (
        registry.call_confirmation_key(name, arguments)
        if tool.spec.side_effect == "destructive"
        else registry.confirmation_key(name)
    )
    context = ExecutionContext(
        confirmed_side_effects=(frozenset({confirmation}) if confirm else frozenset())
    )
    batch = await executor.execute_batch(
        [
            ToolCall(
                call_id=f"{name}-1",
                tool_name=name,
                schema_version=tool.spec.version,
                schema_hash=tool.spec.schema_hash,
                registry_generation=generation,
                arguments=arguments,
            )
        ],
        context,
    )
    return batch.results[0]


@pytest.mark.asyncio
async def test_read_memory_tools_execute_without_confirmation(runtime, manager) -> None:
    """只读检索不得要求写确认——这正是原 P0 缺陷的核心。"""

    # semantic 层是 RAG 检索层的目标；episodic 层用于验证跨层检索与 list。
    manager.semantic.add("星云 项目 计划 周三 交付 第一版")
    manager.episodic.record("问：星云计划什么时候交付\n答：周三")

    search = await _call(
        runtime, "memory.query", {"action": "search", "query": "星云 项目 计划"}, confirm=False
    )
    assert search.ok is True
    assert search.data["count"] >= 1

    retrieve = await _call(
        runtime,
        "memory.rag_search",
        {"action": "retrieve", "query": "星云 项目 计划"},
        confirm=False,
    )
    assert retrieve.ok is True
    assert retrieve.data["count"] >= 1

    listing = await _call(
        runtime, "memory.query", {"action": "list", "memory_type": "episodic"}, confirm=False
    )
    assert listing.ok is True
    assert listing.data["count"] >= 1


@pytest.mark.asyncio
async def test_memory_add_needs_confirmation_then_succeeds(runtime) -> None:
    """增量写入仍需确认；拿到确认钥匙后必须真正落库。"""

    denied = await _call(
        runtime,
        "memory.add",
        {"content": "用户偏好：浅色主题", "memory_type": "episodic"},
        confirm=False,
    )
    assert denied.ok is False
    assert denied.error is not None
    assert denied.error.code == "CONFIRMATION_REQUIRED"

    allowed = await _call(
        runtime,
        "memory.add",
        {"content": "用户偏好：浅色主题", "memory_type": "episodic"},
        confirm=True,
    )
    assert allowed.ok is True
    assert allowed.data["count"] == 1


@pytest.mark.asyncio
async def test_destructive_and_ingest_writes_need_confirmation(runtime) -> None:
    """删除/清空/入库属于破坏性或外部写入，绝不自动放行。"""

    for name, arguments in (
        ("memory.manage", {"action": "clear", "memory_type": "working"}),
        ("memory.rag", {"action": "ingest", "text": "一段待入库的文本"}),
    ):
        denied = await _call(runtime, name, arguments, confirm=False)
        assert denied.ok is False, name
        assert denied.error is not None
        assert denied.error.code == "CONFIRMATION_REQUIRED", name


@pytest.mark.asyncio
async def test_confirmed_memory_delete_is_the_single_destructive_path(runtime, manager) -> None:
    item = manager.semantic.add_fact("用户", "明确删除", "旧记忆", confidence=0.9)
    assert manager.semantic.graph_store.get_relations("用户")
    arguments = {"action": "delete", "memory_type": "semantic", "item_id": item.id}

    denied = await _call(runtime, "memory.manage", arguments, confirm=False)
    assert denied.error is not None
    assert denied.error.code == "CONFIRMATION_REQUIRED"
    assert manager.get(item.id) is not None

    allowed = await _call(runtime, "memory.manage", arguments, confirm=True)
    assert allowed.ok is True
    assert allowed.data["count"] == 1
    assert manager.get(item.id) is None
    assert manager.semantic.graph_store.get_relations("用户") == []


@pytest.mark.asyncio
async def test_destructive_confirmation_is_bound_to_exact_arguments(runtime, manager) -> None:
    first = manager.semantic.add_fact("甲", "关联", "乙", confidence=0.9)
    second = manager.semantic.add_fact("丙", "关联", "丁", confidence=0.9)
    registry, executor = runtime
    tool, generation = registry.resolve("memory.manage")
    approved = {"action": "delete", "memory_type": "semantic", "item_id": first.id}
    context = ExecutionContext(
        confirmed_side_effects=frozenset(
            {registry.call_confirmation_key("memory.manage", approved)}
        )
    )
    substitutions = (
        {"action": "delete", "memory_type": "semantic", "item_id": second.id},
        {"action": "clear", "memory_type": "semantic"},
    )
    for index, substituted in enumerate(substitutions):
        call = ToolCall(
            call_id=f"substituted-{index}",
            tool_name="memory.manage",
            schema_version=tool.spec.version,
            schema_hash=tool.spec.schema_hash,
            registry_generation=generation,
            arguments=substituted,
        )
        result = (await executor.execute_batch([call], context)).results[0]
        assert result.error is not None
        assert result.error.code == "CONFIRMATION_REQUIRED"

    assert manager.get(first.id) is not None
    assert manager.get(second.id) is not None


@pytest.mark.asyncio
async def test_semantic_clear_removes_graph_relations(runtime, manager) -> None:
    manager.semantic.add_fact("清空主体", "关联", "清空客体", confidence=0.9)
    assert manager.semantic.graph_store.get_relations("清空主体")
    assert manager.semantic.graph_store.graph_snapshot()["entities"]

    result = await _call(
        runtime,
        "memory.manage",
        {"action": "clear", "memory_type": "semantic"},
        confirm=True,
    )

    assert result.ok is True
    assert result.data["count"] >= 1
    assert manager.semantic.graph_store.get_relations("清空主体") == []
    snapshot = manager.semantic.graph_store.graph_snapshot()
    assert snapshot["entities"] == []
    assert snapshot["observations"] == []
    assert snapshot["relations"] == []


@pytest.mark.asyncio
async def test_rag_ingest_rejects_source_outside_workspace(
    runtime, tmp_path, monkeypatch
) -> None:
    """``memory.rag`` 的 source 是模型可控输入，必须经工作区沙箱解析。"""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))
    outside = tmp_path / "secret.txt"
    outside.write_text("SECRET-OUTSIDE-WORKSPACE", encoding="utf-8")

    denied = await _call(
        runtime,
        "memory.rag",
        {"action": "ingest", "source": str(outside)},
        confirm=True,
    )
    assert denied.ok is False
    assert denied.error is not None
    # 运行时按设计不把内部异常细节回传给模型，只报告执行失败。
    assert denied.error.code == "EXECUTION_ERROR"
    # 越界文件内容绝不能进入记忆库。
    assert not [
        item
        for item in runtime[0].resolve("memory.query")[0].manager.list(
            memory_type="semantic", include_expired=True
        )
        if "SECRET-OUTSIDE-WORKSPACE" in item.content
    ]


def test_read_and_write_tool_names_are_distinct() -> None:
    """读/写拆分后的工具名与副作用标注必须互相对应。"""

    read_tools = (
        MemoryQueryTool().spec,
        RAGSearchTool().spec,
    )
    write_tools = (
        MemoryAddTool().spec,
        MemoryManageTool().spec,
        RAGTool().spec,
    )
    assert {spec.name for spec in read_tools} == {"memory.query", "memory.rag_search"}
    assert all(spec.side_effect == "read" for spec in read_tools)
    assert {spec.name for spec in write_tools} == {
        "memory.add",
        "memory.manage",
        "memory.rag",
    }
    assert MemoryManageTool().spec.side_effect == "destructive"
    assert {spec.side_effect for spec in write_tools} == {"write", "destructive"}
