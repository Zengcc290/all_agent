"""F1 边强化（"回忆即强化"）的最小回归集。

覆盖：
- 检索命中会真正给路径边 +1 次回忆、权重增长（有上限）；
- 幂等重写（抽取重复抽取同一三元组）不把计数清零；
- 强化总开关关闭后检索完全无副作用、排序退回纯 confidence；
- path_query 按路径总权重排序，强化过的路径排前面。
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from memory.rag import GraphRAGPipeline


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


def edge(manager: MemoryManager, source: str, relation: str, target: str) -> dict:
    for value in manager.semantic.graph_store.get_relations(source):
        if value["relation"] == relation and value["target"] == target:
            return value
    raise AssertionError(f"edge {source}-[{relation}]->{target} not found")


def test_recall_reinforces_adopted_edges(manager: MemoryManager):
    """同一条边被检索命中后，recall_count/weight 增长，且每次检索只 +1。"""

    manager.semantic.add_fact("A", "knows", "B", confidence=0.9)

    pipeline = GraphRAGPipeline(manager)
    pipeline.retrieve("A", limit=5, hops=1)
    first = edge(manager, "A", "knows", "B")["properties"]
    assert first["recall_count"] == 1
    assert first["weight"] == pytest.approx(1.5)
    assert first["last_accessed_at"]

    pipeline.retrieve("A", limit=5, hops=1)
    second = edge(manager, "A", "knows", "B")["properties"]
    assert second["recall_count"] == 2
    assert second["weight"] == pytest.approx(2.25)


def test_weight_is_capped(manager: MemoryManager):
    """权重有上限，不会"富者越富"到无穷大。"""

    manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    pipeline = GraphRAGPipeline(manager)
    for _ in range(10):
        pipeline.retrieve("A", limit=5, hops=1)

    from constants import MEMORY_EDGE_WEIGHT_MAX

    assert edge(manager, "A", "knows", "B")["properties"]["weight"] == pytest.approx(
        MEMORY_EDGE_WEIGHT_MAX
    )


def test_idempotent_ingest_does_not_reset_counters(manager: MemoryManager):
    """常规写入（合并分支）不碰计数/权重：只有回忆才强化。"""

    manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    GraphRAGPipeline(manager).retrieve("A", limit=5, hops=1)
    before = edge(manager, "A", "knows", "B")["properties"]

    manager.semantic.add_fact("A", "knows", "B", confidence=0.95)

    after = edge(manager, "A", "knows", "B")["properties"]
    assert after["recall_count"] == before["recall_count"] == 1
    assert after["weight"] == before["weight"]


def test_reinforcement_can_be_disabled(manager: MemoryManager, monkeypatch):
    """总开关关闭：检索零副作用、排序退回纯 confidence（回归基线）。"""

    monkeypatch.setenv("HELLOAGENTS_MEMORY_EDGE_REINFORCE", "0")
    manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    GraphRAGPipeline(manager).retrieve("A", limit=5, hops=1)

    properties = edge(manager, "A", "knows", "B")["properties"]
    assert properties["recall_count"] == 0
    assert properties["weight"] == 1.0
    assert properties["last_accessed_at"] == ""


def test_bump_ignores_missing_edges(manager: MemoryManager):
    """强化不凭空造边：bump 一条不存在的边是空操作。"""

    assert manager.semantic.graph_store.add_relation("X", "knows", "Y", bump=True) is None
    assert manager.semantic.graph_store.get_relations("X") == []


def test_repeated_recall_moves_path_ahead(manager: MemoryManager):
    """验收：被反复命中的路径在结果里位置前移。"""

    # 两条同置信度路径：A->B->C 与 A->D->C；先强化 A->B->C。
    manager.semantic.add_fact("A", "knows", "B", confidence=0.8)
    manager.semantic.add_fact("B", "knows", "C", confidence=0.8)
    manager.semantic.add_fact("A", "knows", "D", confidence=0.8)
    manager.semantic.add_fact("D", "knows", "C", confidence=0.8)
    pipeline = GraphRAGPipeline(manager)
    warmup = pipeline.retrieve("A", limit=5, hops=2)
    assert warmup.paths

    # 反复强化 A->B->C 这条路径（模拟"用户反复问 A"）。
    for _ in range(3):
        manager.semantic.graph_store.add_relation("A", "knows", "B", bump=True)
        manager.semantic.graph_store.add_relation("B", "knows", "C", bump=True)

    result = GraphRAGPipeline(manager).retrieve("A", limit=5, hops=2)
    assert result.paths
    two_hop = [path for path in result.paths if len(path.entities) == 3]
    assert two_hop[0].entities == ("A", "B", "C")


def test_disabled_recall_keeps_original_ordering(manager: MemoryManager, monkeypatch):
    """关掉强化后同图不重排：effective 退化为 confidence。"""

    monkeypatch.setenv("HELLOAGENTS_MEMORY_EDGE_REINFORCE", "0")
    manager.semantic.add_fact("A", "knows", "B", confidence=0.8)
    manager.semantic.add_fact("B", "knows", "C", confidence=0.8)
    manager.semantic.add_fact("A", "knows", "D", confidence=0.8)
    manager.semantic.add_fact("D", "knows", "C", confidence=0.8)

    paths = GraphRAGPipeline(manager)._expand(["A"], hops=2, path_limit=5)

    # 无强化时全部权重为 1.0：effective == confidence，排序与旧行为一致。
    assert paths
    assert all(path.effective == pytest.approx(path.confidence) for path in paths)


def test_path_query_orders_by_weight(manager: MemoryManager):
    """path_query 返回边权重，强路径在前。"""

    graph = manager.semantic.graph_store
    graph.add_relation("A", "knows", "B")
    graph.add_relation("B", "knows", "C")
    graph.add_relation("A", "knows", "D")
    graph.add_relation("D", "knows", "C")
    graph.add_relation("A", "knows", "B", bump=True)
    graph.add_relation("A", "knows", "B", bump=True)
    graph.add_relation("B", "knows", "C", bump=True)

    paths = graph.path_query("A", "C", max_depth=3)

    assert paths[0]["entities"] == ["A", "B", "C"]
    assert all("weight" in step for path in paths for step in path["relations"])
