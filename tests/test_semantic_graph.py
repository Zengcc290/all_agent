"""Phase 4: 图边一致性（F6）、实体属性、路径查询。

覆盖方案 P4：合并/撤回分支必须刷新图边属性（原先只有新建分支写边）、
实体 domain/aliases/importance 通过 add_relation 落到图上、
path_query 返回简单路径（内存回退与 Cypher 同形）。
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from memory.rag import Document, RAGPipeline
from memory.rag.knowledge import EntityCandidate, ExtractionResult, RelationCandidate


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


@pytest.fixture()
def graph(manager: MemoryManager) -> Neo4jGraphStore:
    return manager.semantic.graph_store


def edge(graph: Neo4jGraphStore, source: str, relation: str, target: str) -> dict:
    for value in graph.get_relations(source):
        if value["relation"] == relation and value["target"] == target:
            return value
    raise AssertionError(f"edge {source}-[{relation}]->{target} not found")


class GraphExtractor:
    """抽取「Qdrant 用于 语义检索」，用来驱动真实的实体/关系写入链。"""

    def extract(self, text: str, *, metadata=None) -> ExtractionResult:
        return ExtractionResult(
            domain="人工智能",
            entities=[
                EntityCandidate(name="Qdrant", entity_type="数据库", confidence=0.95, aliases=["向量库"]),
                EntityCandidate(name="语义检索", entity_type="能力", confidence=0.9),
            ],
            relations=[
                RelationCandidate(
                    subject="Qdrant",
                    predicate="用于",
                    object="语义检索",
                    confidence=0.92,
                    evidence="Qdrant用于语义检索",
                )
            ],
        )


def test_retract_refreshes_graph_edge(manager: MemoryManager, graph: Neo4jGraphStore):
    """F6：撤回走合并分支，原先不写图 → 边会一直停在旧属性。"""

    item = manager.semantic.add_fact("混合检索", "使用", "RRF", confidence=0.9)
    assert edge(graph, "混合检索", "使用", "RRF")["properties"]["memory_id"] == item.id

    manager.semantic.add_fact("混合检索", "使用", "RRF", metadata={"active": False})

    assert edge(graph, "混合检索", "使用", "RRF")["properties"]["active"] is False


def test_supersede_writes_edge_markers(manager: MemoryManager, graph: Neo4jGraphStore):
    manager.semantic.add_fact("当前版本", "是", "v1", confidence=0.9)

    manager.semantic.add_fact(
        "当前版本",
        "是",
        "v1",
        metadata={
            "active": False,
            "superseded_at": "2026-09-15T00:00:00+00:00",
            "superseded_by": ["relation:newer"],
        },
    )

    properties = edge(graph, "当前版本", "是", "v1")["properties"]
    assert properties["active"] is False
    assert properties["superseded_by"] == ["relation:newer"]
    assert properties["superseded_at"] == "2026-09-15T00:00:00+00:00"


def test_reactivation_clears_retired_markers_on_the_edge(
    manager: MemoryManager, graph: Neo4jGraphStore
):
    manager.semantic.add_fact("服务", "状态", "下线", metadata={"active": False})
    manager.semantic.add_fact("服务", "状态", "下线")

    properties = edge(graph, "服务", "状态", "下线")["properties"]
    assert properties["active"] is True
    assert properties["superseded_by"] == []


def test_path_query_returns_simple_paths(graph: Neo4jGraphStore):
    graph.add_relation("A", "knows", "B")
    graph.add_relation("B", "knows", "C")
    graph.add_relation("A", "knows", "D")
    graph.add_relation("D", "knows", "C")

    paths = graph.path_query("A", "C", max_depth=3)

    assert len(paths) == 2
    assert sorted(path["entities"] for path in paths) == [["A", "B", "C"], ["A", "D", "C"]]
    assert paths[0]["relations"][0]["source"] in {"A", "B"}  # 关系方向按真实边
    # 深度/条数受限；反向也能查到（无向遍历）
    assert graph.path_query("A", "C", max_depth=1) == []
    assert len(graph.path_query("A", "C", max_depth=3, limit=1)) == 1
    assert len(graph.path_query("C", "A", max_depth=3)) == 2
    assert graph.path_query("A", "不存在", max_depth=3) == []


def test_path_query_rejects_bad_arguments(graph: Neo4jGraphStore):
    with pytest.raises(ValueError, match="start and target"):
        graph.path_query(" ", "B")
    with pytest.raises(ValueError, match="max_depth"):
        graph.path_query("A", "B", max_depth=0)
    with pytest.raises(ValueError, match="limit"):
        graph.path_query("A", "B", limit=0)


def test_entity_attributes_land_on_the_graph(manager: MemoryManager, graph: Neo4jGraphStore):
    """实体属性由抽取管道写进 memories，边写入时再投影到图（7.1）。"""

    pipeline = RAGPipeline(manager, extractor=GraphExtractor())
    pipeline.ingest(Document("Qdrant用于语义检索。", id="doc-g"))

    qdrant = graph.entity("Qdrant")
    assert qdrant["domain"] == "人工智能"
    assert qdrant["importance"] == pytest.approx(0.95)
    assert qdrant["aliases"] == ["向量库"]
    assert graph.entity("语义检索")["importance"] == pytest.approx(0.9)


def test_add_relation_entity_clauses_on_create_vs_match(graph: Neo4jGraphStore):
    graph.add_relation("A", "r1", "B", source_domain="人工智能", source_aliases=["a1"], source_importance=0.9)
    assert graph.entity("A") == {"domain": "人工智能", "aliases": ["a1"], "importance": 0.9}

    # ON MATCH：空的 domain/importance 不覆盖已有值，别名非空时更新
    graph.add_relation("A", "r2", "C", source_aliases=["a2"])
    assert graph.entity("A") == {"domain": "人工智能", "aliases": ["a2"], "importance": 0.9}

    graph.add_relation("A", "r3", "D", source_aliases=[])
    assert graph.entity("A")["aliases"] == ["a2"]
    assert graph.entity("C") == {"domain": "", "aliases": [], "importance": 0.5}
    assert graph.entity("未见过的实体") == {}
