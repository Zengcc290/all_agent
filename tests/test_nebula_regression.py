"""Regression: 全局同实体 + 时序观察保留 + 原句作为独立行星连接所有实体。"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from memory.rag import Document, EntityCandidate, ExtractionResult, RAGPipeline, RelationCandidate
from web.graph_builder import build_graph


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


class TimedCatExtractor:
    """13:00 小猫在洗澡；14:00 小猫在睡觉 -> 同一实体「小猫」，两条时序观察都保留。"""

    def __init__(self, verb: str, event_at: str) -> None:
        self._verb = verb
        self._event_at = event_at

    def extract(self, text: str, *, metadata=None, graph_context=""):
        return ExtractionResult(
            domain="生活",
            entities=[EntityCandidate(name="小猫", entity_type="动物")],
            relations=[
                RelationCandidate(
                    subject="小猫",
                    predicate=self._verb,
                    object="状态",
                    cardinality="temporal",
                    event_at=self._event_at,
                    action="assert",
                    confidence=0.95,
                    evidence=text,
                )
            ],
        )


def test_same_entity_keeps_both_timed_observations(manager: MemoryManager) -> None:
    pipeline = RAGPipeline(manager, extractor=TimedCatExtractor("在洗澡", "2026-09-18T13:00:00+08:00"))
    pipeline.ingest(Document("13:00 小猫在洗澡。", id="cat-13", metadata={"filename": "note.txt", "event_at": "2026-09-18T13:00:00+08:00"}))

    pipeline.extractor = TimedCatExtractor("在睡觉", "2026-09-18T14:00:00+08:00")
    pipeline.ingest(Document("14:00 小猫在睡觉。", id="cat-14", metadata={"filename": "note.txt", "event_at": "2026-09-18T14:00:00+08:00"}))

    entities = [item for item in manager.semantic.list() if item.metadata.get("kind") == "entity"]
    cats = [item for item in entities if item.metadata.get("canonical_name") == "小猫"]
    assert len(cats) == 1, "全局同一实体只能有一个"

    facts = manager.semantic.facts("小猫")
    preds = sorted(item.metadata.get("predicate") for item in facts)
    assert preds == ["在洗澡", "在睡觉"], "两条时序观察都保留"
    assert all(item.metadata.get("active", True) is not False for item in facts)

    graph = build_graph(manager)
    cat_nodes = [n for n in graph["nodes"] if n["kind"] == "entity" and n["title"] == "小猫"]
    assert len(cat_nodes) == 1
    obs_edges = [e for e in graph["edges"] if e["relation"] in ("在洗澡", "在睡觉")]
    assert len(obs_edges) == 2  # 两条有向观察边


class MultiRelationExtractor:
    """一句话抽出全部实体和多条具体关系。"""

    def extract(self, text: str, *, metadata=None, graph_context=""):
        return ExtractionResult(
            domain="生活",
            entities=[
                EntityCandidate(name="小猫", entity_type="动物"),
                EntityCandidate(name="小狗", entity_type="动物"),
                EntityCandidate(name="浴缸", entity_type="物品"),
            ],
            relations=[
                RelationCandidate(subject="小猫", predicate="在洗澡", object="浴缸", confidence=0.95, evidence=text),
                RelationCandidate(subject="小狗", predicate="守着", object="浴缸", confidence=0.9, evidence=text),
            ],
        )


class OrphanEntityExtractor:
    """原句提取出一个还没有任何关系的孤立实体，它也必须被原句行星连接。"""

    def extract(self, text: str, *, metadata=None, graph_context=""):
        return ExtractionResult(
            domain="生活",
            entities=[EntityCandidate(name="地板", entity_type="物品")],
            relations=[],
        )


def test_orphan_entity_still_linked_from_source_planet(manager: MemoryManager) -> None:
    """孤立实体（无关系边）不能在图上变成没有入辙的悬浮节点：原句行星必须用「提及」边连到它。"""

    pipeline = RAGPipeline(manager, extractor=OrphanEntityExtractor())
    pipeline.ingest(Document("地板上有一层水。", id="src-orphan", metadata={"filename": "floor.txt"}))

    graph = build_graph(manager)
    doc = next(n for n in graph["nodes"] if n["kind"] == "chunk" and n["title"].startswith("floor.txt"))
    mentions = [e for e in graph["edges"] if e["source"] == doc["id"] and e["relation"] == "提及"]
    by_id = {n["id"]: n["title"] for n in graph["nodes"]}
    assert {by_id[e["target"]] for e in mentions} == {"地板"}
    assert len(mentions) == 1

def test_source_sentence_planet_connects_all_entities(manager: MemoryManager) -> None:
    pipeline = RAGPipeline(manager, extractor=MultiRelationExtractor())
    pipeline.ingest(Document("小猫在浴缸洗澡，小狗在旁边守着。", id="src-1", metadata={"filename": "note.txt"}))

    graph = build_graph(manager)
    doc = next(n for n in graph["nodes"] if n["kind"] == "chunk" and n["title"].startswith("note.txt"))
    assert doc["kind"] == "chunk"
    assert doc["parent"].startswith("dom:")

    # 原句行星 -> 所有提取实体 都有「提及」边
    mentions = [e for e in graph["edges"] if e["source"] == doc["id"] and e["relation"] == "提及"]
    by_id = {n["id"]: n["title"] for n in graph["nodes"]}
    assert {by_id[e["target"]] for e in mentions} == {"小猫", "小狗", "浴缸"}
    assert len(mentions) == 3

    # 抽取的具体关系是有向边，而非关系节点
    assert not any(n["kind"] == "relation" for n in graph["nodes"])
    washing = [e for e in graph["edges"] if e["relation"] == "在洗澡"]
    guarding = [e for e in graph["edges"] if e["relation"] == "守着"]
    assert len(washing) == 1 and len(guarding) == 1
def test_every_entity_and_chunk_links_to_its_domain_star(manager: MemoryManager) -> None:
    """恒星 ↔ 行星连线：每个实体/原句行星都有一条指向所属领域恒星的「属于」有向边。"""

    pipeline = RAGPipeline(manager, extractor=MultiRelationExtractor())
    pipeline.ingest(Document("小猫在浴缸洗澡，小狗在旁边守着。", id="src-1", metadata={"filename": "note.txt"}))

    graph = build_graph(manager)
    belong = [e for e in graph["edges"] if e["relation"] == "属于"]
    assert len(belong) > 0
    by_id = {n["id"]: n["title"] for n in graph["nodes"]}

    for node in graph["nodes"]:
        if node["kind"] not in {"entity", "chunk"}:
            continue
        star_id = f"dom:{node['domain']}"
        assert star_id in by_id, f"行星 {node['id']} 的领域恒星缺失"
        matches = [
            e for e in belong if e["source"] == star_id and e["target"] == node["id"]
        ]
        assert len(matches) == 1, f"{node['id']} 缺少唯一的「属于」恒星边"
        assert matches[0]["structural"] is True
