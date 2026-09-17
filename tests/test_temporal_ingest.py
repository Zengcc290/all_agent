"""F4 逐句多元关系 + 时间/状态分类的最小回归集。

覆盖：
- 句级切分（chunks 表出现句级块，偏移精确、chunk_index 连续）；
- RelationCandidate 时间/状态字段透传到 facts 与 Neo4j 边；
- 过期时间段检索不命中；未来时间段也不命中；uncertain 降权不排除；
- active=False 语义不变；旧数据（无新字段）仍能检索（向后兼容）。
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from memory.rag import Document, DocumentProcessor, GraphRAGPipeline, RAGPipeline
from memory.rag.knowledge import ExtractionResult, RelationCandidate


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


def test_sentences_with_spans_split_on_punctuation():
    processor = DocumentProcessor()
    document = Document("小红 2024 年结婚。他们有一个孩子！名字叫小蓝？", id="doc-s")

    spans = processor.sentences_with_spans(document)

    assert [span.chunk.content for span in spans] == [
        "小红 2024 年结婚。",
        "他们有一个孩子！",
        "名字叫小蓝？",
    ]
    # 偏移指向 normalized_text，可直接重切
    text = processor.normalized_text(document)
    for span in spans:
        assert text[span.char_start : span.char_end] == span.chunk.content
    # chunk_index 连续、granularity 标记
    assert [span.chunk.metadata["chunk_index"] for span in spans] == [0, 1, 2]
    assert all(span.chunk.metadata["granularity"] == "sentences" for span in spans)


def test_sentences_tail_without_punctuation_is_kept():
    processor = DocumentProcessor()
    spans = processor.sentences_with_spans(Document("第一句。残句没有句读号", id="doc-t"))

    assert [span.chunk.content for span in spans] == ["第一句。", "残句没有句读号"]


def test_ingest_granularity_sentences(tmp_path):
    """句级导入：chunks 表出现句级块，偏移精确。"""

    from memory.storage.document_repo import DocumentRepository

    db = tmp_path / "memory.sqlite3"
    manager2 = MemoryManager(
        MemoryConfig(sqlite_path=str(db)),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )
    pipeline = RAGPipeline(manager2)
    pipeline.ingest(
        Document("Qdrant用于语义检索。Neo4j存关系。", id="doc-g"),
        granularity="sentences",
    )

    repository = DocumentRepository(str(db))
    chunks = repository.list_chunks("doc-g")
    assert [chunk.text for chunk in chunks] == ["Qdrant用于语义检索。", "Neo4j存关系。"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    # 偏移索引进 normalized_text，可重切出原句
    text = DocumentProcessor().normalized_text(Document("Qdrant用于语义检索。Neo4j存关系。", id="doc-g"))
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text
    repository.close()
    manager2.close()


def test_temporal_relation_lands_on_graph(manager: MemoryManager):
    """时间/状态字段透传：facts 与 Neo4j 边都带上 valid_from/event_at。"""

    manager.semantic.add_fact(
        "小红",
        "结婚",
        "小蓝",
        metadata={
            "confidence": 0.9,
            "valid_from": "2024-01-01T00:00:00+00:00",
            "status": "fact",
            "event_at": "2024-06-01T00:00:00+00:00",
        },
    )

    item = manager.semantic.facts("小红")[0]
    assert item.metadata["valid_from"] == "2024-01-01T00:00:00+00:00"
    assert item.metadata["status"] == "fact"
    assert item.metadata["event_at"] == "2024-06-01T00:00:00+00:00"

    edge = next(
        value
        for value in manager.semantic.graph_store.get_relations("小红")
        if value["relation"] == "结婚"
    )
    assert edge["properties"]["valid_from"] == "2024-01-01T00:00:00+00:00"
    assert edge["properties"]["event_at"] == "2024-06-01T00:00:00+00:00"


def test_expired_window_is_filtered_out(manager: MemoryManager):
    """用过期时间段检索时该边不再命中。"""

    from memory.rag import GraphRAGPipeline

    manager.semantic.add_fact(
        "A",
        "租用",
        "V1",
        metadata={"confidence": 0.9, "valid_to": "2020-01-01T00:00:00+00:00"},
    )
    manager.semantic.add_fact("A", "租用", "V2", confidence=0.9)

    paths = GraphRAGPipeline(manager)._expand(["A"], hops=1, path_limit=10)

    assert [path.target for path in paths] == ["V2"]


def test_future_window_is_filtered_out(manager: MemoryManager):
    manager.semantic.add_fact(
        "A",
        "部署",
        "V3",
        metadata={"confidence": 0.9, "valid_from": "2030-01-01T00:00:00+00:00"},
    )
    manager.semantic.add_fact("A", "部署", "V4", confidence=0.9)

    paths = GraphRAGPipeline(manager)._expand(["A"], hops=1, path_limit=10)

    assert [path.target for path in paths] == ["V4"]


def test_uncertain_status_is_downweighted_not_excluded(manager: MemoryManager):
    """status=uncertain 降权而非排除；expired 不命中。"""

    from memory.rag import GraphRAGPipeline

    manager.semantic.add_fact(
        "A", "支持", "X", metadata={"status": "uncertain"}, confidence=0.9
    )
    manager.semantic.add_fact(
        "A", "支持", "Y", metadata={"status": "expired"}, confidence=0.9
    )
    manager.semantic.add_fact("A", "支持", "Z", confidence=0.9)

    paths = GraphRAGPipeline(manager)._expand(["A"], hops=1, path_limit=10)

    targets = [path.target for path in paths]
    assert "Y" not in targets  # expired 不命中
    assert "X" in targets  # uncertain 降权但仍召回
    uncertain = next(path for path in paths if path.target == "X")
    certain = next(path for path in paths if path.target == "Z")
    assert uncertain.confidence == pytest.approx(certain.confidence * 0.5)


def test_legacy_items_without_new_fields_still_retrieve(manager: MemoryManager):
    """旧数据（无 valid_from/status）向后兼容：无界时间默认命中。"""

    from memory.rag import GraphRAGPipeline

    manager.semantic.add_fact("A", "属于", "B", confidence=0.9)

    paths = GraphRAGPipeline(manager)._expand(["A"], hops=1, path_limit=10)

    assert [path.target for path in paths] == ["B"]


def test_relation_candidate_schema_defaults():
    candidate = RelationCandidate(subject="A", predicate="属于", object="B")

    assert candidate.valid_from == ""
    assert candidate.valid_to == ""
    assert candidate.status == "fact"
    assert candidate.event_at == ""

    timed = RelationCandidate.model_validate(
        {
            "subject": "A",
            "predicate": "结婚",
            "object": "B",
            "valid_from": "2024-01-01T00:00:00+00:00",
            "valid_to": "",
            "status": "fact",
            "event_at": "2024-06-01T00:00:00+00:00",
        }
    )
    assert timed.valid_from == "2024-01-01T00:00:00+00:00"

    with pytest.raises(Exception):
        RelationCandidate.model_validate(
            {"subject": "A", "predicate": "属于", "object": "B", "status": "bad"}
        )


def test_extraction_result_with_temporal_relations_materializes(manager: MemoryManager):
    """抽取管道逐句多元关系 + 时间字段的端到端：一条抽取出多组三元组。"""

    from memory.rag.knowledge import materialize_extraction

    item = manager.semantic.add("小红 2024 年结婚。他们有一个孩子。")
    extraction = ExtractionResult.model_validate(
        {
            "domain": "人物",
            "entities": [
                {"name": "小红", "entity_type": "人物"},
                {"name": "小蓝", "entity_type": "人物"},
                {"name": "孩子", "entity_type": "人物"},
            ],
            "relations": [
                {
                    "subject": "小红",
                    "predicate": "结婚",
                    "object": "小蓝",
                    "action": "assert",
                    "cardinality": "multi",
                    "confidence": 0.9,
                    "evidence": "小红 2024 年结婚",
                    "valid_from": "2024-01-01T00:00:00+00:00",
                    "event_at": "2024-06-01T00:00:00+00:00",
                },
                {
                    "subject": "小红",
                    "predicate": "有",
                    "object": "孩子",
                    "action": "assert",
                    "cardinality": "multi",
                    "confidence": 0.85,
                    "evidence": "他们有一个孩子",
                },
            ],
        }
    )
    materialized = materialize_extraction(
        manager, extraction, source_item=item, source_metadata={}, resolver=None
    )

    assert materialized["relations"] == 2
    facts = manager.semantic.facts("小红")
    marriage = next(fact for fact in facts if fact.metadata["predicate"] == "结婚")
    assert marriage.metadata["valid_from"] == "2024-01-01T00:00:00+00:00"
    assert marriage.metadata["event_at"] == "2024-06-01T00:00:00+00:00"
    child = next(fact for fact in facts if fact.metadata["predicate"] == "有")
    assert child.metadata["status"] == "fact"
