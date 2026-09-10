from __future__ import annotations

from pathlib import Path

import pytest

from memory import MemoryConfig, MemoryManager
from memory.rag import (
    Document,
    DocumentProcessor,
    EntityCandidate,
    ExtractionResult,
    GraphRAGPipeline,
    RelationCandidate,
    RAGPipeline,
)

from conftest import HashEmbedding


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(MemoryConfig(sqlite_path=":memory:"), embedding=HashEmbedding())


@pytest.fixture()
def pipeline(manager: MemoryManager) -> RAGPipeline:
    return RAGPipeline(manager)


def test_document_requires_non_empty_content_and_id():
    with pytest.raises(ValueError, match="content"):
        Document("   ")
    with pytest.raises(ValueError, match="id"):
        Document("text", id=" ")
    assert Document("hello").metadata == {}


def test_processor_chunks_with_overlap_and_shared_metadata():
    document = Document("a" * 250, id="doc-1", metadata={"source": "unit-test"})
    chunks = DocumentProcessor().chunks(document, chunk_size=100, overlap=20)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["document_id"] == "doc-1" for chunk in chunks)
    assert all(chunk.metadata["source"] == "unit-test" for chunk in chunks)
    assert chunks[0].content == "a" * 100
    assert chunks[1].id == "doc-1:1"
    # Overlapping windows repeat the tail of the previous chunk.
    assert chunks[1].content[:20] == chunks[0].content[80:]
    assert len(chunks[2].content) == 90


def test_processor_chunk_validation():
    processor = DocumentProcessor()
    document = Document("hello world")
    with pytest.raises(ValueError, match="chunk_size"):
        processor.chunks(document, chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        processor.chunks(document, chunk_size=10, overlap=10)


@pytest.mark.parametrize(
    ("filename", "raw", "expected_fragment"),
    [
        ("notes.jsonl", '{"a": 1}\n{"b": 2}\n', '"a": 1'),
        ("notes.json", '{"a": [1, 2]}', '"a"'),
        ("notes.csv", "name,age\nli,20\n", "li | 20"),
        ("notes.html", "<p>hello <b>world</b></p>", "hello"),
        ("notes.txt", "plain text", "plain text"),
    ],
)
def test_processor_parses_local_formats(tmp_path: Path, filename: str, raw: str, expected_fragment: str):
    path = tmp_path / filename
    path.write_text(raw, encoding="utf-8")

    document = DocumentProcessor().parse(path)

    assert expected_fragment in document.content
    assert document.metadata["filename"] == filename
    assert document.metadata["extension"] == path.suffix


def test_processor_parse_rejects_unknown_types():
    with pytest.raises(TypeError, match="source"):
        DocumentProcessor().parse(123)  # type: ignore[arg-type]


def test_pipeline_ingest_retrieve_and_context(pipeline: RAGPipeline):
    document = Document(
        "OpenSquilla is a mantis shrimp themed agent runtime. " * 30,
        id="doc-open",
        metadata={"source": "intro"},
    )
    items = pipeline.ingest(document, chunk_size=200, overlap=40)

    assert len(items) > 1
    assert all(item.memory_type.value == "semantic" for item in items)

    results = pipeline.retrieve("mantis shrimp runtime", limit=3)
    assert results and results[0].score > 0
    assert all(result.metadata["document_id"] == "doc-open" for result in results)

    context = pipeline.build_context("mantis shrimp runtime", limit=2)
    assert context
    assert context.count("\n\n") <= 1


class StaticExtractor:
    def extract(self, text: str, *, metadata=None) -> ExtractionResult:
        return ExtractionResult(
            domain="人工智能",
            topics=["检索"],
            entities=[
                EntityCandidate(name="Qdrant", entity_type="数据库", confidence=0.95),
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


def test_pipeline_auto_extracts_entities_and_graph_paths(manager: MemoryManager):
    pipeline = RAGPipeline(manager, extractor=StaticExtractor())
    items = pipeline.ingest(
        Document("Qdrant用于语义检索。", id="doc-qdrant", metadata={"filename": "qdrant.txt"}),
        chunk_size=100,
        overlap=0,
    )

    assert len(items) == 1
    assert pipeline.last_ingest_report["entities"] == 2
    assert pipeline.last_ingest_report["relations"] == 1
    facts = manager.semantic.facts()
    assert any(item.metadata.get("predicate") == "用于" for item in facts)

    result = GraphRAGPipeline(manager).retrieve("Qdrant", limit=3, hops=1)
    assert result.entities == ["Qdrant"]
    assert any(path.target == "语义检索" for path in result.paths)
    assert "Qdrant用于语义检索" in result.build_context()


def test_pipeline_reuses_relation_id_and_accumulates_evidence(manager: MemoryManager):
    pipeline = RAGPipeline(manager, extractor=StaticExtractor())
    for index in range(2):
        pipeline.ingest(
            Document(f"Qdrant用于语义检索。{index}", id=f"doc-{index}", metadata={"filename": f"{index}.txt"}),
            chunk_size=100,
            overlap=0,
        )
    relations = [
        item for item in manager.semantic.facts()
        if item.metadata.get("predicate") == "用于"
    ]
    assert len(relations) == 1
    assert len(relations[0].metadata["evidence_items"]) == 2


def test_pipeline_ingest_source_and_delete_document(pipeline: RAGPipeline, tmp_path: Path):
    path = tmp_path / "facts.txt"
    path.write_text("The memory package stores semantic facts.", encoding="utf-8")

    items = pipeline.ingest_source(str(path), chunk_size=50, overlap=10)
    assert items

    document_id = items[0].metadata["document_id"]
    removed = pipeline.delete_document(document_id)
    assert removed == len(items)
    assert pipeline.retrieve("semantic facts") == []


def test_pipeline_answer_requires_callable_generator(pipeline: RAGPipeline):
    with pytest.raises(TypeError, match="generator"):
        pipeline.answer("anything", "not-callable")  # type: ignore[arg-type]

    pipeline.ingest(Document("React answers use thoughts and actions.", id="doc-react"))
    answer = pipeline.answer("thoughts and actions", lambda prompt: f"answer-of:{len(prompt)}")
    assert answer.startswith("answer-of:")


def test_tool_default_sqlite_path_prefers_env(monkeypatch: pytest.MonkeyPatch):
    from memory import default_sqlite_path

    monkeypatch.setenv("MEMORY_DB_PATH", "custom/memory.sqlite3")
    assert default_sqlite_path() == "custom/memory.sqlite3"
    monkeypatch.delenv("MEMORY_DB_PATH")
    assert default_sqlite_path() == "memory.sqlite3"


def test_memory_tool_persists_across_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "tool-memory.sqlite3"))
    from memory import default_sqlite_path
    from tool.memory_tool import MemoryTool, MemoryToolInput

    def make_tool() -> MemoryTool:
        manager = MemoryManager(MemoryConfig(sqlite_path=default_sqlite_path()), embedding=HashEmbedding())
        return MemoryTool(manager=manager)

    first = make_tool()
    first.execute(MemoryToolInput(action="add", content="persistent fact", memory_type="semantic"))

    second = make_tool()
    output = second.execute(MemoryToolInput(action="search", query="persistent fact", memory_type="semantic"))

    assert output.count >= 1
    assert any("persistent fact" in item["content"] for item in output.items)


def test_rag_tool_ingest_and_retrieve_with_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "rag-memory.sqlite3"))
    from memory import default_sqlite_path
    from tool.rag_tool import RAGTool, RAGToolInput

    def make_tool() -> RAGTool:
        pipeline = RAGPipeline(MemoryManager(MemoryConfig(sqlite_path=default_sqlite_path()), embedding=HashEmbedding()))
        return RAGTool(pipeline=pipeline)

    tool = make_tool()
    ingest = tool.execute(RAGToolInput(action="ingest", text="The zebra lives in savannah. " * 20, chunk_size=100, overlap=20))
    assert ingest.count >= 2

    retrieved = tool.execute(RAGToolInput(action="retrieve", query="zebra savannah", limit=2))
    assert retrieved.count >= 1
    assert "zebra" in " ".join(item["content"] for item in retrieved.items)

    context = tool.execute(RAGToolInput(action="context", query="zebra savannah", limit=2))
    assert context.count == 1
    assert "zebra" in context.context
