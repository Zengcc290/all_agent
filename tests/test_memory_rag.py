from __future__ import annotations

from pathlib import Path

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager
from memory.rag import (
    Document,
    DocumentProcessor,
    EntityCandidate,
    ExtractionResult,
    GraphRAGPipeline,
    RAGPipeline,
    RelationCandidate,
)


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"), embedding=HashEmbedding()
    )


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
    processor = DocumentProcessor()
    spans = processor.chunks_with_spans(document, chunk_size=100, overlap=20)
    chunks = [span.chunk for span in spans]

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["document_id"] == "doc-1" for chunk in chunks)
    assert all(chunk.metadata["source"] == "unit-test" for chunk in chunks)
    assert chunks[0].content == "a" * 100
    assert chunks[1].id == "doc-1:1"
    # Overlapping windows repeat the tail of the previous chunk.
    assert chunks[1].content[:20] == chunks[0].content[80:]
    assert len(chunks[2].content) == 90
    # 偏移必须落在规范化文本上：这是 documents/chunks 真值源的 char_start/char_end。
    text = processor.normalized_text(document)
    assert [text[span.char_start:span.char_end] for span in spans] == [
        chunk.content for chunk in chunks
    ]


def test_processor_chunk_validation():
    processor = DocumentProcessor()
    document = Document("hello world")
    with pytest.raises(ValueError, match="chunk_size"):
        processor.chunks_with_spans(document, chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        processor.chunks_with_spans(document, chunk_size=10, overlap=10)


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
def test_processor_parses_local_formats(
    tmp_path: Path, filename: str, raw: str, expected_fragment: str
):
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
        Document(
            "Qdrant用于语义检索。", id="doc-qdrant", metadata={"filename": "qdrant.txt"}
        ),
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


def test_graph_retrieve_degrades_when_the_graph_store_fails(
    manager: MemoryManager, monkeypatch: pytest.MonkeyPatch
):
    pipeline = RAGPipeline(manager, extractor=StaticExtractor())
    pipeline.ingest(
        Document("Qdrant用于语义检索。", id="doc-qdrant", metadata={"filename": "qdrant.txt"}),
        chunk_size=100,
        overlap=0,
    )

    def boom(self, seeds, *, hops, path_limit):
        raise RuntimeError("Aura routing table unavailable")

    monkeypatch.setattr(GraphRAGPipeline, "_expand", boom)
    result = GraphRAGPipeline(manager).retrieve("Qdrant", limit=3, hops=1)

    assert result.evidence
    assert result.paths == []
    assert result.entities == []


def test_pipeline_reuses_relation_id_and_accumulates_evidence(manager: MemoryManager):
    pipeline = RAGPipeline(manager, extractor=StaticExtractor())
    for index in range(2):
        pipeline.ingest(
            Document(
                f"Qdrant用于语义检索。{index}",
                id=f"doc-{index}",
                metadata={"filename": f"{index}.txt"},
            ),
            chunk_size=100,
            overlap=0,
        )
    relations = [
        item
        for item in manager.semantic.facts()
        if item.metadata.get("predicate") == "用于"
    ]
    assert len(relations) == 1
    assert len(relations[0].metadata["evidence_items"]) == 2


def test_graph_rag_does_not_return_cycles(manager: MemoryManager):
    manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    manager.semantic.add_fact("B", "knows", "A", confidence=0.8)

    paths = GraphRAGPipeline(manager)._expand(["A"], hops=2, path_limit=20)

    assert paths
    assert all(len(set(path.entities)) == len(path.entities) for path in paths)


def test_add_fact_without_item_id_is_idempotent(manager: MemoryManager):
    manager.semantic.add_fact("A", "knows", "B", confidence=0.8)
    manager.semantic.add_fact("A", "knows", "B", confidence=0.9)

    facts = manager.semantic.facts("A")
    assert len(facts) == 1
    assert facts[0].importance == 0.9


def test_pipeline_ingest_source_and_delete_document(
    pipeline: RAGPipeline, tmp_path: Path
):
    path = tmp_path / "facts.txt"
    path.write_text("The memory package stores semantic facts.", encoding="utf-8")

    items = pipeline.ingest_source(str(path), chunk_size=50, overlap=10)
    assert items

    document_id = items[0].metadata["document_id"]
    removed = pipeline.delete_document(document_id)
    assert removed == len(items)
    assert pipeline.retrieve("semantic facts") == []


def test_parse_treats_strings_as_text_and_paths_as_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """回归：``parse`` 的 str 必须一律当作文本，不得探测文件系统。

    历史缺陷：``parse`` 对任何不含换行的字符串做 ``Path(source).exists()``
    探测并 ``read_bytes()``，于是（1）与现存文件同名的短文本会被当文件读取，
    （2）模型通过 ``memory.rag`` 传来的绝对路径能读到工作区外的任意文件，
    绕开 ``fs.*`` 工具的沙箱。
    """
    from memory.rag import DocumentProcessor

    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-OUTSIDE-WORKSPACE", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    processor = DocumentProcessor()

    # 同名文件存在时，字符串仍然是文本。
    assert processor.parse("secret.txt").content == "secret.txt"
    # 绝对路径字符串同样是文本，绝不触发读取。
    assert processor.parse(str(secret)).content == str(secret)
    # 显式 Path 才表示文件。
    assert processor.parse(secret).content == "SECRET-OUTSIDE-WORKSPACE"
    # 非 str/Path/bytes/stream 仍然报类型错误。
    with pytest.raises(TypeError):
        processor.parse(123)  # type: ignore[arg-type]


def test_ingest_source_enforces_base_dir_containment(
    pipeline: RAGPipeline, tmp_path: Path
):
    """``ingest_source(base_dir=...)`` 必须拒绝越界路径。

    这是模型驱动调用（``memory.rag`` 的 source）的兜底边界：即使调用方忘了
    预先解析路径，越界也会在管道层被拒。
    """
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "note.txt"
    inside.write_text("The zebra lives in savannah.", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET-OUTSIDE-WORKSPACE", encoding="utf-8")

    items = pipeline.ingest_source(inside, base_dir=allowed, chunk_size=50, overlap=10)
    assert items

    with pytest.raises(ValueError, match="outside"):
        pipeline.ingest_source(outside, base_dir=allowed, chunk_size=50, overlap=10)

    # 字符串路径依旧被接受（内部转 Path），保持既有调用方兼容。
    assert pipeline.ingest_source(str(inside), base_dir=allowed, chunk_size=50, overlap=10)


def test_pipeline_answer_requires_callable_generator(pipeline: RAGPipeline):
    with pytest.raises(TypeError, match="generator"):
        pipeline.answer("anything", "not-callable")  # type: ignore[arg-type]

    pipeline.ingest(Document("React answers use thoughts and actions.", id="doc-react"))
    answer = pipeline.answer(
        "thoughts and actions", lambda prompt: f"answer-of:{len(prompt)}"
    )
    assert answer.startswith("answer-of:")


def test_tool_default_sqlite_path_prefers_env(monkeypatch: pytest.MonkeyPatch):
    from pathlib import Path as _Path

    from memory import default_sqlite_path

    monkeypatch.setenv("MEMORY_DB_PATH", "custom/memory.sqlite3")
    assert default_sqlite_path() == "custom/memory.sqlite3"
    monkeypatch.delenv("MEMORY_DB_PATH")
    # 未设环境变量：回落到仓库根目录的 memory.sqlite3（绝对路径，单点收拢）
    fallback = default_sqlite_path()
    assert _Path(fallback).is_absolute()
    assert _Path(fallback).name == "memory.sqlite3"
    assert _Path(fallback).parent == _Path(__file__).resolve().parent.parent


def test_memory_tool_persists_across_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "tool-memory.sqlite3"))
    from memory import default_sqlite_path
    from tool.memory_add import MemoryAddInput, MemoryAddTool
    from tool.memory_query import MemoryQueryInput, MemoryQueryTool

    def make_manager() -> MemoryManager:
        return MemoryManager(
            MemoryConfig(sqlite_path=default_sqlite_path()), embedding=HashEmbedding()
        )

    first = MemoryAddTool(manager=make_manager())
    first.execute(MemoryAddInput(content="persistent fact", memory_type="semantic"))

    # 第二个 manager 在写入之后构造，必须从 SQLite 恢复出可检索的向量。
    second = MemoryQueryTool(manager=make_manager())
    output = second.execute(
        MemoryQueryInput(action="search", query="persistent fact", memory_type="semantic")
    )

    assert output.count >= 1
    assert any("persistent fact" in item["content"] for item in output.items)


def test_memory_tool_search_without_type_covers_episodic(manager: MemoryManager):
    """回归：不指定 memory_type 的 search 必须能查到 episodic 里的问答/经历。

    历史缺陷：search 默认只搜 working，而问答留痕写在 episodic，导致
    「我这两天问过什么」这类问题永远检索不到。
    """
    from tool.memory_add import MemoryAddInput, MemoryAddTool
    from tool.memory_query import MemoryQueryInput, MemoryQueryTool

    writer = MemoryAddTool(manager=manager)
    writer.execute(
        MemoryAddInput(
            content="问：我这两天的计划\n答：先把后端修好",
            memory_type="episodic",
        )
    )
    reader = MemoryQueryTool(manager=manager)

    # 注意：conftest 的 HashEmbedding 按 \w+ 分词，查询需与内容存在同 token
    # （未配置真实嵌入模型时中文语义检索能力有限，这里只验证跨层作用域）。
    found = reader.execute(MemoryQueryInput(action="search", query="我这两天的计划"))
    assert found.count >= 1
    assert any("先把后端修好" in item["content"] for item in found.items)

    # 显式限定 working 时仍然查不到（保持分层过滤语义）
    scoped = reader.execute(
        MemoryQueryInput(action="search", query="我这两天的计划", memory_type="working")
    )
    assert scoped.count == 0


def test_memory_tool_clear_without_type_only_touches_working(
    manager: MemoryManager,
):
    """安全回归：clear 省略 memory_type 时不得清空全库（仍只清 working）。"""
    from tool.memory_add import MemoryAddInput, MemoryAddTool
    from tool.memory_tool import MemoryManageInput, MemoryManageTool

    writer = MemoryAddTool(manager=manager)
    writer.execute(MemoryAddInput(content="要保留的问答", memory_type="episodic"))
    writer.execute(MemoryAddInput(content="临时工作记忆"))

    MemoryManageTool(manager=manager).execute(MemoryManageInput(action="clear"))

    assert len(manager.list(memory_type="episodic")) == 1
    assert manager.list(memory_type="working") == []


def test_rag_tool_ingest_and_retrieve_with_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "rag-memory.sqlite3"))
    from memory import default_sqlite_path
    from tool.rag_search import RAGSearchInput, RAGSearchTool
    from tool.rag_tool import RAGTool, RAGToolInput

    def make_manager() -> MemoryManager:
        return MemoryManager(
            MemoryConfig(sqlite_path=default_sqlite_path()),
            embedding=HashEmbedding(),
        )

    ingest = RAGTool(pipeline=RAGPipeline(make_manager())).execute(
        RAGToolInput(
            action="ingest",
            text="The zebra lives in savannah. " * 20,
            chunk_size=100,
            overlap=20,
        )
    )
    assert ingest.count >= 2

    # 读取侧同样是「写入之后新建」的实例：验证落盘 → 恢复向量链路。
    search = RAGSearchTool(pipeline=RAGPipeline(make_manager()))
    retrieved = search.execute(
        RAGSearchInput(action="retrieve", query="zebra savannah", limit=2)
    )
    assert retrieved.count >= 1
    assert "zebra" in " ".join(item["content"] for item in retrieved.items)

    context = search.execute(
        RAGSearchInput(action="context", query="zebra savannah", limit=2)
    )
    assert context.count == 1
    assert "zebra" in context.context


# ---------------------------------------------------------------------------
# Phase 2: 摄取双写（documents/chunks 真值源 + 状态机）
# ---------------------------------------------------------------------------


@pytest.fixture()
def persistent_pipeline(tmp_path):
    """File-backed manager: the repository and the store share one sqlite file."""

    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
    )
    pipeline = RAGPipeline(manager)
    yield pipeline
    pipeline.close()


class ExplodingExtractor:
    def extract(self, text: str, *, metadata=None) -> ExtractionResult:
        raise RuntimeError("知识抽取服务不可用")


def test_ingest_persists_document_and_chunks(persistent_pipeline: RAGPipeline):
    document = Document(
        "Qdrant 是向量库。" * 40,
        id="doc-persist",
        metadata={
            "source": "课程设计.txt",
            "title": "存储架构",
            "tags": ["课程设计", "权限A"],
            "permission": "shared",
        },
    )
    items = persistent_pipeline.ingest(document, chunk_size=200, overlap=40)

    repo = persistent_pipeline.document_repo()
    assert repo is not None
    stored = repo.get_document("doc-persist")
    assert stored is not None
    assert stored.status == "extracted"
    assert stored.title == "存储架构"
    assert stored.source == "课程设计.txt"
    assert stored.tags == ["课程设计", "权限A"]
    assert stored.permission == "shared"
    assert stored.raw_text == persistent_pipeline.processor.normalized_text(document)

    chunks = repo.list_chunks("doc-persist")
    assert len(chunks) == len(items) > 1
    assert [chunk.chunk_id for chunk in chunks] == [item.id for item in items]
    assert all(chunk.vector_status == "indexed" for chunk in chunks)
    assert repo.stats() == {
        "documents": 1,
        "chunks": len(items),
        "chunks_indexed": len(items),
    }


def test_ingest_raw_text_matches_chunk_source(persistent_pipeline: RAGPipeline):
    """raw_text 与分块必须同源，否则 char_start/char_end 定位不到原文（方案 2.3）。"""

    document = Document("混合检索 RRF 融合。" * 30, id="doc-spans")
    persistent_pipeline.ingest(document, chunk_size=120, overlap=20)

    repo = persistent_pipeline.document_repo()
    stored = repo.get_document("doc-spans")
    chunks = repo.list_chunks("doc-spans")

    assert stored is not None
    for chunk in (chunks[0], chunks[-1]):
        assert stored.raw_text[chunk.char_start : chunk.char_end] == chunk.text
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end <= len(stored.raw_text)


def test_ingest_marks_vectorized_when_extraction_is_off(persistent_pipeline: RAGPipeline):
    persistent_pipeline.auto_extract = False
    persistent_pipeline.ingest(Document("关闭抽取时终态是 vectorized。", id="doc-nokx"))

    repo = persistent_pipeline.document_repo()
    stored = repo.get_document("doc-nokx")
    assert stored is not None and stored.status == "vectorized"


def test_extractor_failure_marks_document_failed(tmp_path):
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
    )
    pipeline = RAGPipeline(manager, extractor=ExplodingExtractor())
    try:
        items = pipeline.ingest(Document("抽取会抛异常。", id="doc-fail"))

        repo = pipeline.document_repo()
        stored = repo.get_document("doc-fail")
        assert stored is not None
        assert stored.status == "failed"
        assert stored.error and "知识抽取服务不可用" in stored.error
        # 抽取失败不能丢原文：分块照样已入库且已索引。
        assert repo.stats() == {"documents": 1, "chunks": len(items), "chunks_indexed": len(items)}
        assert pipeline.last_ingest_report["errors"]
    finally:
        pipeline.close()


def test_reingest_is_idempotent(persistent_pipeline: RAGPipeline):
    document = Document("重复导入必须幂等。" * 20, id="doc-again")
    first_items = persistent_pipeline.ingest(document, chunk_size=100, overlap=20)
    created_at = persistent_pipeline.document_repo().get_document("doc-again").created_at

    second_items = persistent_pipeline.ingest(document, chunk_size=100, overlap=20)

    repo = persistent_pipeline.document_repo()
    assert repo.count_documents() == 1
    assert len(second_items) == len(first_items)
    assert repo.stats()["chunks"] == len(first_items)
    assert repo.get_document("doc-again").created_at == created_at


def test_delete_document_also_clears_repository(persistent_pipeline: RAGPipeline):
    persistent_pipeline.ingest(Document("待删除的文档。" * 10, id="doc-del", metadata={"source": "x"}))
    repo = persistent_pipeline.document_repo()
    assert repo.stats()["chunks"] > 0

    removed = persistent_pipeline.delete_document("doc-del")

    assert removed > 0
    assert repo.get_document("doc-del") is None
    assert repo.stats() == {"documents": 0, "chunks": 0, "chunks_indexed": 0}


def test_in_memory_store_skips_repository_and_still_ingests(pipeline: RAGPipeline):
    """:memory: 存储没有可共享的真值文件，必须跳过双写而不是写进另一个内存库。"""

    assert pipeline.document_repo() is None
    items = pipeline.ingest(Document("内存模式照常可摄入。", id="doc-mem"))
    assert [item.id for item in items] == ["doc-mem:0"]
    assert pipeline.manager.document_store.get("doc-mem:0") is not None


def test_unknown_permission_and_scalar_tags_are_safe(persistent_pipeline: RAGPipeline):
    persistent_pipeline.ingest(
        Document("权限值非法时必须按 private 处理。", id="doc-perm", metadata={"permission": "secret", "tags": "单个标签"})
    )

    stored = persistent_pipeline.document_repo().get_document("doc-perm")
    assert stored is not None
    assert stored.permission == "private"
    assert stored.tags == ["单个标签"]
