"""Phase 3: Qdrant 轻量化 payload、命中回查 chunks、FTS5 × 向量 RRF 混合检索。

覆盖方案 P3 的验收点：payload 只留回查/过滤所需键、search 优先 chunk_id、
维度校验不变、FTS5 触发器同步、精确词命中、转义不 500、RRF 融合排序、
配置开关回到纯向量、嵌入不可达时降级为纯关键词（D8）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager
from memory.base import MemoryItem, MemoryType
from memory.rag import Document
from memory.rag.pipeline import RAGPipeline, _rrf_fuse
from memory.storage.document_repo import ChunkRecord, DocumentRecord, DocumentRepository
from memory.storage.qdrant import QdrantVectorStore


class FakeQdrantClient:
    """Minimal QdrantClient stand-in: records upserts, replays canned search hits."""

    def __init__(
        self,
        *,
        exists: bool = False,
        existing_size: int | None = None,
        hits: list | None = None,
        named_vectors: bool = False,
    ) -> None:
        self.exists, self.existing_size, self.hits = exists, existing_size, hits or []
        self.named_vectors = named_vectors
        self.points: list = []
        self.vectors_config = None
        self.searches: list[dict] = []
        self.deleted_collections: list[str] = []

    def collection_exists(self, *, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, *, collection_name: str, vectors_config) -> None:
        self.exists = True
        self.vectors_config = vectors_config
        size = getattr(vectors_config, "size", None)
        if size is not None:
            self.existing_size = size

    def delete_collection(self, *, collection_name: str) -> None:
        self.exists = False
        self.deleted_collections.append(collection_name)
        self.points.clear()
        self.existing_size = None

    def get_collection(self, *, collection_name: str):
        if self.named_vectors:
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=self.existing_size)
                    )
                )
            )
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(size=self.existing_size)))

    def upsert(self, *, collection_name: str, points: list) -> None:
        if points and self.existing_size is not None:
            vector = getattr(points[0], "vector", None)
            if vector is not None and len(vector) != self.existing_size:
                raise RuntimeError(
                    "Unexpected Response: 400 (Bad Request) "
                    f"expected dim: {self.existing_size}, got {len(vector)}"
                )
        self.points.extend(points)

    def search(self, *, collection_name: str, query_vector, query_filter, limit: int) -> list:
        self.searches.append({"query_vector": query_vector, "query_filter": query_filter, "limit": limit})
        return self.hits[:limit]


# ---------------------------------------------------------------------------
# Qdrant payload / look-back key
# ---------------------------------------------------------------------------


def test_upsert_chunk_payload_lightweight():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, namespace="tests")

    store.upsert_chunk("doc:3", [0.1, 0.2], document_id="doc", chunk_index=3, source="file.pdf")

    payload = client.points[0].payload
    # 方案 2.2 的 5 个业务键 + 存储层补的 namespace（search 强制按它过滤，缺了就永远查不到）。
    assert payload == {
        "chunk_id": "doc:3",
        "document_id": "doc",
        "chunk_index": 3,
        "source": "file.pdf",
        "memory_type": "semantic",
        "namespace": "tests",
    }
    assert "embedding" not in payload


def test_upsert_memory_payload_drops_heavy_fields():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client, namespace="tests")
    item = MemoryItem(content="整段正文", memory_type=MemoryType.SEMANTIC, embedding=[0.1, 0.2], metadata={"a": 1})

    store.upsert(item)

    payload = client.points[0].payload
    assert set(payload) == {"id", "memory_type", "created_at", "expires_at", "namespace"}
    assert payload["id"] == item.id
    assert payload["memory_type"] == "semantic"
    assert "embedding" not in payload and "content" not in payload and "metadata" not in payload


def test_search_prefers_chunk_id():
    hits = [
        SimpleNamespace(id="uuid-point", payload={"chunk_id": "doc:1", "id": "old-field"}, score=0.9),
        SimpleNamespace(id="uuid-point-2", payload={"id": "doc:2"}, score=0.5),
        SimpleNamespace(id="uuid-point-3", payload=None, score=0.1),
    ]
    store = QdrantVectorStore(client=FakeQdrantClient(hits=hits), namespace="tests")
    store._ensure_collection(2)

    assert store.search([0.1, 0.2], limit=3) == [("doc:1", 0.9), ("doc:2", 0.5), ("uuid-point-3", 0.1)]


def test_dimension_mismatch_still_raises():
    client = FakeQdrantClient(exists=True, existing_size=8)
    store = QdrantVectorStore(client=client, dimension=4, namespace="tests")

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.upsert(MemoryItem(content="x", memory_type=MemoryType.SEMANTIC, embedding=[0.1] * 4))
    assert client.deleted_collections == []


def test_dimension_mismatch_on_upsert_does_not_recreate():
    """写入路径发现维度冲突必须抬错，不能静默 recreate 丢掉旧投影。"""

    client = FakeQdrantClient(exists=True, existing_size=1024)
    store = QdrantVectorStore(client=client, namespace="tests")
    store._ensure_collection(1024)

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.upsert_chunk("d1:0", [0.1] * 4096)

    assert client.deleted_collections == []
    assert client.existing_size == 1024
    assert client.points == []


def test_configured_dimension_is_not_treated_as_collection_size():
    """配置护栏 1024 不能冒充集合现有维度：集合已是 4096 时同维写入必须成功。"""

    client = FakeQdrantClient(exists=True, existing_size=4096)
    store = QdrantVectorStore(client=client, dimension=1024, namespace="tests")

    store.upsert_chunk("d1:0", [0.1] * 4096)

    assert len(client.points) == 1
    assert store.dimension == 4096


def test_dimension_mismatch_reads_qdrant_vectors_size():
    """真实 qdrant-client 把尺寸放在 config.params.vectors.size，不能只读扁平 size。"""

    client = FakeQdrantClient(exists=True, existing_size=1024, named_vectors=True)
    store = QdrantVectorStore(client=client, namespace="tests")

    with pytest.raises(ValueError, match="dimension mismatch") as excinfo:
        store.upsert_chunk("d1:0", [0.1] * 4096)

    assert "1024" in str(excinfo.value) and "4096" in str(excinfo.value)


def test_dimension_mismatch_message_points_to_single_knob_and_rebuild():
    """1024/4096 冲突时：报错必须点名集合、说明唯一开关并给出重建命令。"""

    client = FakeQdrantClient(exists=True, existing_size=1024)
    store = QdrantVectorStore(client=client, namespace="tests")

    with pytest.raises(ValueError, match="dimension mismatch") as excinfo:
        store.upsert_chunk("d1:0", [0.1] * 4096)

    message = str(excinfo.value)
    assert "1024" in message and "4096" in message
    assert "[embedding].model" in message  # 唯一可改的开关
    assert "migrate_to_cloud.py --recreate-collection" in message
    assert "reindex_embeddings.py" in message


def test_recreate_collection_rebuilds_projection_at_new_dimension():
    client = FakeQdrantClient(exists=True, existing_size=1024)
    store = QdrantVectorStore(client=client, namespace="tests")

    store.recreate_collection(4096)

    assert client.deleted_collections == [store.collection_name]  # 旧投影被丢弃
    assert client.vectors_config.size == 4096  # 新集合按新维度建立
    assert store.dimension == 4096
    # 重建后同维度 upsert 不再报维度不一致。
    store.upsert_chunk("d1:0", [0.1] * 4096)
    assert len(client.points) == 1


# ---------------------------------------------------------------------------
# FTS5 keyword path
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path):
    repository = DocumentRepository(tmp_path / "memory.sqlite3")
    yield repository
    repository.close()


def seed_chunks(repository: DocumentRepository, texts: list[str]) -> None:
    repository.upsert_document(DocumentRecord(document_id="d1", raw_text=" ".join(texts), status="parsed"))
    repository.upsert_chunks(
        [ChunkRecord(f"d1:{index}", "d1", index, 0, len(text), text) for index, text in enumerate(texts)]
    )


def test_fts_index_syncs_on_insert_update_delete(repo: DocumentRepository):
    assert repo.fts_tokenizer in {"trigram", "unicode61"}  # 本机实测为 trigram
    seed_chunks(repo, ["第一个分块讲混合检索。"])
    assert [chunk_id for chunk_id, _ in repo.search_keywords("混合检索")] == ["d1:0"]

    repo.upsert_chunk(ChunkRecord("d1:1", "d1", 1, 0, 10, "第二个分块讲图数据库 Neo4j。"))
    assert [chunk_id for chunk_id, _ in repo.search_keywords("Neo4j")] == ["d1:1"]

    repo.upsert_chunk(ChunkRecord("d1:1", "d1", 1, 0, 10, "第二个分块被改写，不再有那个词。"))
    assert repo.search_keywords("Neo4j") == []

    repo.delete_document("d1")
    assert repo.search_keywords("混合检索") == []


def test_keyword_search_finds_exact_identifier(repo: DocumentRepository):
    seed_chunks(repo, ["设备型号 X200 支持混合检索。", "完全无关的另一段文本。"])

    hits = repo.search_keywords("X200")

    assert [chunk_id for chunk_id, _ in hits] == ["d1:0"]
    # 分数口径是「越大越相关」（bm25 取负），便于调用方排序。
    assert hits[0][1] > 0


def test_fts_query_escapes_dash_and_colon(repo: DocumentRepository):
    """本机实测：不转义时 abc-123 抛 no such column: 123，接口直接 500。"""

    seed_chunks(repo, ["编号 abc-123 已登记。"])

    assert [chunk_id for chunk_id, _ in repo.search_keywords("abc-123")] == ["d1:0"]
    assert repo.search_keywords("型号: X200") == []  # 不抛异常即达标
    assert DocumentRepository._fts_query("abc-123") == '"abc-123"'


def test_fts_query_escapes_embedded_quote(repo: DocumentRepository):
    seed_chunks(repo, ['引号 " 不能破坏 MATCH 语法。'])

    assert DocumentRepository._fts_query('a"b') == '"a""b"'
    repo.search_keywords('引号 " 不')  # 不抛异常


def test_keyword_search_ignores_blank_query_and_rejects_bad_limit(repo: DocumentRepository):
    seed_chunks(repo, ["有内容。"])

    assert repo.search_keywords("   ") == []

    with pytest.raises(ValueError, match="limit"):
        repo.search_keywords("内容", limit=0)


# ---------------------------------------------------------------------------
# RRF fusion and degradation
# ---------------------------------------------------------------------------


def test_rrf_fuse_prefers_common_hits():
    fused = _rrf_fuse([["a", "b", "c"], ["b", "a"]])

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)  # hit by both paths
    assert _rrf_fuse([[], []]) == []


def _break_vector_search(pipeline: RAGPipeline, monkeypatch) -> None:
    """让向量路在查询时抛 RuntimeError（云端端点故障的等价模拟）。

    历史版本用「本机没人监听的端口 + TCP 预检」模拟网关掉线；预检删除后，
    降级信号就是 ``manager.search`` 在嵌入查询时抛出的异常本身。
    入库仍用哈希嵌入（对应「云端在线时已入库」的存量数据）。
    """

    def broken_search(*args, **kwargs):
        raise RuntimeError("embedding API request failed: cloud endpoint unreachable")

    monkeypatch.setattr(pipeline.manager, "search", broken_search)


def build_pipeline(tmp_path, embedding) -> RAGPipeline:
    manager = MemoryManager(MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")), embedding=embedding)
    return RAGPipeline(manager, auto_extract=False)


def test_hybrid_retrieve_uses_both_paths(tmp_path):
    pipeline = build_pipeline(tmp_path, HashEmbedding())
    try:
        pipeline.ingest(
            Document("设备编号 abc-123 的混合检索配置。" * 8, id="doc-hybrid"),
            chunk_size=120,
            overlap=20,
        )

        results = pipeline.hybrid_retrieve("abc-123", limit=3)

        assert results
        assert any("abc-123" in result.content for result in results)
        assert results[0].metadata["document_id"] == "doc-hybrid"
        assert pipeline.last_retrieval_note == ""
    finally:
        pipeline.close()


def test_hybrid_falls_back_to_vector_when_disabled(tmp_path, monkeypatch):
    import memory.rag.pipeline as rag_pipeline

    monkeypatch.setattr(rag_pipeline, "MEMORY_HYBRID", False)
    pipeline = build_pipeline(tmp_path, HashEmbedding())
    try:
        pipeline.ingest(Document("关闭混合检索后走纯向量路径。" * 8, id="doc-off"), chunk_size=120, overlap=20)

        assert [chunk.memory_id for chunk in pipeline.hybrid_retrieve("纯向量路径", limit=3)] == [
            chunk.memory_id for chunk in pipeline.retrieve("纯向量路径", limit=3)
        ]
    finally:
        pipeline.close()


def test_hybrid_degrades_to_keyword_when_embedding_down(tmp_path, monkeypatch):
    """D8: 云端嵌入在查询时故障，检索必须降级为 FTS5，而不是整体失败。"""

    pipeline = build_pipeline(tmp_path, HashEmbedding())
    try:
        pipeline.ingest(Document("设备编号 abc-123 的降级检索。" * 8, id="doc-down"), chunk_size=120, overlap=20)
        _break_vector_search(pipeline, monkeypatch)

        results = pipeline.hybrid_retrieve("abc-123", limit=3)

        assert results and "abc-123" in results[0].content
        assert "降级" in pipeline.last_retrieval_note
        assert "向量检索失败" in pipeline.last_retrieval_note
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# Phase 7 U4: 溯源分数明细
# ---------------------------------------------------------------------------


def test_hybrid_retrieve_exposes_per_path_scores(tmp_path):
    """U4：融合前要留住两路原始分，否则面板只剩一个 RRF 分数、贡献不可见。"""

    pipeline = build_pipeline(tmp_path, HashEmbedding())
    try:
        pipeline.ingest(
            Document("设备编号 abc-123 的混合检索配置。" * 8, id="doc-score"),
            chunk_size=120,
            overlap=20,
        )

        results = pipeline.hybrid_retrieve("abc-123", limit=3)

        assert results
        top = results[0]
        assert set(top.detail) == {"rrf_score", "vector_score", "keyword_score"}
        # 两路都命中：三个分数都在，且 RRF 分数与对外 score 一致
        assert top.detail["vector_score"] is not None
        assert top.detail["keyword_score"] is not None
        assert top.detail["rrf_score"] == pytest.approx(top.score)
        # RRF 是名次分：两位有效数字内必然小于 2/(k+1)，用它区分「真分数」实现
        assert 0 < top.detail["rrf_score"] < 0.033
        assert all(result.detail["rrf_score"] is not None for result in results)
    finally:
        pipeline.close()


def test_hybrid_detail_marks_the_missing_path_when_degraded(tmp_path, monkeypatch):
    """降级时向量分为 None（面板显示「—」），关键词分仍在——降级原因因此在界面上可见。"""

    pipeline = build_pipeline(tmp_path, HashEmbedding())
    try:
        pipeline.ingest(Document("设备编号 abc-123 的降级检索。" * 8, id="doc-score-down"), chunk_size=120, overlap=20)
        _break_vector_search(pipeline, monkeypatch)

        results = pipeline.hybrid_retrieve("abc-123", limit=3)

        assert results
        assert results[0].detail["vector_score"] is None
        assert results[0].detail["keyword_score"] is not None
    finally:
        pipeline.close()
