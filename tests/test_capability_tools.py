"""能力工具契约：混合索引 / 混合召回 / 多路混合召回 / 图节点更新。

这四个能力原先内嵌在 ``RAGPipeline``（``hybrid_retrieve``/``hybrid_retrieve_multi``/
``_vector_hits``/``_rrf_fuse``/``ingest`` 里的双写）、``GraphRAGPipeline.retrieve_multi``
和 ``EntityResolver.resolve`` 里，只能整条管道调用。拆到 ``tool/`` 下成为独立工具后
必须同时满足三件事：

1. 能被 ``core.discover_tools`` 自动发现并注册（``TOOL_ENABLED`` + ``create_tool``）；
2. 读/写副作用标注正确——只读召回不得要求写确认，索引与节点更新必须要求；
3. 管道内部仍调用同一份实现（不允许出现第二份拷贝），因此这些用例同时是
   「管道行为不变」的回归。
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from core import ToolRegistry, discover_tools
from memory import MemoryConfig, MemoryManager
from memory.base import MemoryType
from memory.rag import Document, RAGPipeline
from memory.rag.knowledge import EntityResolver
from tool.graph_node_update import (
    GraphNodeUpdateInput,
    GraphNodeUpdateTool,
    find_entity_node,
    update_entity_node,
)
from tool.hybrid_index import HybridIndexInput, HybridIndexTool, index_chunk, repository_for
from tool.hybrid_recall import HybridRecallInput, HybridRecallTool, hybrid_recall
from tool.multi_recall import (
    MultiRecallInput,
    MultiRecallTool,
    NullQueryDecomposer,
    build_decomposer,
    graph_recall_multi,
    hybrid_recall_multi,
)

CAPABILITY_TOOLS = (
    "knowledge.hybrid_index",
    "knowledge.hybrid_recall",
    "knowledge.multi_recall",
    "knowledge.graph_node_update",
)


@pytest.fixture()
def pipeline(tmp_path):
    """SQLite-backed pipeline: the keyword side only exists on a real file."""

    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
    )
    instance = RAGPipeline(manager, auto_extract=False)
    try:
        yield instance
    finally:
        instance.close()


def _ingest(pipeline: RAGPipeline, text: str, document_id: str) -> None:
    pipeline.ingest(Document(text, id=document_id), chunk_size=120, overlap=20)


def test_capability_tools_are_discovered_and_registered() -> None:
    """四个能力工具必须能被自动发现注册，且发现过程零错误。"""

    registry = ToolRegistry()
    report = discover_tools(registry)

    assert report.ok, [record.error for record in report.errors]
    assert {record.tool_name for record in report.registered} >= set(CAPABILITY_TOOLS)
    for name in CAPABILITY_TOOLS:
        assert registry.resolve(name)[0].spec.name == name


def test_read_and_write_side_effects_are_declared_correctly() -> None:
    """召回类只读（免确认），索引与节点更新类写入（必须确认）。"""

    read_tools = (HybridRecallTool().spec, MultiRecallTool().spec)
    write_tools = (HybridIndexTool().spec, GraphNodeUpdateTool().spec)

    assert {spec.name for spec in read_tools} == {
        "knowledge.hybrid_recall",
        "knowledge.multi_recall",
    }
    assert all(spec.side_effect == "read" for spec in read_tools)
    assert {spec.name for spec in write_tools} == {
        "knowledge.hybrid_index",
        "knowledge.graph_node_update",
    }
    assert all(spec.side_effect == "write" for spec in write_tools)


def test_hybrid_index_writes_both_projections(pipeline) -> None:
    """双写顺序：真值源（chunks+FTS5）先落，再写向量，最后置 vector_status。"""

    manager = pipeline.manager
    repository = repository_for(manager)
    assert repository is not None

    item = index_chunk(
        manager,
        repository,
        chunk_id="doc-1:0",
        document_id="doc-1",
        chunk_index=0,
        char_start=0,
        char_end=12,
        text="设备编号 abc-123",
        metadata={"document_id": "doc-1", "chunk_index": 0},
    )

    assert repository.get_chunk("doc-1:0").text == "设备编号 abc-123"
    assert manager.document_store.get(item.id) is not None
    assert repository.chunk_ids(vector_status="indexed") == ["doc-1:0"]
    # 关键词路立刻可召回（FTS5 触发器随 upsert_chunk 生效）
    assert [chunk_id for chunk_id, _ in repository.search_keywords("abc-123")] == ["doc-1:0"]


def test_hybrid_index_tool_is_idempotent_per_chunk_id(pipeline) -> None:
    """同一分块重复索引不产生重复行（chunk_id 既是主键也是记忆 id）。"""

    tool = HybridIndexTool(manager=pipeline.manager)
    arguments = HybridIndexInput(
        text="同一段文本重复入库。",
        document_id="doc-idem",
        chunk_index=0,
        source="幂等.txt",
    )

    first = tool.execute(arguments)
    second = tool.execute(arguments)

    assert first.chunk_id == second.chunk_id == "doc-idem:0"
    assert first.keyword_indexed is True
    assert first.vector_status == "indexed"
    repository = repository_for(pipeline.manager)
    assert repository is not None
    assert repository.chunk_ids() == ["doc-idem:0"]
    assert len(
        [
            item
            for item in pipeline.manager.list(
                memory_type=MemoryType.SEMANTIC, include_expired=True
            )
            if item.id == "doc-idem:0"
        ]
    ) == 1


def test_hybrid_recall_tool_exposes_per_path_scores(pipeline) -> None:
    """召回结果必须能溯源：融合分与两路原始分同时可见。"""

    _ingest(pipeline, "设备编号 abc-123 的混合召回配置。" * 8, "doc-recall")
    tool = HybridRecallTool(pipeline=pipeline)

    output = tool.execute(HybridRecallInput(query="abc-123", limit=3))

    assert output.count == len(output.hits) >= 1
    assert output.note == ""
    assert output.vector_available is True
    top = output.hits[0]
    assert "abc-123" in top.snippet
    assert top.document_id == "doc-recall"
    assert top.rrf_score == pytest.approx(top.score)
    assert top.vector_score is not None and top.keyword_score is not None


def test_hybrid_recall_tool_reports_degradation_instead_of_failing(
    pipeline, monkeypatch
) -> None:
    """向量路故障时只读工具降级为纯关键词，并把原因写进 note。"""

    _ingest(pipeline, "设备编号 abc-123 的降级召回。" * 8, "doc-down")

    def broken_search(*args, **kwargs):
        raise RuntimeError("embedding API request failed: cloud endpoint unreachable")

    monkeypatch.setattr(pipeline.manager, "search", broken_search)

    output = HybridRecallTool(pipeline=pipeline).execute(
        HybridRecallInput(query="abc-123", limit=3)
    )

    assert output.hits and "abc-123" in output.hits[0].snippet
    assert output.vector_available is False
    assert "降级" in output.note
    assert output.hits[0].vector_score is None
    assert output.hits[0].keyword_score is not None


def test_multi_recall_runs_both_modes_over_decomposed_queries(pipeline) -> None:
    """多路召回：分解出的每条子查询都参与，融合后一次返回块与关系路径。"""

    _ingest(pipeline, "DSV4.1 部署在本地 Qdrant 与 Neo4j 上。" * 8, "doc-multi")
    tool = MultiRecallTool(pipeline=pipeline, decomposer=NullQueryDecomposer())

    output = tool.execute(
        MultiRecallInput(query="DSV4.1 部署在哪", mode="both", limit=3)
    )

    assert output.sub_queries == ["DSV4.1 部署在哪"]
    assert output.hits, "混合路必须有命中"
    assert output.note == ""
    # 本次入库未跑 LLM 抽取：图里没有实体与关系路径，但语义证据仍应作为上下文给出，
    # 且绝不允许伪造路径。
    assert output.entities == []
    assert output.paths == []
    assert output.context.startswith("[证据|")


def test_multi_recall_rejects_empty_queries_and_merges_paths(pipeline) -> None:
    """空查询列表是调用错误；多路图召回按 effective 合并去重。"""

    with pytest.raises(ValueError, match="queries"):
        graph_recall_multi(pipeline, [])

    assert hybrid_recall_multi(pipeline, []).chunks == []
    assert hybrid_recall_multi(pipeline, ["", "   "]).chunks == []


def test_multi_recall_single_query_matches_single_path_recall(pipeline) -> None:
    """只有一条查询时多路必须与单路逐条一致（不引入额外排序差异）。"""

    _ingest(pipeline, "设备编号 abc-123 的单路与多路一致性。" * 8, "doc-single")

    single = hybrid_recall(pipeline, "abc-123", limit=3).chunks
    multi = hybrid_recall_multi(pipeline, ["abc-123"], limit=3).chunks

    assert [chunk.memory_id for chunk in multi] == [chunk.memory_id for chunk in single]


def test_decomposer_falls_back_to_null_without_a_real_api_key() -> None:
    """没有真实密钥时分解器必须退化为原句，绝不因分解失败而答不出来。"""

    decomposer = build_decomposer()

    assert isinstance(decomposer, NullQueryDecomposer)
    assert decomposer.decompose("  星云计划 何时交付  ") == ["星云计划 何时交付"]
    assert decomposer.decompose("   ") == []


def test_graph_node_update_creates_then_merges_attributes(manager) -> None:
    """节点更新：别名取并集、重要度只升不降、id 稳定。"""

    created = update_entity_node(
        manager, "Qdrant", domain="存储", aliases=["qdrant"], importance=0.4
    )
    assert created.metadata["kind"] == "entity"
    assert created.metadata["domain"] == "存储"
    assert created.metadata["aliases"] == ["qdrant"]
    assert created.importance == pytest.approx(0.4)

    updated = update_entity_node(
        manager,
        "Qdrant",
        existing=created,
        description="向量数据库",
        aliases=["向量库"],
        importance=0.9,
    )
    assert updated.id == created.id
    assert updated.metadata["aliases"] == ["qdrant", "向量库"]  # 并集，不丢旧别名
    assert updated.metadata["domain"] == "存储"  # 未传即保持
    assert updated.metadata["description"] == "向量数据库"
    assert updated.importance == pytest.approx(0.9)

    downgraded = update_entity_node(manager, "Qdrant", existing=updated, importance=0.1)
    assert downgraded.importance == pytest.approx(0.9)  # 低置信抽取不得降级节点


def test_graph_node_update_tool_projects_onto_existing_graph_node(manager) -> None:
    """图投影只更新已存在的节点，绝不凭空造节点。"""

    manager.graph_store.add_relation(
        "Qdrant",
        "用于",
        "语义检索",
        source_domain="存储",
        source_importance=0.5,
    )
    # 真值源先有实体行（created 描述的是真值源，不是图投影）
    update_entity_node(manager, "Qdrant", domain="存储", importance=0.5)
    assert manager.graph_store.entity("Qdrant")["domain"] == "存储"

    tool = GraphNodeUpdateTool(manager=manager)
    output = tool.execute(
        GraphNodeUpdateInput(
            name="Qdrant", domain="向量存储", aliases=["向量库"], importance=0.9
        )
    )

    assert output.created is False
    assert output.graph_projected is True
    assert output.domain == "向量存储"
    assert manager.graph_store.entity("Qdrant")["domain"] == "向量存储"
    assert manager.graph_store.entity("Qdrant")["importance"] == pytest.approx(0.9)

    # 不存在的实体只写真值源，不往图里塞孤立节点
    absent = tool.execute(GraphNodeUpdateInput(name="尚未出现的实体"))
    assert absent.created is True
    assert absent.graph_projected is False
    assert manager.graph_store.entity("尚未出现的实体") == {}


def test_graph_node_update_can_refuse_to_create(manager) -> None:
    """``create_if_missing=False`` 时找不到节点必须报错，而不是静默新建。"""

    assert find_entity_node(manager, "不存在的实体") is None
    with pytest.raises(LookupError):
        update_entity_node(manager, "不存在的实体", create_if_missing=False)


def test_entity_resolver_shares_the_node_update_implementation(manager) -> None:
    """``EntityResolver.resolve`` 的属性写入必须与工具同源（别名并集/重要度取大）。"""

    resolver = EntityResolver(manager)
    resolver.resolve("星云", domain="项目", aliases=["Nebula"], confidence=0.4)
    canonical = resolver.resolve("星云", domain="", aliases=["知识库"], confidence=0.9)

    stored = find_entity_node(manager, canonical)
    assert stored is not None
    assert stored.metadata["domain"] == "项目"
    assert stored.metadata["aliases"] == ["Nebula", "知识库"]
    assert stored.importance == pytest.approx(0.9)
