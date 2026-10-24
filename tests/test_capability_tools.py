"""能力工具契约：混合索引 / 混合召回 / 多路召回 / 图节点更新 / 三库对账 / 导出导入。

这些能力原先内嵌在 ``RAGPipeline``（``hybrid_retrieve``/``hybrid_retrieve_multi``/
``_vector_hits``/``_rrf_fuse``/``ingest`` 里的双写）、``GraphRAGPipeline.retrieve_multi``、
``EntityResolver.resolve`` 与 ``web/app.py``（三库对账、漂移自愈、导出/导入）里，
只能整条管道或整个 HTTP 端点调用。拆到 ``tool/`` 下成为独立工具后必须同时满足三件事：

1. 能被 ``core.discover_tools`` 自动发现并注册（``TOOL_ENABLED`` + ``create_tool``）；
2. 读/写副作用标注正确——只读召回不得要求写确认，索引与节点更新必须要求；
3. 管道与 Web 层内部仍调用同一份实现（不允许出现第二份拷贝），因此这些用例同时是
   「原有行为不变」的回归。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import HashEmbedding

from constants import WEB_DOCUMENTS_PAGE_SIZE_MAX
from core import ToolRegistry, discover_tools
from memory import MemoryConfig, MemoryManager
from memory.base import MemoryType
from memory.rag import Document, ExtractionResult, RAGPipeline
from memory.rag.knowledge import EntityResolver
from memory.storage.document_repo import DocumentRecord
from memory.storage.vector import InMemoryVectorStore
from tool.add_fact import AddFactInput, AddFactTool, add_fact
from tool.document_get import DocumentGetInput, DocumentGetTool, get_document
from tool.document_list import DocumentListInput, DocumentListTool, list_documents
from tool.document_revectorize import (
    DocumentRevectorizeInput,
    DocumentRevectorizeTool,
    revectorize_document,
)
from tool.domain_classify import (
    KNOWN_DOMAINS,
    ClassifyDomainInput,
    ClassifyDomainTool,
    classify_domain,
    majority_domain,
)
from tool.export_knowledge import (
    EXPORT_FORMAT,
    ExportKnowledgeInput,
    ExportKnowledgeTool,
    export_filename,
    export_payload,
)
from tool.graph_node_update import (
    GraphNodeUpdateInput,
    GraphNodeUpdateTool,
    find_entity_node,
    update_entity_node,
)
from tool.graph_snapshot import (
    GraphSnapshotInput,
    GraphSnapshotTool,
    build_graph,
    domain_color,
)
from tool.hybrid_index import HybridIndexInput, HybridIndexTool, index_chunk, repository_for
from tool.hybrid_recall import HybridRecallInput, HybridRecallTool, hybrid_recall
from tool.import_knowledge import (
    ImportKnowledgeInput,
    ImportKnowledgeTool,
    import_items,
    parse_import_payload,
)
from tool.ingest_image import TEXT_ONLY_WARNING, IngestImageInput, IngestImageTool
from tool.knowledge_stats import KnowledgeStatsInput, KnowledgeStatsTool, knowledge_stats
from tool.multi_recall import (
    MultiRecallInput,
    MultiRecallTool,
    NullQueryDecomposer,
    build_decomposer,
    graph_recall_multi,
    hybrid_recall_multi,
)
from tool.orphan_entities import (
    OrphanEntitiesInput,
    OrphanEntitiesTool,
    find_orphan_entities,
)
from tool.reconcile import (
    ReconcileInput,
    ReconcileTool,
    fact_items,
    reconcile_report,
)
from tool.repair_drift import RepairDriftInput, RepairDriftTool, repair_drift
from tool.seed_knowledge import SEED_MARK, SeedKnowledgeInput, SeedKnowledgeTool, seed

#: 图片入库用例用的最小 PNG 头（内容由抽取器解释，不需要真图片）。
PNG_BYTES = b"\x89PNG\r\n\x1a\nimage"

CAPABILITY_TOOLS = (
    "knowledge.hybrid_index",
    "knowledge.hybrid_recall",
    "knowledge.multi_recall",
    "knowledge.graph_node_update",
    "knowledge.reconcile",
    "knowledge.repair_drift",
    "knowledge.export",
    "knowledge.import",
    "knowledge.classify_domain",
    "knowledge.orphan_entities",
    "knowledge.graph_snapshot",
    "knowledge.document_list",
    "knowledge.document_get",
    "knowledge.document_revectorize",
    "knowledge.add_fact",
    "knowledge.seed",
    "knowledge.ingest_image",
    "knowledge.stats",
)


class EnumerableVectorStore(InMemoryVectorStore):
    """``InMemoryVectorStore`` + Qdrant 形状的 ``list_ids`` / ``upsert_chunk``。

    三库对账要求投影「可枚举」（``list_ids``），漂移自愈要求投影「能按 chunk 写回」
    （``upsert_chunk``）——这两个方法只有 ``QdrantVectorStore`` 实现，所以对账与自愈
    天然是「远程投影」路径。测试用这个双替身把同一契约搬到内存里，从而在不依赖
    Qdrant 的前提下验证真实代码路径。
    """

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._vectors)

    def upsert_chunk(
        self,
        chunk_id: str,
        vector: list[float],
        *,
        document_id: str,
        chunk_index: int,
        source: str = "",
        memory_type: MemoryType | str | None = None,
    ) -> None:
        with self._lock:
            self._vectors[chunk_id] = (
                list(vector),
                MemoryType(memory_type) if memory_type is not None else MemoryType.SEMANTIC,
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


@pytest.fixture()
def drift_pipeline(tmp_path):
    """带「可枚举向量投影」的管道：三库对账需要能列出向量库里的 id。"""

    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
        vector_store=EnumerableVectorStore(),
    )
    instance = RAGPipeline(manager, auto_extract=False)
    try:
        yield instance
    finally:
        instance.close()


def test_capability_tools_are_discovered_and_registered() -> None:
    """四个能力工具必须能被自动发现注册，且发现过程零错误。"""

    registry = ToolRegistry()
    report = discover_tools(registry)

    assert report.ok, [record.error for record in report.errors]
    assert {record.tool_name for record in report.registered} >= set(CAPABILITY_TOOLS)
    for name in CAPABILITY_TOOLS:
        assert registry.resolve(name)[0].spec.name == name


def test_read_and_write_side_effects_are_declared_correctly() -> None:
    """只读类免确认；索引、修复、导入与其他数据写入必须确认。"""

    read_tools = (
        HybridRecallTool().spec,
        MultiRecallTool().spec,
        ReconcileTool().spec,
        ExportKnowledgeTool().spec,
        ClassifyDomainTool().spec,
        OrphanEntitiesTool().spec,
        GraphSnapshotTool().spec,
        DocumentListTool().spec,
        DocumentGetTool().spec,
        KnowledgeStatsTool().spec,
    )
    write_tools = (
        HybridIndexTool().spec,
        GraphNodeUpdateTool().spec,
        RepairDriftTool().spec,
        ImportKnowledgeTool().spec,
        DocumentRevectorizeTool().spec,
        AddFactTool().spec,
        SeedKnowledgeTool().spec,
        IngestImageTool().spec,
    )

    assert {spec.name for spec in read_tools} == {
        "knowledge.hybrid_recall",
        "knowledge.multi_recall",
        "knowledge.reconcile",
        "knowledge.export",
        "knowledge.classify_domain",
        "knowledge.orphan_entities",
        "knowledge.graph_snapshot",
        "knowledge.document_list",
        "knowledge.document_get",
        "knowledge.stats",
    }
    assert all(spec.side_effect == "read" for spec in read_tools)
    assert {spec.name for spec in write_tools} == {
        "knowledge.hybrid_index",
        "knowledge.graph_node_update",
        "knowledge.repair_drift",
        "knowledge.import",
        "knowledge.document_revectorize",
        "knowledge.add_fact",
        "knowledge.seed",
        "knowledge.ingest_image",
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


def test_reconcile_tool_reports_clean_three_store_counts(drift_pipeline) -> None:
    """对账工具：真值源/向量/图三库计数一致时 consistent=true。"""

    _ingest(drift_pipeline, "设备编号 abc-123 的对账基线。" * 8, "doc-clean")
    repository = repository_for(drift_pipeline.manager)
    assert repository is not None

    output = ReconcileTool(manager=drift_pipeline.manager).execute(ReconcileInput())

    assert output.consistent is True
    assert output.drift == []
    assert output.counts.chunks == len(repository.chunk_ids())
    assert output.counts.chunks_indexed_sqlite == output.counts.chunks
    assert output.counts.qdrant_points == output.counts.chunks
    assert output.counts.facts == 0

    # include_ids=false 只影响 id 列表，不影响计数（长列表可省）
    without_ids = reconcile_report(
        drift_pipeline.manager, repository, include_ids=False
    )
    assert all(entry["ids"] == [] for entry in without_ids["drift"]) or not without_ids["drift"]


def test_repair_drift_tool_rebuilds_a_missing_vector_projection(drift_pipeline) -> None:
    """漂移自愈：真值源标了 indexed 但向量缺失 → 重嵌入补回，且幂等。"""

    _ingest(drift_pipeline, "设备编号 abc-123 的三库对账与自愈。" * 8, "doc-drift")
    manager = drift_pipeline.manager
    repository = repository_for(manager)
    assert repository is not None
    total = len(repository.chunk_ids())
    missing_id = repository.chunk_ids()[0]
    assert repository.chunk_ids(vector_status="indexed") == repository.chunk_ids()

    # 模拟投影丢失：真值源仍标 indexed，但向量库里已经没有这一条
    assert manager.vector_store.delete(missing_id) is True

    report = reconcile_report(manager, repository)
    assert report["counts"]["chunks_indexed_sqlite"] == total
    assert report["counts"]["qdrant_points"] == total - 1
    assert [(entry["kind"], entry["count"]) for entry in report["drift"]] == [
        ("missing_vector", 1)
    ]

    tool = RepairDriftTool(manager=manager)
    repaired = tool.execute(RepairDriftInput(repair=["missing_vector"]))
    assert repaired.repaired.missing_vector == 1
    assert repaired.repaired.missing_edge == 0

    # 幂等：再修一次补 0 条，对账恢复干净
    assert tool.execute(RepairDriftInput(repair=["missing_vector"])).repaired.missing_vector == 0
    assert reconcile_report(manager, repository)["drift"] == []


def test_repair_drift_only_accepts_self_healing_kinds(drift_pipeline) -> None:
    """未知类型与不可逆的 orphan_vector 都必须被拒绝，而不是悄悄扩大副作用面。"""

    with pytest.raises(ValueError, match="不支持的修复类型"):
        repair_drift(drift_pipeline.manager, None, ["不存在的类型"])
    with pytest.raises(ValueError, match="不支持的修复类型"):
        repair_drift(drift_pipeline.manager, None, ["orphan_vector"])
    # 空清单是合法的「只报告不修」
    assert repair_drift(drift_pipeline.manager, None, [])["repaired"] == {
        "missing_vector": 0,
        "missing_edge": 0,
    }


def test_repair_drift_replays_missing_graph_edges(drift_pipeline) -> None:
    """missing_edge：语义层有 fact 但图里没有关系 → 用同一 item_id 重放，幂等。"""

    manager = drift_pipeline.manager
    item = manager.semantic.add_fact(
        "星云", "部署于", "本机", confidence=0.9, item_id="fact:nebula"
    )
    assert [fact.id for fact in fact_items(manager)] == [item.id]
    # add_fact 同时写了图投影；这里把它删掉以模拟「真值源有、图里没有」
    assert manager.graph_store.relation_memory_ids() == [item.id]
    assert manager.graph_store.delete_memory_relation(item.id) is True

    repository = repository_for(manager)
    report = reconcile_report(manager, repository)
    assert [entry["kind"] for entry in report["drift"]] == ["missing_edge"]

    output = RepairDriftTool(manager=manager).execute(
        RepairDriftInput(repair=["missing_edge"])
    )
    assert output.repaired.missing_edge == 1
    assert reconcile_report(manager, repository)["drift"] == []


def test_export_tool_serializes_a_self_describing_payload(drift_pipeline) -> None:
    """导出：带 format 版本号与计数口径的纯 JSON，且 limit 只截断 items 不骗 counts。"""

    manager = drift_pipeline.manager
    _ingest(drift_pipeline, "设备编号 abc-123 的导出基线。" * 8, "doc-export")
    manager.semantic.add_fact("星云", "部署于", "本机", confidence=0.9)

    output = ExportKnowledgeTool(manager=manager).execute(ExportKnowledgeInput())

    assert output.format == EXPORT_FORMAT
    assert output.exported_at
    assert output.filename.startswith("knowledge_export_")
    assert output.filename.endswith(".json")
    assert output.counts.total == len(output.items)
    # 计数口径必须与 items 自洽（导入方先验后写，不靠猜）
    assert output.counts.semantic == sum(
        1 for item in output.items if item["memory_type"] == "semantic"
    )
    assert output.counts.semantic >= 1
    assert output.counts.episodic == 0
    assert all(isinstance(item, dict) for item in output.items)

    # 全量导出（HTTP /api/export 走 limit=0）与记忆库条目数一致
    full = export_payload(manager, limit=0)
    assert full["counts"]["total"] == len(list(manager.list(include_expired=False)))
    assert export_filename(stamp="20260101-000000") == "knowledge_export_20260101-000000.json"


def test_export_tool_limit_is_bounded_for_tool_results(drift_pipeline) -> None:
    """工具默认 limit=100：导出结果要能被塞进模型上下文，不能无界增长。"""

    for index in range(3):
        drift_pipeline.manager.add(f"条目 {index}", memory_type=MemoryType.WORKING)

    bounded = ExportKnowledgeTool(manager=drift_pipeline.manager).execute(
        ExportKnowledgeInput(limit=2)
    )
    assert bounded.counts.total == 2
    assert len(bounded.items) == 2


def test_import_tool_round_trips_and_is_idempotent(drift_pipeline) -> None:
    """导出 → 导入到另一个库：事实与普通条目都回得来；重复导入不产生重复/重边。"""

    source = drift_pipeline.manager
    _ingest(drift_pipeline, "设备编号 abc-123 的导入基线。" * 8, "doc-import")
    source.semantic.add_fact("星云", "部署于", "本机", confidence=0.9)
    exported = export_payload(source, limit=0)
    payload = json.dumps(exported, ensure_ascii=False)

    target = MemoryManager(MemoryConfig(), embedding=HashEmbedding())
    tool = ImportKnowledgeTool(manager=target)
    first = tool.execute(ImportKnowledgeInput(payload=payload))
    second = tool.execute(ImportKnowledgeInput(payload=payload))

    assert first.imported == len(exported["items"])
    assert first.skipped == 0
    assert first.errors == []
    # 幂等：第二次全部按 id / 三元组跳过
    assert second.imported == 0
    assert second.skipped == first.imported
    assert len(list(target.list(include_expired=False))) == len(
        list(source.list(include_expired=False))
    )


def test_import_tool_rejects_non_finite_and_malformed_payloads() -> None:
    """严格 JSON：NaN/Infinity、缺 items、坏 UTF-8 都必须是 ValueError（HTTP 层映射 400）。"""

    with pytest.raises(ValueError, match="不是合法的 JSON"):
        parse_import_payload(b'{"items": [{"id": "x", "importance": NaN}]}')
    with pytest.raises(ValueError, match="找不到 items 数组"):
        parse_import_payload(b'{"format": "x"}')
    with pytest.raises(ValueError, match="不是合法的 JSON"):
        parse_import_payload(b"\xff\xfe\x00")
    # 裸数组是合法的（历史导出文件没有外层包装）
    assert parse_import_payload(b'[{"id": "a", "content": "b"}]') == [
        {"id": "a", "content": "b"}
    ]


def test_import_tool_reports_per_item_reasons_within_the_cap(manager) -> None:
    """单条失败不拖垮整批：能写的照写，失败原因有上限且逐条说明。"""

    entries = [
        "不是对象",
        {"id": "", "content": "缺 id"},
        {"id": "ok-1", "content": "正常条目", "memory_type": "working"},
        {"id": "bad-type", "content": "x", "memory_type": "不存在的层"},
        {"id": "ok-2", "content": "又一条", "metadata": "不是对象"},
    ]
    result = import_items(manager, entries, max_errors=2)

    assert result["imported"] == 1
    assert result["skipped"] == 4
    assert len(result["errors"]) == 2
    assert "不是 JSON 对象" in result["errors"][0]
    assert manager.get("ok-1") is not None


def test_classify_domain_tool_is_offline_and_deterministic() -> None:
    """领域分类工具：纯规则、可离线、同输入同输出，并如实报告「未命中」。"""

    tool = ClassifyDomainTool()
    hit = tool.execute(ClassifyDomainInput(text="Python 的函数与变量", title="c语言笔记.txt"))
    assert hit.domain == "编程开发"
    assert hit.matched is True
    assert hit.known_domains == list(KNOWN_DOMAINS)

    # 同一输入必须永远同一结果（跨进程稳定，词表与权重都在 constants 里）
    assert classify_domain("微分方程与矩阵特征值") == "数学"
    assert classify_domain("微分方程与矩阵特征值") == classify_domain("微分方程与矩阵特征值")
    assert majority_domain(["数学", "数学", "物理"]) == "数学"

    # 命不中任何关键词时如实返回兜底领域，且 matched=false
    miss = tool.execute(ClassifyDomainInput(text="今天天气很好，出门散步"))
    assert miss.matched is False
    assert miss.domain == "未分类"
    assert majority_domain([]) == "未分类"


def test_graph_snapshot_tool_projects_the_four_layers(drift_pipeline) -> None:
    """星图投影：文档→行星、领域→恒星、事实→边，统计与节点自洽，可截断但不骗计数。"""

    manager = drift_pipeline.manager
    _ingest(drift_pipeline, "Python 的函数定义与变量作用域。" * 8, "doc-graph")
    manager.semantic.add_fact("星云", "部署于", "本机", confidence=0.9)

    output = GraphSnapshotTool(manager=manager).execute(GraphSnapshotInput())

    kinds = {node["kind"] for node in output.nodes}
    assert "chunk" in kinds
    assert "domain" in kinds
    assert output.stats.chunks >= 1
    assert output.stats.domains >= 1
    assert output.stats.edges == len(output.edges)
    assert output.stats.total == len(output.nodes)
    assert output.as_of == ""
    assert output.truncated is False
    # 领域配色必须跨进程稳定（crc32，不是内建 hash）
    assert domain_color("编程开发") == domain_color("编程开发")

    # 只要统计时节点清空但 stats 仍是全量，且如实标注截断
    stats_only = GraphSnapshotTool(manager=manager).execute(
        GraphSnapshotInput(include_nodes=False)
    )
    assert stats_only.nodes == []
    assert stats_only.edges == []
    assert stats_only.truncated is True
    assert stats_only.stats.total == output.stats.total

    # max_nodes 只截断节点/边，不改统计口径
    capped = GraphSnapshotTool(manager=manager).execute(GraphSnapshotInput(max_nodes=1))
    assert capped.truncated is True
    assert len(capped.nodes) == 1
    assert capped.stats.total == output.stats.total
    # Web 端仍直接调用纯函数，返回结构不变（revision/unchanged 由 app.py 追加）
    assert set(build_graph(manager)) == {"graph_source", "as_of", "stats", "nodes", "edges"}


def test_orphan_entities_tool_matches_the_cleanup_criteria(manager) -> None:
    """孤儿检测工具：只有「完全孤立」的实体才算，且工具只读不删。"""

    manager.add("尘埃", memory_type="semantic", metadata={"kind": "entity", "title": "尘埃"})
    manager.add("恒星", memory_type="semantic", metadata={"kind": "entity", "title": "恒星"})
    manager.semantic.add_fact("恒星", "照亮", "行星", confidence=0.9)

    output = OrphanEntitiesTool(manager=manager).execute(OrphanEntitiesInput())

    assert output.count == 1
    assert [entity.name for entity in output.entities] == ["尘埃"]
    assert len(find_orphan_entities(manager)) == 1
    # 只读：实体还在
    assert manager.get(output.entities[0].id) is not None


def test_document_tools_read_the_truth_source(drift_pipeline) -> None:
    """文档列表/详情：读真值源；分页上限与缺失文档都如实报错，截断不骗计数。"""

    _ingest(drift_pipeline, "设备编号 abc-123 的文档中心基线。" * 8, "doc-center")
    manager = drift_pipeline.manager
    repository = repository_for(manager)
    assert repository is not None

    listing = DocumentListTool(manager=manager).execute(DocumentListInput(page_size=10))
    assert listing.total == 1
    assert listing.page == 1
    assert [item.document_id for item in listing.items] == ["doc-center"]
    assert listing.items[0].chunk_count >= 1
    assert listing.items[0].status

    detail = DocumentGetTool(manager=manager).execute(
        DocumentGetInput(document_id="doc-center")
    )
    assert detail.chunk_count == len(detail.chunks) == listing.items[0].chunk_count
    assert detail.raw_text
    assert all(chunk.vector_status == "indexed" for chunk in detail.chunks)
    assert detail.truncated is False

    # max_chunks 只裁剪列表，chunk_count 始终是全量
    capped = DocumentGetTool(manager=manager).execute(
        DocumentGetInput(document_id="doc-center", max_chunks=1)
    )
    assert capped.chunk_count == detail.chunk_count
    assert len(capped.chunks) == 1
    assert capped.truncated is (detail.chunk_count > 1)

    with pytest.raises(LookupError, match="文档不存在"):
        get_document(repository, "不存在的文档")
    with pytest.raises(ValueError, match="page_size"):
        list_documents(repository, page_size=WEB_DOCUMENTS_PAGE_SIZE_MAX + 1)


def test_document_revectorize_tool_only_touches_that_document(drift_pipeline) -> None:
    """重嵌入：只重灌指定文档的分块，另一篇一条不动；缺文档/缺分块都如实报错。"""

    _ingest(drift_pipeline, "设备编号 abc-123 的重嵌入基线。" * 8, "doc-revec")
    _ingest(drift_pipeline, "另一篇文档，用于验证互不影响。" * 8, "doc-other")
    manager = drift_pipeline.manager
    repository = repository_for(manager)
    assert repository is not None
    other_before = [chunk.chunk_id for chunk in repository.list_chunks("doc-other")]

    # 模拟投影落后：把目标文档的分块标回待投影
    for chunk in repository.list_chunks("doc-revec"):
        repository.set_chunk_vector_status(chunk.chunk_id, "pending")

    output = DocumentRevectorizeTool(manager=manager).execute(
        DocumentRevectorizeInput(document_id="doc-revec")
    )

    assert output.document_id == "doc-revec"
    assert output.chunks_reindexed == len(repository.list_chunks("doc-revec"))
    assert output.status == "vectorized"
    assert all(
        chunk.vector_status == "indexed" for chunk in repository.list_chunks("doc-revec")
    )
    # 另一篇文档一条不动
    assert [chunk.chunk_id for chunk in repository.list_chunks("doc-other")] == other_before
    assert all(
        chunk.vector_status == "indexed" for chunk in repository.list_chunks("doc-other")
    )

    with pytest.raises(LookupError, match="文档不存在"):
        revectorize_document(manager, repository, "不存在")

    # 没有分块的文档必须明确拒绝（422 语义），而不是静默成功
    repository.upsert_document(
        DocumentRecord(document_id="doc-empty", raw_text="正文", source="e.txt")
    )
    with pytest.raises(ValueError, match="没有分块"):
        revectorize_document(manager, repository, "doc-empty")


def test_ingest_image_tool_stores_payload_and_reads_the_file(manager, tmp_path, monkeypatch) -> None:
    """图片入库：按路径读文件 → 感知记忆留下原始字节 → 视觉抽取器拿到图片与媒体类型。"""

    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    image_path = tmp_path / "camera.png"
    image_path.write_bytes(PNG_BYTES)
    seen: list[tuple[Any, ...]] = []

    class Extractor:
        def extract(self, text, *, metadata=None, graph_context="", image=None, mime_type=""):
            seen.append((text, metadata, image, mime_type))
            return ExtractionResult()

    pipeline = RAGPipeline(manager, extractor=Extractor())
    tool = IngestImageTool(pipeline=pipeline)

    output = tool.execute(
        IngestImageInput(
            path=str(image_path),
            text="电脑在书桌上",
            captured_at="2025-01-01T13:00:00+00:00",
        )
    )

    item = manager.get(output.item_id)
    assert item is not None
    assert item.payload == PNG_BYTES
    assert item.modality == "image"
    assert output.modality == "image"
    assert output.extraction.modality == "image"
    # 媒体类型按扩展名推断（调用方没填 mime_type）
    assert seen[0][3] == "image/png"
    assert seen[0][2] == PNG_BYTES
    assert pipeline.last_ingest_report["modality"] == "image"
    # 非 VL 嵌入必须诚实提示，而不是假装多模态
    assert output.extraction.multimodal_embedding is False
    assert output.warning == TEXT_ONLY_WARNING

    # 参数与文件错误如实报错，不静默成功
    with pytest.raises(LookupError, match="图片文件不存在"):
        tool.execute(IngestImageInput(path=str(tmp_path / "nope.png")))
    bad = tmp_path / "note.txt"
    bad.write_text("不是图片", encoding="utf-8")
    with pytest.raises(ValueError, match="image media type"):
        tool.execute(IngestImageInput(path=str(bad), mime_type="text/plain"))


def test_add_fact_tool_writes_the_truth_row_and_the_graph_edge(manager) -> None:
    """事实写入：三元组进真值源，图里同步长出同 id 的边；领域留空落兜底值。"""

    output = AddFactTool(manager=manager).execute(
        AddFactInput(
            subject="星云", predicate="部署于", object="本机", domain="项目", confidence=0.8
        )
    )

    item = manager.get(output.item_id)
    assert item is not None
    assert item.metadata["subject"] == "星云"
    assert item.metadata["object"] == "本机"
    assert item.metadata["domain"] == "项目"
    assert manager.graph_store.relation_memory_ids() == [output.item_id]
    assert output.confidence == 0.8

    # 领域/备注留空时与旧端点一致：落到兜底领域，置信度默认 1.0
    fallback = AddFactTool(manager=manager).execute(
        AddFactInput(subject="甲", predicate="是", object="乙")
    )
    assert manager.get(fallback.item_id).metadata["domain"] == "未分类"
    assert fallback.confidence == 1.0
    assert add_fact(manager, subject="丙", predicate="是", object="丁").metadata["domain"] == "未分类"


def test_seed_tool_is_idempotent_and_its_rows_are_not_orphans(drift_pipeline, tmp_path) -> None:
    """播种：写入实体/关系/备注并打 seed 标记；第二次幂等跳过；seed 实体不算孤儿。"""

    seed_file = tmp_path / "seed.json"
    seed_file.write_text(
        json.dumps(
            {
                "entities": [{"name": "星云", "domain": "项目"}, {"name": "尘埃"}],
                "relations": [
                    {"subject": "星云", "predicate": "部署于", "object": "本机", "confidence": 0.9}
                ],
                "notes": [{"content": "一条备注", "entity": "星云", "title": "档案"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = drift_pipeline.manager
    tool = SeedKnowledgeTool(manager=manager)

    first = tool.execute(SeedKnowledgeInput(path=str(seed_file)))
    assert first.seeded is True
    assert (first.entities, first.relations, first.notes) == (2, 1, 1)
    assert first.source == str(seed_file)

    # 幂等：第二次全部跳过，并回报库内已有条目数
    second = tool.execute(SeedKnowledgeInput(path=str(seed_file)))
    assert second.seeded is False
    assert "已播种过" in second.reason
    assert second.existing >= 1

    # 缺文件时如实说明原因，而不是静默成功
    missing = tool.execute(SeedKnowledgeInput(path=str(tmp_path / "nope.json")))
    assert missing.seeded is False
    assert "种子文件不存在" in missing.reason

    # 跨工具契约：seed 实体即使完全孤立也不能被当成垃圾候选
    seeded_entities = {
        item.metadata.get("title")
        for item in manager.list(memory_type="semantic")
        if item.metadata.get("seed") == SEED_MARK and item.metadata.get("kind") == "entity"
    }
    assert seeded_entities == {"星云", "尘埃"}
    # 「尘埃」没有任何关系/提及/备注，仅靠 seed 标记逃过孤儿判定
    orphans = OrphanEntitiesTool(manager=manager).execute(OrphanEntitiesInput())
    assert orphans.count == 0
    assert seed(manager, seed_file)["seeded"] is False


def test_stats_tool_matches_the_three_store_truth(drift_pipeline) -> None:
    """统计工具：计数口径必须与真值源/记忆层一致，且只给计数不给内容。"""

    manager = drift_pipeline.manager
    tool = KnowledgeStatsTool(manager=manager)
    empty = tool.execute(KnowledgeStatsInput())
    assert (empty.documents, empty.chunks, empty.chunks_indexed, empty.facts) == (0, 0, 0, 0)

    # 走真实入库路径（建 documents 行 + 分块 + 向量投影），统计口径才有的可数
    _ingest(drift_pipeline, "统计口径：一段用于计数的正文。" * 6, "doc-stats")
    AddFactTool(manager=manager).execute(
        AddFactInput(subject="星", predicate="属于", object="云")
    )

    stats = tool.execute(KnowledgeStatsInput())
    repository = repository_for(manager)
    assert repository is not None
    counts = repository.stats()
    assert stats.documents == counts["documents"] == 1
    assert stats.chunks == counts["chunks"] >= 1
    assert stats.chunks_indexed == counts["chunks_indexed"] == stats.chunks
    assert stats.facts == 1
    assert stats.memories_total == len(manager.document_store.list(include_expired=True))
    # 只给计数：输出模型里没有任何内容正文字段
    assert "统计口径" not in stats.model_dump_json()

    # 函数式入口与工具入口同源（web 的 /api/stats 走的就是这个函数）
    assert knowledge_stats(manager, repository) == stats
    assert knowledge_stats(manager, None).documents == 0

    spec = tool.spec
    assert (spec.side_effect, spec.idempotent, spec.parallel_safe) == ("read", True, True)
    assert spec.guidance.strip()
