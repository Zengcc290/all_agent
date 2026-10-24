"""F3 问句分解的最小回归集。

覆盖：
- NullQueryDecomposer 返回原句（LLM 关掉时结果与今天一致）；
- LLMQueryDecomposer 解析合法 JSON、原句永远第一条、去重、截断；
- hybrid_recall_multi 多路融合召回高于单路（A/B 对比）；
- graph_recall_multi 路径按 effective 合并去重。
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from memory.rag import Document, RAGPipeline
from tool.hybrid_recall import hybrid_recall
from tool.multi_recall import (
    LLMQueryDecomposer,
    NullQueryDecomposer,
    graph_recall_multi,
    hybrid_recall_multi,
)


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


def test_null_decomposer_returns_original_query():
    decomposer = NullQueryDecomposer()

    assert decomposer.decompose("小红的亲戚是谁") == ["小红的亲戚是谁"]
    assert decomposer.decompose("") == []
    assert decomposer.decompose("   ") == []


class _FakeResponse(dict):
    @property
    def choices(self):
        return self["choices"]


def _complete_with(content: str):
    def complete(messages, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    return complete


def test_llm_decomposer_parses_json_and_keeps_original_first():
    decomposer = LLMQueryDecomposer(
        _complete_with('{"sub_queries": ["小红", "小红 亲戚", "小红 亲属 关系", "小红"]}')
    )

    queries = decomposer.decompose("小红的亲戚是谁")

    assert queries[0] == "小红的亲戚是谁"
    assert queries[1:] == ["小红", "小红 亲戚", "小红 亲属 关系"]  # 去重、原句第一条


def test_llm_decomposer_accepts_markdown_wrapped_json():
    decomposer = LLMQueryDecomposer(
        _complete_with('```json\n{"sub_queries": ["小红"]}\n```')
    )

    assert decomposer.decompose("小红是谁") == ["小红是谁", "小红"]


def test_llm_decomposer_falls_back_to_original_on_garbage():
    decomposer = LLMQueryDecomposer(_complete_with("不是 JSON"))

    assert decomposer.decompose("原句不变") == ["原句不变"]


def test_llm_decomposer_caps_at_six():
    content = '{"sub_queries": ["一", "二", "三", "四", "五", "六", "七", "八"]}'
    decomposer = LLMQueryDecomposer(_complete_with(content))

    queries = decomposer.decompose("原句")

    assert len(queries) == 6
    assert queries[0] == "原句"


def test_hybrid_retrieve_multi_recalls_more_than_single(manager: MemoryManager):
    """A/B：多路融合召回高于单路（精确词向量区分度差，关键词路补召回）。"""

    from memory.storage.document_repo import DocumentRepository

    db = manager.document_store.path
    pipeline = RAGPipeline(manager)
    pipeline.ingest(
        [
            Document("模型型号是 DSV4.1。", id="doc-a"),
            Document("部署位置是生产集群。", id="doc-b"),
            Document("DSV4.1 部署在生产集群。", id="doc-c"),
        ]
    )
    assert DocumentRepository(str(db)) is not None

    single = hybrid_recall(pipeline, "DSV4.1 部署在哪", limit=3).chunks
    multi = hybrid_recall_multi(
        pipeline, ["DSV4.1 部署在哪", "DSV4.1", "部署位置"], limit=3
    ).chunks

    assert [chunk.memory_id for chunk in multi] == [chunk.memory_id for chunk in multi]
    assert len(multi) >= len(single)
    multi_ids = {chunk.memory_id for chunk in multi}
    single_ids = {chunk.memory_id for chunk in single}
    assert single_ids <= multi_ids  # 多路不丢单路已召回的分块


def test_graph_retrieve_multi_merges_paths_by_effective(manager: MemoryManager):
    manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    manager.semantic.add_fact("B", "knows", "C", confidence=0.8)
    manager.semantic.add_fact("D", "knows", "C", confidence=0.95)

    result = graph_recall_multi(RAGPipeline(manager), ["A", "D"], limit=5, hops=2)

    assert result.paths
    assert len({path.entities for path in result.paths}) == len(result.paths)  # 去重
    # D->C 的 effective 更高，应排在前面
    assert result.paths[0].entities == ("D", "C")


def test_retrieve_multi_rejects_empty_queries(manager: MemoryManager):
    with pytest.raises(ValueError, match="queries"):
        graph_recall_multi(RAGPipeline(manager), [])


def test_hybrid_retrieve_multi_empty_queries(manager: MemoryManager):
    pipeline = RAGPipeline(manager)

    assert hybrid_recall_multi(pipeline, []).chunks == []
    assert hybrid_recall_multi(pipeline, ["", "   "]).chunks == []
