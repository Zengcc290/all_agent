"""web.domain_classifier 自动领域分类单测 + 图构建集成验证。"""
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager
from memory.rag import Document, RAGPipeline
from web.domain_classifier import (
    DEFAULT,
    DOMAIN_KEYWORDS,
    KNOWN_DOMAINS,
    classify_domain,
    majority_domain,
)
from web.graph_builder import build_graph


def test_known_domains_nonempty() -> None:
    assert len(KNOWN_DOMAINS) >= 8
    assert DEFAULT not in KNOWN_DOMAINS


def test_keywords_nonempty() -> None:
    for domain, kws in DOMAIN_KEYWORDS.items():
        assert kws, f"{domain} 关键词表为空"


def test_classify_programming_text() -> None:
    assert classify_domain("Python 的函数定义与变量作用域，调试时打印堆栈") == "编程开发"


def test_classify_math_text() -> None:
    assert classify_domain("微分方程与矩阵特征值在概率统计中的应用") == "数学"


def test_classify_medicine_text() -> None:
    assert classify_domain("病毒的基因序列与疫苗免疫应答机制") == "生物医学"


def test_title_weighted_above_text() -> None:
    # 标题命中加权：正文无关键词、文件名含主题词也应归类
    assert classify_domain("第一章 概述", title="c语言笔记.txt") == "编程开发"


def test_title_and_text_mismatch_uses_text() -> None:
    # 正文命中多个主题词（牛顿/力学/电磁/波动 = 4 分）vs 标题权重 3 → 正文胜
    assert classify_domain("牛顿力学与电磁场波动", title="数学课.txt") == "物理"


def test_unmatched_returns_default() -> None:
    assert classify_domain("今天天气很好，出门散步") == DEFAULT
    assert classify_domain("") == DEFAULT
    assert classify_domain("", title="") == DEFAULT


def test_deterministic() -> None:
    text = "数据库的索引与查询优化，前端调用 API 接口"
    assert classify_domain(text) == classify_domain(text)


def test_majority_domain() -> None:
    assert majority_domain(["编程开发", "编程开发", "数学"]) == "编程开发"
    assert majority_domain([]) == DEFAULT
    assert majority_domain(["数学"]) == "数学"


def test_known_domains_cover_major_topics() -> None:
    required = {"编程开发", "数学", "物理", "化学", "生物医学", "历史人文", "经济管理", "文学艺术"}
    assert required <= set(KNOWN_DOMAINS)


def test_ingested_docs_auto_classify_to_different_domains() -> None:
    """图构建：不同主题的文档自动落入不同恒星系，而非统一「文档库」。"""
    manager = MemoryManager(
        MemoryConfig(sqlite_path=":memory:"), embedding=HashEmbedding()
    )
    pipe = RAGPipeline(manager)
    try:
        topics = {
            "编程笔记": "Python 函数与变量，前端调用 API 接口，数据库索引查询" * 10,
            "高数笔记": "微积分方程与矩阵特征值，概率统计与导数应用" * 10,
            "历史笔记": "唐朝与宋朝的战争与王朝更迭，古代文明遗址考古" * 10,
        }
        for title, text in topics.items():
            pipe.ingest(
                Document(text, id=title, metadata={"source": f"{title}.txt", "filename": f"{title}.txt"}),
                chunk_size=40,
                overlap=5,
            )

        graph = build_graph(manager)
        doc_nodes = {
            n["title"]: n["domain"]
            for n in graph["nodes"]
            if n["kind"] == "entity" and n["title"].startswith("文档：")
        }
        # 三份文档落到三个不同主题恒星系
        assert doc_nodes["文档：编程笔记.txt"] == "编程开发"
        assert doc_nodes["文档：高数笔记.txt"] == "数学"
        assert doc_nodes["文档：历史笔记.txt"] == "历史人文"
        # 没有文档被扔进「文档库」
        assert "文档库" not in {n["domain"] for n in graph["nodes"]}
    finally:
        manager.close()
