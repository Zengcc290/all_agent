"""web 层 API 测试：内存库 + FastAPI TestClient 端到端。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memory import MemoryConfig, MemoryManager  # noqa: E402
from web import create_app  # noqa: E402
from web.support import HashEmbedding  # noqa: E402
from memory.rag import EntityCandidate, ExtractionResult, RelationCandidate  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("WEB_AUTOSEED", "0")  # 测试显式控制播种
    manager = MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        embedding=HashEmbedding(),  # 离线确定性嵌入，测试无需任何 API key
    )
    app = create_app(manager=manager)
    app.state.manager = manager
    app.state.pipeline = None  # 延迟到首次使用
    with TestClient(app) as test_client:
        yield test_client
    manager.close()


def _make_ingest_payload(filename: str, text: str) -> dict:
    return {"file": (filename, io.BytesIO(text.encode("utf-8")), "text/plain")}


def test_graph_empty_then_seeded(client: TestClient) -> None:
    graph = client.get("/api/graph").json()
    # 空库也有内置的「时间线」恒星与实体，但没有任何边和事实
    assert graph["edges"] == []
    assert not [
        n for n in graph["nodes"] if n["kind"] in ("fact", "chunk", "note", "event")
    ]

    seeded = client.post("/api/seed").json()
    assert seeded["seeded"] is True

    graph = client.get("/api/graph").json()
    kinds = {node["kind"] for node in graph["nodes"]}
    assert {"domain", "entity", "fact", "note"} <= kinds
    assert graph["stats"]["edges"] >= 10
    # 幂等：再种一次不再增长
    again = client.post("/api/seed").json()
    assert again["seeded"] is False
    stats2 = client.get("/api/graph").json()["stats"]
    assert stats2 == graph["stats"]


def test_facts_add_and_appear(client: TestClient) -> None:
    body = {
        "subject": "Qdrant",
        "predicate": "支持",
        "object": "本地嵌入式模式",
        "domain": "Technology",
        "note": "测试事实",
    }
    response = client.post("/api/facts", json=body)
    assert response.status_code == 200
    assert response.json()["ok"] is True

    graph = client.get("/api/graph").json()
    subjects = {node["title"] for node in graph["nodes"] if node["kind"] == "entity"}
    assert "Qdrant" in subjects
    assert any(edge["relation"] == "支持" for edge in graph["edges"])


def test_ingest_grows_nebula(client: TestClient) -> None:
    text = "知识星云把领域映射为恒星、实体映射为行星。" * 8
    response = client.post("/api/ingest", files=_make_ingest_payload("note.txt", text))
    assert response.status_code == 200
    assert response.json()["chunks"] >= 1

    graph = client.get("/api/graph").json()
    kinds = {}
    for node in graph["nodes"]:
        kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
    assert kinds.get("chunk", 0) >= 1
    assert kinds.get("domain", 0) >= 1
    assert any("note.txt" in (node.get("source") or "") for node in graph["nodes"])


def test_ingest_rejects_empty_file(client: TestClient) -> None:
    response = client.post("/api/ingest", files=_make_ingest_payload("empty.txt", ""))
    assert response.status_code == 400


def test_graph_rag_endpoint_returns_vector_evidence_and_paths(
    client: TestClient,
) -> None:
    from memory.rag import RAGPipeline

    class Extractor:
        def extract(self, text: str, *, metadata=None) -> ExtractionResult:
            return ExtractionResult(
                domain="测试",
                entities=[EntityCandidate(name="A"), EntityCandidate(name="B")],
                relations=[
                    RelationCandidate(
                        subject="A", predicate="关联", object="B", confidence=0.9
                    )
                ],
            )

    client.app.state.pipeline = RAGPipeline(
        client.app.state.manager, extractor=Extractor()
    )
    client.post("/api/ingest", files=_make_ingest_payload("graph.txt", "A关联B"))
    response = client.post("/api/graph-rag", json={"query": "A", "hops": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence"]
    assert any(path["target"] == "B" for path in payload["paths"])


def test_export_import_roundtrip_idempotent(client: TestClient) -> None:
    client.post("/api/facts", json={"subject": "A", "predicate": "关联", "object": "B"})

    export = client.get("/api/export")
    assert export.status_code == 200
    payload = export.json()
    assert payload["counts"]["total"] >= 1  # 一个三元组事实 = 一条记忆
    assert "attachment" in export.headers["content-disposition"]

    # 全量回导：既有的 item_id/三元组应全部跳过
    files = {
        "file": (
            "export.json",
            io.BytesIO(json.dumps(payload).encode("utf-8")),
            "application/json",
        )
    }
    imported = client.post("/api/import", files=files).json()
    assert imported["imported"] == 0
    assert imported["skipped"] == payload["counts"]["total"]

    stats_before = client.get("/api/graph").json()["stats"]
    assert stats_before["entities"] >= 3  # A、B、事件时间线


def test_chat_disabled_without_provider(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 503
    assert "provider" in response.json()["detail"]


def test_knowledge_sentence_ingests_and_extracts(client: TestClient) -> None:
    from memory.rag import RAGPipeline

    class Extractor:
        def extract(self, text: str, *, metadata=None) -> ExtractionResult:
            return ExtractionResult(
                domain="Technology",
                entities=[EntityCandidate(name="向量数据库"), EntityCandidate(name="语义检索")],
                relations=[RelationCandidate(subject="向量数据库", predicate="用于", object="语义检索", confidence=0.9)],
            )

    client.app.state.pipeline = RAGPipeline(client.app.state.manager, extractor=Extractor())

    response = client.post(
        "/api/knowledge",
        json={"text": "向量数据库把非结构化文本编码成稠密向量，用于语义检索。"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["chunks"] >= 1
    assert payload["extraction"]["entities"] >= 2
    assert payload["extraction"]["relations"] >= 1

    graph = client.get("/api/graph").json()
    assert any(edge["relation"] == "用于" for edge in graph["edges"])
    assert any(node["kind"] == "chunk" for node in graph["nodes"])


def test_knowledge_sentence_requires_text(client: TestClient) -> None:
    response = client.post("/api/knowledge", json={"text": "   "})
    assert response.status_code == 422


def test_health(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert "chat_ready" in health and "embedding_mode" in health


def test_graph_cache_invalidates_on_writes(client: TestClient) -> None:
    """/api/graph 结果缓存：未变化时命中缓存，写操作后正确失效重建。"""
    import time

    client.post("/api/seed")

    # 冷构建（或缓存未命中）应较慢，命中缓存应显著更快
    def timed_get() -> tuple[float, dict]:
        t0 = time.perf_counter()
        payload = client.get("/api/graph").json()
        return (time.perf_counter() - t0) * 1000, payload

    cold_ms, first = timed_get()
    warm_ms, second = timed_get()
    assert first == second  # 数据未变，负载必须一致
    assert warm_ms < max(30.0, cold_ms)  # 缓存命中应显著更快

    # 写入新事实 → 图必须反映新数据（缓存失效）
    client.post("/api/facts", json={"subject": "缓存验证", "predicate": "使", "object": "缓存失效"})
    _, after = timed_get()
    titles = {node["title"] for node in after["nodes"] if node["kind"] == "entity"}
    assert "缓存验证" in titles
    assert after != second
