"""web 层 API 测试：内存库 + FastAPI TestClient 端到端。"""

from __future__ import annotations

import io
import json
import socket
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memory import InMemoryVectorStore, MemoryConfig, MemoryManager  # noqa: E402
from memory.rag import EntityCandidate, ExtractionResult, RelationCandidate  # noqa: E402
from memory.storage import ChunkRecord, DocumentRecord, DocumentRepository  # noqa: E402
from web import create_app, support  # noqa: E402
from web.support import HashEmbedding  # noqa: E402


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


class DriftVectorStore(InMemoryVectorStore):
    """内存向量库 + 可枚举 id：对账接口需要，且能人为制造缺向量的漂移。"""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def upsert(self, item) -> None:
        super().upsert(item)
        self.ids.add(item.id)

    def upsert_chunk(self, chunk_id, vector, **kwargs) -> None:
        self.ids.add(chunk_id)

    def list_ids(self) -> list[str]:
        return sorted(self.ids)


@pytest.fixture()
def file_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """文件库客户端：documents/chunks 真值源只在文件型 SQLite 上存在。"""

    monkeypatch.setenv("WEB_AUTOSEED", "0")
    monkeypatch.setenv("EMBEDDING_TUNNEL_HINT", "")   # 断言固定，不受本机 .env 影响
    store = DriftVectorStore()
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
        vector_store=store,
    )
    app = create_app(manager=manager)
    with TestClient(app) as test_client:
        yield test_client, store
    manager.close()


def _ingest_documents(client: TestClient, names: list[str]) -> list[str]:
    """上传若干文档，返回它们的 document_id（按创建顺序）。"""

    for name in names:
        response = client.post(
            "/api/ingest",
            files=_make_ingest_payload(f"{name}.txt", f"{name} 的内容，讲的是混合检索与向量库。" * 8),
        )
        assert response.status_code == 200, response.text
    listing = client.get("/api/documents", params={"page_size": 50}).json()
    return [item["document_id"] for item in listing["items"]]


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
    # 回归：错误提示必须是可读中文，不得是编码损坏的问号串
    detail = response.json()["detail"]
    assert "空" in detail
    assert "?" not in detail


def test_ingest_rejects_oversized_file(client: TestClient) -> None:
    from constants import MAX_UPLOAD_BYTES

    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/api/ingest",
        files={"file": ("big.txt", io.BytesIO(oversized), "text/plain")},
    )
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "上限" in detail
    assert "?" not in detail


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


def test_import_rejects_oversized_file(client: TestClient) -> None:
    """回归：/api/import 曾把整个请求体读进内存且无大小限制。"""

    from constants import MAX_UPLOAD_BYTES

    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
    files = {
        "file": ("huge.json", io.BytesIO(oversized), "application/json"),
    }
    response = client.post("/api/import", files=files)
    assert response.status_code == 413
    assert "上限" in response.json()["detail"]


def test_import_reports_why_items_were_skipped(client: TestClient) -> None:
    """回归：导入失败曾被静默吞掉，只留下一个 skipped 计数。"""

    payload = {
        "format": "knowledge-nebula-export/v1",
        "items": [
            {"id": "ok-1", "content": "A 关联 B", "memory_type": "episodic"},
            {"id": "no-content"},
            "not-an-object",
        ],
    }
    files = {
        "file": (
            "partial.json",
            io.BytesIO(json.dumps(payload).encode("utf-8")),
            "application/json",
        )
    }
    result = client.post("/api/import", files=files).json()

    assert result["imported"] == 1
    assert result["skipped"] == 2
    assert len(result["errors"]) == 2
    assert any("no-content" in message for message in result["errors"])
    assert any("不是 JSON 对象" in message for message in result["errors"])


def test_import_error_list_is_bounded(client: TestClient) -> None:
    from constants import WEB_IMPORT_ERRORS_MAX

    payload = {"items": [{"id": ""} for _ in range(WEB_IMPORT_ERRORS_MAX + 5)]}
    files = {
        "file": (
            "many.json",
            io.BytesIO(json.dumps(payload).encode("utf-8")),
            "application/json",
        )
    }
    result = client.post("/api/import", files=files).json()

    assert result["skipped"] == WEB_IMPORT_ERRORS_MAX + 5
    # 明细有上限，避免超长响应；计数仍然完整。
    assert len(result["errors"]) == WEB_IMPORT_ERRORS_MAX


def test_ingest_still_rejects_oversized_file(client: TestClient) -> None:
    """共享上传辅助后，/api/ingest 的行为必须保持。"""

    from constants import MAX_UPLOAD_BYTES

    response = client.post(
        "/api/ingest",
        files=_make_ingest_payload("big.txt", "x" * (MAX_UPLOAD_BYTES + 1)),
    )
    assert response.status_code == 413


def test_chat_disabled_without_provider(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 503
    assert "provider" in response.json()["detail"]


def test_chat_rejects_unknown_mode(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "你好", "mode": "quantum"})
    assert response.status_code == 422


def _force_search_env(monkeypatch: pytest.MonkeyPatch, *, on: bool) -> None:
    names = [
        "SEARCH_BASE_URL", "ANYSEARCH_BASE_URL",
        "SEARCH_API", "SEARCH_API_KEY", "ANYSEARCH_API_KEY",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    if on:
        monkeypatch.setenv("SEARCH_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("SEARCH_API", "test-search-key")


def _make_fake_agent():
    class FakeTools:
        def snapshot(self):
            return {
                "memory.query": None,
                "memory.add": None,
                "memory.rag_search": None,
                "memory.rag": None,
                "system.current_time": None,
                "web.search": None,
            }

        def confirmation_key(self, name: str) -> str:
            return f"{name}:test-generation"

    class FakeAgent:
        tools = FakeTools()

        def __init__(self) -> None:
            self.last_tool_names = None
            self.last_context = None

        def run(self, query: str, **kwargs):
            self.last_tool_names = kwargs.get("tool_names")
            self.last_context = kwargs.get("context")
            return "这是测试回答"

    return FakeAgent()


def test_chat_records_qa_into_episodic_memory(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_search_env(monkeypatch, on=False)
    agent = _make_fake_agent()
    monkeypatch.setattr("web.app.chat_ready", lambda: (True, ""))
    monkeypatch.setattr("web.app.get_agent", lambda: agent)

    response = client.post(
        "/api/chat", json={"message": "我这两天在忙什么", "mode": "offline"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "这是测试回答"
    assert payload["mode"] == "offline"

    items = client.app.state.manager.list(memory_type="episodic")
    qa_items = [item for item in items if item.metadata.get("kind") == "qa"]
    assert len(qa_items) == 1
    meta = qa_items[0].metadata
    assert meta["mode"] == "offline"
    assert meta["question"] == "我这两天在忙什么"
    assert "asked_at" in meta
    assert qa_items[0].timestamp is not None

    # 端到端：模拟 agent 用 memory.query 检索（不指定 memory_type 应能命中）
    from tool.memory_query import MemoryQueryInput, MemoryQueryTool

    tool = MemoryQueryTool(manager=client.app.state.manager)
    found = tool.execute(MemoryQueryInput(action="search", query="我这两天在忙什么"))
    assert found.count >= 1
    assert any("这是测试回答" in item["content"] for item in found.items)


def test_chat_tool_names_follow_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _make_fake_agent()
    monkeypatch.setattr("web.app.chat_ready", lambda: (True, ""))
    monkeypatch.setattr("web.app.get_agent", lambda: agent)

    # 非联网：工具清单里不得有 web.search
    _force_search_env(monkeypatch, on=False)
    client.post("/api/chat", json={"message": "q1", "mode": "offline"})
    assert agent.last_tool_names is not None
    assert "web.search" not in agent.last_tool_names

    # 已配置搜索且选择联网：tool_names 为 None（全部工具，含 web.search）
    _force_search_env(monkeypatch, on=True)
    client.post("/api/chat", json={"message": "q2", "mode": "online"})
    assert agent.last_tool_names is None

    # 未配置搜索却选择联网：回退为非联网
    _force_search_env(monkeypatch, on=False)
    response = client.post("/api/chat", json={"message": "q3", "mode": "online"})
    assert response.json()["mode"] == "offline"
    assert agent.last_tool_names is not None
    assert "web.search" not in agent.last_tool_names


def test_chat_confirms_only_additive_memory_write(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """聊天回合只为 memory.add 预置写确认。

    只读工具本就不需要确认（A1 拆分后）；delete/clear/ingest 属于破坏性或外部
    写入，聊天层不得代用户授权，因此确认集合里必须只有 memory.add。
    """
    _force_search_env(monkeypatch, on=False)
    agent = _make_fake_agent()
    monkeypatch.setattr("web.app.chat_ready", lambda: (True, ""))
    monkeypatch.setattr("web.app.get_agent", lambda: agent)

    response = client.post("/api/chat", json={"message": "记住我偏好浅色主题"})
    assert response.status_code == 200

    context = agent.last_context
    assert context is not None
    assert context.confirmed_side_effects == frozenset({"memory.add:test-generation"})


def test_chat_serializes_concurrent_requests(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """并发问答必须排队：agent 是共享单例，其对话历史无内部锁。

    未串行化时第二个请求会在第一个尚未结束时进入 agent.run，两者读写同一份
    历史并互相污染。
    """
    import threading
    import time

    entered = threading.Event()
    release = threading.Event()
    entered_order: list[str] = []
    order_lock = threading.Lock()

    class BlockingAgent:
        class _Tools:
            def snapshot(self):
                return {"memory.rag": None, "memory.manage": None}

        tools = _Tools()

        def run(self, query: str, **kwargs):
            with order_lock:
                entered_order.append(query)
            entered.set()
            release.wait(timeout=10)
            return f"answer:{query}"

    monkeypatch.setattr("web.app.chat_ready", lambda: (True, ""))
    monkeypatch.setattr("web.app.get_agent", lambda: BlockingAgent())

    results: list[int] = []

    def call(message: str) -> None:
        results.append(client.post("/api/chat", json={"message": message}).status_code)

    first = threading.Thread(target=call, args=("q1",))
    first.start()
    assert entered.wait(timeout=10), "第一个请求未进入 agent.run"

    second = threading.Thread(target=call, args=("q2",))
    second.start()
    time.sleep(0.3)  # 给第二个请求足够机会进入（若未串行化则会进入）

    assert entered_order == ["q1"], "第二个请求与第一个并发进入了 agent.run"

    release.set()
    first.join(timeout=15)
    second.join(timeout=15)

    assert sorted(results) == [200, 200]
    assert entered_order == ["q1", "q2"]


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
    # 回归：模式必须按实际生效的嵌入实现报告，而不是只看 DASHSCOPE_API_KEY
    #（注入的测试 embedding 不是 APIEmbedding，因此这里必须是 local-hash）。
    assert health["embedding_mode"] == "local-hash"
    assert health["embedding"]["type"] == "HashEmbedding"
    assert health["embedding"]["dimension"] == client.app.state.manager.embedding.dimension
    assert "search_available" in health


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


def test_get_agent_registers_the_four_memory_tools(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回归：web 知识管家的真实装配路径必须能跑通。

    历史缺陷：装配代码写成 ``RagSearchTool(...)``，而 ``tool/rag_search.py``
    导出的类名是 ``RAGSearchTool``（ruff F821 才暴露），于是 get_agent() 抛
    NameError，Web 聊天在真实运行中直接 500。此前的测试都用 FakeAgent 替换了
    get_agent，所以没有任何用例覆盖这段真实装配代码。
    """

    from web import support

    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "agent-tools.sqlite3"))
    monkeypatch.setattr(support, "DB_PATH", tmp_path / "agent-tools.sqlite3")
    monkeypatch.setattr(support, "_manager", None)
    monkeypatch.setattr(support, "_pipeline", None)
    monkeypatch.setattr(support, "_agent", None)
    try:
        agent = support.get_agent()
        names = {spec.name for spec in agent.tools.specs()}
        assert {
            "memory.query",
            "memory.add",
            "memory.rag_search",
            "memory.rag",
        } <= names
        # 只读工具不该要求确认，memory.add 才是聊天唯一自动确认的写入。
        assert support.chat_confirmed_side_effects(agent) == frozenset(
            {agent.tools.confirmation_key("memory.add")}
        )
    finally:
        support.close_manager()
        monkeypatch.setattr(support, "_agent", None)
        monkeypatch.setattr(support, "_pipeline", None)


# ---------------------------------------------------------------------------
# Phase 6: 文档中心、统计、三库对账
# ---------------------------------------------------------------------------


def test_list_documents_pagination(file_client) -> None:
    client, _ = file_client
    _ingest_documents(client, ["文档甲", "文档乙", "文档丙"])

    first = client.get("/api/documents", params={"page_size": 2}).json()
    second = client.get("/api/documents", params={"page_size": 2, "page": 2}).json()

    assert first["total"] == second["total"] == 3
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert first["items"][0]["chunk_count"] > 0
    assert client.get("/api/documents", params={"page": 0}).status_code == 422


def test_get_document_raw_text(file_client) -> None:
    client, _ = file_client
    document_id = _ingest_documents(client, ["边界文档"])[0]

    payload = client.get(f"/api/documents/{document_id}").json()

    assert payload["document_id"] == document_id
    assert payload["chunks"]
    for chunk in payload["chunks"]:
        assert payload["raw_text"][chunk["char_start"] : chunk["char_end"]] == chunk["text"]
    assert payload["status"] in {"vectorized", "extracted"}
    assert client.get("/api/documents/不存在").status_code == 404


def test_document_endpoints_need_a_file_database(client: TestClient) -> None:
    """:memory: 存储没有真值源，必须明确报错而不是静默返回空列表。"""

    response = client.get("/api/documents")

    assert response.status_code == 400
    assert "内存模式" in response.json()["detail"]


def test_stats_endpoint(file_client) -> None:
    client, _ = file_client
    document_id = _ingest_documents(client, ["统计文档"])[0]
    detail = client.get(f"/api/documents/{document_id}").json()

    stats = client.get("/api/stats").json()

    assert stats["documents"] == 1
    assert stats["chunks"] == len(detail["chunks"])
    assert stats["chunks_indexed"] == len(detail["chunks"])
    assert stats["facts"] == 0
    assert stats["memories_total"] >= stats["chunks"]


def test_reconcile_reports_drift(file_client) -> None:
    client, store = file_client
    document_id = _ingest_documents(client, ["漂移文档"])[0]
    chunks = client.get(f"/api/documents/{document_id}").json()["chunks"]
    store.ids.discard(chunks[0]["chunk_id"])

    report = client.get("/api/reconcile").json()

    assert report["counts"]["chunks"] == len(chunks)
    # 向量库里除了 chunk 还有导入时记录的那条 episodic 记忆，所以只比对集合本身。
    assert report["counts"]["qdrant_points"] == len(store.ids)
    drift = {entry["kind"]: entry for entry in report["drift"]}
    assert drift["missing_vector"]["count"] == 1
    assert drift["missing_vector"]["ids"] == [chunks[0]["chunk_id"]]
    assert "orphan_vector" not in drift


def test_reconcile_repair_is_idempotent(file_client) -> None:
    client, store = file_client
    document_id = _ingest_documents(client, ["自愈文档"])[0]
    chunk_id = client.get(f"/api/documents/{document_id}").json()["chunks"][0]["chunk_id"]
    store.ids.discard(chunk_id)

    first = client.post("/api/reconcile", json={"repair": ["missing_vector"]}).json()
    second = client.post("/api/reconcile", json={"repair": ["missing_vector"]}).json()

    assert first["repaired"]["missing_vector"] == 1
    assert second["repaired"]["missing_vector"] == 0
    assert client.get("/api/reconcile").json()["drift"] == []
    assert chunk_id in store.ids
    assert client.post("/api/reconcile", json={"repair": ["不存在的类型"]}).status_code == 422


def test_health_reports_store_modes_and_degraded(file_client) -> None:
    client, _ = file_client

    health = client.get("/api/health").json()

    assert health["store_modes"] == {
        "document": "sqlite",
        "vector": "DriftVectorStore",
        "graph": "inmemory",
    }
    assert health["degraded"] == {
        "embedding": "hash",
        "embedding_endpoint": "",
        "chat_ready": health["chat_ready"],
        "keyword_fallback": False,
        "embedding_hint": "",
    }


def test_revectorize_rebuilds_every_chunk(file_client) -> None:
    client, store = file_client
    document_id = _ingest_documents(client, ["重嵌入文档"])[0]
    chunks = client.get(f"/api/documents/{document_id}").json()["chunks"]
    store.ids.difference_update(chunk["chunk_id"] for chunk in chunks)   # 模拟索引丢失

    res = client.post(f"/api/documents/{document_id}/revectorize").json()

    # 已抽取过知识的文档不因重嵌入而回退状态：extracted 保持 extracted。
    assert res == {"document_id": document_id, "chunks_reindexed": len(chunks), "status": "extracted"}
    detail = client.get(f"/api/documents/{document_id}").json()
    assert {chunk["vector_status"] for chunk in detail["chunks"]} == {"indexed"}
    assert all(chunk["chunk_id"] in store.ids for chunk in detail["chunks"])
    assert client.post("/api/documents/不存在/revectorize").status_code == 404


# ---------------------------------------------------------------------------
# Phase 7 U6: 增量图 ?since=<revision>
# ---------------------------------------------------------------------------


def test_graph_since_returns_only_revision_when_nothing_changed(file_client) -> None:
    client, _ = file_client
    _ingest_documents(client, ["增量文档"])

    full = client.get("/api/graph").json()
    assert full["unchanged"] is False
    assert full["revision"] > 0 and full["nodes"]

    delta = client.get("/api/graph", params={"since": full["revision"]}).json()

    # 无写入：不回传节点/边，但 revision 与计数保持一致，前端沿用本地图即可
    assert delta["unchanged"] is True
    assert delta["nodes"] == [] and delta["edges"] == []
    assert delta["revision"] == full["revision"]
    assert delta["stats"] == full["stats"]


def test_graph_since_is_stale_after_a_write(file_client) -> None:
    """写入后必须重新变成全量，否则前端会永远停在旧图上。"""

    client, _ = file_client
    _ingest_documents(client, ["第一版"])
    first = client.get("/api/graph").json()

    _ingest_documents(client, ["第二版"])
    after_write = client.get("/api/graph", params={"since": first["revision"]}).json()

    assert after_write["unchanged"] is False
    assert after_write["revision"] > first["revision"]
    # 「差量合并后节点总数与全量一致」：客户端用 delta.revision 再问一次即得到全量
    refetched = client.get("/api/graph", params={"since": after_write["revision"]}).json()
    assert refetched["unchanged"] is True
    full = client.get("/api/graph").json()
    assert len(full["nodes"]) == len(after_write["nodes"]) == full["stats"]["total"]


def test_graph_without_since_stays_backward_compatible(client: TestClient) -> None:
    payload = client.get("/api/graph").json()

    assert payload["unchanged"] is False
    assert "nodes" in payload and "edges" in payload


def test_graph_since_zero_is_a_valid_cursor_not_a_missing_parameter(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """进程刚启动时 revision 就是 0：客户端带 0 来问必须走增量，而不是被判成没带参数。

    revision 是进程级共享计数（同一进程里别的用例已经把它推高），所以这里显式压回 0
    来复现「刚启动」这一档。
    """

    monkeypatch.setattr(support, "GRAPH_REVISION", 0)
    full = client.get("/api/graph").json()
    assert full["revision"] == 0 and full["unchanged"] is False

    delta = client.get("/api/graph", params={"since": 0}).json()

    assert delta["unchanged"] is True
    assert delta["nodes"] == []
    assert delta["stats"] == full["stats"]


class DeadGatewayEmbedding(HashEmbedding):
    """已配置网关但连不上：用真实会被拒连的端口，逼出降级分支。"""

    def __init__(self, port: int) -> None:
        super().__init__()
        self.base_url = f"http://127.0.0.1:{port}"


def closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _dead_gateway_client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_AUTOSEED", "0")
    monkeypatch.setenv("EMBEDDING_TUNNEL_HINT", "ssh -N -L 10800:127.0.0.1:18000 root@example -p 10034")
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=DeadGatewayEmbedding(closed_port()),
    )
    return create_app(manager=manager), manager


def test_ingest_fails_fast_with_the_tunnel_command_when_gateway_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """方案 §11.6：隧道断开时入库必须明确报错（含隧道命令），且不留下垃圾记录。"""

    app, manager = _dead_gateway_client(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/api/ingest",
            files=_make_ingest_payload("降级.txt", "隧道没通时不应该写库。" * 10),
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "嵌入网关不可达" in detail
        assert "ssh -N -L 10800:127.0.0.1:18000 root@example -p 10034" in detail
        # 快速失败：真值源里不应出现半吊子文档
        assert client.get("/api/documents").json()["total"] == 0
        assert manager.document_store.list(include_expired=True) == []
        # health 把同一条命令下发给前端（前端不再硬编码）
        assert client.get("/api/health").json()["degraded"] == {
            "embedding": "unreachable",
            "embedding_endpoint": manager.embedding.base_url,
            "chat_ready": False,
            "keyword_fallback": True,
            "embedding_hint": "ssh -N -L 10800:127.0.0.1:18000 root@example -p 10034",
        }
    manager.close()


def test_revectorize_reports_the_tunnel_command_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """先有文档再断隧道：重嵌入要 503 且带修复命令，而不是静默失败。"""

    app, manager = _dead_gateway_client(tmp_path, monkeypatch)
    monkeypatch.delenv("EMBEDDING_TUNNEL_HINT", raising=False)
    with TestClient(app) as client:
        repository = DocumentRepository(manager.config.sqlite_path)
        try:
            repository.upsert_document(DocumentRecord(document_id="doc-1", raw_text="正文", source="a.txt"))
            repository.upsert_chunks([ChunkRecord(
                chunk_id="doc-1:0", document_id="doc-1", chunk_index=0,
                char_start=0, char_end=2, text="正文",
            )])
        finally:
            repository.close()
        response = client.post("/api/documents/doc-1/revectorize")
        assert response.status_code == 503
        assert "重嵌入已中止" in response.json()["detail"]
        # 没有配 EMBEDDING_TUNNEL_HINT 时给通用指引，而不是编造命令
        assert "见本地部署说明" in response.json()["detail"]
    manager.close()
