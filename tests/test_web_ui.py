"""Phase 7 前端验收：静态页可服务、面板存在、前端只调用真实存在的端点。

前端是单文件零构建（web/static/index.html），没有打包器兜底，所以「调用了不存在的
/api/*」这类错误只能在浏览器里才发现。这里用最小的静态检查把它挡在提交前。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memory import MemoryConfig, MemoryManager  # noqa: E402
from web import create_app  # noqa: E402
from web.app import app as module_app  # noqa: E402
from web.support import HashEmbedding  # noqa: E402

INDEX = Path(__file__).resolve().parent.parent / "web" / "static" / "index.html"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("WEB_AUTOSEED", "0")
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
    )
    with TestClient(create_app(manager=manager)) as test_client:
        yield test_client
    manager.close()


def test_static_index_serves_the_star_map(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="universe"' in response.text


def test_index_always_calls_only_registered_endpoints() -> None:
    called = set(re.findall(r"/api/[a-z0-9\-]+", INDEX.read_text(encoding="utf-8")))
    registered = {
        route.path for route in module_app.routes if str(getattr(route, "path", "")).startswith("/api/")
    }

    missing = sorted(
        path for path in called
        if not any(route == path or route.startswith(path + "/") for route in registered)
    )

    assert called, "前端应当至少调用一个 /api 端点"
    assert missing == [], f"前端调用了不存在的端点：{missing}"


def test_u1_graph_reasoning_entry_and_highlight_are_wired() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert '<option value="graphrag">' in html
    assert 'apiPostJson("/api/graph-rag"' in html
    # 高亮态必须被绘制循环消费，否则 U1 只是算了不用
    assert "highlightPath(" in html
    assert "highlight.edges.has(edgeKey(s.id, t.id))" in html
    assert "highlight.entities.has(n.title)" in html
    assert 'id="evidence-list"' in html


def test_u3_storage_panel_is_wired_to_health_and_reconcile() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="storage-panel"' in html
    assert 'apiGet("/api/reconcile"' in html
    assert 'apiPostJson("/api/reconcile"' in html
    # 降级时必须给出隧道命令，否则用户只知道坏了、不知道怎么修（D8）
    assert "unreachable" in html and "embedding_hint" in html
    # 命令由后端下发，前端不得硬编码服务器地址
    assert "103.240.196.39" not in html


def test_graphrag_mode_does_not_require_the_chat_model() -> None:
    """D9：图谱推理是纯本地检索，聊天模型没配也要能发问。"""

    html = INDEX.read_text(encoding="utf-8")

    assert "chatReady || chatMode === \"graphrag\"" in html


def test_u2_document_center_is_wired() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="documents-panel"' in html
    assert 'id="doc-list"' in html and 'id="doc-detail"' in html
    assert "apiGet(`/api/documents?" in html
    assert "apiGet(`/api/documents/${encodeURIComponent(documentId)}`" in html
    assert "/revectorize`" in html
    # 方案 §10.1 点名 /api/import 零入口，U2 必须补上
    assert 'apiPostForm("/api/import"' in html


def test_u2_chunk_highlight_is_driven_by_truth_source_offsets() -> None:
    """分块高亮必须用真值源的 char_start/char_end，不能在前端重新切分。"""

    html = INDEX.read_text(encoding="utf-8")

    assert "chunk.char_start" in html and "chunk.char_end" in html
    assert "raw_text" in html
    # 状态机进度与失败原因都要能看见（U2/U5）
    assert "docBadge(item.status)" in html or "docBadge(doc.status)" in html
    assert "doc.error" in html


def test_u6_delta_refresh_and_lod_are_wired() -> None:
    """U6：带 revision 轮询、无变化不重排；LOD 分档集中在一处。"""

    html = INDEX.read_text(encoding="utf-8")

    assert "/api/graph?since=${since}" in html
    assert "data.unchanged" in html
    assert "GRAPH_POLL_MS" in html and "loadGraph({ since: true })" in html
    assert "function lodLevel()" in html
    assert "lod < 1 && n.level === 3" in html
    assert "lod >= 2" in html
