"""前端验收：SPA 可服务、构建产物存在、且前端只调用真实注册的端点。

前端已从「单文件零构建 HTML」迁移为 ``web/frontend`` 下的 Vite + React 工程
（构建产物落在 ``web/static``，由 FastAPI 挂载到 ``/``）。这里仍然把
「前端调用了不存在的 /api 端点」这类错误挡在提交前，只是检查对象从一份 HTML
变成前端源码里的端点调用集合。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memory import HashEmbedding, MemoryConfig, MemoryManager  # noqa: E402
from web import create_app  # noqa: E402
from web.app import app as module_app  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = PROJECT_ROOT / "web" / "frontend" / "src"
STATIC_DIR = PROJECT_ROOT / "web" / "static"
SOURCE_FILES = sorted(FRONTEND_SRC.rglob("*.js")) + sorted(FRONTEND_SRC.rglob("*.jsx"))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
    )
    with TestClient(create_app(manager=manager)) as test_client:
        yield test_client
    manager.close()


def test_frontend_sources_exist() -> None:
    """迁移后源码必须在位：入口、面板、样式、API 客户端。"""

    assert (FRONTEND_SRC / "main.jsx").is_file()
    assert (FRONTEND_SRC / "App.jsx").is_file()
    assert (FRONTEND_SRC / "api.js").is_file()
    assert (FRONTEND_SRC / "styles.css").is_file()
    components = {path.name for path in (FRONTEND_SRC / "components").glob("*.jsx")}
    assert components >= {
        "NebulaGraph.jsx",
        "ChatPanel.jsx",
        "IngestPanel.jsx",
        "QueryPanel.jsx",
        "DocumentsPanel.jsx",
        "JobsPanel.jsx",
        "DashboardPanel.jsx",
    }


def test_static_index_serves_the_spa(client: TestClient) -> None:
    """GET / 返回构建出的 SPA 入口，并加载其 JS / CSS 资产。"""

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="root"' in response.text
    assert "text/html" in response.headers.get("content-type", "")
    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', response.text)
    assert assets, "SPA 入口必须引用构建产物"
    for asset in assets:
        served = client.get(asset)
        assert served.status_code == 200, f"{asset} 不可访问：构建产物未同步"


def test_frontend_only_calls_registered_endpoints() -> None:
    """前端源码里出现的每个 /api 端点都必须在后端注册过。"""

    registered = {
        route.path
        for route in module_app.routes
        if str(getattr(route, "path", "")).startswith("/api/")
    }
    called: set[str] = set()
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        called.update(re.findall(r"/api/[a-z0-9\-]+", text))
        # api.js 用统一 BASE + 模板串路径，这里把它声明的路径也抓出来
        called.update(re.findall(r"req\(\s*`?(/api/[a-z0-9\-/{}$]+)", text))
        called.update(re.findall(r"'(/api/[a-z0-9\-/{}]+)'", text))
        called.update(re.findall(r"`(/api/[a-z0-9\-/${}]+)`", text))

    assert called, "前端应当至少调用一个 /api 端点"
    missing = sorted(
        path for path in called
        if not any(route == path or route.startswith(path + "/") for route in registered)
    )
    assert missing == [], f"前端调用了不存在的端点：{missing}"


def test_every_backend_endpoint_has_a_frontend_entry() -> None:
    """反过来：后端每个端点都该有前端入口，避免出现无人调用的死接口。"""

    bodies = "\n".join(path.read_text(encoding="utf-8") for path in SOURCE_FILES)
    bodies += (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    registered = sorted(
        route.path
        for route in module_app.routes
        if str(getattr(route, "path", "")).startswith("/api/")
    )
    # 「静态 GET /」由 vite 构建产物提供；其余端点都要在前端源码里出现。
    # 端点地址在 api.js 里以 BASE + 模板串形式书写，这里按去掉路径参数后的
    # 资源名匹配（/api/documents/{document_id} -> documents）。
    ignored = {"/api/export"}
    missing = []
    for route in registered:
        if route in ignored:
            continue
        segments = [seg for seg in route.strip("/").split("/") if seg and not seg.startswith("{")]
        resource = segments[-1] if segments else ""
        if route in bodies or (resource and resource in bodies):
            continue
        missing.append(route)
    assert missing == [], f"后端端点没有前端入口：{missing}"


def test_nebula_graph_renders_domain_planet_and_moon() -> None:
    """U1：星云图要区分领域恒星 / 实体行星 / 事实卫星，并支持缩放与选中。"""

    graph = (FRONTEND_SRC / "components" / "NebulaGraph.jsx").read_text(encoding="utf-8")

    assert 'kindColor' in graph and 'kindLabel' in graph
    assert "domain" in graph and "entity" in graph
    assert "onWheel" in graph and "onPointerMove" in graph
    assert "onSelect" in graph


def test_graphrag_panel_shows_evidence_and_paths() -> None:
    """图谱检索面板必须同时展示证据与关系路径，并能反哺星云图高亮。"""

    panel = (FRONTEND_SRC / "components" / "QueryPanel.jsx").read_text(encoding="utf-8")

    assert "api.graphRag" in panel
    assert "res.evidence" in panel
    assert "res.paths" in panel
    assert "onHighlight" in panel
    # 图谱推理不依赖聊天模型（D9）
    assert "未配置聊天模型也能用" in panel


def test_document_center_uses_truth_source_offsets() -> None:
    """分块高亮必须用真值源的 char_start/char_end，不能在前端重新切分。"""

    panel = (FRONTEND_SRC / "components" / "DocumentsPanel.jsx").read_text(encoding="utf-8")

    assert "api.document(" in panel
    assert "api.revectorize" in panel
    assert "api.exportUrl" in panel
    assert "api.importFile" in panel
    assert "char_start" in panel and "char_end" in panel
    assert "raw_text" in panel


def test_embedding_lock_guard_is_wired_in_the_frontend() -> None:
    """409 embedding_lock_mismatch 必须先询问，再带 confirm_rebuild 重试。"""

    client_source = (FRONTEND_SRC / "api.js").read_text(encoding="utf-8")

    assert "embedding_lock_mismatch" in client_source
    assert "window.confirm" in client_source
    assert "confirm_rebuild" in client_source
    assert "withEmbeddingGuard" in client_source


def test_chat_confirmation_flow_is_wired() -> None:
    """危险写操作必须走确认协议：确认后才带 confirmation 重发。"""

    chat = (FRONTEND_SRC / "components" / "ChatPanel.jsx").read_text(encoding="utf-8")

    assert "confirmation" in chat
    assert "确认" in chat
    assert "同意执行" in chat


def test_jobs_panel_retries_failed_jobs() -> None:
    """失败的一句话入库任务要能从队列页重试。"""

    jobs = (FRONTEND_SRC / "components" / "JobsPanel.jsx").read_text(encoding="utf-8")

    assert "api.retryJob" in jobs
    assert "retryable" in jobs


def test_dashboard_panel_covers_health_stats_and_reconcile() -> None:
    """监控台要把健康度、规模统计与三库对账都接上。"""

    dash = (FRONTEND_SRC / "components" / "DashboardPanel.jsx").read_text(encoding="utf-8")

    assert "api.health()" in dash
    assert "api.stats()" in dash
    assert "api.reconcile()" in dash
    assert "api.reconcileRepair" in dash
    assert "embedding_hint" in dash
    assert "window.confirm" in dash


def test_health_reports_graph_revision_and_delta_polling_contract(client: TestClient) -> None:
    """增量刷新契约：since 命中 revision 时返回 unchanged=true 且不重发节点。"""

    home = client.get("/api/graph")
    assert home.status_code == 200
    payload = home.json()
    revision = payload["revision"]

    delta = client.get(f"/api/graph?since={revision}").json()
    assert delta["unchanged"] is True
    assert delta["nodes"] == []
    assert "stats" in delta


def test_keyboard_reachability_and_small_screen_layout() -> None:
    """U8：原生按钮可达、焦点可见、小屏与中屏都有对应样式。"""

    styles = (FRONTEND_SRC / "styles.css").read_text(encoding="utf-8")

    assert ":focus-visible" in styles
    assert "@media (max-width: 720px)" in styles
    assert "min-height" in styles
    app_source = (FRONTEND_SRC / "App.jsx").read_text(encoding="utf-8")
    assert "<button" in app_source
    assert 'aria-label="知识星云图"' in (FRONTEND_SRC / "components" / "NebulaGraph.jsx").read_text(encoding="utf-8")
    assert 'aria-live="polite"' in app_source