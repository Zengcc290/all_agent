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

from memory import HashEmbedding, MemoryConfig, MemoryManager  # noqa: E402
from web import create_app  # noqa: E402
from web.app import app as module_app  # noqa: E402

INDEX = Path(__file__).resolve().parent.parent / "web" / "static" / "index.html"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
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


def test_u7_domain_filter_and_alias_sidebar_are_wired() -> None:
    """U7：领域筛选（纯前端）+ 实体侧栏展示图库别名。"""

    html = INDEX.read_text(encoding="utf-8")

    assert 'id="filter-domain"' in html
    assert "function applyDomainFilter(value)" in html
    assert "visibleNodes()" in html
    # 筛选后不画通向视野外的边，否则会连线到上一次布局的坐标
    assert "graph.visible.has(s.id)" in html
    # 别名只读展示自图库（P4 已写入实体属性），来源与重要度沿用节点字段
    assert "node.meta.aliases" in html


def test_u4_retrieval_breakdown_panel_is_wired() -> None:
    """U4：回答气泡下方可展开「依据」，逐条给出向量分/关键词分/融合分。"""

    html = INDEX.read_text(encoding="utf-8")

    assert "appendRetrieval(think, res.retrieval)" in html
    assert "function appendRetrieval(bubble, report)" in html
    # 原生 <details> 而非自绘开关：键盘可达（U8 也受益）
    assert 'document.createElement("details")' in html
    assert "retrieval.snippet" in html or "retrieval-snippet" in html
    for label in ("向量 ", "关键词 ", "RRF "):
        assert label in html
    assert "fmtScore" in html


def test_u8_keyboard_reachability_and_small_screen_layout() -> None:
    """U8：交互行用原生 button、ESC 逐层关闭、抽屉手势、小屏布局。"""

    html = INDEX.read_text(encoding="utf-8")

    # 1) 可点击的行改成原生 button（Tab/Enter/空格由浏览器负责）
    assert 'card.className = "subnode-card"' in html
    for cls in ("subnode-card", "search-hit", "evidence-item", "parent-uplink-btn"):
        assert f'{cls}"' in html
    assert html.count('createElement("button")') >= 4
    # 曾经的 div 写法不该再出现在这几处
    assert 'const card = document.createElement("div");\n        card.className = "subnode-card";' not in html
    assert 'const row = document.createElement("div");\n        row.className = "search-hit";' not in html

    # 2) 焦点可见 + 关闭按钮有可读名称
    assert ":focus-visible { outline:" in html
    assert 'aria-label="关闭详情抽屉"' in html and 'aria-label="关闭对话面板"' in html
    assert 'aria-label="搜索知识星云"' in html and 'aria-label="向知识管家提问"' in html
    # 画布对读屏有说明；动态区域会播报
    assert 'aria-label="知识星云图' in html
    assert 'id="chat-msgs" aria-live="polite"' in html
    assert 'id="toasts" aria-live="polite"' in html

    # 3) ESC 逐层关闭 + 抽屉手势
    assert 'event.key !== "Escape"' in html
    assert "function endDrawerSwipe" in html or "const endDrawerSwipe" in html
    assert "pointerdown" in html and "pointermove" in html and "pointerup" in html
    # 手势不与内容滚动打架：竖直方向只在滚动到顶时才跟随
    assert "drawerScroll.scrollTop <= 0" in html

    # 4) 小屏布局
    assert "@media (max-width: 720px)" in html
    assert "#detail-drawer.visible { transform: translateY(0); }" in html
    assert "min-height: 44px" in html

    # 5) 中屏：顶栏/聊天输入换行，面板不再顶出视口边框
    assert "@media (max-width: 1100px)" in html
    assert "flex-wrap: wrap" in html
    assert "max-width: min(920px, calc(100vw - 56px))" in html
    assert "width: min(400px, calc(100vw - 24px))" in html
    assert "flex: 1 1 140px" in html


def test_embedding_lock_confirm_dialog_is_wired() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert "embedding_lock_mismatch" in html
    assert "window.confirm" in html
    assert "confirm_rebuild=true" in html
    assert "已保持锁定配置" in html
    assert "function withEmbeddingGuard" in html
    assert "formatEmbeddingLock" in html
    assert "h.embedding_lock" in html
    assert "function offerEmbeddingRebuild" in html
    assert "/api/embedding/rebuild" in html
    assert "embedding_mismatch" in html
    assert "data-retry-job" in html
    assert "function retryIngestJob" in html
    assert "/api/knowledge/jobs/" in html
    assert "job-retry" in html
