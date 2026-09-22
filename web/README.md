# 知识星云 · Web 层

把 `memory` 四层记忆系统暴露为 HTTP API，并托管 React 单页前端。

## 快速开始

```bash
# 项目根目录下（后端）
.venv\Scripts\python.exe -m web.app
# 打开 http://127.0.0.1:8765 —— 已内置构建好的前端
```

无需任何 API key 即可运行：向量检索自动降级为本地 `HashEmbedding`，
聊天端点返回 503 并提示如何配置。

## 前端开发（改界面时用）

前端是 Vite + React 工程，源码在 `web/frontend/`，构建产物落在 `web/static/`
（FastAPI 直接挂载它，所以不跑 Vite 也能打开页面）：

```bash
cd web/frontend
npm install          # 首次
npm run dev          # 开发态 http://127.0.0.1:5173，/api 代理到 8765
npm run build        # 产物输出到 ../static，刷新后端页面即可生效
```

面板一览（`web/frontend/src/components/`）：

| 面板 | 对应端点 |
|---|---|
| `NebulaGraph.jsx` | `GET /api/graph`（revision 增量轮询 + 历史视图 `at`） |
| `ChatPanel.jsx` | `POST /api/chat`（写操作确认、检索依据、逐条打分） |
| `IngestPanel.jsx` | `POST /api/ingest`、`/api/knowledge`、`/api/knowledge/image`、`/api/facts` |
| `QueryPanel.jsx` | `POST /api/graph-rag`（证据 / 路径 / 上下文，命中反哺星云图高亮） |
| `DocumentsPanel.jsx` | `GET/POST /api/documents…`、`/api/export`、`/api/import` |
| `JobsPanel.jsx` | `GET /api/knowledge/jobs`、`POST /api/knowledge/jobs/{id}/retry` |
| `DashboardPanel.jsx` | `GET /api/health`、`/api/stats`、`/api/reconcile`、`POST /api/reconcile`、`POST /api/embedding/rebuild`、`POST /api/seed` |

所有写请求统一走 `web/frontend/src/api.js` 的 `withEmbeddingGuard`：
遇到 409 `embedding_lock_mismatch` 先弹窗说明，用户确认后才带
`confirm_rebuild=true` 重发，不会静默重建向量投影。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/graph` | 全图节点+边（星云图数据源，支持 `since` 增量与 `at` 历史视图） |
| POST | `/api/chat` | 与知识管家对话（ReAct，未配置模型时 503） |
| POST | `/api/ingest` | 上传文档 → RAG 切块、LLM 自动分类、实体关系抽取 |
| POST | `/api/graph-rag` | 向量证据 + 图关系路径混合检索 |
| POST | `/api/facts` | 手工添加三元组 `{subject, predicate, object, domain?, note?}` |
| POST | `/api/knowledge` | 一句话入库：原文向量化 + LLM 抽取 → 图结构（`wait=false` 走后台队列） |
| POST | `/api/knowledge/image` | 图片/相机 → VL 嵌入 + 视觉模型抽取多元关系 |
| GET | `/api/knowledge/jobs` | 一句话入库历史（排队中/正在入库/成功/失败） |
| POST | `/api/knowledge/jobs/{job_id}/retry` | 失败任务重新入队 |
| POST | `/api/seed` | 重新播种种子数据（幂等） |
| GET | `/api/export` | 导出全部记忆为 JSON 文件 |
| POST | `/api/import` | 导入导出过的 JSON（幂等去重） |
| GET | `/api/documents` · `/api/documents/{id}` | 文档中心列表 / 详情（chunk 含 `char_start/char_end`） |
| POST | `/api/documents/{id}/revectorize` | 单篇重嵌入 |
| GET | `/api/stats` | 知识库规模统计 |
| GET | `/api/reconcile` · `POST /api/reconcile` | 三库对账报告 / 修复漂移 |
| POST | `/api/embedding/rebuild` | 按当前 embedding 重建向量投影 |
| GET | `/api/health` | 健康检查：嵌入模式、聊天可用性、降级原因 |

## 数据流

```
上传文档 / 一句话 / 图片 / 手工三元组
        │
        ▼
文档真值源（SQLite documents + chunks）──┐
        │                                 │ 对账 / 重嵌入
        ▼                                 ▼
LLM 抽取 domain/entity/relation ──► Qdrant 向量投影 + Neo4j 图投影
        │                                 ▲
        ▼                                 │
Agent 工具 memory.query / memory.add /   │ GraphRAG 一跳/多跳关系扩展
memory.rag_search / memory.manage        │
        │                                 │
        └────────► 星云图 nodes+edges ────┘
             （恒星=领域 行星=实体 卫星=事实/知识块）
```

## 结构

- `app.py` —— FastAPI 路由与应用工厂（`create_app(manager=None)` 可注入测试内存库）
- `support.py` —— 单例、聊天可用性与嵌入装配；`STATIC_DIR` 指向 `web/static`
- `seed.py` —— Aetheria 种子数据播种（幂等；开关 `constants.WEB_AUTOSEED`）
- `ingest_queue.py` —— 后台入库任务队列（任务状态由 `/api/knowledge/jobs` 查询）
- `frontend/` —— React 单页工程（源码；`npm run build` 输出到 `web/static`）
- `static/` —— 构建产物 + FastAPI 静态挂载点（勿手改，改 `frontend/` 后重新构建）
- `seed_data.json` —— 种子数据

> 记忆项 → 星云 `nodes/edges` 的映射规则、领域分类与孤儿实体统计都已工具化，
> 见 `tool/graph_snapshot.py`、`tool/domain_classify.py`、`tool/orphan_entities.py`。
> 删除统一使用 `memory.manage` 的运行时写确认，不再维护第二套未接线的提案协议。