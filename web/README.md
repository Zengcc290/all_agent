# 知识星云 · Web 层

把 `memory` 四层记忆系统暴露为 HTTP API，并托管 Canvas 2D 星云图前端。

## 快速开始

```bash
# 项目根目录下
.venv\Scripts\python.exe -m web.app
# 打开 http://127.0.0.1:8765
```

无需任何 API key 即可运行：向量检索自动降级为本地 `HashEmbedding`，
聊天端点返回 503 并提示如何配置。

## 升级为完整体验（可选）

1. **聊天模型**：复制 `config/provider.example.toml` 为 `config/provider.toml`，
   填入真实 `api_key`（或改用 `api_key_env` 指向环境变量），重启服务。
2. **真实嵌入**：在 `.env` 中填入 `DASHSCOPE_API_KEY=sk-...`，
   重启后自动切换到 qwen3-embedding-0.6b，检索质量显著提升。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/graph` | 全图节点+边（星云图数据源） |
| POST | `/api/chat` | 与知识管家对话（ReAct，未配置模型时 503） |
| POST | `/api/ingest` | 上传文档 → RAG 切块、LLM 自动分类、实体关系抽取 |
| POST | `/api/graph-rag` | 向量证据 + 图关系路径混合检索 |
| POST | `/api/facts` | 手工添加三元组 `{subject, predicate, object, domain?, note?}` |
| POST | `/api/knowledge` | 一句话入库：原文向量化 + LLM 自动抽取实体/三元组 → 图结构（未配置聊天模型时仅向量化） |
| POST | `/api/seed` | 重新播种种子数据（幂等） |
| GET | `/api/export` | 导出全部记忆为 JSON 文件 |
| POST | `/api/import` | 导入导出过的 JSON（幂等去重） |
| GET | `/api/health` | 健康检查：嵌入模式、聊天可用性 |

## 数据流

```
上传文档/添加事实/聊天沉淀
        │
        ▼
        │                    ▲
        │                    │ GraphRAG 一跳/多跳关系扩展
        ▼                    │
LLM 抽取 domain/entity/relation ────────────────┘
        │                                            │
        ▼                                            ▼
Agent 工具 memory.manage / memory.rag          星云图 nodes+edges
（同一份记忆 = 记忆共享）                    （恒星=领域 行星=实体 卫星=事实/知识块）
```

## 结构

- `app.py` —— FastAPI 路由与应用工厂（`create_app(manager=None)` 可注入测试内存库）
- `graph_builder.py` —— 记忆项 → 星云 nodes/edges 的映射规则
- `support.py` —— HashEmbedding 离线降级、单例、聊天可用性检测
- `seed.py` —— Aetheria 种子数据播种（幂等，可用 `WEB_AUTOSEED=0` 关闭自动播种）
- `static/index.html` —— 星云图前端（改造自 Aetheria 单文件 HTML）
- `seed_data.json` —— 种子数据（从原 HTML 的 celestialTree 提取）
