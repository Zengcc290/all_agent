# 知识星云前端（React 单页）

对应后端 `web/app.py`（默认 `http://127.0.0.1:8765`）。构建产物输出到
`../static`，由 FastAPI 直接挂载到 `/`，因此**不启动 Vite 也能用**。

## 开发

```bash
npm install
npm run dev      # http://127.0.0.1:5173，/api 代理到 127.0.0.1:8765
npm run build    # 输出到 ../static（会清空该目录后重写）
```

`VITE_API_TARGET=http://host:port npm run dev` 可改代理目标；
`VITE_API_BASE=/api` 可改接口前缀（默认 `/api`）。

## 目录

```
src/
  main.jsx                入口
  App.jsx                 顶栏 / tab / 星云图筛选与刷新 / 节点详情
  api.js                  全部端点 + withEmbeddingGuard（409 → 询问 → confirm_rebuild 重试）
  util.js                 领域配色（crc32，与后端 constants.NEBULA_PALETTE 一致）、类型色、格式化
  styles.css              深色主题
  components/
    NebulaGraph.jsx       Canvas 力导向星云图（恒星=领域、行星=实体、卫星=事实/知识块）
    ChatPanel.jsx         知识管家对话（写操作确认 / 检索依据 / 逐条打分）
    IngestPanel.jsx       文档上传 / 一句话入库 / 图片入库 / 手工三元组
    QueryPanel.jsx        GraphRAG 检索（证据、关系路径、组装上下文）
    DocumentsPanel.jsx    文档中心（真值源 char_start/char_end 高亮、重嵌入、导出导入）
    JobsPanel.jsx         入库任务队列与失败重试
    DashboardPanel.jsx    健康度 / 规模统计 / 三库对账与修复 / 重建向量 / 重新播种
```

## 约定

- 只调用 `web/app.py` 里真实注册的端点；`tests/test_web_ui.py` 会在提交前校验这一点。
- 危险写操作必须显式确认：`withEmbeddingGuard` 处理嵌入锁 409，聊天里的
  `memory.manage` 由后端返回待确认调用，用户点「同意执行」后才带 `confirmation` 重发。
- 星云图按 `revision` 增量轮询（`/api/graph?since=`），无变化时后端只回 `stats`，
  前端不重排布局；节点位置按 id 保留，跨刷新稳定。