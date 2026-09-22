# 知识图谱入库系统（精简版）

FastAPI + React + Qdrant + Neo4j + SQLite + OpenAI 兼容流式 LLM。
核心能力：**一句话入库**（LLM 抽取实体关系 → 双库并行写入 → 状态机转正）+
**实体星球可视化** + **多路混合检索（向量 + FTS5 + RRF）** + **可自动发现的工具系统**。

---

## 1. 一键启动（推荐）

所有依赖服务都在 Windows 原生运行，不需要 Docker / WSL。

```bat
双击 启动所有服务.bat
```

会按顺序拉起并等待就绪：

| 服务 | 地址 | 说明 |
|---|---|---|
| Qdrant | `127.0.0.1:6333` | 向量数据库，Windows 原生版（`qdrant-x86_64-pc-windows-msvc.zip`） |
| Neo4j | `127.0.0.1:7687` | 图数据库，管理台 `http://127.0.0.1:7474/browser/` |
| FastAPI | `127.0.0.1:8000` | 后端，接口文档 `http://127.0.0.1:8000/docs` |
| Vite | `127.0.0.1:5173` | 前端，`/api` 已代理到后端 |

然后浏览器打开 <http://127.0.0.1:5173>。

```bat
双击 停止所有服务.bat     REM 一键停掉全部
双击 查看状态.bat         REM 看各服务状态 + 连通性自检
```

### 两种运行模式

- **真实模式**：`.env` 里填好 `LLM_API_KEY` / `EMBEDDING_API_KEY`（如硅基流动）
- **离线演示模式**：未填 key 时脚本**自动**改用内置的假 LLM
  （`_fake_llm_server.py`，无需联网和 key，可完整跑通入库→查询全流程）

也可以强制演示模式：`powershell -File scripts\start-all.ps1 -Demo`

### 手动启动（不用脚本）

```powershell
# Qdrant
$env:QDRANT__SERVICE__HOST="127.0.0.1"; $env:QDRANT__SERVICE__HTTP_PORT="6333"
C:\qdrant\qdrant.exe

# Neo4j（5.x 的 .bat/.ps1 是 PowerShell 包装，直接调 java 更稳）
& "$env:JAVA_HOME\bin\java.exe" -cp "C:\neo4j\lib\*" -Dbasedir=C:\neo4j `
    org.neo4j.server.startup.Neo4jCommand console

# 后端 / 前端
python run.py
cd frontend; npm run dev
```

---

## 2. 手动安装依赖（不用一键脚本时）

```bash
# 1) Python 依赖
pip install -r requirements.txt

# 2) 配置：复制模板并按需修改（模板里的默认值可直接跑）
copy .example.env .env        # Windows
# Linux/macOS: cp .example.env .env

# 3) 启动后端（默认 http://127.0.0.1:8000）
python run.py

# 4) 前端（另开一个终端）
cd frontend
npm install
npm run dev                   # http://localhost:5173，/api 已代理到后端
```

> **Windows 原生 Qdrant**：官方不发 Windows 老版本，1.19+ 有
> `qdrant-x86_64-pc-windows-msvc.zip`，解压后直接 `qdrant.exe` 即可。
> 如果确实没有服务，把 `.env` 里的 `QDRANT_LOCAL_PATH` 填成 `./data/qdrant`
> 可用嵌入式模式跑在进程内（同一时刻只允许一个进程打开该目录）。
> Neo4j 始终需要本地服务，见上面的一键启动。

---

## 2. 配置说明（.example.env）

| 分组 | 变量 | 说明 |
|---|---|---|
| **LLM** | `LLM_MODEL` | 模型名，如 `Qwen/Qwen2.5-7B-Instruct` |
| | `LLM_API_KEY` | API 密钥 |
| | `LLM_BASE_URL` | 请求地址，代码自动拼 `/chat/completions`，走 OpenAI 协议 |
| | `LLM_STREAM` / `LLM_TEMPERATURE` / `LLM_TIMEOUT` / `LLM_MAX_TOKENS` | 流式开关、温度、超时、最大 token |
| **Embedding** | `EMBEDDING_BASE_URL` | 自动拼 `/embeddings`，按硅基流动标准格式 |
| | `EMBEDDING_MODEL` | 如 `BAAI/bge-m3` |
| | `EMBEDDING_API_KEY` | API 密钥 |
| | `EMBEDDING_DIM` | 向量维度（必须与模型一致：bge-m3 = 1024） |
| | `EMBEDDING_BATCH_SIZE` / `EMBEDDING_TIMEOUT` | 批量编码条数、超时 |
| **Qdrant** | `QDRANT_HOST` / `QDRANT_PORT` | 本地开放端口（默认 `127.0.0.1:6333`） |
| | `QDRANT_COLLECTION` | 集合名 |
| | `QDRANT_DISTANCE` | `cosine` / `dot` / `euclid` |
| | `QDRANT_API_KEY` / `QDRANT_PREFER_GRPC` | 鉴权与传输方式 |
| | `QDRANT_LOCAL_PATH` | 填目录则用**嵌入式本地模式**，不需要起服务 |
| **Neo4j** | `NEO4J_URI` | 本地回环，如 `bolt://127.0.0.1:7687` |
| | `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | 初始账号密码 |
| **SQLite** | `SQLITE_PATH` | 本地数据库文件路径 |

---

## 3. 目录结构

```
python--course/
├── .example.env              # 配置模板（复制为 .env）
├── requirements.txt
├── run.py                    # 入口：python run.py
├── app/
│   ├── config.py             # 读 .env 的全部配置
│   ├── core/
│   │   ├── registry.py       # 工具注册表 + discover() 自动发现 + 提示词生成
│   │   ├── validation.py     # 参数定义与校验（Tool / ToolParam）
│   │   ├── parser.py         # parse_llm_output：解析 LLM 固定格式输出
│   │   ├── prompt.py         # 动态拼装系统提示词（含全部工具说明）
│   │   ├── chunker.py        # 分块：当前保留空实现（原样透传）
│   │   ├── rrf.py            # RRF 倒数排序融合
│   │   └── ingest.py         # 入库流水线
│   ├── db/
│   │   ├── sqlite_store.py   # documents / ingest_queue / chunks / chunk_entities / chunks_fts
│   │   ├── qdrant_store.py   # Qdrant 封装（服务模式 + 本地嵌入式模式）
│   │   └── neo4j_store.py    # Neo4j 封装（实体/关系/chunk CRUD + 多跳）
│   ├── llm/client.py         # OpenAI 流式 chat + 硅基流动 embeddings
│   ├── tools/                # ★ 所有工具都放这里，discover 自动发现
│   │   ├── ingest_sentence.py, ingest_chunk.py, list_pending_chunks.py
│   │   ├── get_graph_entities.py, manage_graph.py, get_graph_snapshot.py
│   │   ├── query_graph.py, search_similar_chunks.py, search_fulltext.py
│   │   ├── hybrid_search.py, parse_llm_output.py, get_current_time.py
│   │   ├── get_stats.py, run_tools_parallel.py
│   ├── api/routes.py         # FastAPI 路由（REST + SSE）
│   └── main.py
├── frontend/                 # React + Vite
│   └── src/
│       ├── App.jsx  api.js  styles.css
│       └── components/  ForceGraph.jsx（实体星球）PendingQueue.jsx  IngestPanel.jsx
│                        QueryPanel.jsx  HybridPanel.jsx  ToolConsole.jsx
└── _selftest.py / _e2e_test.py / _hybrid_test.py    # 自检脚本
```

---

## 4. SQLite 表设计与入库状态机

四张表（其中前三张是需求指定的核心表）：

| 表 | 作用 |
|---|---|
| `documents` | **原始文档数据表**，存用户输入的原话 |
| `ingest_queue` | **入库状态表**，逐 chunk 记录 `qdrant_status` / `neo4j_status`（`pending`/`success`/`failed`）与错误信息 |
| `chunks` | **chunk 表**，只有两条入库线都成功才从队列「转正」到这里 |
| `chunk_entities` | chunk ↔ 实体的**多对多映射**（一个 `chunk_id` → 多个实体，一个实体 ← 多个 chunk） |
| `chunks_fts` | FTS5 全文索引（trigram，中文可用），chunk 转正时自动同步 |

状态机：

```
            ┌─────────── qdrant 线 ───────────┐
  一句话 ──▶ │  embedding → upsert 到 Qdrant   │── 都 success ──▶ chunks 表（转正）
            └──────────────────────────────────┘
            ┌─────────── neo4j 线 ────────────┐
            │ LLM 抽实体/关系 → MERGE 写图库   │── 任一失败 ──▶ 继续留在 ingest_queue
            └──────────────────────────────────┘
```

两条线在同一个 chunk 上**并行**执行（`asyncio.gather`）；多个 chunk 之间也并行，
受 `TOOL_MAX_CONCURRENCY` 限流。任一线失败，chunk 就一直留在队列表里等「重新入库」。

Neo4j 侧模型：

```cypher
(:Entity {key, name, type, time, aliases})
  -[:REL {predicate, directed, time, chunk_id, evidence}]->(:Entity)
(:Chunk {id, content})-[:MENTIONS]->(:Entity)
```

---

## 5. 工具系统
### 5.1 自动发现（核心机制）

所有工具都是 `app/tools/` 下的普通模块，**import 时自行调用 `registry.register(...)`**。
`registry.discover()` 用 `pkgutil.iter_modules` 扫描整个 `tools` 包并导入：

```python
# app/tools/my_tool.py
from app.core.registry import registry
from app.core.validation import Tool, ToolParam

async def my_handler(arg_a: str, arg_b: int = 5):
    return {"ok": True}

registry.register(Tool(
    name="my_tool",
    description="这个工具干什么用的（会写进提示词）",
    params=[
        ToolParam("arg_a", "string", "参数说明", required=True),   # 必填
        ToolParam("arg_b", "integer", "可选参数", default=5),      # 可省略
    ],
    handler=my_handler,
))
```

**往 `tools/` 目录丢一个这样的文件，下一次调用就自动被发现、登记、并出现在提示词里，
不需要手动改任何地方。** 也可以运行时 `POST /api/tools/rediscover` 热更新。

### 5.2 参数校验

- 必填缺失 / 未知参数 / 类型错误 / 枚举越界 / 数值超范围 → 统一抛 `ToolValidationError`
- 支持自动类型转换（`"5"` → `5`、`"true"` → `True`）、默认值补全
- 校验错误返回 HTTP 422，前端直接展示 `detail.message`

### 5.3 提示词由 registry 动态生成

`registry.describe()` 输出「可用工具清单」，每个工具列出
**作用、每个参数的类型、`[必填]` 还是 `[可选]`、默认值、枚举、取值范围**。
入库时 `build_extract_messages()` 把这份清单直接拼进提示词，所以：

> 新工具 → 自动进入提示词 → LLM 立刻知道能调用它。

### 5.4 已注册的 15 个工具

| 工具 | 作用 |
|---|---|
| `ingest_sentence` | **一句话入库**：sqlite 存档 → 取已有实体 → LLM 流式抽取 → 解析 → 时间兜底 → 双库并行写入 → 转正 |
| `ingest_chunk` | 重新入库：把队列里某个 chunk 再过一遍两条入库线 |
| `list_pending_chunks` | 取 sqlite 中**未入库成功**的所有 chunk，正文截取前 N 个字符 |
| `get_all_entities` | 从 neo4j 抽出**所有实体**（实体复用的依据） |
| `get_all_relations` | 从 neo4j 抽出所有关系三元组 |
| `parse_llm_output` | **解析器**：解析 LLM 固定格式 JSON 输出成实体/关系，去围栏、修截断、归一化去重 |
| `get_current_time` | 取系统时间，给抽不到时间的实体/关系兜底 |
| `query_graph` | **Neo4j 多跳查询** `-[:REL*1..N]-`，支持有向/无向 |
| `search_similar_chunks` | **Qdrant 向量相似度查询** |
| `search_fulltext` | **SQLite FTS5** 全文检索（trigram，中文可用） |
| `hybrid_search` | **多路混合检索**：LLM 拆子问题 → 向量 + FTS5 多路并行 → RRF 融合 |
| `get_graph_snapshot` | 实体星球数据源（节点 + 有向/无向连线） |
| `manage_graph` | 图库增删改：加/改/删实体、加/删关系 |
| `run_tools_parallel` | 并行调用多个工具（`asyncio.gather`） |
| `get_stats` | 三库运行状态 |

### 5.5 一句话入库的时间规则

1. LLM 先检查这句话里有没有时间（可具体到**年/月/日/时**），有就如实抽取填 `time`；
2. 抽不到就留空串；
3. 流水线检测到有缺失时，调用 `get_current_time` 工具取系统时间，
   给**每个实体、每条关系**都打上时间标记。

---

## 6. API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/tools` | 所有工具 schema + 动态提示词（前端据此渲染表单） |
| POST | `/api/tools/rediscover` | 重新发现工具 |
| POST | `/api/tools/{name}/call` | 统一工具调用入口 `{"args": {...}}` |
| POST | `/api/tools/call_many` | 并行调用多个工具 |
| GET | `/api/chunks/pending?limit&preview_chars` | 待入库列表（截断预览） |
| POST | `/api/chunks/{chunk_id}/reingest` | 重新入库（前端按钮真实入口） |
| GET | `/api/chunks` / `/api/documents` / `/api/graph` | 已转正 chunk / 原文 / 图谱 |
| POST | `/api/ingest/sentence` | 非流式一句话入库 |
| POST | `/api/ingest/stream` | **SSE 流式**一句话入库，实时推送 LLM 增量与各阶段事件 |
| POST | `/api/llm/parse` | 直接把原始输出丢给解析器 |
| GET | `/api/stats` / `/api/llm/test` | 三库状态 / LLM 与 Embedding 连通性探测 |

---

## 7. 前端页面

| Tab | 内容 |
|---|---|
| 🪐 **实体星球** | 自研 canvas 力导向图：节点=实体（半径随度数、颜色随类型），
  连线=关系，**实线箭头=有向，虚线=无向**，线上标注谓词与时间；可拖拽/缩放/悬浮查看详情 |
| ✨ **一句话入库** | 输入框 + Ctrl+Enter，SSE 实时看 LLM 增量输出、阶段事件、抽取出的实体与关系 |
| 📥 **待入库队列** | 未入库 chunk 的**可折叠列表**，正文截断为前 10 个字符；
  每条后面有「重新入库」按钮 → 点击立即禁用并显示「入库中」→ 真实发起请求 |
| 🧬 **多路混合检索** | 子问题拆解结果、各路原始命中、RRF 融合排序（标注命中了哪几路与每路排名） |
| 🔭 **多跳/向量查询** | Neo4j 多跳 与 Qdrant 向量检索，可一个按钮并行执行 |
| 🧰 **工具台** | 由 `/api/tools` 的 schema **动态生成**调用表单，可直接执行任意工具；
  同时展示自动生成的系统提示词 |
| 📊 **监控台** | 三库指标、LLM/Embedding 连通性、当前配置、表结构 |

---

## 8. 多路混合检索说明（向量 + FTS5 + RRF）

```
用户提问
   │
   ├─▶ LLM 拆成 N 个可独立检索的子问题
   │
   ├─▶ 对每个子问题 × 每条检索路并行执行
   │      ├── 向量路：embedding → qdrant 余弦最近邻（语义召回）
   │      └── 词法路：sqlite FTS5 trigram 全文检索（关键词精准召回）
   │
   └─▶ RRF 融合：  score(d) = Σ 1 / (k + rank_route(d)),   k 默认 60
```

- **只按排名融合，不比绝对分值**，天然解决不同引擎打分量纲不可比的问题；
- **被多路同时命中的 chunk 排更前**（多次累加），测试中 overlap 可达 0.6；
- 中文检索用 FTS5 **trigram** 分词器（3 字以上子串可命中），查询串会被切成
  「短词条 + 长词条 n-gram 窗口」以提高召回，并按 FTS5 内置 `bm25()` 排序；
- `use_llm_split=false` 可跳过拆解直接单路查询；`routes` 可关掉任意一路。

---

## 9. 自检脚本

```bash
python _selftest.py          # 工具发现 / 解析器 / 参数校验 / 并行调用
python _e2e_test.py          # 全链路：入库→复用→状态机→失败重试→恢复转正
python _hybrid_test.py       # 多路混合检索 + RRF
python _neo4j_verify.py      # Neo4j 每一条 Cypher 的实机验证（需本地 neo4j 已启动）
python _integration_test.py  # 真实 neo4j + 真实 qdrant + 真实 sqlite 全集成
python _seed_demo.py         # 灌入一份演示数据，打开前端即可看到实体星球
python _fake_llm_server.py   # 本地假 LLM（OpenAI 流式 + 硅基流动 embeddings 协议）
python _fake_check.py        # 假 LLM 服务协议自检
python _http_check.py        # HTTP 接口自检（需后端在跑）
python _http_e2e.py          # HTTP 端到端：真实入库 + SSE 流式 + 三库查询
```

实测结果：`_neo4j_verify` 58 PASS / `_integration_test` 51 PASS / `_e2e_test` / `_hybrid_test` / HTTP e2e 全绿。

> `_e2e_test.py` / `_hybrid_test.py` 用 `QDRANT_LOCAL_PATH` 让真实 qdrant 引擎
> 跑在进程内，因此不依赖外部服务即可验证完整链路。
> `_http_e2e.py` 走真实 HTTP + SSE，配合 `_fake_llm_server.py` 可离线验证
> LLM 流式解析与硅基流动 embeddings 的请求/响应处理。

---

## 10. 常见问题

**Q：启动后「监控台」neo4j 一直红？**
没启动 Neo4j。本地起一个即可（默认 `bolt://127.0.0.1:7687`），
首次登录需改密：`ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '你的密码'`，
然后同步改 `.env` 的 `NEO4J_PASSWORD`。

**Q：Qdrant 报维度不一致？**
集合创建后维度固定。换模型/维度时删掉集合重建，或换 `QDRANT_COLLECTION` 名字。

**Q：只想本地跑通不想装 Qdrant？**
`.env` 设 `QDRANT_LOCAL_PATH=./data/qdrant`。注意：嵌入式模式同一时刻
只允许**一个进程**打开同一个目录。

**Q：LLM 返回的不是 JSON？**
解析器会去围栏、截取 JSON 块、甚至修复被 `max_tokens` 截断的输出；
仍失败时工具会返回带 `warnings` 的结果，chunk 留在队列里可重试。

**Q：Windows 上 `neo4j.bat` / `.ps1` 报 `spawn powershell.exe ENOENT`？**
Neo4j 5.x 的 Windows 脚本是 PowerShell 包装，某些受限沙箱不允许再 spawn
`powershell.exe`。可以直接用 java 起，绕过这些脚本：

```powershell
# 设初始密码（首次启动前）
& "$env:JAVA_HOME\bin\java.exe" -cp "<NEO4J_HOME>\lib\*" -Dbasedir="<NEO4J_HOME>" `
    org.neo4j.server.startup.Neo4jAdminCommand dbms set-initial-password "你的密码"

# 启动（console 前台，用 Start-Process 可后台）
$p = Start-Process -FilePath "$env:JAVA_HOME\bin\java.exe" -ArgumentList @(
  "-cp", "<NEO4J_HOME>\lib\*", "-Dbasedir=<NEO4J_HOME>",
  "org.neo4j.server.startup.Neo4jCommand", "console") `
  -WorkingDirectory "<NEO4J_HOME>" -WindowStyle Hidden -PassThru
```

等 `http://127.0.0.1:7474` 返回 200 即就绪。注意 `conf/neo4j.conf` 里
同一个 key 不能出现两次（会报 `declared multiple times`）。
