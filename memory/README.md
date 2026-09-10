# HelloAgents Memory

这是一个暂时独立于现有 Agent runtime 的四层记忆系统。入口是
`memory.MemoryManager`，默认组合 SQLite 文档存储、进程内向量索引和
`APIEmbedding`（qwen3-embedding-0.6b，DashScope OpenAI 兼容端点），
无需启动外部数据库服务即可使用。

## 包结构

```
memory/
├── __init__.py        # 公共导出（from memory import MemoryManager, ...）
├── base.py            # 数据结构（MemoryItem / MemoryConfig）与 BaseMemory
├── embedding.py       # 统一嵌入接口：BaseEmbedding + APIEmbedding（任意 OpenAI 兼容厂商）
├── manager.py         # 记忆管理器（统一协调调度）
├── types/             # 记忆类型实现
│   ├── working.py     # 工作记忆（TTL + 容量淘汰）
│   ├── episodic.py    # 情景记忆（事件时间线）
│   ├── semantic.py    # 语义记忆（知识图谱三元组）
│   └── perceptual.py  # 感知记忆（多模态 payload）
├── storage/           # 存储后端实现
│   ├── document.py    # BaseDocumentStore + SQLite 文档存储
│   ├── vector.py      # BaseVectorStore + 进程内向量索引
│   ├── qdrant.py      # Qdrant 向量数据库适配器
│   └── graph.py       # Neo4j 图存储（带进程内回退）
└── rag/               # RAG 系统
    ├── pipeline.py    # RAG 管道（端到端处理）
    └── document.py    # 文档解析与切块
```

## 快速开始

先提供 embedding API key（默认从环境变量 `DASHSCOPE_API_KEY` 读取，
对应 DashScope 上的 `qwen3-embedding-0.6b` 模型，1024 维）：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-..."
```

```python
from memory import MemoryConfig, MemoryManager

memory = MemoryManager(MemoryConfig(sqlite_path="memory.sqlite3"))
memory.working.set("user_name", "小明")
memory.episodic.record("用户在上海参加了会议")
memory.semantic.add_fact("上海", "位于", "中国")
memory.perceptual.store(b"...", modality="image", content="会议室照片")

for result in memory.search("会议"):
    print(result.item.content, result.score)
```

`MemoryManager` 支持上下文管理器，退出时会关闭文档、向量和图存储：

```python
with MemoryManager(MemoryConfig(sqlite_path="memory.sqlite3")) as memory:
    memory.working.set("request_id", "req-1", ttl_seconds=300)
```

`memory.rag` 提供 `DocumentProcessor` 和 `RAGPipeline`：支持纯文本、
JSON/JSONL、CSV、HTML，以及可选的 PDF 解析（需 `pypdf`）；文档会按块写入
语义记忆，并可通过 `retrieve`/`build_context` 取回上下文。

导入管道可以注入 `LLMKnowledgeExtractor`，自动抽取领域、实体和有证据的
三元组关系，并通过稳定实体 ID 合并重复实体。`GraphRAGPipeline` 在向量命中
之后扩展实体关系路径，把原文证据与图路径一起构建上下文；没有配置聊天模型
时默认使用 `NullKnowledgeExtractor`，只保存原始知识块，不会伪造关系。

## Embedding 接口

嵌入层只保留一个通用实现 `APIEmbedding`，实现 `BaseEmbedding` 的
`embed`/`embed_batch` 接口，走 **OpenAI 兼容的 `/embeddings`** 协议，
因此可以接入任意提供该协议的厂商（DashScope、OpenAI、SiliconFlow、
智谱、本地 vLLM/one-api 网关等）。默认配置指向 DashScope 的
`qwen3-embedding-0.6b`：

```python
from memory import APIEmbedding, MemoryManager, MemoryConfig

# 默认即 qwen3-embedding-0.6b @ DashScope（key 来自 DASHSCOPE_API_KEY）
manager = MemoryManager(MemoryConfig(sqlite_path="memory.sqlite3"))

# 换任意 OpenAI 兼容厂商：只改 base_url / model / api_key
embedding = APIEmbedding(
    api_key="sk-...",
    base_url="https://api.siliconflow.cn/v1",          # 举例：SiliconFlow
    model="BAAI/bge-m3",                                # 举例：任意厂商的模型名
    dimension=1024,
)
manager = MemoryManager(MemoryConfig(sqlite_path="memory.sqlite3"), embedding=embedding)
```

`MemoryConfig` 也支持全环境变量配置（见 `.env.example`）：
`HELLOAGENTS_MEMORY_EMBEDDING_API_KEY`、`..._EMBEDDING_MODEL`、
`..._EMBEDDING_BASE_URL`、`..._EMBEDDING_DIMENSION`、`..._EMBEDDING_TIMEOUT`、
`..._EMBEDDING_BATCH_SIZE`。项目根目录的 `.env` 文件会被自动加载
（`memory.embedding.load_dotenv_once`，需 `python-dotenv`，已列入依赖），
因此最简单的方式就是在 `.env` 里写 `DASHSCOPE_API_KEY=sk-...`。若既没有
显式配置也没有 `DASHSCOPE_API_KEY`，构造管理器时会抛出清晰的错误提示。

验证接入与中文检索效果：

```bash
.venv\Scripts\python.exe check_embedding.py            # 完整验证（需 key）
.venv\Scripts\python.exe check_embedding.py --offline  # 只检查配置
```

## 后端配置

* `SQLiteDocumentStore` 保存标准化 `MemoryItem`，可使用文件路径持久化。
* `QdrantVectorStore` 接受 `url`、`api_key` 和 `collection_name`，并在首次写入时按嵌入维度创建 collection。
* `Neo4jGraphStore` 接受 `uri`、`username`、`password`；未提供 `uri` 时使用进程内图实现，便于测试。`get_relations` 的 `relation` 与 `direction` 过滤对驱动路径同样生效。
* `APIEmbedding` 是嵌入的唯一实现：自动分批（默认 `batch_size=10`）、按响应中的 `index`/`text_index` 还原顺序、校验维度与数值有限性，并兼容 DashScope 原生的 `output.embeddings` 响应格式。

如果 `MemoryConfig.qdrant_url` 已配置且没有显式注入 `vector_store`，管理器会自动使用 Qdrant；否则默认使用进程内向量索引。所有向量都会校验维度和有限数值，空查询直接返回空结果。

可以把后端注入管理器：

```python
from memory import MemoryManager, QdrantVectorStore, Neo4jGraphStore, APIEmbedding

manager = MemoryManager(
    embedding=APIEmbedding(api_key="sk-..."),          # qwen3-embedding-0.6b
    vector_store=QdrantVectorStore(url="http://localhost:6333"),
    graph_store=Neo4jGraphStore("bolt://localhost:7687", "neo4j", "password"),
)
```

Qdrant、Neo4j Python 客户端已经加入项目依赖；数据库服务本身仍需按部署环境单独运行。

## Agent 工具的默认持久化

`memory.manage`（`tool/memory_tool.py`）和 `memory.rag`（`tool/rag_tool.py`）
默认把记忆写入 `MEMORY_DB_PATH` 指向的 SQLite 文件；未设置该环境变量时默认
是项目工作目录下的 `memory.sqlite3`。默认管理器在首次调用工具时才创建，
导入与发现工具不会打开数据库。需要其他后端时，显式注入自定义的
`MemoryManager` / `RAGPipeline` 即可。
