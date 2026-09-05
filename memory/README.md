# HelloAgents Memory

这是一个暂时独立于现有 Agent runtime 的四层记忆系统。入口是
`memory.MemoryManager`，默认组合 SQLite 文档存储、进程内向量索引和
TF-IDF 嵌入，无需启动外部服务即可使用。

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

## 后端配置

* `SQLiteDocumentStore` 保存标准化 `MemoryItem`，可使用文件路径持久化。
* `QdrantVectorStore` 接受 `url`、`api_key` 和 `collection_name`，并在首次写入时按嵌入维度创建 collection。
* `Neo4jGraphStore` 接受 `uri`、`username`、`password`；未提供 `uri` 时使用进程内图实现，便于测试。
* `DashScopeEmbedding`、`LocalTransformerEmbedding` 和 `TFIDFEmbedding` 都实现统一的 `embed`/`embed_batch` 接口。

可以把后端注入管理器：

```python
from memory import MemoryManager, QdrantVectorStore, Neo4jGraphStore

manager = MemoryManager(
    vector_store=QdrantVectorStore(url="http://localhost:6333"),
    graph_store=Neo4jGraphStore("bolt://localhost:7687", "neo4j", "password"),
)
```

Qdrant、Neo4j Python 客户端已经加入项目依赖，并安装在本项目 `.venv` 中；数据库服务本身仍需按部署环境单独运行。
