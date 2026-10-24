"""数据存储层。

注意：这里的单例工厂函数特意用与子模块不同的名字，避免
`app.db.qdrant_store`（模块）与同名实例互相遮蔽。
"""
from .sqlite_store import SQLiteStore, store
from .qdrant_store import QdrantStore, vector_store
from .neo4j_store import Neo4jStore, graph_store

__all__ = [
    "SQLiteStore", "store",
    "QdrantStore", "vector_store",
    "Neo4jStore", "graph_store",
]
