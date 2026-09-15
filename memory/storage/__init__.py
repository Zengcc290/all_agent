"""Storage backends for memory records, vectors and graph relations.

- ``document``      ``BaseDocumentStore`` and the SQLite implementation
- ``document_repo`` ``documents``/``chunks`` 真值源（原文 + 分块边界）
- ``vector``        ``BaseVectorStore``, the local in-process index and cosine similarity
- ``qdrant``        the Qdrant vector database adapter
- ``graph``         the Neo4j graph store with an in-memory fallback
"""

from .document import BaseDocumentStore, SQLiteDocumentStore
from .document_repo import ChunkRecord, DocumentRecord, DocumentRepository
from .graph import Neo4jGraphStore
from .qdrant import QdrantVectorStore
from .vector import BaseVectorStore, InMemoryVectorStore, cosine_similarity

__all__ = [
    "BaseDocumentStore",
    "BaseVectorStore",
    "ChunkRecord",
    "DocumentRecord",
    "DocumentRepository",
    "InMemoryVectorStore",
    "Neo4jGraphStore",
    "QdrantVectorStore",
    "SQLiteDocumentStore",
    "cosine_similarity",
]
