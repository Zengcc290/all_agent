"""HelloAgents four-layer memory system.

Applications opt in with ``from memory import MemoryManager``; the built-in
``memory.manage`` and ``memory.rag`` agent tools reuse the same package.

Package layout:

- ``base``      data structures (``MemoryItem``, ``MemoryConfig``) and ``BaseMemory``
- ``embedding`` DashScope / local-transformer / TF-IDF embedding services
- ``types``     working, episodic, semantic and perceptual memories
- ``storage``   SQLite documents, local/Qdrant vector indexes, Neo4j graph
- ``rag``       document parsing, chunking and the RAG pipeline
- ``manager``   the coordinating ``MemoryManager``
"""

from .base import (
    BaseMemory,
    MemoryConfig,
    MemoryItem,
    MemorySearchResult,
    MemoryType,
    default_sqlite_path,
    ensure_datetime,
    utc_now,
)
from .embedding import (
    BaseEmbedding,
    DashScopeEmbedding,
    EmbeddingService,
    LocalTransformerEmbedding,
    TFIDFEmbedding,
)
from .manager import MemoryManager
from .storage import (
    BaseDocumentStore,
    BaseVectorStore,
    InMemoryVectorStore,
    Neo4jGraphStore,
    QdrantVectorStore,
    SQLiteDocumentStore,
    cosine_similarity,
)
from .types import EpisodicMemory, PerceptualMemory, SemanticMemory, WorkingMemory
from .rag import Document, DocumentProcessor, RAGPipeline, RetrievedChunk

__all__ = [
    "BaseMemory", "MemoryConfig", "MemoryItem", "MemoryManager", "MemorySearchResult", "MemoryType",
    "WorkingMemory", "EpisodicMemory", "SemanticMemory", "PerceptualMemory",
    "BaseEmbedding", "EmbeddingService", "DashScopeEmbedding", "LocalTransformerEmbedding", "TFIDFEmbedding",
    "BaseDocumentStore", "BaseVectorStore", "InMemoryVectorStore", "QdrantVectorStore", "Neo4jGraphStore", "SQLiteDocumentStore", "cosine_similarity",
    "ensure_datetime", "utc_now", "default_sqlite_path",
    "Document", "DocumentProcessor", "RAGPipeline", "RetrievedChunk",
]
