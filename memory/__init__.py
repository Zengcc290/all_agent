"""HelloAgents four-layer memory system.

Applications opt in with ``from memory import MemoryManager``; the built-in
``memory.manage`` and ``memory.rag`` agent tools reuse the same package.

Package layout:

- ``base``      data structures (``MemoryItem``, ``MemoryConfig``) and ``BaseMemory``
- ``embedding`` vendor-neutral API embedding (qwen3-embedding-0.6b via OpenAI-compatible endpoint)
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
    make_default_embedding,
    utc_now,
)
from .embedding import (
    APIEmbedding,
    BaseEmbedding,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
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
    "APIEmbedding", "BaseEmbedding", "BaseMemory", "DEFAULT_EMBEDDING_BASE_URL", "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingService", "MemoryConfig", "MemoryItem", "MemoryManager", "MemorySearchResult", "MemoryType",
    "WorkingMemory", "EpisodicMemory", "SemanticMemory", "PerceptualMemory",
    "BaseDocumentStore", "BaseVectorStore", "InMemoryVectorStore", "QdrantVectorStore", "Neo4jGraphStore", "SQLiteDocumentStore", "cosine_similarity",
    "ensure_datetime", "utc_now", "default_sqlite_path", "make_default_embedding",
    "Document", "DocumentProcessor", "RAGPipeline", "RetrievedChunk",
]
