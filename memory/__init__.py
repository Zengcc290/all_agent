"""HelloAgents four-layer memory system.

Applications opt in with ``from memory import MemoryManager``; the built-in
agent tools (``memory.query``/``memory.add``/``memory.manage`` and
``memory.rag_search``/``memory.rag``) reuse the same package.

Package layout:

- ``base``      data structures (``MemoryItem``, ``MemoryConfig``) and ``BaseMemory``
- ``embedding`` API embedding (qwen3-embedding-0.6b via an OpenAI-compatible
                endpoint) plus the deterministic offline ``HashEmbedding``
- ``ids``       stable entity/fact id derivation shared by the RAG writers
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
    HashEmbedding,
    load_dotenv_once,
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
    "EmbeddingService", "HashEmbedding", "MemoryConfig", "MemoryItem", "MemoryManager", "MemorySearchResult", "MemoryType",
    "load_dotenv_once",
    "WorkingMemory", "EpisodicMemory", "SemanticMemory", "PerceptualMemory",
    "BaseDocumentStore", "BaseVectorStore", "InMemoryVectorStore", "QdrantVectorStore", "Neo4jGraphStore", "SQLiteDocumentStore", "cosine_similarity",
    "ensure_datetime", "utc_now", "default_sqlite_path", "make_default_embedding",
    "Document", "DocumentProcessor", "RAGPipeline", "RetrievedChunk",
]
