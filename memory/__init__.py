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
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMINI_EMBEDDING_BASE_URL,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    APIEmbedding,
    BaseEmbedding,
    EmbedServerEmbedding,
    GeminiEmbedding,
    HashEmbedding,
    load_dotenv_once,
)
from .manager import MemoryManager
from .rag import Document, DocumentProcessor, RAGPipeline, RetrievedChunk
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

__all__ = [
    "DEFAULT_EMBEDDING_BASE_URL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GEMINI_EMBEDDING_BASE_URL",
    "DEFAULT_GEMINI_EMBEDDING_MODEL",
    "APIEmbedding",
    "BaseDocumentStore",
    "BaseEmbedding",
    "BaseMemory",
    "BaseVectorStore",
    "Document",
    "DocumentProcessor",
    "EmbedServerEmbedding",
    "EpisodicMemory",
    "GeminiEmbedding",
    "HashEmbedding",
    "InMemoryVectorStore",
    "MemoryConfig",
    "MemoryItem",
    "MemoryManager",
    "MemorySearchResult",
    "MemoryType",
    "Neo4jGraphStore",
    "PerceptualMemory",
    "QdrantVectorStore",
    "RAGPipeline",
    "RetrievedChunk",
    "SQLiteDocumentStore",
    "SemanticMemory",
    "WorkingMemory",
    "cosine_similarity",
    "default_sqlite_path",
    "ensure_datetime",
    "load_dotenv_once",
    "make_default_embedding",
    "utc_now",
]
