"""HelloAgents four-layer memory system.

Applications opt in with ``from memory import MemoryManager``; the built-in
agent tools (``memory.query``/``memory.add``/``memory.manage`` and
``memory.rag_search``/``memory.rag``) reuse the same package.

Package layout:

- ``base``      data structures (``MemoryItem``, ``MemoryConfig``) and ``BaseMemory``
- ``embedding`` cloud API embedding (OpenAI-compatible ``/embeddings``，支持
                文本与图文 VL 输入) plus the deterministic offline ``HashEmbedding``
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
    DEFAULT_EMBEDDING_MODEL,
    APIEmbedding,
    BaseEmbedding,
    HashEmbedding,
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
    "DEFAULT_EMBEDDING_MODEL",
    "APIEmbedding",
    "BaseDocumentStore",
    "BaseEmbedding",
    "BaseMemory",
    "BaseVectorStore",
    "Document",
    "DocumentProcessor",
    "EpisodicMemory",
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
    "make_default_embedding",
    "utc_now",
]
