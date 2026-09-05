"""Standalone HelloAgents four-layer memory system.

The package is intentionally not imported by the existing agent runtime yet;
applications can opt in with ``from memory import MemoryManager``.
"""

from .base import BaseMemory
from .config import MemoryConfig
from .embeddings import BaseEmbedding, DashScopeEmbedding, EmbeddingService, LocalTransformerEmbedding, TFIDFEmbedding
from .manager import MemoryManager
from .memory_types import EpisodicMemory, PerceptualMemory, SemanticMemory, WorkingMemory
from .models import MemoryItem, MemorySearchResult, MemoryType, ensure_datetime, utc_now
from .storage import BaseDocumentStore, BaseVectorStore, InMemoryVectorStore, Neo4jGraphStore, QdrantVectorStore, SQLiteDocumentStore, cosine_similarity
from .rag import Document, DocumentProcessor, RAGPipeline, RetrievedChunk

__all__ = [
    "BaseMemory", "MemoryConfig", "MemoryItem", "MemoryManager", "MemorySearchResult", "MemoryType",
    "WorkingMemory", "EpisodicMemory", "SemanticMemory", "PerceptualMemory",
    "BaseEmbedding", "EmbeddingService", "DashScopeEmbedding", "LocalTransformerEmbedding", "TFIDFEmbedding",
    "BaseDocumentStore", "BaseVectorStore", "InMemoryVectorStore", "QdrantVectorStore", "Neo4jGraphStore", "SQLiteDocumentStore", "cosine_similarity",
    "ensure_datetime", "utc_now",
    "Document", "DocumentProcessor", "RAGPipeline", "RetrievedChunk",
]
