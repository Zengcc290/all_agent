"""Small, dependency-light retrieval augmented generation helpers."""

from .document import Document, DocumentProcessor
from .graph_rag import GraphPath, GraphRAGPipeline, GraphRAGResult
from .knowledge import (
    EntityCandidate,
    EntityResolver,
    ExtractionResult,
    LLMKnowledgeExtractor,
    NullKnowledgeExtractor,
    RelationCandidate,
    RelationRole,
    build_graph_context,
    entity_id_for,
    materialize_extraction,
    normalize_entity_name,
    observation_id_for,
    predicate_key_for,
    relation_id_for,
)
from .pipeline import RAGPipeline, RetrievedChunk

__all__ = [
    "Document", "DocumentProcessor", "EntityCandidate", "EntityResolver",
    "ExtractionResult", "GraphPath", "GraphRAGPipeline", "GraphRAGResult",
    "LLMKnowledgeExtractor", "NullKnowledgeExtractor", "RAGPipeline",
    "RelationCandidate", "RelationRole", "RetrievedChunk", "build_graph_context", "entity_id_for",
    "materialize_extraction", "normalize_entity_name", "observation_id_for", "predicate_key_for",
    "relation_id_for",
]
