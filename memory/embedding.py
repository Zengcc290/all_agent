"""Compatibility alias for integrations using the singular module name."""

from .embeddings import (
    BaseEmbedding,
    DashScopeEmbedding,
    EmbeddingService,
    LocalTransformerEmbedding,
    TFIDFEmbedding,
)

__all__ = [
    "BaseEmbedding",
    "EmbeddingService",
    "DashScopeEmbedding",
    "LocalTransformerEmbedding",
    "TFIDFEmbedding",
]
