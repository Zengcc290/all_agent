"""Shared schema pieces and default backends for the memory agent tools.

This module is deliberately NOT discoverable: ``core.discovery`` ignores
modules whose names start with an underscore, and this file defines neither
``TOOL_ENABLED`` nor a ``create_tool()`` factory. Importing it opens no
database and performs no network call; every backend is built on first use
inside the individual tools.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory import MemoryConfig, MemoryManager, default_sqlite_path
from memory.rag import LLMKnowledgeExtractor, NullKnowledgeExtractor, RAGPipeline

LOGGER = logging.getLogger(__name__)

MemoryScope = Literal["working", "episodic", "semantic", "perceptual"]


class MemoryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: str = Field(min_length=1)
    value: str


def normalize_metadata_payload(value: Any) -> Any:
    """Accept the friendly ``{"k": "v"}`` mapping form models often emit."""
    if isinstance(value, dict) and isinstance(value.get("metadata"), dict):
        value = dict(value)
        value["metadata"] = [
            {"key": key, "value": str(item)} for key, item in value["metadata"].items()
        ]
    return value


def metadata_dict(entries: list[MemoryMetadata] | None) -> dict[str, str]:
    """Flatten the tool's metadata list into the manager's mapping form."""

    return {entry.key: entry.value for entry in (entries or [])}


def build_default_manager() -> MemoryManager:
    """Open the shared on-disk memory store used when no manager is injected."""

    # 与 web.support.get_manager 同一口径：HELLOAGENTS_MEMORY_* 控制 Qdrant/Neo4j，
    # SQLite 路径仍由 MEMORY_DB_PATH 优先。
    config = MemoryConfig.from_env()
    config.sqlite_path = default_sqlite_path()
    return MemoryManager(config)


def build_default_pipeline() -> RAGPipeline:
    """Build the default RAG pipeline, degrading to a null extractor.

    A configured provider enables LLM knowledge extraction; anything else
    (missing key, placeholder key, unreadable config) keeps ingestion working
    with ``NullKnowledgeExtractor`` and records why in the log.
    """

    extractor = NullKnowledgeExtractor()
    try:
        from agents.llm import LLM
        from agents.providers import ProviderRegistry
        from core.services_config import load_services_config

        registry = ProviderRegistry()
        profile = registry.get(registry.active_profile)
        key = registry.resolve_api_key(profile.name)
        if key and not key.startswith("replace-with"):
            client = LLM(api_key=key, base_url=profile.base_url, model=profile.default_model)
            # 与 web.support.build_knowledge_extractor 同口径：视觉模型名来自
            # config/services.toml 的 [vision] 段。
            vision_model = load_services_config().vision.model or ""
            extractor = LLMKnowledgeExtractor(
                client.complete,
                model=profile.default_model,
                vision_model=vision_model or profile.default_model,
            )
    except Exception:
        LOGGER.warning(
            "RAG 知识抽取器不可用，降级为 NullKnowledgeExtractor（仅做向量检索）",
            exc_info=True,
        )
    return RAGPipeline(build_default_manager(), extractor=extractor)


__all__ = [
    "LOGGER",
    "MemoryMetadata",
    "MemoryScope",
    "build_default_manager",
    "build_default_pipeline",
    "metadata_dict",
    "normalize_metadata_payload",
]
