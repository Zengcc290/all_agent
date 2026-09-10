"""LLM knowledge extraction and graph materialization helpers.

The extractor is deliberately separated from persistence.  An LLM may suggest
entities and relations, but only validated Pydantic data reaches the memory
store.  This also makes the ingestion path easy to test with a deterministic
extractor when no provider is configured.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from difflib import SequenceMatcher
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..base import MemoryItem, MemoryType
from ..manager import MemoryManager


def _clean_text(value: Any, *, max_length: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_length].strip()


def normalize_entity_name(value: str) -> str:
    """Return a stable comparison key while preserving the display label."""

    value = _clean_text(value, max_length=200).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value, flags=re.UNICODE)


def entity_id_for(name: str) -> str:
    key = normalize_entity_name(name)
    if not key:
        raise ValueError("entity name must contain at least one searchable character")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"entity:{digest}"


def relation_id_for(subject: str, predicate: str, object: str) -> str:
    key = "|".join(
        (normalize_entity_name(subject), normalize_entity_name(predicate), normalize_entity_name(object))
    )
    return f"relation:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}"


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=200)
    entity_type: str = Field(default="概念", max_length=80)
    description: str = Field(default="", max_length=1000)
    confidence: float = Field(default=0.8, ge=0, le=1)
    aliases: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("name", "entity_type", "description", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> str:
        return _clean_text(value, max_length=1000)


class RelationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    object: str = Field(min_length=1, max_length=200)
    confidence: float = Field(default=0.75, ge=0, le=1)
    evidence: str = Field(default="", max_length=1200)

    @field_validator("subject", "predicate", "object", "evidence", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> str:
        return _clean_text(value, max_length=1200)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    domain: str = Field(default="未分类", min_length=1, max_length=100)
    topics: list[str] = Field(default_factory=list, max_length=20)
    entities: list[EntityCandidate] = Field(default_factory=list, max_length=50)
    relations: list[RelationCandidate] = Field(default_factory=list, max_length=80)
    keywords: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("domain", mode="before")
    @classmethod
    def normalize_domain(cls, value: Any) -> str:
        return _clean_text(value, max_length=100) or "未分类"

    @field_validator("topics", "keywords", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("topics and keywords must be lists")
        return [_clean_text(item, max_length=100) for item in value if _clean_text(item, max_length=100)]


class KnowledgeExtractor(Protocol):
    def extract(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> ExtractionResult: ...


class NullKnowledgeExtractor:
    """Safe fallback used when no chat model is configured."""

    def extract(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> ExtractionResult:
        return ExtractionResult()


class LLMKnowledgeExtractor:
    """Extract structured knowledge through an OpenAI-compatible chat client."""

    def __init__(
        self,
        complete: Callable[..., Any],
        *,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        self.complete = complete
        self.model = model
        self.timeout = timeout

    def extract(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> ExtractionResult:
        if not isinstance(text, str) or not text.strip():
            return ExtractionResult()
        source = _clean_text((metadata or {}).get("filename") or (metadata or {}).get("source"), max_length=300)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是知识图谱抽取器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。"
                    "从给定文本中抽取一个最合适的 domain、topics、实体和有文本证据支持的关系。"
                    "实体名称使用简短、稳定、可复用的规范名称；不要把句子或代词当实体。"
                    "关系必须是 subject-predicate-object 三元组；无法确认的内容不要猜测。"
                    "字段格式：domain:string, topics:string[], entities:[{name,entity_type,description,confidence,aliases}],"
                    "relations:[{subject,predicate,object,confidence,evidence}], keywords:string[]。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"source": source, "text": text[:12000]},
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.complete(
            messages,
            model=self.model,
            temperature=0.0,
            timeout=self.timeout,
            stream=False,
        )
        raw = self._content(response)
        return ExtractionResult.model_validate(self._parse_json(raw))

    @staticmethod
    def _content(response: Any) -> str:
        if isinstance(response, str):
            return response
        choices = response.get("choices") if isinstance(response, Mapping) else getattr(response, "choices", None)
        if not choices:
            raise ValueError("knowledge extraction response contained no choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else getattr(choices[0], "message", None)
        content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("knowledge extraction response contained no text")
        return content

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate).strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("knowledge extraction response was not valid JSON")
            value = json.loads(candidate[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("knowledge extraction response must be a JSON object")
        return value


def _entity_similarity(left: str, right: str, aliases: list[str] | None = None) -> float:
    left_key = normalize_entity_name(left)
    right_values = [right, *(aliases or [])]
    scores = [SequenceMatcher(None, left_key, normalize_entity_name(value)).ratio() for value in right_values]
    left_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", left.casefold()))
    right_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", right.casefold()))
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return max(max(scores, default=0.0), overlap)


class EntityResolver:
    """Resolve extracted names to stable semantic-memory entity records."""

    def __init__(self, manager: MemoryManager, *, similarity_threshold: float = 0.88) -> None:
        self.manager = manager
        self.similarity_threshold = similarity_threshold
        self._entities: dict[str, MemoryItem] = {}
        self._load()

    def _load(self) -> None:
        self._entities.clear()
        for item in self.manager.semantic.list():
            if item.metadata.get("kind") != "entity":
                continue
            name = item.metadata.get("canonical_name") or item.metadata.get("title") or item.content
            key = normalize_entity_name(str(name))
            if key:
                self._entities[key] = item

    def resolve(
        self,
        name: str,
        *,
        domain: str,
        entity_type: str = "概念",
        description: str = "",
        confidence: float = 0.8,
        aliases: list[str] | None = None,
        source_id: str | None = None,
    ) -> str:
        name = _clean_text(name, max_length=200)
        key = normalize_entity_name(name)
        if not key:
            raise ValueError("entity name must not be empty")
        existing = self._entities.get(key)
        if existing is None:
            for item in self._entities.values():
                canonical = str(item.metadata.get("canonical_name") or item.content)
                known_aliases = list(item.metadata.get("aliases") or [])
                if _entity_similarity(name, canonical, known_aliases) >= self.similarity_threshold:
                    existing = item
                    break
        item_id = existing.id if existing is not None else entity_id_for(name)
        metadata = dict(existing.metadata if existing is not None else {})
        canonical = str(metadata.get("canonical_name") or (existing.content if existing else name))
        known_aliases = {
            str(value) for value in metadata.get("aliases", []) if str(value).strip()
        }
        if name != canonical:
            known_aliases.add(name)
        if aliases:
            known_aliases.update(str(value).strip() for value in aliases if str(value).strip() and str(value).strip() != canonical)
        source_ids = {
            str(value) for value in metadata.get("source_ids", []) if str(value).strip()
        }
        if source_id:
            source_ids.add(source_id)
        metadata.update(
            {
                "kind": "entity",
                "title": canonical,
                "canonical_name": canonical,
                "entity_type": entity_type or metadata.get("entity_type", "概念"),
                "description": description or metadata.get("description", ""),
                "domain": domain or metadata.get("domain", "未分类"),
                "aliases": sorted(known_aliases),
                "source_ids": sorted(source_ids),
            }
        )
        item = self.manager.semantic.add(
            canonical,
            metadata=metadata,
            importance=max(float(confidence), float(existing.importance) if existing else 0.0),
            item_id=item_id,
        )
        self._entities[normalize_entity_name(canonical)] = item
        return canonical


def materialize_extraction(
    manager: MemoryManager,
    extraction: ExtractionResult,
    *,
    source_item: MemoryItem,
    source_metadata: Mapping[str, Any] | None = None,
    relation_threshold: float = 0.6,
) -> dict[str, Any]:
    """Persist entities and evidence-backed relations from one chunk."""

    metadata = dict(source_metadata or {})
    source = metadata.get("filename") or metadata.get("source") or ""
    resolver = EntityResolver(manager)
    canonical_by_key: dict[str, str] = {}
    entities = 0
    for candidate in extraction.entities:
        canonical = resolver.resolve(
            candidate.name,
            domain=extraction.domain,
            entity_type=candidate.entity_type,
            description=candidate.description,
            confidence=candidate.confidence,
            aliases=candidate.aliases,
            source_id=source_item.id,
        )
        canonical_by_key[normalize_entity_name(candidate.name)] = canonical
        entities += 1

    relations = 0
    skipped_relations = 0
    relation_items: list[MemoryItem] = []
    for candidate in extraction.relations:
        if candidate.confidence < relation_threshold:
            skipped_relations += 1
            continue
        subject = canonical_by_key.get(normalize_entity_name(candidate.subject)) or resolver.resolve(
            candidate.subject, domain=extraction.domain, source_id=source_item.id
        )
        object_name = canonical_by_key.get(normalize_entity_name(candidate.object)) or resolver.resolve(
            candidate.object, domain=extraction.domain, source_id=source_item.id
        )
        relation_id = relation_id_for(subject, candidate.predicate, object_name)
        existing = manager.get(relation_id, memory_type=MemoryType.SEMANTIC)
        existing_metadata = dict(existing.metadata) if existing is not None else {}
        source_ids = {
            str(value)
            for value in existing_metadata.get("source_ids", [])
            if str(value).strip()
        }
        evidence_items = list(existing_metadata.get("evidence_items") or [])
        if source_item.id:
            source_ids.add(source_item.id)
        evidence_record = {
            "source": source,
            "chunk_id": source_item.id,
            "evidence": candidate.evidence or source_item.content[:600],
        }
        if evidence_record not in evidence_items:
            evidence_items.append(evidence_record)
        fact_metadata = {
            "domain": extraction.domain,
            "topics": extraction.topics,
            "source": source,
            "source_document": source,
            "chunk_id": source_item.id,
            "evidence": candidate.evidence or source_item.content[:600],
            "source_ids": sorted(source_ids),
            "evidence_items": evidence_items[-20:],
            "created_by": "llm",
            "extraction_confidence": candidate.confidence,
        }
        relation_items.append(
            manager.semantic.add_fact(
                subject,
                candidate.predicate,
                object_name,
                metadata=fact_metadata,
                confidence=candidate.confidence,
                item_id=relation_id,
            )
        )
        relations += 1
    return {
        "domain": extraction.domain,
        "topics": extraction.topics,
        "entities": entities,
        "relations": relations,
        "skipped_relations": skipped_relations,
        "relation_items": relation_items,
    }


__all__ = [
    "EntityCandidate",
    "EntityResolver",
    "ExtractionResult",
    "KnowledgeExtractor",
    "LLMKnowledgeExtractor",
    "NullKnowledgeExtractor",
    "RelationCandidate",
    "entity_id_for",
    "materialize_extraction",
    "normalize_entity_name",
    "relation_id_for",
]
