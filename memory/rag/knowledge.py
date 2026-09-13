"""LLM knowledge extraction and graph materialization helpers.

The extractor is deliberately separated from persistence.  An LLM may suggest
entities and relations, but only validated Pydantic data reaches the memory
store.  This also makes the ingestion path easy to test with a deterministic
extractor when no provider is configured.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from difflib import SequenceMatcher
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from constants import (
    DEFAULT_DOMAIN,
    ENTITY_DEFAULT_CONFIDENCE,
    ENTITY_DEFAULT_TYPE,
    ENTITY_NAME_MAX_LENGTH,
    ENTITY_PREFIX_MIN_LENGTH,
    ENTITY_SIMILARITY_THRESHOLD,
    GRAPH_CONTEXT_MAX_RELATIONS,
    RAG_CONTEXT_MAX_CHARS,
)

from ..base import MemoryItem, MemoryType, utc_now
from ..ids import (
    _clean_text,
    entity_id_for,
    normalize_entity_name,
    predicate_key_for,
    relation_id_for,
)
from ..manager import MemoryManager

#: 前缀命中允许的分隔符：较短的名字必须是完整前缀，且后面紧跟这些字符之一，
#: 或者两者完全相等。``web`` 命中 ``web 中转站``、``deepseek`` 命中
#: ``deepseek-v4.1-flash``；而 ``web`` 不会命中 ``webfoo``，``hub`` 不会命中 ``hubby``。
ENTITY_PREFIX_BOUNDARIES = frozenset({" ", "-", "_", ".", "/", "|", ":", "·", "（", "("})


def _normalize_for_match(value: str) -> str:
    """Lowercase and collapse whitespace while keeping word separators.

    ``normalize_entity_name`` strips punctuation for stable ids, which erases
    the word boundary needed to tell ``web``/``web 中转站`` from an unrelated
    ``webfoo``. Matching therefore uses this separators-preserving variant.
    """

    value = _clean_text(value, max_length=ENTITY_NAME_MAX_LENGTH).casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip("".join(sorted(ENTITY_PREFIX_BOUNDARIES - {" "}))).strip()


def _starts_at_boundary(shorter: str, longer: str) -> bool:
    """True when ``shorter`` prefixes ``longer`` only at a word boundary."""

    if not shorter or not longer.startswith(shorter):
        return False
    if len(shorter) == len(longer):
        return True
    return longer[len(shorter)] in ENTITY_PREFIX_BOUNDARIES


def is_prefix_match(left_key: str, right_key: str) -> bool:
    """True when one matching key is a word-boundary prefix of the other.

    A complete prefix plus a separator keeps a bare hub name pointing at
    ``web 中转站`` and ``deepseek`` at ``deepseek-v4.1-flash``, while never
    merging different entities such as ``web``/``world`` or ``hub``/``hubby``.
    """

    shorter, longer = (
        (left_key, right_key)
        if len(left_key) <= len(right_key)
        else (right_key, left_key)
    )
    if len(shorter) < ENTITY_PREFIX_MIN_LENGTH:
        return False
    return _starts_at_boundary(shorter, longer)


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=ENTITY_NAME_MAX_LENGTH)
    entity_type: str = Field(default=ENTITY_DEFAULT_TYPE, max_length=80)
    description: str = Field(default="", max_length=1000)
    confidence: float = Field(default=ENTITY_DEFAULT_CONFIDENCE, ge=0, le=1)
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
    #: 补丁动作：assert 断言、supersede 更新（旧值退场）、retract 撤回。
    action: Literal["assert", "supersede", "retract"] = "assert"
    #: single 表示该 (subject, predicate) 只能有一个当前值；multi 可并列累积。
    cardinality: Literal["single", "multi"] = "multi"
    confidence: float = Field(default=0.75, ge=0, le=1)
    evidence: str = Field(default="", max_length=1200)

    @field_validator("subject", "predicate", "object", "evidence", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> str:
        return _clean_text(value, max_length=1200)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    domain: str = Field(default=DEFAULT_DOMAIN, min_length=1, max_length=100)
    topics: list[str] = Field(default_factory=list, max_length=20)
    entities: list[EntityCandidate] = Field(default_factory=list, max_length=50)
    relations: list[RelationCandidate] = Field(default_factory=list, max_length=80)
    keywords: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("domain", mode="before")
    @classmethod
    def normalize_domain(cls, value: Any) -> str:
        return _clean_text(value, max_length=100) or DEFAULT_DOMAIN

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

    SYSTEM_PROMPT = (
        "你是知识图谱抽取器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。\n"
        "任务：把文本转换成对既有知识图的补丁（patch），而不是孤立的事实快照。\n"
        "\n"
        "【实体】\n"
        "1. 若文本提到「已知实体」里的对象，name 必须原样使用该已知规范名，不要新造变体。\n"
        "2. 文本出现的其他写法、简称、旧称、别名，一律放进该实体的 aliases，不要另开实体。\n"
        "3. 实体名必须简短、稳定、可复用；不要把整句、代词、泛指词当实体。\n"
        "\n"
        "【关系动作 action】\n"
        "1. assert：新增或强化。文本给出了新成立的事实。\n"
        "2. supersede：更新。文本表示某个既有取值被替换（例如余额从 1 元变成 0 元、"
        "状态、当前版本、现居地、负责人、价格发生变化）。\n"
        "3. retract：撤回。文本明确表示某条关系不再成立，且没有给出新取值。\n"
        "\n"
        "【取值基数 cardinality】\n"
        "1. single：同一 (subject, predicate) 只能有一个当前值，出现新值即旧值失效。"
        "典型：余额、当前版本、状态、价格、负责人、所在地。\n"
        "2. multi：可并列累积，多条同时成立。典型：支持、属于、部署于、位于、别名、包含。\n"
        "3. 凡是 single，或文本表达「更新/改成/不再是/现在没有了」，action 用 supersede。\n"
        "4. 判断不确定时用 cardinality=multi、action=assert；不确定的新值不要猜测。\n"
        "\n"
        "【约束】\n"
        "1. subject/predicate/object 必须都能在文本或已知图中找到依据；无法确认的关系不要输出。\n"
        "2. 不要输出「历史值」的关系，历史值由程序按 supersede 自动退场。\n"
        "3. 每条关系必须给出 evidence，且 evidence 必须是原文片段。\n"
        "4. 单次最多 50 个实体、80 条关系；宁少勿滥。\n"
        "\n"
        "字段格式：domain:string, topics:string[], "
        "entities:[{name,entity_type,description,confidence,aliases}], "
        "relations:[{subject,predicate,object,action,cardinality,confidence,evidence}], "
        "keywords:string[]。"
    )

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

    def extract(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        graph_context: str = "",
    ) -> ExtractionResult:
        if not isinstance(text, str) or not text.strip():
            return ExtractionResult()
        source = _clean_text((metadata or {}).get("filename") or (metadata or {}).get("source"), max_length=300)
        payload: dict[str, Any] = {"source": source, "text": text[:12000]}
        if graph_context.strip():
            payload["已知图"] = graph_context
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
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
    """Resolve extracted names to stable semantic-memory entity records.

    Matching order is exact normalized name, exact alias, prefix, then fuzzy
    similarity. Aliases are indexed explicitly, so aggregate labels such as an
    endpoint name written in full always converge on the same planet.
    """

    def __init__(self, manager: MemoryManager, *, similarity_threshold: float = ENTITY_SIMILARITY_THRESHOLD) -> None:
        self.manager = manager
        self.similarity_threshold = similarity_threshold
        self._entities: dict[str, MemoryItem] = {}
        self._index: dict[str, MemoryItem] = {}
        self._load()

    def _load(self) -> None:
        self._entities.clear()
        self._index.clear()
        for item in self.manager.semantic.list():
            if item.metadata.get("kind") != "entity":
                continue
            self._remember(item)

    def _remember(self, item: MemoryItem) -> None:
        """Index one entity under its canonical name and every known alias."""

        metadata = item.metadata
        canonical = str(metadata.get("canonical_name") or metadata.get("title") or item.content)
        key = _normalize_for_match(canonical)
        if not key:
            return
        self._entities[key] = item
        self._index.setdefault(key, item)
        for alias in metadata.get("aliases") or []:
            alias_key = _normalize_for_match(str(alias))
            if alias_key:
                self._index[alias_key] = item

    def _exact(self, key: str) -> MemoryItem | None:
        return self._entities.get(key) or self._index.get(key)

    def _alias(self, key: str) -> MemoryItem | None:
        item = self._index.get(key)
        if item is None:
            return None
        canonical = _normalize_for_match(
            str(item.metadata.get("canonical_name") or item.content)
        )
        return item if key != canonical else None

    def _prefix_candidate(self, key: str) -> MemoryItem | None:
        """Longest full-prefix match wins; partial character overlap never does.

        Read-only on purpose: the alias a prefix hit is worth is the name the
        caller actually wrote, not the normalized lookup key, so
        :meth:`resolve` records it.
        """

        best: MemoryItem | None = None
        best_length = 0
        for known_key, item in list(self._index.items()):
            if not is_prefix_match(key, known_key):
                continue
            shared = min(len(key), len(known_key))
            if shared <= best_length:
                continue
            best, best_length = item, shared
        return best

    def _store_alias(self, item: MemoryItem, name: str) -> MemoryItem:
        """Store ``name`` as an alias of ``item`` and repoint every index key.

        The lookup index keeps normalized keys, so recording a spelling never
        widens prefix matching. Every key that pointed at the pre-write
        snapshot is repointed at the stored object; otherwise the next
        ``resolve`` would read the stale snapshot back and drop the alias.
        """

        canonical = str(item.metadata.get("canonical_name") or item.content)
        metadata = dict(item.metadata)
        aliases = {
            str(value) for value in metadata.get("aliases", []) if str(value).strip()
        }
        aliases.add(name)
        metadata["aliases"] = sorted(aliases)
        stored = self.manager.semantic.add(
            canonical,
            metadata=metadata,
            importance=item.importance,
            item_id=item.id,
        )
        for key, indexed in list(self._index.items()):
            if indexed.id == item.id:
                self._index[key] = stored
        self._entities[_normalize_for_match(canonical)] = stored
        return stored

    def match(self, name: str) -> MemoryItem | None:
        """Return the entity a name refers to, or None when nothing matches."""

        key = _normalize_for_match(name)
        if not key:
            return None
        return (
            self._exact(key)
            or self._alias(key)
            or self._prefix_candidate(key)
        )

    def resolve(
        self,
        name: str,
        *,
        domain: str,
        entity_type: str = ENTITY_DEFAULT_TYPE,
        description: str = "",
        confidence: float = ENTITY_DEFAULT_CONFIDENCE,
        aliases: list[str] | None = None,
        source_id: str | None = None,
    ) -> str:
        name = _clean_text(name, max_length=ENTITY_NAME_MAX_LENGTH)
        key = _normalize_for_match(name)
        if not key:
            raise ValueError("entity name must not be empty")
        existing = self._exact(key) or self._alias(key)
        # Only an exact canonical/alias hit may donate the written name as a new
        # alias, so a lookalike name never pollutes the entity it resembles.
        name_is_known = existing is not None
        if existing is None:
            existing = self._prefix_candidate(key)
            if existing is not None:
                written = str(existing.metadata.get("canonical_name") or existing.content)
                if name != written and key not in self._index:
                    existing = self._store_alias(existing, name)
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
        if name != canonical and name_is_known:
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
                "domain": domain or metadata.get("domain", DEFAULT_DOMAIN),
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
        self._remember(item)
        return canonical


def materialize_extraction(
    manager: MemoryManager,
    extraction: ExtractionResult,
    *,
    source_item: MemoryItem,
    source_metadata: Mapping[str, Any] | None = None,
    relation_threshold: float = 0.6,
    resolver: EntityResolver | None = None,
) -> dict[str, Any]:
    """Persist entities and evidence-backed relation patches from one chunk.

    The extractor only proposes patches. Every graph mutation is executed here
    with deterministic ids:

    - ``assert``    add or strengthen one edge;
    - ``supersede`` retire the other current values of a single-valued slot,
      then add the new one;
    - ``retract``   retire the named edge without deleting its history.

    Retired edges stay in SQLite with ``active=false`` so the graph keeps an
    audit trail while current-state queries only see live edges.
    """

    metadata = dict(source_metadata or {})
    source = metadata.get("filename") or metadata.get("source") or ""
    resolver = resolver if resolver is not None else EntityResolver(manager)
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
        canonical_by_key[_normalize_for_match(candidate.name)] = canonical
        entities += 1

    # One index covers every patch in this chunk; the resolver mirror stays valid
    # because every fact write below goes through add_fact only. The index maps a
    # slot to *all* its facts: a single-valued slot can legitimately hold several
    # active rows (历史数据、并发抽取), and supersede must retire every one of
    # them rather than whichever row happened to be cached first.
    known_items: dict[str, list[MemoryItem]] = {}
    facts_indexed = False

    def slot_key(subject: str, predicate: str) -> str:
        return f"{normalize_entity_name(subject)}|{normalize_entity_name(predicate)}"

    def index_all_facts() -> None:
        nonlocal facts_indexed
        if facts_indexed:
            return
        facts_indexed = True
        for item in manager.semantic.facts():
            subject = str(item.metadata.get("subject") or "")
            predicate = str(item.metadata.get("predicate") or "")
            if subject and predicate:
                known_items.setdefault(slot_key(subject, predicate), []).append(item)

    def remember(item: MemoryItem) -> None:
        """Refresh the cached copies after one fact write in this chunk."""
        subject = str(item.metadata.get("subject") or "")
        predicate = str(item.metadata.get("predicate") or "")
        if not subject or not predicate:
            return
        index_all_facts()
        slot = known_items.setdefault(slot_key(subject, predicate), [])
        for position, cached in enumerate(slot):
            if cached.id == item.id:
                slot[position] = item
                return
        slot.append(item)

    def facts_for(subject: str, predicate: str) -> list[MemoryItem]:
        """Return every active fact sharing one (subject, predicate) slot."""
        index_all_facts()
        return [
            item
            for item in known_items.get(slot_key(subject, predicate), [])
            if item.metadata.get("active", True) is not False
        ]

    def resolve_endpoint(name: str) -> str:
        cached = canonical_by_key.get(_normalize_for_match(name))
        if cached:
            return cached
        return resolver.resolve(
            name, domain=extraction.domain, source_id=source_item.id
        )

    relations = 0
    superseded = 0
    retracted = 0
    skipped_relations = 0
    relation_items: list[MemoryItem] = []
    now = utc_now().isoformat()
    for candidate in extraction.relations:
        if candidate.confidence < relation_threshold:
            skipped_relations += 1
            continue
        subject = resolve_endpoint(candidate.subject)
        object_name = resolve_endpoint(candidate.object)
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

        if candidate.action == "retract":
            if existing is None:
                skipped_relations += 1
                continue
            retracted_metadata = dict(existing_metadata)
            retracted_metadata.update(
                {
                    "active": False,
                    "superseded_at": now,
                    "source_ids": sorted(source_ids),
                    "evidence_items": evidence_items[-20:],
                }
            )
            relation_items.append(
                manager.semantic.add_fact(
                    subject,
                    candidate.predicate,
                    object_name,
                    metadata=retracted_metadata,
                    confidence=float(existing.importance),
                    item_id=relation_id,
                )
            )
            remember(relation_items[-1])
            retracted += 1
            continue

        retire = candidate.action == "supersede" or candidate.cardinality == "single"
        superseded_by: list[str] = []
        if retire:
            for stale in facts_for(subject, candidate.predicate):
                if stale.id == relation_id:
                    continue
                if stale.metadata.get("active", True) is False:
                    continue
                stale_metadata = dict(stale.metadata)
                stale_metadata.update(
                    {
                        "active": False,
                        "superseded_at": now,
                        "superseded_by": relation_id,
                    }
                )
                remember(
                    manager.semantic.add_fact(
                        subject,
                        candidate.predicate,
                        str(stale.metadata.get("object") or ""),
                        metadata=stale_metadata,
                        confidence=float(stale.importance),
                        item_id=stale.id,
                    )
                )
                superseded_by.append(stale.id)
                superseded += 1

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
            "predicate_key": predicate_key_for(subject, candidate.predicate),
            "action": candidate.action,
            "cardinality": candidate.cardinality,
            "active": True,
            "superseded_by": [],
            "superseded_at": "",
            "supersedes": superseded_by,
        }
        written = manager.semantic.add_fact(
            subject,
            candidate.predicate,
            object_name,
            metadata=fact_metadata,
            confidence=candidate.confidence,
            item_id=relation_id,
        )
        remember(written)
        relation_items.append(written)
        relations += 1

    return {
        "domain": extraction.domain,
        "topics": extraction.topics,
        "entities": entities,
        "relations": relations,
        "superseded": superseded,
        "retracted": retracted,
        "skipped_relations": skipped_relations,
        "relation_items": relation_items,
    }


def build_graph_context(
    manager: MemoryManager,
    text: str,
    *,
    max_relations: int = GRAPH_CONTEXT_MAX_RELATIONS,
    max_chars: int = RAG_CONTEXT_MAX_CHARS,
) -> str:
    """Render the relevant slice of the existing graph for the extractor prompt.

    Matching is deliberately conservative: an entity matches when its canonical
    name or a known alias appears in the text, or when a distinctive prefix is
    shared. Retrieved entities expand one hop, so the model can reuse canonical
    names and retire the right old value instead of inventing a second planet.
    """

    if (
        isinstance(max_relations, bool)
        or not isinstance(max_relations, int)
        or max_relations < 1
        or isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or max_chars < 1
    ):
        raise ValueError("max_relations and max_chars must be positive integers")
    resolver = EntityResolver(manager)
    seeds: dict[str, MemoryItem] = {}
    folded = (text or "").casefold()
    if not folded.strip():
        return ""
    words = re.findall(r"[\w\u4e00-\u9fff]+", folded)
    for word in words:
        item = resolver.match(word)
        if item is not None:
            seeds.setdefault(item.id, item)
    # Whole-phrase matches cover names that contain separators themselves.
    for key, item in resolver._index.items():
        if is_prefix_match(_normalize_for_match(folded), key):
            seeds.setdefault(item.id, item)
    if not seeds:
        return ""
    # Neighbor names come from the fact metadata, which stores canonical names,
    # so a one-hop context lines up with what the graph projection will draw.
    names = {
        str(item.metadata.get("canonical_name") or item.content)
        for item in seeds.values()
    }
    for item in manager.semantic.facts():
        metadata = item.metadata
        if metadata.get("active", True) is False:
            continue
        subject = str(metadata.get("subject") or "")
        object_name = str(metadata.get("object") or "")
        if subject in names or object_name in names:
            if subject:
                names.add(subject)
            if object_name:
                names.add(object_name)
    lines = ["已知实体（必须复用，不要新造）："]
    for item in seeds.values():
        metadata = item.metadata
        canonical = str(metadata.get("canonical_name") or item.content)
        aliases = [str(value) for value in metadata.get("aliases") or [] if str(value)]
        suffix = f"；别名：{'、'.join(aliases)}" if aliases else ""
        lines.append(f"- {canonical}{suffix}")
    relations: list[tuple[float, str]] = []
    retired: list[tuple[float, str]] = []
    for item in manager.semantic.facts():
        metadata = item.metadata
        subject = str(metadata.get("subject") or "")
        predicate = str(metadata.get("predicate") or "")
        object_name = str(metadata.get("object") or "")
        if not (subject and predicate and object_name):
            continue
        if subject not in names and object_name not in names:
            continue
        line = f"{subject} --{predicate}--> {object_name}"
        confidence = float(metadata.get("confidence", item.importance) or 0.0)
        if metadata.get("active", True) is False:
            retired.append((confidence, f"{line}（历史，已失效）"))
        else:
            relations.append((confidence, line))
    relations.sort(key=lambda entry: -entry[0])
    retired.sort(key=lambda entry: -entry[0])
    lines.append("已知关系（当前有效）：")
    lines.extend(line for _, line in relations[:max_relations])
    lines.extend(line for _, line in retired[: max(1, max_relations // 3)])
    return _fit_lines(lines, max_chars)


def _fit_lines(lines: list[str], max_chars: int) -> str:
    """Join whole lines only: a half-cut relation would mislead the extractor."""

    kept: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) + (1 if kept else 0)
        if used + cost > max_chars:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)


__all__ = [
    "EntityCandidate",
    "EntityResolver",
    "ExtractionResult",
    "KnowledgeExtractor",
    "LLMKnowledgeExtractor",
    "NullKnowledgeExtractor",
    "RelationCandidate",
    "build_graph_context",
    "entity_id_for",
    "is_prefix_match",
    "materialize_extraction",
    "normalize_entity_name",
    "predicate_key_for",
    "relation_id_for",
]
