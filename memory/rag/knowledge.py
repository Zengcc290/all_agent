"""LLM knowledge extraction and graph materialization helpers.

The extractor is deliberately separated from persistence.  An LLM may suggest
entities and relations, but only validated Pydantic data reaches the memory
store.  This also makes the ingestion path easy to test with a deterministic
extractor when no provider is configured.
"""

from __future__ import annotations

import base64
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
    observation_id_for,
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
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1, max_length=ENTITY_NAME_MAX_LENGTH)
    entity_type: str = Field(default=ENTITY_DEFAULT_TYPE, max_length=80)
    description: str = Field(default="", max_length=1000)
    confidence: float = Field(default=ENTITY_DEFAULT_CONFIDENCE, ge=0, le=1)
    aliases: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("name", "entity_type", "description", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> str:
        return _clean_text(value, max_length=1000)


class RelationRole(BaseModel):
    """One additional participant in an n-ary observation."""

    model_config = ConfigDict(extra="ignore", strict=True)

    role: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=200)
    entity_type: str = Field(default=ENTITY_DEFAULT_TYPE, max_length=80)

    @field_validator("role", "value", "entity_type", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> str:
        return _clean_text(value, max_length=200)


class RelationCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    object: str = Field(min_length=1, max_length=200)
    #: 补丁动作：assert 断言、supersede 更新（旧值退场）、retract 撤回。
    action: Literal["assert", "supersede", "retract"] = "assert"
    #: temporal 表示同一槽位按时间追加观察，不覆盖历史。
    cardinality: Literal["single", "multi", "temporal"] = "multi"
    #: 除主语/宾语外的任意参与者，例如地点、活动、设备、人员。
    roles: list[RelationRole] = Field(default_factory=list, max_length=20)
    #: F4 时间/状态分类：关系成立/失效时间点与时间段，空值视为无界。
    valid_from: str = Field(default="", max_length=60)
    valid_to: str = Field(default="", max_length=60)
    #: fact 现在成立 / plan 计划中 / expired 已失效 / uncertain 不确定（降权不排除）。
    status: Literal["fact", "plan", "expired", "uncertain", ""] = "fact"
    #: 事件发生时刻（区别于关系写入时刻）。
    event_at: str = Field(default="", max_length=60)
    confidence: float = Field(default=0.75, ge=0, le=1)
    evidence: str = Field(default="", max_length=1200)

    @field_validator("subject", "predicate", "object", "evidence", "valid_from", "valid_to", "event_at", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> str:
        return _clean_text(value, max_length=1200)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: Any) -> str:
        return _clean_text(value, max_length=20) if value is not None else ""


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

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
    def extract(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        graph_context: str = "",
        image: Any = None,
        mime_type: str = "image/jpeg",
    ) -> ExtractionResult: ...


class NullKnowledgeExtractor:
    """Safe fallback used when no chat model is configured."""

    def extract(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        graph_context: str = "",
        image: Any = None,
        mime_type: str = "image/jpeg",
    ) -> ExtractionResult:
        return ExtractionResult()


def response_content(response: Any) -> str:
    """Read the assistant text out of a chat-completion response (or a raw string).

    Shared by the extractor and by ``tool/multi_recall``'s query decomposer, so
    both accept the same provider response shapes.
    """

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


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating ```json fences and surrounding prose."""

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
        raise TypeError("knowledge extraction response must be a JSON object")
    return value


class LLMKnowledgeExtractor:
    """Extract structured knowledge through an OpenAI-compatible chat client."""

    SYSTEM_PROMPT = (
        "你是知识图谱抽取器。只输出一个合法 JSON 对象，不要 Markdown、解释或额外文字。\n"
        "任务：把文本或图片转成对现有知识图的补充（patch），而不是独立的快照。\n"
        "\n"
        "【实体】\n"
        "1. 抽取文本中出现的【所有】实体：人物、地点、组织、系统、模型、概念、文件、事件主体等，宁多勿漏。\n"
        "2. 若文本提到「已知实体」里的对象，name 必须原样使用该已知规范名，不要新造变体；不确定时优先复用已知实体。\n"
        "3. 同一实体的其它写法、简称、旧称、别名一律放进 aliases，不要另开实体。\n"
        "4. 实体名必须简短、稳定、可复用；不要把整句、代词、泛指词当实体。\n"
        "\n"
        "【关系动作 action】\n"
        "1. assert：新增或强化。文本给出了新成立的事实。\n"
        "2. supersede：更新。文本表示某个既有取值被替换。\n"
        "3. retract：撤回。文本明确表示某条关系不再成立，且没有给出新取值。\n"
        "\n"
        "【取值基数 cardinality】\n"
        "1. single：同一 (subject, predicate) 只能有一个当前值，出现新值即旧值失效。\n"
        "2. multi：可并列累积，多条同时成立。\n"
        "3. temporal：有 event_at 的位置、状态、活动等按时间追加，不覆盖其它时刻。\n"
        "4. 带明确观察时间且没有纠错语义时，使用 cardinality=temporal、action=assert。\n"
        "\n"
        "【多元关系与时间】\n"
        "1. 一个事实除 subject/predicate/object 外还有地点、活动、工具、人员等参与者时，放进 roles。\n"
        "2. roles 格式为 {role,value,entity_type}；不要把时间重复放进 roles，时间写 event_at。\n"
        "3. 相对时间以输入里的 reference_time 为基准；图片拍摄时间优先使用 captured_at。\n"
        "\n"
        "【领域 domain】\n"
        "1. 优先复用「已知领域」里的现有领域名，不要为同一主题新建领域。\n"
        "2. 仅当文本构成一个已知领域之外的新主题（例如一批围绕新题材的实体/关系），"
        "且无法归入任何已知领域时，才新建领域名；新领域名 2~10 个汉字，简洁概括主题。\n"
        "3. 禁止使用「未分类」「其他」「默认」作为领域名；"
        "拿不准时复用与实体关系最相关的现有领域。\n"
        "\n"
        "【约束】\n"
        "1. subject/predicate/object 必须都能在文本或已知图中找到依据；无法确认的关系不要输出。\n"
        "2. 每条关系必须给出 evidence，且 evidence 必须是原文片段。\n"
        "3. 单次最多 50 个实体、80 条关系；宁少勿滥。\n"
        "4. 简单句也必须抽取。例如「小猫和小狗是亲兄弟」应输出 entities:[{name:小猫},{name:小狗}] 与 relations:[{subject:小猫,predicate:亲兄弟,object:小狗,action:assert,cardinality:multi,evidence:小猫和小狗是亲兄弟}]。\n"
        "5. 关系描述必须具体：不要用「相关」「关联」「有关系」这类含糊词；用明确的动词或短语。\n"
        "6. predicate 最多 10 个汉字（或等长字符），例如「包含子系统」「毕业于」「在」「是」「支持」「运行于」。\n"
        "字段格式：domain:string, topics:string[], "
        "entities:[{name,entity_type,description,confidence,aliases}], "
        "relations:[{subject,predicate,object,action,cardinality,roles:[{role,value,entity_type}],"
        "valid_from,valid_to,status,event_at,confidence,evidence}], "
        "keywords:string[]。"
    )
    def __init__(
        self,
        complete: Callable[..., Any],
        *,
        model: str | None = None,
        vision_model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        self.complete = complete
        self.model = model
        self.vision_model = vision_model
        self.timeout = timeout

    def extract(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        graph_context: str = "",
        image: Any = None,
        mime_type: str = "image/jpeg",
    ) -> ExtractionResult:
        if (not isinstance(text, str) or not text.strip()) and image is None:
            return ExtractionResult()
        metadata = dict(metadata or {})
        source = _clean_text(
            metadata.get("filename") or metadata.get("source"), max_length=300
        )
        payload: dict[str, Any] = {
            "source": source,
            "text": text[:12000] if isinstance(text, str) else "",
            "reference_time": metadata.get("reference_time") or utc_now().isoformat(),
        }
        for key in ("captured_at", "event_at", "modality"):
            if metadata.get(key):
                payload[key] = metadata[key]
        if graph_context.strip():
            payload["已知图"] = graph_context
        prompt = json.dumps(payload, ensure_ascii=False)
        if image is None:
            user_content: Any = prompt
            selected_model = self.model
        else:
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image, mime_type)},
                },
            ]
            selected_model = self.vision_model or self.model
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = self.complete(
            messages,
            model=selected_model,
            temperature=0.0,
            timeout=self.timeout,
            stream=False,
        )
        raw = response_content(response)
        return ExtractionResult.model_validate(parse_json_object(raw))

    @staticmethod
    def _image_data_url(image: Any, mime_type: str) -> str:
        if isinstance(image, (bytes, bytearray, memoryview)):
            encoded = base64.b64encode(bytes(image)).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        if isinstance(image, str) and image.strip():
            return image.strip()
        raise TypeError("image must be bytes or a non-empty URL/data URI")


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
        # 属性合并/写回/图投影的唯一实现在 tool/graph_node_update.py。
        # 函数内导入：``tool.graph_node_update`` 属于上层能力模块，模块级导入会
        # 形成 memory.rag -> tool -> memory 的初始化环。
        from tool.graph_node_update import update_entity_node

        stored = update_entity_node(
            self.manager,
            name,
            existing=item,
            aliases=[name],
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
        return self._exact(key) or self._prefix_candidate(key)

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
        existing = self._exact(key)
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
        # 节点属性（domain/type/description/aliases/importance/source_ids）的合并与
        # 写回已收敛到 tool/graph_node_update.py 的 update_entity_node：这里是唯一的
        # 属性更新实现，本类只负责「名字 -> 已有实体」的匹配。
        from tool.graph_node_update import update_entity_node

        stored = update_entity_node(
            self.manager,
            name,
            existing=existing,
            domain=domain,
            entity_type=entity_type,
            description=description,
            aliases=aliases,
            importance=confidence,
            source_id=source_id,
            add_written_name=name_is_known,
            item_id=existing.id if existing is not None else entity_id_for(name),
        )
        self._remember(stored)
        return str(stored.metadata.get("canonical_name") or stored.content)


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
        roles = [
            {
                "role": role.role,
                "value": resolve_endpoint(role.value),
                "entity_type": role.entity_type,
            }
            for role in candidate.roles
        ]
        temporal = bool(
            candidate.event_at
            or candidate.valid_from
            or candidate.valid_to
            or candidate.cardinality == "temporal"
            or roles
        )
        relation_id = (
            observation_id_for(
                subject,
                candidate.predicate,
                object_name,
                roles=roles,
                event_at=candidate.event_at,
                valid_from=candidate.valid_from,
                valid_to=candidate.valid_to,
                source_id=source_item.id,
            )
            if temporal
            else relation_id_for(subject, candidate.predicate, object_name)
        )
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

        # Timed observations append history. Only an explicit supersede is
        # allowed to retire another timed observation; ordinary single-value
        # behavior remains unchanged for timeless state facts.
        retire = candidate.action == "supersede" or (
            candidate.cardinality == "single" and not temporal
        )
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
            "roles": roles,
            "observation_id": relation_id if temporal else "",
            "modality": source_item.modality or metadata.get("modality") or "text",
            "captured_at": metadata.get("captured_at", ""),
            # F4：时间/状态透传，空值由检索侧视为无界/默认。
            "valid_from": candidate.valid_from,
            "valid_to": candidate.valid_to,
            "status": candidate.status or "fact",
            "event_at": candidate.event_at,
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
    resolver: EntityResolver | None = None,
    max_relations: int = GRAPH_CONTEXT_MAX_RELATIONS,
    max_chars: int = RAG_CONTEXT_MAX_CHARS,
) -> str:
    """Render the relevant slice of the existing graph for the extractor prompt.

    Matching is deliberately conservative: an entity matches when its canonical
    name or a known alias appears in the text, or when a distinctive prefix is
    shared. Retrieved entities expand one hop, so the model can reuse canonical
    names and retire the right old value instead of inventing a second planet.

    ``resolver`` is injectable so one ingest call can pass its chunk-shared
    :class:`EntityResolver`: building one here would re-scan every stored entity
    once per chunk. A fresh resolver is created when the caller has none, which
    keeps the standalone helper usable.
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
    resolver = resolver if resolver is not None else EntityResolver(manager)
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
    # 已知领域清单：让 LLM 优先复用现有恒星，只有真正新主题才新建。
    known_domains = _known_domains()
    lines = [f"已知领域（优先复用，不要随意新建）：{'、'.join(known_domains)}"]
    lines += ["已知实体（必须复用，不要新造）："]
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


def _known_domains() -> list[str]:
    """已知领域清单（同步 web/domain_classifier 的 KNOWN_DOMAINS）。"""

    try:
        from web.domain_classifier import KNOWN_DOMAINS
    except Exception:  # noqa: BLE001 - 顺便导入失败时返回常规集
        return ["未分类"]
    return list(KNOWN_DOMAINS)


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
    "RelationRole",
    "build_graph_context",
    "entity_id_for",
    "is_prefix_match",
    "materialize_extraction",
    "normalize_entity_name",
    "parse_json_object",
    "predicate_key_for",
    "relation_id_for",
    "response_content",
]
