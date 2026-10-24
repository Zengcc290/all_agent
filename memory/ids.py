"""Stable identifier helpers for memory entities and facts.

Leaf module by design: it imports only the standard library and ``constants``,
so both ``memory.types.semantic`` and ``memory.rag.knowledge`` can derive the
same ids. Keeping these functions here (instead of inside ``rag.knowledge``)
avoids the import cycle ``manager -> types.semantic -> rag.knowledge -> manager``
and guarantees one single id scheme for facts written by either path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from constants import ENTITY_CLEAN_MAX_LENGTH, ENTITY_NAME_MAX_LENGTH


def _clean_text(value: Any, *, max_length: int = ENTITY_CLEAN_MAX_LENGTH) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_length].strip()


def normalize_entity_name(value: str) -> str:
    """Return a stable comparison key while preserving the display label."""

    value = _clean_text(value, max_length=ENTITY_NAME_MAX_LENGTH).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value, flags=re.UNICODE)


def entity_id_for(name: str) -> str:
    key = normalize_entity_name(name)
    if not key:
        raise ValueError("entity name must contain at least one searchable character")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"entity:{digest}"


def relation_id_for(subject: str, predicate: str, object: str) -> str:
    key = "|".join(
        (
            normalize_entity_name(subject),
            normalize_entity_name(predicate),
            normalize_entity_name(object),
        )
    )
    return f"relation:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}"


def observation_id_for(
    subject: str,
    predicate: str,
    object: str,
    *,
    roles: Sequence[Mapping[str, Any]] | None = None,
    event_at: str = "",
    valid_from: str = "",
    valid_to: str = "",
    source_id: str = "",
) -> str:
    """Return the stable id of one n-ary, time-qualified observation.

    Unlike :func:`relation_id_for`, this identity includes time, source and all
    extra participants.  Therefore two observations of the same triple at
    different times coexist, while retrying the same source chunk is idempotent.
    """

    normalized_roles = sorted(
        (
            {
                "role": normalize_entity_name(str(role.get("role") or "")),
                "value": normalize_entity_name(str(role.get("value") or "")),
            }
            for role in (roles or [])
            if str(role.get("role") or "").strip()
            and str(role.get("value") or "").strip()
        ),
        key=lambda role: (role["role"], role["value"]),
    )
    payload = {
        "subject": normalize_entity_name(subject),
        "predicate": normalize_entity_name(predicate),
        "object": normalize_entity_name(object),
        "roles": normalized_roles,
        "event_at": _clean_text(event_at, max_length=80),
        "valid_from": _clean_text(valid_from, max_length=80),
        "valid_to": _clean_text(valid_to, max_length=80),
        "source_id": _clean_text(source_id, max_length=300),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"observation:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def predicate_key_for(subject: str, predicate: str) -> str:
    """Stable key identifying one (subject, predicate) slot.

    Single-valued slots such as ``余额`` or ``当前版本`` hold one current value.
    The key ignores the object so a newer value can retire its predecessor.
    """

    key = "|".join(
        (normalize_entity_name(subject), normalize_entity_name(predicate))
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def legacy_fact_id_for(subject: str, predicate: str, object: str) -> str:
    """The pre-unification ``fact:<s>|<p>|<o>`` id kept for existing databases.

    Facts written before the id scheme was unified used this readable form.
    ``add_fact`` falls back to it when the canonical id is absent so old rows
    are updated in place instead of being duplicated.
    """

    return f"fact:{subject.strip()}|{predicate.strip()}|{object.strip()}"


__all__ = [
    "entity_id_for",
    "legacy_fact_id_for",
    "normalize_entity_name",
    "observation_id_for",
    "predicate_key_for",
    "relation_id_for",
]
