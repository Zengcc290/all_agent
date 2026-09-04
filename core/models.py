from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class ToolCall(StrictModel):
    """A model-produced request to execute one registered tool version."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=200)
    schema_version: str = Field(min_length=1, max_length=32)
    schema_hash: str = Field(min_length=1, max_length=128)
    registry_generation: int | None = Field(default=None, ge=1)
    arguments: dict[str, Any]
    depends_on: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list
    )


class ToolError(StrictModel):
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    retryable: bool = False


class ToolResult(StrictModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=200)
    ok: bool
    data: Any | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> ToolResult:
        if self.ok and self.error is not None:
            raise ValueError("successful tool results cannot contain an error")
        if not self.ok and self.error is None:
            raise ValueError("failed tool results must contain an error")
        if not self.ok and self.data is not None:
            raise ValueError("failed tool results cannot contain data")
        return self


class BatchToolResult(StrictModel):
    type: Literal["batch_tool_result"] = "batch_tool_result"
    results: list[ToolResult]

    @model_validator(mode="after")
    def validate_unique_call_ids(self) -> BatchToolResult:
        grouped: dict[str, list[ToolResult]] = defaultdict(list)
        for result in self.results:
            grouped[result.call_id].append(result)
        for duplicate_results in grouped.values():
            if len(duplicate_results) < 2:
                continue
            # The runtime returns one diagnostic result per input when it
            # detects duplicate IDs; those diagnostics are the sole permitted
            # duplicate representation.
            if not all(
                result.error is not None and result.error.code == "DUPLICATE_CALL_ID"
                for result in duplicate_results
            ):
                raise ValueError("batch tool results must have unique call_id values")
        return self


@dataclass(frozen=True)
class ToolSpec:
    """Static metadata. Pydantic input/output models are the source of truth."""

    name: str
    description: str
    version: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    side_effect: str = "read"
    permissions: tuple[str, ...] = ()
    timeout_seconds: float = 30.0
    idempotent: bool = True
    parallel_safe: bool = True
    max_concurrency: int | None = None
    tags: tuple[str, ...] = ()
    _schema_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", self.name
        ):
            raise ValueError(
                "tool names must use a namespace, for example 'web.search'"
            )
        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or len(self.description) > 2000
        ):
            raise ValueError(
                "tool descriptions must be non-empty and at most 2000 characters"
            )
        if (
            not isinstance(self.version, str)
            or not self.version.strip()
            or len(self.version) > 32
        ):
            raise ValueError("tool versions must be between 1 and 32 characters")
        if not isinstance(self.input_model, type) or not issubclass(
            self.input_model, BaseModel
        ):
            raise TypeError("input_model must be a Pydantic model class")
        if not isinstance(self.output_model, type) or not issubclass(
            self.output_model, BaseModel
        ):
            raise TypeError("output_model must be a Pydantic model class")
        for field_name, model in (
            ("input_model", self.input_model),
            ("output_model", self.output_model),
        ):
            if model.model_config.get("extra") != "forbid":
                raise ValueError(
                    f"{field_name} must configure extra='forbid' so the tool "
                    "contract cannot silently discard unknown fields"
                )
        if (
            not isinstance(self.side_effect, str)
            or not self.side_effect.strip()
            or len(self.side_effect) > 32
        ):
            raise ValueError(
                "side_effect must be a non-empty string of at most 32 characters"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        if not isinstance(self.idempotent, bool) or not isinstance(
            self.parallel_safe, bool
        ):
            raise TypeError("idempotent and parallel_safe must be booleans")
        if not isinstance(self.permissions, tuple) or not all(
            isinstance(item, str) and item for item in self.permissions
        ):
            raise TypeError("permissions must be a tuple of non-empty strings")
        if not isinstance(self.tags, tuple) or not all(
            isinstance(item, str) and item for item in self.tags
        ):
            raise TypeError("tags must be a tuple of non-empty strings")
        if self.max_concurrency is not None and (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be positive")
        schema = {
            "input": self.input_model.model_json_schema(),
            "output": self.output_model.model_json_schema(),
        }
        encoded = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        object.__setattr__(
            self, "_schema_hash", hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )

    @property
    def schema_hash(self) -> str:
        return self._schema_hash

    @property
    def confirmation_key(self) -> str:
        """Bind side-effect confirmation to the exact executable contract."""
        return f"{self.name}@{self.version}#{self.schema_hash}"

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    @property
    def output_schema(self) -> dict[str, Any]:
        return self.output_model.model_json_schema()

    def summary(self) -> dict[str, Any]:
        return {
            "tool_name": self.name,
            "description": self.description,
            "version": self.version,
            "schema_hash": self.schema_hash,
            "side_effect": self.side_effect,
            "max_concurrency": self.max_concurrency,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ExecutionContext:
    """Execution context for one tool request.

    Entries in ``confirmed_side_effects`` should be the registry's
    generation-bound confirmation keys, so a replacement implementation cannot
    reuse an old confirmation. ``permissions`` remains available as tool
    compatibility/audit metadata, but is not currently enforced by the runtime
    or agents.
    """

    subject: str = "default"
    permissions: frozenset[str] = frozenset()
    confirmed_side_effects: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("subject must be a non-empty string")
        if not isinstance(self.permissions, frozenset) or not all(
            isinstance(item, str) and item.strip() for item in self.permissions
        ):
            raise TypeError("permissions must be a frozenset of non-empty strings")
        if not isinstance(self.confirmed_side_effects, frozenset) or not all(
            isinstance(item, str) and item.strip()
            for item in self.confirmed_side_effects
        ):
            raise TypeError(
                "confirmed_side_effects must be a frozenset of non-empty strings"
            )
