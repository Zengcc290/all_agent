from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ToolCall(StrictModel):
    """A model-produced request to execute one registered tool version."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=200)
    schema_version: str = Field(min_length=1, max_length=32)
    schema_hash: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any]
    depends_on: list[str] = Field(default_factory=list)


class ToolError(StrictModel):
    code: str
    message: str
    retryable: bool = False


class ToolResult(StrictModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    tool_name: str
    ok: bool
    data: Any | None = None
    error: ToolError | None = None


class BatchToolResult(StrictModel):
    type: Literal["batch_tool_result"] = "batch_tool_result"
    results: list[ToolResult]


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
        if not self.name or "." not in self.name:
            raise ValueError("tool names must use a namespace, for example 'web.search'")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_concurrency is not None and self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        schema = {
            "input": self.input_model.model_json_schema(),
            "output": self.output_model.model_json_schema(),
        }
        encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        object.__setattr__(self, "_schema_hash", hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    @property
    def schema_hash(self) -> str:
        return self._schema_hash

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
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ExecutionContext:
    subject: str = "default"
    permissions: frozenset[str] = frozenset()
    confirmed_side_effects: frozenset[str] = frozenset()
