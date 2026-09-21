"""知识库导入工具：把导出载荷幂等地写回记忆库（写工具）。

三条不可动摇的规则
==================

1. **严格 JSON**：拒绝 ``NaN``/``Infinity``。项目的模型/API 全链路都保证数值严格有限，
   导入是唯一的"外来数据入口"，如果这里放行非有限值，它就会顺着记忆库扩散到嵌入、
   相似度和图权重里。
2. **幂等**：已存在的 id 跳过；事实按 ``(主语, 谓语, 宾语)`` 三元组去重，
   避免重复导入把图长成多重边。
3. **单条失败不拖垮整批**：逐条 try，失败只跳过该条并把原因写进 ``errors``
   （历史版本静默吞掉异常，用户只看到 skipped 计数却不知道哪条失败、为什么失败）。
   ``errors`` 有上限，保证响应有界。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 里原先内联的导入循环与
``_reject_json_constant`` 已删除，``POST /api/import`` 只保留上传与错误码映射。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from constants import WEB_IMPORT_ERRORS_MAX
from core import BaseTool, ToolSpec
from memory.base import MemoryType
from memory.manager import MemoryManager

TOOL_ENABLED = True


def reject_json_constant(value: str) -> None:
    """Reject ``NaN``/``Infinity`` so imported JSON stays strictly finite."""

    raise ValueError(f"invalid JSON constant: {value}")


def parse_import_payload(raw: bytes) -> list[Any]:
    """Decode an export payload into the raw item list.

    Raises ``ValueError`` with a user-facing reason; the HTTP layer maps it to 400.
    """

    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=reject_json_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"不是合法的 JSON：{exc}") from exc
    entries = data.get("items") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        # 载荷结构错误是**数据**问题不是编程错误：HTTP 层据此回 400，
        # 所以这里刻意用 ValueError 而不是 TRY004 建议的 TypeError。
        raise ValueError("JSON 中找不到 items 数组")  # noqa: TRY004
    return entries


def import_items(
    manager: MemoryManager,
    entries: list[Any],
    *,
    max_errors: int = WEB_IMPORT_ERRORS_MAX,
) -> dict[str, Any]:
    """Import items idempotently; returns ``{imported, skipped, errors}``."""

    if isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 1:
        raise ValueError("max_errors must be a positive integer")
    existing_facts = {
        (
            item.metadata.get("subject"),
            item.metadata.get("predicate"),
            item.metadata.get("object"),
        )
        for item in manager.list(memory_type=MemoryType.SEMANTIC)
        if item.metadata.get("subject")
        and item.metadata.get("predicate")
        and item.metadata.get("object")
    }
    known_types = {type_.value for type_ in MemoryType}
    imported = skipped = 0
    errors: list[str] = []

    def note_error(message: str) -> None:
        """Keep the response bounded: first ``max_errors`` reasons."""

        if len(errors) < max_errors:
            errors.append(message)

    for position, raw_item in enumerate(entries, start=1):
        if not isinstance(raw_item, dict):
            skipped += 1
            note_error(f"第 {position} 项：不是 JSON 对象")
            continue
        item_id = raw_item.get("id")
        content = raw_item.get("content") or ""
        if not item_id or not content:
            skipped += 1
            note_error(f"第 {position} 项（id={item_id or '缺失'}）：缺少 id 或 content")
            continue
        if manager.get(item_id) is not None:
            skipped += 1
            continue
        md = raw_item.get("metadata") or {}
        if not isinstance(md, dict):
            skipped += 1
            note_error(f"{item_id}: metadata 必须是 JSON 对象")
            continue
        memory_type = raw_item.get("memory_type") or "semantic"
        if not isinstance(memory_type, str) or memory_type not in known_types:
            skipped += 1
            note_error(f"{item_id}: 未知 memory_type：{memory_type!r}")
            continue
        importance = raw_item.get("importance", 0.5)
        subject, predicate, obj = (
            md.get("subject"),
            md.get("predicate"),
            md.get("object"),
        )
        try:
            if subject and predicate and obj:
                # 事实：按三元组幂等，避免重复导入时长出重边。
                if (subject, predicate, obj) in existing_facts:
                    skipped += 1
                    continue
                existing_facts.add((subject, predicate, obj))
                manager.semantic.add_fact(
                    subject,
                    predicate,
                    obj,
                    metadata=md,
                    confidence=float(importance),
                )
            else:
                manager.add(
                    content,
                    memory_type=memory_type,
                    metadata=md,
                    item_id=item_id,
                    importance=float(importance),
                )
            imported += 1
        except Exception as exc:  # noqa: BLE001 - 单条失败只跳过该条并记录原因
            skipped += 1
            note_error(f"{item_id}: {type(exc).__name__}: {exc}")

    return {"imported": imported, "skipped": skipped, "errors": errors}


class ImportKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    payload: str = Field(
        min_length=2,
        description="导出的 JSON 文本：knowledge-nebula-export/v1 载荷，或裸 items 数组。",
    )
    max_errors: int = Field(
        default=WEB_IMPORT_ERRORS_MAX,
        ge=1,
        le=100,
        description="最多回报多少条失败原因（其余只计入 skipped）。",
    )


class ImportKnowledgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    imported: int = Field(description="真正写入的条目数。")
    skipped: int = Field(description="跳过数：已存在、缺字段、类型未知或写入失败。")
    errors: list[str] = Field(default_factory=list, description="失败原因（有上限）。")


class ImportKnowledgeTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.import",
        description=(
            "Import a knowledge-nebula-export/v1 JSON payload (or a bare items "
            "array) back into the memory library. Idempotent: existing ids and "
            "duplicate (subject, predicate, object) facts are skipped. Rejects "
            "NaN/Infinity and reports per-item failure reasons."
        ),
        version="1.0.0",
        input_model=ImportKnowledgeInput,
        output_model=ImportKnowledgeOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=600.0,
        idempotent=True,
        parallel_safe=False,
        tags=("knowledge", "import", "restore", "write"),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: ImportKnowledgeInput) -> ImportKnowledgeOutput:
        entries = parse_import_payload(arguments.payload.encode("utf-8"))
        result = import_items(self.manager, entries, max_errors=arguments.max_errors)
        return ImportKnowledgeOutput(**result)


def create_tool() -> BaseTool:
    return ImportKnowledgeTool()


__all__ = [
    "ImportKnowledgeInput",
    "ImportKnowledgeOutput",
    "ImportKnowledgeTool",
    "create_tool",
    "import_items",
    "parse_import_payload",
    "reject_json_constant",
]
