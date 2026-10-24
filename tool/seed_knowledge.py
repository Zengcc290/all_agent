"""种子播种工具：把 Aetheria 星图种子数据导入记忆库（写工具，幂等）。

为什么这是一个独立能力
======================

"给一个空库灌一份可演示的星图"是独立的运维能力：它幂等（以
``metadata.seed == SEED_MARK`` 为标记）、可重复调用、失败无副作用。
它过去只藏在 ``web/seed.py`` 里，由应用启动钩子与 ``POST /api/seed`` 使用；
Agent 无法主动"播种"，也无法解释"为什么图里有这些种子节点"。

幂等标记为什么必须留在 metadata 里
==================================

判定依据是**记忆库自身**的内容，而不是外部状态文件：库被复制、被清空、
被换机器，判定都自动跟着走。这也是"完全孤立实体"判定里把 seed 实体排除在外的依据
（见 ``knowledge.orphan_entities``），所以这个标记是跨工具契约，不能改。

本模块是这段逻辑的**唯一实现**：``web/seed.py`` 已删除，启动钩子与端点改为调用这里。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryManager, MemoryType

TOOL_ENABLED = True

#: 幂等标记：写进每条种子项的 metadata.seed。
SEED_MARK = "aetheria-seed-v1"

#: 种子数据文件（仓库内固定位置；``tool/`` 的上一级即仓库根）。
SEED_FILE = Path(__file__).resolve().parent.parent / "web" / "seed_data.json"


def seed(manager: MemoryManager, path: Path | None = None) -> dict[str, Any]:
    """Import the Aetheria seed data; idempotent via ``metadata.seed == SEED_MARK``."""

    seed_path = Path(path) if path is not None else SEED_FILE
    if not seed_path.is_file():
        return {"seeded": False, "reason": f"种子文件不存在：{seed_path}"}

    semantic_items = manager.list(memory_type=MemoryType.SEMANTIC)
    if any(item.metadata.get("seed") == SEED_MARK for item in semantic_items):
        return {"seeded": False, "reason": "已播种过（幂等跳过）",
                "existing": len(semantic_items)}

    data = json.loads(seed_path.read_text(encoding="utf-8"))
    entities = data.get("entities") or []
    relations = data.get("relations") or []
    notes = data.get("notes") or []

    entity_count = 0
    for entity in entities:
        name = str(entity.get("name") or "").strip()
        if not name:
            continue
        manager.add(
            name,
            memory_type=MemoryType.SEMANTIC,
            metadata={
                "kind": "entity",
                "title": name,
                "domain": entity.get("domain") or "未分类",
                "seed": SEED_MARK,
            },
            importance=float(entity.get("importance", 0.85)),
        )
        entity_count += 1

    relation_count = 0
    for relation in relations:
        subject = str(relation.get("subject") or "").strip()
        predicate = str(relation.get("predicate") or "").strip()
        obj = str(relation.get("object") or "").strip()
        if not (subject and predicate and obj):
            continue
        manager.semantic.add_fact(
            subject,
            predicate,
            obj,
            metadata={
                "domain": relation.get("domain") or "未分类",
                "note": relation.get("note") or "",
                "date": relation.get("date") or "",
                "seed": SEED_MARK,
            },
            confidence=float(relation.get("confidence", 1.0)),
        )
        relation_count += 1

    note_count = 0
    for note in notes:
        content = str(note.get("content") or "").strip()
        if not content:
            continue
        manager.add(
            content,
            memory_type=MemoryType.SEMANTIC,
            metadata={
                "kind": "note",
                "entity": note.get("entity") or "",
                "domain": note.get("domain") or "未分类",
                "title": note.get("title") or "档案",
                "date": note.get("date") or "",
                "seed": SEED_MARK,
            },
            importance=0.4,
        )
        note_count += 1

    return {
        "seeded": True,
        "source": str(seed_path),
        "entities": entity_count,
        "relations": relation_count,
        "notes": note_count,
    }


class SeedKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(
        default="",
        max_length=1000,
        description="种子数据文件路径；空串表示用仓库内的 web/seed_data.json。",
    )


class SeedKnowledgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seeded: bool = Field(description="true 表示本次真的写入了；false 表示幂等跳过或文件缺失。")
    reason: str = Field(default="", description="未播种时的原因（文件不存在/已播种过）。")
    source: str = Field(default="", description="实际读取的种子文件路径。")
    existing: int = Field(default=0, description="已播种过时库内现有的语义条目数。")
    entities: int = Field(default=0, description="本次写入的实体数。")
    relations: int = Field(default=0, description="本次写入的关系（事实）数。")
    notes: int = Field(default=0, description="本次写入的备注数。")


class SeedKnowledgeTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.seed",
        description=(
            "Seed the demo Aetheria star map into the memory library. Idempotent: "
            "a second call is skipped because seeded rows carry metadata.seed. "
            "Returns what was written, or why nothing was."
        ),
        version="1.0.0",
        input_model=SeedKnowledgeInput,
        output_model=SeedKnowledgeOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=120.0,
        idempotent=True,
        parallel_safe=False,
        tags=("knowledge", "seed", "demo", "write"),
        guidance=(
            "只有在用户要求灌演示数据或初始化星图时调用。它是幂等的，重复调用会返回跳过原因，不要因为 seeded 为 false 就重试。"
            "种子数据默认取仓库内的 web/seed_data.json，要换数据源才传 path。"
        ),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: SeedKnowledgeInput) -> SeedKnowledgeOutput:
        result = seed(self.manager, arguments.path or None)
        return SeedKnowledgeOutput(
            seeded=bool(result.get("seeded")),
            reason=str(result.get("reason") or ""),
            source=str(result.get("source") or ""),
            existing=int(result.get("existing") or 0),
            entities=int(result.get("entities") or 0),
            relations=int(result.get("relations") or 0),
            notes=int(result.get("notes") or 0),
        )


def create_tool() -> BaseTool:
    return SeedKnowledgeTool()


__all__ = [
    "SEED_FILE",
    "SEED_MARK",
    "SeedKnowledgeInput",
    "SeedKnowledgeOutput",
    "SeedKnowledgeTool",
    "create_tool",
    "seed",
]
