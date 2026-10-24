"""事实写入工具：写一条 (主语, 谓语, 宾语) 语义事实（写工具）。

为什么这是一个独立能力
======================

事实是知识图谱的**边**：``semantic.add_fact`` 一次写入同时落到真值源（语义记忆行）
与图投影（有向关系），因此它是"最小可审计的写入单元"——比"写一段文本让 LLM 去抽取"
可控得多。Agent 需要它来把用户明确陈述的事实立刻固化，而不必走整条抽取管道。

与 ``memory.add`` 的区别（刻意分开）
====================================

``memory.add`` 写的是**内容**（一段文本 + 任意元数据），走的是通用记忆写入；
本工具写的是**结构化三元组**，强制 subject/predicate/object 三件套、
带领域与置信度、并且保证图里长出对应的边。两者语义不同，合并会让
"写一段话"和"断言一个事实"混成一个动作。

本模块是这段逻辑的**唯一实现**：``web/app.py`` 的 ``POST /api/facts`` 原先内联的
``semantic.add_fact`` 调用已删除，端点只保留请求校验与错误码映射。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from constants import (
    DEFAULT_DOMAIN,
    WEB_FACT_DOMAIN_MAX,
    WEB_FACT_NOTE_MAX,
    WEB_FACT_OBJECT_MAX,
    WEB_FACT_PREDICATE_MAX,
    WEB_FACT_SUBJECT_MAX,
)
from core import BaseTool, ToolSpec
from memory.base import MemoryItem
from memory.manager import MemoryManager

TOOL_ENABLED = True


def add_fact(
    manager: MemoryManager,
    *,
    subject: str,
    predicate: str,
    object: str,
    domain: str = "",
    note: str = "",
    confidence: float = 1.0,
) -> MemoryItem:
    """Write one fact; the graph projection is updated by ``semantic.add_fact``."""

    return manager.semantic.add_fact(
        subject,
        predicate,
        object,
        metadata={"domain": domain or DEFAULT_DOMAIN, "note": note or ""},
        confidence=confidence,
    )


class AddFactInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subject: str = Field(min_length=1, max_length=WEB_FACT_SUBJECT_MAX, description="主语实体名。")
    predicate: str = Field(min_length=1, max_length=WEB_FACT_PREDICATE_MAX, description="谓语/关系名。")
    object: str = Field(min_length=1, max_length=WEB_FACT_OBJECT_MAX, description="宾语实体名。")
    domain: str = Field(default="", max_length=WEB_FACT_DOMAIN_MAX, description="所属领域；空串表示未分类。")
    note: str = Field(default="", max_length=WEB_FACT_NOTE_MAX, description="备注/证据说明。")
    confidence: float = Field(default=1.0, ge=0, le=1, description="置信度 0~1。")


class AddFactOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_id: str = Field(description="写入的语义记忆项 id（同时也是图关系的 memory_id）。")
    subject: str
    predicate: str
    object: str
    domain: str
    confidence: float


class AddFactTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.add_fact",
        description=(
            "Assert one structured fact (subject, predicate, object) into semantic "
            "memory and the knowledge graph. Use it when the user states a fact "
            "explicitly; use memory.add for plain content."
        ),
        version="1.0.0",
        input_model=AddFactInput,
        output_model=AddFactOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=60.0,
        idempotent=False,
        parallel_safe=False,
        tags=("knowledge", "fact", "graph", "write"),
        guidance=(
            "用户明确陈述一条事实（谁-怎么样-谁或什么）时用它，它同时写语义记忆与图上的边。要写一段内容让系统去抽取知识请用 memory.rag；只写普通文本用 memory.add。三元组三项都必须非空；"
            "关系有更新时写新事实（旧值会被取代），不要试图改写历史。"
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

    def execute(self, arguments: AddFactInput) -> AddFactOutput:
        item = add_fact(
            self.manager,
            subject=arguments.subject,
            predicate=arguments.predicate,
            object=arguments.object,
            domain=arguments.domain,
            note=arguments.note,
            confidence=arguments.confidence,
        )
        return AddFactOutput(
            item_id=item.id,
            subject=arguments.subject,
            predicate=arguments.predicate,
            object=arguments.object,
            domain=arguments.domain or DEFAULT_DOMAIN,
            confidence=arguments.confidence,
        )


def create_tool() -> BaseTool:
    return AddFactTool()


__all__ = [
    "AddFactInput",
    "AddFactOutput",
    "AddFactTool",
    "add_fact",
    "create_tool",
]
