"""按任务类型分节的系统提示词：结构、模式专门化与「工具名不漂移」守护。"""

from __future__ import annotations

import re

from agents.prompts import (
    CONSISTENCY_DISCIPLINE,
    HONESTY_RULES,
    MODE_OFFLINE,
    MODE_ONLINE,
    PROMPT_SECTIONS,
    RETRIEVAL_DISCIPLINE,
    TIME_DISCIPLINE,
    WRITE_DISCIPLINE,
    build_system_prompt,
)
from agents.react import ReActAgent
from core import ToolCatalogTool, ToolRegistry, discover_tools
from web.support import SYSTEM_PROMPT

#: 提示词里提到的工具名（用于防漂移校验）。
TOOL_NAME_PATTERN = re.compile(r"\b(?:knowledge|memory|system|web)\.[a-z_]+\b")


def _known_tool_names() -> set[str]:
    registry = ToolRegistry()
    report = discover_tools(registry)
    assert report.ok, [record.error for record in report.errors]
    return set(registry.snapshot()) | {ToolCatalogTool(registry).spec.name}


def test_prompt_is_composed_of_task_specific_sections() -> None:
    prompt = build_system_prompt()
    for name, block in PROMPT_SECTIONS:
        assert block in prompt, name
    # 每个纪律块都有自己的标题，模型才能按任务区分
    for header in (
        "【检索纪律】",
        "【时间纪律】",
        "【写入纪律】",
        "【一致性治理纪律】",
        "【多模态纪律】",
        "【诚实与引用】",
    ):
        assert header in prompt
    assert prompt.count("【模式：") == 2


def test_prompt_sections_are_mutually_distinct() -> None:
    blocks = [block for _, block in PROMPT_SECTIONS]
    assert len(set(blocks)) == len(blocks)
    # 分节必须各管一件事：写入纪律不能混进检索纪律的内容
    assert "写入" not in RETRIEVAL_DISCIPLINE
    assert "knowledge.reconcile" in CONSISTENCY_DISCIPLINE
    assert "knowledge.reconcile" not in WRITE_DISCIPLINE
    assert "system.current_time" in TIME_DISCIPLINE
    assert "检索不到" in HONESTY_RULES


def test_prompt_can_drop_mode_sections() -> None:
    without = build_system_prompt(include_modes=False)
    assert MODE_ONLINE not in without
    assert MODE_OFFLINE not in without
    assert RETRIEVAL_DISCIPLINE in without


def test_web_system_prompt_is_the_composed_prompt() -> None:
    assert SYSTEM_PROMPT == build_system_prompt()
    assert "【检索纪律】" in SYSTEM_PROMPT


def test_every_tool_name_in_the_prompt_is_registered() -> None:
    """提示词提到的工具名必须真的注册在案——改名时这里会失败，防止提示词与工具漂移。"""

    known = _known_tool_names()
    mentioned = set(TOOL_NAME_PATTERN.findall(build_system_prompt()))
    assert mentioned, "提示词里应当提到具体工具名"
    unknown = sorted(mentioned - known)
    assert unknown == [], f"提示词提到未注册的工具：{unknown}"


def test_react_instructions_carry_tool_usage_discipline() -> None:
    instructions = ReActAgent.REACT_INSTRUCTIONS
    assert "工具使用纪律" in instructions
    assert "必填" in instructions
    assert "写操作 · 需确认" in instructions
    assert "不要重复重试" in instructions
    assert "编造 Observation" in instructions
