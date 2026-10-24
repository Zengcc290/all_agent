"""``core.tool_docs``：把工具契约渲染成 prompt 可读文本的回归测试。

覆盖三件事：
1. **完整性**——每个注册工具的每个输入变量、约束、默认值、输出字段、副作用都要进 prompt；
2. **确定性**——同一批注册表每次渲染逐字节相同（OpenAI 前缀缓存复用的前提）；
3. **边界**——``guidance`` 进 prompt 但不进 ``schema_hash``（改提示词不该让已存确认失效）。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolRegistry, ToolSpec, discover_tools
from core.tool_docs import (
    render_schema_block,
    render_tool_catalog,
    render_tool_catalog_text,
    render_tool_entry,
)


class ProbeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=120, description="检索词，必填。")
    limit: int = Field(default=5, ge=1, le=50, description="返回条数上限。")
    zone: str | None = Field(default=None, description="可选区域；null 表示不限。")
    tags: list[str] = Field(default_factory=list, description="标签过滤。")
    nested: ProbeNested | None = Field(default=None, description="嵌套对象参数。")


class ProbeNested(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: str = Field(min_length=1, description="嵌套键。")


class ProbeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ok: bool = Field(description="是否成功。")
    items: list[str] = Field(default_factory=list)


class ProbeTool(BaseTool):
    spec = ToolSpec(
        name="test.probe",
        description="Probe one query.",
        version="2.1.0",
        input_model=ProbeInput,
        output_model=ProbeOutput,
        side_effect="write",
        timeout_seconds=45.0,
        idempotent=False,
        parallel_safe=False,
        permissions=("network",),
        recommended_before_tools=("system.current_time",),
        guidance="只在需要外部检索时使用；不要用它写记忆。",
    )

    def execute(self, arguments: ProbeInput) -> ProbeOutput:  # pragma: no cover - 渲染不需要执行
        raise NotImplementedError


def test_entry_renders_every_variable_with_type_required_default_and_constraints() -> None:
    rendered = "\n".join(render_tool_entry("test.probe", ProbeTool().spec))

    # 头部：版本 + 副作用/确认/超时/幂等/并行/权限
    assert "- test.probe@2.1.0 [" in rendered
    assert "写操作" in rendered and "需确认" in rendered
    assert "超时 45s" in rendered
    assert "非幂等" in rendered and "不可并行" in rendered
    assert "权限 network" in rendered

    # 用途 / 使用规范 / 副作用说明 / 推荐前置
    assert "用途: Probe one query." in rendered
    assert "使用规范: 只在需要外部检索时使用；不要用它写记忆。" in rendered
    assert "推荐前置: system.current_time" in rendered
    assert "写操作：执行前必须持有该工具的人工确认钥匙" in rendered

    # 逐变量：类型、必填/默认、约束、字段说明一个都不能少
    assert "- query: 字符串 (必填) 约束: 最短 1 字，最长 120 字 — 检索词，必填。" in rendered
    assert "- limit: 整数 (可选, 默认=5) 约束: ≥ 1，≤ 50 — 返回条数上限。" in rendered
    assert "- zone: 字符串 | 空 (可选, 默认=null) — 可选区域；null 表示不限。" in rendered
    # default_factory 的字段在 JSON Schema 里没有 default（pydantic 不物化工厂），
    # 因此只声明「可选」——这正是模型需要知道的可执行信息。
    assert "- tags: 数组[字符串] (可选) — 标签过滤。" in rendered
    assert "- nested: 对象{key} | 空 (可选, 默认=null) — 嵌套对象参数。" in rendered

    # 输出字段同样进 prompt
    assert "输出字段:" in rendered
    assert "- ok: 布尔 (必填) — 是否成功。" in rendered
    assert "- items: 数组[字符串] (可选" in rendered


def test_read_tool_is_rendered_as_confirmation_free() -> None:
    spec = ProbeTool().spec
    read_spec = ToolSpec(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        input_model=spec.input_model,
        output_model=spec.output_model,
    )
    rendered = "\n".join(render_tool_entry("test.probe", read_spec))
    assert "只读" in rendered and "免确认" in rendered
    assert "只读操作：免确认，可直接调用" in rendered


def test_guidance_is_prompt_only_and_does_not_change_schema_hash() -> None:
    spec = ProbeTool().spec
    without = ToolSpec(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        input_model=spec.input_model,
        output_model=spec.output_model,
        side_effect=spec.side_effect,
    )
    # 提示词变了，但可执行契约（输入/输出 schema）没变 → 哈希必须一致，
    # 否则每次润色提示词都会让已存的写确认钥匙失效。
    assert without.schema_hash == spec.schema_hash
    assert "使用规范" not in "\n".join(render_tool_entry("test.probe", without))
    assert "使用规范" in "\n".join(render_tool_entry("test.probe", spec))


def test_guidance_must_stay_short() -> None:
    spec = ProbeTool().spec
    with pytest.raises(TypeError, match="guidance must be a string"):
        ToolSpec(
            name=spec.name,
            description=spec.description,
            version=spec.version,
            input_model=spec.input_model,
            output_model=spec.output_model,
            guidance="x" * 1201,
        )


def test_schema_block_renders_a_bare_schema() -> None:
    lines = render_schema_block("test.lazy", ProbeInput.model_json_schema())
    rendered = "\n".join(lines)
    assert lines[0] == "- test.lazy"
    assert "输入变量:" in rendered
    assert "- query: 字符串 (必填)" in rendered


def test_catalog_is_sorted_and_byte_stable() -> None:
    first: dict[str, tuple[Any, int]] = {
        "test.probe": (ProbeTool(), 1),
        "system.current_time": (ProbeTool(), 1),
    }
    reversed_order: dict[str, tuple[Any, int]] = dict(reversed(list(first.items())))

    assert render_tool_catalog(first) == render_tool_catalog(reversed_order)
    text = render_tool_catalog_text(first)
    assert text == render_tool_catalog_text(first)
    # 工具名排序：system.* 在 test.* 之前
    assert text.index("system.current_time") < text.index("test.probe")


def test_every_registered_tool_renders_its_full_contract() -> None:
    registry = ToolRegistry()
    report = discover_tools(registry)
    assert report.ok, [record.error for record in report.errors]

    snapshot = registry.snapshot()
    text = render_tool_catalog_text(snapshot)

    for name, (tool, _) in snapshot.items():
        spec = tool.spec
        block = "\n".join(render_tool_entry(name, spec))
        assert f"- {name}@{spec.version} [" in block
        assert f"用途: {spec.description}" in block
        assert "输入变量:" in block and "输出字段:" in block
        assert "副作用与确认:" in block
        # 每个声明的输入变量都必须出现在 prompt 里
        for field in spec.input_schema.get("properties", {}):
            assert f"- {field}:" in block, f"{name} 缺少输入变量 {field}"
        # 每个声明的输出字段同理
        for field in spec.output_schema.get("properties", {}):
            assert f"- {field}:" in block, f"{name} 缺少输出字段 {field}"
    assert text.count("- knowledge.") >= 18


def test_every_registered_tool_declares_its_own_usage_guidance() -> None:
    """每个工具都必须写自己的使用规范：何时用、何时不用、硬约束。

    这是「按不同功能加强规范与约束、强化专门性」的落点——guidance 会原样进提示词，
    因此缺失或互相抄同一段都算缺陷。
    """

    registry = ToolRegistry()
    report = discover_tools(registry)
    assert report.ok, [record.error for record in report.errors]

    snapshot = registry.snapshot()
    missing = [
        name for name, (tool, _) in snapshot.items() if not tool.spec.guidance.strip()
    ]
    assert missing == [], f"这些工具缺少使用规范：{missing}"

    seen: dict[str, str] = {}
    for name, (tool, _) in snapshot.items():
        guidance = tool.spec.guidance
        assert 10 <= len(guidance) <= 1200, name
        assert guidance not in seen, f"{name} 与 {seen.get(guidance)} 的使用规范完全相同"
        seen[guidance] = name
        rendered = "\n".join(render_tool_entry(name, tool.spec))
        assert f"使用规范: {guidance}" in rendered


def test_catalog_tool_also_carries_guidance() -> None:
    """catalog 工具由 agent 注册（不在自动发现里），它的规范同样必须进提示词。"""

    from core import ToolCatalogTool

    spec = ToolCatalogTool(ToolRegistry()).spec
    assert spec.guidance.strip()
    assert "intent" in spec.guidance
    assert "使用规范: " in "\n".join(render_tool_entry(spec.name, spec))
