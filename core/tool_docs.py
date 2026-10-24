"""Render every registered tool's full contract into prompt-ready text.

为什么需要它
============

ReAct 提示词过去只给「工具名 + 一句描述 + 原始 JSON Schema」。原始 JSON Schema 对模型
并不友好：必填/可选要看 ``required`` 数组，约束散落在 ``minLength``/``maximum``/``enum``，
字段说明埋在 ``properties.*.description`` 里，而**输出字段、副作用、是否需要人工确认**
完全没有进 prompt。结果是模型经常猜参数名、漏必填项、对写工具反复试探。

本模块把 ``ToolSpec`` 渲染成一份**逐变量的契约文本**：

    工具名@版本 [读/写 · 是否需确认 · 超时 · 幂等 · 并行]
      用途: <model_description>
      使用规范: <guidance，逐工具专门性约束>
      输入变量:
        - 字段名: 类型 (必填/可选, 默认=…) 约束: … — 字段说明
      输出字段:
        - 字段名: 类型 — 字段说明
      推荐前置: <recommended_before_tools>

同一份渲染同时供两条链路使用，避免两处漂移：

* ReAct 文本协议（``agents/react.py`` 的 ``_with_tool_instructions``）；
* 原生 function-calling（``agents/agent.py`` 的 ``_definitions_for_registrations``
  已经把 schema 放进 ``tools`` 字段，这里额外把同一份契约文本放进 system 消息，
  让「prompt 里有全部工具信息」这件事对两种协议都成立）。

确定性
======

渲染顺序是「工具名排序 + 字段定义顺序」，且不含时间戳/随机值，因此同一批注册表
每次都渲染出**逐字节相同**的文本——这是 OpenAI 前缀缓存（KV cache）复用的前提，
也保证工具契约变化时缓存失效是可解释的。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .models import ToolSpec

#: 嵌套模型（``$ref``）展开的最大深度，防止自引用模型把提示词撑爆。
MAX_REF_DEPTH = 2

#: 单个工具渲染出的文本行数上限（超出则截断字段列表并说明）。
MAX_FIELD_LINES = 60

#: 中文类型名映射：让模型看到的不是 JSON Schema 的英文关键字。
_TYPE_LABELS = {
    "string": "字符串",
    "integer": "整数",
    "number": "数值",
    "boolean": "布尔",
    "array": "数组",
    "object": "对象",
    "null": "空",
}


def _type_label(schema: Mapping[str, Any], defs: Mapping[str, Any], depth: int) -> str:
    """Render a JSON-schema node as a short type label, resolving refs/anyOf."""

    resolved = _resolve(schema, defs, depth)
    if "enum" in resolved:
        values = ", ".join(json.dumps(item, ensure_ascii=False) for item in resolved["enum"])
        return f"枚举[{values}]"
    if "const" in resolved:
        return f"固定值 {json.dumps(resolved['const'], ensure_ascii=False)}"
    if "anyOf" in resolved or "oneOf" in resolved:
        branches = resolved.get("anyOf") or resolved.get("oneOf") or []
        labels = [
            _type_label(branch, defs, depth + 1)
            for branch in branches
            if not _is_null(branch, defs)
        ]
        if any(_is_null(branch, defs) for branch in branches):
            labels.append("空")
        unique = list(dict.fromkeys(label for label in labels if label))
        return " | ".join(unique) if unique else "任意"
    raw_type = resolved.get("type")
    if isinstance(raw_type, list):
        labels = [_TYPE_LABELS.get(str(item), str(item)) for item in raw_type]
        return " | ".join(dict.fromkeys(labels))
    if raw_type == "array":
        items = resolved.get("items") or {}
        return f"数组[{_type_label(items, defs, depth + 1)}]"
    if raw_type == "object":
        extra = resolved.get("additionalProperties")
        if isinstance(extra, Mapping):
            return f"对象{{任意键: {_type_label(extra, defs, depth + 1)}}}"
        properties = resolved.get("properties")
        if isinstance(properties, Mapping) and properties:
            inner = ", ".join(str(key) for key in properties)
            return f"对象{{{inner}}}"
        return "对象"
    if isinstance(raw_type, str):
        return _TYPE_LABELS.get(raw_type, raw_type)
    if "properties" in resolved:
        return "对象"
    return "任意"


def _is_null(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> bool:
    resolved = _resolve(schema, defs, 0)
    return resolved.get("type") == "null"


def _resolve(
    schema: Mapping[str, Any], defs: Mapping[str, Any], depth: int
) -> Mapping[str, Any]:
    """Follow ``$ref`` into ``$defs`` (bounded by ``MAX_REF_DEPTH``)."""

    node: Mapping[str, Any] = schema
    hops = 0
    while isinstance(node, Mapping) and "$ref" in node and hops <= MAX_REF_DEPTH:
        ref = str(node["$ref"])
        name = ref.rsplit("/", 1)[-1]
        target = defs.get(name)
        if not isinstance(target, Mapping):
            return {"type": name}
        node = target
        hops += 1
    if depth > MAX_REF_DEPTH:
        return {"type": "…"}
    return node


def _constraints(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> str:
    """Render the numeric/length/pattern constraints a model must respect."""

    resolved = _resolve(schema, defs, 0)
    parts: list[str] = []
    if "minLength" in resolved:
        parts.append(f"最短 {resolved['minLength']} 字")
    if "maxLength" in resolved:
        parts.append(f"最长 {resolved['maxLength']} 字")
    if "minimum" in resolved:
        parts.append(f"≥ {resolved['minimum']}")
    if "exclusiveMinimum" in resolved:
        parts.append(f"> {resolved['exclusiveMinimum']}")
    if "maximum" in resolved:
        parts.append(f"≤ {resolved['maximum']}")
    if "exclusiveMaximum" in resolved:
        parts.append(f"< {resolved['exclusiveMaximum']}")
    if "minItems" in resolved:
        parts.append(f"至少 {resolved['minItems']} 项")
    if "maxItems" in resolved:
        parts.append(f"最多 {resolved['maxItems']} 项")
    if "pattern" in resolved:
        parts.append(f"须匹配 {resolved['pattern']}")
    return "，".join(parts)


def _format_default(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False) if value else '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return rendered if len(rendered) <= 60 else rendered[:57] + "…"
    return str(value)


def _field_lines(schema: Mapping[str, Any], *, limit: int = MAX_FIELD_LINES) -> list[str]:
    """Render one line per declared property: 名称/类型/必填/默认/约束/说明。"""

    defs = schema.get("$defs")
    defs = defs if isinstance(defs, Mapping) else {}
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        return ["    （无输入变量）"]
    required = set(schema.get("required") or ())
    lines: list[str] = []
    for name, node in properties.items():
        node = node if isinstance(node, Mapping) else {}
        resolved = _resolve(node, defs, 0)
        label = _type_label(node, defs, 0)
        if name in required:
            status = "必填"
        elif "default" in resolved:
            status = f"可选, 默认={_format_default(resolved['default'])}"
        else:
            status = "可选"
        constraints = _constraints(node, defs)
        description = str(node.get("description") or resolved.get("description") or "").strip()
        line = f"    - {name}: {label} ({status})"
        if constraints:
            line += f" 约束: {constraints}"
        if description:
            line += f" — {description}"
        lines.append(line)
        if len(lines) >= limit:
            lines.append(f"    …（其余 {len(properties) - len(lines) + 1} 个变量从略）")
            break
    return lines


def _confirmation_note(spec: ToolSpec) -> str:
    if spec.side_effect == "write":
        return "写操作：执行前必须持有该工具的人工确认钥匙，模型不得自行假定已授权"
    return "只读操作：免确认，可直接调用"


def render_schema_block(name: str, schema: Mapping[str, Any]) -> list[str]:
    """Render a bare input schema (lazily resolved tools without a live spec)."""

    return [f"- {name}", "  输入变量:", *_field_lines(dict(schema))]


def render_tool_entry(name: str, spec: ToolSpec) -> list[str]:
    """Render one tool as a block of prompt lines (deterministic order)."""

    flags = [
        "写操作" if spec.side_effect == "write" else "只读",
        "需确认" if spec.side_effect == "write" else "免确认",
        f"超时 {spec.timeout_seconds:g}s",
        "幂等" if spec.idempotent else "非幂等",
        "可并行" if spec.parallel_safe else "不可并行",
    ]
    if spec.permissions:
        flags.append("权限 " + "/".join(spec.permissions))
    lines = [f"- {name}@{spec.version} [{(' · '.join(flags))}]"]
    lines.append(f"  用途: {spec.description}")
    guidance = getattr(spec, "guidance", "") or ""
    if guidance:
        lines.append(f"  使用规范: {guidance}")
    lines.append("  输入变量:")
    lines.extend(_field_lines(spec.input_schema))
    lines.append("  输出字段:")
    lines.extend(_field_lines(spec.output_schema))
    lines.append(f"  副作用与确认: {_confirmation_note(spec)}")
    if spec.recommended_before_tools:
        lines.append("  推荐前置: " + ", ".join(spec.recommended_before_tools))
    return lines


def render_tool_catalog(
    registrations: Mapping[str, tuple[Any, int]],
) -> list[str]:
    """Render every registration as prompt lines, sorted by tool name."""

    lines: list[str] = []
    for name in sorted(registrations):
        tool, _ = registrations[name]
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, ToolSpec):
            continue
        if lines:
            lines.append("")
        lines.extend(render_tool_entry(name, spec))
    return lines


def render_tool_catalog_text(
    registrations: Mapping[str, tuple[Any, int]],
) -> str:
    """Render the catalog as a single block for a system message."""

    lines = render_tool_catalog(registrations)
    return "\n".join(lines) if lines else "(no tools are registered)"


__all__ = [
    "MAX_FIELD_LINES",
    "MAX_REF_DEPTH",
    "render_schema_block",
    "render_tool_catalog",
    "render_tool_catalog_text",
    "render_tool_entry",
]
