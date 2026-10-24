"""解析器：把 LLM 输出的固定格式（JSON）解析成实体与关系，并补全时间标记。

LLM 被要求输出：
{
  "entities":  [{"name": "...", "type": "...", "key"?: "...", "time"?: "..."}],
  "relations": [{"source": "...", "target": "...", "predicate": "...",
                 "directed"?: true, "time"?: "...", "evidence"?: "..."}]
}

时间规则：
  1. LLM 检测一句话里是否有时间（年/月/日/时），有则如实抽取；
  2. 抽不到就返回空字符串；
  3. 最终由 resolve_times() 统一兜底为「系统当前时间」。
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

from app.core.registry import registry
from app.core.textutil import norm_key
from app.core.validation import Tool, ToolParam

FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*|\s*```$", re.M)
TIME_HINT_RE = re.compile(r"\b\d{4}([-/年]\d{1,2}([-/月]\d{1,2})?)?\b|\d{1,2}[:：时]\d{2}|去年|今年|明年|去年|上个月|本月|目前|近期|现在|当前|最近|之前|以后|未来|早期|当代|近代|古代")
_TIME_TOKENS = ("年", "月", "日", "时", "上午", "下午", "早上", "晚上")


# ---------------------------------------------------------------- 基础工具
def strip_fences(s: str) -> str:
    return FENCE_RE.sub("", (s or "")).strip()


def extract_json_block(s: str) -> dict | list | None:
    """从任意文本中提取 JSON：先整体试，再按花括号配对截取。"""
    s = strip_fences(s)
    if not s:
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, (dict, list)) else None
    except json.JSONDecodeError:
        pass
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            start = i
            break
        elif ch == "[":
            start = i
            break
    if start is None:
        return None
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:j + 1])
                except json.JSONDecodeError:
                    return _salvage(s[start:j + 1])
    return _salvage(s[start:])


def _salvage(fragment: str) -> dict | None:
    """尽力修复被 max_tokens 截断的 JSON：截到最后一个完整元素再补括号。"""
    idx = max(fragment.rfind("}"), fragment.rfind("]"))
    if idx <= 0:
        return None
    head = fragment[:idx + 1]
    stack: list[str] = []
    in_str = esc = False
    for ch in head:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()
    tail = ""
    # 截断处可能停在半句话里，先补一个引号再收尾
    if in_str:
        tail += '"'
    tail += "".join(reversed(stack))
    for candidate in (tail, tail + "]}", tail + "]}", "".join(reversed(stack)) + "]}",
                      "".join(reversed(stack))):
        try:
            v = json.loads(head + candidate)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _clean_str(v: Any, limit: int = 200) -> str:
    if v is None:
        return ""
    s = str(v).strip().strip('"').strip()
    return s[:limit]


def _extract_time(*vals: Any) -> str:
    """从若干候选值中找出第一个像时间的东西。"""
    for v in vals:
        s = _clean_str(v, 40)
        if not s:
            continue
        if TIME_HINT_RE.search(s):
            return s
        if any(t in s for t in _TIME_TOKENS):
            return s
    return ""


def _norm_entities(raw: Iterable) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        name = _clean_str(e.get("name") or e.get("entity") or e.get("text"))
        if not name:
            continue
        key = norm_key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "name": name,
            "type": _clean_str(e.get("type") or e.get("label") or "概念", 30) or "概念",
            # key 一律用实体名归一化，与关系表 src_key/tgt_key 对齐（忽略 LLM 拼音 key）
            "key": norm_key(name),
            "aliases": [_clean_str(a, 60) for a in (e.get("aliases") or e.get("alias") or []) if _clean_str(a, 60)],
            "time": _extract_time(e.get("time"), e.get("date"), e.get("timestamp")),
            "raw_time": _clean_str(e.get("time") or "", 40),
        })
    return out


def _norm_relations(raw: Iterable) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        src = _clean_str(r.get("source") or r.get("subject") or r.get("from"))
        tgt = _clean_str(r.get("target") or r.get("object") or r.get("to"))
        pred = _clean_str(r.get("predicate") or r.get("relation") or r.get("type"), 60)
        if not src or not tgt or not pred:
            continue
        sig = (src.lower(), tgt.lower(), pred.lower())
        if sig in seen:
            continue
        seen.add(sig)
        directed_raw = r.get("directed", r.get("direction"))
        directed = True if directed_raw is None else str(directed_raw).lower() in {
            "true", "1", "yes", "y", "on", "有向", "单向", "directed"}
        out.append({
            "source": src, "target": tgt, "predicate": pred,
            "src_key": norm_key(src), "tgt_key": norm_key(tgt),
            "directed": directed,
            "time": _extract_time(r.get("time"), r.get("date")),
            "raw_time": _clean_str(r.get("time") or "", 40),
            "evidence": _clean_str(r.get("evidence") or r.get("sentence") or "", 300),
        })
    return out


def parse_llm_output(raw: str) -> tuple[dict, list[str]]:
    """解析 LLM 固定格式 -> (normalized_dict, warnings)。"""
    warnings: list[str] = []
    data = extract_json_block(raw)
    if data is None:
        raise ValueError(f"无法从 LLM 输出中解析出 JSON。原始输出（前500字）: {(raw or '')[:500]}")

    if isinstance(data, list):
        data = {"entities": [], "relations": data}
        warnings.append("顶层是数组，已按 relations 处理")

    ents_raw = data.get("entities") or data.get("entity") or []
    rels_raw = data.get("relations") or data.get("triples") or data.get("relation") or []
    if not isinstance(ents_raw, list):
        ents_raw = [ents_raw]
    if not isinstance(rels_raw, list):
        rels_raw = [rels_raw]

    entities = _norm_entities(ents_raw)
    relations = _norm_relations(rels_raw)

    # 关系里出现但实体表里没有的实体 -> 自动补建
    known = {e["key"] for e in entities}
    for r in relations:
        for side in ("src_key", "tgt_key"):
            if r[side] not in known:
                known.add(r[side])
                entities.append({"name": r["source" if side == "src_key" else "target"],
                                 "type": "概念", "key": r[side], "aliases": [],
                                 "time": r["time"], "raw_time": ""})
                warnings.append(f"关系中的实体 {r[side]!r} 未出现在 entities 列表中，已自动补建")
    if not entities:
        warnings.append("未解析到任何实体")
    if not relations:
        warnings.append("未解析到任何关系")

    return {"entities": entities, "relations": relations,
            "counts": {"entities": len(entities), "relations": len(relations)}}, warnings


def has_time(text: str) -> bool:
    """判断一句话里是否含有可识别的时间表达。"""
    return bool(TIME_HINT_RE.search(text or "") or any(t in (text or "") for t in _TIME_TOKENS))


def resolve_times(parsed: dict, default_time: str) -> dict:
    """给缺时间的实体/关系统一打上时间标记。"""
    for e in parsed.get("entities", []):
        if not e.get("time"):
            e["time"] = default_time
            e["time_filled"] = True
        else:
            e.setdefault("time_filled", False)
    for r in parsed.get("relations", []):
        if not r.get("time"):
            r["time"] = default_time
            r["time_filled"] = True
        else:
            r.setdefault("time_filled", False)
    return parsed


# ---------------------------------------------------------------- 注册为工具
async def _parse_handler(raw: str, fallback_time: str = "") -> dict:
    if not raw or not raw.strip():
        raise ValueError("参数 raw 不能为空（LLM 原始输出）")
    parsed, warnings = parse_llm_output(raw)
    if fallback_time:
        parsed = resolve_times(parsed, fallback_time)
    elif parsed["entities"] or parsed["relations"]:
        # 未指定兜底时间且存在缺时间的项 -> 调用 get_current_time 工具
        missing = any(not e.get("time") for e in parsed["entities"]) or \
                  any(not r.get("time") for r in parsed["relations"])
        if missing and registry.has("get_current_time"):
            sys_time = (await registry.call("get_current_time", {}))["result"].get("datetime", "")
            if sys_time:
                parsed = resolve_times(parsed, sys_time)
                warnings.append("存在缺失时间的实体/关系，已用系统当前时间兜底")
    return {
        "ok": True,
        "entities": parsed["entities"],
        "relations": parsed["relations"],
        "counts": parsed["counts"],
        "warnings": warnings,
    }


registry.register(Tool(
    name="parse_llm_output",
    description=(
        "解析器工具：解析并获取 LLM 输出的固定格式（JSON），转换为结构化实体与实体关系三元组。"
        "输入是 LLM 的原始字符串输出，内部会去除 markdown 代码围栏、截取 JSON 块、"
        "尝试修复被截断的 JSON，并对实体名/关系做归一化与去重；"
        "若 LLM 输出中缺少 time 字段，可传入 fallback_time 兜底。"
    ),
    params=[
        ToolParam("raw", "string", "LLM 的原始输出文本（含 JSON 的那段，带 ```json 围栏也可以）", required=True, max_length=200000),
        ToolParam("fallback_time", "string",
                  "可选：当实体/关系缺少时间时，统一使用的时间标记（如今天日期）。省略时不自动兜底。",
                  default=""),
    ],
    handler=_parse_handler,
    tags=["解析", "核心"],
    timeout=60,
))
