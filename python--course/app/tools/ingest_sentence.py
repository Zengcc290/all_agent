"""工具：ingest_sentence —— 一句话入库。

流程：
  1. sqlite 存档原文并进入 ingest_queue；
  2. 调用 get_all_entities / get_all_relations 工具，从 neo4j 拿已有实体与关系（供 LLM 复用）；
  3. 组装提示词（含全部可用工具说明 + 抽取要求），流式调用 LLM；
  4. 用 parse_llm_output 解析 LLM 固定格式输出 -> 实体与关系三元组；
  5. 时间处理：LLM 检测一句话里是否有时间（可到年/月/日/时），
     没有时间就调用 get_current_time 工具取系统时间，给每个实体/每条关系都打上时间标记；
  6. qdrant 与 neo4j 两条线并行写入，两条都成功才把 chunk 转正进 chunks 表。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from app import config
from app.core.ingest import (
    Event, _extract_with_llm, _ingest_qdrant, _read_existing_graph, _sys_time, _write_neo4j,
)
from app.core.registry import registry
from app.core.textutil import norm_key
from app.core.validation import Tool, ToolParam
from app.db.sqlite_store import store
from app.llm import LLMError

_ = (_sys_time, config)  # 保留导入：子流程 time_resolved 事件对外可见，便于调试

# 供 /api/ingest/stream 使用的全局事件总线（发布订阅，多个 SSE 客户端可并存）
_LISTENERS: list[Callable[[str, dict], Any]] = []


def subscribe(fn: Callable[[str, dict], Any]) -> Callable[[], None]:
    _LISTENERS.append(fn)

    def un() -> None:
        if fn in _LISTENERS:
            _LISTENERS.remove(fn)

    return un


def _broadcast(stage: str, kw: dict) -> None:
    for fn in list(_LISTENERS):
        try:
            fn(stage, kw)
        except Exception:  # noqa: BLE001
            pass


async def ingest_sentence(text: str, source: str = "sentence",
                          on_event: Event | None = None) -> dict:
    """一条龙入库，返回全链路结果（含 LLM 抽取明细）。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("参数 text 不能为空（要入库的那句话）")
    if len(text) > 20000:
        raise ValueError("单句话过长（>20000 字符），请拆分成多条")

    t0 = time.perf_counter()

    async def emit(stage: str, data: dict | None = None, **kw: Any) -> None:
        """对外统一的事件出口：接收 (stage, dict) 或 (stage, **kw) 两种形式。

        与 core.ingest._emit 的约定一致（Callable[[str, dict], Any]），
        这样上层用关键字调用、底层用位置调用都不会错位。
        """
        payload = dict(data) if isinstance(data, dict) else dict(kw)
        if on_event is not None:
            r = on_event(stage, payload)
            if hasattr(r, "__await__"):
                await r
        _broadcast(stage, payload)

    # ---------- 1. sqlite：原文 -> 队列 ----------
    doc = await asyncio.to_thread(store.add_document, text, source, {"kind": "sentence"})
    doc_id = doc["id"]
    chunk_id = await asyncio.to_thread(store.enqueue_chunk, doc_id, text, 0)
    await emit("document", document_id=doc_id, chunk_id=chunk_id, text=text)
    await emit("queued", chunk_id=chunk_id, document_id=doc_id)

    # ---------- 2. 已有实体 / 关系（通过工具系统拿，不直接穿透） ----------
    existing, existing_rels = await _read_existing_graph(emit)
    await emit("existing_entities", count=len(existing))

    # ---------- 3~4. LLM 抽取 -> 解析 -> 时间兜底（只抽一次，不重复） ----------
    try:
        out = await _extract_with_llm(text, existing, existing_rels, emit)
    except Exception as e:  # noqa: BLE001
        await emit("llm_end", error=str(e), chars=0)
        await asyncio.to_thread(store.update_chunk_status, chunk_id, None, "failed", str(e)[:500])
        raise
    parsed, raw, warnings = out["parsed"], out["raw"], out["warnings"]
    time_source, default_time = out["time_source"], out["default_time"]
    for w in warnings:
        await emit("warn", message=w)
    if not raw.strip():
        await asyncio.to_thread(store.update_chunk_status, chunk_id, None, "failed", "LLM 输出为空")
        raise LLMError("LLM 返回了空内容")

    entities = parsed["entities"]
    triples = parsed["relations"]
    existing_keys = {e.get("key") or norm_key(e.get("name") or "") for e in existing}
    for e in entities:
        e["reused"] = e["key"] in existing_keys
    for r in triples:
        r["chunk_id"] = chunk_id

    # 时间裁决已在 _extract_with_llm 里完成，这里直接沿用，不要用已补全的数据反推
    await emit("parsed", entities=len(entities), relations=len(triples),
               new_entities=sum(1 for e in entities if not e.get("reused")),
               reused_entities=sum(1 for e in entities if e.get("reused")))

    # ---------- 5. 写库：qdrant 线 ∥ neo4j 线并行 ----------
    async def run_q() -> dict:
        try:
            await asyncio.to_thread(store.update_chunk_status, chunk_id, "success", None, "")
            res = await _ingest_qdrant(chunk_id, text)
            await emit("qdrant_ok", chunk_id=chunk_id, **res)
            return {"line": "qdrant", "ok": True, **res}
        except Exception as e:  # noqa: BLE001
            await asyncio.to_thread(store.update_chunk_status, chunk_id, "failed", None, str(e)[:500])
            await emit("qdrant_fail", chunk_id=chunk_id, error=str(e)[:300])
            return {"line": "qdrant", "ok": False, "error": str(e)}

    async def run_n() -> dict:
        try:
            res = await _write_neo4j(chunk_id, text, parsed, emit)
            await asyncio.to_thread(store.update_chunk_status, chunk_id, None, "success", "")
            await emit("neo4j_ok", chunk_id=chunk_id, **res)
            return {"line": "neo4j", "ok": True, **res}
        except Exception as e:  # noqa: BLE001
            await asyncio.to_thread(store.update_chunk_status, chunk_id, None, "failed", str(e)[:500])
            await emit("neo4j_fail", chunk_id=chunk_id, error=str(e)[:300])
            return {"line": "neo4j", "ok": False, "error": str(e)}

    res = await asyncio.gather(run_q(), run_n())
    ok_q = next(r for r in res if r["line"] == "qdrant")["ok"]
    ok_n = next(r for r in res if r["line"] == "neo4j")["ok"]

    promoted = None
    if ok_q and ok_n:
        promoted = await asyncio.to_thread(store.promote_chunk, chunk_id)
        if promoted:
            await emit("promoted", chunk_id=chunk_id, document_id=doc_id, preview=text[:60])
    else:
        await emit("stayed_in_queue", chunk_id=chunk_id,
                   message="qdrant 或 neo4j 有一条未成功，chunk 继续留在入库队列中")

    return {
        "ok": bool(promoted),
        "document_id": doc_id,
        "chunk_id": chunk_id,
        "qdrant": "success" if ok_q else "failed",
        "neo4j": "success" if ok_n else "failed",
        "promoted": bool(promoted),
        "time": default_time,
        "time_source": time_source,
        "time_llm_count": out["time_llm_count"],
        "time_fallback_count": out["time_fallback_count"],
        "llm_raw": raw[:4000],
        "extracted": {
            "entities": [
                {k: e.get(k) for k in ("name", "type", "key", "time", "reused", "aliases")}
                for e in entities
            ],
            "relations": [
                {k: r.get(k) for k in ("source", "target", "predicate", "directed", "time", "evidence")}
                for r in triples
            ],
            "new_entities": sum(1 for e in entities if not e.get("reused")),
            "reused_entities": sum(1 for e in entities if e.get("reused")),
        },
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


async def _handler(text: str, source: str = "sentence") -> dict:
    return await ingest_sentence(text, source)


_ = config  # noqa: F401（保留在模块作用域便于其它工具引用配置）

registry.register(Tool(
    name="ingest_sentence",
    description=(
        "一句话入库工具：在前端输入框输入一句话，本工具把它整体入库。"
        "处理链路：sqlite 存档原文并进队列 -> 调用 get_all_entities 取回已有实体供 LLM 复用 -> "
        "流式调用 LLM 按提示词要求抽取实体与关系（已有实体复用、不存在的创建，允许多个三元组与实体重复利用，"
        "要求输出规范 JSON）-> parse_llm_output 解析 -> 时间兜底（LLM 检测不到时间时调用 get_current_time "
        "给每个实体和关系都打上时间标记）-> qdrant 与 neo4j 并行写入，两条都成功才把 chunk 转正进 chunks 表。"
    ),
    params=[
        ToolParam("text", "string", "要入库的那句话（原始文本，会原样作为唯一一个 chunk）",
                  required=True, max_length=20000, min_length=1),
        ToolParam("source", "string", "来源标记，写入 documents.source", default="sentence", max_length=64),
    ],
    handler=_handler,
    tags=["入库", "LLM", "qdrant", "neo4j"],
    timeout=600,
))
