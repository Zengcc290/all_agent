"""入库流水线：文档 -> 分块(透传) -> 队列 -> (qdrant 线 ∥ neo4j 线) -> 两条都成功才转正进 chunks 表。

qdrant 线与 neo4j 线在同一个 chunk 上是并行执行的（asyncio.gather），
多个 chunk 之间也并行，但受 app.max_concurrency 并发上限约束。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from app import config
from app.core import chunker
from app.core.parser import has_time, parse_llm_output, resolve_times
from app.core.registry import registry
from app.core.textutil import norm_key
from app.db.qdrant_store import vector_store as qdrant_store
from app.db.neo4j_store import graph_store as neo4j_store
from app.db.sqlite_store import now_iso, store
from app.llm.client import LLMError, chat_stream, embed_texts

Event = Callable[[str, dict], Any]


async def _emit(on_event: Event | None, stage: str, **kw: Any) -> None:
    """统一事件出口。

    on_event 可能是同步函数也可能是 async 函数（SSE 端点传的就是 async），
    所以返回值必须判断是否为协程并 await —— 否则上层的事件会被静默丢弃。
    """
    if on_event is None:
        return
    r = on_event(stage, kw)
    if hasattr(r, "__await__"):
        await r


async def _sys_time() -> str:
    """通过 get_current_time 工具取系统时间（走注册表，不直接写死实现）。"""
    if registry.has("get_current_time"):
        try:
            r = await registry.call("get_current_time", {"format": "date"})
            return r["result"].get("date", "") or r["result"].get("datetime", "")
        except Exception:  # noqa: BLE001
            pass
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d")


# ------------------------------------------------------------------ qdrant 线
async def _ingest_qdrant(chunk_id: str, content: str) -> dict:
    vecs = await embed_texts([content])
    if not vecs:
        raise LLMError("embedding 返回为空")
    vec = vecs[0]
    await qdrant_store.ensure_collection(dim=len(vec))
    await qdrant_store.upsert_chunk(chunk_id, vec, {
        "content": content,
        "document_id": (await asyncio.to_thread(store.get_queue_item, chunk_id) or {}).get("document_id", ""),
        "created_at": (await asyncio.to_thread(store.get_queue_item, chunk_id) or {}).get("created_at", ""),
    })
    return {"dim": len(vec), "point_id": chunk_id}


# ------------------------------------------------------------------ neo4j 线
async def _read_existing_graph(on_event: Event | None = None) -> tuple[list[dict], list[dict]]:
    """(2) 通过注册表里的工具从 neo4j 读回已有实体与关系，供 LLM 复用。"""
    existing: list[dict] = []
    existing_rels: list[dict] = []
    if registry.has("get_all_entities"):
        try:
            res = (await registry.call("get_all_entities", {"limit": 300}))["result"]
            existing = res.get("entities", [])
        except Exception as e:  # noqa: BLE001
            await _emit(on_event, "warn", message=f"读取已有实体失败（将全部按新实体写入）: {e}")
    if registry.has("get_all_relations"):
        try:
            existing_rels = (await registry.call("get_all_relations", {"limit": 150}))["result"].get("relations", [])
        except Exception:  # noqa: BLE001
            pass
    return existing, existing_rels


async def _write_neo4j(chunk_id: str, content: str, parsed: dict,
                       on_event: Event | None = None) -> dict:
    """把已解析好的实体/关系写入 neo4j，并建立 chunk <-> 实体映射（多对多）。"""
    t0 = time.perf_counter()
    entities = parsed["entities"]
    triples = parsed["relations"]
    for r in triples:
        r.setdefault("chunk_id", chunk_id)

    n_ent = await neo4j_store.upsert_entities(entities)
    n_rel = await neo4j_store.upsert_relations(triples)
    n_link = await neo4j_store.link_chunk(chunk_id, [e["key"] for e in entities], content)
    # sqlite 侧同步 chunk <-> 实体映射（一个 chunk_id -> 多个实体，一个实体 <- 多个 chunk）
    await asyncio.to_thread(store.set_chunk_entities, chunk_id, entities)

    await _emit(on_event, "graph_written", chunk_id=chunk_id, entities=n_ent,
          relations=n_rel, linked_entities=n_link)
    return {
        "entities": n_ent, "relations": n_rel, "linked_entities": n_link,
        "new_entities": sum(1 for e in entities if not e.get("reused")),
        "reused_entities": sum(1 for e in entities if e.get("reused")),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


async def _extract_with_llm(text: str, existing: list[dict], existing_rels: list[dict],
                            on_event: Event | None = None) -> dict:
    """LLM 流式抽取 + 解析器解析 + 时间兜底。

    返回 {"parsed", "raw", "warnings", "time_source", "default_time",
          "time_llm_count", "time_fallback_count"}。

    时间裁决规则（与需求一致）：
      1. LLM 先检测这句话里有没有时间（可到年/月/日/时），有就原样填 time；
      2. 抽不到的一律留空；
      3. 再统一兜底：所有仍然为空的时间都填上 get_current_time 取到的系统时间。
    因此常见结果是「混合」的 —— 部分用 LLM 抽到的，部分用系统时间。
    time_source 反映的是"这句话本身是否被 LLM 抽到了时间"。
    """
    from app.core.prompt import build_extract_messages

    messages = build_extract_messages(text, existing, existing_rels)

    async def _delta(d: str) -> None:
        await _emit(on_event, "llm_delta", delta=d)

    await _emit(on_event, "llm_start", model=config.llm.model, known_entities=len(existing))
    try:
        raw = await chat_stream(messages, on_delta=_delta)
    except Exception as e:  # noqa: BLE001
        raise LLMError(f"LLM 抽取失败: {e}") from e
    await _emit(on_event, "llm_end", chars=len(raw), raw=raw)

    parsed, warnings = parse_llm_output(raw)

    # 兜底前先数一下 LLM 真正抽到几个时间
    llm_times = sum(1 for e in parsed["entities"] if e.get("time")) + \
                sum(1 for r in parsed["relations"] if r.get("time"))

    need_fallback = (not has_time(text)) or llm_times == 0 or \
        any(not e.get("time") for e in parsed["entities"]) or \
        any(not r.get("time") for r in parsed["relations"])

    if need_fallback:
        default_time = await _sys_time()
        parsed = resolve_times(parsed, default_time)
    else:
        default_time = parsed["entities"][0].get("time") if parsed["entities"] else ""

    fallback_count = sum(1 for e in parsed["entities"] if e.get("time_filled")) + \
        sum(1 for r in parsed["relations"] if r.get("time_filled"))

    time_source = "llm" if llm_times > 0 else "system"
    await _emit(on_event, "time_resolved", source=time_source, time=default_time,
          llm_count=llm_times, fallback_count=fallback_count,
          note=(f"LLM 从这句话里抽到 {llm_times} 个时间；另有 {fallback_count} 个实体/关系"
                f"抽不到时间，已用 get_current_time 兜底为 {default_time}")
          if (llm_times and fallback_count) else
          (f"这句话里没检测到时间，已调用 get_current_time 取系统时间 {default_time} 兜底"
           if not llm_times else "时间全部由 LLM 从这句话中抽取，无需兜底"))

    # 已有实体复用标记
    existing_keys = {e.get("key") or norm_key(e.get("name") or "") for e in existing}
    for e in parsed["entities"]:
        e["reused"] = e["key"] in existing_keys
    for r in parsed["relations"]:
        r.setdefault("time", default_time)

    return {
        "parsed": parsed, "raw": raw, "warnings": warnings,
        "time_source": time_source, "default_time": default_time,
        "time_llm_count": llm_times, "time_fallback_count": fallback_count,
    }


async def _ingest_neo4j(chunk_id: str, content: str, text: str,
                        on_event: Event | None = None) -> dict:
    """完整的 neo4j 入库线：读已有实体 -> LLM 抽取 -> 解析 -> 写入。"""
    existing, existing_rels = await _read_existing_graph(on_event)
    out = await _extract_with_llm(text, existing, existing_rels, on_event)
    parsed, raw, warnings = out["parsed"], out["raw"], out["warnings"]
    for w in warnings:
        await _emit(on_event, "warn", chunk_id=chunk_id, message=w)
    res = await _write_neo4j(chunk_id, content, parsed, on_event)
    res.update({"parsed": parsed, "llm_raw": raw[:2000],
                "time_source": out["time_source"], "time": out["default_time"]})
    return res


# ------------------------------------------------------------------ 单个 chunk
async def ingest_chunk(chunk_id: str, on_event: Event | None = None) -> dict:
    """把队列里的一个 chunk 过一遍两条入库线，都成功才 promote 进 chunks 表。

    兼容两种来源：
    - 还在 ingest_queue 的 chunk（待入库/失败重试）：跑完后 promote 进 chunks 表；
    - 已转正的 chunk（chunks 表已有）：重新跑两条线并刷新 ingested_at（重新入库）。
    """
    row = await asyncio.to_thread(store.get_queue_item, chunk_id)
    promoted_row = None
    if not row:
        promoted_row = await asyncio.to_thread(store.get_chunk, chunk_id)
        if not promoted_row:
            raise ValueError(f"chunk {chunk_id} 既不在入库队列中，也不在 chunks 表中")
        content = promoted_row["content"]
        await asyncio.to_thread(store.update_chunk_status, chunk_id, error="")
    else:
        content = row["content"]
        await asyncio.to_thread(store.update_chunk_status, chunk_id, error="")

    await asyncio.to_thread(store.update_chunk_status, chunk_id, error="")

    async def run_qdrant() -> dict:
        try:
            r = await _ingest_qdrant(chunk_id, content)
            await asyncio.to_thread(store.update_chunk_status, chunk_id, "success", None, "")
            await _emit(on_event, "qdrant_ok", chunk_id=chunk_id, **r)
            return {"line": "qdrant", "ok": True, **r}
        except Exception as e:  # noqa: BLE001
            await asyncio.to_thread(store.update_chunk_status, chunk_id, "failed", None, str(e)[:500])
            await _emit(on_event, "qdrant_fail", chunk_id=chunk_id, error=str(e)[:300])
            return {"line": "qdrant", "ok": False, "error": str(e)}

    async def run_neo4j() -> dict:
        try:
            if promoted_row is not None:
                # 重新入库：先清理这个 chunk 的旧图产物（断开 MENTIONS、删旧 REL；
                # 实体若不再被其它 chunk 引用则删除，否则保留复用）
                cleaned = await neo4j_store.cleanup_chunk(chunk_id)
                await _emit(on_event, "cleanup", chunk_id=chunk_id, **cleaned)
            r = await _ingest_neo4j(chunk_id, content, content, on_event)
            await asyncio.to_thread(store.update_chunk_status, chunk_id, None, "success", "")
            await _emit(on_event, "neo4j_ok", chunk_id=chunk_id, **{k: v for k, v in r.items() if k != "parsed"})
            return {"line": "neo4j", "ok": True, **{k: v for k, v in r.items() if k != "parsed"}}
        except Exception as e:  # noqa: BLE001
            await asyncio.to_thread(store.update_chunk_status, chunk_id, None, "failed", str(e)[:500])
            await _emit(on_event, "neo4j_fail", chunk_id=chunk_id, error=str(e)[:300])
            return {"line": "neo4j", "ok": False, "error": str(e)}

    # 两条入库线并行执行
    res = await asyncio.gather(run_qdrant(), run_neo4j())

    ok_q = next(r for r in res if r["line"] == "qdrant")["ok"]
    ok_n = next(r for r in res if r["line"] == "neo4j")["ok"]

    promoted = False
    if ok_q and ok_n:
        if promoted_row is not None:
            # 已转正的 chunk：重新入库成功后刷新 ingested_at（chunk_id 不变）
            ts = now_iso()
            await asyncio.to_thread(store.update_chunk_ingested_at, chunk_id, ts)
            await _emit(on_event, "promoted", chunk_id=chunk_id,
                  document_id=promoted_row["document_id"], reingested=True, ingested_at=ts)
            promoted = True
        else:
            promoted = await asyncio.to_thread(store.promote_chunk, chunk_id)
            if promoted:
                # promote_chunk 已返回 chunk_id，这里显式覆盖一次，避免重复传参
                await _emit(on_event, "promoted", **{**promoted, "chunk_id": chunk_id})
    else:
        # 任一失败就继续留在 ingest_queue 里等重试
        await _emit(on_event, "stayed_in_queue", chunk_id=chunk_id,
              message="qdrant 或 neo4j 有一条未成功，chunk 继续留在入库队列中")

    return {
        "chunk_id": chunk_id,
        "qdrant": "success" if ok_q else "failed",
        "neo4j": "success" if ok_n else "failed",
        "promoted": bool(promoted),
        "preview": content[:60],
        "details": {r["line"]: r for r in res},
    }


# ------------------------------------------------------------------ 整篇文档
async def ingest_document(text: str, source: str = "manual",
                          on_event: Event | None = None,
                          meta: dict | None = None) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("文本不能为空")

    t0 = time.perf_counter()
    doc = await asyncio.to_thread(store.add_document, text, source, meta or {})
    doc_id = doc["id"]
    await _emit(on_event, "document", document_id=doc_id, chars=len(text))

    blocks = chunker.split_text(text)   # 保留空实现：原样返回
    chunk_ids: list[str] = []
    for i, blk in enumerate(blocks):
        cid = await asyncio.to_thread(store.enqueue_chunk, doc_id, blk, i)
        chunk_ids.append(cid)
    await _emit(on_event, "chunked", document_id=doc_id, count=len(chunk_ids),
          info=chunker.chunk_info(text))

    sem = asyncio.Semaphore(max(1, config.app.max_concurrency))

    async def _one(cid: str) -> dict:
        async with sem:
            return await ingest_chunk(cid, on_event)

    results = await asyncio.gather(*(_one(c) for c in chunk_ids))

    done = sum(1 for r in results if r["promoted"])
    return {
        "ok": True,
        "document_id": doc_id,
        "chunks_total": len(chunk_ids),
        "chunks_promoted": done,
        "chunks_in_queue": len(chunk_ids) - done,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "results": results,
    }
