"""FastAPI 路由：工具发现 / 调用、待入库列表、重新入库、一句话入库（SSE）、查询。"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app import config
from app.core.parser import parse_llm_output, resolve_times
from app.core.registry import registry
from app.core.validation import ToolValidationError
from app.db.sqlite_store import store
from app.llm import test_connection
from app.tools.ingest_sentence import ingest_sentence, subscribe
from app.tools.list_pending_chunks import _list_pending

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------- 基础
@router.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "kg-ingest", "time": time.time()}


@router.get("/tools")
async def list_tools() -> dict:
    """把注册表里所有工具（含参数 schema）吐给前端，前端动态生成调用表单与提示词预览。"""
    return {
        "ok": True,
        "count": len(registry.names()),
        "tools": registry.schemas(),
        "prompt_preview": registry.describe(),
    }


@router.post("/tools/rediscover")
async def rediscover() -> dict:
    """手动触发一次工具发现（热更新 tools 目录）。"""
    registry.reload()
    info = await registry.discover()
    return {"ok": True, **info}


@router.post("/tools/{name}/call")
async def call_tool(name: str, body: dict | None = None) -> dict:
    """统一工具入口：严格参数校验 -> 执行 -> 统一返回。"""
    body = body or {}
    args = body.get("args", body)
    try:
        return await registry.call(name, args, body.get("timeout"))
    except ToolValidationError as e:
        raise HTTPException(status_code=422, detail={"stage": "validation", "message": str(e)}) from e
    except Exception as e:  # noqa: BLE001
        detail = {"stage": "execute", "message": str(e), "type": type(e).__name__}
        raise HTTPException(status_code=400, detail=detail) from e


@router.post("/tools/call_many")
async def call_many(body: dict) -> dict:
    calls = body.get("calls")
    if not isinstance(calls, list) or not calls:
        raise HTTPException(422, {"stage": "validation", "message": "需要一个非空 calls 数组"})
    t0 = time.perf_counter()
    results = await registry.call_many(calls)
    return {"ok": all(r.get("ok") for r in results), "results": results,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------------------------------------------------------------- 待入库 / 重新入库
@router.get("/chunks/pending")
async def chunks_pending(limit: int = 20, preview_chars: int = 10) -> dict:
    return await _list_pending(limit=limit, preview_chars=preview_chars)


@router.post("/chunks/{chunk_id}/reingest")
async def reingest_chunk(chunk_id: str) -> dict:
    """前端「重新入库」按钮的真实入口：Chunk 重新过 qdrant + neo4j 两条线。"""
    try:
        return await registry.call("ingest_chunk", {"chunk_id": chunk_id})
    except ToolValidationError as e:
        raise HTTPException(422, {"stage": "validation", "message": str(e)}) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, {"stage": "execute", "message": str(e)}) from e


@router.get("/documents")
async def documents(limit: int = 50) -> dict:
    return {"ok": True, "items": await asyncio.to_thread(store.list_documents, limit)}


@router.get("/chunks")
async def chunks(limit: int = 50) -> dict:
    rows = await asyncio.to_thread(store.list_chunks, limit)
    for r in rows:
        r["entities"] = [e.get("entity_name") for e in await asyncio.to_thread(store.get_chunk_entities, r["chunk_id"])]
    return {"ok": True, "items": rows}


@router.get("/graph")
async def graph() -> dict:
    """实体星球数据源。成功与失败都返回顶层 nodes/links，保证前端拿到的形状一致。"""
    try:
        snap = await registry.call("get_graph_snapshot", {"limit": 500, "relation_limit": 3000})
        res = snap.get("result") or {}
        return {
            "ok": bool(res.get("ok", True)),
            "nodes": res.get("nodes") or [],
            "links": res.get("links") or [],
            "stats": res.get("stats") or {},
            "directed_links": res.get("directed_links"),
            "undirected_links": res.get("undirected_links"),
            "truncated": res.get("truncated"),
            "elapsed_ms": snap.get("elapsed_ms"),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "nodes": [], "links": [], "stats": {}, "error": str(e)}


@router.get("/stats")
async def stats() -> dict:
    return await registry.call("get_stats", {})


@router.get("/llm/test")
async def llm_test() -> dict:
    return await test_connection()


# ---------------------------------------------------------------- 一句话入库（SSE 流式）
@router.post("/ingest/stream")
async def ingest_stream(request: Request, body: dict) -> StreamingResponse:
    text = (body or {}).get("text") or ""
    if not text.strip():
        raise HTTPException(422, {"stage": "validation", "message": "text 不能为空"})

    queue: asyncio.Queue = asyncio.Queue()

    def _push(stage: str, kw: dict) -> None:
        queue.put_nowait({"event": stage, "data": kw})

    async def _runner() -> None:
        try:
            await ingest_sentence(text, body.get("source") or "sentence", _push)
            _push("done", {"ok": True})
        except Exception as e:  # noqa: BLE001
            _push("done", {"ok": False, "error": str(e)})

    async def gen() -> AsyncGenerator[str, None]:
        # 只通过 on_event 单通道下发，避免再 subscribe 一次导致事件重复
        task = asyncio.create_task(_runner())
        try:
            yield f"data: {json.dumps({'event': 'start', 'data': {'text': text[:200]}}, ensure_ascii=False)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(item, ensure_ascii=False, default=str)}\n\n"
                if item["event"] == "done":
                    break
        finally:
            task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/ingest/sentence")
async def ingest_one(body: dict) -> dict:
    """非流式版一句话入库（结果一次性返回）。"""
    text = (body or {}).get("text") or ""
    if not text.strip():
        raise HTTPException(422, {"stage": "validation", "message": "text 不能为空"})
    try:
        return await ingest_sentence(text, body.get("source") or "sentence")
    except ToolValidationError as e:
        raise HTTPException(422, {"stage": "validation", "message": str(e)}) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, {"stage": "execute", "message": str(e)}) from e


@router.post("/llm/parse")
async def llm_parse(body: dict) -> dict:
    """直接把 LLM 输出丢给解析器工具，返回结构化实体与关系。"""
    raw = (body or {}).get("raw") or ""
    if not raw.strip():
        raise HTTPException(422, {"stage": "validation", "message": "raw 不能为空"})
    return await registry.call("parse_llm_output", {"raw": raw})


# ---------------------------------------------------------------- 应用工厂
def create_app() -> FastAPI:
    application = FastAPI(title="知识图谱入库系统", version="1.0.0")

    @application.middleware("http")
    async def _utf8_json(request: Request, call_next):  # noqa: Ann202
        """给 JSON 响应补上 charset=utf-8。

        FastAPI 默认只写 `application/json`，部分客户端会因此按 latin-1 解码中文。
        （浏览器本身按 UTF-8 解析 JSON，这一步主要是为兼容性兜底。）
        """
        resp = await call_next(request)
        ct = resp.headers.get("content-type", "")
        if ct.startswith("application/json") and "charset" not in ct.lower():
            resp.headers["content-type"] = "application/json; charset=utf-8"
        return resp

    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.app.cors_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)

    @application.on_event("startup")
    async def _startup() -> None:  # noqa: Ann202
        await registry.discover()
        print(f"[startup] 已自动发现并登记 {len(registry.names())} 个工具: {registry.names()}")
        # 尽量把 qdrant 集合建好（qdrant 没起也不阻塞启动）
        try:
            from app.db.qdrant_store import vector_store as qdrant_store
            print(f"[startup] qdrant: {await qdrant_store.ensure_collection()}")
        except Exception as e:  # noqa: BLE001
            print(f"[startup] qdrant 未就绪（不影响启动，入库时会重试）: {e}")

    @application.exception_handler(ToolValidationError)
    async def _validation_handler(_: Request, exc: ToolValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"stage": "validation", "message": str(exc)})

    return application


app = create_app()
