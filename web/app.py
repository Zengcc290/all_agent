"""知识星云 · FastAPI 应用。

路由一览：
- GET  /api/graph    全图 nodes+edges（星云图数据源）
- GET  /api/graph-rag 向量证据 + 图关系路径混合检索
- POST /api/chat     与知识管家对话（未配置聊天模型时 503）
- POST /api/ingest   上传文档 → RAG 切块入库（星云长出新星星）
- POST /api/facts    手工添加三元组知识
- POST /api/knowledge 一句话入库：原文向量化 + LLM 自动抽取实体/关系 → 图结构
- POST /api/knowledge/image 图片/相机 → VL embedding + 视觉模型抽取实体、时间和多元关系
- GET  /api/knowledge/jobs 一句话入库历史（后台队列状态：排队中/正在入库/成功/失败）
- POST /api/knowledge/jobs/{job_id}/retry 失败的一句话入库任务重新入队
- POST /api/seed     （重新）播种 Aetheria 种子数据（幂等）
- GET  /api/export   导出全部记忆为 JSON 文件（课设「库→文件」要求）
- POST /api/import   导入此前导出的 JSON（课设「文件→库」要求）
- GET  /api/health   健康检查：嵌入模式、聊天可用性
- /                星云图前端静态页（web/static/index.html）

运行：``python -m web.app``（默认 http://127.0.0.1:8765）
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from constants import (
    LOCALHOST,
    MAX_UPLOAD_BYTES,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_GRAPH_HOPS,
    RAG_GRAPH_MAX_HOPS,
    RAG_RETRIEVE_LIMIT,
    WEB_AUTOSEED,
    WEB_CHAT_MAX_CHARS,
    WEB_FACT_DOMAIN_MAX,
    WEB_FACT_NOTE_MAX,
    WEB_FACT_OBJECT_MAX,
    WEB_FACT_PREDICATE_MAX,
    WEB_FACT_SUBJECT_MAX,
    WEB_GRAPH_RAG_LIMIT_MAX,
    WEB_GRAPH_RAG_QUERY_MAX,
    WEB_INGEST_CHUNK_SIZE,
    WEB_KNOWLEDGE_MAX_CHARS,
)
from core import ExecutionContext
from memory import MemoryManager
from memory.embedding_lock import (
    EmbeddingLockMismatch,
    apply_embedding_lock,
    inspect_embedding_lock,
    mismatch_from_exception,
)
from memory.rag import RAGPipeline
from memory.storage.document_repo import DocumentRepository
from tool.add_fact import add_fact as write_fact
from tool.document_get import get_document as document_payload
from tool.document_list import list_documents as list_documents_payload
from tool.document_revectorize import revectorize_document as revectorize_document_payload
from tool.export_knowledge import export_filename, export_payload
from tool.graph_snapshot import build_graph
from tool.hybrid_recall import hybrid_recall
from tool.import_knowledge import import_items, parse_import_payload
from tool.ingest_image import ingest_image
from tool.reconcile import fact_items, reconcile_report
from tool.repair_drift import repair_drift
from tool.seed_knowledge import seed

from .ingest_queue import IngestJobQueue, job_to_dict
from .support import (
    SEARCH_TOOL_NAME,
    STATIC_DIR,
    build_knowledge_extractor,
    bump_graph_revision,
    chat_confirmed_side_effects,
    chat_ready,
    chat_tool_names,
    close_manager,
    get_agent,
    get_manager,
    graph_revision,
    record_qa,
    schedule_qa_extraction,
    search_available,
)


class ReconcileBody(BaseModel):
    """对账修复请求；``repair`` 为空表示只报告不修。"""

    repair: list[str] = Field(default_factory=list)


def embedding_config_hint(manager: MemoryManager) -> str:
    """云端嵌入未配置时的配置指引（已配置返回空串）。

    历史版本这里是「网关不可达 → 503 门禁」（隧道时代的预检）；云端时代
    端点失败在请求时自然报错，预检与门禁已删，这个提示只用于 /api/health
    的 degraded.embedding_hint，告诉用户为什么检索退化成了关键词。
    """

    base_url = getattr(getattr(manager, "embedding", None), "base_url", None)
    if base_url:
        return ""
    return (
        "云端嵌入未配置，向量检索退化为关键词（离线兜底向量与云端不兼容，"
        "不会静默混用）。请在 config/services.toml 的 [embedding] 段填写 "
        "base_url / api_key / model，然后重启服务。"
    )


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=WEB_CHAT_MAX_CHARS)
    #: 回答模式：offline 只靠本地记忆，online 额外允许联网搜索（web.search）。
    mode: Literal["offline", "online"] = Field(default="offline")


class FactBody(BaseModel):
    subject: str = Field(min_length=1, max_length=WEB_FACT_SUBJECT_MAX)
    predicate: str = Field(min_length=1, max_length=WEB_FACT_PREDICATE_MAX)
    object: str = Field(min_length=1, max_length=WEB_FACT_OBJECT_MAX)
    domain: str | None = Field(default=None, max_length=WEB_FACT_DOMAIN_MAX)
    note: str | None = Field(default=None, max_length=WEB_FACT_NOTE_MAX)
    confidence: float = Field(default=1.0, ge=0, le=1)


class GraphRAGBody(BaseModel):
    query: str = Field(min_length=1, max_length=WEB_GRAPH_RAG_QUERY_MAX)
    limit: int = Field(default=RAG_RETRIEVE_LIMIT, ge=1, le=WEB_GRAPH_RAG_LIMIT_MAX)
    hops: int = Field(default=RAG_GRAPH_HOPS, ge=0, le=RAG_GRAPH_MAX_HOPS)
    at: str | None = Field(default=None, max_length=80)


class KnowledgeBody(BaseModel):
    text: str = Field(min_length=1, max_length=WEB_KNOWLEDGE_MAX_CHARS)
    event_at: str | None = Field(default=None, max_length=80)
    #: True（默认，兼容旧行为/测试）同步等待结果；False 提交后台队列立即返回。
    wait: bool = True


async def _save_upload(
    file: UploadFile, *, prefix: str, suffix: str | None = None
) -> Path:
    """Stream one upload to a temp file while enforcing ``MAX_UPLOAD_BYTES``.

    Both upload endpoints share this: reading ``await file.read()`` into memory
    let a single large body exhaust the process, and only ``/api/ingest`` used to
    be bounded. Oversized input is rejected with 413 before any parsing, empty
    input with 400; the caller unlinks the returned path.
    """

    filename = file.filename or "untitled"
    extension = suffix if suffix is not None else (Path(filename).suffix or ".txt")
    total_size = 0
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=extension, prefix=prefix
    ) as tmp:
        while chunk := await file.read(1024 * 1024):
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"文件超过上传上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB"
                    ),
                )
            tmp.write(chunk)
        if total_size == 0:
            tmp.close()
            os.unlink(tmp.name)
            raise HTTPException(status_code=400, detail="上传内容为空")
        return Path(tmp.name)


def create_app(manager: MemoryManager | None = None) -> FastAPI:
    """应用工厂。``manager`` 可注入（测试用内存库）；默认用共享单例。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_manager = manager is None
        app.state.manager = manager if manager is not None else get_manager()
        app.state.pipeline = RAGPipeline(
            app.state.manager,
            extractor=build_knowledge_extractor(),
        )
        # 一句话后台入库队列：提交即返回，状态持久化在 ingest_jobs 表，
        # 异常退出后由 start() 复位续跑。:memory: 测试库下 available=False。
        # on_progress：任务终态时刷星云图缓存，前端 since 轮询自动看到新图。
        app.state.ingest_queue = IngestJobQueue(
            app.state.manager,
            app.state.pipeline.extractor,
            on_progress=lambda _job_id: invalidate_graph(),
        )
        app.state.ingest_queue.start()
        # 知识管家是进程级单例，且其对话历史是一份共享的可变状态：并发问答会
        # 互相覆盖历史。这里串行化聊天请求（单人本地应用，排队是可接受的代价）。
        # 在 lifespan 内创建以保证锁绑定到当前事件循环。
        app.state.chat_lock = asyncio.Lock()
        if WEB_AUTOSEED:
            # 首次启动自动播种，让星云图一打开就有内容（开关在 constants.py）。
            seed(app.state.manager)
        yield
        app.state.ingest_queue.shutdown()
        if owns_manager:
            close_manager()

    app = FastAPI(title="知识星云 · 个人知识库", version="0.1.0", lifespan=lifespan)

    def the_manager() -> MemoryManager:
        return app.state.manager

    def guard_embedding(*, confirm_rebuild: bool = False) -> None:
        """409 before ingest/reindex when SQLite lock or live Qdrant dimension mismatches."""

        manager = the_manager()
        repository = app.state.pipeline.document_repo()
        try:
            apply_embedding_lock(manager, repository, confirm_rebuild=confirm_rebuild)
        except EmbeddingLockMismatch as exc:
            raise HTTPException(status_code=409, detail=exc.to_detail()) from exc

    def raise_embedding_http(exc: BaseException) -> None:
        mapped = mismatch_from_exception(
            exc, the_manager(), app.state.pipeline.document_repo()
        )
        if mapped is not None:
            raise HTTPException(status_code=409, detail=mapped.to_detail()) from exc

    # ------------------------------------------------------------------
    # 星云图缓存：任何写操作递增 revision，/api/graph 命中缓存避免全量重建。
    # 数据量大时 build_graph 是全库 O(N) 遍历，每请求重建会拖慢打开/刷新。
    # 后台问答抽取线程不改本字典，而是递增 support.GRAPH_REVISION；读请求
    # 比对两个 revision，落后才重建。
    # ------------------------------------------------------------------
    graph_cache: dict[str, Any] = {
        "external": graph_revision(),
        "payload": None,
    }

    def invalidate_graph() -> None:
        # 本地写入与后台抽取共用同一个进程级计数：/api/graph?since= 只有一个真相来源，
        # 否则刚通过 API 写完就会被 since 判成「无变化」。
        bump_graph_revision()
        graph_cache["payload"] = None

    # ------------------------------------------------------------------
    # 星云图数据
    # ------------------------------------------------------------------
    @app.get("/api/graph")
    def graph(since: int = -1, at: str | None = None) -> dict[str, Any]:
        """Read the real graph; ``at`` selects a historical ISO-8601 instant.

        Process-local cache is keyed by revision + as-of time. Writes bump
        GRAPH_REVISION so Neo4j Aura updates from this app still invalidate.
        """

        revision = graph_revision()
        cache_at = at or ""
        try:
            if (
                graph_cache["payload"] is None
                or graph_cache["external"] != revision
                or graph_cache.get("at") != cache_at
            ):
                graph_cache["payload"] = build_graph(the_manager(), at=at)
                graph_cache["external"] = revision
                graph_cache["at"] = cache_at
            payload = graph_cache["payload"]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"时间参数无效：{exc}") from exc
        if not cache_at and since >= 0 and since == revision:
            return {
                "revision": revision,
                "unchanged": True,
                "nodes": [],
                "edges": [],
                "stats": payload.get("stats", {}),
            }
        return {**payload, "revision": revision, "unchanged": False}

    @app.post("/api/graph-rag")
    def graph_rag(body: GraphRAGBody) -> dict[str, Any]:
        result = app.state.pipeline.graph_retrieve(
            body.query,
            limit=body.limit,
            hops=body.hops,
            at=body.at,
        )
        return result.to_dict() | {"context": result.build_context()}

    # ------------------------------------------------------------------
    # 聊天（得力助手）
    # ------------------------------------------------------------------
    @app.post("/api/chat")
    async def chat(body: ChatBody) -> dict[str, Any]:
        ready, reason = chat_ready()
        if not ready:
            raise HTTPException(status_code=503, detail=reason)
        agent = get_agent()
        online = body.mode == "online"
        # 联网模式只在 AnySearch 已配置时真正开放 web.search；否则按非联网处理。
        tool_names = chat_tool_names(agent, online=online)
        effective_mode = "online" if SEARCH_TOOL_NAME in tool_names else "offline"
        try:
            # agent.run 是同步阻塞调用，丢进线程避免卡住事件循环。
            # chat_lock：agent 是共享单例且内部历史无锁，串行化避免并发问答
            # 互相污染上下文（后发请求排队，而不是并发改写同一份历史）。
            # 上下文只为 memory.add 预置写确认：用户这一轮明确要求「记住」时
            # 模型才能落库；删除/清空/入库仍需人工确认。
            context = ExecutionContext(
                confirmed_side_effects=chat_confirmed_side_effects(agent)
            )
            async with app.state.chat_lock:
                answer = await asyncio.to_thread(
                    agent.run,
                    body.message,
                    tool_names=tool_names,
                    context=context,
                )
        except Exception as exc:  # noqa: BLE001 - 任意上游失败都归一为可读的 502
            raise HTTPException(
                status_code=502, detail=f"聊天模型调用失败：{type(exc).__name__}: {exc}"
            )
        # 问答留痕：每次问答都写进 episodic 记忆（带时间戳、可检索），
        # 时间线上会新增一颗「问：…」事件星。
        record_qa(the_manager(), body.message, answer, mode=effective_mode)
        # 再把这次问答交给 LLM 转成图补丁，后台执行：抽取是第二次模型往返，
        # 不能让用户为它多等一轮。抽取成功会递增 GRAPH_REVISION，图缓存自动失效。
        schedule_qa_extraction(
            body.message, answer, manager=the_manager()
        )
        invalidate_graph()
        try:
            retrieval = app.state.pipeline.graph_retrieve(
                body.message, limit=RAG_RETRIEVE_LIMIT, hops=RAG_GRAPH_HOPS
            )
        except Exception:  # noqa: BLE001 - graph retrieve is post-answer; keep the chat 200
            from memory.rag.graph_rag import GraphRAGResult

            retrieval = GraphRAGResult(query=body.message)
        # U4 溯源：同一句问话再走一遍混合召回（向量 × FTS5 RRF），把每条的
        # 向量分/关键词分/融合分交给前端做「依据」面板；网关不可用时这里退化为
        # 纯关键词，正好让降级原因对用户可见，而不是只显示一个空来源列表。
        # 混合召回的唯一实现在 tool/hybrid_recall.py。
        hybrid = hybrid_recall(app.state.pipeline, body.message, limit=RAG_RETRIEVE_LIMIT)
        retrieval_report = {
            "note": hybrid.note,
            "hits": [
                {
                    "chunk_id": hit.memory_id,
                    "document_id": hit.metadata.get("document_id"),
                    "chunk_index": hit.metadata.get("chunk_index"),
                    "snippet": (hit.content or "")[:200],
                    "score": hit.score,
                    "rrf_score": hit.detail.get("rrf_score"),
                    "vector_score": hit.detail.get("vector_score"),
                    "keyword_score": hit.detail.get("keyword_score"),
                }
                for hit in hybrid.chunks
            ],
        }
        return {
            "answer": answer,
            "mode": effective_mode,
            "sources": [
                {
                    "memory_id": result.item.id,
                    "score": result.score,
                    "source": result.item.metadata.get("filename")
                    or result.item.metadata.get("source"),
                    "chunk_id": result.item.metadata.get("chunk_id")
                    or result.item.metadata.get("document_id"),
                }
                for result in retrieval.evidence
            ],
            "paths": [path.to_dict() for path in retrieval.paths],
            "retrieval": retrieval_report,
        }

    # ------------------------------------------------------------------
    # 文档导入（RAG 摄取）
    # ------------------------------------------------------------------
    @app.post("/api/ingest")
    async def ingest(
        file: UploadFile,
        confirm_rebuild: bool = Query(default=False),
    ) -> dict[str, Any]:
        filename = file.filename or "untitled"
        tmp_path = await _save_upload(file, prefix="nebula-ingest-")
        try:
            guard_embedding(confirm_rebuild=confirm_rebuild)
            items = app.state.pipeline.ingest_source(
                tmp_path,
                metadata={"source": filename, "filename": filename},
                chunk_size=WEB_INGEST_CHUNK_SIZE,
                overlap=RAG_CHUNK_OVERLAP,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - 解析失败归一为 422，附错误类型
            raise_embedding_http(exc)
            raise HTTPException(
                status_code=422, detail=f"文档解析失败：{type(exc).__name__}: {exc}"
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        the_manager().episodic.record(
            f"上传并导入了文档《{filename}》（{len(items)} 个知识块）",
            metadata={"title": "导入文档", "filename": filename},
        )
        invalidate_graph()
        return {
            "filename": filename,
            "chunks": len(items),
            "extraction": dict(app.state.pipeline.last_ingest_report),
        }

    # ------------------------------------------------------------------
    # 手工添加三元组
    # ------------------------------------------------------------------
    @app.post("/api/facts")
    def add_fact(body: FactBody) -> dict[str, Any]:
        # 事实写入的唯一实现在 tool/add_fact.py（工具名 knowledge.add_fact）。
        item = write_fact(
            the_manager(),
            subject=body.subject,
            predicate=body.predicate,
            object=body.object,
            domain=body.domain or "",
            note=body.note or "",
            confidence=body.confidence,
        )
        invalidate_graph()
        return {"ok": True, "item_id": item.id}

    # ------------------------------------------------------------------
    # 一句话添加知识（原文向量化 + LLM 自动抽取实体/关系 → 图结构）
    # wait=false：写一条 ingest_jobs 记录立即返回，后台 worker 并发入库，
    # 状态（排队中/正在入库/成功/失败）持久化在 SQLite，可查历史、重启续跑。
    # ------------------------------------------------------------------
    @app.post("/api/knowledge")
    def add_knowledge(
        body: KnowledgeBody,
        confirm_rebuild: bool = Query(default=False),
    ) -> dict[str, Any]:
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="一句话内容不能为空")
        guard_embedding(confirm_rebuild=confirm_rebuild)
        queue = getattr(app.state, "ingest_queue", None)

        if queue is not None and queue.available:
            job = queue.submit(text, event_at=body.event_at or "")
            if not body.wait:
                return {
                    "ok": True,
                    "job_id": job.job_id,
                    "status": job.status,
                    "label": "排队中",
                    "async": True,
                }
            finished = queue.wait(job.job_id)
            if finished is None or finished.status != "done":
                error = finished.error if finished is not None else "等待入库超时"
                raise HTTPException(
                    status_code=500,
                    detail=f"入库失败：{error or '未知错误'}",
                )
            try:
                result = json.loads(finished.result) if finished.result else {}
            except ValueError as exc:
                # 后台任务写入的 result 必须是合法 JSON；对端损坏时给可读错误，
                # 而不是让解析异常变成 500 内部错误。
                raise HTTPException(
                    status_code=500, detail=f"入库结果解析失败：{exc}"
                ) from exc
            report = result.get("report") if isinstance(result.get("report"), dict) else {}
            return {
                "ok": True,
                "job_id": finished.job_id,
                "chunks": result.get("chunks", 0),
                "items": [],
                "extraction": report,
            }

        # 回退：无持久化 SQLite（如 :memory: 测试库）时保持原同步行为。
        pipeline = app.state.pipeline
        from memory.rag import Document

        preview = text.splitlines()[0][:40]
        items = pipeline.ingest(
            Document(
                text,
                metadata={
                    "source": "一句话入库",
                    "filename": preview,
                    "note": text[:400],
                    "event_at": body.event_at or "",
                    "reference_time": datetime.now(UTC).isoformat(),
                },
            ),
            chunk_size=RAG_CHUNK_SIZE,
            overlap=RAG_CHUNK_OVERLAP,
        )
        report = pipeline.last_ingest_report
        the_manager().episodic.record(
            f"添加了一条知识：{text[:80]}",
            metadata={"title": "一句话入库", "source": "一句话入库"},
        )
        invalidate_graph()
        return {
            "ok": True,
            "chunks": len(items),
            "items": [item.to_dict() for item in items],
            "extraction": report,
        }

    @app.get("/api/knowledge/jobs")
    def list_knowledge_jobs(status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """入库历史记录：每条一句话任务的持久化状态，前端轮询展示。"""

        queue = getattr(app.state, "ingest_queue", None)
        if queue is None or not queue.available:
            return {"items": [], "available": False, "workers": 0}
        if status is not None and status not in ("pending", "running", "done", "failed"):
            raise HTTPException(status_code=422, detail="status 取值必须是 pending/running/done/failed")
        # 仓储层内部有 200 条硬上限，这里在 API 层再夹一层，避免超大步进直接
        # 打到 SQL 层（同时让前端拿到的 limit 就是实际生效值）。
        jobs = queue.list(status=status, limit=max(1, min(int(limit), 200)))
        return {
            "items": [job_to_dict(job) for job in jobs],
            "available": True,
            "workers": getattr(queue, "_workers", 0),
        }

    @app.post("/api/knowledge/jobs/{job_id}/retry")
    def retry_knowledge_job(
        job_id: str,
        confirm_rebuild: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Re-queue a failed one-sentence ingest job after the embedding gate."""

        guard_embedding(confirm_rebuild=confirm_rebuild)
        queue = getattr(app.state, "ingest_queue", None)
        if queue is None or not queue.available:
            raise HTTPException(status_code=503, detail="入库队列不可用")
        try:
            job = queue.retry(job_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="入库任务不存在") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "async": True, **job_to_dict(job)}

    @app.post("/api/knowledge/image")
    async def add_image_knowledge(
        file: UploadFile,
        text: str = Form(default=""),
        captured_at: str = Form(default=""),
        confirm_rebuild: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Index a camera/image observation and extract its n-ary graph facts."""

        filename = file.filename or "camera.jpg"
        mime_type = (file.content_type or "").split(";", 1)[0].strip().lower()
        if not mime_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="只接受图片文件")
        if len(text) > WEB_KNOWLEDGE_MAX_CHARS:
            raise HTTPException(status_code=422, detail="图片说明过长")
        tmp_path = await _save_upload(file, prefix="nebula-image-")
        try:
            image = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
        metadata = {
            "source": filename,
            "filename": filename,
            "captured_at": captured_at.strip(),
            "reference_time": datetime.now(UTC).isoformat(),
            "modality": "image",
        }
        try:
            guard_embedding(confirm_rebuild=confirm_rebuild)
            # 图片入库的唯一实现在 tool/ingest_image.py（工具名 knowledge.ingest_image）。
            result = ingest_image(
                app.state.pipeline,
                image=image,
                text=text,
                mime_type=mime_type,
                metadata=metadata,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise_embedding_http(exc)
            if isinstance(exc, (TypeError, ValueError)):
                raise HTTPException(status_code=422, detail=f"图片入库参数无效：{exc}") from exc
            raise
        invalidate_graph()
        return {"ok": True, **result}

    # ------------------------------------------------------------------
    # 播种 / 导出 / 导入（课设硬性要求）
    # ------------------------------------------------------------------
    @app.post("/api/seed")
    def reseed() -> dict[str, Any]:
        result = seed(the_manager())
        invalidate_graph()
        return result

    @app.get("/api/export")
    def export(request: Request) -> JSONResponse:
        # 载荷构造的唯一实现在 tool/export_knowledge.py（limit=0 表示全量导出）。
        payload = export_payload(the_manager(), limit=0)
        # 导出文件名用本地时间戳（面向用户，非持久化时间语义）。
        return JSONResponse(
            payload,
            headers={
                "Content-Disposition": f'attachment; filename="{export_filename()}"'
            },
        )

    @app.post("/api/import")
    async def import_file(file: UploadFile) -> dict[str, Any]:
        tmp_path = await _save_upload(file, prefix="nebula-import-", suffix=".json")
        try:
            raw = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
        # 解析与逐条写入的唯一实现在 tool/import_knowledge.py；非法 JSON 仍是 400。
        try:
            entries = parse_import_payload(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = import_items(the_manager(), entries)
        if result["imported"]:
            invalidate_graph()
        return result

    # ------------------------------------------------------------------
    # 文档中心与三库对账（documents/chunks 真值源 → 向量/图投影）
    #
    # 对账与漂移自愈的逻辑本身已收敛到 tool/reconcile.py 与 tool/repair_drift.py
    # （各自是可被 Agent 调用的独立工具），这里只保留 HTTP 边界与错误码映射。
    # ------------------------------------------------------------------
    def the_repository() -> DocumentRepository:
        """复用管道缓存的那个仓储：同一 sqlite 文件、自带锁、每作用域独立连接。"""

        repository = app.state.pipeline.document_repo()
        if repository is None:
            raise HTTPException(
                status_code=400,
                detail="当前记忆库是内存模式（:memory:），没有 documents/chunks 真值源",
            )
        return repository

    @app.get("/api/documents")
    def list_documents(tag: str = "", status: str = "", page: int = 1, page_size: int = 20) -> dict[str, Any]:
        # 列表逻辑的唯一实现在 tool/document_list.py（含 page_size 上限校验）。
        try:
            return list_documents_payload(
                the_repository(), tag=tag, status=status, page=page, page_size=page_size
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str) -> dict[str, Any]:
        # 详情逻辑的唯一实现在 tool/document_get.py。
        try:
            return document_payload(the_repository(), document_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/documents/{document_id}/revectorize")
    def revectorize_document(
        document_id: str,
        confirm_rebuild: bool = Query(default=False),
    ) -> dict[str, Any]:
        """重建该文档的向量投影；网关不可达时明确失败，绝不切换到别的向量空间（D8）。"""

        # 重嵌入的唯一实现在 tool/document_revectorize.py（锁闸门顺序也一致）。
        try:
            return revectorize_document_payload(
                the_manager(),
                the_repository(),
                document_id,
                confirm_rebuild=confirm_rebuild,
            )
        except EmbeddingLockMismatch as exc:
            raise HTTPException(status_code=409, detail=exc.to_detail()) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise_embedding_http(exc)
            raise HTTPException(status_code=502, detail=f"重嵌入失败：{type(exc).__name__}: {exc}") from exc

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        manager = the_manager()
        repository = app.state.pipeline.document_repo()
        counts = repository.stats() if repository is not None else {"documents": 0, "chunks": 0, "chunks_indexed": 0}
        return {
            **counts,
            "facts": len(fact_items(manager)),
            "memories_total": len(manager.document_store.list(include_expired=True)),
        }

    @app.get("/api/reconcile")
    def reconcile() -> dict[str, Any]:
        return reconcile_report(the_manager(), app.state.pipeline.document_repo())

    @app.post("/api/reconcile")
    def reconcile_repair(body: ReconcileBody) -> dict[str, Any]:
        try:
            return repair_drift(
                the_manager(), app.state.pipeline.document_repo(), body.repair
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/embedding/rebuild")
    def rebuild_embedding(confirm_rebuild: bool = Query(default=False)) -> dict[str, Any]:
        """Confirm and rebuild the vector projection at the current embedding."""

        if not confirm_rebuild:
            guard_embedding(confirm_rebuild=False)
        else:
            guard_embedding(confirm_rebuild=True)
        invalidate_graph()
        snapshot = inspect_embedding_lock(the_manager(), app.state.pipeline.document_repo())
        return {
            "ok": True,
            "rebuilt": bool(confirm_rebuild),
            "embedding_lock": snapshot["locked"],
            "qdrant_dimension": snapshot["qdrant_dimension"],
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        ready, _ = chat_ready()
        # 报告实际生效的嵌入实现，而不是猜配置：调用方可注入自定义 embedding，
        # 所以看实例属性（云端实现带 base_url，离线 HashEmbedding 没有）。
        embedding = getattr(the_manager(), "embedding", None)
        base_url = getattr(embedding, "base_url", None)
        embedding_mode = "api" if base_url else "local-hash"
        embedding_state = "api" if base_url else "hash"
        manager = the_manager()
        repository = app.state.pipeline.document_repo()
        snapshot = inspect_embedding_lock(manager, repository)
        lock = snapshot["locked"]
        return {
            "ok": True,
            "chat_ready": ready,
            "embedding_mode": embedding_mode,
            "embedding_reachable": bool(base_url),
            "embedding": getattr(embedding, "to_dict", dict)(),
            "vision_model": getattr(getattr(app.state, "pipeline", None), "extractor", None)
            and getattr(app.state.pipeline.extractor, "vision_model", None),
            "search_available": search_available(),
            # 三个存储各自的实现（D1：全部在本机），UI 用它区分「本地真值 / 本地投影」。
            "store_modes": {
                "document": "memory" if str(getattr(manager.document_store, "path", "")) == ":memory:" else "sqlite",
                "vector": type(manager.vector_store).__name__,
                "graph": "neo4j" if getattr(manager.graph_store, "driver", None) is not None else "inmemory",
            },
            # 出网点的实时状态（D8/D9）：云端嵌入未配置时检索只能走 FTS5，
            # UI 必须如实提示（指引文案由 embedding_config_hint 生成）。
            "degraded": {
                "embedding": embedding_state,
                "embedding_endpoint": base_url or "",
                "chat_ready": ready,
                "keyword_fallback": not bool(base_url),
                "embedding_hint": embedding_config_hint(manager),
            },
            "embedding_lock": lock,
            "embedding_current": snapshot["current"],
            "embedding_projection": snapshot["projection"],
            "qdrant_dimension": snapshot["qdrant_dimension"],
            "embedding_mismatch": snapshot["mismatch"],
            "knowledge_extractor": type(getattr(app.state.pipeline, "extractor", None)).__name__,
        }

    # ------------------------------------------------------------------
    # 静态前端（注册在 API 路由之后，避免吞掉 /api/*）
    # ------------------------------------------------------------------
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    # 监听端口唯一来源：constants.DEFAULT_WEB_PORT（历史 NEBULA_PORT 环境变量已删）。
    from constants import DEFAULT_WEB_PORT

    uvicorn.run(app, host=LOCALHOST, port=DEFAULT_WEB_PORT)
