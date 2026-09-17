"""知识星云 · FastAPI 应用。

路由一览：
- GET  /api/graph    全图 nodes+edges（星云图数据源）
- GET  /api/graph-rag 向量证据 + 图关系路径混合检索
- POST /api/chat     与知识管家对话（未配置聊天模型时 503）
- POST /api/ingest   上传文档 → RAG 切块入库（星云长出新星星）
- POST /api/facts    手工添加三元组知识
- POST /api/knowledge 一句话入库：原文向量化 + LLM 自动抽取实体/关系 → 图结构
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

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from constants import (
    DEFAULT_DOMAIN,
    DEFAULT_WEB_PORT,
    LOCALHOST,
    MAX_UPLOAD_BYTES,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_GRAPH_HOPS,
    RAG_GRAPH_MAX_HOPS,
    RAG_RETRIEVE_LIMIT,
    WEB_CHAT_MAX_CHARS,
    WEB_FACT_DOMAIN_MAX,
    WEB_FACT_NOTE_MAX,
    WEB_FACT_OBJECT_MAX,
    WEB_FACT_PREDICATE_MAX,
    WEB_FACT_SUBJECT_MAX,
    WEB_GRAPH_RAG_LIMIT_MAX,
    WEB_GRAPH_RAG_QUERY_MAX,
    WEB_IMPORT_ERRORS_MAX,
    WEB_INGEST_CHUNK_SIZE,
    WEB_KNOWLEDGE_MAX_CHARS,
)
from core import ExecutionContext
from memory import MemoryManager, MemoryType
from memory.embedding import EmbedServerEmbedding, gateway_reachable
from memory.rag import RAGPipeline
from memory.storage.document_repo import DocumentRepository

from .graph_builder import build_graph
from .seed import seed
from .support import (
    STATIC_DIR,
    build_knowledge_extractor,
    bump_graph_revision,
    chat_confirmed_side_effects,
    chat_ready,
    chat_tool_names,
    close_manager,
    ensure_embedding_tunnel,
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


def _is_fact_item(item: Any) -> bool:
    """A semantic row that represents one (subject, predicate, object) fact."""

    metadata = getattr(item, "metadata", {}) or {}
    return all(metadata.get(key) for key in ("subject", "predicate", "object"))


def embedding_tunnel_hint() -> str:
    """隧道命令由部署方通过环境变量下发，避免把服务器地址写进仓库。"""

    return os.getenv("EMBEDDING_TUNNEL_HINT", "").strip()


def embedding_unavailable_detail(manager: MemoryManager) -> str:
    """网关已配置但连不上时给出可执行的说明（方案 §11.6）；可用时返回空串。"""

    base_url = getattr(getattr(manager, "embedding", None), "base_url", None)
    if not base_url or gateway_reachable(base_url):
        return ""
    hint = embedding_tunnel_hint()
    return (
        f"嵌入网关不可达（{base_url}），向量检索与入库已停用（不会切换到别的向量空间）。"
        + (f"请先建立隧道：{hint}" if hint else "请先恢复嵌入网关，命令见本地部署说明。")
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


class KnowledgeBody(BaseModel):
    text: str = Field(min_length=1, max_length=WEB_KNOWLEDGE_MAX_CHARS)


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
        # 按需自动建立嵌入隧道（配置了 EMBEDDING_BASE_URL 且 10800 不可达时）；
        # 失败只降级不阻塞（D8：绝不因隧道问题换向量空间）。
        ensure_embedding_tunnel()
        app.state.manager = manager if manager is not None else get_manager()
        app.state.pipeline = RAGPipeline(
            app.state.manager,
            extractor=build_knowledge_extractor(),
        )
        # 知识管家是进程级单例，且其对话历史是一份共享的可变状态：并发问答会
        # 互相覆盖历史。这里串行化聊天请求（单人本地应用，排队是可接受的代价）。
        # 在 lifespan 内创建以保证锁绑定到当前事件循环。
        app.state.chat_lock = asyncio.Lock()
        if os.getenv("WEB_AUTOSEED", "1") != "0":
            # 首次启动自动播种，让星云图一打开就有内容。
            seed(app.state.manager)
        yield
        if owns_manager:
            close_manager()

    app = FastAPI(title="知识星云 · 个人知识库", version="0.1.0", lifespan=lifespan)

    def the_manager() -> MemoryManager:
        return app.state.manager

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
    def graph(since: int = -1) -> dict[str, Any]:
        """全图；``?since=<revision>`` 时若期间无写入则只回 revision（U6 增量刷新）。

        ``since`` 默认 -1 表示「不带增量语义」——不能用 0 当哨兵，因为进程刚启动时
        revision 就是 0，客户端带着 0 来问会被误判成「没带参数」而永远拿全量。
        """

        revision = graph_revision()
        if graph_cache["payload"] is None or graph_cache["external"] != revision:
            graph_cache["payload"] = build_graph(the_manager())
            graph_cache["external"] = revision
        payload = graph_cache["payload"]
        if since >= 0 and since == revision:
            # 自 since 起没有任何图可见的写入：不回传节点/边，前端沿用本地图，
            # 省掉一次全量 build_graph 与整棵星系树的重排布局。
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
        effective_mode = "online" if tool_names is None else "offline"
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
        # U4 溯源：同一句问话再走一遍混合检索（向量 × FTS5 RRF），把每条的
        # 向量分/关键词分/融合分交给前端做「依据」面板；网关不可用时这里退化为
        # 纯关键词，正好让降级原因对用户可见，而不是只显示一个空来源列表。
        hybrid = app.state.pipeline.hybrid_retrieve(body.message, limit=RAG_RETRIEVE_LIMIT)
        retrieval_report = {
            "note": app.state.pipeline.last_retrieval_note,
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
                for hit in hybrid
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
    async def ingest(file: UploadFile) -> dict[str, Any]:
        filename = file.filename or "untitled"
        # 降级验收（方案 §11.6）：嵌入网关不可达时明确报错并给出隧道命令，
        # 而不是写出一条注定失败的文档记录或返回空结果。
        embedding_hint = embedding_unavailable_detail(the_manager())
        if embedding_hint:
            raise HTTPException(status_code=503, detail=embedding_hint)
        tmp_path = await _save_upload(file, prefix="nebula-ingest-")
        try:
            items = app.state.pipeline.ingest_source(
                tmp_path,
                metadata={"source": filename, "filename": filename},
                chunk_size=WEB_INGEST_CHUNK_SIZE,
                overlap=RAG_CHUNK_OVERLAP,
            )
        except Exception as exc:  # noqa: BLE001 - 解析失败归一为 422，附错误类型
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
        item = the_manager().semantic.add_fact(
            body.subject,
            body.predicate,
            body.object,
            metadata={"domain": body.domain or DEFAULT_DOMAIN, "note": body.note or ""},
            confidence=body.confidence,
        )
        invalidate_graph()
        return {"ok": True, "item_id": item.id}

    # ------------------------------------------------------------------
    # 一句话添加知识（原文向量化 + LLM 自动抽取实体/关系 → 图结构）
    # ------------------------------------------------------------------
    @app.post("/api/knowledge")
    def add_knowledge(body: KnowledgeBody) -> dict[str, Any]:
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="一句话内容不能为空")
        pipeline = app.state.pipeline
        from memory.rag import Document

        items = pipeline.ingest(
            Document(
                text,
                metadata={
                    "source": "一句话入库",
                    "filename": "一句话入库",
                    "note": text[:400],
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
        manager = the_manager()
        items = [item.to_dict() for item in manager.list(include_expired=False)]
        payload = {
            "format": "knowledge-nebula-export/v1",
            "exported_at": datetime.now(UTC).isoformat(),
            "counts": {
                "total": len(items),
                "semantic": sum(
                    1 for item in items if item["memory_type"] == "semantic"
                ),
                "episodic": sum(
                    1 for item in items if item["memory_type"] == "episodic"
                ),
            },
            "items": items,
        }
        # 导出文件名用本地时间戳（面向用户，非持久化时间语义）。
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        return JSONResponse(
            payload,
            headers={
                "Content-Disposition": f'attachment; filename="knowledge_export_{stamp}.json"'
            },
        )

    @app.post("/api/import")
    async def import_file(file: UploadFile) -> dict[str, Any]:
        tmp_path = await _save_upload(file, prefix="nebula-import-", suffix=".json")
        try:
            raw = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"不是合法的 JSON：{exc}")
        entries = data.get("items") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise HTTPException(status_code=400, detail="JSON 中找不到 items 数组")

        manager = the_manager()
        existing_facts = {
            (
                item.metadata.get("subject"),
                item.metadata.get("predicate"),
                item.metadata.get("object"),
            )
            for item in manager.list(memory_type=MemoryType.SEMANTIC)
            if item.metadata.get("subject")
            and item.metadata.get("predicate")
            and item.metadata.get("object")
        }
        imported = skipped = 0
        errors: list[str] = []

        def note_error(message: str) -> None:
            """Keep the response bounded: first WEB_IMPORT_ERRORS_MAX reasons."""
            if len(errors) < WEB_IMPORT_ERRORS_MAX:
                errors.append(message)

        for position, raw_item in enumerate(entries, start=1):
            if not isinstance(raw_item, dict):
                skipped += 1
                note_error(f"第 {position} 项：不是 JSON 对象")
                continue
            item_id = raw_item.get("id")
            content = raw_item.get("content") or ""
            if not item_id or not content:
                skipped += 1
                note_error(
                    f"第 {position} 项（id={item_id or '缺失'}）：缺少 id 或 content"
                )
                continue
            if manager.get(item_id) is not None:
                skipped += 1
                continue
            md = raw_item.get("metadata") or {}
            memory_type = raw_item.get("memory_type") or "semantic"
            importance = raw_item.get("importance", 0.5)
            subject, predicate, obj = (
                md.get("subject"),
                md.get("predicate"),
                md.get("object"),
            )
            try:
                if subject and predicate and obj:
                    # 事实：按三元组幂等，避免重复导入时长出重边。
                    if (subject, predicate, obj) in existing_facts:
                        skipped += 1
                        continue
                    existing_facts.add((subject, predicate, obj))
                    manager.semantic.add_fact(
                        subject,
                        predicate,
                        obj,
                        metadata=md,
                        confidence=float(importance),
                    )
                else:
                    manager.add(
                        content,
                        memory_type=memory_type,
                        metadata=md,
                        item_id=item_id,
                        importance=float(importance),
                    )
                imported += 1
            except Exception as exc:  # noqa: BLE001 - 单条失败只跳过该条并记录原因
                # 历史上这里静默吞掉所有异常，用户只看到 skipped 计数却不知道
                # 哪些条目失败、为什么失败。
                skipped += 1
                note_error(f"{item_id}: {type(exc).__name__}: {exc}")
        if imported:
            invalidate_graph()
        return {"imported": imported, "skipped": skipped, "errors": errors}

    # ------------------------------------------------------------------
    # 文档中心与三库对账（documents/chunks 真值源 → 向量/图投影）
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

    def fact_items() -> list[Any]:
        return [item for item in the_manager().semantic.facts() if _is_fact_item(item)]

    def projected_vector_ids() -> set[str] | None:
        """Qdrant 里的 app 级 id 集合；无法枚举（存储不支持或不可达）时返回 None。"""

        list_ids = getattr(the_manager().vector_store, "list_ids", None)
        if not callable(list_ids):
            return None
        try:
            return {str(value) for value in list_ids()}
        except Exception:  # noqa: BLE001 - 读不到就跳过向量对账，不误报漂移
            return None

    def projected_edge_ids() -> set[str] | None:
        """图投影里的 memory_id 集合（内存回退与 Neo4j 都实现同一方法）。"""

        relation_ids = getattr(the_manager().graph_store, "relation_memory_ids", None)
        if not callable(relation_ids):
            return None
        try:
            return {str(value) for value in relation_ids() if str(value)}
        except Exception:  # noqa: BLE001 - 同上
            return None

    def reconcile_report() -> dict[str, Any]:
        """三库计数与漂移（只看不改）：真值源 ↔ 向量投影 ↔ 图投影。"""

        manager = the_manager()
        repository = app.state.pipeline.document_repo()
        chunk_ids = set(repository.chunk_ids()) if repository is not None else set()
        indexed = set(repository.chunk_ids(vector_status="indexed")) if repository is not None else set()
        memory_ids = {item.id for item in manager.document_store.list(include_expired=True)}
        facts = fact_items()
        vectors = projected_vector_ids()
        edges = projected_edge_ids()

        drift: list[dict[str, Any]] = []
        if vectors is not None:
            missing = sorted(indexed - vectors)
            orphan = sorted(vectors - chunk_ids - memory_ids)
            if missing:
                drift.append({"kind": "missing_vector", "count": len(missing), "ids": missing})
            if orphan:
                drift.append({"kind": "orphan_vector", "count": len(orphan), "ids": orphan})
        if edges is not None:
            missing_edges = sorted({item.id for item in facts} - edges)
            if missing_edges:
                drift.append({"kind": "missing_edge", "count": len(missing_edges), "ids": missing_edges})
        return {
            "counts": {
                "chunks": len(chunk_ids),
                "chunks_indexed_sqlite": len(indexed),
                "qdrant_points": len(vectors) if vectors is not None else -1,
                "facts": len(facts),
                "neo4j_edges": len(edges) if edges is not None else -1,
            },
            "drift": drift,
        }

    def repair_drift(kinds: list[str]) -> dict[str, Any]:
        """幂等自愈：只补缺失的投影，绝不删除或改写真值源。"""

        requested = list(dict.fromkeys(kinds or []))
        unknown = [kind for kind in requested if kind not in {"missing_vector", "missing_edge"}]
        if unknown:
            raise HTTPException(status_code=422, detail=f"不支持的修复类型：{', '.join(unknown)}")
        repository = app.state.pipeline.document_repo()
        manager = the_manager()
        entries = {entry["kind"]: entry["ids"] for entry in reconcile_report()["drift"]}
        repaired = {"missing_vector": 0, "missing_edge": 0}

        if "missing_vector" in requested and entries.get("missing_vector"):
            ids = entries["missing_vector"]
            chunks = [chunk for chunk in (repository.get_chunk(chunk_id) for chunk_id in ids) if chunk is not None]
            unavailable = embedding_unavailable_detail(manager)
            if unavailable:
                raise HTTPException(status_code=503, detail=f"补向量已中止：{unavailable}")
            if chunks:
                vectors = manager.embedding.embed_batch([chunk.text for chunk in chunks])
                for chunk, vector in zip(chunks, vectors, strict=True):
                    manager.vector_store.upsert_chunk(
                        chunk.chunk_id,
                        vector,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        source="",
                        memory_type=MemoryType.SEMANTIC.value,
                    )
                    repository.set_chunk_vector_status(chunk.chunk_id, "indexed")
                repaired["missing_vector"] = len(chunks)
                for document_id in {chunk.document_id for chunk in chunks}:
                    document = repository.get_document(document_id)
                    if document is not None and document.status == "parsed":
                        repository.set_status(document_id, "vectorized")

        if "missing_edge" in requested and entries.get("missing_edge"):
            wanted = set(entries["missing_edge"])
            for item in fact_items():
                if item.id not in wanted:
                    continue
                manager.semantic.add_fact(
                    str(item.metadata["subject"]),
                    str(item.metadata["predicate"]),
                    str(item.metadata["object"]),
                    metadata=item.metadata,
                    confidence=float(item.importance),
                    item_id=item.id,
                )
                repaired["missing_edge"] += 1

        return {"repaired": repaired}

    @app.get("/api/documents")
    def list_documents(tag: str = "", status: str = "", page: int = 1, page_size: int = 20) -> dict[str, Any]:
        repository = the_repository()
        try:
            items, total = repository.list_documents(tag=tag, status=status, page=page, page_size=page_size)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        counts = repository.chunk_counts()
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "source": item.source,
                    "tags": item.tags,
                    "status": item.status,
                    "chunk_count": counts.get(item.document_id, 0),
                    "created_at": item.created_at,
                }
                for item in items
            ],
        }

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str) -> dict[str, Any]:
        repository = the_repository()
        document = repository.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {
            "document_id": document.document_id,
            "title": document.title,
            "raw_text": document.raw_text,
            "source": document.source,
            "tags": document.tags,
            "permission": document.permission,
            "status": document.status,
            "error": document.error,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "text": chunk.text,
                    "vector_status": chunk.vector_status,
                }
                for chunk in repository.list_chunks(document_id)
            ],
        }

    @app.post("/api/documents/{document_id}/revectorize")
    def revectorize_document(document_id: str) -> dict[str, Any]:
        """重建该文档的向量投影；网关不可达时明确失败，绝不切换到别的向量空间（D8）。"""

        repository = the_repository()
        document = repository.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        chunks = repository.list_chunks(document_id)
        if not chunks:
            raise HTTPException(status_code=422, detail="该文档没有分块，无法重嵌入")
        manager = the_manager()
        unavailable = embedding_unavailable_detail(manager)
        if unavailable:
            raise HTTPException(status_code=503, detail=f"重嵌入已中止：{unavailable}")
        try:
            vectors = manager.embedding.embed_batch([chunk.text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                manager.vector_store.upsert_chunk(
                    chunk.chunk_id,
                    vector,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    source=document.source,
                    memory_type=MemoryType.SEMANTIC.value,
                )
                repository.set_chunk_vector_status(chunk.chunk_id, "indexed")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"重嵌入失败：{type(exc).__name__}: {exc}") from exc
        status = "extracted" if document.status == "extracted" else "vectorized"
        repository.set_status(document_id, status)
        return {"document_id": document_id, "chunks_reindexed": len(chunks), "status": status}

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        manager = the_manager()
        repository = app.state.pipeline.document_repo()
        counts = repository.stats() if repository is not None else {"documents": 0, "chunks": 0, "chunks_indexed": 0}
        return {
            **counts,
            "facts": len(fact_items()),
            "memories_total": len(manager.document_store.list(include_expired=True)),
        }

    @app.get("/api/reconcile")
    def reconcile() -> dict[str, Any]:
        return reconcile_report()

    @app.post("/api/reconcile")
    def reconcile_repair(body: ReconcileBody) -> dict[str, Any]:
        return repair_drift(body.repair)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        ready, _ = chat_ready()
        # 报告实际生效的嵌入实现，而不是猜某个环境变量：MemoryConfig 还支持
        # HELLOAGENTS_MEMORY_EMBEDDING_API_KEY，且调用方可注入自定义 embedding。
        embedding = getattr(the_manager(), "embedding", None)
        # APIEmbedding 与 EmbedServerEmbedding 都带 base_url，都是远端向量实现；
        # HashEmbedding 离线兜底没有 base_url。用属性而非类名判断，避免重复导入。
        base_url = getattr(embedding, "base_url", None)
        embedding_mode = "api" if base_url else "local-hash"
        # 只上报隧道可达性，供 UI/调用方判断是否已降级为关键词检索（D8）；
        # 绝不因为不可达就换向量空间。
        embedding_reachable = gateway_reachable(base_url) if base_url else True
        manager = the_manager()
        if not base_url:
            embedding_state = "hash"
        elif isinstance(embedding, EmbedServerEmbedding):
            embedding_state = "gateway"
        else:
            embedding_state = "api"
        if base_url and not embedding_reachable:
            embedding_state = "unreachable"
        return {
            "ok": True,
            "chat_ready": ready,
            "embedding_mode": embedding_mode,
            "embedding_reachable": embedding_reachable,
            "embedding": getattr(embedding, "to_dict", dict)(),
            "search_available": search_available(),
            # 三个存储各自的实现（D1：全部在本机），UI 用它区分「本地真值 / 本地投影」。
            "store_modes": {
                "document": "memory" if str(getattr(manager.document_store, "path", "")) == ":memory:" else "sqlite",
                "vector": type(manager.vector_store).__name__,
                "graph": "neo4j" if getattr(manager.graph_store, "driver", None) is not None else "inmemory",
            },
            # 两个出网点的实时状态（D8/D9）：嵌入不可达时检索只能走 FTS5，UI 必须如实提示。
            "degraded": {
                "embedding": embedding_state,
                "embedding_endpoint": base_url or "",
                "chat_ready": ready,
                "keyword_fallback": bool(base_url) and not embedding_reachable,
                # 隧道命令由部署方通过 EMBEDDING_TUNNEL_HINT 下发，前端只负责显示。
                "embedding_hint": embedding_tunnel_hint(),
            },
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

    uvicorn.run(app, host=LOCALHOST, port=int(os.getenv("NEBULA_PORT", str(DEFAULT_WEB_PORT))))
