"""知识星云 · FastAPI 应用。

路由一览：
- GET  /api/graph    全图 nodes+edges（星云图数据源）
- POST /api/chat     与知识管家对话（未配置聊天模型时 503）
- POST /api/ingest   上传文档 → RAG 切块入库（星云长出新星星）
- POST /api/facts    手工添加三元组知识
- POST /api/seed     （重新）播种 Aetheria 种子数据（幂等）
- GET  /api/export   导出全部记忆为 JSON 文件（课设「库→文件」要求）
- POST /api/import   导入此前导出的 JSON（课设「文件→库」要求）
- /                星云图前端静态页（web/static/index.html）

运行：``python -m web.app``（默认 http://127.0.0.1:8765）
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memory import MemoryManager, MemoryType
from memory.base import utc_now
from memory.rag import RAGPipeline

from .graph_builder import build_graph
from .seed import seed
from .support import chat_ready, close_manager, get_agent, get_manager, STATIC_DIR


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class FactBody(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    object: str = Field(min_length=1, max_length=200)
    domain: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=4000)
    confidence: float = Field(default=1.0, ge=0, le=1)


def create_app(manager: MemoryManager | None = None) -> FastAPI:
    """应用工厂。``manager`` 可注入（测试用内存库）；默认用共享单例。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_manager = manager is None
        app.state.manager = manager if manager is not None else get_manager()
        app.state.pipeline = RAGPipeline(app.state.manager)
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
    # 星云图数据
    # ------------------------------------------------------------------
    @app.get("/api/graph")
    def graph() -> dict[str, Any]:
        return build_graph(the_manager())

    # ------------------------------------------------------------------
    # 聊天（得力助手）
    # ------------------------------------------------------------------
    @app.post("/api/chat")
    async def chat(body: ChatBody) -> dict[str, Any]:
        ready, reason = chat_ready()
        if not ready:
            raise HTTPException(status_code=503, detail=reason)
        agent = get_agent()
        try:
            # agent.run 是同步阻塞调用，丢进线程避免卡住事件循环。
            answer = await asyncio.to_thread(agent.run, body.message)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"聊天模型调用失败：{type(exc).__name__}: {exc}")
        return {"answer": answer}

    # ------------------------------------------------------------------
    # 文档导入（RAG 摄取）
    # ------------------------------------------------------------------
    @app.post("/api/ingest")
    async def ingest(file: UploadFile) -> dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传的文件是空的")
        filename = file.filename or "untitled"
        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="nebula-ingest-") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            items = app.state.pipeline.ingest_source(
                tmp_path,
                metadata={"source": filename, "filename": filename},
                chunk_size=800,
                overlap=100,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"文档解析失败：{type(exc).__name__}: {exc}")
        finally:
            os.unlink(tmp_path)
        the_manager().episodic.record(
            f"上传并导入了文档《{filename}》（{len(items)} 个知识块）",
            metadata={"title": "导入文档", "filename": filename},
        )
        return {"filename": filename, "chunks": len(items)}

    # ------------------------------------------------------------------
    # 手工添加三元组
    # ------------------------------------------------------------------
    @app.post("/api/facts")
    def add_fact(body: FactBody) -> dict[str, Any]:
        item = the_manager().semantic.add_fact(
            body.subject,
            body.predicate,
            body.object,
            metadata={"domain": body.domain or "未分类", "note": body.note or ""},
            confidence=body.confidence,
        )
        return {"ok": True, "item_id": item.id}

    # ------------------------------------------------------------------
    # 播种 / 导出 / 导入（课设硬性要求）
    # ------------------------------------------------------------------
    @app.post("/api/seed")
    def reseed() -> dict[str, Any]:
        return seed(the_manager())

    @app.get("/api/export")
    def export(request: Request) -> JSONResponse:
        manager = the_manager()
        items = [item.to_dict() for item in manager.list(include_expired=False)]
        payload = {
            "format": "knowledge-nebula-export/v1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "total": len(items),
                "semantic": sum(1 for item in items if item["memory_type"] == "semantic"),
                "episodic": sum(1 for item in items if item["memory_type"] == "episodic"),
            },
            "items": items,
        }
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return JSONResponse(
            payload,
            headers={"Content-Disposition": f'attachment; filename="knowledge_export_{stamp}.json"'},
        )

    @app.post("/api/import")
    async def import_file(file: UploadFile) -> dict[str, Any]:
        raw = await file.read()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"不是合法的 JSON：{exc}")
        entries = data.get("items") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise HTTPException(status_code=400, detail="JSON 中找不到 items 数组")

        manager = the_manager()
        existing_facts = {
            (item.metadata.get("subject"), item.metadata.get("predicate"), item.metadata.get("object"))
            for item in manager.list(memory_type=MemoryType.SEMANTIC)
            if item.metadata.get("subject") and item.metadata.get("predicate") and item.metadata.get("object")
        }
        imported = skipped = 0
        for raw_item in entries:
            if not isinstance(raw_item, dict):
                continue
            item_id = raw_item.get("id")
            content = raw_item.get("content") or ""
            if not item_id or not content:
                skipped += 1
                continue
            if manager.get(item_id) is not None:
                skipped += 1
                continue
            md = raw_item.get("metadata") or {}
            memory_type = raw_item.get("memory_type") or "semantic"
            importance = raw_item.get("importance", 0.5)
            subject, predicate, obj = md.get("subject"), md.get("predicate"), md.get("object")
            try:
                if subject and predicate and obj:
                    # 事实：按三元组幂等，避免重复导入时长出重边。
                    if (subject, predicate, obj) in existing_facts:
                        skipped += 1
                        continue
                    existing_facts.add((subject, predicate, obj))
                    manager.semantic.add_fact(
                        subject, predicate, obj, metadata=md, confidence=float(importance),
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
            except Exception:
                skipped += 1
        return {"imported": imported, "skipped": skipped}

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        ready, _ = chat_ready()
        return {
            "ok": True,
            "chat_ready": ready,
            "embedding_mode": "api" if os.getenv("DASHSCOPE_API_KEY") else "local-hash",
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

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("NEBULA_PORT", "8765")))
