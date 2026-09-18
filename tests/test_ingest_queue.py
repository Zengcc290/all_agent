"""一句话后台入库队列：异步提交、SQLite 状态持久化、重启续跑与并发。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from conftest import HashEmbedding
from fastapi.testclient import TestClient

from memory import MemoryConfig, MemoryManager, MemoryType
from memory.rag import NullKnowledgeExtractor
from memory.storage.document_repo import DocumentRepository
from web import create_app
from web.ingest_queue import MAX_ATTEMPTS, IngestJobQueue


def make_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=str(tmp_path / "memory.sqlite3")),
        embedding=HashEmbedding(),
    )


def wait_for(predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached within timeout")


def test_async_submit_completes_and_updates_status(tmp_path):
    manager = make_manager(tmp_path)
    queue = IngestJobQueue(manager, NullKnowledgeExtractor())
    assert queue.available
    queue.start()
    try:
        job = queue.submit("电脑 13:00 在书桌上跑项目。")
        assert job.status == "pending"

        finished = queue.wait(job.job_id, timeout=10)

        assert finished is not None and finished.status == "done"
        result = json.loads(finished.result)
        assert result["chunks"] >= 1
        assert isinstance(result["report"], dict)
        # 文档真的落库，事件记忆也记录了一条。
        repo = DocumentRepository(manager.document_store.path)
        docs, _total = repo.list_documents()
        assert any(doc.source == "一句话入库" for doc in docs)
        assert any(
            "添加了一条知识" in item.content
            for item in manager.list(memory_type=MemoryType.EPISODIC)
        )
    finally:
        queue.shutdown()
        manager.close()


def test_crash_recovery_reruns_stale_running_jobs(tmp_path):
    manager = make_manager(tmp_path)
    repo = DocumentRepository(manager.document_store.path)
    # 模拟进程在入库中途退出：任务停在 running，从未完成。
    stale = repo.create_ingest_job("电脑 14:00 在图书馆里。")
    repo.set_ingest_job_status(stale.job_id, "running")

    queue = IngestJobQueue(manager, NullKnowledgeExtractor())
    queue.start()
    try:
        finished = queue.wait(stale.job_id, timeout=10)
        assert finished is not None and finished.status == "done"
        assert finished.attempts == 2  # 崩溃前 1 次 + 续跑 1 次
    finally:
        queue.shutdown()
        manager.close()


def test_retry_cap_marks_job_failed(tmp_path):
    manager = make_manager(tmp_path)
    repo = DocumentRepository(manager.document_store.path)
    # 模拟连续多次崩溃重启：每次都 start 运行、崩回 pending，尝试次数耗尽。
    exhausted = repo.create_ingest_job("反复失败的一句话。")
    for _ in range(MAX_ATTEMPTS):
        repo.set_ingest_job_status(exhausted.job_id, "running")
        repo.set_ingest_job_status(exhausted.job_id, "pending")
    assert repo.get_ingest_job(exhausted.job_id).attempts == MAX_ATTEMPTS

    queue = IngestJobQueue(manager, NullKnowledgeExtractor())
    queue.start()
    try:
        wait_for(lambda: "重试次数超限" in repo.get_ingest_job(exhausted.job_id).error)
        job = repo.get_ingest_job(exhausted.job_id)
        assert job.status == "failed"
        assert job.attempts == MAX_ATTEMPTS
    finally:
        queue.shutdown()
        manager.close()


def test_knowledge_endpoint_async_flow_and_history(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    app = create_app(manager)
    with TestClient(app) as client:
        response = client.post(
            "/api/knowledge",
            json={"text": "向量数据库把文本编码成稠密向量，用于语义检索。", "wait": False},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["async"] is True
        assert payload["job_id"].startswith("job_")

        def done() -> bool:
            items = client.get("/api/knowledge/jobs").json()["items"]
            return bool(items) and items[0]["status"] == "done"

        wait_for(done)
        history = client.get("/api/knowledge/jobs").json()
        assert history["available"] is True
        job = history["items"][0]
        assert job["label"] == "入库成功"
        assert job["result"]["chunks"] >= 1

        # wait=true（默认）走同一条队列但同步等待，保持旧响应契约。
        sync = client.post("/api/knowledge", json={"text": "Qdrant 支持混合检索。"})
        assert sync.status_code == 200
        assert sync.json()["chunks"] >= 1
        assert isinstance(sync.json()["extraction"], dict)
    manager.close()


def test_queue_unavailable_on_memory_store(tmp_path):
    manager = MemoryManager(MemoryConfig(sqlite_path=":memory:"), embedding=HashEmbedding())
    queue = IngestJobQueue(manager, NullKnowledgeExtractor())
    try:
        assert not queue.available
        assert queue.list() == []
    finally:
        queue.shutdown()
        manager.close()


def test_multiple_jobs_complete_concurrently(tmp_path):
    manager = make_manager(tmp_path)
    queue = IngestJobQueue(manager, NullKnowledgeExtractor())
    queue.start()
    try:
        jobs = [queue.submit(f"第 {i} 条知识：并发入库测试。") for i in range(3)]
        for job in jobs:
            finished = queue.wait(job.job_id, timeout=15)
            assert finished is not None and finished.status == "done"
        statuses = [queue.job(job.job_id).status for job in jobs]
        assert statuses == ["done", "done", "done"]
    finally:
        queue.shutdown()
        manager.close()
