"""一句话后台入库队列：异步提交、SQLite 状态持久化、重启续跑与串行消费。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from conftest import HashEmbedding
from fastapi.testclient import TestClient

from memory import MemoryConfig, MemoryManager, MemoryType
from memory.rag import NullKnowledgeExtractor
from memory.storage.document_repo import DocumentRepository
from web import create_app
from web.ingest_queue import MAX_ATTEMPTS, IngestJobQueue, job_to_dict


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
        assert "workers" not in history  # 旧字段从未连接真实 worker 数，已移除
        job = history["items"][0]
        assert job["label"] == "入库成功"
        assert job["result"]["chunks"] >= 1

        # wait=true（默认）走同一条队列但同步等待，保持旧响应契约。
        sync = client.post("/api/knowledge", json={"text": "Qdrant 支持混合检索。"})
        assert sync.status_code == 200
        assert sync.json()["chunks"] >= 1
        assert isinstance(sync.json()["extraction"], dict)
    manager.close()


def test_user_retry_resets_failed_job_and_succeeds(tmp_path):
    manager = make_manager(tmp_path)
    repo = DocumentRepository(manager.document_store.path)
    failed = repo.create_ingest_job("失败后可点重试。")
    repo.set_ingest_job_status(failed.job_id, "running")
    repo.set_ingest_job_status(failed.job_id, "failed", error="ValueError: boom")
    done = repo.create_ingest_job("已经成功的任务。")
    repo.set_ingest_job_status(done.job_id, "done", result="{}")

    queue = IngestJobQueue(manager, NullKnowledgeExtractor())
    queue.start()
    try:
        with pytest.raises(LookupError):
            queue.retry("job_missing")
        with pytest.raises(ValueError, match="只有失败任务可重试"):
            queue.retry(done.job_id)

        reset = queue.retry(failed.job_id)
        assert reset.status == "pending"
        assert reset.attempts == 0
        assert reset.error == ""
        assert job_to_dict(reset)["retryable"] is False

        finished = queue.wait(failed.job_id, timeout=10)
        assert finished is not None and finished.status == "done"
        assert job_to_dict(finished)["retryable"] is False
    finally:
        queue.shutdown()
        manager.close()


def test_retry_endpoint_requeues_failed_job(tmp_path):
    manager = make_manager(tmp_path)
    app = create_app(manager)
    with TestClient(app) as client:
        repo = DocumentRepository(manager.document_store.path)
        job = repo.create_ingest_job("点按钮重试这条。")
        repo.set_ingest_job_status(job.job_id, "failed", error="boom")
        listed = {item["job_id"]: item for item in client.get("/api/knowledge/jobs").json()["items"]}
        assert listed[job.job_id]["retryable"] is True
        assert listed[job.job_id]["label"] == "入库失败"

        missing = client.post("/api/knowledge/jobs/job_missing/retry")
        assert missing.status_code == 404

        done = repo.create_ingest_job("成功过的不能重试。")
        repo.set_ingest_job_status(done.job_id, "done", result="{}")
        refused = client.post(f"/api/knowledge/jobs/{done.job_id}/retry")
        assert refused.status_code == 400

        retried = client.post(f"/api/knowledge/jobs/{job.job_id}/retry")
        assert retried.status_code == 200, retried.text
        payload = retried.json()
        assert payload["ok"] is True
        assert payload["async"] is True
        assert payload["status"] == "pending"
        assert payload["retryable"] is False

        def done_again() -> bool:
            items = {item["job_id"]: item for item in client.get("/api/knowledge/jobs").json()["items"]}
            return items[job.job_id]["status"] == "done"

        wait_for(done_again)
        history = {item["job_id"]: item for item in client.get("/api/knowledge/jobs").json()["items"]}
        assert history[job.job_id]["label"] == "入库成功"
        assert history[job.job_id]["retryable"] is False
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


def test_multiple_jobs_complete_in_order(tmp_path):
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

def test_extraction_error_marks_job_failed_and_retryable(tmp_path):
    class BoomExtractor:
        def extract(self, text, *, metadata=None, graph_context=""):
            raise RuntimeError("routing boom")

    manager = make_manager(tmp_path)
    queue = IngestJobQueue(manager, BoomExtractor())
    queue.start()
    try:
        job = queue.submit("小猫和小狗是亲兄弟")
        finished = queue.wait(job.job_id, timeout=10)
        assert finished is not None and finished.status == "failed"
        assert "routing boom" in (finished.error or "")
        assert job_to_dict(finished)["retryable"] is True
    finally:
        queue.shutdown()
        manager.close()
