"""一句话后台入库队列：SQLite 持久化状态 + 线程池并发消费。

提交立即返回（前端即刻清空输入框），worker 线程在后台完成向量化与
LLM 抽取。每条任务的 ``pending → running → done/failed`` 状态写在
memories 同一个 SQLite 文件里（``ingest_jobs`` 表），随时可查历史；
进程异常退出后 :meth:`IngestJobQueue.start` 把遗留 ``running`` 任务复位
为 ``pending`` 并重新入队（``attempts`` 封顶防死循环）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from constants import KNOWLEDGE_INGEST_WORKERS
from memory.rag import Document, RAGPipeline
from memory.storage.document_repo import DocumentRepository, IngestJobRecord

LOGGER = logging.getLogger(__name__)

#: 单条任务跨重启的最大尝试次数（含首次）。
MAX_ATTEMPTS = 3

#: 状态 → 展示文案（API/前端共用一份口径）。
JOB_LABELS = {
    "pending": "排队中",
    "running": "正在入库",
    "done": "入库成功",
    "failed": "入库失败",
}


def job_to_dict(job: IngestJobRecord, *, text_preview: int = 120) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if job.result:
        try:
            decoded = json.loads(job.result)
            if isinstance(decoded, dict):
                result = decoded
        except ValueError:
            result = {}
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "text": job.text[:text_preview],
        "status": job.status,
        "label": JOB_LABELS.get(job.status, job.status),
        "attempts": job.attempts,
        "error": job.error,
        "result": result,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


class IngestJobQueue:
    """后台一句话入库；``available=False``（如 ``:memory:`` 测试库）时不可用。"""

    def __init__(
        self,
        manager: Any,
        extractor: Any,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._manager = manager
        self._extractor = extractor
        self._on_progress = on_progress
        path = getattr(manager.document_store, "path", None)
        usable = bool(path) and str(path) != ":memory:"
        # 文件库走 DocumentRepository（每次调用独立连接，线程安全）。
        self._repo = DocumentRepository(path) if usable else None
        # 并发 worker 数唯一来源：constants.KNOWLEDGE_INGEST_WORKERS（I/O 为主，
        # 2-3 即可吃满 LLM 延迟；历史 KNOWLEDGE_INGEST_WORKERS 环境变量已删）。
        self._workers = max(1, KNOWLEDGE_INGEST_WORKERS)
        self._executor: ThreadPoolExecutor | None = None

    @property
    def available(self) -> bool:
        return self._repo is not None

    def start(self) -> None:
        """启动 worker 池，并续跑上次退出时未完成的任务。"""

        if self._repo is None:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="ingest-job"
        )
        recovered = self._repo.restart_stale_ingest_jobs()
        if recovered:
            LOGGER.info("入库队列恢复：%d 条上次未完成的任务已重新排队", recovered)
        for job in self._repo.list_ingest_jobs(status="pending"):
            self._enqueue(job.job_id)

    def submit(self, text: str, *, event_at: str = "") -> IngestJobRecord:
        """落库一条 ``pending`` 任务并立即入队；返回记录供 API 即刻回显。"""

        if self._repo is None:
            raise RuntimeError("ingest queue unavailable (no persistent sqlite store)")
        job = self._repo.create_ingest_job(text, event_at=event_at)
        self._enqueue(job.job_id)
        return job

    def job(self, job_id: str) -> IngestJobRecord | None:
        return None if self._repo is None else self._repo.get_ingest_job(job_id)

    def list(self, *, status: str | None = None, limit: int = 50) -> list[IngestJobRecord]:
        return [] if self._repo is None else self._repo.list_ingest_jobs(status=status, limit=limit)

    def wait(self, job_id: str, timeout: float = 180.0) -> IngestJobRecord | None:
        """阻塞到任务终态（done/failed）或超时；给 ``wait=true`` 的同步调用方用。"""

        import time

        deadline = time.monotonic() + timeout
        job = self.job(job_id)
        while job is not None and job.status not in ("done", "failed") and time.monotonic() < deadline:
            time.sleep(0.05)
            job = self.job(job_id)
        return job

    def shutdown(self) -> None:
        """不再等待在跑的任务：它们在 SQLite 里保持 ``running``，下次启动续跑。"""

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._repo is not None:
            self._repo.close()
            self._repo = None

    # -- internals ------------------------------------------------------
    def _enqueue(self, job_id: str) -> None:
        if self._executor is not None:
            self._executor.submit(self._run_job, job_id)

    def _run_job(self, job_id: str) -> None:
        repo = self._repo
        if repo is None:
            return
        job = repo.get_ingest_job(job_id)
        if job is None or job.status == "done":
            return
        if job.attempts >= MAX_ATTEMPTS:
            repo.set_ingest_job_status(job_id, "failed", error="重试次数超限")
            self._notify(job_id)
            return
        repo.set_ingest_job_status(job_id, "running")
        try:
            # 每任务独立 pipeline：last_ingest_report 不能被并发任务互相覆盖。
            pipeline = RAGPipeline(self._manager, extractor=self._extractor)
            items = pipeline.ingest(
                Document(
                    job.text,
                    metadata={
                        "source": "一句话入库",
                        "filename": "一句话入库",
                        "note": job.text[:400],
                        "event_at": job.event_at,
                        "reference_time": datetime.now(UTC).isoformat(),
                        "ingest_job_id": job_id,
                    },
                )
            )
            report = pipeline.last_ingest_report
            self._manager.episodic.record(
                f"添加了一条知识：{job.text[:80]}",
                metadata={"title": "一句话入库", "source": "一句话入库", "ingest_job_id": job_id},
            )
            summary = json.dumps(
                {"chunks": len(items), "report": report}, ensure_ascii=False
            )
            repo.set_ingest_job_status(job_id, "done", result=summary)
        except Exception as exc:
            LOGGER.exception("入库任务 %s 失败", job_id)
            repo.set_ingest_job_status(job_id, "failed", error=f"{type(exc).__name__}: {exc}")
        self._notify(job_id)

    def _notify(self, job_id: str) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(job_id)
        except Exception:
            LOGGER.warning("入库进度回调失败", exc_info=True)


__all__ = [
    "JOB_LABELS",
    "MAX_ATTEMPTS",
    "IngestJobQueue",
    "job_to_dict",
]
