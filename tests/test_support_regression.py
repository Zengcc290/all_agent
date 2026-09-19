"""Web 支撑设施回归：进程级单例的初始化路径不得死锁。

历史版本 ``get_pipeline()`` 在持有 ``_manager_lock`` 时再次调用
``get_manager()``（同一把非可重入锁），首次调用直接永久挂起。
"""

from __future__ import annotations

import threading

import pytest

from web import support


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support, "_manager", None)
    monkeypatch.setattr(support, "_pipeline", None)
    yield
    # 单例关闭顺序固定：先 pipeline（引用 manager），再 manager。
    support._pipeline = None
    monkeypatch.setattr(support, "_manager", None)
    close = getattr(support, "close_manager", None)
    if callable(close):
        close()


def test_get_pipeline_does_not_deadlock_on_first_call() -> None:
    result: dict[str, object] = {}

    def build() -> None:
        try:
            pipeline = support.get_pipeline()
            result["pipeline"] = pipeline is not None
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion below
            result["error"] = repr(exc)

    thread = threading.Thread(target=build, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive(), "get_pipeline() 首次调用必须能在超时内返回"
    assert "error" not in result, result["error"]
    assert result.get("pipeline") is True
    assert support.get_pipeline() is support.get_pipeline(), "进程级单例应保持同一实例"
