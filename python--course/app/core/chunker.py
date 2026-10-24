"""分块器。

当前阶段输入的一句话不会太长，因此分块保留空实现：
传入原文档，原样返回一段，chunk_id 由上层生成。
"""
from __future__ import annotations

from typing import Any


def split_text(text: str, max_chars: int = 2000, **_: Any) -> list[str]:
    """预留的分块接口。现在直接透传（保留空实现）。"""
    if text is None:
        return []
    return [text]


def chunk_count(text: str) -> int:
    return 1 if (text or "").strip() else 0


def chunk_info(text: str) -> dict:
    t = text or ""
    return {
        "strategy": "passthrough",
        "input_len": len(t),
        "block_count": 1 if t.strip() else 0,
        "note": "当前阶段保留空实现：传入原文档，返回原文档",
    }
