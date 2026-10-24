"""轻量文本工具：实体 key 归一化等。"""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)


def norm_key(s: str) -> str:
    """把实体名归一化成稳定 key：统一小写、去掉一切非字母数字汉字的字符。
    保证「糖尿病」和「 糖尿病 」映射到同一个节点，实现实体复用。"""
    s = (s or "").strip().lower()
    s = _NON_ALNUM.sub("", s)
    return s[:128] or "unknown"


def is_empty(s: str | None) -> bool:
    return not (s or "").strip()
