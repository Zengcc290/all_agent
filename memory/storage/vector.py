"""Local and remote vector indexes for semantic memory search."""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod

from ..base import MemoryItem, MemoryType


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def delete(self, item_id: str) -> bool: ...

    @abstractmethod
    def search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]: ...

    def clear(self) -> None:
        pass


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    if any(not math.isfinite(float(value)) for value in (*left, *right)):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class InMemoryVectorStore(BaseVectorStore):
    """Fast local vector index, useful as the default and for unit tests."""

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], MemoryType]] = {}
        self._lock = threading.RLock()

    def upsert(self, item: MemoryItem) -> None:
        if item.embedding is not None:
            if not item.embedding or any(not math.isfinite(float(value)) for value in item.embedding):
                raise ValueError("item embedding must contain finite values")
            with self._lock:
                self._vectors[item.id] = (list(item.embedding), item.memory_type)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            return self._vectors.pop(item_id, None) is not None

    def search(self, vector: list[float], *, limit: int = 10, memory_type: MemoryType | str | None = None) -> list[tuple[str, float]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        wanted = MemoryType(memory_type) if memory_type is not None else None
        with self._lock:
            scores = [
                (item_id, cosine_similarity(vector, stored), stored_type)
                for item_id, (stored, stored_type) in self._vectors.items()
                if wanted is None or stored_type == wanted
            ]
        scores.sort(key=lambda row: (-row[1], row[0]))
        return [(item_id, score) for item_id, score, _ in scores[:limit]]

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()


__all__ = ["BaseVectorStore", "InMemoryVectorStore", "cosine_similarity"]
