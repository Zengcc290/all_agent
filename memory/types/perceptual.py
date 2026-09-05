"""Perceptual memory for multimodal payloads."""

from __future__ import annotations

from typing import Any, Mapping

from ..base import BaseMemory, MemoryItem, MemoryType


class PerceptualMemory(BaseMemory):
    memory_type = MemoryType.PERCEPTUAL

    def store(self, data: Any, *, modality: str, content: str | None = None, metadata: Mapping[str, Any] | None = None, importance: float = 0.5, item_id: str | None = None) -> MemoryItem:
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError("modality must be a non-empty string")
        text = content if content is not None else (data if isinstance(data, str) else f"{modality} perceptual memory")
        return self.add(str(text), metadata=metadata, importance=importance, payload=data, modality=modality, item_id=item_id)

    def get_data(self, item_id: str, default: Any = None) -> Any:
        item = self.get(item_id)
        return item.payload if item is not None else default

    remember = store


__all__ = ["PerceptualMemory"]
