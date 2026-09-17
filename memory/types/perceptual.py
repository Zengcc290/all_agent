"""Perceptual memory for multimodal payloads."""

from __future__ import annotations

from ..base import BaseMemory, MemoryType


class PerceptualMemory(BaseMemory):
    memory_type = MemoryType.PERCEPTUAL


__all__ = ["PerceptualMemory"]
