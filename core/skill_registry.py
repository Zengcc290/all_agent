"""Thread-safe registry for discovered skills.

Unlike :class:`~core.registry.ToolRegistry`, this registry stores immutable
metadata specs only. It intentionally does not keep content strings: content is
read from disk by the catalog tool so a ``view`` call always reflects the
current on-disk skill file.
"""

from __future__ import annotations

import threading
from typing import Any

from .skill_models import SkillSpec


class SkillRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, SkillSpec] = {}
        self._generations: dict[str, int] = {}
        self._lock = threading.RLock()

    def register(self, spec: SkillSpec, *, replace: bool = False) -> int:
        """Register one skill spec and return its generation.

        Re-registering the exact same spec is idempotent: the generation is not
        bumped, so prompt-cache key material stays stable across restarts or
        repeated discovery runs.
        """

        if not isinstance(spec, SkillSpec):
            raise TypeError("spec must be a SkillSpec instance")
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean")
        with self._lock:
            current = self._specs.get(spec.name)
            if current is not None:
                if current == spec:
                    return self._generations[spec.name]
                if not replace:
                    raise ValueError(
                        f"skill '{spec.name}' has a different active spec; "
                        "pass replace=True to replace it"
                    )
            self._specs[spec.name] = spec
            if current is None or current != spec:
                self._generations[spec.name] = self._generations.get(spec.name, 0) + 1
            return self._generations[spec.name]

    def resolve(self, name: str) -> tuple[SkillSpec, int]:
        """Atomically return a spec and its current generation."""

        if not isinstance(name, str) or not name:
            raise ValueError("skill name must be a non-empty string")
        with self._lock:
            try:
                return self._specs[name], self._generations[name]
            except KeyError as exc:
                raise KeyError(f"skill '{name}' is not registered") from exc

    def maybe_resolve(self, name: str) -> tuple[SkillSpec, int] | None:
        if not isinstance(name, str) or not name:
            return None
        with self._lock:
            spec = self._specs.get(name)
            if spec is None:
                return None
            return spec, self._generations[name]

    def get(self, name: str) -> SkillSpec:
        return self.resolve(name)[0]

    def maybe_get(self, name: str) -> SkillSpec | None:
        registration = self.maybe_resolve(name)
        return registration[0] if registration is not None else None

    def snapshot(self) -> dict[str, tuple[SkillSpec, int]]:
        """Capture a stable name-to-spec view, ordered by skill name."""

        with self._lock:
            return {
                name: (self._specs[name], self._generations[name])
                for name in sorted(self._specs)
            }

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._specs

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)
