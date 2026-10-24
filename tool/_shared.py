"""Shared path-containment helpers for the filesystem-facing tools.

This module is deliberately NOT discoverable: ``core.discovery`` ignores
modules whose names start with an underscore, and this file defines neither
``TOOL_ENABLED`` nor a ``create_tool()`` factory. Every helper here is pure
standard-library code with no network, file-write or process-spawn side effect
at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT_ENV = "WORKSPACE_ROOT"
ALLOW_OUTSIDE_ENV = "WORKSPACE_ALLOW_OUTSIDE"


def workspace_root() -> Path:
    """Return the configured workspace root (defaults to the process CWD)."""
    raw = os.getenv(WORKSPACE_ROOT_ENV)
    if raw is not None and raw.strip():
        return Path(raw.strip()).expanduser().resolve()
    return Path.cwd().resolve()


def allow_outside_workspace() -> bool:
    """Return whether tools may touch paths outside the workspace root."""
    raw = os.getenv(ALLOW_OUTSIDE_ENV)
    if raw is None:
        return False
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def resolve_path(
    base_dir: str | Path,
    path: str,
    *,
    allow_outside: bool | None = None,
) -> Path:
    """Resolve ``path`` against ``base_dir`` and enforce workspace containment.

    Relative paths are anchored at ``base_dir``. Absolute paths are allowed
    only when they stay inside the resolved ``base_dir`` unless
    ``allow_outside`` (or the ``WORKSPACE_ALLOW_OUTSIDE`` environment switch)
    permits full access. ``allow_outside`` is a deployment decision made by the
    tool owner, never by the LLM through an input field.
    """
    if not isinstance(base_dir, (str, Path)):
        raise TypeError("base_dir must be a string or pathlib.Path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    base = Path(base_dir).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if allow_outside is None:
        allow_outside = allow_outside_workspace()
    if not allow_outside and not _is_within(base, resolved):
        raise ValueError(
            f"path '{path}' resolves outside the workspace root '{base}'"
        )
    return resolved


def _is_within(base: Path, candidate: Path) -> bool:
    base_text = os.path.normcase(str(base))
    candidate_text = os.path.normcase(str(candidate))
    return candidate_text == base_text or candidate_text.startswith(
        base_text + os.sep
    )
