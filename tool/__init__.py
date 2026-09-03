"""Tool plugin package.

Concrete plugins are imported lazily so one broken implementation cannot make
the package itself unimportable before the discovery scanner can isolate it.
"""

from importlib import import_module
from typing import Any

from .base import BaseTool, ToolRegistry

_SEARCH_EXPORTS = frozenset({"SearchInput", "SearchItem", "SearchOutput", "SearchTool"})

__all__ = [
    "BaseTool",
    "SearchInput",
    "SearchItem",
    "SearchOutput",
    "SearchTool",
    "ToolRegistry",
]


def __getattr__(name: str) -> Any:
    if name not in _SEARCH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".search", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SEARCH_EXPORTS)
