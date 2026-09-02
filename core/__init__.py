"""Core contracts and runtime for agent tools."""

from .models import (
    BatchToolResult,
    ExecutionContext,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from .catalog import CatalogInput, CatalogOutput, ToolCatalogTool
from .registry import BaseTool, ToolRegistry
from .runtime import ToolExecutionManager
from .parser import parse_openai_tool_calls, parse_tool_calls
from .repository import ToolSpecRepository

__all__ = [
    "BaseTool",
    "CatalogInput",
    "CatalogOutput",
    "BatchToolResult",
    "ExecutionContext",
    "ToolCall",
    "ToolCatalogTool",
    "ToolError",
    "ToolExecutionManager",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolSpecRepository",
    "parse_openai_tool_calls",
    "parse_tool_calls",
]
