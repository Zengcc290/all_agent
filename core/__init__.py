"""Core contracts and runtime for agent tools."""

from .catalog import CatalogInput, CatalogOutput, ToolCatalogTool
from .discovery import (
    ToolDiscoveryError,
    ToolDiscoveryRecord,
    ToolDiscoveryReport,
    discover_tools,
)
from .models import (
    BatchToolResult,
    ExecutionContext,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from .parser import parse_openai_tool_calls, parse_tool_calls
from .registry import BaseTool, ToolRegistry
from .repository import ToolSpecRepository
from .runtime import ToolExecutionManager
from .tool_loop import ToolLoop
from .update_log import DEFAULT_UPDATE_LOG_FILENAME, UpdateLogRepository

__all__ = [
    "BaseTool",
    "BatchToolResult",
    "CatalogInput",
    "CatalogOutput",
    "ExecutionContext",
    "ToolCall",
    "ToolCatalogTool",
    "ToolDiscoveryError",
    "ToolDiscoveryRecord",
    "ToolDiscoveryReport",
    "ToolError",
    "ToolExecutionManager",
    "ToolLoop",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolSpecRepository",
    "DEFAULT_UPDATE_LOG_FILENAME",
    "UpdateLogRepository",
    "discover_tools",
    "parse_openai_tool_calls",
    "parse_tool_calls",
]
