"""知识库规模统计：库里有文档、事实、记忆各多少。

这是从 ``web/app.py`` 的 ``/api/stats`` 端点里搬出来的**只读聚合能力**：
端点仍然存在（前端仪表盘要用），但它现在只是调用本工具并把结果转成字典。
搬出来的意义与其它工具一致——同一份统计逻辑既能被 HTTP 端点用，也能被 LLM
直接调用（回答「我库里现在有多少内容」时不必先列出全部记忆再数）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core import BaseTool, ToolSpec
from memory import MemoryManager
from tool.hybrid_index import repository_for
from tool.reconcile import fact_items

TOOL_ENABLED = True


class KnowledgeStatsInput(BaseModel):
    """该工具不需要调用方提供参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class KnowledgeStatsOutput(BaseModel):
    """知识库规模统计；只返回计数，不返回任何内容正文。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    documents: int = Field(ge=0, description="真值源里的文档数。")
    chunks: int = Field(ge=0, description="真值源里的分块总数。")
    chunks_indexed: int = Field(ge=0, description="已标记为 indexed（向量投影就绪）的分块数。")
    facts: int = Field(ge=0, description="语义记忆里的事实条数（不含备注行）。")
    memories_total: int = Field(
        ge=0,
        description="四层记忆的条目总数，包含已过期条目（用于展示真实存量）。",
    )


def knowledge_stats(manager: Any, repository: Any = None) -> KnowledgeStatsOutput:
    """Aggregate the knowledge base size across the truth source and memory layers.

    ``repository`` 为 ``None``（非 SQLite 文档库或内存库）时，文档与分块计数按 0 返回，
    与重构前 ``/api/stats`` 的行为逐字一致。
    """

    counts = (
        repository.stats()
        if repository is not None
        else {"documents": 0, "chunks": 0, "chunks_indexed": 0}
    )
    return KnowledgeStatsOutput(
        documents=int(counts.get("documents", 0)),
        chunks=int(counts.get("chunks", 0)),
        chunks_indexed=int(counts.get("chunks_indexed", 0)),
        facts=len(fact_items(manager)),
        memories_total=len(manager.document_store.list(include_expired=True)),
    )

class KnowledgeStatsTool(BaseTool):
    """只读：统计知识库规模。"""

    spec = ToolSpec(
        name="knowledge.stats",
        description=(
            "Report how much is stored: document/chunk counts from the truth "
            "source, indexed-chunk count, fact count and the total number of "
            "memory items. Read-only; it returns counts only, never content. "
            "Use it to answer 'how much do I have' questions or to check that an "
            "ingest actually landed; use the recall tools to see what the content "
            "says."
        ),
        version="1.0.0",
        input_model=KnowledgeStatsInput,
        output_model=KnowledgeStatsOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=30.0,
        idempotent=True,
        parallel_safe=True,
        tags=("knowledge", "stats", "counts", "read"),
        guidance=(
            "回答「库里有多少内容」、确认一次入库是否真的落库、或做前后对比时用它。"
            "它只给计数、不给内容：要回答具体内容请用 knowledge.hybrid_recall / "
            "knowledge.multi_recall / memory.rag_search，不要用它代替检索。"
            "chunks_indexed 小于 chunks 说明向量投影落后，"
            "此时应先用 knowledge.reconcile 确认，再决定是否 knowledge.repair_drift。"
        ),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            from ._memory import build_default_manager

            self._manager = build_default_manager()
        return self._manager

    def execute(self, arguments: KnowledgeStatsInput) -> KnowledgeStatsOutput:
        return knowledge_stats(self.manager, repository_for(self.manager))


def create_tool() -> BaseTool:
    return KnowledgeStatsTool()
