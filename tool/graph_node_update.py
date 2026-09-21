"""图节点更新工具：修改实体节点（domain / 类型 / 描述 / 别名 / 重要度）。

为什么单独成工具
================

图里的「节点」是实体：它的可写属性只有 ``domain``、``entity_type``、
``description``、``aliases``、``importance``（以及来源 ``source_ids``）。
在抽取管道里这些属性原先是在 ``EntityResolver.resolve`` / ``_store_alias``
内联合并后直接 ``semantic.add`` 的——匹配逻辑（精确/前缀/模糊）和**属性更新
逻辑**混在一个方法里，无法被外部单独调用，也无法显式地「改一个节点」。

本工具把**属性更新**这一段收敛为唯一实现：

1. 合并：以已有实体记录为基准，把调用方给的新值覆盖上去（``domain`` /
   ``description`` 为空表示不改；``aliases`` 取并集；``importance`` 取
   ``max(旧值, 新值)``，只升不降，避免一次低置信抽取把既有节点降级）。
2. 写回真值源：``manager.semantic.add``（``memories`` 里的实体行）。
3. 镜像到图投影：调用图存储的 ``update_entity``。它只 ``MATCH`` 已存在的节点，
   **不会凭空造节点**——造节点仍然只由 ``add_relation`` / ``add_observation`` 负责，
   这样「更新节点」不会改变图的节点集合。

``EntityResolver`` 保留匹配职责，属性更新改为调用 :func:`update_entity_node`。
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from constants import (
    DEFAULT_DOMAIN,
    ENTITY_DEFAULT_CONFIDENCE,
    ENTITY_DEFAULT_TYPE,
    ENTITY_NAME_MAX_LENGTH,
)
from core import BaseTool, ToolSpec
from memory.base import MemoryItem, MemoryType
from memory.ids import _clean_text, entity_id_for, normalize_entity_name
from memory.manager import MemoryManager

TOOL_ENABLED = True


def _default_manager() -> MemoryManager:
    """Build the shared on-disk manager (imported lazily: ``_memory`` pulls ``memory.rag``)."""

    from ._memory import build_default_manager

    return build_default_manager()


def push_graph_entity(
    manager: MemoryManager,
    name: str,
    *,
    domain: str,
    aliases: Iterable[str],
    importance: float,
) -> bool:
    """Mirror entity attributes onto the graph projection; False when no node exists."""

    updater = getattr(getattr(manager, "graph_store", None), "update_entity", None)
    if not callable(updater):
        return False
    return bool(
        updater(
            name,
            domain=domain,
            aliases=[str(value) for value in aliases],
            importance=float(importance),
        )
    )


def update_entity_node(
    manager: MemoryManager,
    name: str,
    *,
    existing: MemoryItem | None = None,
    domain: str = "",
    entity_type: str = ENTITY_DEFAULT_TYPE,
    description: str = "",
    aliases: Iterable[str] | None = None,
    importance: float = ENTITY_DEFAULT_CONFIDENCE,
    source_id: str | None = None,
    add_written_name: bool = False,
    item_id: str | None = None,
    create_if_missing: bool = True,
) -> MemoryItem:
    """Merge the given attributes into one entity node and persist the result.

    ``existing`` is the already-matched entity item (``None`` means "no match").
    ``add_written_name`` records ``name`` itself as an alias, which is what an
    exact-hit lookup wants but a lookalike prefix hit must not do.
    """

    cleaned = _clean_text(name, max_length=ENTITY_NAME_MAX_LENGTH)
    if not cleaned:
        raise ValueError("entity name must not be empty")
    if existing is None and not create_if_missing:
        raise LookupError(f"entity node not found: {cleaned}")
    metadata = dict(existing.metadata) if existing is not None else {}
    canonical = str(
        metadata.get("canonical_name") or (existing.content if existing else cleaned)
    )
    known_aliases = {
        str(value) for value in metadata.get("aliases", []) if str(value).strip()
    }
    if add_written_name and cleaned != canonical:
        known_aliases.add(cleaned)
    if aliases:
        known_aliases.update(
            str(value).strip()
            for value in aliases
            if str(value).strip() and str(value).strip() != canonical
        )
    source_ids = {
        str(value) for value in metadata.get("source_ids", []) if str(value).strip()
    }
    if source_id:
        source_ids.add(source_id)
    metadata.update(
        {
            "kind": "entity",
            "title": canonical,
            "canonical_name": canonical,
            "entity_type": entity_type or metadata.get("entity_type") or ENTITY_DEFAULT_TYPE,
            "description": description or metadata.get("description", ""),
            "domain": domain or metadata.get("domain", DEFAULT_DOMAIN),
            "aliases": sorted(known_aliases),
            "source_ids": sorted(source_ids),
        }
    )
    target_id = item_id or (existing.id if existing is not None else entity_id_for(canonical))
    # 只升不降：一次低置信抽取不该把既有节点的重要度打回去。
    resolved_importance = max(
        float(importance), float(existing.importance) if existing is not None else 0.0
    )
    item = manager.semantic.add(
        canonical,
        metadata=metadata,
        importance=resolved_importance,
        item_id=target_id,
    )
    push_graph_entity(
        manager,
        canonical,
        domain=str(metadata["domain"]),
        aliases=metadata["aliases"],
        importance=resolved_importance,
    )
    return item


def find_entity_node(manager: MemoryManager, name: str) -> MemoryItem | None:
    """Locate one entity node by id first, then by normalized canonical name."""

    cleaned = _clean_text(name, max_length=ENTITY_NAME_MAX_LENGTH)
    if not cleaned:
        return None
    direct = manager.document_store.get(entity_id_for(cleaned))
    if (
        direct is not None
        and direct.memory_type == MemoryType.SEMANTIC
        and direct.metadata.get("kind") == "entity"
    ):
        return direct
    key = normalize_entity_name(cleaned)
    for item in manager.semantic.list():
        if item.metadata.get("kind") != "entity":
            continue
        canonical = str(item.metadata.get("canonical_name") or item.content)
        if normalize_entity_name(canonical) == key:
            return item
        if any(
            normalize_entity_name(str(alias)) == key
            for alias in item.metadata.get("aliases") or []
        ):
            return item
    return None


class GraphNodeUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(
        min_length=1,
        max_length=ENTITY_NAME_MAX_LENGTH,
        description="要更新的实体名（canonical 名或已知别名都可定位到同一节点）。",
    )
    domain: str | None = Field(
        default=None, max_length=100, description="新领域；null 或空串表示不修改。"
    )
    entity_type: str | None = Field(
        default=None, max_length=80, description="新实体类型；null 或空串表示不修改。"
    )
    description: str | None = Field(
        default=None, max_length=1000, description="新描述；null 或空串表示不修改。"
    )
    aliases: list[str] | None = Field(
        default=None,
        max_length=20,
        description="要并入的别名（取并集，不会删除已有别名）；null 表示不修改。",
    )
    importance: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="重要度；只会取 max(旧值, 新值)，不会把节点降级。null 表示不修改。",
    )
    create_if_missing: bool = Field(
        default=True,
        description="找不到节点时是否新建；false 时返回错误。",
    )


class GraphNodeUpdateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(description="实际写入的 canonical 实体名。")
    node_id: str
    created: bool
    domain: str
    entity_type: str
    aliases: list[str]
    importance: float
    graph_projected: bool = Field(
        description="图投影里是否已存在该节点并被同步刷新（不存在则只写真值源）。"
    )


class GraphNodeUpdateTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.graph_node_update",
        description=(
            "Update one existing knowledge-graph entity node: domain, entity "
            "type, description, aliases or importance. Aliases are merged "
            "(never dropped) and importance only ever increases. Use it to fix "
            "or enrich a node; use the ingest tools to create new facts."
        ),
        version="1.0.0",
        input_model=GraphNodeUpdateInput,
        output_model=GraphNodeUpdateOutput,
        side_effect="write",
        permissions=(),
        timeout_seconds=30.0,
        idempotent=True,
        parallel_safe=False,
        tags=("graph", "entity", "node", "update", "write"),
        guidance=(
            "只用于修正或丰富**已存在**的实体节点（领域/类型/描述/别名/重要度）。绝不用它创建新实体或新关系——新知识走 memory.rag 入库或 knowledge.add_fact。"
            "别名只增不减、重要度只升不降，因此不要指望用空值清掉已有字段；create_if_missing 默认 false，找不到节点会明确报错而不是新建。"
        ),
    )

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MemoryManager:
        if self._manager is None:
            self._manager = _default_manager()
        return self._manager

    def execute(self, arguments: GraphNodeUpdateInput) -> GraphNodeUpdateOutput:
        manager = self.manager
        existing = find_entity_node(manager, arguments.name)
        item = update_entity_node(
            manager,
            arguments.name,
            existing=existing,
            domain=arguments.domain or "",
            entity_type=arguments.entity_type or ENTITY_DEFAULT_TYPE,
            description=arguments.description or "",
            aliases=arguments.aliases,
            importance=(
                ENTITY_DEFAULT_CONFIDENCE
                if arguments.importance is None
                else arguments.importance
            ),
            create_if_missing=arguments.create_if_missing,
        )
        projected = bool(
            getattr(manager.graph_store, "entity", lambda _name: {})(item.metadata["canonical_name"])
        )
        return GraphNodeUpdateOutput(
            name=str(item.metadata["canonical_name"]),
            node_id=item.id,
            created=existing is None,
            domain=str(item.metadata.get("domain") or ""),
            entity_type=str(item.metadata.get("entity_type") or ""),
            aliases=[str(value) for value in item.metadata.get("aliases") or []],
            importance=float(item.importance),
            graph_projected=projected,
        )


def create_tool() -> BaseTool:
    return GraphNodeUpdateTool()


__all__ = [
    "GraphNodeUpdateInput",
    "GraphNodeUpdateOutput",
    "GraphNodeUpdateTool",
    "create_tool",
    "find_entity_node",
    "push_graph_entity",
    "update_entity_node",
]
