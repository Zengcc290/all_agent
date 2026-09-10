"""把 Aetheria 星图种子数据（web/seed_data.json）导入记忆库。

幂等：以 ``metadata.seed == SEED_MARK`` 作为标记，重复调用不会重复导入。
支持三种运行方式：
- ``python -m web.seed``            命令行手动播种
- 应用启动时自动播种（manager 为空时，可用 WEB_AUTOSEED=0 关闭）
- ``POST /api/seed``               强制重新检查播种
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory import MemoryManager, MemoryType

from .support import SEED_FILE

SEED_MARK = "aetheria-seed-v1"


def seed(manager: MemoryManager, path: Path | None = None) -> dict[str, Any]:
    seed_path = Path(path) if path is not None else SEED_FILE
    if not seed_path.is_file():
        return {"seeded": False, "reason": f"种子文件不存在：{seed_path}"}

    semantic_items = manager.list(memory_type=MemoryType.SEMANTIC)
    if any(item.metadata.get("seed") == SEED_MARK for item in semantic_items):
        return {"seeded": False, "reason": "已播种过（幂等跳过）",
                "existing": len(semantic_items)}

    data = json.loads(seed_path.read_text(encoding="utf-8"))
    entities = data.get("entities") or []
    relations = data.get("relations") or []
    notes = data.get("notes") or []

    entity_count = 0
    for entity in entities:
        name = str(entity.get("name") or "").strip()
        if not name:
            continue
        manager.add(
            name,
            memory_type=MemoryType.SEMANTIC,
            metadata={
                "kind": "entity",
                "title": name,
                "domain": entity.get("domain") or "未分类",
                "seed": SEED_MARK,
            },
            importance=float(entity.get("importance", 0.85)),
        )
        entity_count += 1

    relation_count = 0
    for relation in relations:
        subject = str(relation.get("subject") or "").strip()
        predicate = str(relation.get("predicate") or "").strip()
        obj = str(relation.get("object") or "").strip()
        if not (subject and predicate and obj):
            continue
        manager.semantic.add_fact(
            subject,
            predicate,
            obj,
            metadata={
                "domain": relation.get("domain") or "未分类",
                "note": relation.get("note") or "",
                "date": relation.get("date") or "",
                "seed": SEED_MARK,
            },
            confidence=float(relation.get("confidence", 1.0)),
        )
        relation_count += 1

    note_count = 0
    for note in notes:
        content = str(note.get("content") or "").strip()
        if not content:
            continue
        manager.add(
            content,
            memory_type=MemoryType.SEMANTIC,
            metadata={
                "kind": "note",
                "entity": note.get("entity") or "",
                "domain": note.get("domain") or "未分类",
                "title": note.get("title") or "档案",
                "date": note.get("date") or "",
                "seed": SEED_MARK,
            },
            importance=0.4,
        )
        note_count += 1

    return {
        "seeded": True,
        "source": str(seed_path),
        "entities": entity_count,
        "relations": relation_count,
        "notes": note_count,
    }


if __name__ == "__main__":
    from .support import get_manager

    manager = get_manager()
    print(seed(manager))
    manager.close()
