"""一次性迁移：把 memories 里已有的分块回填到 documents/chunks，并让向量真值归位。

用法：
    .venv\\Scripts\\python.exe scripts\\migrate_storage.py [--db memory.sqlite3] [--dry-run] [--no-backup]

设计要点（方案 P5 / D1 / D3）：
- 迁移前默认用 SQLite 在线备份写出 ``{db}.bak.pre_migration``；
- 幂等：``documents`` 按 document_id、``chunks`` 按 chunk_id 存在即跳过；
- ``memories`` 表本身不动：事实条目（subject/predicate/object）与
  working/episodic/perceptual 全部保留，只新增 documents/chunks 两张表；
- ``memories.embedding`` **只在重嵌入成功之后**才清空：先清空再失败会既丢向量又
  没有新向量（Qdrant 是唯一向量真值，D3），所以嵌入网关不可用时整段跳过并保留原值；
- Neo4j 未开启时不重放图，只报告 0。

结束输出：``{"documents": n, "chunks": m, "facts_kept": k, "graph_replayed": g}``
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.base import MemoryConfig, MemoryItem, MemoryType  # noqa: E402
from memory.manager import MemoryManager  # noqa: E402
from memory.storage.document_repo import ChunkRecord, DocumentRecord, DocumentRepository  # noqa: E402

BACKUP_SUFFIX = ".bak.pre_migration"


def backup_database(db: Path) -> Path:
    """Consistent online backup; a plain file copy can catch a half-written page."""

    target = db.with_name(db.name + BACKUP_SUFFIX)
    with sqlite3.connect(db) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    return target


def chunk_documents(items: list[MemoryItem]) -> dict[str, list[MemoryItem]]:
    """Group semantic chunk rows by document_id, ordered by chunk_index."""

    grouped: dict[str, list[MemoryItem]] = {}
    for item in items:
        if item.memory_type is not MemoryType.SEMANTIC:
            continue
        document_id = str(item.metadata.get("document_id") or "")
        if not document_id or "chunk_index" not in item.metadata:
            continue
        grouped.setdefault(document_id, []).append(item)
    for chunks in grouped.values():
        chunks.sort(key=lambda item: int(item.metadata.get("chunk_index") or 0))
    return grouped


def document_record(document_id: str, chunks: list[MemoryItem]) -> tuple[DocumentRecord, list[ChunkRecord]]:
    """Rebuild the真值源 rows for one document from its chunk rows.

    ``raw_text`` is the chunk texts joined by newlines, and each chunk's
    ``char_start``/``char_end`` are measured in exactly that text - the offsets
    are only meaningful relative to the text they were sliced from (方案 2.3).
    """

    separator = "\n"
    raw_text = separator.join(item.content for item in chunks)
    records: list[ChunkRecord] = []
    cursor = 0
    for item in chunks:
        start = cursor
        end = start + len(item.content)
        records.append(
            ChunkRecord(
                chunk_id=item.id,
                document_id=document_id,
                chunk_index=int(item.metadata.get("chunk_index") or 0),
                char_start=start,
                char_end=end,
                text=item.content,
            )
        )
        cursor = end + len(separator)
    source = str(chunks[0].metadata.get("source") or document_id)
    document = DocumentRecord(
        document_id=document_id,
        title=str(chunks[0].metadata.get("title") or ""),
        raw_text=raw_text,
        source=source,
        tags=[str(tag) for tag in chunks[0].metadata.get("tags") or []],
        permission=str(chunks[0].metadata.get("permission") or "private"),
        status="parsed",  # 向量是否补齐由重嵌入阶段决定
    )
    return document, records


def is_fact(item: MemoryItem) -> bool:
    return all(item.metadata.get(key) for key in ("subject", "predicate", "object"))


def migrate(manager: MemoryManager, *, dry_run: bool = False, repository: DocumentRepository | None = None) -> dict[str, int]:
    """Run the SQLite-side migration; returns the run report."""

    owns_repo = repository is None
    repo = repository if repository is not None else DocumentRepository(manager.document_store.path)
    report = {"documents": 0, "chunks": 0, "facts_kept": 0, "graph_replayed": 0}
    try:
        items = manager.document_store.list(include_expired=True)
        for document_id, chunks in chunk_documents(items).items():
            existing = repo.get_document(document_id)
            document, records = document_record(document_id, chunks)
            if dry_run:
                report["documents"] += int(existing is None)
                report["chunks"] += len(records) if existing is None else 0
                continue
            repo.upsert_document(document)
            repo.upsert_chunks(records)
            report["documents"] += int(existing is None)
            report["chunks"] += len(records) if existing is None else 0

        for item in items:
            if not is_fact(item):
                continue
            report["facts_kept"] += 1
            if dry_run or not manager.config.neo4j_uri:
                continue
            # 复用真实写入路径，保证图边属性与在线抽取完全一致（含实体属性）。
            manager.semantic.add_fact(
                str(item.metadata["subject"]),
                str(item.metadata["predicate"]),
                str(item.metadata["object"]),
                metadata=item.metadata,
                confidence=float(item.importance),
                item_id=item.id,
            )
            report["graph_replayed"] += 1
    finally:
        if owns_repo:
            repo.close()
    return report


def reindex(manager: MemoryManager, *, dry_run: bool = False) -> dict[str, int]:
    """Re-embed every memory item into Qdrant, then blank ``memories.embedding``.

    Returns ``{"reindexed": n, "blanked": m, "skipped": reason}``. Skipping is
    deliberate and total: without a reachable gateway the existing vectors are
    the only copy, so nothing is blanked.
    """

    base_url = getattr(manager.embedding, "base_url", None)
    if not base_url:
        return {"reindexed": 0, "blanked": 0, "skipped": "no_cloud_embedding"}
    items = manager.document_store.list(include_expired=True)
    if dry_run:
        return {"reindexed": len(items), "blanked": len(items), "skipped": ""}

    repo = DocumentRepository(manager.document_store.path)
    try:
        # 一次批量请求：远端网关按批推理，逐条发会让耗时随条数线性增长。
        vectors = manager.embedding.embed_batch([item.content for item in items])
        for item, vector in zip(items, vectors, strict=True):
            item.embedding = vector
            chunk = repo.get_chunk(item.id)
            if chunk is None:
                manager.vector_store.upsert(item)
                continue
            manager.vector_store.upsert_chunk(
                chunk.chunk_id,
                vector,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source=str(item.metadata.get("source") or ""),
            )
            repo.set_chunk_vector_status(chunk.chunk_id, "indexed")
        for document_id in chunk_documents(items):
            repo.set_status(document_id, "vectorized")
        # 向量已在 Qdrant，memories.embedding 不再持有副本（D3）。
        for item in items:
            item.embedding = None
            manager.document_store.upsert(item)
    finally:
        repo.close()
    return {"reindexed": len(items), "blanked": len(items), "skipped": ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 memories 回填到 documents/chunks 并归位向量真值")
    parser.add_argument("--db", default=None, help="SQLite 路径，默认 MEMORY_DB_PATH 或仓库根目录 memory.sqlite3")
    parser.add_argument("--dry-run", action="store_true", help="只报告将要发生的变化，不写库")
    parser.add_argument("--no-backup", action="store_true", help="跳过迁移前备份（不推荐）")
    parser.add_argument("--no-reindex", action="store_true", help="不做向量重嵌入与 memories.embedding 清空")
    args = parser.parse_args(argv)

    db = Path(args.db or (Path(__file__).resolve().parent.parent / "memory.sqlite3")).expanduser()
    if not db.exists():
        print(f"数据库不存在：{db}")
        return 1
    if not args.dry_run and not args.no_backup:
        print(f"已备份：{backup_database(db)}")

    config = MemoryConfig.from_config()
    config.sqlite_path = str(db)
    manager = MemoryManager(config)
    try:
        report = migrate(manager, dry_run=args.dry_run)
        if not args.no_reindex:
            outcome = reindex(manager, dry_run=args.dry_run)
            report["reindexed"] = outcome["reindexed"]
            report["blanked"] = outcome["blanked"]
            if outcome["skipped"]:
                print(
                    f"跳过向量重嵌入（{outcome['skipped']}）：memories.embedding 保持原值，"
                    "隧道恢复后重跑本脚本即可补齐。"
                )
    finally:
        manager.close()
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
