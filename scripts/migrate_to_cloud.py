"""把 memory.sqlite3 里已有的 chunks 用当前嵌入配置重新向量化并入库云端。

场景：嵌入模型切换后（qwen 网关 -> SiliconFlow BAAI/bge-m3），SQLite 里已落库的
chunks（documents/chunks 表）必须用新模型重算向量，并写入云端 Qdrant 集合；
documents 行同时回填（此前可能缺失），chunks.vector_status 置为 indexed。

用法：
    .venv\\Scripts\\python.exe scripts\\migrate_to_cloud.py [--db memory.sqlite3]

- 云端连接参数全部来自 config/services.toml（[embedding]/[qdrant]），
  env 优先，.env 里的本地覆盖请先注释（本次迁移已处理）。
- 幂等：按 chunk_id upsert，重复执行不会产生重复点。
- 退出码 0 = 全部成功；任何一步失败会抛错并保留现场（向量状态仍为 pending）。
- 输出：分阶段耗时（嵌入 / 上传 / 总计）与入库统计。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory import HashEmbedding, MemoryConfig, MemoryManager, make_default_embedding  # noqa: E402
from memory.storage.document_repo import DocumentRecord, DocumentRepository  # noqa: E402
from memory.storage.qdrant import QdrantVectorStore  # noqa: E402

BACKUP_SUFFIX = ".bak.pre_migration"


def backup_database(db: Path) -> Path:
    target = db.with_name(db.name + BACKUP_SUFFIX)
    with sqlite3.connect(db) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="chunks 重新向量化并入库云端 Qdrant")
    parser.add_argument("--db", default=None, help="SQLite 路径，默认仓库根目录 memory.sqlite3")
    parser.add_argument("--no-backup", action="store_true", help="跳过迁移前备份（不推荐）")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="嵌入模型维度变更时：先删除云端集合并按新维度重建（向量是 SQLite 真值的投影，随后全量重灌即可恢复）",
    )
    args = parser.parse_args(argv)

    db = Path(args.db or (ROOT / "memory.sqlite3")).expanduser()
    if not db.exists():
        print(f"数据库不存在：{db}")
        return 1
    if not args.no_backup:
        backup = backup_database(db)
        print(f"已备份：{backup}")

    config = MemoryConfig.from_config()
    config.sqlite_path = str(db)
    embedding = make_default_embedding(config)
    if isinstance(embedding, HashEmbedding):
        print("当前嵌入回落到了离线 HashEmbedding，拒绝迁移：请先配置云端嵌入（config/services.toml [embedding]）。")
        return 1
    print(f"嵌入服务：{embedding!r}")
    print(f"Qdrant 集合：{config.qdrant_collection} @ {config.qdrant_url}")

    manager = MemoryManager(config, embedding=embedding)
    repo = DocumentRepository(db)
    report: dict[str, object] = {}
    try:
        chunk_ids = repo.chunk_ids()
        if not chunk_ids:
            print("没有待迁移的 chunks，无事可做。")
            return 0

        chunks = [repo.get_chunk(cid) for cid in chunk_ids]
        chunks = [c for c in chunks if c is not None]
        print(f"待处理 chunks：{len(chunks)} 条")

        # ---- 阶段 1：嵌入（新模型 bge-m3）----
        t0 = time.perf_counter()
        vectors = embedding.embed_batch([chunk.text for chunk in chunks])
        t_embed = time.perf_counter() - t0
        if len(vectors) != len(chunks):
            raise RuntimeError(f"嵌入数量不匹配：{len(vectors)} != {len(chunks)}")
        dims = {len(v) for v in vectors}
        print(f"嵌入完成：{len(vectors)} 条，维度 {dims}，耗时 {t_embed:.3f}s")

        # ---- 可选：维度变更时先重建集合（阶段 2 之前，否则旧集合按新维度 upsert 会报维度不一致）----
        if args.recreate_collection:
            if isinstance(manager.vector_store, QdrantVectorStore):
                manager.vector_store.recreate_collection(len(vectors[0]))
                print(f"已按新维度 {len(vectors[0])} 重建集合 {config.qdrant_collection}（旧投影已丢弃）")
            else:
                print("未配置 Qdrant（内存回退），无需重建集合。")

        # ---- 阶段 2：上传云端 Qdrant（按 chunk_id 幂等 upsert）----
        t1 = time.perf_counter()
        for chunk, vector in zip(chunks, vectors, strict=True):
            manager.vector_store.upsert_chunk(
                chunk.chunk_id,
                vector,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source=chunk.document_id,
            )
        t_upload = time.perf_counter() - t1
        print(f"Qdrant 上传完成：{len(chunks)} 点，耗时 {t_upload:.3f}s")

        # ---- 阶段 3：回填 documents 行 + 状态置 indexed ----
        t2 = time.perf_counter()
        by_document: dict[str, list] = {}
        for chunk in chunks:
            by_document.setdefault(chunk.document_id, []).append(chunk)
        for document_id, doc_chunks in by_document.items():
            doc_chunks.sort(key=lambda c: c.chunk_index)
            repo.upsert_document(
                DocumentRecord(
                    document_id=document_id,
                    raw_text="\n".join(c.text for c in doc_chunks),
                    source=document_id,
                    status="vectorized",
                )
            )
            repo.set_status(document_id, "vectorized")
        for chunk in chunks:
            repo.set_chunk_vector_status(chunk.chunk_id, "indexed")
        t_status = time.perf_counter() - t2

        # ---- 阶段 4：核对云端落库点数 ----
        t3 = time.perf_counter()
        remote_ids = manager.vector_store.list_ids(limit=10000)
        t_check = time.perf_counter() - t3

        report = {
            "chunks": len(chunks),
            "documents": len(by_document),
            "embed_seconds": round(t_embed, 3),
            "upload_seconds": round(t_upload, 3),
            "status_seconds": round(t_status, 3),
            "verify_seconds": round(t_check, 3),
            "total_seconds": round(time.perf_counter() - t0, 3),
            "qdrant_points": len(remote_ids),
        }
    finally:
        manager.close()
        repo.close()

    print("== 入库报告 ==")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
