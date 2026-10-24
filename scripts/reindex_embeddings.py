"""把记忆库中全部条目用当前配置的云端嵌入服务重新向量化。

每家提供方/模型都产出互不兼容的向量空间，切换模型后已入库条目的旧向量必须重算：

    .venv\\Scripts\\python.exe scripts\\reindex_embeddings.py

重索引到哪一套向量空间由 ``config/services.toml`` 的 ``[embedding]`` 段决定
（base_url / api_key / model）；没有云端配置时脚本会拒绝执行——离线
``HashEmbedding`` 不是云端空间的替代品，硬灌进去只会污染现有向量。

向量写入「当前配置的向量存储」：配置了 config/services.toml [qdrant] 就是云端
Qdrant 集合，否则是内存回退——与 Web/Agent 运行时同一份投影，不会写丢。

嵌入空间不一致时必须加 ``--confirm-rebuild``。锁闸门会一次性重建集合并重灌
chunks 和 memories；脚本不会再做第二轮重复嵌入。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory import HashEmbedding, MemoryConfig, MemoryManager, default_sqlite_path, make_default_embedding  # noqa: E402
from memory.embedding_lock import (  # noqa: E402
    EmbeddingLockMismatch,
    apply_embedding_lock,
    inspect_embedding_lock,
    reindex_vector_projection,
)
from memory.storage.document_repo import DocumentRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="用当前嵌入配置重灌记忆条目向量")
    parser.add_argument(
        "--confirm-rebuild",
        action="store_true",
        help="嵌入锁定不一致时：重建向量集合并全量重灌（否则拒绝）",
    )
    args = parser.parse_args()
    config = MemoryConfig.from_config()
    embedding = make_default_embedding(config)
    if isinstance(embedding, HashEmbedding):
        print(
            "当前没有云端嵌入配置（选型回落到了离线 HashEmbedding），拒绝重索引："
            "离线向量与云端向量空间不兼容，硬灌会污染现有向量。\n"
            "请在 config/services.toml 的 [embedding] 段填写 "
            "base_url / api_key / model。"
        )
        return 1
    db_path = Path(default_sqlite_path())
    # 批量重索引时放宽超时（原本就是 180s）：一次请求慢不该判为失败。
    embedding.timeout = max(float(getattr(embedding, "timeout", 0.0) or 0.0), 180.0)
    # 与 Web/Agent 同一份配置（含云端 Qdrant 投影），只覆盖库路径。
    config.sqlite_path = str(db_path)
    manager = MemoryManager(config, embedding=embedding)
    repo = DocumentRepository(db_path) if db_path.exists() else None
    try:
        snapshot = inspect_embedding_lock(manager, repo) if repo is not None else {"mismatch": False}
        rebuilt_projection = bool(snapshot["mismatch"])
        if rebuilt_projection:
            try:
                apply_embedding_lock(manager, repo, confirm_rebuild=args.confirm_rebuild)
            except EmbeddingLockMismatch as exc:
                print(exc)
                print("换模型后请加 --confirm-rebuild 以重建向量集合并全量重灌。")
                return 2
            except Exception as exc:  # noqa: BLE001 - CLI reports rebuild failure cleanly
                print(f"重建失败，嵌入锁未更新：{type(exc).__name__}: {exc}")
                return 2

        print(f"向量存储：{type(manager.vector_store).__name__}")
        if rebuilt_projection:
            print("完成：嵌入空间已变更，锁闸门已一次性重灌 chunks 与 memories。")
            return 0

        if repo is None:  # manager construction normally creates the SQLite file
            print("找不到 SQLite 真值源，拒绝只重建部分投影。")
            return 2
        print(f"按唯一 ID 重索引 chunks 与 memories -> {embedding!r}")
        report = reindex_vector_projection(
            manager,
            repo,
            recreate_collection=False,
            continue_on_error=True,
        )
        failures = report["failed"]
        for failure in failures:
            print(f"  ! {failure['id']}: {failure['error']}")
        print(
            f"完成：{report['chunks']} 个分块、{report['memories']} 条记忆成功，"
            f"{len(failures)} 个唯一 ID 失败。"
        )
        return 0 if not failures else 2
    finally:
        if repo is not None:
            repo.close()
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
