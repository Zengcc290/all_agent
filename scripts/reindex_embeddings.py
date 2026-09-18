"""把记忆库中全部条目用当前配置的云端嵌入服务重新向量化。

每家提供方/模型都产出互不兼容的向量空间，切换模型后已入库条目的旧向量必须重算：

    .venv\\Scripts\\python.exe scripts\\reindex_embeddings.py

重索引到哪一套向量空间由 ``config/services.toml`` 的 ``[embedding]`` 段决定
（base_url / api_key / model）；没有云端配置时脚本会拒绝执行——离线
``HashEmbedding`` 不是云端空间的替代品，硬灌进去只会污染现有向量。

向量写入「当前配置的向量存储」：配置了 config/services.toml [qdrant] 就是云端
Qdrant 集合，否则是内存回退——与 Web/Agent 运行时同一份投影，不会写丢。

维度变更（如 1024 -> 4096）请先运行：
    python scripts/migrate_to_cloud.py --recreate-collection
重建集合并重灌 chunks，再运行本脚本重灌记忆条目向量。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory import HashEmbedding, MemoryConfig, MemoryManager, default_sqlite_path, make_default_embedding  # noqa: E402


def main() -> int:
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
    print(f"向量存储：{type(manager.vector_store).__name__}")
    items = manager.document_store.list(include_expired=True)
    print(f"待重索引：{len(items)} 条 -> {embedding!r}")

    done = failed = 0
    for item in items:
        try:
            item.embedding = embedding.embed(item.content)
            manager.document_store.upsert(item)
            manager.vector_store.upsert(item)
            done += 1
        except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整批重索引
            failed += 1
            print(f"  ! {item.id}: {type(exc).__name__}: {exc}")
    print(f"完成：{done} 条成功，{failed} 条失败，共 {len(items)} 条。")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
