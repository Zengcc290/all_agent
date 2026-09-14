"""把记忆库中全部条目用当前配置的嵌入服务重新向量化。

切换嵌入提供方会改变向量空间（离线 ``HashEmbedding`` 与 ``qwen-embed`` 网关
互不兼容），已入库条目的旧向量必须重算。启用 ``EMBEDDING_BASE_URL`` 后执行：

    .venv\\Scripts\\python.exe scripts\\reindex_embeddings.py

用法：
    EMBEDDING_BASE_URL=http://127.0.0.1:10800 .venv\\Scripts\\python.exe scripts\\reindex_embeddings.py

不影响 Agent 工具与 Web 共享的同一个库：SQLite 文档库与内存向量索引同步更新。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.base import MemoryConfig  # noqa: E402
from memory.embedding import EmbedServerEmbedding, load_dotenv_once  # noqa: E402
from memory.manager import MemoryManager  # noqa: E402


def main() -> int:
    load_dotenv_once()  # 把 .env 里的 EMBEDDING_BASE_URL 读进来
    server_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip()
    if not server_url:
        print("EMBEDDING_BASE_URL 未设置，拒绝重索引（错误的重索引会污染现有向量）。")
        return 1
    db_path = Path(os.getenv("MEMORY_DB_PATH") or (ROOT / "memory.sqlite3"))
    embedding = EmbedServerEmbedding(base_url=server_url, timeout=180.0)
    manager = MemoryManager(
        MemoryConfig(sqlite_path=str(db_path)),
        embedding=embedding,
    )
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