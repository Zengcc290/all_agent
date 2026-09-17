"""把记忆库中全部条目用当前配置的嵌入服务重新向量化。

每家提供方（转发网关 / Gemini ``:embedContent`` / OpenAI 兼容 ``/embeddings``）
都产出互不兼容的向量空间，切换提供方或模型后已入库条目的旧向量必须重算：

    .venv\\Scripts\\python.exe scripts\\reindex_embeddings.py

要重索引到哪一套向量空间，就按 memory/base.py 的选型规则配置环境变量（见
.env.example），没有远端配置时脚本会拒绝执行——离线 ``HashEmbedding`` 不是远端
空间的替代品，硬灌进去只会污染现有向量。

不影响 Agent 工具与 Web 共享的同一个库：SQLite 文档库与内存向量索引同步更新。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory import HashEmbedding, MemoryConfig, MemoryManager, make_default_embedding  # noqa: E402
from memory.embedding import load_dotenv_once  # noqa: E402


def main() -> int:
    load_dotenv_once()  # 把 .env 里的嵌入配置读进来
    embedding = make_default_embedding(MemoryConfig.from_env())
    if isinstance(embedding, HashEmbedding):
        print(
            "当前没有任何远端嵌入配置（选型回落到了离线 HashEmbedding），拒绝重索引："
            "离线向量与远端向量空间不兼容，硬灌会污染现有向量。\n"
            "请先配置 EMBEDDING_BASE_URL，或 HELLOAGENTS_MEMORY_EMBEDDING_API_KEY / "
            "_BASE_URL / _MODEL / _PROVIDER 之一。"
        )
        return 1
    db_path = Path(os.getenv("MEMORY_DB_PATH") or (ROOT / "memory.sqlite3"))
    # 批量重索引时放宽超时（原本就是 180s）：一次请求慢不该判为失败。
    embedding.timeout = max(float(getattr(embedding, "timeout", 0.0) or 0.0), 180.0)
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