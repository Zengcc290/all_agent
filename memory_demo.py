"""四层记忆系统演示与测试脚本。

覆盖三层验证：
  1. 四层记忆 CRUD / 检索 / TTL（working / episodic / semantic / perceptual）
  2. RAG：把本地文档（默认你桌面的 Aetheria HTML）摄取进语义记忆并检索
  3. 向量数据库：Qdrant 本地模式接入（无需启动服务），并展示远程配置方式

Embedding 默认走 qwen3-embedding-0.6b（DashScope OpenAI 兼容 API，1024 维），
api_key 从环境变量 DASHSCOPE_API_KEY 读取；未配置时退回本地 HashEmbedding
占位跑通全流程（只做演示，检索质量有限）。

用法:
    .venv\\Scripts\\python.exe memory_demo.py                      # 用默认 HTML 文档
    .venv\\Scripts\\python.exe memory_demo.py path/to/doc.html     # 指定文档
    .venv\\Scripts\\python.exe memory_demo.py --no-rag             # 只测四层记忆
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 默认示例文档：你桌面上的 Aetheria 页面（路径写死便于直接测试）
DEFAULT_DOC = r"C:\Users\liang\Desktop\python\新建 文本文档 (6).html"


def banner(title: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}\n{title}\n{char * width}")


class HashEmbedding:
    """离线确定性 embedding，仅在未配置 API key 时用于演示占位。"""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
        vector = [0.0] * self.dimension
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0 + math.log(float(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def build_embedding():
    """优先 qwen3-embedding-0.6b（需 DASHSCOPE_API_KEY），否则 HashEmbedding。"""
    from memory import APIEmbedding

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        print(f"[embedding] 使用 qwen3-embedding-0.6b (DashScope API, {api_key[:4]}...)")
        return APIEmbedding(api_key=api_key, model="qwen3-embedding-0.6b", dimension=1024)
    print("[embedding] 未设置 DASHSCOPE_API_KEY，使用本地 HashEmbedding 占位（演示用）")
    return HashEmbedding(1024)


def demo_four_layers() -> None:
    """第一层验证：四层记忆各自的增删查与语义检索。"""
    from memory import MemoryConfig, MemoryManager, MemoryType

    banner("[1/3] 四层记忆系统")
    # sqlite_path=":memory:" 表示不落盘；换成文件路径即可持久化
    memory = MemoryManager(
        MemoryConfig(sqlite_path=":memory:", working_memory_capacity=5),
        embedding=build_embedding(),
    )

    # --- working 工作记忆：TTL + 容量淘汰 ---
    memory.working.set("user_name", "小明", ttl_seconds=300)
    memory.working.set("session_topic", "记忆系统", importance=0.9)
    print("working.get_value('user_name') =", memory.working.get_value("user_name"))

    # --- episodic 情景记忆：事件时间线 ---
    memory.episodic.record("用户在上海参加了关于 RAG 的会议")
    memory.episodic.record("用户把向量数据库接入记忆系统")
    print("episodic 数量 =", len(memory.episodic.list()))

    # --- semantic 语义记忆：知识三元组（自动进入图存储） ---
    memory.semantic.add_fact("Qdrant", "是一种", "向量数据库")
    memory.semantic.add_fact("向量数据库", "存储", "Embedding 向量")
    print("semantic 与 Qdrant 相关关系 =", memory.semantic.related("Qdrant"))

    # --- perceptual 感知记忆：多模态 payload ---
    memory.perceptual.store(b"\x89PNG fake image bytes", modality="image", content="记忆系统架构图")
    print("perceptual 数量 =", len(memory.perceptual.list()))

    # --- 跨层语义检索（manager.search 会合并四层结果并按相似度排序） ---
    banner("跨层检索 query='向量数据库'")
    for result in memory.search("向量数据库", limit=5):
        print(f"  [{result.item.memory_type.value:<9}] score={result.score:.3f}  {result.item.content[:40]}")

    memory.close()
    print("\n四层记忆验证完成 ✅")


def demo_rag(doc_path: str) -> None:
    """第二层验证：RAG 摄取 -> 切块 -> 语义检索 -> 拼装上下文。"""
    from memory import MemoryConfig, MemoryManager
    from memory.rag import RAGPipeline

    path = Path(doc_path)
    if not path.is_file():
        print(f"\n[2/3] 跳过 RAG：找不到文档 {doc_path}")
        return

    banner(f"[2/3] RAG 文档摄取与检索: {path.name}")
    pipeline = RAGPipeline(MemoryManager(MemoryConfig(sqlite_path=":memory:"), embedding=build_embedding()))
    items = pipeline.ingest_source(str(path), chunk_size=300, overlap=50)
    print(f"文档解析并切出 {len(items)} 个 chunk（全部存入 semantic 记忆）")
    print("chunk 示例:", items[0].content[:120], "...")

    queries = ["天体物理 星系系统", "认知记忆 子系统", "GraphRAG"]
    for query in queries:
        results = pipeline.retrieve(query, limit=2)
        print(f"\nquery='{query}' -> {len(results)} 条命中")
        for chunk in results:
            print(f"  score={chunk.score:.3f}  {chunk.content[:60]}")

    context = pipeline.build_context("认知记忆 子系统", limit=2)
    print(f"\nbuild_context 拼接长度 = {len(context)} 字符")
    pipeline.close()
    print("\nRAG 验证完成 ✅")


def demo_qdrant(doc_path: str) -> None:
    """第三层验证：向量数据库（Qdrant）接入。"""
    from memory import MemoryConfig, MemoryManager, QdrantVectorStore

    banner("[3/3] 向量数据库接入（Qdrant 本地模式）")

    # 方式 A（推荐用于本地测试）: QdrantVectorStore 不传 url -> 进程内本地模式，零配置
    vector_store = QdrantVectorStore(collection_name="demo_memory")
    manager = MemoryManager(
        MemoryConfig(sqlite_path=":memory:", embedding_dimension=1024),
        vector_store=vector_store,
        embedding=build_embedding(),
    )

    manager.semantic.add_fact("Qdrant", "支持", "本地嵌入式模式")
    manager.semantic.add_fact("Qdrant", "支持", "分布式部署")
    hits = manager.search("Qdrant 部署方式", memory_type="semantic", limit=3)
    print(f"Qdrant 检索命中 {len(hits)} 条:")
    for result in hits:
        print(f"  score={result.score:.3f}  {result.item.content}")
    manager.close()

    # 方式 B（远程服务）: 配置 qdrant_url 后，MemoryManager 会自动改用 Qdrant
    print("\n远程接入方式（有 Qdrant 服务时启用）：")
    print("  MemoryConfig(qdrant_url='http://localhost:6333', qdrant_collection='my_mem')")
    print("  或环境变量 HELLOAGENTS_MEMORY_QDRANT_URL=http://localhost:6333")
    print("\nQdrant 接入验证完成 ✅")


def main() -> None:
    parser = argparse.ArgumentParser(description="四层记忆系统演示")
    parser.add_argument("doc", nargs="?", default=DEFAULT_DOC, help="要摄取的文档路径（默认 Aetheria HTML）")
    parser.add_argument("--no-rag", action="store_true", help="跳过 RAG 部分")
    args = parser.parse_args()

    demo_four_layers()
    if not args.no_rag:
        demo_rag(args.doc)
    demo_qdrant(args.doc)

    banner("全部完成")
    print("提示：sqlite_path 换成 'memory.sqlite3' 即可持久化；")
    print("      QdrantVectorStore(url='http://localhost:6333') 即连远程向量库。")
    print("      设置 DASHSCOPE_API_KEY 环境变量即可启用 qwen3-embedding-0.6b 真实嵌入。")


if __name__ == "__main__":
    main()
