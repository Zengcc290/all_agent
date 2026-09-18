"""constants.py「连接与端点」小节的守护测试。

这几条用例钉的是「配置只有一个事实来源」这个属性：
  1) 文档化的默认端点/端口不能悄悄变——使用说明与方案文档都引用它们；
  2) 各模块的默认参数必须**引用**常量对象，而不是各自再写一遍字面量
     （因此用 ``is`` 而不是 ``==``：重新写一份等值字面量也要被测出来）；
  3) 嵌入配置没有出厂端点：云端 base_url 只能来自 services.toml [embedding]
     （历史 DEFAULT_EMBEDDING_BASE_URL 指向本机网关，已随网关实现一并删除）。
"""

from __future__ import annotations

import inspect

from constants import (
    DEFAULT_NEO4J_BOLT_PORT,
    DEFAULT_NEO4J_URI,
    DEFAULT_QDRANT_PORT,
    DEFAULT_QDRANT_URL,
    DEFAULT_WEB_PORT,
    KNOWLEDGE_INGEST_WORKERS,
    LOCALHOST,
    MEMORY_HYBRID,
    MEMORY_QDRANT_COLLECTION,
    WEB_AUTOSEED,
    WEB_QA_EXTRACT,
    WEB_QA_EXTRACT_SYNC,
)
from memory import MemoryConfig
from memory.storage.qdrant import QdrantVectorStore


def test_documented_connection_defaults():
    """本机回环地址/端口/端点的取值（= 用户手册里写的那几个）。"""

    assert LOCALHOST == "127.0.0.1"
    assert (
        DEFAULT_QDRANT_PORT,
        DEFAULT_NEO4J_BOLT_PORT,
        DEFAULT_WEB_PORT,
    ) == (6333, 7687, 8765)
    assert DEFAULT_QDRANT_URL == "http://127.0.0.1:6333"
    assert DEFAULT_NEO4J_URI == "bolt://127.0.0.1:7687"
    # 端点由「地址 + 端口」常量拼成，改端口不需要再改端点字面量
    assert DEFAULT_QDRANT_URL == f"http://{LOCALHOST}:{DEFAULT_QDRANT_PORT}"
    assert DEFAULT_NEO4J_URI == f"bolt://{LOCALHOST}:{DEFAULT_NEO4J_BOLT_PORT}"


def test_connection_defaults_are_referenced_not_rehardcoded():
    """回归：这几处曾各自硬编码端点/库名，必须继续引用 constants.py 的对象。"""

    # 出厂仍然是 None = 不连接（走内存回退）；常量只回答「要连就填什么值」
    assert MemoryConfig().qdrant_url is None
    assert MemoryConfig().neo4j_uri is None
    # 嵌入没有出厂端点：空串 = 未配置云端（离线兜底）
    assert MemoryConfig().embedding_base_url == ""
    # 默认参数必须是同一个常量对象：重新写一份等值字面量也会让这个断言失败
    assert inspect.signature(QdrantVectorStore.__init__).parameters["collection_name"].default is MEMORY_QDRANT_COLLECTION


def test_runtime_toggles_have_single_source():
    """原散落环境变量收拢后的运行开关：常量存在且默认值符合文档。"""

    assert WEB_AUTOSEED is True
    assert WEB_QA_EXTRACT is True
    assert WEB_QA_EXTRACT_SYNC is False
    assert MEMORY_HYBRID is True
    assert isinstance(KNOWLEDGE_INGEST_WORKERS, int) and KNOWLEDGE_INGEST_WORKERS >= 1
