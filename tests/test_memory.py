from datetime import UTC, datetime, timedelta

from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, MemoryType, Neo4jGraphStore
from memory.storage import SQLiteDocumentStore


def _manager(**config_kwargs):
    return MemoryManager(MemoryConfig(sqlite_path=":memory:", **config_kwargs), embedding=HashEmbedding())


def test_manager_crud_search_and_type_isolation():
    manager = _manager()
    item = manager.add("Python is a programming language", memory_type=MemoryType.SEMANTIC)
    manager.add("The meeting starts at nine", memory_type=MemoryType.EPISODIC)

    assert manager.get(item.id).id == item.id
    matches = manager.search("programming")
    assert matches and matches[0].item.id == item.id
    assert manager.search("programming", memory_type="episodic") == []
    assert manager.delete(item.id)
    assert manager.get(item.id) is None


def test_working_memory_ttl_and_capacity():
    manager = _manager(working_memory_capacity=2)
    manager.working.set("a", 1, importance=0.1)
    manager.working.set("b", 2, importance=0.9)
    manager.working.set("c", 3, importance=0.9)
    assert manager.working.get_value("a") is None
    assert manager.working.get_value("c") == 3
    expired = manager.working.add("short lived", ttl_seconds=0.01)
    expired.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    manager.document_store.upsert(expired)
    assert manager.working.get(expired.id) is None


def test_semantic_memory_graph_fallback():
    manager = MemoryManager(MemoryConfig(sqlite_path=":memory:"), graph_store=Neo4jGraphStore(), embedding=HashEmbedding())
    manager.semantic.add_fact("Alice", "knows", "Bob")
    relations = manager.semantic.related("Alice")
    assert relations[0]["target"] == "Bob"


def test_hash_embedding_dimension_is_stable():
    embedding = HashEmbedding(32)
    assert len(embedding.embed("hello")) == 32
    assert len(embedding.embed("hello world")) == 32


def test_sqlite_memory_store_creates_nested_parent_directory(tmp_path):
    path = tmp_path / "nested" / "memory" / "records.sqlite3"
    store = SQLiteDocumentStore(path)

    assert path.exists()
    store.close()


def test_manager_preserves_falsey_injected_embedding():
    class FalseyEmbedding(HashEmbedding):
        def __bool__(self):
            return False

    embedding = FalseyEmbedding(8)
    manager = MemoryManager(MemoryConfig(sqlite_path=":memory:"), embedding=embedding)

    assert manager.embedding is embedding


def test_memory_item_default_id_is_a_uuid():
    """回归：MemoryItem 的默认 id 曾引用未导入的 ``uuid``，直接构造即 NameError。"""

    from uuid import UUID

    from memory import MemoryItem

    item = MemoryItem("hello")
    assert UUID(item.id)
    assert MemoryItem("hello").id != item.id


def test_memory_manager_without_api_key_falls_back_to_offline_embedding(monkeypatch):
    """回归：没有任何 API key 时 MemoryManager() 必须可用（离线哈希降级）。

    历史缺陷：``make_default_embedding`` 无条件构造 ``APIEmbedding``，没有 key
    时构造 MemoryManager 直接抛 RuntimeError，导致离线环境下 memory 包、
    memory.* 工具与 web 应用全部不可用。
    """

    from memory import HashEmbedding as LibraryHashEmbedding
    from memory import base as memory_base

    for name in ("DASHSCOPE_API_KEY", "HELLOAGENTS_MEMORY_EMBEDDING_API_KEY", "EMBEDDING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    # .env 里可能存有真实 key，测试必须隔离它才能确定性验证降级路径。
    monkeypatch.setattr(memory_base, "load_dotenv_once", lambda: None)

    manager = MemoryManager(MemoryConfig(sqlite_path=":memory:"))
    try:
        assert isinstance(manager.embedding, LibraryHashEmbedding)
        manager.episodic.record("offline memory search works")
        assert manager.search("offline memory search", limit=3)
    finally:
        manager.close()


def test_explicit_embedding_api_key_still_selects_api_embedding(monkeypatch):
    """降级只发生在确实没有 key 时；显式 key 仍走 APIEmbedding。"""

    from memory import APIEmbedding
    from memory import base as memory_base

    monkeypatch.setattr(memory_base, "load_dotenv_once", lambda: None)
    for name in ("DASHSCOPE_API_KEY", "EMBEDDING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    embedding = memory_base.make_default_embedding(
        MemoryConfig(sqlite_path=":memory:", embedding_api_key="explicit-key")
    )
    assert isinstance(embedding, APIEmbedding)


def test_environment_api_key_still_selects_api_embedding(monkeypatch):
    """仅设置 DASHSCOPE_API_KEY 时仍自动升级到真实嵌入（既有行为保持）。"""

    from memory import APIEmbedding
    from memory import base as memory_base

    monkeypatch.setattr(memory_base, "load_dotenv_once", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    embedding = memory_base.make_default_embedding(MemoryConfig(sqlite_path=":memory:"))
    assert isinstance(embedding, APIEmbedding)


def test_embedding_base_url_overrides_dashscope_key(monkeypatch):
    """端口转发网关（EMBEDDING_BASE_URL）优先级最高，即使也给了 DashScope key。

    Web 与 Agent 工具路径共用同一个 ``make_default_embedding``，保证两边嵌入一致。
    """

    from memory import EmbedServerEmbedding
    from memory import base as memory_base

    monkeypatch.setattr(memory_base, "load_dotenv_once", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:10800")

    embedding = memory_base.make_default_embedding(MemoryConfig(sqlite_path=":memory:"))
    assert isinstance(embedding, EmbedServerEmbedding)
    assert embedding.base_url == "http://127.0.0.1:10800"
    assert embedding.dimension == 1024


def test_only_in_memory_vector_store_is_rebuilt_at_startup(tmp_path):
    """定向重建：远程/持久向量库不应在每次构造 manager 时被全量重灌。"""

    from memory.storage import BaseVectorStore, InMemoryVectorStore

    class CountingVectorStore(BaseVectorStore):
        def __init__(self) -> None:
            self.upserts: list[str] = []

        def upsert(self, item):
            self.upserts.append(item.id)

        def delete(self, item_id: str) -> bool:
            return False

        def search(self, vector, **kwargs):
            return []

        def clear(self) -> None:
            return None

    path = tmp_path / "persisted.sqlite3"
    first = MemoryManager(
        MemoryConfig(sqlite_path=str(path)), embedding=HashEmbedding()
    )
    first.episodic.record("persisted item")
    first.close()

    counting = CountingVectorStore()
    MemoryManager(
        MemoryConfig(sqlite_path=str(path)),
        embedding=HashEmbedding(),
        vector_store=counting,
    )
    assert counting.upserts == []

    local = MemoryManager(
        MemoryConfig(sqlite_path=str(path)),
        embedding=HashEmbedding(),
        vector_store=InMemoryVectorStore(),
    )
    assert local.search("persisted item", limit=3), "本地内存索引必须从 SQLite 恢复"


def test_from_env_treats_blank_values_as_unset(monkeypatch):
    """回归：.env 里写 ``DASHSCOPE_API_KEY=`` / ``VAR=`` 不能让 from_env 崩掉。

    Phase 0 把 from_env 接进 get_manager/build_default_manager 后暴露的既有 bug：
    空字符串被当成「配了空 key」，在 __post_init__ 校验处抛 ValueError，
    导致 Qdrant/Neo4j 开关的装配路径整体不可用。
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_API_KEY", "")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_MODEL", "")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_QDRANT_URL", "")

    config = MemoryConfig.from_env()

    assert config.embedding_api_key is None
    assert config.embedding_model  # 回落到类默认值，而不是空串
    assert config.qdrant_url in (None, "")  # 空 URL 表示「不启用 Qdrant」


# ---------------------------------------------------------------------------
# 方案 §P0 验收（F3 激活路径）：开关环境变量一旦设置，MemoryManager 必须真的
# 换用对应存储，而不是继续用内存回退。两条用例只做构造期断言：
# QdrantClient/neo4j.Driver 的构造都是懒连接，不会触网；SQLite 路径钉到 tmp。
# ---------------------------------------------------------------------------


def test_manager_from_env_selects_qdrant_when_url_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HELLOAGENTS_MEMORY_SQLITE_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setenv("HELLOAGENTS_MEMORY_QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")   # 空即未配置（P0 约定），嵌入回落 Hash

    manager = MemoryManager(MemoryConfig.from_env())

    try:
        from memory.storage.qdrant import QdrantVectorStore

        assert type(manager.vector_store) is QdrantVectorStore
        assert manager.vector_store.collection_name == "helloagents_memory"
        assert manager.vector_store.dimension == manager.config.embedding_dimension
    finally:
        manager.close()


def test_manager_from_env_selects_neo4j_when_uri_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HELLOAGENTS_MEMORY_SQLITE_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setenv("HELLOAGENTS_MEMORY_NEO4J_URI", "bolt://127.0.0.1:7687")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_NEO4J_PASSWORD", "pw")

    manager = MemoryManager(MemoryConfig.from_env())

    try:
        assert manager.graph_store.driver is not None   # 不再是内存回退（driver 为 None）
        # 未配置 URI 的对照组：内存回退，driver 必须是 None
        fallback = MemoryManager(MemoryConfig(sqlite_path=":memory:"), embedding=HashEmbedding())
        try:
            assert fallback.graph_store.driver is None
        finally:
            fallback.close()
    finally:
        manager.close()
