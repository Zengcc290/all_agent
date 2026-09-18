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
    manager.working.add("item a", importance=0.1)
    manager.working.add("item b", importance=0.9)
    manager.working.add("item c", importance=0.9)

    # 容量 2：最低重要性的 "item a" 被淘汰，另外两条保留。
    assert sorted(item.content for item in manager.working.list()) == ["item b", "item c"]

    # TTL 单独用一个容量充足的实例，避免容量淘汰掩盖过期判定。
    ttl_manager = _manager(working_memory_capacity=8)
    expired = ttl_manager.working.add("short lived", ttl_seconds=0.01)
    expired.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    ttl_manager.document_store.upsert(expired)
    assert ttl_manager.working.get(expired.id) is None
    assert ttl_manager.working.list() == []


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


def test_memory_manager_without_cloud_config_falls_back_to_offline_embedding():
    """回归：没有云端配置时 MemoryManager() 必须可用（离线哈希降级）。

    历史缺陷：``make_default_embedding`` 无条件构造 ``APIEmbedding``，没有配置
    时构造 MemoryManager 直接抛 RuntimeError，导致离线环境下 memory 包、
    memory.* 工具与 web 应用全部不可用。
    """

    from memory import HashEmbedding as LibraryHashEmbedding

    manager = MemoryManager(MemoryConfig(sqlite_path=":memory:"))
    try:
        assert isinstance(manager.embedding, LibraryHashEmbedding)
        manager.episodic.record("offline memory search works")
        assert manager.search("offline memory search", limit=3)
    finally:
        manager.close()


def test_explicit_cloud_config_selects_api_embedding():
    """降级只发生在云端配置不齐时；端点+密钥配齐就走 APIEmbedding。"""

    from memory import APIEmbedding
    from memory import base as memory_base

    embedding = memory_base.make_default_embedding(
        MemoryConfig(
            sqlite_path=":memory:",
            embedding_base_url="https://api.example.com/v1",
            embedding_api_key="explicit-key",
        )
    )
    assert isinstance(embedding, APIEmbedding)
    assert embedding.base_url == "https://api.example.com/v1"


def test_environment_api_key_alone_is_not_enough(monkeypatch):
    """环境变量不再是配置来源：只设 DASHSCOPE_API_KEY 不配端点 -> 离线兜底。"""

    from memory import HashEmbedding as LibraryHashEmbedding
    from memory import base as memory_base

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

    embedding = memory_base.make_default_embedding(MemoryConfig(sqlite_path=":memory:"))
    assert isinstance(embedding, LibraryHashEmbedding)


def test_provider_hash_forces_offline_even_with_cloud_config():
    """provider = "hash" 强制离线（测试/审计场景的显式开关）。"""

    from memory import HashEmbedding as LibraryHashEmbedding
    from memory import base as memory_base

    embedding = memory_base.make_default_embedding(
        MemoryConfig(
            sqlite_path=":memory:",
            embedding_provider="hash",
            embedding_base_url="https://api.example.com/v1",
            embedding_api_key="key",
        )
    )
    assert isinstance(embedding, LibraryHashEmbedding)


def test_provider_openai_without_config_falls_back_to_hash():
    """provider = "openai" 但配置不齐：回落离线（/api/health 会如实显示 hash）。"""

    from memory import HashEmbedding as LibraryHashEmbedding
    from memory import base as memory_base

    embedding = memory_base.make_default_embedding(
        MemoryConfig(sqlite_path=":memory:", embedding_provider="openai")
    )
    assert isinstance(embedding, LibraryHashEmbedding)


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


def test_from_config_ignores_environment_entirely(monkeypatch):
    """配置只认 services.toml：环境变量（含空值占位）不再参与优先级。"""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_API_KEY", "")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_MODEL", "")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_QDRANT_URL", "")

    config = MemoryConfig.from_config()

    assert config.embedding_api_key is None
    assert config.embedding_model  # 回落到类默认值，而不是空串
    assert config.qdrant_url is None


# ---------------------------------------------------------------------------
# 方案 §P0 验收（F3 激活路径）：开关环境变量一旦设置，MemoryManager 必须真的
# 换用对应存储，而不是继续用内存回退。两条用例只做构造期断言：
# QdrantClient/neo4j.Driver 的构造都是懒连接，不会触网；SQLite 路径钉到 tmp。
# ---------------------------------------------------------------------------


def test_manager_from_config_selects_qdrant_when_url_set(tmp_path, monkeypatch):
    from core import services_config

    monkeypatch.setattr(
        services_config, "default_config_path", lambda: tmp_path / "services.toml"
    )
    (tmp_path / "services.toml").write_text(
        "[qdrant]\nurl = \"http://127.0.0.1:6333\"\n", encoding="utf-8"
    )

    config = MemoryConfig.from_config()
    config.sqlite_path = str(tmp_path / "memory.sqlite3")
    manager = MemoryManager(config)

    try:
        from memory.storage.qdrant import QdrantVectorStore

        assert type(manager.vector_store) is QdrantVectorStore
        assert manager.vector_store.collection_name == "helloagents_memory"
        assert manager.vector_store.dimension is None
    finally:
        manager.close()


def test_manager_from_config_selects_neo4j_when_uri_set(tmp_path, monkeypatch):
    from core import services_config

    monkeypatch.setattr(
        services_config, "default_config_path", lambda: tmp_path / "services.toml"
    )
    (tmp_path / "services.toml").write_text(
        "[neo4j]\nuri = \"bolt://127.0.0.1:7687\"\nusername = \"neo4j\"\npassword = \"pw\"\n",
        encoding="utf-8",
    )

    config = MemoryConfig.from_config()
    config.sqlite_path = str(tmp_path / "memory.sqlite3")
    manager = MemoryManager(config)

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
