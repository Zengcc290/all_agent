from datetime import datetime, timedelta, timezone

from memory import MemoryConfig, MemoryManager, MemoryType, Neo4jGraphStore
from memory.storage import SQLiteDocumentStore

from conftest import HashEmbedding


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
    expired.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
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
