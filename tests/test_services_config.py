"""config/services.toml 集中配置加载器与各消费方接线测试。"""

from pathlib import Path

import pytest

import core.services_config as services_config_module
from core.services_config import load_services_config
from memory import MemoryConfig


def write_services(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


FULL_SERVICES = """\
[embedding]
provider = "openai"
base_url = "https://api.siliconflow.cn/v1"
model = "Qwen/Qwen3-VL-Embedding-0.6B"
api_key_env = "TEST_EMBED_KEY"
dimension = 1024
batch_size = 8
timeout = 15.0

[vision]
model = "Qwen/Qwen2.5-VL-72B-Instruct"

[search]
base_url = "https://search.example.test/v1"
api_key = "plaintext-search-key"
timeout = 3.0

[qdrant]
url = "https://xxxx.us-east-1-0.aws.cloud.qdrant.io:6333"
api_key_env = "TEST_QDRANT_KEY"
collection = "my_collection"

[neo4j]
uri = "bolt+s://xxxx.databases.neo4j.io"
username = "neo4j"
password = "plaintext-neo4j-password"
"""


def test_loads_all_sections_with_secret_resolution(tmp_path, monkeypatch):
    config = tmp_path / "services.toml"
    write_services(config, FULL_SERVICES)
    monkeypatch.setenv("TEST_EMBED_KEY", "embed-secret")
    monkeypatch.setenv("TEST_QDRANT_KEY", "qdrant-secret")

    services = load_services_config(config)

    assert services.embedding.provider == "openai"
    assert services.embedding.base_url == "https://api.siliconflow.cn/v1"
    assert services.embedding.model == "Qwen/Qwen3-VL-Embedding-0.6B"
    assert services.embedding.api_key == "embed-secret"
    assert services.embedding.dimension == 1024
    assert services.embedding.batch_size == 8
    assert services.embedding.timeout == 15.0

    assert services.vision.model == "Qwen/Qwen2.5-VL-72B-Instruct"

    assert services.search.base_url == "https://search.example.test/v1"
    assert services.search.api_key == "plaintext-search-key"
    assert services.search.timeout == 3.0

    assert services.qdrant.url.startswith("https://xxxx.us-east-1-0.aws")
    assert services.qdrant.api_key == "qdrant-secret"
    assert services.qdrant.collection == "my_collection"

    assert services.neo4j.uri.startswith("bolt+s://")
    assert services.neo4j.username == "neo4j"
    assert services.neo4j.password == "plaintext-neo4j-password"
    assert services.proxy.url is None

    assert services.configured


def test_plaintext_api_key_wins_over_env_indirection(tmp_path, monkeypatch):
    config = tmp_path / "services.toml"
    write_services(
        config,
        """\
[search]
base_url = "https://search.example.test/v1"
api_key = "direct"
api_key_env = "TEST_SEARCH_KEY"
""",
    )
    monkeypatch.setenv("TEST_SEARCH_KEY", "from-env")

    services = load_services_config(config)

    assert services.search.api_key == "direct"


def test_missing_file_yields_empty_config(tmp_path):
    services = load_services_config(tmp_path / "does-not-exist.toml")

    assert not services.configured
    assert services.embedding.api_key is None
    assert services.qdrant.url is None
    assert services.neo4j.uri is None


def test_blank_values_are_treated_as_unset(tmp_path):
    config = tmp_path / "services.toml"
    write_services(
        config,
        """\
[embedding]
provider = ""

[vision]
model = ""

[search]
base_url = ""

[qdrant]
url = ""
api_key_env = ""

[neo4j]
uri = ""
username = ""
""",
    )

    services = load_services_config(config)

    assert services.embedding.provider is None
    assert services.vision.model is None
    assert services.search.base_url is None
    assert services.qdrant.url is None
    assert services.neo4j.uri is None
    assert not services.configured


def test_invalid_toml_raises(tmp_path):
    config = tmp_path / "services.toml"
    config.write_text("this is [ not toml", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid services TOML"):
        load_services_config(config)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            '[embedding]\nbase_url = "ftp://example.test/v1"\n',
            "scheme",
        ),
        (
            '[embedding]\nprovider = "sparkles"\n',
            "provider",
        ),
        (
            '[embedding]\ndimension = -3\n',
            "positive integer",
        ),
        (
            '[search]\ntimeout = 0\n',
            "positive number",
        ),
        (
            '[qdrant]\nurl = "not-a-url"\n',
            "scheme",
        ),
        (
            '[proxy]\nurl = "socks5://127.0.0.1:7890"\n',
            "scheme",
        ),
    ],
)
def test_invalid_section_values_raise(tmp_path, body: str, match: str):
    config = tmp_path / "services.toml"
    write_services(config, body)

    with pytest.raises(ValueError, match=match):
        load_services_config(config)


def test_neo4j_accepts_bolt_scheme_but_rejects_others(tmp_path):
    config = tmp_path / "services.toml"
    write_services(
        config,
        '[neo4j]\nuri = "bolt+s://xxxx.databases.neo4j.io"\nusername = "neo4j"\n',
    )
    assert load_services_config(config).neo4j.uri.startswith("bolt+s://")

    write_services(config, '[neo4j]\nuri = "smb://files.example.test"\n')
    with pytest.raises(ValueError, match="scheme"):
        load_services_config(config)


# ---------------------------------------------------------------------------
# 消费方接线：MemoryConfig.from_config / SearchTool / web.search_available
# ---------------------------------------------------------------------------


def write_and_point(path: Path, body: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_services(path, body)
    monkeypatch.setattr(services_config_module, "default_config_path", lambda: path)
    return path


def test_from_config_merges_services_toml(tmp_path, monkeypatch):
    config = tmp_path / "services.toml"
    write_and_point(
        config,
        """\
[embedding]
provider = "openai"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen3-embedding-0.6b"
api_key_env = "TEST_EMBED_KEY"
dimension = 1024

[qdrant]
url = "https://xxxx.qdrant.tech"
api_key_env = "TEST_QDRANT_KEY"
collection = "cloud_collection"

[neo4j]
uri = "bolt+s://xxxx.databases.neo4j.io"
username = "neo4j"
password_env = "TEST_NEO4J_PASSWORD"
""",
        monkeypatch,
    )
    monkeypatch.setenv("TEST_EMBED_KEY", "embed-secret")
    monkeypatch.setenv("TEST_QDRANT_KEY", "qdrant-secret")
    monkeypatch.setenv("TEST_NEO4J_PASSWORD", "neo4j-secret")

    config_obj = MemoryConfig.from_config()

    assert config_obj.embedding_provider == "openai"
    assert config_obj.embedding_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config_obj.embedding_model == "qwen3-embedding-0.6b"
    assert config_obj.embedding_api_key == "embed-secret"
    assert config_obj.embedding_dimension == 1024
    assert config_obj.qdrant_url == "https://xxxx.qdrant.tech"
    assert config_obj.qdrant_api_key == "qdrant-secret"
    assert config_obj.qdrant_collection == "cloud_collection"
    assert config_obj.neo4j_uri.startswith("bolt+s://")
    assert config_obj.neo4j_username == "neo4j"
    assert config_obj.neo4j_password == "neo4j-secret"
    assert config_obj.proxy_url is None


def test_from_config_ignores_environment_entirely(tmp_path, monkeypatch):
    """环境变量不再是配置层：即使设了 HELLOAGENTS_MEMORY_*，services.toml 说了算。"""

    config = tmp_path / "services.toml"
    write_and_point(
        config,
        """\
[embedding]
provider = "openai"
model = "Qwen/Qwen3-Embedding-0.6B"

[qdrant]
url = "https://toml.example.test"
""",
        monkeypatch,
    )
    monkeypatch.setenv("HELLOAGENTS_MEMORY_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("HELLOAGENTS_MEMORY_QDRANT_URL", "http://127.0.0.1:6333")

    config_obj = MemoryConfig.from_config()

    assert config_obj.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert config_obj.embedding_provider == "openai"
    assert config_obj.qdrant_url == "https://toml.example.test"


def test_search_tool_falls_back_to_services_toml(tmp_path, monkeypatch):
    from tool.search import SearchTool

    config = tmp_path / "services.toml"
    write_and_point(
        config,
        """\
[search]
base_url = "https://search.example.test/v1"
api_key_env = "TEST_SEARCH_KEY"
timeout = 2.5
""",
        monkeypatch,
    )
    for name in ("SEARCH_BASE_URL", "ANYSEARCH_BASE_URL", "SEARCH_API", "SEARCH_API_KEY", "ANYSEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEST_SEARCH_KEY", "search-secret")

    tool = SearchTool()

    assert tool.base_url == "https://search.example.test/v1"
    assert tool.api_key == "search-secret"
    assert tool.timeout == 2.5


def test_search_tool_ignores_environment_sources(tmp_path, monkeypatch):
    """SEARCH_*/ANYSEARCH_* 环境变量入口已删除：services.toml 是唯一来源。"""

    from tool.search import SearchTool

    config = tmp_path / "services.toml"
    write_and_point(
        config,
        '[search]\nbase_url = "https://toml.example.test/v1"\napi_key = "toml-key"\n',
        monkeypatch,
    )
    monkeypatch.setenv("SEARCH_BASE_URL", "https://env.example.test/v1")

    tool = SearchTool()

    assert tool.base_url == "https://toml.example.test/v1"
    assert tool.api_key == "toml-key"


def test_web_search_available_uses_services_toml(tmp_path, monkeypatch):
    from web.support import search_available

    config = tmp_path / "services.toml"
    write_and_point(
        config,
        '[search]\nbase_url = "https://search.example.test/v1"\napi_key_env = "TEST_SEARCH_KEY"\n',
        monkeypatch,
    )
    for name in ("SEARCH_BASE_URL", "ANYSEARCH_BASE_URL", "SEARCH_API", "SEARCH_API_KEY", "ANYSEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEST_SEARCH_KEY", "search-secret")

    assert search_available() is True


def test_manager_passes_qdrant_api_key_from_services_toml(tmp_path, monkeypatch):
    from memory.manager import MemoryManager

    config = tmp_path / "services.toml"
    write_and_point(
        config,
        '[qdrant]\nurl = "https://xxxx.qdrant.tech"\napi_key_env = "TEST_QDRANT_KEY"\n',
        monkeypatch,
    )
    monkeypatch.setenv("TEST_QDRANT_KEY", "qdrant-secret")

    config_obj = MemoryConfig.from_config()
    config_obj.sqlite_path = str(tmp_path / "memory.sqlite3")
    manager = MemoryManager(config_obj)

    try:
        from memory.storage.qdrant import QdrantVectorStore

        assert type(manager.vector_store) is QdrantVectorStore
        assert manager.config.qdrant_api_key == "qdrant-secret"
    finally:
        manager.close()


def test_from_config_merges_proxy_url_from_services_toml(tmp_path, monkeypatch):
    config = tmp_path / "services.toml"
    write_and_point(config, '[proxy]\nurl = "http://127.0.0.1:7890"\n', monkeypatch)

    config_obj = MemoryConfig.from_config()

    assert config_obj.proxy_url == "http://127.0.0.1:7890"


def test_manager_passes_proxy_url_to_qdrant_and_neo4j(tmp_path, monkeypatch):
    from memory.manager import MemoryManager

    config = tmp_path / "services.toml"
    write_and_point(
        config,
        """\
[qdrant]
url = "https://xxxx.qdrant.tech"
api_key = "qdrant-secret"

[neo4j]
uri = "neo4j+s://xxxx.databases.neo4j.io"
username = "neo4j"
password = "neo4j-secret"

[proxy]
url = "http://127.0.0.1:7890"
""",
        monkeypatch,
    )

    captured: dict[str, object] = {}

    class FakeQdrant:
        def __init__(self, **kwargs):
            captured["qdrant"] = kwargs

    class FakeNeo4j:
        def __init__(self, *args, **kwargs):
            captured["neo4j_args"] = args
            captured["neo4j"] = kwargs

        def close(self) -> None:
            return None

    monkeypatch.setattr("memory.manager.QdrantVectorStore", FakeQdrant)
    monkeypatch.setattr("memory.manager.Neo4jGraphStore", FakeNeo4j)

    config_obj = MemoryConfig.from_config()
    config_obj.sqlite_path = str(tmp_path / "memory.sqlite3")
    manager = MemoryManager(config_obj)
    try:
        assert captured["qdrant"]["proxy_url"] == "http://127.0.0.1:7890"
        assert captured["neo4j"]["proxy_url"] == "http://127.0.0.1:7890"
    finally:
        manager.close()


# ---------------------------------------------------------------------------
# 消费方接线：识图抽取模型统一从 services.toml [vision] 读取
# ---------------------------------------------------------------------------

PROVIDER_TOML = """\
[defaults]
active_profile = "local"

[profiles.local]
adapter = "openai_compatible"
api_url = "http://127.0.0.1:8000/v1"
api_key = "test-key"
default_model = "text-model"
models = ["text-model"]
tool_mode = "native_strict"
"""


def _point_provider_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.providers import ProviderRegistry

    provider = tmp_path / "provider.toml"
    provider.write_text(PROVIDER_TOML, encoding="utf-8")
    monkeypatch.setattr(
        ProviderRegistry, "default_config_path", staticmethod(lambda: provider)
    )


def test_vision_model_comes_from_services_toml(tmp_path, monkeypatch):
    from web.support import build_knowledge_extractor

    _point_provider_at(tmp_path, monkeypatch)
    services = tmp_path / "services.toml"
    write_and_point(
        services, '[vision]\nmodel = "Qwen/Qwen2.5-VL-72B-Instruct"\n', monkeypatch
    )

    extractor = build_knowledge_extractor()

    assert extractor.vision_model == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert extractor.model == "text-model"


def test_vision_model_falls_back_to_provider_default_without_section(tmp_path, monkeypatch):
    from web.support import build_knowledge_extractor

    _point_provider_at(tmp_path, monkeypatch)
    services = tmp_path / "services.toml"
    write_and_point(services, "[search]\ntimeout = 1.0\n", monkeypatch)

    extractor = build_knowledge_extractor()

    assert extractor.vision_model == "text-model"


def test_memory_tool_pipeline_uses_services_toml_vision_model(tmp_path, monkeypatch):
    import tool._memory as memory_tool_module

    _point_provider_at(tmp_path, monkeypatch)
    services = tmp_path / "services.toml"
    write_and_point(
        services, '[vision]\nmodel = "Qwen/Qwen2.5-VL-72B-Instruct"\n', monkeypatch
    )
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))

    pipeline = memory_tool_module.build_default_pipeline()

    assert pipeline.extractor.vision_model == "Qwen/Qwen2.5-VL-72B-Instruct"
    pipeline.close()
