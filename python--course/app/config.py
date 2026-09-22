"""全局配置：从 .env / 环境变量读取，字段与 .example.env 一一对应。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")          # 没有 .env 也不会报错
load_dotenv(ROOT / ".example.env")  # 兜底：直接用模板里的默认值也能跑起来


def _s(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _i(key: str, default: int) -> int:
    try:
        return int(_s(key) or default)
    except ValueError:
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key) or default)
    except ValueError:
        return default


def _b(key: str, default: bool) -> bool:
    v = _s(key).lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class LLMConfig:
    model: str = field(default_factory=lambda: _s("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    api_key: str = field(default_factory=lambda: _s("LLM_API_KEY"))
    base_url: str = field(default_factory=lambda: _s("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"))
    temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.1))
    stream: bool = field(default_factory=lambda: _b("LLM_STREAM", True))
    timeout: float = field(default_factory=lambda: _f("LLM_TIMEOUT", 120))
    max_tokens: int = field(default_factory=lambda: _i("LLM_MAX_TOKENS", 2048))

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str = field(default_factory=lambda: _s("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"))
    model: str = field(default_factory=lambda: _s("EMBEDDING_MODEL", "BAAI/bge-m3"))
    api_key: str = field(default_factory=lambda: _s("EMBEDDING_API_KEY"))
    dim: int = field(default_factory=lambda: _i("EMBEDDING_DIM", 1024))
    batch_size: int = field(default_factory=lambda: _i("EMBEDDING_BATCH_SIZE", 32))
    timeout: float = field(default_factory=lambda: _f("EMBEDDING_TIMEOUT", 60))

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/embeddings"


@dataclass(frozen=True)
class QdrantConfig:
    host: str = field(default_factory=lambda: _s("QDRANT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _i("QDRANT_PORT", 6333))
    grpc_port: int = field(default_factory=lambda: _i("QDRANT_GRPC_PORT", 6334))
    prefer_grpc: bool = field(default_factory=lambda: _b("QDRANT_PREFER_GRPC", False))
    collection: str = field(default_factory=lambda: _s("QDRANT_COLLECTION", "kg_chunks"))
    api_key: str = field(default_factory=lambda: _s("QDRANT_API_KEY")) or None
    distance: str = field(default_factory=lambda: _s("QDRANT_DISTANCE", "cosine").lower())
    auto_create: bool = field(default_factory=lambda: _b("QDRANT_AUTO_CREATE", True))
    # 嵌入式本地模式：填了就走 qdrant-client 的 path 模式（qdrant 引擎跑在本进程内），
    # 留空则连接 QDRANT_HOST:QDRANT_PORT 上的 qdrant 服务。
    local_path: str = field(default_factory=lambda: _s("QDRANT_LOCAL_PATH"))

    @property
    def mode(self) -> str:
        return "local" if self.local_path else "server"


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = field(default_factory=lambda: _s("NEO4J_URI", "bolt://127.0.0.1:7687"))
    user: str = field(default_factory=lambda: _s("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: _s("NEO4J_PASSWORD", "neo4j"))
    database: str = field(default_factory=lambda: _s("NEO4J_DATABASE", "neo4j"))
    timeout: float = field(default_factory=lambda: _f("NEO4J_TIMEOUT", 30))


@dataclass(frozen=True)
class AppConfig:
    host: str = field(default_factory=lambda: _s("APP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _i("APP_PORT", 8000))
    cors: str = field(default_factory=lambda: _s("CORS_ORIGINS", "*"))
    sqlite_path: str = field(default_factory=lambda: _s("SQLITE_PATH", "./data/kg.db"))
    max_concurrency: int = field(default_factory=lambda: _i("TOOL_MAX_CONCURRENCY", 8))

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors.split(",") if o.strip()] or ["*"]

    @property
    def sqlite_file(self) -> Path:
        p = Path(self.sqlite_path)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


llm = LLMConfig()
embedding = EmbeddingConfig()
qdrant = QdrantConfig()
neo4j = Neo4jConfig()
app = AppConfig()
