"""Regression coverage for temporal n-ary and visual observations."""

from __future__ import annotations

import base64
import io

from conftest import HashEmbedding
from fastapi.testclient import TestClient

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from memory.rag import (
    Document,
    EntityCandidate,
    ExtractionResult,
    GraphRAGPipeline,
    LLMKnowledgeExtractor,
    RAGPipeline,
    RelationCandidate,
    RelationRole,
)
from web import create_app
from web.graph_builder import build_graph

PNG = b"\x89PNG\r\n\x1a\nimage"


def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


def test_same_triple_at_two_times_is_two_observations_and_as_of_selects_latest():
    store = manager()

    class Extractor:
        def __init__(self) -> None:
            self.index = 0

        def extract(self, text, *, metadata=None, graph_context=""):
            self.index += 1
            place = "书桌" if self.index == 1 else "图书馆"
            moment = (
                "2025-01-01T13:00:00+00:00"
                if self.index == 1
                else "2025-01-01T14:00:00+00:00"
            )
            return ExtractionResult(
                domain="设备",
                entities=[
                    EntityCandidate(name="电脑", entity_type="设备"),
                    EntityCandidate(name="项目", entity_type="项目"),
                    EntityCandidate(name=place, entity_type="地点"),
                ],
                relations=[
                    RelationCandidate(
                        subject="电脑",
                        predicate="运行",
                        object="项目",
                        roles=[RelationRole(role="地点", value=place, entity_type="地点")],
                        cardinality="temporal",
                        event_at=moment,
                        confidence=0.95,
                        evidence=f"电脑在{place}运行项目",
                    )
                ],
            )

    pipeline = RAGPipeline(store, extractor=Extractor())
    pipeline.ingest(Document("电脑 13:00 在书桌上跑项目", id="t13"))
    pipeline.ingest(Document("电脑 14:00 在图书馆里跑项目", id="t14"))

    facts = [item for item in store.semantic.facts("电脑") if item.metadata.get("predicate") == "运行"]
    assert len(facts) == 2
    assert len(store.graph_store.graph_snapshot()["observations"]) == 2

    at_13 = store.graph_store.graph_snapshot(at="2025-01-01T13:30:00+00:00")
    at_14 = store.graph_store.graph_snapshot(at="2025-01-01T14:30:00+00:00")
    assert {
        participant["name"]
        for observation in at_13["observations"]
        for participant in observation["participants"]
        if participant["role"] == "地点"
    } == {"书桌"}
    assert {
        participant["name"]
        for observation in at_14["observations"]
        for participant in observation["participants"]
        if participant["role"] == "地点"
    } == {"图书馆"}

    graph_at_13 = build_graph(store, at="2025-01-01T13:30:00+00:00")
    assert graph_at_13["graph_source"] == "inmemory"
    assert any(edge["relation"] == "地点" for edge in graph_at_13["edges"])
    assert not any(
        edge["relation"] == "地点"
        and any(
            node["title"] == "图书馆" and node["id"] == edge["target"]
            for node in graph_at_13["nodes"]
        )
        for edge in graph_at_13["edges"]
    )
    paths_at_13 = GraphRAGPipeline(store).retrieve(
        "电脑", hops=1, limit=5, at="2025-01-01T13:30:00+00:00"
    ).paths
    assert any(path.target == "书桌" for path in paths_at_13)
    assert all(path.target != "图书馆" for path in paths_at_13)
    store.close()


def test_nary_observation_supports_role_edges_and_multihop():
    store = manager()
    store.semantic.add_fact(
        "电脑",
        "运行",
        "项目",
        item_id="observation:nary",
        metadata={
            "domain": "设备",
            "active": True,
            "cardinality": "temporal",
            "event_at": "2025-01-01T13:00:00+00:00",
            "roles": [
                {"role": "地点", "value": "书桌", "entity_type": "地点"},
                {"role": "操作者", "value": "小明", "entity_type": "人员"},
            ],
            "evidence": "电脑在书桌上由小明运行项目",
        },
    )
    snapshot = store.graph_store.graph_snapshot()
    observation = snapshot["observations"][0]
    assert {item["role"] for item in observation["participants"]} == {
        "subject",
        "object",
        "地点",
        "操作者",
    }
    paths = store.graph_store.path_query("项目", "书桌", max_depth=3)
    assert paths
    assert any("电脑" in path["entities"] for path in paths)
    store.close()


def test_visual_extractor_sends_image_to_configured_vision_model():
    calls = []

    def complete(messages, **kwargs):
        calls.append((messages, kwargs))
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"domain":"设备","entities":[],"relations":[],"keywords":[]}'
                    }
                }
            ]
        }

    extractor = LLMKnowledgeExtractor(
        complete, model="text-model", vision_model="Qwen/Qwen2.5-VL-72B-Instruct"
    )
    extractor.extract(
        "电脑照片",
        metadata={"captured_at": "2025-01-01T13:00:00+00:00"},
        image=PNG,
        mime_type="image/png",
    )
    messages, kwargs = calls[0]
    assert kwargs["model"] == "Qwen/Qwen2.5-VL-72B-Instruct"
    image_url = messages[1]["content"][1]["image_url"]["url"]
    assert image_url == "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


def test_graph_snapshot_includes_graph_only_edge():
    store = manager()
    store.graph_store.add_relation("Neo4j实体", "关联", "外部节点")
    graph = build_graph(store)
    assert graph["graph_source"] == "inmemory"
    assert any(edge["relation"] == "关联" for edge in graph["edges"])
    store.close()


def test_visual_ingest_persists_payload_and_calls_visual_extractor():
    store = manager()
    seen = []

    class Extractor:
        def extract(self, text, *, metadata=None, graph_context="", image=None, mime_type=""):
            seen.append((text, metadata, image, mime_type))
            return ExtractionResult()

    pipeline = RAGPipeline(store, extractor=Extractor())
    item = pipeline.ingest_media(
        PNG,
        text="电脑在书桌上",
        mime_type="image/png",
        metadata={"filename": "camera.png", "captured_at": "2025-01-01T13:00:00+00:00"},
    )
    assert item.payload == PNG
    assert item.modality == "image"
    assert seen[0][2] == PNG
    assert seen[0][3] == "image/png"
    assert pipeline.last_ingest_report["modality"] == "image"
    store.close()


def test_image_knowledge_endpoint_stores_camera_payload(monkeypatch):
    store = manager()
    seen = []

    class Extractor:
        def extract(self, text, *, metadata=None, graph_context="", image=None, mime_type=""):
            seen.append((image, mime_type, metadata))
            return ExtractionResult()

    app = create_app(manager=store)
    with TestClient(app) as client:
        app.state.pipeline = RAGPipeline(store, extractor=Extractor())
        response = client.post(
            "/api/knowledge/image",
            files={"file": ("camera.png", io.BytesIO(PNG), "image/png")},
            data={
                "text": "电脑在书桌上",
                "captured_at": "2025-01-01T13:00:00+00:00",
            },
        )
    assert response.status_code == 200, response.text
    assert seen and seen[0][0] == PNG and seen[0][1] == "image/png"
    assert store.list(memory_type="perceptual")[0].payload == PNG
    store.close()
