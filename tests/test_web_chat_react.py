"""End-to-end regression for the real /api/chat agent path.

``tests/test_web_api.py`` drives ``/api/chat`` with a FakeAgent, so nothing
covered the pieces that actually run in production: the real assembler
(``web.support.get_agent``), the text ReAct protocol, the real memory tools and
the write-confirmation gate.  That blind spot is how a ``RagSearchTool``
NameError survived until a lint pass (see update log 88).

The model here is a scripted ``complete`` callable, so the whole chain is
deterministic and offline: no API key, no sockets.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from web.app import create_app

SCRIPT = [
    (
        "Thought: 先检索知识库\n"
        "Action: memory.rag_search\n"
        'Action Input: {"action": "graph_retrieve", "query": "GraphRAG"}\n'
    ),
    (
        "Thought: 再看记忆层\n"
        "Action: memory.query\n"
        'Action Input: {"action": "search", "query": "GraphRAG"}\n'
    ),
    (
        "Thought: 记录这次验证\n"
        "Action: memory.add\n"
        'Action Input: {"content": "端到端回归：模型经 memory.add 成功落库", '
        '"memory_type": "episodic", "importance": 0.7}\n'
    ),
    "Thought: 已获得证据\nFinal Answer: 已按知识库回答并记录。\n",
]


class ScriptedReActLLM:
    """Minimal stand-in matching the shape ``ReActAgent`` accepts."""

    model = "scripted-test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict]] = []

    def complete(self, messages, **_options):
        self.requests.append([dict(message) for message in messages])
        self.calls += 1
        return SCRIPT[min(self.calls, len(SCRIPT)) - 1]

    def assistant_text(self) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for request in self.requests
            for message in request
            if message.get("role") == "assistant"
        )

    def observations(self) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for request in self.requests
            for message in request
            if message.get("role") == "user"
        )


@contextmanager
def _chat_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch, *, confirm_add: bool
) -> Iterator[tuple[TestClient, object, ScriptedReActLLM]]:
    """A TestClient whose app, agent and tools all share one temp memory 库."""

    support = importlib.import_module("web.support")
    db_path = tmp_path / "chat-agent.sqlite3"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db_path))
    # 嵌入选型只认 config/services.toml（conftest 已指向空配置）：无云端配置
    # 时 make_default_embedding 确定回落 HashEmbedding，memory.* 工具不出网。
    monkeypatch.setattr(support, "DB_PATH", db_path)
    for name in ("_manager", "_pipeline", "_agent"):
        monkeypatch.setattr(support, name, None)

    llm = ScriptedReActLLM()
    agent = support.get_agent()
    agent.llm = llm
    manager = support.get_manager()
    app = create_app(manager=manager)
    # 抽取是第二次模型往返，走真实 provider 配置；测试里必须断开，
    # 否则本机存在 key 的环境会真的发网络请求。
    monkeypatch.setattr("web.app.schedule_qa_extraction", lambda *a, **k: None)
    monkeypatch.setattr("web.app.chat_ready", lambda: (True, ""))
    if not confirm_add:
        # /api/chat 直接绑定了该函数名，必须打在 web.app 上。
        monkeypatch.setattr(
            "web.app.chat_confirmed_side_effects", lambda agent: frozenset()
        )
    try:
        with TestClient(app) as client:
            yield client, manager, llm
    finally:
        support.close_manager()
        for name in ("_agent", "_pipeline"):
            monkeypatch.setattr(support, name, None)


@pytest.fixture()
def real_chat_client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    with _chat_env(tmp_path, monkeypatch, confirm_add=True) as env:
        yield env


def test_real_agent_chat_runs_the_memory_tools_and_persists_add(real_chat_client) -> None:
    client, manager, llm = real_chat_client

    response = client.post(
        "/api/chat",
        json={"message": "先检索知识库，然后记住这次端到端回归验证。"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"] == "已按知识库回答并记录。"
    assert payload["mode"] == "offline"

    # 三个真实工具各跑一次，且模型拿回的 Observation 都是成功结果。
    assert llm.calls == 4
    assistant_text = llm.assistant_text()
    for action in ("memory.rag_search", "memory.query", "memory.add"):
        assert f"Action: {action}" in assistant_text
    observations = llm.observations()
    assert '"ok": true' in observations
    assert '"action": "graph_retrieve"' in observations
    # 只读检索不需要写确认；memory.add 在 /api/chat 注入的确认下必须放行。
    assert "CONFIRMATION_REQUIRED" not in observations

    # memory.add 真的落到记忆库，而不是只通过协议返回。
    stored = manager.search("端到端回归", limit=5)
    assert any("端到端回归" in result.item.content for result in stored)

    # 问答留痕仍走 episodic（record_qa 未被本次改动绕过）。
    qa_items = [
        item
        for item in manager.list(memory_type="episodic")
        if item.metadata.get("kind") == "qa"
    ]
    assert qa_items and qa_items[0].metadata["question"].startswith("先检索知识库")


def test_real_chat_destructive_confirmation_round_trip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _chat_env(tmp_path, monkeypatch, confirm_add=True) as (client, manager, _llm):
        item = manager.semantic.add_fact("待删主体", "待删关系", "待删客体", confidence=0.9)
        action = (
            "Thought: 删除需要确认\n"
            "Action: memory.manage\n"
            f'Action Input: {{"action": "delete", "memory_type": "semantic", "item_id": "{item.id}"}}\n'
        )
        agent = importlib.import_module("web.support").get_agent()
        agent.llm = ScriptedReActLLM()
        agent.llm.complete = lambda messages, **options: (
            action if agent.llm.calls in {0, 2} else "Final Answer: 已处理确认流程。"
        )
        original_complete = agent.llm.complete

        def counted(messages, **options):
            result = original_complete(messages, **options)
            agent.llm.calls += 1
            return result

        agent.llm.complete = counted

        first = client.post("/api/chat", json={"message": "删除这条明确指定的记忆"})
        pending = first.json()["confirmations"]
        assert first.status_code == 200
        assert pending == [{
            "tool_name": "memory.manage",
            "arguments": {
                "action": "delete",
                "memory_type": "semantic",
                "item_id": item.id,
            },
        }]
        assert manager.get(item.id) is not None

        second = client.post(
            "/api/chat",
            json={
                "message": "删除这条明确指定的记忆",
                "confirmation": pending[0],
            },
        )
        assert second.status_code == 200
        assert second.json()["confirmations"] == []
        assert manager.get(item.id) is None
        assert manager.semantic.graph_store.get_relations("待删主体") == []


def test_chat_denies_memory_add_when_the_endpoint_stops_confirming(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """反向对照：去掉 /api/chat 的写确认后，memory.add 必须被拒且不落库。"""

    with _chat_env(tmp_path, monkeypatch, confirm_add=False) as (client, manager, llm):
        response = client.post("/api/chat", json={"message": "记住这次验证"})

    assert response.status_code == 200
    observations = llm.observations()
    assert "CONFIRMATION_REQUIRED" in observations
    # 被拒的写入不得出现在任何记忆层。
    assert not [
        item
        for item in manager.list(memory_type="episodic")
        if "端到端回归" in item.content
    ]
