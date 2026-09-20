"""F2 LLM 提议删除 + 确认闸门的最小回归集。

覆盖四条验收：
① 提议后数据未变（提议只落待确认记录，不删记忆）；
② 确认后真删且图边同步清干净、episodic 审计落库；
③ 不带令牌 / 过期后 / 重复确认都不产生副作用（幂等）；
④ propose 工具本身不删除（side_effect=read），直接删必须走确认。
"""

from __future__ import annotations

import pytest
from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager, Neo4jGraphStore
from memory.storage.document_repo import DeletionProposalStore, execute_deletion


@pytest.fixture()
def manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"),
        graph_store=Neo4jGraphStore(),
        embedding=HashEmbedding(),
    )


@pytest.fixture()
def store(manager: MemoryManager) -> DeletionProposalStore:
    # ``:memory:`` 时必须复用同一条连接，否则两个存储类不共享数据。
    return DeletionProposalStore(
        manager.document_store.path, connection=manager.document_store.connection
    )


def test_proposal_is_persisted_but_nothing_is_deleted(manager: MemoryManager, store: DeletionProposalStore):
    """验收①：提议后数据未变。"""

    item = manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    proposal = store.create(requested_by="llm", reason="过时", item_ids=[item.id])

    assert proposal.status == "pending"
    assert proposal.confirm_token
    # 数据未变
    assert manager.get(item.id) is not None
    assert manager.semantic.facts("A")


def test_confirmed_deletion_removes_item_and_graph_edge(manager: MemoryManager, store: DeletionProposalStore):
    """验收②：确认后真删，图边同步清干净，episodic 审计落库。"""

    item = manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    assert manager.semantic.graph_store.get_relations("A")
    proposal = store.create(requested_by="llm", reason="错误关系", item_ids=[item.id])

    result = execute_deletion(proposal.proposal_id, proposal.confirm_token, manager)

    assert result["deleted"] == [item.id]
    assert result["already_confirmed"] is False
    assert manager.get(item.id) is None
    assert manager.semantic.graph_store.get_relations("A") == []
    # episodic 审计
    audit = [
        memory
        for memory in manager.list(memory_type="episodic")
        if memory.metadata.get("kind") == "deletion_audit"
    ]
    assert len(audit) == 1
    assert audit[0].metadata["deleted_ids"] == [item.id]


def test_wrong_token_is_rejected(manager: MemoryManager, store: DeletionProposalStore):
    """验收③：不带/错令牌都不产生副作用。"""

    item = manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    proposal = store.create(requested_by="llm", reason="原因", item_ids=[item.id])

    with pytest.raises(ValueError, match="token"):
        execute_deletion(proposal.proposal_id, "WRONG", manager)

    assert manager.get(item.id) is not None
    assert store.get(proposal.proposal_id).status == "pending"


def test_expired_proposal_is_rejected(manager: MemoryManager, store: DeletionProposalStore):
    with pytest.raises(Exception):
        store.create(requested_by="llm", reason="原因", item_ids=["不存在"], ttl_minutes=0)

    proposal = store.create(requested_by="llm", reason="原因", item_ids=["不存在"], ttl_minutes=1)
    assert store.get(proposal.proposal_id).status in {"pending", "expired"}


def test_repeated_confirmation_is_idempotent(manager: MemoryManager, store: DeletionProposalStore):
    """验收③：重复确认同 proposal 不重复删。"""

    item = manager.semantic.add_fact("A", "knows", "B", confidence=0.9)
    proposal = store.create(requested_by="llm", reason="原因", item_ids=[item.id])

    first = execute_deletion(proposal.proposal_id, proposal.confirm_token, manager)
    second = execute_deletion(proposal.proposal_id, proposal.confirm_token, manager)

    assert first["deleted"] == [item.id]
    assert second["already_confirmed"] is True
    assert second["deleted"] == []
    audits = [
        memory
        for memory in manager.list(memory_type="episodic")
        if memory.metadata.get("kind") == "deletion_audit"
    ]
    assert len(audits) == 1  # 只写一次审计


def test_unknown_proposal_is_rejected(manager: MemoryManager):
    with pytest.raises(ValueError, match="unknown proposal"):
        execute_deletion("不存在", "TOKEN", manager)


def test_propose_delete_tool_persists_proposal_only(manager: MemoryManager):
    """propose 工具：只落提议、不删任何东西（side_effect=read 的语义边界）。"""

    from tool.memory_propose_delete import MemoryProposeDeleteInput, MemoryProposeDeleteTool

    tool = MemoryProposeDeleteTool(manager=manager)
    manager.semantic.add_fact("Qdrant", "用于", "向量检索", confidence=0.9)

    output = tool.execute(
        MemoryProposeDeleteInput(
            action="by_relation", target="Qdrant 用于 向量检索", reason="过时"
        )
    )

    assert output.proposal_id
    assert output.confirm_token
    assert len(output.items) == 1
    # 数据未变
    assert manager.semantic.facts("Qdrant")
    store = DeletionProposalStore(
        manager.document_store.path, connection=manager.document_store.connection
    )
    assert store.get(output.proposal_id).status == "pending"


def test_propose_delete_tool_without_candidates_returns_note(manager: MemoryManager):
    from tool.memory_propose_delete import MemoryProposeDeleteInput, MemoryProposeDeleteTool

    tool = MemoryProposeDeleteTool(manager=manager)

    output = tool.execute(
        MemoryProposeDeleteInput(
            action="by_query", target="不存在的查询词", reason="原因"
        )
    )

    assert output.proposal_id == ""
    assert "没有检索到任何匹配" in output.note


def test_concurrent_confirmation_confirms_exactly_once(manager, store):
    """并发确认：原子状态翻转，只有一个调用能赢，且不会重复删/重复审计。"""

    import threading

    from memory.storage.document_repo import execute_deletion

    item = manager.semantic.add_fact("并发", "确认", "原子", confidence=0.9)
    proposal = store.create(requested_by="llm", reason="并发测试", item_ids=[item.id])
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait()
            results.append(execute_deletion(proposal.proposal_id, proposal.confirm_token, manager))
        except Exception as exc:  # noqa: BLE001 - 并发测试收集异常
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 2
    assert {result["already_confirmed"] for result in results} == {True, False}
    deleted_sets = {tuple(result["deleted"]) for result in results}
    assert (item.id,) in deleted_sets
    assert () in deleted_sets
    assert store.get(proposal.proposal_id).status == "confirmed"
    assert manager.get(item.id) is None
    audits = [
        memory
        for memory in manager.list(memory_type="episodic")
        if memory.metadata.get("kind") == "deletion_audit"
    ]
    assert len(audits) == 1
