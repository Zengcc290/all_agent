"""Unit tests for DocumentRepository (documents/chunks source of truth, P1)."""

from __future__ import annotations

import pytest

from memory.storage.document_repo import ChunkRecord, DocumentRecord, DocumentRepository


@pytest.fixture
def repo(tmp_path):
    repository = DocumentRepository(tmp_path / "memory.sqlite3")
    yield repository
    repository.close()


def make_document(document_id: str, **overrides) -> DocumentRecord:
    fields = {
        "document_id": document_id,
        "title": f"标题-{document_id}",
        "raw_text": f"正文 {document_id}",
        "source": f"{document_id}.pdf",
        "tags": ["默认"],
        "permission": "private",
        "status": "parsed",
    }
    fields.update(overrides)
    return DocumentRecord(**fields)


def make_chunk(chunk_id: str, document_id: str, index: int, **overrides) -> ChunkRecord:
    fields = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": index,
        "char_start": index * 10,
        "char_end": index * 10 + 10,
        "text": f"分块 {chunk_id}",
    }
    fields.update(overrides)
    return ChunkRecord(**fields)


def test_upsert_document_idempotent(repo: DocumentRepository):
    repo.upsert_document(make_document("d1", title="旧标题"))
    repo.upsert_document(make_document("d1", title="新标题"))

    items, total = repo.list_documents()
    assert total == 1
    assert len(items) == 1
    assert items[0].title == "新标题"
    assert repo.count_documents() == 1


def test_upsert_keeps_original_created_at(repo: DocumentRepository):
    repo.upsert_document(make_document("d1"))
    first = repo.get_document("d1")
    repo.upsert_document(make_document("d1", title="改过"))
    second = repo.get_document("d1")

    assert first is not None and second is not None
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_raw_text_and_tags_round_trip(repo: DocumentRepository):
    repo.upsert_document(
        make_document("d1", raw_text="归一化后的 全文\n换行", tags=["权限A", "课程设计"], permission="shared")
    )
    stored = repo.get_document("d1")

    assert stored is not None
    assert stored.raw_text == "归一化后的 全文\n换行"
    assert stored.tags == ["权限A", "课程设计"]
    assert stored.permission == "shared"


def test_list_with_tag_filter(repo: DocumentRepository):
    repo.upsert_document(make_document("d1", tags=["权限A"]))
    repo.upsert_document(make_document("d2", tags=["权限B"]))
    repo.upsert_document(make_document("d3", tags=["权限B", "权限C"]))

    items, total = repo.list_documents(tag="权限A")

    assert total == 1
    assert [item.document_id for item in items] == ["d1"]


def test_list_with_status_filter(repo: DocumentRepository):
    repo.upsert_document(make_document("d1", status="parsed"))
    repo.upsert_document(make_document("d2", status="failed", error="解析失败"))

    items, total = repo.list_documents(status="failed")

    assert total == 1
    assert items[0].document_id == "d2"
    assert items[0].error == "解析失败"


def test_list_pagination_and_total(repo: DocumentRepository):
    for index in range(3):
        repo.upsert_document(make_document(f"d{index}"))

    first_page, total = repo.list_documents(page=1, page_size=2)
    second_page, second_total = repo.list_documents(page=2, page_size=2)
    beyond, beyond_total = repo.list_documents(page=9, page_size=2)

    assert total == second_total == beyond_total == 3
    assert len(first_page) == 2
    assert len(second_page) == 1
    assert beyond == []


@pytest.mark.parametrize("page,page_size", [(0, 20), (1, 0), (True, 20)])
def test_pagination_rejects_bad_arguments(repo: DocumentRepository, page, page_size):
    with pytest.raises(ValueError):
        repo.list_documents(page=page, page_size=page_size)


def test_status_transition_persists_error(repo: DocumentRepository):
    repo.upsert_document(make_document("d1", status="parsed"))
    repo.set_status("d1", "vectorized")
    repo.set_status("d1", "failed", error="网关不可达")

    stored = repo.get_document("d1")
    assert stored is not None
    assert stored.status == "failed"
    assert stored.error == "网关不可达"


def test_status_machine_rejects_unknown_value(repo: DocumentRepository):
    with pytest.raises(ValueError, match="status"):
        repo.upsert_document(make_document("d1", status="不存在的状态"))
    with pytest.raises(ValueError, match="status"):
        repo.set_status("d1", "不存在的状态")
    with pytest.raises(ValueError, match="permission"):
        repo.upsert_document(make_document("d2", permission="secret"))


def test_delete_document_cascades_chunks(repo: DocumentRepository):
    repo.upsert_document(make_document("d1"))
    repo.upsert_chunks([make_chunk(f"d1:{index}", "d1", index) for index in range(3)])

    removed = repo.delete_document("d1")

    assert removed == 3
    assert repo.get_document("d1") is None
    assert repo.get_chunk("d1:0") is None
    assert repo.stats()["chunks"] == 0


def test_chunk_upsert_is_idempotent_and_ordered(repo: DocumentRepository):
    repo.upsert_document(make_document("d1"))
    repo.upsert_chunks([make_chunk("d1:1", "d1", 1), make_chunk("d1:0", "d1", 0)])
    repo.upsert_chunk(make_chunk("d1:1", "d1", 1, text="改写后的分块"))

    chunks = repo.list_chunks("d1")

    assert [chunk.chunk_id for chunk in chunks] == ["d1:0", "d1:1"]
    assert chunks[1].text == "改写后的分块"
    assert repo.stats()["chunks"] == 2


def test_chunk_vector_status_transitions(repo: DocumentRepository):
    repo.upsert_document(make_document("d1"))
    repo.upsert_chunk(make_chunk("d1:0", "d1", 0))
    assert repo.get_chunk("d1:0").vector_status == "pending"

    repo.set_chunk_vector_status("d1:0", "indexed")
    assert repo.get_chunk("d1:0").vector_status == "indexed"

    with pytest.raises(ValueError, match="vector_status"):
        repo.set_chunk_vector_status("d1:0", "done")


def test_stats_counts(repo: DocumentRepository):
    repo.upsert_document(make_document("d1"))
    repo.upsert_document(make_document("d2"))
    repo.upsert_chunks([make_chunk(f"d1:{index}", "d1", index) for index in range(3)])
    repo.upsert_chunks([make_chunk(f"d2:{index}", "d2", index) for index in range(2)])
    for index in range(3):
        repo.set_chunk_vector_status(f"d1:{index}", "indexed")

    assert repo.stats() == {"documents": 2, "chunks": 5, "chunks_indexed": 3}
