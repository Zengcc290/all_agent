"""Regression cover for LLM graph patches: aliases, updates, retraction.

The user story under test is multi-step ingestion of one endpoint's state:

1. a relay named ``web`` has a 1 yuan balance
2. the relay serves grok-4.6 and deepseek-v4.1-flash
3. the balance becomes 0
4. deepseek-v4.1-flash is deployed behind a second relay, ``world``
5. deepseek-v4.1-flash belongs to deepseek and is commonly called ds / dsv4.1

After all five steps the graph must expose one current balance edge, one
planet for the model (reachable through its aliases), and one-hop neighbours
for both relays.
"""

from __future__ import annotations

from conftest import HashEmbedding

from memory import MemoryConfig, MemoryManager
from memory.rag import (
    Document,
    EntityCandidate,
    EntityResolver,
    ExtractionResult,
    GraphRAGPipeline,
    RAGPipeline,
    RelationCandidate,
)
from memory.rag.knowledge import build_graph_context


def _manager() -> MemoryManager:
    return MemoryManager(
        MemoryConfig(sqlite_path=":memory:"), embedding=HashEmbedding()
    )


def test_resolver_merges_explicit_alias_and_prefix_without_fuzzy_collisions():
    manager = _manager()
    resolver = EntityResolver(manager)
    resolver.resolve(
        "deepseek-v4.1-flash",
        domain="模型",
        aliases=["ds", "dsv4.1"],
    )
    resolver.resolve("web 中转站", domain="中转站")
    resolver.resolve("world 中转站", domain="中转站")

    # Explicit alias and distinctive prefix both land on the same planet.
    assert resolver.resolve("ds", domain="模型") == "deepseek-v4.1-flash"
    assert resolver.resolve("dsv4.1", domain="模型") == "deepseek-v4.1-flash"
    assert resolver.resolve("deepseek-v4.1-flash", domain="模型") == "deepseek-v4.1-flash"

    # A lookalike name resolved by prefix is recorded as written, not as a
    # normalized key, so prefix matching never widens on its own.
    resolver.resolve("deepseek-v4.1", domain="模型")
    canonical = resolver.match("deepseek-v4.1-flash")
    assert canonical is not None
    assert canonical.metadata["canonical_name"] == "deepseek-v4.1-flash"
    assert canonical.metadata["aliases"] == ["deepseek-v4.1", "ds", "dsv4.1"]

    # A short shared prefix must not merge unrelated entities.
    assert resolver.match("web") is not None
    assert resolver.match("world") is not None
    assert len([item for item in manager.semantic.list() if item.metadata.get("kind") == "entity"]) == 3


def test_supersede_retires_previous_single_valued_value():
    manager = _manager()
    pipeline = RAGPipeline(manager)

    class Step1:
        def extract(self, text, *, metadata=None, graph_context=""):
            return ExtractionResult(
                domain="中转站",
                entities=[EntityCandidate(name="web 中转站", aliases=["web"]), EntityCandidate(name="1 元")],
                relations=[
                    RelationCandidate(
                        subject="web 中转站",
                        predicate="余额",
                        object="1 元",
                        cardinality="single",
                        confidence=0.9,
                        evidence="web 当前有余额 1 元",
                    )
                ],
            )

    pipeline.extractor = Step1()
    pipeline.ingest(
        Document("web 有余额 1 元。", metadata={"filename": "step1.txt"})
    )
    assert pipeline.last_ingest_report["relations"] == 1

    class Extractor:
        def extract(self, text, *, metadata=None, graph_context=""):
            assert "已知实体" in graph_context  # the subgraph is injected
            return ExtractionResult(
                domain="中转站",
                entities=[
                    EntityCandidate(name="web", entity_type="中转站"),
                    EntityCandidate(name="0 元", entity_type="状态"),
                ],
                relations=[
                    RelationCandidate(
                        subject="web",
                        predicate="余额",
                        object="0 元",
                        action="supersede",
                        cardinality="single",
                        confidence=0.95,
                        evidence="当前 web 没有余额了",
                    )
                ],
            )

    pipeline.extractor = Extractor()
    pipeline.ingest(
        Document("当前 web 没有余额了。", metadata={"filename": "step3.txt"})
    )
    report = pipeline.last_ingest_report
    assert report["relations"] == 1
    assert report["superseded"] == 1

    balances = [
        item
        for item in manager.semantic.facts("web 中转站")
        if item.metadata.get("predicate") == "余额"
    ]
    assert len(balances) == 2  # history is retained, never deleted
    current = [item for item in balances if item.metadata.get("active", True)]
    stale = [item for item in balances if item.metadata.get("active") is False]
    assert len(current) == 1 and current[0].metadata["object"] == "0 元"
    assert len(stale) == 1 and stale[0].metadata["object"] == "1 元"
    assert stale[0].metadata["superseded_by"] == current[0].id
    assert current[0].metadata["supersedes"] == [stale[0].id]

    # The retired edge must not carry a retrieval hop.
    result = GraphRAGPipeline(manager).retrieve("web 中转站", limit=5, hops=1)
    assert all(path.target != "1 元" for path in result.paths)
    assert any(path.target == "0 元" for path in result.paths)


def test_retract_keeps_history_without_new_value():
    manager = _manager()
    pipeline = RAGPipeline(manager)

    class Extractor:
        def extract(self, text, *, metadata=None, graph_context=""):
            return ExtractionResult(
                domain="中转站",
                entities=[EntityCandidate(name="web"), EntityCandidate(name="1 元")],
                relations=[
                    RelationCandidate(
                        subject="web",
                        predicate="余额",
                        object="1 元",
                        confidence=0.9,
                        evidence="当前有余额 1 元",
                    )
                ],
            )

    pipeline = RAGPipeline(manager, extractor=Extractor())
    pipeline.ingest(Document("web 当前有余额 1 元。", metadata={"filename": "a.txt"}))
    assert len(manager.semantic.facts("web")) == 1

    class Retractor:
        def extract(self, text, *, metadata=None, graph_context=""):
            return ExtractionResult(
                domain="中转站",
                relations=[
                    RelationCandidate(
                        subject="web",
                        predicate="余额",
                        object="1 元",
                        action="retract",
                        confidence=0.9,
                        evidence="余额信息已作废",
                    )
                ],
            )

    pipeline.extractor = Retractor()
    pipeline.ingest(Document("web 的余额信息作废。", metadata={"filename": "b.txt"}))
    assert pipeline.last_ingest_report["retracted"] == 1

    facts = manager.semantic.facts("web")
    assert len(facts) == 1
    assert facts[0].metadata["active"] is False
    assert GraphRAGPipeline(manager).retrieve("web", limit=5, hops=1).paths == []


class ScriptedExtractor:
    """Deterministic stand-in for the LLM: emits one patch per ingest call."""

    def __init__(self, results: list[ExtractionResult]) -> None:
        self._results = list(results)

    def extract(self, text, *, metadata=None, graph_context=""):
        return self._results.pop(0)


def _step_results() -> list[ExtractionResult]:
    return [
        ExtractionResult(
            domain="AI 中转站",
            entities=[
                EntityCandidate(name="web 中转站", aliases=["web"]),
                EntityCandidate(name="1 元"),
            ],
            relations=[
                RelationCandidate(
                    subject="web 中转站",
                    predicate="余额",
                    object="1 元",
                    cardinality="single",
                    confidence=0.9,
                    evidence="web 当前有余额 1 元",
                )
            ],
        ),
        ExtractionResult(
            domain="AI 中转站",
            entities=[
                EntityCandidate(name="web 中转站", aliases=["web"]),
                EntityCandidate(name="grok-4.6"),
                EntityCandidate(name="deepseek-v4.1-flash", aliases=["ds", "dsv4.1"]),
            ],
            relations=[
                RelationCandidate(
                    subject="web 中转站",
                    predicate="支持",
                    object="grok-4.6",
                    confidence=0.9,
                    evidence="web 里有满血 grok-4.6",
                ),
                RelationCandidate(
                    subject="web 中转站",
                    predicate="支持",
                    object="deepseek-v4.1-flash",
                    confidence=0.9,
                    evidence="web 里有 deepseek-v4.1-flash",
                ),
            ],
        ),
        ExtractionResult(
            domain="AI 中转站",
            entities=[EntityCandidate(name="0 元")],
            relations=[
                RelationCandidate(
                    subject="web 中转站",
                    predicate="余额",
                    object="0 元",
                    action="supersede",
                    cardinality="single",
                    confidence=0.95,
                    evidence="当前 web 没有余额了",
                )
            ],
        ),
        ExtractionResult(
            domain="AI 中转站",
            entities=[EntityCandidate(name="world 中转站", aliases=["world"])],
            relations=[
                RelationCandidate(
                    subject="deepseek-v4.1-flash",
                    predicate="部署于",
                    object="world 中转站",
                    confidence=0.9,
                    evidence="world 也有 deepseek-v4.1-flash",
                )
            ],
        ),
        ExtractionResult(
            domain="AI 中转站",
            entities=[
                EntityCandidate(name="deepseek-v4.1-flash", aliases=["ds", "dsv4.1"])
            ],
            relations=[
                RelationCandidate(
                    subject="deepseek-v4.1-flash",
                    predicate="属于",
                    object="deepseek",
                    confidence=0.9,
                    evidence="deepseek-v4.1-flash 是 deepseek 旗下的",
                )
            ],
        ),
    ]


def test_five_step_endpoint_story_converges_on_one_planet_per_entity():
    manager = _manager()
    pipeline = RAGPipeline(manager, extractor=ScriptedExtractor(_step_results()))

    texts = [
        "web 是一个中转站，当前有余额 1 元。",
        "web 里面有满血 grok-4.6 和 deepseek-v4.1-flash。",
        "当前 web 没有余额了。",
        "world 这个中转站也有 deepseek-v4.1-flash。",
        "deepseek-v4.1-flash 是 deepseek 旗下的，通常叫他 ds、dsv4.1。",
    ]
    for index, text in enumerate(texts, start=1):
        pipeline.ingest(Document(text, metadata={"filename": f"step{index}.txt"}))

    entities = [
        item for item in manager.semantic.list() if item.metadata.get("kind") == "entity"
    ]
    names = {item.metadata["canonical_name"] for item in entities}
    # Exactly one planet per概念. "deepseek" is a word-boundary prefix of
    # "deepseek-v4.1-flash", so the strict prefix rule folds it in as an alias
    # instead of creating a second planet.
    assert names == {
        "web 中转站",
        "1 元",
        "0 元",
        "grok-4.6",
        "deepseek-v4.1-flash",
        "world 中转站",
    }
    model = next(
        item
        for item in entities
        if item.metadata["canonical_name"] == "deepseek-v4.1-flash"
    )
    assert {"ds", "dsv4.1", "deepseek"} <= set(model.metadata["aliases"])

    balances = [
        item
        for item in manager.semantic.facts("web 中转站")
        if item.metadata.get("predicate") == "余额"
    ]
    assert len([item for item in balances if item.metadata.get("active", True)]) == 1

    # Asking by alias finds the model planet plus both relays one hop away.
    result = GraphRAGPipeline(manager).retrieve("dsv4.1 部署在哪", limit=5, hops=1)
    assert "deepseek-v4.1-flash" in result.entities
    targets = {path.target for path in result.paths}
    assert "world 中转站" in targets
    assert "web 中转站" in targets


def test_graph_projection_keeps_history_satellite_but_drops_stale_edge():
    from web.graph_builder import build_graph

    manager = _manager()
    manager.semantic.add_fact(
        "web",
        "余额",
        "1 元",
        metadata={"active": False, "superseded_by": ["relation:x"]},
        confidence=0.9,
        item_id="relation:old",
    )
    manager.semantic.add_fact(
        "web",
        "余额",
        "0 元",
        metadata={"active": True, "cardinality": "single"},
        confidence=0.95,
        item_id="relation:new",
    )

    graph = build_graph(manager)
    assert graph["stats"]["facts"] == 0
    assert graph["stats"]["historical_facts"] == 1
    stale_nodes = [node for node in graph["nodes"] if node["id"] == "relation:old"]
    assert stale_nodes == []
    assert all(node["kind"] != "relation" for node in graph["nodes"])
    titles = {node["id"]: node["title"] for node in graph["nodes"]}
    edges = [edge for edge in graph["edges"] if edge["relation"] == "余额"]
    assert len(edges) == 1
    assert titles[edges[0]["source"]] == "web"
    assert titles[edges[0]["target"]] == "0 元"


def test_manual_and_extracted_facts_share_one_id_scheme():
    """回归：同一个三元组经「手工/种子」与「抽取」两条路径只落一条记录。

    历史缺陷：``SemanticMemory.add_fact`` 默认用可读的 ``fact:s|p|o`` 作 id，
    而抽取链路用 ``relation_id_for`` 哈希 id，同一事实因此存成两行（图上出现
    重复行星与重复边），``/api/import`` 还为此写了一段手工三元组去重。
    """

    from memory.ids import legacy_fact_id_for, relation_id_for

    manager = _manager()
    subject, predicate, object_ = "Alice", "余额", "1 元"

    manual = manager.semantic.add_fact(subject, predicate, object_)
    assert manual.id == relation_id_for(subject, predicate, object_)

    extracted = manager.semantic.add_fact(
        subject,
        predicate,
        object_,
        metadata={"action": "assert", "active": True},
        confidence=0.9,
    )
    assert extracted.id == manual.id
    matching = [
        item
        for item in manager.semantic.facts()
        if item.metadata.get("subject") == subject
        and item.metadata.get("predicate") == predicate
        and item.metadata.get("object") == object_
    ]
    assert len(matching) == 1
    # 抽取路径的 metadata 合并进同一条记录，而不是另起一行。
    assert matching[0].metadata.get("action") == "assert"

    # 存量库兼容：旧 ``fact:s|p|o`` 行仍被就地更新，不产生第二条记录。
    legacy_manager = _manager()
    legacy_id = legacy_fact_id_for("Bob", "余额", "2 元")
    legacy_manager.semantic.add(
        "Bob 余额 2 元",
        metadata={
            "subject": "Bob",
            "predicate": "余额",
            "object": "2 元",
            "active": True,
        },
        item_id=legacy_id,
    )
    updated = legacy_manager.semantic.add_fact(
        "Bob", "余额", "2 元", metadata={"active": False}, confidence=0.9
    )
    assert updated.id == legacy_id
    legacy_rows = [
        item
        for item in legacy_manager.semantic.facts()
        if item.metadata.get("subject") == "Bob"
        and item.metadata.get("predicate") == "余额"
        and item.metadata.get("object") == "2 元"
    ]
    assert len(legacy_rows) == 1
    assert legacy_rows[0].metadata.get("active") is False


def test_qa_extraction_writes_into_the_injected_manager(monkeypatch):
    """后台问答抽取必须使用调用方的库，绝不能去抢全局单例。"""

    from web import support

    manager = _manager()
    monkeypatch.setattr(
        support, "build_knowledge_extractor", lambda: ScriptedExtractor(_step_results()[:2])
    )

    report = support.extract_graph_patches(
        "web 当前有余额 1 元，支持 grok-4.6。",
        "已记录。",
        manager=manager,
    )

    assert report is not None
    assert report["relations"] >= 1
    assert manager.semantic.facts()
    assert support.GRAPH_REVISION >= 1


def test_qa_extraction_is_skipped_without_a_chat_model(monkeypatch):
    from web import support

    called: list[str] = []
    monkeypatch.setattr(support, "chat_ready", lambda: (False, "no key"))
    monkeypatch.setattr(
        support,
        "extract_graph_patches",
        lambda *args, **kwargs: called.append("ran"),
    )

    support.schedule_qa_extraction("q", "a", manager=_manager())
    assert called == []


def test_graph_context_feeds_known_entities_and_trims_whole_lines():
    from memory.rag import build_graph_context

    manager = _manager()
    pipeline = RAGPipeline(manager, extractor=ScriptedExtractor(_step_results()[:2]))
    for index, text in enumerate(["web 中转站当前有余额 1 元", "web 中转站支持 grok-4.6"], 1):
        pipeline.ingest(Document(text, metadata={"filename": f"ctx{index}.txt"}))

    context = build_graph_context(manager, "web 中转站现在的余额是多少")
    assert "已知实体（必须复用，不要新造）" in context
    assert "web 中转站" in context
    assert "已知关系（当前有效）" in context

    # A tight budget must drop whole lines instead of cutting one in half.
    squeezed = build_graph_context(manager, "web 中转站现在的余额是多少", max_chars=60)
    assert squeezed
    assert all(
        line.endswith("：") or line.startswith("- ") or "--" in line
        for line in squeezed.splitlines()
    )


def test_supersede_retires_every_active_value_in_the_slot():
    """回归：单值槽内有多个 active 旧值时，supersede 必须全部退役。

    历史缺陷：``materialize_extraction`` 的 ``known_items`` 用 ``setdefault``
    让每个 (subject, predicate) 槽只缓存一条，``facts_for`` 命中即返回该条，
    槽内其余 active 旧值永远不会进入退役循环，图里因此同时存在多个「当前值」。
    """

    manager = _manager()
    pipeline = RAGPipeline(manager)

    # 直接构造历史脏数据：同一单值槽两个 active 值。
    manager.semantic.add_fact(
        "web 中转站", "余额", "1 元", metadata={"active": True}, confidence=0.9
    )
    manager.semantic.add_fact(
        "web 中转站", "余额", "2 元", metadata={"active": True}, confidence=0.9
    )

    class Extractor:
        def extract(self, text, *, metadata=None, graph_context=""):
            return ExtractionResult(
                domain="中转站",
                entities=[
                    EntityCandidate(name="web 中转站", entity_type="中转站"),
                    EntityCandidate(name="0 元", entity_type="状态"),
                ],
                relations=[
                    RelationCandidate(
                        subject="web 中转站",
                        predicate="余额",
                        object="0 元",
                        action="supersede",
                        cardinality="single",
                        confidence=0.95,
                        evidence="当前 web 没有余额了",
                    )
                ],
            )

    pipeline.extractor = Extractor()
    pipeline.ingest(
        Document("当前 web 没有余额了。", metadata={"filename": "multi-active.txt"})
    )

    active = sorted(
        str(item.metadata.get("object"))
        for item in manager.semantic.facts()
        if item.metadata.get("subject") == "web 中转站"
        and item.metadata.get("predicate") == "余额"
        and item.metadata.get("active", True) is not False
    )
    assert active == ["0 元"]
    # 历史仍然可审计：被退役的旧值保留在库里。
    retired = sorted(
        str(item.metadata.get("object"))
        for item in manager.semantic.facts()
        if item.metadata.get("subject") == "web 中转站"
        and item.metadata.get("predicate") == "余额"
        and item.metadata.get("active") is False
    )
    assert retired == ["1 元", "2 元"]


def test_build_graph_context_reuses_the_supplied_resolver():
    """回归：build_graph_context 过去每次都自建 EntityResolver，等于每个 chunk
    全量重载一次实体；接入 ingest 级 resolver 后必须复用同一份索引。"""

    manager = _manager()
    resolver = EntityResolver(manager)
    resolver.resolve("web 中转站", domain="中转站", aliases=["web"])
    manager.semantic.add_fact("web 中转站", "余额", "1 元")

    # 注入的 resolver 生效：不传则内部自建。
    injected = build_graph_context(
        manager, "web 中转站的余额是多少", resolver=resolver
    )
    assert "web 中转站" in injected

    original_init = EntityResolver.__init__
    calls: list[int] = []

    def counting_init(self, *args, **kwargs):
        calls.append(1)
        return original_init(self, *args, **kwargs)

    EntityResolver.__init__ = counting_init  # type: ignore[method-assign]
    try:
        build_graph_context(manager, "web 中转站的余额是多少", resolver=resolver)
        assert calls == [], "注入 resolver 时不得再次全量加载实体"
        build_graph_context(manager, "web 中转站的余额是多少")
        assert len(calls) == 1, "未注入时仍应自建 resolver，保持独立可用"
    finally:
        EntityResolver.__init__ = original_init  # type: ignore[method-assign]


def test_ingest_builds_one_resolver_per_call(monkeypatch):
    """整次 ingest 只构造一个 EntityResolver（不再每个 chunk 重载实体）。"""

    class Extractor:
        def extract(self, text, *, metadata=None, graph_context=""):
            return ExtractionResult(
                domain="中转站",
                entities=[EntityCandidate(name="web 中转站")],
                relations=[],
            )

    manager = _manager()
    pipeline = RAGPipeline(manager, extractor=Extractor())

    created: list[int] = []
    original_init = EntityResolver.__init__

    def counting_init(self, *args, **kwargs):
        created.append(1)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(EntityResolver, "__init__", counting_init)
    items = pipeline.ingest(
        Document("x" * 2000, metadata={"filename": "many-chunks.txt"}),
        chunk_size=200,
        overlap=0,
    )

    assert len(items) > 1, "用例需要多 chunk 才有意义"
    # 修复前：每个 chunk 在 build_graph_context 里各建一个 resolver。
    assert len(created) == 1
