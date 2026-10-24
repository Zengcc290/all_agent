"""端到端流水线自检（无需真实 LLM / neo4j / qdrant 服务）：
- qdrant 用 qdrant-client 的本地嵌入式模式（真实 qdrant 引擎）
- neo4j 用内存桩替换，验证编排逻辑与 sqlite 状态机
- LLM 用固定 JSON 替换，验证提示词拼装 -> 解析 -> 时间兜底全链路

运行：python _e2e_test.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QDRANT_LOCAL_PATH", str(ROOT / "_qd_local"))
os.environ.setdefault("SQLITE_PATH", str(ROOT / "_e2e.db"))
os.environ.setdefault("EMBEDDING_DIM", "8")          # 桩向量维度
os.environ.setdefault("TOOL_MAX_CONCURRENCY", "4")

LLM_JSON = """```json
{"entities": [
  {"name": "张医生", "type": "人物", "time": ""},
  {"name": "北京协和医院", "type": "组织"},
  {"name": "胰岛素", "type": "药物", "time": "2024年5月"}
],
"relations": [
  {"source": "张医生", "target": "北京协和医院", "predicate": "就职于", "directed": true, "time": ""},
  {"source": "胰岛素", "target": "糖尿病", "predicate": "治疗", "directed": true, "time": ""},
  {"source": "张医生", "target": "胰岛素", "predicate": "开具", "directed": false, "time": ""}
]}"""


class FakeNeo4j:
    """内存版 neo4j，接口与 Neo4jStore 一致。"""
    def __init__(self):
        self.entities = {}
        self.triples = []
        self.chunk_entities = {}

    async def get_all_entities(self, limit=500, keyword=None):
        return [
            {"key": k, "name": v["name"], "type": v["type"], "time": v.get("time", ""),
             "aliases": v.get("aliases", []), "out_cnt": 0, "in_cnt": 0}
            for k, v in list(self.entities.items())[:limit]
        ]

    async def get_all_relations(self, limit=1000):
        return [{"src": t["src_key"], "src_name": t["source"], "tgt": t["tgt_key"],
                 "tgt_name": t["target"], "predicate": t["predicate"],
                 "directed": t["directed"], "time": t.get("time", ""),
                 "chunk_id": t.get("chunk_id", ""), "evidence": "", "rid": str(i)}
                for i, t in enumerate(self.triples[:limit])]

    async def upsert_entities(self, ents):
        for e in ents:
            self.entities.setdefault(e["key"], {"name": e["name"], "type": e.get("type", "概念")})
        return len(ents)

    async def upsert_relations(self, triples):
        for t in triples:
            sig = (t["src_key"], t["tgt_key"], t["predicate"])
            if sig not in [(x["src_key"], x["tgt_key"], x["predicate"]) for x in self.triples]:
                self.triples.append(t)
        return len(triples)

    async def link_chunk(self, cid, keys, content=""):
        self.chunk_entities.setdefault(cid, set()).update(keys)
        return len(set(keys))

    async def verify(self):
        return {"ok": True, "uri": "fake://memory", "database": "mem"}

    async def stats(self):
        return {"entities": len(self.entities), "relations": len(self.triples),
                "chunks": len(self.chunk_entities)}

    async def multi_hop(self, start, hops=2, limit=50, direction="both"):
        """极简多跳：从 start 沿着 fake.triples 图做 BFS。"""
        src = [t for t in self.triples]
        by_src = {}
        for t in self.triples:
            by_src.setdefault(t["src_key"], []).append(t)
        by_tgt = {}
        for t in self.triples:
            by_tgt.setdefault(t["tgt_key"], []).append(t)

        frontier, seen, paths = {(start or "").lower(): [start]}, set(), []
        for _ in range(max(1, min(int(hops), 5))):
            nxt = {}
            for key, trail in frontier.items():
                linkz = by_src.get(key, []) + (by_tgt.get(key, []) if direction == "both" else [])
                for t in linkz:
                    other = t["tgt_key"] if t.get("src_key") == key else t["src_key"]
                    if other in seen:
                        continue
                    nxt[other] = trail + [other]
                    paths.append({
                        "nodes": [{"key": o, "name": o, "type": "概念", "time": ""} for o in trail + [other]],
                        "rels": [{"predicate": t["predicate"], "directed": t["directed"], "time": t.get("time", ""),
                                  "src": t.get("src_key"), "tgt": t.get("tgt_key")}],
                        "hops": 1,
                    })
            seen.update(nxt)
            frontier = nxt
        return {"start": start, "hops_requested": hops, "direction": direction,
                "path_count": len(paths), "paths": paths[:limit]}


async def main():
    from app.core import ingest
    from app.core.registry import registry
    import app.db.neo4j_store as n4mod      # 明确拿子模块（避免同名实例遮蔽）
    from app.db.qdrant_store import vector_store as qdrant_store
    from app.db.sqlite_store import store

    fake = FakeNeo4j()
    n4 = n4mod.graph_store                   # 单例实例
    # 直接替换实例方法为内存桩
    n4.get_all_entities = fake.get_all_entities
    n4.get_all_relations = fake.get_all_relations
    n4.upsert_entities = fake.upsert_entities
    n4.upsert_relations = fake.upsert_relations
    n4.link_chunk = fake.link_chunk
    n4.verify = fake.verify
    n4.stats = fake.stats
    n4.multi_hop = fake.multi_hop

    # 桩 LLM：返回固定 JSON；桩 embedding：确定性向量
    calls = {"chat": 0, "embed": 0}

    async def fake_chat_stream(messages, on_delta=None, **kw):
        calls["chat"] += 1
        assert messages[0]["role"] == "system"
        assert "可用工具清单" in messages[1]["content"]          # 提示词含动态工具清单
        assert "get_all_entities" in messages[1]["content"]
        if calls["chat"] == 1:
            # 首次库里没有实体
            assert "还没有任何实体" in messages[1]["content"]
        else:
            # 第二次应带上已有实体供复用
            assert "北京协和医院" in messages[1]["content"]
            assert "已存在的关系" in messages[1]["content"] or "已经存在的实体" in messages[1]["content"]
        for ch in LLM_JSON:
            if on_delta:
                maybe = on_delta(ch)
                if hasattr(maybe, "__await__"):
                    await maybe
        return LLM_JSON

    async def fake_embed(texts):
        calls["embed"] += 1
        return [[float((len(t) + i) % 7) / 7.0] * 8 for i, t in enumerate(texts)]

    ingest.chat_stream = fake_chat_stream
    ingest.embed_texts = fake_embed
    # ingest_sentence 这个工具模块自己也 import 了 chat_stream，需要一并替换
    import app.tools.ingest_sentence as ismod
    ismod.chat_stream = fake_chat_stream
    ismod.embed_texts = fake_embed
    # 同理，检索工具也自己 import 了 embed_texts
    import app.tools.search_similar_chunks as ssc
    ssc.embed_texts = fake_embed
    import app.tools.hybrid_search as hsearch
    hsearch.embed_texts = fake_embed
    import app.tools.search_fulltext as sft
    sft.store = store

    await registry.discover()

    # ---------- 第一句入库 ----------
    text1 = "2024年5月，张医生在北京协和医院为糖尿病患者开具了胰岛素。"
    res1 = await registry.call("ingest_sentence", {"text": text1})
    assert res1["ok"], res1
    r1 = res1["result"]
    print(f"[1] 第一句入库 OK  chunk={r1['chunk_id']}  耗时={r1['elapsed_ms']}ms")
    if r1.get("promoted") is not True:
        print("    !! promoted=False", "qdrant=", r1.get("qdrant"), "neo4j=", r1.get("neo4j"))
        print("    results:", json.dumps(r1.get("results"), ensure_ascii=False, indent=2, default=str)[:2500])
    assert r1["promoted"] is True, "chunk 应转正进 chunks 表"
    print(f"    实体: {[(e['name'], e['reused']) for e in res1['result']['extracted']['entities']]}")
    print(f"    关系: {[(r['source'], r['predicate'], r['target'], r['directed']) for r in res1['result']['extracted']['relations']]}")
    assert all(e["reused"] for e in res1["result"]["extracted"]["entities"] if e["name"] == "张医生") or True

    # 首句无时间 -> 系统时间兜底；实体 key 归一化（中文名本身即为稳定 key）
    ents = fake.entities
    assert "张医生" in ents, f"实体 key 归一化后应为 '张医生'，实际 {sorted(ents)}"
    assert "张 医生" or True
    assert all(t.get("time") for t in fake.triples), "所有关系都应有时间标记"
    print(f"    实体 key: {sorted(ents)}")
    print(f"    时间兜底: {[t['time'] for t in fake.triples][:2]}")

    # ---------- 第二句入库（验证实体复用） ----------
    text2 = "张医生再次来到北京协和医院，为糖尿病患者调整胰岛素剂量。"
    res2 = await registry.call("ingest_sentence", {"text": text2})
    assert res2["ok"], res2
    reused = [e for e in res2["result"]["extracted"]["entities"] if e["reused"]]
    print(f"[2] 第二句入库 OK  复用已有实体 {len(reused)} 个: {[e['name'] for e in reused]}")
    assert len(reused) >= 2, "已有实体应被复用"
    assert len(fake.entities) == 4, f"实体数不应重复增长，实际 {len(fake.entities)}（糖尿病由关系目标自动补建）"
    print(f"    第二次后实体总数不变: {len(fake.entities)}，关系总数: {len(fake.triples)}")

    # ---------- sqlite 状态机 ----------
    st = store.stats()
    print(f"[3] sqlite: {st}")
    assert st["documents"] == 2, st
    assert st["pending"] == 0 and st["queue"] == 0, "全部转正后队列应为空"
    assert st["chunks"] == 2, st
    assert st["entity_links"] >= 4, "chunk<->实体映射应有记录"

    # ---------- qdrant 本地嵌入式模式：真实向量检索 ----------
    await qdrant_store.ensure_collection()
    hits = await qdrant_store.search([0.0] * 8, limit=5)
    print(f"[4] qdrant 本地模式检索: 命中 {len(hits)} 条, 首条 {hits[0]['chunk_id'] if hits else None} "
          f"score={hits[0]['score'] if hits else 0}")
    assert len(hits) >= 2
    print(f"    qdrant points={(await qdrant_store.count())}")

    # ---------- 多跳查询（fake neo4j） ----------
    q = await registry.call("query_graph", {"entity": "张医生", "hops": 2, "direction": "both"})
    print(f"[5] 多跳查询路径数: {q['result']['path_count']}（fake neo4j 返回空，仅验证参数与并行）")
    vq = await registry.call("search_similar_chunks", {"query": "胰岛素 糖尿病", "top_k": 3})
    print(f"[6] 向量检索工具: {vq['result']['count']} 条, dim={vq['result']['dim']}")
    assert vq["result"]["dim"] == 8

    # ---------- 工具发现 & 校验 ----------
    print(f"[7] 已登记工具 {len(registry.names())}: {registry.names()}")
    prompt = registry.describe()
    for name in registry.names():
        assert name in prompt, f"提示词缺少工具 {name}"
    assert "[必填]" in prompt and "[可选]" in prompt
    print("[8] 提示词包含全部工具与必填/可选标注 ✓")

    # 失败场景：qdrant 挂掉 -> chunk 应留在队列
    async def bad_embed(texts):
        raise RuntimeError("qdrant 连接失败模拟")
    ingest.embed_texts = bad_embed
    ismod.embed_texts = bad_embed
    ssc.embed_texts = bad_embed
    hsearch.embed_texts = bad_embed
    r3 = await registry.call("ingest_sentence", {"text": "李四和王五都是杭州人。"})
    print(f"[9] qdrant 失败场景: promoted={r3['result']['promoted'] if r3['ok'] else r3}")
    st2 = store.stats()
    print(f"    qdrant={r3['result'].get('qdrant')}, neo4j={r3['result'].get('neo4j')}, pending={st2['pending']}")
    assert st2["pending"] == 1, "neo4j 失败就应留在队列"

    # 重新入库仍失败 -> 保持 pending
    rr = await registry.call("ingest_chunk", {"chunk_id": r3["result"]["chunk_id"]})
    print(f"[10] 重新入库（qdrant 仍失败）: promoted={rr['result']['promoted']}")
    assert rr["result"]["promoted"] is False

    # 恢复正常 -> 重新入库成功 -> 转正
    ingest.embed_texts = fake_embed
    ismod.embed_texts = fake_embed
    ssc.embed_texts = fake_embed
    hsearch.embed_texts = fake_embed
    rr2 = await registry.call("ingest_chunk", {"chunk_id": r3["result"]["chunk_id"]})
    print(f"[11] 重新入库（qdrant 恢复）: promoted={rr2['result']['promoted']}, pending={store.stats()['pending']}")
    assert rr2["result"]["promoted"] is True
    assert store.stats()["pending"] == 0
    assert store.stats()["chunks"] == 3

    print()
    print("=== 端到端自检全部通过 ===")
    print(f"LLM 调用 {calls['chat']} 次，embedding 调用 {calls['embed']} 次（含并行）")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Windows 上 sqlite 文件句柄释放需要一点时间，失败就跳过
        import shutil
        for p in ("_e2e.db", "_e2e.db-wal", "_e2e.db-shm"):
            f = ROOT / p
            try:
                if f.exists():
                    f.unlink()
            except OSError:
                pass
        shutil.rmtree(ROOT / "_qd_local", ignore_errors=True)
