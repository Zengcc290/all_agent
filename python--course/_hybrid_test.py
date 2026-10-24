"""多路混合检索自检：python _hybrid_test.py"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QDRANT_LOCAL_PATH", str(ROOT / "_qd_hyb"))
os.environ.setdefault("SQLITE_PATH", str(ROOT / "_hyb.db"))
os.environ.setdefault("EMBEDDING_DIM", "8")

DOCS = [
    "胰岛素注射液由诺和诺德生产，用于治疗糖尿病引起的血糖升高，常规剂量为每日两次皮下注射。",
    "二甲双胍是二型糖尿病的一线用药，通常随餐服用以减少胃肠道不适。",
    "糖尿病患者应定期监测糖化血红蛋白，目标值一般控制在 7% 以下。",
    "北京协和医院的内分泌科在糖尿病诊疗方面经验丰富，张医生擅长胰岛素泵治疗。",
    "糖尿病视网膜病变是常见并发症，需要每年进行眼底检查。",
]

LLM_JSON = ('{"sub_queries": ["糖尿病的治疗方案有哪些", "胰岛素适用于哪些疾病",'
            ' "糖尿病的常用药物"]}')


async def main():
    from app.core.registry import registry
    from app.db.sqlite_store import SQLiteStore
    import app.tools.hybrid_search as hs
    import app.tools.ingest_sentence as ismod

    # 直接用 store 造数据（跳过 LLM 抽取）
    st = SQLiteStore(os.environ["SQLITE_PATH"])
    cids = []
    for i, d in enumerate(DOCS):
        doc = st.add_document(d, "seed")
        cid = st.enqueue_chunk(doc["id"], d, i)
        # 直接改写状态并转正
        st.update_chunk_status(cid, qdrant_status="success", neo4j_status="success")
        st.promote_chunk(cid)
        cids.append(cid)

    print(f"sqlite: {st.stats()}")

    # 向量路：直接往 qdrant 灌向量
    from app.db.qdrant_store import vector_store
    from app.llm import embed_texts

    async def fake_embed(texts):
        return [[float(sum(ord(c) for c in t) % 11) / 11.0] * 8 for t in texts]

    hs.embed_texts = fake_embed
    await vector_store.ensure_collection()
    for cid, d in zip(cids, DOCS):
        v = await fake_embed([d])
        await vector_store.upsert_chunk(cid, v[0], {"content": d, "document_id": "seed"})

    print(f"qdrant points: {await vector_store.count()}")

    # LLM 拆解打桩
    async def fake_chat(messages, **kw):
        return LLM_JSON
    hs.chat = fake_chat

    await registry.discover()

    # ---------- 1) LLM 拆解 ----------
    r = await registry.call("hybrid_search", {
        "question": "糖尿病患者能不能用胰岛素，剂量怎么定？",
        "top_k": 5, "per_route_k": 3, "rrf_k": 60, "use_llm_split": True,
    })
    res = r["result"]
    assert res["ok"], res
    print(f"\n[1] 子问题拆解 -> {res['sub_query_count']} 个: {res['sub_queries']}")
    assert len(res["sub_queries"]) == 3
    print(f"    启用检索路: {res['routes_enabled']}, 各路命中: {res['agreement']['per_route']}")
    print(f"    多路重合: {res['agreement']['multi_route']}/{res['agreement']['total_unique']} "
          f"(overlap {res['agreement']['overlap_ratio']}), 耗时 {res['elapsed_ms']}ms")

    print("\n[2] RRF 融合结果:")
    for i, row in enumerate(res["fused"][:5], 1):
        print(f"    #{i} rrf={row['rrf_score']:.5f} 命中路={row['route_hits']} "
              f"子问题={row['sub_queries_hit']}")
        print(f"       {(row.get('content') or '')[:56]}")
    assert res["fused"], "融合结果不应为空"

    # ---------- 3) 关掉 LLM 拆分 ----------
    r2 = await registry.call("hybrid_search", {
        "question": "糖尿病的常用药物", "top_k": 3, "use_llm_split": False, "rrf_k": 60,
    })
    res2 = r2["result"]
    print(f"\n[3] 关闭 LLM 拆分 -> 子问题 {res2['sub_query_count']} 个: {res2['sub_queries']}")
    assert res2["sub_queries"] == ["糖尿病的常用药物"]
    for i, row in enumerate(res2["fused"][:3], 1):
        print(f"    #{i} rrf={row['rrf_score']:.5f} 命中路={row['route_hits']}")

    # ---------- 4) 只启用单路 ----------
    r3 = await registry.call("hybrid_search", {
        "question": "胰岛素", "top_k": 3, "use_llm_split": False, "routes": ["fts"],
    })
    res3 = r3["result"]
    print(f"\n[4] 只启用 fts 路 -> 命中 {res3['count']} 条: "
          f"[{[x['route_hits'] for x in res3['fused']]}]")
    assert all(x["route_hits"] == ["fts"] for x in res3["fused"])

    # ---------- 5) 单独 FTS ----------
    r4 = await registry.call("search_fulltext", {"query": "胰岛素", "top_k": 3})
    print(f"[5] search_fulltext -> {r4['result']['count']} 条, FTS5 available={r4['result']['fts_ok']}")

    # ---------- 6) 参数校验 ----------
    try:
        await registry.call("hybrid_search", {"question": "x", "rrf_k": 0})
        print("[6] rrf_k=0 未被拦截 <- 应该报错")
    except Exception as e:
        print(f"[6] rrf_k=0 正确拦截: {str(e)[:60]}")
    try:
        await registry.call("hybrid_search", {"question": "x", "routes": ["bogus"]})
        print("[7] routes=bogus 未被拦截 <- 应该报错")
    except Exception as e:
        print(f"[7] routes=bogus 正确提示: {str(e)[:70]}")

    print(f"\n[8] 已登记工具 {len(registry.names())} 个，prompt 已包含 hybrid_search: "
          f"{'hybrid_search' in registry.describe()}")
    print("=== 多路混合检索自检通过 ===")


asyncio.run(main())
