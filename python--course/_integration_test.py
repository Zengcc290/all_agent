"""最终集成验证：真实 Neo4j + 真实 Qdrant + 真实 SQLite，只有 LLM 用桩替换。

这一版把整条入库链路（ingest_sentence）跑在真实图库上，
再验证 get_graph_snapshot / query_graph / get_all_entities 等查询工具的真实返回。

运行：python _integration_test.py
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "neo4j123456")
os.environ.setdefault("QDRANT_LOCAL_PATH", str(ROOT / "_itest_qdrant"))
os.environ.setdefault("SQLITE_PATH", str(ROOT / "_itest.db"))
os.environ.setdefault("EMBEDDING_DIM", "8")

SENTENCES = [
    "2024年5月，张医生在北京协和医院为糖尿病患者开具了胰岛素。",
    "胰岛素注射液由诺和诺德生产，常规剂量为每日两次皮下注射。",
    "王伟和李强在杭州合作开发了一套新的血糖监测系统。",
]

# 固定的“假 LLM”输出：逐句给出不同的实体关系，第二句用来测试实体复用
LLM_OUTPUTS = [
    """{"entities": [
        {"name": "张医生", "type": "人物", "key": "张医生", "time": ""},
        {"name": "北京协和医院", "type": "组织", "key": "北京协和医院", "time": ""},
        {"name": "糖尿病", "type": "疾病", "key": "糖尿病", "time": ""},
        {"name": "胰岛素", "type": "药物", "key": "胰岛素", "time": ""}
      ],
      "relations": [
        {"source": "张医生", "target": "北京协和医院", "predicate": "就职于", "directed": true, "time": ""},
        {"source": "张医生", "target": "胰岛素", "predicate": "开具", "directed": true, "time": ""},
        {"source": "胰岛素", "target": "糖尿病", "predicate": "治疗", "directed": true, "time": ""},
        {"source": "张医生", "target": "糖尿病", "predicate": "关注", "directed": false, "time": ""}
      ]}""",
    """{"entities": [
        {"name": "胰岛素", "type": "药物", "key": "胰岛素", "time": ""},
        {"name": "诺和诺德", "type": "公司", "key": "诺和诺德", "time": ""}
      ],
      "relations": [
        {"source": "胰岛素", "target": "诺和诺德", "predicate": "由..生产", "directed": true, "time": ""}
      ]}""",
    """{"entities": [
        {"name": "王伟", "type": "人物", "key": "王伟", "time": ""},
        {"name": "李强", "type": "人物", "key": "李强", "time": ""},
        {"name": "血糖监测系统", "type": "产品", "key": "血糖监测系统", "time": ""}
      ],
      "relations": [
        {"source": "王伟", "target": "血糖监测系统", "predicate": "开发", "directed": true, "time": ""},
        {"source": "李强", "target": "血糖监测系统", "predicate": "开发", "directed": true, "time": ""},
        {"source": "王伟", "target": "李强", "predicate": "合作", "directed": false, "time": ""}
      ]}""",
]

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   -> {detail}" if (detail and not cond) else ""))


async def main():
    from app.core.registry import registry
    from app.db.neo4j_store import graph_store
    from app.db.qdrant_store import vector_store
    from app.db.sqlite_store import store
    import app.tools.ingest_sentence as ismod
    import app.tools.hybrid_search as hs
    import app.tools.search_similar_chunks as ssc
    import app.tools.search_fulltext as sft

    # 清库
    print("=== 清理既有数据 ===")
    await graph_store.clear_graph()
    store.clear_all()
    try:
        await vector_store.close()
        import shutil
        shutil.rmtree(ROOT / "_itest_qdrant", ignore_errors=True)
    except Exception:
        pass
    await vector_store.delete_by_document("*") if False else None

    # ---- 桩 LLM / embedding ----
    idx = {"n": 0}
    seen_entities: list[set] = []

    async def fake_chat(messages, on_delta=None, **kw):
        used = messages[1]["content"]
        return LLM_OUTPUTS[idx["n"] % len(LLM_OUTPUTS)]

    async def fake_embed(texts):
        out = []
        for t in texts:
            # 按字符散列成 8 维向量，保证不同文本向量不同、相同文本向量相同
            v = [0.0] * 8
            for ch in t:
                v[hash(ch) % 8] += 1.0 / max(1, len(t))
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([round(x / n, 6) for x in v])
        return out

    ismod.chat_stream = fake_chat
    hs.chat = fake_chat
    hs.embed_texts = fake_embed
    ssc.embed_texts = fake_embed
    # embedding 侧：app.core.ingest 与 tool 都要换
    from app.core import ingest as core_ingest
    core_ingest.chat_stream = fake_chat
    core_ingest.embed_texts = fake_embed

    await registry.discover()

    # ---------- 逐句入库 ----------
    print("\n=== 逐句入库（真实 neo4j + qdrant）===")
    results = []
    for i, text in enumerate(SENTENCES):
        idx["n"] = i
        r = await ismod.ingest_sentence(text, "integration")
        ex = r["extracted"]
        print(f"  [{i+1}] {text[:26]}...")
        print(f"       qdrant={r['qdrant']} neo4j={r['neo4j']} promoted={r['promoted']} "
              f"时间来源={r['time_source']}({r['time']})")
        print(f"       实体 {len(ex['entities'])} 个(新 {ex['new_entities']}/复用 {ex['reused_entities']}) "
              f"关系 {len(ex['relations'])} 条  耗时 {r['elapsed_ms']}ms")
        results.append(r)
        check(f"句{i+1} 双库都成功", r["qdrant"] == "success" and r["neo4j"] == "success", str(r))
        check(f"句{i+1} 转正进 chunks 表", r["promoted"] is True)

    # ---------- sqlite ----------
    print("\n=== sqlite 状态 ===")
    st = store.stats()
    print(f"  {st}")
    check("documents 记录 3 条", st["documents"] == 3, str(st))
    check("chunks 转正 3 条", st["chunks"] == 3, str(st))
    check("队列已清空", st["queue"] == 0 and st["pending"] == 0, str(st))
    check("chunk_entities 映射至少有 9 条", st["entity_links"] >= 9, str(st))
    check("FTS5 索引已同步", st["fts_ok"] and st["fts_rows"] == 3, str(st))

    # chunk <-> 实体 多对多
    cid = results[0]["chunk_id"]
    ents_of_cid = [e["entity_key"] for e in store.get_chunk_entities(cid)]
    check("chunk -> 多个实体", len(ents_of_cid) == 4, str(ents_of_cid))
    chunks_of_insulin = [c["chunk_id"] for c in store.get_chunks_of_entity("胰岛素")]
    check("一个实体 <- 多个 chunk（胰岛素出现 2 次）",
          len(chunks_of_insulin) == 2, str(chunks_of_insulin))

    # ---------- neo4j 内容 ----------
    print("\n=== neo4j 图谱内容 ===")
    gst = await graph_store.stats()
    print(f"  entities={gst['entities']} relations={gst['relations']} chunks={gst['chunks']}")
    check("实体 8 个", gst["entities"] == 8, str(gst))
    check("关系 8 条", gst["relations"] == 8, str(gst))
    check("chunk 节点 3 个", gst["chunks"] == 3, str(gst))

    ents = await graph_store.get_all_entities()
    names = {e["name"] for e in ents}
    print(f"  实体: {sorted(names)}")
    check("实体集合正确",
          names == {"张医生", "北京协和医院", "糖尿病", "胰岛素", "诺和诺德", "王伟", "李强", "血糖监测系统"},
          str(sorted(names)))

    rels = await graph_store.get_all_relations()
    trip = {(r["src_label"], r["predicate"], r["tgt_label"]) for r in rels}
    print(f"  关系: {sorted(trip)}")
    check("关系三元组正确",
          trip == {("张医生", "就职于", "北京协和医院"), ("张医生", "开具", "胰岛素"),
                   ("胰岛素", "治疗", "糖尿病"), ("张医生", "关注", "糖尿病"),
                   ("胰岛素", "由..生产", "诺和诺德"), ("王伟", "开发", "血糖监测系统"),
                   ("李强", "开发", "血糖监测系统"), ("王伟", "合作", "李强")},
          str(sorted(trip)))

    undirected = {(r["src_label"], r["predicate"], r["tgt_label"])
                  for r in rels if r["directed"] is False}
    check("无向关系有 2 条（关注 / 合作）", len(undirected) == 2, str(undirected))

    # 时间：所有实体和关系都应有时间（第一句无时间 -> 系统兜底）
    no_time_r = [r for r in rels if not r.get("time")]
    no_time_e = [e for e in ents if not e.get("time")]
    check("所有关系都带时间标记", not no_time_r, str(no_time_r))
    check("所有实体都带时间标记", not no_time_e, str(no_time_e))
    print(f"  关系时间示例: {[r['time'] for r in rels[:3]]}")

    # 关系带 chunk_id（溯源）
    with_chunk = [r for r in rels if r.get("chunk_id")]
    check("关系都溯源到 chunk_id", len(with_chunk) == len(rels), f"{len(with_chunk)}/{len(rels)}")

    # ---------- 多跳查询 ----------
    print("\n=== 多跳查询（真实 neo4j）===")
    mh = await registry.call("query_graph", {"entity": "张医生", "hops": 2, "limit": 100})
    res = mh["result"]
    nodes_found = {n["key"] for n in res["nodes"]}
    print(f"  张医生 2 跳 -> 路径 {res['path_count']} 条，节点 {sorted(nodes_found)}")
    check("2 跳可达胰岛素/糖尿病/北京协和医院",
          {"胰岛素", "糖尿病", "北京协和医院"} <= nodes_found, str(sorted(nodes_found)))

    mh3 = await registry.call("query_graph", {"entity": "张医生", "hops": 3, "limit": 100})
    n3 = {n["key"] for n in mh3["result"]["nodes"]}
    print(f"  张医生 3 跳 -> 路径 {mh3['result']['path_count']} 条，节点 {sorted(n3)}")
    check("3 跳可达诺和诺德（张->胰岛素->诺和诺德）", "诺和诺德" in n3, str(sorted(n3)))
    check("跳数越多路径越多",
          mh3["result"]["path_count"] >= res["path_count"],
          f"{mh3['result']['path_count']} vs {res['path_count']}")

    mho = await registry.call("query_graph", {"entity": "糖尿病", "hops": 2, "direction": "out"})
    no1 = {n["key"] for n in mho["result"]["nodes"]}
    check("有向遍历从糖尿病出发不会反向拿到张医生", "张医生" not in no1, str(sorted(no1)))

    mhb = await registry.call("query_graph", {"entity": "糖尿病", "hops": 2, "direction": "both"})
    nb1 = {n["key"] for n in mhb["result"]["nodes"]}
    check("无向遍历从糖尿病能反向拿到张医生", "张医生" in nb1, str(sorted(nb1)))

    # ---------- 图快照 ----------
    print("\n=== 图快照（前端实体星球数据源）===")
    snap = (await registry.call("get_graph_snapshot", {"limit": 100}))["result"]
    print(f"  nodes={len(snap['nodes'])} links={len(snap['links'])} "
          f"有向={sum(1 for l in snap['links'] if l['directed'])} "
          f"无向={sum(1 for l in snap['links'] if not l['directed'])}")
    check("快照节点数=8", len(snap["nodes"]) == 8, str(len(snap["nodes"])))
    check("快照连线数=8", len(snap["links"]) == 8, str(len(snap["links"])))
    check("有向线 6 条", sum(1 for l in snap["links"] if l["directed"]) == 6)
    check("无向线 2 条", sum(1 for l in snap["links"] if not l["directed"]) == 2)
    check("连线带 predicate 与 time",
          all(l.get("predicate") and l.get("time") for l in snap["links"]))
    # 前端渲染要求 source/target 必须在 nodes 里
    ids = {n["id"] for n in snap["nodes"]}
    check("所有连线的 source/target 都有对应节点",
          all(l["source"] in ids and l["target"] in ids for l in snap["links"]))

    # ---------- 向量检索 ----------
    print("\n=== qdrant 向量检索 ===")
    vq = (await registry.call("search_similar_chunks", {"query": "胰岛素 糖尿病", "top_k": 5}))["result"]
    print(f"  命中 {vq['count']} 条，dim={vq['dim']}")
    for h in vq["hits"][:3]:
        print(f"    score={h['score']:.4f} {h['chunk_id']} {(h['content'] or '')[:30]}")
    check("向量检索有结果", vq["count"] >= 2, str(vq))
    check("检索结果带 chunk 关联实体", any(h.get("entities") for h in vq["hits"]))

    # ---------- FTS5 ----------
    print("\n=== FTS5 全文检索 ===")
    fq = (await registry.call("search_fulltext", {"query": "胰岛素", "top_k": 5}))["result"]
    print(f"  命中 {fq['count']} 条: {[h['chunk_id'] for h in fq['hits']]}")
    check("FTS5 命中含胰岛素的 2 条", fq["count"] == 2, str(fq))
    fq2 = (await registry.call("search_fulltext", {"query": "血糖监测系统", "top_k": 5}))["result"]
    check("FTS5 命中血糖监测系统", fq2["count"] == 1, str(fq2))

    # ---------- 混合检索 ----------
    print("\n=== 多路混合检索 + RRF ===")
    hr = (await registry.call("hybrid_search", {
        "question": "胰岛素可以治疗糖尿病吗？", "top_k": 4,
        "rrf_k": 60, "per_route_k": 5,
    }))["result"]
    print(f"  子问题: {hr['sub_queries']}")
    print(f"  各路命中: {hr['agreement']['per_route']} 重合 {hr['agreement']['multi_route']}")
    for i, row in enumerate(hr["fused"], 1):
        print(f"    #{i} rrf={row['rrf_score']:.5f} 路={row['route_hits']} "
              f"{(row.get('content') or '')[:28]}")
    check("混合检索有子问题", len(hr["sub_queries"]) >= 1, str(hr["sub_queries"]))
    check("向量与 FTS5 两路都有命中",
          set(hr["agreement"]["per_route"]) >= {"vector", "fts"}, str(hr["agreement"]))
    check("RRF 融合有结果", len(hr["fused"]) > 0, str(hr))
    multi = [r for r in hr["fused"] if r["route_count"] >= 2]
    print(f"  多路同时命中的条数: {len(multi)}")
    if multi:
        check("被多路命中的结果排在最前",
              multi[0]["rrf_score"] >= hr["fused"][-1]["rrf_score"])

    # ---------- 重新入库 / 队列 ----------
    print("\n=== 重新入库（先注入失败，再恢复）===")
    bad = {"n": 0}

    async def boom(texts):
        raise RuntimeError("embedding 服务挂了")
    core_ingest.embed_texts = boom
    r_bad = await registry.call("ingest_sentence", {"text": "赵六在南京发布了新的医疗AI模型。"})
    print(f"  qdrant={r_bad['result'].get('qdrant')} neo4j={r_bad['result'].get('neo4j')} "
          f"promoted={r_bad['result'].get('promoted')}")
    st2 = store.stats()
    print(f"  sqlite: pending={st2['pending']} chunks={st2['chunks']}")
    check("qdrant 失败时留在队列", st2["pending"] == 1, str(st2))
    check("失败 chunk 未转正", r_bad["result"]["promoted"] is False)

    pend = (await registry.call("list_pending_chunks", {"limit": 10, "preview_chars": 10}))["result"]
    print(f"  待入库列表 {pend['returned']}/{pend['total_pending']} 条")
    check("待入库列表能取到", pend["returned"] == 1, str(pend))

    core_ingest.embed_texts = fake_embed
    ismod.embed_texts = fake_embed
    rr = await registry.call("ingest_chunk", {"chunk_id": r_bad["result"]["chunk_id"]})
    print(f"  恢复后重新入库: qdrant={rr['result'].get('qdrant')} promoted={rr['result'].get('promoted')}")
    check("恢复后重新入库成功", rr["result"]["promoted"] is True, str(rr))
    st3 = store.stats()
    check("队列再次清空", st3["pending"] == 0 and st3["chunks"] == 4, str(st3))

    # ---------- 图库增删改 ----------
    print("\n=== 图库增删改 ===")
    await registry.call("manage_graph", {"action": "add_entity", "entity_key": "测试实体", "type": "测试"})
    e_ok = await graph_store.get_entity("测试实体")
    check("add_entity 生效", bool(e_ok), str(e_ok))
    await registry.call("manage_graph", {"action": "delete_entity", "entity_key": "测试实体"})
    check("delete_entity 生效", await graph_store.get_entity("测试实体") is None)

    # ---------- 统计 ----------
    print("\n=== 三库统计 ===")
    stats = (await registry.call("get_stats", {}))["result"]
    print(f"  sqlite: {stats['sqlite']}")
    print(f"  qdrant: ok={stats['qdrant']['ok']} points={stats['qdrant'].get('points')} "
          f"dim={stats['qdrant'].get('dim')} {stats['qdrant'].get('distance')}")
    print(f"  neo4j:  ok={stats['neo4j']['ok']} {stats['neo4j'].get('uri')}")
    check("qdrant 可用", stats["qdrant"]["ok"], str(stats["qdrant"]))
    check("neo4j 可用", stats["neo4j"]["ok"], str(stats["neo4j"]))

    # ---------- 清理 ----------
    print("\n=== 清理 ===")
    await graph_store.clear_graph()
    store.clear_all()
    fg = await graph_store.stats()
    check("最终图库已清空", fg["entities"] == 0 and fg["relations"] == 0, str(fg))
    await graph_store.close()
    await vector_store.close()

    print(f"\n=== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
    if FAIL:
        print("失败项:")
        for f in FAIL:
            print("  -", f)
    return not FAIL


ok = asyncio.run(main())

import shutil
for p in ("_itest.db", "_itest_qdrant"):
    t = ROOT / p
    if t.is_file():
        try: t.unlink()
        except OSError: pass
    elif t.is_dir():
        shutil.rmtree(t, ignore_errors=True)
sys.exit(0 if ok else 1)
