"""真实 HTTP 端到端：后端连假 LLM 服务，全程走 SSacke/embeddings 协议。

运行：python _http_e2e.py     （需先起 _fake_llm_server.py 与 run.py）
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8899/api"


def _req(path, obj=None, timeout=300, method=None):
    data = json.dumps(obj).encode("utf-8") if obj is not None else None
    r = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8", "ignore")), None
    except Exception as e:
        return 0, {"_error": f"{type(e).__name__}: {e}"}, None


print("\n=== 脏数据检查：先记录基线（data/kg.db 可能残留历史数据）===")
st, s0, _ = _req("/stats")
base_chunks = s0["result"]["sqlite"]["chunks"]
base_docs = s0["result"]["sqlite"]["documents"]
base_ents = s0["result"]["neo4j"]["entities"]
base_rels = s0["result"]["neo4j"]["relations"]
print(f"  基线 chunks={base_chunks} documents={base_docs} 实体={base_ents} 关系={base_rels}")

print("\n=== 排干历史遗留的待入库队列，保证从干净状态开始 ===")
st, p0, _ = _req("/chunks/pending?limit=50")
if p0["total_pending"]:
    print(f"  发现 {p0['total_pending']} 条遗留，逐条重新入库...")
    for it in p0["items"]:
        try:
            _req(f"/chunks/{it['chunk_id']}/reingest", timeout=300)
            print(f"    {it['chunk_id']} 已处理")
        except Exception as e:
            print(f"    {it['chunk_id']} 失败: {e}")
    st, p0, _ = _req("/chunks/pending?limit=50")
print(f"  当前待入库: {p0['total_pending']} 条")

print("\n=== 真实一句话入库（HTTP + 假 LLM 流式 + 硅基流动格式 embeddings）===")
TEXT = "2024年5月，张医生在北京协和医院为糖尿病患者进行了系统诊治。"
st, res, _ = _req("/ingest/sentence", {"text": TEXT, "source": "http-e2e"})
print(f"  HTTP {st}")
print(f"  qdrant={res.get('qdrant')} neo4j={res.get('neo4j')} promoted={res.get('promoted')}")
print(f"  时间来源={res.get('time_source')} -> {res.get('time')}")
ex = res.get("extracted") or {}
print(f"  实体 {len(ex.get('entities') or [])} 个（新 {ex.get('new_entities')}/复用 {ex.get('reused_entities')}）"
      f" 关系 {len(ex.get('relations') or [])} 条")
for r in (ex.get("relations") or []):
    arrow = "->" if r["directed"] else "--"
    print(f"    {r['source']} -[{r['predicate']} {arrow} {r['time']}]-> {r['target']}")
print("  LLM 抽到的时间数: %d，走系统时间兜底的实体/关系数: %d"
      % (res.get("time_llm_count"), res.get("time_fallback_count")))
assert st == 200 and res["qdrant"] == "success" and res["neo4j"] == "success", res
assert res["promoted"] is True, "应转正进 chunks 表"
# 句子里有 2024年5月，LLM 确实抽到了时间（混合来源：部分用 LLM，部分兜底）
assert res["time_source"] == "llm", "句子里有时间，LLM 应至少抽到一个"
assert res.get("time_llm_count", 0) >= 1
assert res["time"], "时间不应为空"
assert all(r.get("time") for r in (ex.get("relations") or [])), "每条关系都必须有时间"
assert all(e.get("time") for e in (ex.get("entities") or [])), "每个实体都必须有时间"

print("\n=== 再入一句（假 LLM 不给时间 -> 全部系统时间兜底）===")
TEXT2 = "胰岛素由诺和诺德生产，常规剂量为每日两次皮下注射。"
st, res2, _ = _req("/ingest/sentence", {"text": TEXT2, "source": "http-e2e"})
print(f"  HTTP {st} 时间来源={res2.get('time_source')} -> {res2.get('time')} promoted={res2.get('promoted')}")
print(f"  LLM 抽到 {res2.get('time_llm_count')} 个时间，兜底 {res2.get('time_fallback_count')} 个")
assert st == 200 and res2["promoted"] is True
ex2 = res2.get("extracted") or {}
print(f"  实体 {len(ex2.get('entities') or [])} 个（新 {ex2.get('new_entities')}/复用 {ex2.get('reused_entities')}）")
reused = [e["name"] for e in (ex2.get("entities") or []) if e.get("reused")]
print(f"  复用实体: {reused}")
assert all(e.get("time") for e in (ex2.get("entities") or [])), "实体必须有时间"
assert all(r.get("time") for r in (ex2.get("relations") or [])), "关系必须有时间"

print("\n=== SSE 流式入库 ===")
r = urllib.request.Request(BASE + "/ingest/stream",
                           data=json.dumps({"text": "张医生再次来到北京协和医院复诊。",
                                            "source": "sse"}).encode(),
                           headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(r, timeout=300) as resp:
    body = resp.read().decode("utf-8")
events, llm_text = [], ""
for line in body.split("\n"):
    if not line.startswith("data:"):
        continue
    ev = json.loads(line[5:].strip())
    events.append(ev["event"])
    if ev["event"] == "llm_delta":
        llm_text += ev["data"].get("delta", "")
print(f"  收到事件 {len(events)} 个: {events}")
print(f"  LLM 流式增量拼回 {len(llm_text)} 字符, 开头: {llm_text[:20]!r}")
assert "start" in events and "done" in events and "llm_delta" in events and "promoted" in events, events
assert len(llm_text) > 50, "应真的收到 LLM 增量"

print("\n=== 入库后数据落库情况 ===")
st, g, _ = _req("/graph")
print(f"  /api/graph -> nodes={len(g['nodes'])} links={len(g['links'])} "
      f"有向={g['directed_links']} 无向={g['undirected_links']}")
assert len(g["nodes"]) >= 3 and len(g["links"]) >= 2

st, s, _ = _req("/stats")
sq, ng = s["result"]["sqlite"], s["result"]["neo4j"]
print(f"  sqlite: documents={sq['documents']} chunks={sq['chunks']} pending={sq['pending']} "
      f"entity_links={sq['entity_links']} fts_rows={sq['fts_rows']}")
print(f"  neo4j : entities={ng['entities']} relations={ng['relations']}")
print(f"  qdrant: points={s['result']['qdrant']['points']} dim={s['result']['qdrant']['dim']}")
assert ng["entities"] >= 3 and ng["relations"] >= 2
assert sq["chunks"] >= base_chunks  # 每次成功入库都至少不减少
assert sq["entity_links"] >= 3, "chunk<->实体 映射应至少有 3 条"

# 入库的三句话在 neo4j 里确实存在，且都带时间
st, al, _ = _req("/tools/get_all_entities/call", {"args": {"limit": 500}})
all_ents = al["result"]["entities"]
all_names = {e["name"] for e in all_ents}
print(f"  实体清单（前 8）: {sorted(all_names)[:8]}")
assert {"张医生", "北京协和医院", "糖尿病"} <= all_names
# 本次入库涉及的两个实体必须带时间（无时间的老种子数据不算）
for nm in ("张医生", "北京协和医院", "糖尿病"):
    e = next(x for x in all_ents if x["name"] == nm)
    print(f"    {nm}: type={e['type']} time={e['time']!r} out={e['out_count']} in={e['in_count']}")
    assert e["time"], f"实体 {nm} 应有时间标记"

st, rl, _ = _req("/tools/get_all_relations/call", {"args": {"limit": 2000}})
all_rels = rl["result"]["relations"]
print(f"  关系总数 {len(all_rels)}，全部带时间: {all(r.get('time') for r in all_rels)}")
assert all(r.get("time") for r in all_rels), "所有关系都应有时间标记"
has_target = any(r["source"] == "张医生" and r["predicate"] == "就职于"
                 and r["target"] == "北京协和医院" for r in all_rels)
print(f"  含「张医生-就职于->北京协和医院」: {has_target}")
assert has_target

print("\n=== 查询工具（真实三库）===")
st, q, _ = _req("/tools/query_graph/call", {"args": {"entity": "张医生", "hops": 3, "limit": 100}})
rq = q["result"]
print(f"  query_graph 3跳 -> 路径 {rq['path_count']} 条, 节点 {sorted(n['key'] for n in rq['nodes'])}")
assert rq["path_count"] > 0

st, v, _ = _req("/tools/search_similar_chunks/call",
                {"args": {"query": "胰岛素治疗糖尿病", "top_k": 3}})
rv = v["result"]
print(f"  search_similar_chunks -> 命中 {rv['count']} 条 dim={rv['dim']}")
for h in rv["hits"]:
    print(f"    score={h['score']} {h['chunk_id']} {(h['content'] or '')[:26]} entities={h['entities']}")
assert rv["count"] >= 1

st, f, _ = _req("/tools/search_fulltext/call", {"args": {"query": "胰岛素", "top_k": 5}})
print(f"  search_fulltext -> 命中 {f['result']['count']} 条")
assert f["result"]["count"] >= 1

st, h, _ = _req("/tools/hybrid_search/call",
                {"args": {"question": "胰岛素可以治疗糖尿病吗？剂量怎么定？",
                          "top_k": 4, "rrf_k": 60, "per_route_k": 5}})
rh = h["result"]
print(f"  hybrid_search -> 子问题 {rh['sub_query_count']} 个: {rh['sub_queries']}")
print(f"    各路命中 {rh['agreement']['per_route']}  重合 {rh['agreement']['multi_route']}")
for i, row in enumerate(rh["fused"], 1):
    print(f"    #{i} rrf={row['rrf_score']:.5f} 路={row['route_hits']} {(row.get('content') or '')[:24]}")
assert len(rh["sub_queries"]) >= 2, "LLM 应拆出多个子问题"
assert set(rh["agreement"]["per_route"]) >= {"vector", "fts"}
assert len(rh["fused"]) > 0

print("\n=== 待入库 / 重新入库 ===")
st, p, _ = _req("/chunks/pending?limit=10&preview_chars=10")
print(f"  待入库 {p['total_pending']} 条")
assert p["total_pending"] == 0

print("\n=== 失败重试：让 embedding 挂掉，验证 chunk 留在队列 ===")
try:
    urllib.request.urlopen("http://127.0.0.1:8890/mode?fail=emb", timeout=10)
    st, bad, _ = _req("/ingest/sentence", {"text": "这句应该只写进 sqlite 但卡住。"})
    print(f"  HTTP {st} qdrant={bad.get('qdrant')} neo4j={bad.get('neo4j')} promoted={bad.get('promoted')}")
    assert "qdrant" in bad and bad["qdrant"] == "failed", bad
    st, p2, _ = _req("/chunks/pending?limit=10&preview_chars=10")
    print(f"  队列中现在有 {p2['total_pending']} 条 -> 预览: "
          f"{[i['preview'] for i in p2['items']]}")
    assert p2["total_pending"] == 1
    item = p2["items"][0]
    print(f"  状态: qdrant={item['qdrant_status']} neo4j={item['neo4j_status']}")
    assert item["qdrant_status"] == "failed" and item["neo4j_status"] == "success"

    urllib.request.urlopen("http://127.0.0.1:8890/mode", timeout=10)
    print("\n  恢复 embedding，重新入库：")
    st, rr, _ = _req(f"/chunks/{item['chunk_id']}/reingest", {})
    rrr = rr["result"]
    print(f"  HTTP {st} qdrant={rrr.get('qdrant')} promoted={rrr.get('promoted')}")
    assert rrr["promoted"] is True
    st, p3, _ = _req("/chunks/pending")
    print(f"  队列清空: pending={p3['total_pending']}")
    assert p3["total_pending"] == 0
except AssertionError:
    raise
finally:
    try:
        urllib.request.urlopen("http://127.0.0.1:8890/mode", timeout=10)
    except Exception:
        pass

print("\n=== 三库统计 ===")
st, s, _ = _req("/stats")
print(f"  sqlite={s['result']['sqlite']}")
print(f"  qdrant={s['result']['qdrant']}")
print(f"  neo4j ={s['result']['neo4j']}")
assert s["result"]["qdrant"]["ok"] and s["result"]["neo4j"]["ok"]

print("\n=== HTTP 端到端全部通过 ===")
sys.exit(0)
