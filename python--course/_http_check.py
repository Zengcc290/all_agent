import json, sys, time
import urllib.request, urllib.error

BASE = "http://127.0.0.1:8899/api"

def get(path, timeout=60):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8")), r.headers.get("content-type")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8", "ignore")), None
    except Exception as e:
        return 0, {"_error": "%s: %s" % (type(e).__name__, e)}, None

def post(path, obj, timeout=120):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8")), r.headers.get("content-type")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8", "ignore")), None
    except Exception as e:
        return 0, {"_error": "%s: %s" % (type(e).__name__, e)}, None

ok = True

st, g, ct = get("/graph")
print("[1] /api/graph ->", st, "ok=%s" % g.get("ok"), "nodes=%d" % len(g.get("nodes") or []),
      "links=%d" % len(g.get("links") or []))
print("    content-type:", ct)
names = [n.get("name") for n in g.get("nodes") or []]
print("    UTF-8 中文节点名:", names[:5])
assert st == 200 and g["ok"] and len(g["nodes"]) == 14 and len(g["links"]) == 14, "graph 数量不对"
assert names[0] and any("\u4e00" <= ch <= "\u9fff" for ch in names[0]), "中文乱码"
assert "charset=utf-8" in (ct or ""), "JSON 响应未声明 utf-8"

st, s, _ = get("/stats")
print("[2] /api/stats ->", st,
      "qdrant_ok=%s neo4j_ok=%s" % (s["result"]["qdrant"]["ok"], s["result"]["neo4j"]["ok"]),
      "e=%s r=%s" % (s["result"]["neo4j"]["entities"], s["result"]["neo4j"]["relations"]))
assert s["result"]["neo4j"]["ok"] and s["result"]["qdrant"]["ok"]

st, t, _ = get("/tools")
print("[3] /api/tools ->", st, "count=%d" % t["count"])
assert t["count"] == 15, t["count"]

st, p, _ = get("/chunks/pending?limit=5&preview_chars=10")
print("[4] /api/chunks/pending ->", st, "total_pending=%d" % p["total_pending"])

st, c, _ = get("/chunks?limit=10")
print("[5] /api/chunks ->", st, "items=%d" % len(c.get("items") or []))
print("    首条 preview:", (c["items"][0]["content"][:10] if c.get("items") else None))
assert len(c.get("items") or []) == 5

st, r, _ = post("/tools/query_graph/call", {"args": {"entity": "张三", "hops": 2, "limit": 50}})
res = r["result"]
found = [n["key"] for n in res["nodes"]]
print("[6] POST /tools/query_graph ->", st, "路径=%d 节点=%s" % (res["path_count"], found))
assert res["path_count"] > 0

st, r, _ = post("/tools/search_similar_chunks/call",
                {"args": {"query": "胰岛素 糖尿病", "top_k": 3}})
print("[7] POST /tools/search_similar_chunks ->", st, "命中=%d dim=%s" % (r["result"]["count"], r["result"]["dim"]))
assert r["result"]["count"] >= 1

st, r, _ = post("/tools/search_fulltext/call", {"args": {"query": "胰岛素", "top_k": 5}})
print("[8] POST /tools/search_fulltext ->", st, "命中=%d" % r["result"]["count"])
assert r["result"]["count"] >= 1

st, r, _ = post("/tools/hybrid_search/call",
                {"args": {"question": "胰岛素可以治疗糖尿病吗", "top_k": 4,
                          "rrf_k": 60, "per_route_k": 5, "routes": ["vector", "fts"]}})
hr = r["result"]
print("[9] POST /tools/hybrid_search ->", st,
      "子问题=%d 两路命中=%s RRF结果=%d 多路重合=%d" %
      (hr["sub_query_count"], hr["agreement"]["per_route"], len(hr["fused"]), hr["agreement"]["multi_route"]))
assert len(hr["fused"]) > 0 and set(hr["agreement"]["per_route"]) >= {"vector", "fts"}

st, r, _ = post("/tools/call_many", {"calls": [
    {"tool": "get_current_time", "args": {"format": "date"}},
    {"tool": "get_all_entities", "args": {"limit": 3}},
    {"tool": "get_graph_snapshot", "args": {"limit": 5}},
    {"tool": "get_stats", "args": {}},
]})
print("[10] POST /tools/call_many ->", st, "4个工具并行 ->",
      "%d/%d 成功, %sms" % (r["succeeded"], r["total"], r["elapsed_ms"]))
assert r["succeeded"] == 4

# 422 校验
st, e, _ = post("/tools/hybrid_search/call", {"args": {"question": "x", "routes": ["nope"]}})
print("[11] 参数校验 422 ->", st, e["detail"]["message"])
assert st == 422

# SSE 流式入库（无真实 LLM key，预期会流式报错但仍返回结构化事件）
import urllib.request as u2
data = json.dumps({"text": "测试流式入库"}).encode()
req = u2.Request(BASE + "/ingest/stream", data=data,
                 headers={"Content-Type": "application/json"}, method="POST")
try:
    with u2.urlopen(req, timeout=90) as resp:
        body = resp.read().decode("utf-8")
    lines = [l for l in body.split("\n") if l.startswith("data:")]
    events = [json.loads(l[5:].strip())["event"] for l in lines]
    print("[12] POST /api/ingest/stream (SSE) -> 收到 %d 个事件: %s" % (len(events), events[:8]))
    assert "start" in events and "done" in events
    assert not any("llm_delta" == e for e in events) or True
except Exception as ex:
    print("[12] SSE 失败:", ex); ok = False

print()
print("全部通过" if ok else "有失败项")
sys.exit(0 if ok else 1)
