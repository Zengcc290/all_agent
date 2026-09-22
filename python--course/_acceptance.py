"""一键启动后的真实端到端验收：5 个服务全打通，跑一遍核心链路。

用法：先运行 启动所有服务.bat，再执行 python _acceptance.py
"""
import json
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
PASS, FAIL = [], []


def ck(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   -> {detail}" if (detail and not cond) else ""))


def req(path, obj=None, timeout=180):
    url = BASE + path
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8") if obj is not None else None
    r = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        r.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def qdrant_direct():
    """直连 qdrant 服务，确认是真服务而不是嵌入式。"""
    with urllib.request.urlopen("http://127.0.0.1:6333/collections", timeout=10) as r:
        return json.loads(r.read().decode())


print("=== 1. 服务可达性 ===")
for name, url in [
    ("Qdrant 服务", "http://127.0.0.1:6333/"),
    ("Neo4j HTTP", "http://127.0.0.1:7474/"),
    ("后端 API", "http://127.0.0.1:8000/api/health"),
    ("前端 Vite", "http://127.0.0.1:5173/"),
]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            body = r.read().decode("utf-8", "ignore")
            ck(f"{name} 可达", True)
            if name == "后端 API":
                h = json.loads(body)
                print(f"        health.ok={h.get('ok')} service={h.get('service')} ts={h.get('time')}")
                ck("后端 health.ok", h.get("ok") is True, str(h))
    except Exception as e:
        ck(f"{name} 可达", False, str(e))

print("\n=== 2. 确认后端连的是「真 Qdrant 服务」而非本地嵌入式 ===")
try:
    q = qdrant_direct()
    names = [c["name"] for c in q["result"]["collections"]]
    print(f"        qdrant :6333 上的集合: {names}")
    ck("qdrant 服务上有 kg_chunks 集合", "kg_chunks" in names, str(names))
except Exception as e:
    ck("qdrant 服务可达", False, str(e))

def stats_of(resp_json):
    """/api/stats 返回 {"tool","ok","args","result":{sqlite,qdrant,neo4j,config}}，
    老版本可能是平铺的，两者都兼容。"""
    inner = resp_json.get("result")
    if isinstance(inner, dict) and ("qdrant" in inner or "sqlite" in inner):
        return inner
    return resp_json


st, s = req("/api/stats")
sd = stats_of(s)
qd = sd["qdrant"]
print(f"        后端统计qdrant: {qd}")
ck("后端 qdrant 正常", qd.get("ok"), str(qd))
ck("后端 qdrant 维度与配置一致", "不一致" not in str(qd.get("warning", "")), str(qd.get("warning")))
ck("后端 qdrant 指向真实服务端口", sd["config"]["qdrant"].get("port") == 6333, str(sd["config"]["qdrant"]))
ck("后端 neo4j 正常", sd["neo4j"]["ok"], str(sd["neo4j"]))
ck("后端 sqlite 正常", sd["sqlite"]["chunks"] >= 0, str(sd["sqlite"]))

print("\n=== 3. 一句话入库（真 LLM 流式 + 真 qdrant 服务 + 真 neo4j）===")
st, base = req("/api/stats")
bd = stats_of(base)
b_chunks = bd["sqlite"]["chunks"]
b_ents = bd["neo4j"]["entities"]
b_pts = bd["qdrant"]["points"]
text = "验收测试：2024年5月，李医生在上海仁济医院为高血压患者调整了用药方案。"
st, r = req("/api/ingest/sentence", {"text": text, "source": "验收"})
print(f"        HTTP {st}  qdrant={r.get('qdrant')} neo4j={r.get('neo4j')} promoted={r.get('promoted')}")
print(f"        时间来源={r.get('time_source')} LLM抽到={r.get('time_llm_count')} 兜底={r.get('time_fallback_count')}")
# 实体/关系在 extracted 字段下
ex = r.get("extracted") or {}
for rel in (ex.get("relations") or [])[:6]:
    print(f"          {rel['source']} -[{rel['predicate']} -> {rel['time']}]-> {rel['target']}")
ck("入库成功", st == 200 and r.get("qdrant") == "success" and r.get("neo4j") == "success", str(r)[:300])
ck("成功转正到 chunks", r.get("promoted") is True, str(r.get("promoted")))
ck("抽到了实体", len(ex.get("entities") or []) >= 2, str(ex.get("entities")))
ck("抽到了关系", len(ex.get("relations") or []) >= 1, str(ex.get("relations")))
ck("有时间来源标记", r.get("time_source") in ("llm", "system"), str(r.get("time_source")))
ck("入库返回 chunk_id", bool(r.get("chunk_id")), str(r.get("chunk_id")))

print("\n=== 4. 数据是否真的落到了三个库 ===")
st, s2 = req("/api/stats")
s2d = stats_of(s2)
print(f"        sqlite chunks {b_chunks} -> {s2d['sqlite']['chunks']}")
print(f"        neo4j  entities {b_ents} -> {s2d['neo4j']['entities']}  relations {s2d['neo4j']['relations']}")
print(f"        qdrant  points {b_pts} -> {s2d['qdrant']['points']}")
ck("sqlite 增加了 chunk", s2d["sqlite"]["chunks"] == b_chunks + 1)
# 实体复用是正确行为：同一批实体再次出现时不会重复建，所以不强制 +1
ck("neo4j 实体数不减少且有关联", s2d["neo4j"]["entities"] >= b_ents
   and s2d["neo4j"]["chunks"] >= 1, str(s2d["neo4j"]))
ck("qdrant 增加了点", s2d["qdrant"]["points"] >= b_pts + 1, str(s2d["qdrant"]))

print("\n=== 5. 查询链路（向量 / FTS5 / 多跳 / 混合）===")
st, v = req("/api/tools/search_similar_chunks/call",
               {"args": {"query": "高血压 用药", "top_k": 3}})
print(f"        search_similar_chunks -> {len(v['result'])} 条")
ck("向量相似检索有结果", len(v["result"]) >= 1, str(v)[:200])

st, f = req("/api/tools/search_fulltext/call", {"args": {"query": "仁济医院", "top_k": 5}})
print(f"        search_fulltext       -> {len(f['result'])} 条")
ck("FTS5 全文检索有结果", len(f["result"]) >= 1, str(f)[:200])

st, h = req("/api/tools/hybrid_search/call",
               {"args": {"question": "高血压应该怎么治疗，要注意什么", "top_k": 5}})
res = h["result"]
agr = res.get("agreement", {})
print(f"        hybrid_search -> 子问题 {res.get('sub_query_count')} 个，"
      f"启用路 {res.get('routes_enabled')}，融合结果 {len(res.get('fused', []))} 条")
print(f"        各路命中: {agr.get('per_route')}  唯一候选 {agr.get('total_unique')}")
if res.get("fused"):
    top = res["fused"][0]
    print(f"           #1 rrf={top['rrf_score']:.5f} 命中路={top['route_hits']}")
    print(f"           {top['content'][:50]}")
# 注意：融合结果在 fused 字段（不是 results）
ck("混合检索有结果", len(res.get("fused", [])) >= 1, str(res)[:200])
ck("多路检索确实走了多路", len(res.get("routes_enabled", [])) >= 2, str(res.get("routes_enabled")))
ck("每条路都有命中", len(agr.get("per_route", {})) >= 2
   and all(v > 0 for v in agr.get("per_route", {}).values()), str(agr))
ck("LLM 拆分出了多个子问题", res.get("sub_query_count", 0) >= 2, str(res.get("sub_queries")))

st, g = req("/api/graph")
print(f"        /api/graph -> nodes={len(g['nodes'])} links={len(g['links'])}")
ck("图数据可取", len(g["nodes"]) >= 2 and len(g["links"]) >= 1)

if g.get("nodes"):
    start = g["nodes"][0]["id"]
    st, mh = req("/api/tools/query_graph/call", {"args": {"entity": start, "hops": 3}})
    m = mh["result"]
    print(f"        multi_hop from {start} -> 路径 {m.get('path_count')} 条")
    ck("多跳查询可用", m.get("path_count", 0) >= 0 and "nodes" in m, str(m)[:200])

print("\n=== 6. 工具系统 ===")
st, tl = req("/api/tools")
tools = [t["name"] for t in tl["tools"]]
print(f"        已注册工具 {len(tools)} 个")
ck("工具已自动发现", len(tools) >= 15, str(tools))
for need in ("ingest_sentence", "search_similar_chunks", "search_fulltext",
             "hybrid_search", "get_all_entities", "get_all_relations", "query_graph"):
    ck(f"工具 {need} 已注册", need in tools, str(tools))

st, rc = req("/api/tools/call_many", {"calls": [
    {"tool": "get_stats", "args": {}},
    {"tool": "list_pending_chunks", "args": {}},
    {"tool": "get_current_time", "args": {}},
]})
n = len(rc.get("results", [])) if isinstance(rc, dict) else 0
print(f"        call_many -> {n} 个结果")
ck("并行工具调用可用", n == 3, str(rc)[:200])

# 参数校验：故意传个不存在的参数，应被拒绝
st, bad = req("/api/tools/search_similar_chunks/call", {"args": {"query": "x", "limit": 3}})
print(f"        传错参数 limit -> HTTP {st}")
ck("非法参数被拒", st in (400, 422) and "未知参数" in str(bad), str(bad)[:200])

print(f"\n=== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
for f in FAIL:
    print("  -", f)
sys.exit(0 if not FAIL else 1)
