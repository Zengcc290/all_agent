"""Neo4j 实机验证：把 app/db/neo4j_store.py 里每条 Cypher 都跑一遍。

运行：python _neo4j_verify.py     （需要本地 neo4j 已启动）
"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "neo4j123456")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))


async def main():
    from app.db.neo4j_store import graph_store

    print("=== connect ===")
    v = await graph_store.verify()
    print(f"  verify: {v}")
    check("verify() 连通", bool(v.get("ok")), str(v))

    # 先清干净
    print("\n=== clear ===")
    try:
        await graph_store.clear_graph()
    except Exception as e:
        print(f"  clear_graph: {e}")

    print("\n=== upsert_entities（含新建 + 复用 + 别名）===")
    ents = [
        {"key": "张三", "name": "张三", "type": "人物", "time": "2024年度"},
        {"key": "北京协和医院", "name": "北京协和医院", "type": "组织"},
        {"key": "胰岛素", "name": "胰岛素", "type": "药物", "time": "2024-05-01",
         "aliases": ["insulin", "胰岛素注射液"]},
        {"key": "糖尿病", "name": "糖尿病", "type": "疾病"},
    ]
    n = await graph_store.upsert_entities(ents)
    print(f"  upsert {n} entities")
    check("upsert_entities 返回数量", n == 4, f"got {n}")

    # 再写一次 = 复用，不应新增
    n2 = await graph_store.upsert_entities(ents)
    st = await graph_store.stats()
    check("重复 upsert 不产生新实体", st["entities"] == 4, f"entities={st['entities']}")

    # 别名追加（同 key，新别名）
    await graph_store.upsert_entities([
        {"key": "胰岛素", "name": "胰岛素", "type": "药物", "aliases": ["insulin", "重组人胰岛素"]}
    ])
    all_e = await graph_store.get_all_entities()
    ins = next(e for e in all_e if e["key"] == "胰岛素")
    check("别名合并写入", set(ins.get("aliases") or []) >= {"insulin", "重组人胰岛素"}, str(ins))

    print("\n=== upsert_relations ===")
    tris = [
        {"source": "张三", "src_key": "张三", "target": "北京协和医院", "tgt_key": "北京协和医院",
         "predicate": "就职于", "directed": True, "time": "2024-01-01", "chunk_id": "ck_test1",
         "evidence": "张医生在北京协和医院工作"},
        {"source": "胰岛素", "src_key": "胰岛素", "target": "糖尿病", "tgt_key": "糖尿病",
         "predicate": "治疗", "directed": True, "time": "2024-05-01", "chunk_id": "ck_test1"},
        {"source": "张三", "src_key": "张三", "target": "胰岛素", "tgt_key": "胰岛素",
         "predicate": "开具", "directed": False, "time": "2024-05-01", "chunk_id": "ck_test2"},
    ]
    r = await graph_store.upsert_relations(tris)
    print(f"  upsert {r} relations")
    check("upsert_relations 返回数量", r == 3, f"got {r}")

    rels = await graph_store.get_all_relations()
    print(f"  get_all_relations -> {len(rels)} 条")
    check("关系可读回", len(rels) == 3, f"got {len(rels)}")
    if rels:
        sample = rels[0]
        for field in ("predicate", "directed", "time", "src", "tgt", "src_label", "tgt_label",
                      "chunk_id", "evidence", "rid"):
            check(f"get_all_relations 含字段 {field}", field in sample, f"sample keys={list(sample)}")
        undirected = [x for x in rels if x["predicate"] == "开具"]
        check("无向关系 directed=false", bool(undirected) and undirected[0]["directed"] is False,
              str(undirected))

    ents2 = await graph_store.get_all_entities()
    print(f"  get_all_entities -> {len(ents2)} 个，字段: {list(ents2[0])}")
    check("get_all_entities 含 out_rels/in_rels",
          all("out_rels" in e and "in_rels" in e for e in ents2))
    degs = {e["key"]: (e["out_rels"], e["in_rels"]) for e in ents2}
    # 张三：就职于(->协和) + 开具(--胰岛素) 都是出向；无向边只存一份，不会双向重复计数
    check("度数统计正确（张三 out=2 in=0）", degs["张三"] == (2, 0), str(degs))
    check("度数统计正确（胰岛素 out=1 in=1）", degs["胰岛素"] == (1, 1), str(degs))

    print("\n=== link_chunk / get_chunks_of_entity ===")
    m = await graph_store.link_chunk("ck_test1", ["张三", "北京协和医院", "胰岛素", "糖尿病"], "测试句子一")
    print(f"  link_chunk 关联 {m} 个实体")
    ch = await graph_store.get_chunks_of_entity("张三")
    check("get_chunks_of_entity 可通过实体反查 chunk", "ck_test1" in ch, str(ch))
    gc = await graph_store.get_chunk("ck_test1")
    check("get_chunk 可读回 chunk 节点", bool(gc and gc.get("content") == "测试句子一"), str(gc))

    print("\n=== multi_hop（核心：之前无法验证）===")
    for hops, direction in ((1, "both"), (2, "both"), (2, "out"), (5, "both")):
        res = await graph_store.multi_hop("张三", hops=hops, direction=direction, limit=50)
        print(f"  hops={hops} dir={direction} -> {res['path_count']} 条路径, "
              f"{len(res['nodes'])} 节点, rels={len(res['rels'])}")
        check(f"multi_hop hops={hops}/{direction} 有结果", res["path_count"] > 0, str(res))
        if res["paths"]:
            p0 = res["paths"][0]
            check(f"  hops={hops} 路径含 nodes", len(p0["nodes"]) >= 2, str(p0))
            check(f"  hops={hops} 路径含 rels", len(p0["rels"]) >= 1, str(p0))
    res5 = await graph_store.multi_hop("张三", hops=5, direction="both", limit=100)
    check("5 跳可达糖尿病（张三->胰岛素->糖尿病）",
          any(n["key"] == "糖尿病" for n in res5["nodes"]), str([n["key"] for n in res5["nodes"]]))
    # 无向应该能反向走
    res_n = await graph_store.multi_hop("糖尿病", hops=2, direction="both", limit=50)
    check("无向遍历可反向发现张三", any(n["key"] == "张三" for n in res_n["nodes"]),
          str([n["key"] for n in res_n["nodes"]]))
    res_o = await graph_store.multi_hop("糖尿病", hops=2, direction="out", limit=50)
    check("有向遍历不会反向走", not any(n["key"] == "张三" for n in res_o["nodes"]),
          str([n["key"] for n in res_o["nodes"]]))
    # 未知起点
    res_miss = await graph_store.multi_hop("不存在的实体", hops=2, limit=10)
    check("未知起点返回 0 条而不报错", res_miss["path_count"] == 0, str(res_miss))

    print("\n=== neighbors ===")
    nb = await graph_store.neighbors("张三")
    print(f"  neighbors(张三) -> {nb['count']} 条")
    check("neighbors 找齐相邻关系（就职于 + 开具）", nb["count"] == 2, str(nb))
    # 无向边本身存在，所以从胰岛素侧也能看到
    nb2 = await graph_store.neighbors("胰岛素")
    check("neighbors 能看到无向边", any(r["predicate"] == "开具" for r in nb2["relations"]), str(nb2))

    print("\n=== graph_snapshot ===")
    snap = await graph_store.graph_snapshot()
    print(f"  nodes={len(snap['nodes'])} links={len(snap['links'])}")
    check("快照有节点", len(snap["nodes"]) >= 4)
    check("快照有连线", len(snap["links"]) >= 3)
    lnk = snap["links"][0]
    for f in ("id", "source", "target", "predicate", "directed", "time"):
        check(f"link 含字段 {f}", f in lnk, f"keys={list(lnk)}")
    check("快照含 group（前端配色）", all("group" in n for n in snap["nodes"]))

    print("\n=== get_entity / update / delete ===")
    e = await graph_store.get_entity("胰岛素")
    check("get_entity 命中", bool(e), str(e))
    check("get_entity 未命中返回 None", await graph_store.get_entity("不存在") is None)

    print("\n=== delete_relation / delete_entity ===")
    rels_before = await graph_store.get_all_relations()
    target_rel = next(r for r in rels_before if r["predicate"] == "开具")
    await graph_store.delete_relation(str(target_rel["rid"]))
    rels_after = await graph_store.get_all_relations()
    check("delete_relation 生效", len(rels_after) == len(rels_before) - 1,
          f"{len(rels_before)} -> {len(rels_after)}")

    await graph_store.delete_entity("胰岛素")
    all_e2 = await graph_store.get_all_entities()
    check("DETACH DELETE 级联删除关系", not any(x["key"] == "胰岛素" for x in all_e2),
          str([x["key"] for x in all_e2]))
    rels_final = await graph_store.get_all_relations()
    check("级联删关系后只剩 1 条", len(rels_final) == 1, str(len(rels_final)))

    print("\n=== stats ===")
    st2 = await graph_store.stats()
    print(f"  {st2}")
    check("stats 键完整", set(st2) == {"entities", "relations", "chunks"}, str(st2))

    print("\n=== query_graph 工具（走注册表）===")
    from app.core.registry import registry
    info = await registry.discover()
    q = await registry.call("query_graph", {"entity": "张三", "hops": 2, "limit": 50})
    res_q = q["result"]
    print(f"  query_graph -> {res_q['path_count']} 条路径, 节点 {len(res_q['nodes'])}")
    check("query_graph 工具可用", res_q["path_count"] > 0, str(res_q))
    qd = await registry.call("get_all_entities", {"limit": 100})
    check("get_all_entities 工具可用", qd["result"]["count"] >= 1)

    print("\n=== 收尾清库 ===")
    await graph_store.clear_graph()
    st3 = await graph_store.stats()
    check("clear_graph 清空", st3["entities"] == 0 and st3["relations"] == 0, str(st3))
    await graph_store.close()

    print(f"\n=== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
    if FAIL:
        print("失败项:")
        for f in FAIL:
            print(f"  - {f}")
    return not FAIL


ok = asyncio.run(main())
sys.exit(0 if ok else 1)
