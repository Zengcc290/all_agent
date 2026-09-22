"""验证原生 Qdrant :6333 与 Neo4j :7687 都能正常连接与增删改查。

运行：python _verify_native_services.py
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
os.environ.setdefault("QDRANT_HOST", "127.0.0.1")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("QDRANT_COLLECTION", "kg_chunks")
os.environ.setdefault("EMBEDDING_DIM", "8")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if (detail and not cond) else ""))


async def main():
    from app import config
    from app.db.qdrant_store import QdrantStore
    from app.db.neo4j_store import Neo4jStore

    print("=== 连接配置 ===")
    print(f"  qdrant: {config.qdrant.host}:{config.qdrant.port} 集合={config.qdrant.collection} "
          f"mode={config.qdrant.mode}")
    print(f"  neo4j : {config.neo4j.uri} 用户={config.neo4j.user}")
    check("qdrant 走服务模式（非本地嵌入式）", config.qdrant.mode == "server")
    check("neo4j 用回环 bolt", config.neo4j.uri.startswith("bolt://127.0.0.1"))

    # ================= Qdrant =================
    print("\n=== Qdrant 原生服务 ===")
    q = QdrantStore()
    try:
        info = await q.ensure_collection()
        print(f"  ensure_collection: {info}")
        check("建/连集合成功", bool(info.get("dim")), str(info))
        h = await q.health()
        print(f"  health: ok={h['ok']} points={h['points']} dim={h['dim']} "
              f"distance={h['distance']} status={h['status']}")
        check("health 正常", h["ok"], str(h))
        check("距离度量是 Cosine", h["distance"] == "Cosine", h["distance"])

        # 写入 + 查询 + 删除
        vecs = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        ids = ["ck_vtest1", "ck_vtest2", "ck_vtest3"]
        payloads = [{"content": t, "document_id": "doc_verify"} for t in
                    ["向量测试一", "向量测试二", "接近向量测试一"]]
        out = await q.upsert_many(ids, vecs, payloads)
        print(f"  upsert {len(out)} 个点")
        check("upsert 成功", len(out) == 3)

        hits = await q.search(vecs[0], limit=3)
        print(f"  search 命中 {len(hits)} 条:")
        for x in hits:
            print(f"    {x['chunk_id']} score={x['score']:.4f} {x['content']}")
        check("search 有结果", len(hits) >= 3, str(hits))
        check("最相似的是自身（score≈1）", abs(hits[0]["score"] - 1.0) < 1e-3, str(hits[0]))
        check("第三点与第一点近似度高于第二点",
              hits[1]["chunk_id"] == "ck_vtest3" and hits[2]["chunk_id"] == "ck_vtest2",
              str([h["chunk_id"] for h in hits]))
        check("payload 完整带出", hits[0].get("content") and hits[0].get("document_id"), str(hits[0]))

        n = await q.count()
        print(f"  集合点数: {n}")
        check("count 反映写入", n >= 3, str(n))

        docs = await q.scroll(limit=5)
        check("scroll 能取回 payload", any(d.get("content") for d in docs), str(docs))

        d = await q.delete_points(["ck_vtest1", "ck_vtest2", "ck_vtest3"])
        print(f"  删除 {d} 个点")
        check("delete 成功", d == 3, str(d))
    except Exception as e:
        check("Qdrant 整体可用", False, f"{type(e).__name__}: {e}")
    finally:
        await q.close()

    # ================= Neo4j =================
    print("\n=== Neo4j 原生服务 ===")
    g = Neo4jStore()
    try:
        v = await g.verify()
        print(f"  verify: {v}")
        check("verify 连通", v.get("ok"), str(v))

        await g.clear_graph()
        n = await g.upsert_entities([
            {"key": "原生测试实体A", "name": "原生测试实体A", "type": "测试"},
            {"key": "原生测试实体B", "name": "原生测试实体B", "type": "测试"},
        ])
        check("upsert_entities", n == 2, str(n))

        r = await g.upsert_relations([{
            "source": "原生测试实体A", "src_key": "原生测试实体A",
            "target": "原生测试实体B", "tgt_key": "原生测试实体B",
            "predicate": "连接到", "directed": True, "time": "2026年", "chunk_id": "ck_vtest",
        }])
        check("upsert_relations", r == 1, str(r))

        ents = await g.get_all_entities()
        check("读回 2 个实体", len(ents) == 2, str(ents))
        rels = await g.get_all_relations()
        check("读回 1 条关系且带时间", len(rels) == 1 and rels[0]["time"] == "2026年", str(rels))

        mh = await g.multi_hop("原生测试实体A", hops=2, direction="both", limit=10)
        print(f"  multi_hop: 路径 {mh['path_count']} 条, 节点 {[n['key'] for n in mh['nodes']]}")
        # norm_key 会把 key 归一成小写，所以命中项是小写形式
        check("多跳查询可达 B",
              any(n["key"] == "原生测试实体b" for n in mh["nodes"])
              and any(n["name"] == "原生测试实体B" for n in mh["nodes"]), str(mh))

        snap = await g.graph_snapshot()
        print(f"  graph_snapshot: nodes={len(snap['nodes'])} links={len(snap['links'])}")
        check("快照 2 节点 1 连线", len(snap["nodes"]) == 2 and len(snap["links"]) == 1, str(snap))

        st = await g.stats()
        print(f"  stats: {st}")
        check("stats 正确", st == {"entities": 2, "relations": 1, "chunks": 0}, str(st))

        await g.clear_graph()
        check("清理后为空", (await g.stats())["entities"] == 0)
    except Exception as e:
        check("Neo4j 整体可用", False, f"{type(e).__name__}: {e}")
    finally:
        await g.close()

    print(f"\n=== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
    if FAIL:
        for f in FAIL:
            print("  -", f)
    return not FAIL


ok = asyncio.run(main())
sys.exit(0 if ok else 1)
