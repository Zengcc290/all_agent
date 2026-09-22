"""往真实 neo4j / sqlite / qdrant 灌入一批种子数据，便于人工查看前端效果。

运行：python _seed_demo.py
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
os.environ.setdefault("QDRANT_LOCAL_PATH", str(ROOT / "data" / "qdrant"))
os.environ.setdefault("SQLITE_PATH", str(ROOT / "data" / "kg.db"))
os.environ.setdefault("EMBEDDING_DIM", "8")

DEMO = [
    ("张三", "人物", [("就职于", "北京协和医院", True, "2023-07-01"),
                     ("开具", "胰岛素", True, "2024-05-10"),
                     ("关注", "糖尿病", False, "2024-05-10")]),
    ("北京协和医院", "组织", [("位于", "北京", True, "1921-01-01"), ("合作", "诺和诺德", False, "2024-03-15")]),
    ("胰岛素", "药物", [("治疗", "糖尿病", True, "2024-05-10"), ("由..生产", "诺和诺德", True, "2024-05-01")]),
    ("糖尿病", "疾病", [("引发", "视网膜病变", True, "2024-05-10"), ("监测", "糖化血红蛋白", False, "2024-05-10")]),
    ("王伟", "人物", [("开发", "血糖监测系统", True, "2024-04-20"), ("合作", "李强", False, "2024-04-15")]),
    ("李强", "人物", [("任职于", "阿里巴巴", True, "2020-09-01")]),
    ("张仲景", "人物", [("撰写", "伤寒杂病论", True, "210年"), ("治疗", "糖尿病", False, "210年")]),
]


async def main():
    from app.db.neo4j_store import graph_store
    from app.db.sqlite_store import store
    from app.db.qdrant_store import vector_store

    print("清理旧数据...")
    await graph_store.clear_graph()
    store.clear_all()
    # qdrant 集合清空（local 模式下重新建集合同样有效）
    try:
        await vector_store.close()
    except Exception:
        pass

    print(f"\n写入 {len(DEMO)} 个实体及其关系...")
    entities = [{"key": k, "name": k, "type": t} for k, t, _ in DEMO]
    n = await graph_store.upsert_entities(entities)
    print(f"  entities: {n}")

    total_rel = 0
    for src, _, rels in DEMO:
        tris = [{"source": src, "src_key": src, "target": tgt, "tgt_key": tgt,
                 "predicate": pred, "directed": d, "time": tm, "chunk_id": "seed"}
                for pred, tgt, d, tm in rels]
        total_rel += await graph_store.upsert_relations(tris)
    print(f"  relations: {total_rel}")

    # chunk + FTS + qdrant 向量
    texts = [
        ("ck_seed1", "2024年5月，张医生在北京协和医院为糖尿病患者开具了胰岛素，并建议定期监测糖化血红蛋白。"),
        ("ck_seed2", "胰岛素注射液由诺和诺德生产，用于治疗糖尿病引起的血糖升高，常规剂量为每日两次皮下注射。"),
        ("ck_seed3", "糖尿病视网膜病变需要每年眼底检查，张仲景曾在伤寒杂病论中记述相关治疗。"),
        ("ck_seed4", "王伟与李强合作开发了一套血糖监测系统，可实时上传数据到阿里云。"),
        ("ck_seed5", "北京协和医院与诺和诺德在新药临床研究方面保持长期合作。"),
    ]
    ents_by_key = {e["key"]: e for e in entities}
    linked = {"ck_seed1": ["张医生", "北京协和医院", "糖尿病", "胰岛素"],
              "ck_seed2": ["胰岛素", "诺和诺德", "糖尿病"],
              "ck_seed3": ["糖尿病", "视网膜病变", "张仲景", "伤寒杂病论"],
              "ck_seed4": ["王伟", "李强", "血糖监测系统", "阿里巴巴"],
              "ck_seed5": ["北京协和医院", "诺和诺德"]}
    for cid, text in texts:
        doc = store.add_document(text, "seed")
        kid = store.enqueue_chunk(doc["id"], text, 0)
        store.update_chunk_status(kid, qdrant_status="success", neo4j_status="success")
        store.promote_chunk(kid)
        keys = linked.get(cid) or []
        if keys:
            store.set_chunk_entities(kid, [ents_by_key[k] for k in keys if k in ents_by_key])
        v = [0.0] * 8
        for ch in text:
            v[hash(ch) % 8] += 1.0 / len(text)
        await vector_store.ensure_collection()
        await vector_store.upsert_chunk(kid, v, {"content": text, "document_id": doc["id"],
                                                 "created_at": doc["created_at"]})
        print(f"  chunk {cid} -> {kid}")

    gst = await graph_store.stats()
    st = store.stats()
    print(f"\nneo4j : {gst}")
    print(f"sqlite: {st}")
    print(f"qdrant points: {await vector_store.count()}")

    entities_full = await graph_store.get_all_entities()
    rels = await graph_store.get_all_relations()
    print(f"\n实体 {len(entities_full)} 个，关系 {len(rels)} 条")
    for r in rels[:12]:
        arrow = "->" if r["directed"] else "--"
        print(f"  {r['src_label']} -[{r['predicate']} {arrow} {r['time']}]-> {r['tgt_label']}")

    snap = await graph_store.graph_snapshot()
    print(f"\ngraph_snapshot: nodes={len(snap['nodes'])} links={len(snap['links'])} "
          f"有向={sum(1 for l in snap['links'] if l['directed'])} "
          f"无向={sum(1 for l in snap['links'] if not l['directed'])}")
    await graph_store.close()
    await vector_store.close()


asyncio.run(main())
