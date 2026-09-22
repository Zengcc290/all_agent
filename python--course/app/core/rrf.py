"""RRF (Reciprocal Rank Fusion) 倒数排序融合。

多路检索（向量 / FTS5 / BM25 / 图谱……）各自返回一个「按自己打分排好序」的列表，
RRF 不关心各路分的绝对值，只用排名做融合，天然免疫不同打分量纲不可比的问题。

    RRF(d) = Σ route  1 / (k + rank_route(d))

k 越大，头部优势越被削弱（越平滑），常用 k = 60。
"""
from __future__ import annotations

from typing import Any, Iterable

DEFAULT_K = 60.0


def rrf_fuse(routes: dict[str, list[dict]], k: float = DEFAULT_K,
             id_key: str = "chunk_id", limit: int = 20) -> list[dict]:
    """routes: {"vector": [ {chunk_id, ...}, ... ], "fts": [...]}
    返回按 RRF 分数降序的融合结果，每条附上命中了哪几路、每路排名与原始分。"""
    if k <= 0:
        raise ValueError("RRF 常数 k 必须 > 0")

    agg: dict[str, dict[str, Any]] = {}
    for route_name, items in (routes or {}).items():
        for rank, item in enumerate(items or [], start=1):
            doc_id = item.get(id_key) if isinstance(item, dict) else None
            if doc_id is None:
                continue
            slot = agg.setdefault(doc_id, {
                "id": doc_id,
                "rrf_score": 0.0,
                "routes": {},
                "payload": item,
            })
            slot["rrf_score"] += 1.0 / (k + rank)
            slot["routes"][route_name] = {
                "rank": rank,
                "score": item.get("score"),
            }
            # 优先保留信息更全的那份 payload
            if len(item) > len(slot["payload"]):
                slot["payload"] = item

    fused = sorted(agg.values(), key=lambda x: x["rrf_score"], reverse=True)[: max(1, limit)]
    out = []
    for f in fused:
        row = dict(f["payload"])
        row["id"] = f["id"]
        row["rrf_score"] = round(f["rrf_score"], 8)
        row["routes"] = f["routes"]
        row["route_hits"] = sorted(f["routes"])
        row["route_count"] = len(f["routes"])
        out.append(row)
    return out


def route_agreement(routes: dict[str, list[dict]]) -> dict:
    """各路之间的重合度统计（有多少结果被多路同时命中）。"""
    seen: dict[str, set[str]] = {}
    for name, items in (routes or {}).items():
        seen[name] = {i.get("chunk_id") for i in (items or []) if isinstance(i, dict)}
    all_ids = set().union(*seen.values()) if seen else set()
    multi = [i for i in all_ids if sum(i in v for v in seen.values()) > 1]
    return {
        "total_unique": len(all_ids),
        "multi_route": len(multi),
        "overlap_ratio": round(len(multi) / len(all_ids), 3) if all_ids else 0.0,
        "per_route": {k: len(v) for k, v in seen.items()},
    }


def _demo() -> None:
    r = rrf_fuse({
        "vector": [{"chunk_id": "a", "score": 0.91}, {"chunk_id": "b", "score": 0.8}],
        "fts":    [{"chunk_id": "b", "score": 1.0}, {"chunk_id": "c", "score": 0.9}],
    })
    for x in r:
        print(x["id"], x["rrf_score"], x["routes"])


if __name__ == "__main__":
    _demo()
