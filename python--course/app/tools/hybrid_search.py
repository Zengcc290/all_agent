"""工具：多路混合检索 —— LLM 拆分子问题 -> 向量 + FTS5 多路并行 -> RRF 融合。

一条提问往往是「复合问题」：
  「糖尿病患者能不能用胰岛素，剂量怎么定？」
可以拆成：
  1. 胰岛素适用于哪些疾病
  2. 胰岛素的用法用量
  3. 糖尿病患者使用胰岛素的注意事项

每一路子问题都并行跑两条检索路：
  · 向量路  —— qdrant 相似度（语义召回）
  · 词法路  —— sqlite FTS5 全文检索（关键词精准召回）
最后用 RRF 把「子问题 × 检索路」的所有排序融合成一份最终列表。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.registry import registry
from app.core.rrf import DEFAULT_K, route_agreement, rrf_fuse
from app.core.validation import Tool, ToolParam, ToolValidationError
from app.db.qdrant_store import vector_store
from app.db.sqlite_store import store
from app.llm import chat, embed_texts

ROUTES = ("vector", "fts")

DECOMPOSE_SYSTEM = """你是检索查询改写专家。用户会给你一个问题或查询，
请把它拆成 2 到 5 个更具体、更适合检索的独立子问题。

要求：
- 只输出 JSON：{"sub_queries": ["子问题1", "子问题2", ...]}
- 每个子问题只包含一个可独立检索的意图，保留关键实体与限定词
- 不要输出任何解释、不要加 markdown 代码围栏
- 如果原问题本身已经很具体，就返回 1 条即可"""


async def _decompose(question: str, max_sub: int = 5) -> list[str]:
    """让 LLM 把一个问题拆成多个可独立检索的子问题；失败时退化为原问题。"""
    try:
        raw = await chat([
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user", "content": f"请拆解下面这个问题：\n{question}"},
        ], temperature=0.2, max_tokens=400)
        from app.core.parser import extract_json_block
        data = extract_json_block(raw)
        if isinstance(data, dict) and isinstance(data.get("sub_queries"), list):
            subs = [str(s).strip() for s in data["sub_queries"] if str(s).strip()]
            if subs:
                return subs[:max_sub]
    except Exception:  # noqa: BLE001
        pass
    return [question]


async def _vector_route(query: str, top_k: int, score_threshold: float = 0.0) -> list[dict]:
    vecs = await embed_texts([query])
    if not vecs:
        return []
    hits = await vector_store.search(vecs[0], limit=top_k)
    if score_threshold:
        hits = [h for h in hits if (h.get("score") or 0) >= score_threshold]
    return hits


async def _fts_route(query: str, top_k: int) -> list[dict]:
    return await asyncio.to_thread(store.fts_search, query, top_k)


async def hybrid_search(question: str, top_k: int = 8, per_route_k: int | None = None,
                        rrf_k: float = DEFAULT_K, use_llm_split: bool = True,
                        routes: list[str] | None = None,
                        score_threshold: float = 0.0) -> dict:
    """多路混合检索主流程。"""
    question = (question or "").strip()
    if not question:
        raise ToolValidationError("必填参数 question 不能为空")
    if rrf_k <= 0:
        raise ToolValidationError(f"参数 rrf_k 必须大于 0，收到 {rrf_k}")

    wanted = set(routes or ROUTES)
    bad = sorted(wanted - set(ROUTES))
    if bad:
        raise ToolValidationError(f"参数 routes 含未知检索路 {bad}，可选 {list(ROUTES)}")
    if not wanted:
        raise ToolValidationError(f"参数 routes 不能为空数组，可选 {list(ROUTES)}")
    enabled = [r for r in ROUTES if r in wanted]
    if not enabled:
        raise ValueError(f"routes 不能为空数组，可选 {list(ROUTES)}")
    per_k = per_route_k or max(top_k * 2, 10)
    t0 = time.perf_counter()

    # 1) LLM 把问题拆成多个子问题（可关掉）
    sub_queries = (await _decompose(question)) if use_llm_split else [question]
    if not sub_queries:
        sub_queries = [question]

    # 2) 子问题 × 检索路，全部并行（任一路抛错只记 warn，不影响其它路）
    async def one(sub: str, route: str) -> dict:
        try:
            hits = (await _vector_route(sub, per_k, score_threshold)) if route == "vector" \
                else (await _fts_route(sub, per_k))
        except Exception as e:  # noqa: BLE001
            hits = [{"chunk_id": None, "error": f"{type(e).__name__}: {e}", "score": 0}]
        return {"sub_query": sub, "route": route, "hits": hits}

    route_results = await asyncio.gather(*[
        one(sub, route) for sub in sub_queries for route in enabled
    ])

    # 3) 按路汇总 + 按子问题汇总（供前端分别展示）
    jobs_by_route: dict[str, list[dict]] = {}
    per_query: dict[str, dict[str, list[dict]]] = {}
    for rr in route_results:
        per_query.setdefault(rr["sub_query"], {})[rr["route"]] = rr["hits"]
        jobs_by_route.setdefault(rr["route"], []).extend(rr["hits"])

    # 4) RRF 融合
    fused = rrf_fuse(jobs_by_route, k=rrf_k, limit=top_k)
    agreement = route_agreement(
        {k: [h for h in v if h.get("chunk_id")] for k, v in jobs_by_route.items()}
    )

    # 5) 附上命中该 chunk 的子问题与关联实体
    hit_ids: dict[str, set[str]] = {}
    for rr in route_results:
        for h in rr["hits"]:
            if h.get("chunk_id"):
                hit_ids.setdefault(h["chunk_id"], set()).add(rr["sub_query"])
    for row in fused:
        cid = row.get("chunk_id")
        row["sub_queries_hit"] = sorted(hit_ids.get(cid) or [])
        if cid:
            row["entities"] = [e.get("entity_name")
                               for e in await asyncio.to_thread(store.get_chunk_entities, cid)]

    return {
        "ok": True,
        "question": question,
        "sub_queries": sub_queries,
        "sub_query_count": len(sub_queries),
        "routes_enabled": enabled,
        "rrf_k": rrf_k,
        "agreement": agreement,
        "per_query": per_query,
        "fused": fused,
        "count": len(fused),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


async def _handler(question: str, top_k: int = 8, per_route_k: int | None = None,
                   rrf_k: float = DEFAULT_K, use_llm_split: bool = True,
                   routes: list[str] | None = None,
                   score_threshold: float = 0.0) -> dict:
    return await hybrid_search(question, top_k, per_route_k, rrf_k, use_llm_split,
                              routes, score_threshold)


registry.register(Tool(
    name="hybrid_search",
    description=(
        "多路混合检索（向量 + FTS5 + RRF）。"
        "① LLM 把用户的提问拆成多个可独立检索的子问题（use_llm_split=false 则跳过拆分）"
        "② 每个子问题并行跑多条检索路：vector（qdrant 向量相似度，语义召回）"
        "与 fts（sqlite FTS5 全文检索，关键词精准召回）"
        "③ 用 RRF 倒数排序融合，把所有「子问题 × 检索路」的排名合并成一份最终排序"
        "④ 返回融合结果，含每条命中了哪几路、每路排名、命中的子问题、以及 chunk 关联实体。"
        "适合复合问题：任一单路往往只命中一部分意图，融合后召回与精度都更好。"
    ),
    params=[
        ToolParam("question", "string", "用户的提问或查询（可以是复合问题）",
                  required=True, max_length=5000),
        ToolParam("top_k", "integer", "RRF 融合后最终返回多少条（默认 8）",
                  default=8, min_value=1, max_value=100),
        ToolParam("per_route_k", "integer",
                  "每一路每个子问题取前 k 条参与融合；省略时取 max(top_k*2, 10)",
                  default=None, min_value=1, max_value=100),
        ToolParam("rrf_k", "number", "RRF 常数 k，越大头部优势越平滑，常用 60",
                  default=60, min_value=1, max_value=1000),
        ToolParam("use_llm_split", "boolean", "是否让 LLM 先把问题拆成多个子问题（默认 true）",
                  default=True),
        ToolParam("routes", "array",
                  "启用的检索路，子元素只能取 vector / fts；省略时两条都开",
                  default=None, items="string"),
        ToolParam("score_threshold", "number", "向量路相似度下限，0 表示不过滤",
                  default=0.0, min_value=0.0, max_value=1.0),
    ],
    handler=_handler,
    tags=["检索", "混合检索", "RRF", "qdrant", "sqlite"],
    timeout=300,
))
