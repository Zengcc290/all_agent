"""工具：run_tools_parallel —— 演示/使用异步并行调用多个工具。

互不依赖的工具会被 asyncio.gather 同时执行，总耗时约等于最慢的那个，
而不是所有耗时之和。前端「并行调用」面板与 LLM Agent 循环都会用到它。
"""
from __future__ import annotations

from typing import Any

from app.core.registry import registry
from app.core.validation import Tool, ToolParam


async def _run(calls: list[dict], fail_fast: bool = False) -> dict:
    if not calls:
        raise ValueError("参数 calls 不能为空数组")
    if len(calls) > 20:
        raise ValueError("单次并行调用最多 20 个工具")
    clean = []
    for i, c in enumerate(calls):
        if not isinstance(c, dict):
            raise ValueError(f"calls[{i}] 必须是 object，形如 {{\"tool\": \"...\", \"args\": {{...}}}}")
        name = (c.get("tool") or c.get("name") or "").strip()
        if not name:
            raise ValueError(f"calls[{i}] 缺少 tool 字段")
        if not registry.has(name):
            raise ValueError(f"calls[{i}] 指定的工具 {name!r} 未注册，已注册: {registry.names()}")
        clean.append({"tool": name, "args": c.get("args") or {}})

    results = await registry.call_many(clean)
    return {
        "ok": all(r.get("ok") for r in results),
        "total": len(results),
        "succeeded": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "results": results,
    }


registry.register(Tool(
    name="run_tools_parallel",
    description=(
        "并行调用多个互不依赖的工具：传入一个调用数组 [{tool, args}, ...]，"
        "内部用 asyncio.gather 同时执行并汇总结果。"
        "适合批量查询（例如同时做一次 neo4j 多跳查询 + 一次 qdrant 向量检索）。"
    ),
    params=[
        ToolParam("calls", "array",
                  "调用数组，每项形如 {\"tool\": 工具名, \"args\": {参数}}，最多 20 项",
                  required=True, items="object"),
        ToolParam("fail_fast", "boolean", "是否任一项失败即整体失败（默认 false，失败项照常返回错误）",
                  default=False),
    ],
    handler=_run,
    tags=["并行", "编排"],
    timeout=300,
))


_ = Any
