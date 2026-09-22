"""开发期自检脚本：python _selftest.py"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW = """```json
{"entities": [{"name": "糖尿病", "type": "疾病", "key": "糖尿病"},
              {"name": "张三", "type": "人物"},
              {"name": "胰岛素", "type": "药物"}],
 "relations": [{"source": "张三", "target": "糖尿病", "predicate": "患有", "directed": true, "time": ""},
               {"source": "胰岛素", "target": "糖尿病", "predicate": "治疗", "directed": true, "time": "2024年5月"}]}
```"""

BAD_JSON = """{"entities": [{"name": "甲", "type": "人物"}, {"name": "乙"""


async def main() -> None:
    from app.core import parser
    from app.core.registry import registry
    from app.db.sqlite_store import store

    info = await registry.discover()
    print(f"[discover] 扫描 {len(info['scanned'])} 个模块，登记工具: {info['registered']}")
    print()

    p, w = parser.parse_llm_output(RAW)
    print(f"[parse] 实体 {len(p['entities'])} 个 -> {[e['name'] for e in p['entities']]}")
    print(f"[parse] 关系 {len(p['relations'])} 条 -> {[r['predicate'] for r in p['relations']]}")
    assert parser.norm_key(" 糖尿病 ") == parser.norm_key("糖尿病")
    assert len(p["relations"]) == 2

    p2, _ = parser.parse_llm_output(BAD_JSON)
    print(f"[salvage] 截断 JSON 抢救出实体: {[e['name'] for e in p2['entities']]}")

    r = await registry.call("parse_llm_output", {"raw": RAW})
    filled = [e["time"] for e in r["result"]["entities"]]
    print(f"[tool] parse_llm_output ok={r['ok']} 时间兜底={filled}")
    assert all(filled), "缺时间应被 get_current_time 兜底"

    r = await registry.call("get_current_time", {})
    print(f"[tool] get_current_time -> {r['result']['datetime']}")

    r = await registry.call("list_pending_chunks", {"limit": 5, "preview_chars": 10})
    print(f"[tool] list_pending_chunks -> 待入库 {r['result']['total_pending']} 条")

    # 参数校验
    for bad in ({"limit": "abc"}, {"unknown": 1}, {"status": "nope"}):
        try:
            await registry.call("list_pending_chunks", bad)
            print(f"[validate] 未拦截 {bad} <-- 应该报错")
        except Exception as e:
            print(f"[validate] 正确拦截 {bad} -> {type(e).__name__}: {str(e)[:70]}")

    # 并行调用
    r = await registry.call("run_tools_parallel", {"calls": [
        {"tool": "get_current_time", "args": {"format": "date"}},
        {"tool": "list_pending_chunks", "args": {}},
        {"tool": "get_stats", "args": {}},
    ]})
    print(f"[parallel] {r['result']['succeeded']}/{r['result']['total']} 成功, "
          f"耗时 {r['elapsed_ms']}ms（串行约 {sum(x['elapsed_ms'] for x in r['result']['results'])}ms）")

    print()
    print("[sqlite]", store.stats())


if __name__ == "__main__":
    asyncio.run(main())
