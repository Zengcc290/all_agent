"""ReAct 问答全链路演示 v2：强制真实工具调用，完整记录 LLM 输入/输出与工具信封/结果。

用法:
    .venv\\Scripts\\python.exe react_demo.py
"""

from __future__ import annotations

import json
import sys

from agents import ReActAgent
from agents.llm import LLM
from core import ExecutionContext

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STATE = {"llm_rounds": 0, "executed_tools": 0}


def dump(obj) -> str:
    try:
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), ensure_ascii=False, default=str)
    except Exception:
        pass
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return repr(obj)


def banner(title: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}\n{title}\n{char * width}")


def extract_message(response):
    try:
        if isinstance(response, dict):
            return response["choices"][0]["message"]
        return response.choices[0].message
    except Exception:
        return None


def main() -> None:
    agent = ReActAgent("react-demo", llm=LLM())

    # ---------- 0. 工具发现与访问视图 ----------
    banner("[0] 工具自动发现报告")
    for rec in agent.tool_discovery_report.records:
        print(" ", json.dumps(rec.as_dict(), ensure_ascii=False))

    update_log_key = agent.tools.confirmation_key("system.update_log")
    context = ExecutionContext(
        confirmed_side_effects=frozenset({update_log_key}),
    )
    print("\n  update_log 确认 key =", update_log_key)
    print("  可见工具（权限检查已禁用）：")
    for name, (tool, generation) in agent.tools.snapshot().items():
        print(
            f"    - {name:<22} 权限元数据={list(tool.spec.permissions) or '无'} "
            f"可见=True generation={generation}"
        )

    # ---------- 1. 强约束系统提示（防编造） ----------
    agent.set_system_prompt(
        "你是全链路演示代理。严格遵守以下规则：\n"
        "1. 你没有任何内置联网能力，外部信息只能来自系统给出的 Observation。\n"
        "2. 绝对禁止自己编造 Observation、搜索结果或 JSON 数据块；"
        "Observation 只能等待系统回传。\n"
        "3. 在收到至少一个 Observation 之前，禁止输出 'Final Answer'。\n"
        "4. 每轮只输出一个 Action，然后停下等待 Observation。按顺序完成：\n"
        "   a) 用 system.tool_catalog 检索工具："
        '{"action": "search", "intent": "web search", "tool_name": null, '
        '"version": null, "limit": 5}\n'
        "   b) 用 web.search 真实搜索（query 必须以 ' 校验码ZK7QF' 结尾）。\n"
        "   c) 用 system.update_log 记录本次演示，Action Input 按此模板补全：\n"
        '      {"executor": "deepseek-v4-pro ReAct demo", "update_type": "demo", '
        '"title": "ReAct 全链路演示运行", "task_background": "用户要求完整复现 '
        'ReAct 问答链路", "update_details": "依次真实执行了 tool_catalog 检索、'
        'web.search 搜索、update_log 写入", "added_features": "none", "files": '
        '[{"path": "react_demo.py", "action": "modified", "description": "演示'
        '脚本新增工具调用日志"}], "behavior_impact": "none", "validation": "真实'
        '调用了 web.search 和 tool_catalog", "risks": "none", "follow_up": "none"}\n'
        "   三个 Observation 全部收到后，才输出 Final Answer：用中文总结搜索到的"
        " function calling 与 ReAct 的区别（必须引用至少 2 条搜索结果的标题和 URL），"
        "并附上 update_log 返回的 update_id。如果搜索结果为空，如实说明。"
    )

    # ---------- 2. 拦截 LLM.complete ----------
    original_complete = agent.llm.complete

    def logging_complete(messages, **kwargs):
        STATE["llm_rounds"] += 1
        n = STATE["llm_rounds"]
        banner(f"[LLM 第 {n} 轮] >>> 发送给模型的完整输入", "=")
        for message in messages:
            print(f"  ---- role: {message.get('role')} ----")
            content = message.get("content")
            if isinstance(content, str) and len(content) > 4000:
                content = content[:4000] + f"\n  ...(截断, 原文 {len(message['content'])} 字符)"
            print(content)
        print("  ---- 调用参数 ----")
        print(" ", dump(kwargs))

        response = original_complete(messages, **kwargs)

        banner(f"[LLM 第 {n} 轮] <<< 模型原始输出", "=")
        message = extract_message(response)
        if message is not None:
            get = message.get if isinstance(message, dict) else (
                lambda k, d=None: getattr(message, k, d)
            )
            text = get("content")
            print(text if text else "(空文本)")
            native_calls = get("tool_calls")
            if native_calls:
                print("  ---- 原生 tool_calls ----")
                print(" ", dump(native_calls))
        else:
            print("  (无法解析)", dump(response))
        return response

    agent.llm.complete = logging_complete

    # ---------- 3. 拦截 execute_tool_calls ----------
    original_execute = agent.execute_tool_calls

    async def logging_execute(calls, context_=None):
        banner("[工具调用信封 ToolCall]", "-")
        for call in calls:
            print(dump(call))
        batch = await original_execute(calls, context_)
        banner("[工具执行结果 ToolResult]", "-")
        for result in batch.results:
            print(dump(result))
            if result.ok:
                STATE["executed_tools"] += 1
        return batch

    agent.execute_tool_calls = logging_execute

    # ---------- 4. 运行 ----------
    query = (
        "请按系统提示的顺序，依次调用 system.tool_catalog、web.search、"
        "system.update_log 三个工具，完成后给出中文总结。"
    )
    banner("[开始 ReAct 问答]")
    print("  用户问题:", query)

    answer = agent.run(
        query,
        context=context,
        max_rounds=8,
        temperature=0.2,
        timeout=120,
    )

    banner("[最终答案 (agent.run 返回值)]")
    print(answer)
    banner("[统计]")
    print(f"  LLM 轮次: {STATE['llm_rounds']}")
    print(f"  成功执行的工具调用数: {STATE['executed_tools']}")
    if STATE["executed_tools"] == 0:
        print("  ⚠ 警告: 没有任何真实工具调用发生, 上述答案内容不可信!")


if __name__ == "__main__":
    main()
