import asyncio

from agents.react import ReActAgent


async def main():
    # 整个对话期间只创建一次 Agent
    agent = ReActAgent("persistent-chat")

    print("持久对话已启动")
    print("输入 /clear 清空历史，输入 /exit 退出\n")

    while True:
        try:
            query = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束")
            break

        if not query:
            continue

        if query == "/exit":
            print("对话结束")
            break

        if query == "/clear":
            agent.clear_history()
            print("历史对话已清空")
            continue

        try:
            answer = await agent.run_with_react(
                query,
                # Agent 默认复用同一 profile 的历史；/clear 可显式清空。
                use_history=True,
                # 不限制目录调用、逐条日志读取和最终总结的轮数。
                max_rounds=None,
                # 交互程序已经自动发现并注册工具，直接暴露完整契约，
                # 避免模型先猜目录意图而在中文查询上反复重试。
                defer_tool_loading=False,
                prompt_cache_retention="24h",
            )
            print(f"AI：{answer}\n")
        except Exception as exc:
            print(f"请求失败：{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
