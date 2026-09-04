from agents import ReActAgent
from core import ExecutionContext, ToolSpecRepository

repository = ToolSpecRepository("tools.sqlite3")

agent = ReActAgent(
    "demo",
    repository=repository,
)

context = ExecutionContext(permissions=frozenset({"network.read"}))
answer = agent.run(
    "现在是什么时候?今天的12星座运势怎么样",
    context=context,
    profile_name="deepseek",
)
# print(answer)
