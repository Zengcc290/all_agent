from agents import LLM, ReActAgent
from core import ExecutionContext, ToolSpecRepository

# 方式一：从 .env 读取 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
llm = LLM()

repository = ToolSpecRepository("tools.sqlite3")

agent = ReActAgent(
    "demo",
    llm=llm,
    repository=repository,
)

context = ExecutionContext(permissions=frozenset({"network.read"}))
answer = agent.run("请搜索 Python 3.12 的资料", context=context)
print(answer)
