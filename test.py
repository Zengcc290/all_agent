from agents import ReActAgent
from core import ToolSpecRepository

repository = ToolSpecRepository("tools.sqlite3")

agent = ReActAgent(
    "demo",
    repository=repository,
)

answer = agent.run(
    "关于最新的openai公司的消息，你了解多少，特别是关于最新的大模型的消息。我要的是最新的，最近两天的",
)
# print(answer)
