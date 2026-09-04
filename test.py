from agents import ReActAgent
from core import ToolSpecRepository

repository = ToolSpecRepository("tools.sqlite3")

agent = ReActAgent(
    "demo",
    repository=repository,
)

answer = agent.run(
    "关于现在最新的国内外新闻，你了解多少",
)
# print(answer)
