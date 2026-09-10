"""知识星云 · 个人知识库 Web 层。

FastAPI 应用把 ``memory`` 四层记忆系统暴露为 HTTP API，
并以静态文件方式托管星云图前端（``web/static/index.html``）。
"""

from .app import create_app

__all__ = ["create_app"]
