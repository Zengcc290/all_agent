"""知识星云 · 个人知识库 Web 层。

FastAPI 应用把 ``memory`` 四层记忆系统暴露为 HTTP API，
并以静态文件方式托管前端单页（``web/static``，由 ``web/frontend`` 的
Vite 工程构建产物生成）。
"""

from .app import create_app

__all__ = ["create_app"]