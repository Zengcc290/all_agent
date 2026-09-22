"""应用入口：python run.py   （等价 uvicorn app.main:app --reload）"""
from __future__ import annotations

import uvicorn

from app import config
from app.api.routes import app  # noqa: F401  （导入即创建 FastAPI app）

if __name__ == "__main__":
    print("=" * 62)
    print("  知识图谱入库系统 · 精简版")
    print(f"  后端: http://{config.app.host}:{config.app.port}")
    print("  文档: /docs      前端: 见 frontend/ 目录（npm run dev）")
    print("=" * 62)
    uvicorn.run("app.main:app", host=config.app.host, port=config.app.port, reload=False)
