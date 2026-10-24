"""命令行播种入口：``python -m web.seed``。

播种逻辑已经工具化——幂等标记、种子文件路径与写入规则都在
``tool/seed_knowledge.py``（工具名 ``knowledge.seed``）。本文件只保留这个
手动播种入口，方便运维在不开服务的情况下灌一份演示星图。
"""

from __future__ import annotations

from tool.seed_knowledge import SEED_FILE, SEED_MARK, seed

__all__ = ["SEED_FILE", "SEED_MARK", "seed"]


if __name__ == "__main__":
    from .support import get_manager

    manager = get_manager()
    print(seed(manager))
    manager.close()
