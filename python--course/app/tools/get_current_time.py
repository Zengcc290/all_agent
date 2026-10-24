"""工具：get_current_time —— 当一句话里检测不到时间时，为实体/关系打上系统时间标记。"""
from __future__ import annotations

import datetime as _dt

from app.core.registry import registry
from app.core.validation import Tool, ToolParam

try:  # 允许配置时区降级
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # noqa: BLE001
    _TZ = None


async def _now(format: str = "datetime") -> dict:
    now = _dt.datetime.now(_TZ) if _TZ else _dt.datetime.now()
    if format == "date":
        iso = now.strftime("%Y-%m-%d")
    elif format == "time":
        iso = now.strftime("%H:%M:%S")
    elif format == "timestamp":
        return {"ok": True, "timestamp": int(now.timestamp()), "iso": now.isoformat(timespec="seconds")}
    else:
        iso = now.isoformat(timespec="seconds")
    return {
        "ok": True,
        "datetime": iso,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "year": now.year, "month": now.month, "day": now.day,
        "format": format,
        "timezone": "Asia/Shanghai",
    }


registry.register(Tool(
    name="get_current_time",
    description=(
        "获取系统当前时间。用于「一句话里检测不到任何时间表达」的场景："
        "LLM 检测一句话里是否有时间（可具体到年/月/日/时），有则原样抽取，"
        "没有时间时必须调用本工具获取系统时间，然后为每个实体、每条关系的成立都加上这个时间标记。"
    ),
    params=[
        ToolParam("format", "string",
                  "返回格式：datetime=完整时间戳（默认）、date=仅日期、time=仅时间、timestamp=秒级时间戳",
                  default="datetime", enum=["datetime", "date", "time", "timestamp"]),
    ],
    handler=_now,
    tags=["时间", "工具"],
    timeout=10,
))
