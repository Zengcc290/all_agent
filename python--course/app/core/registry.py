"""工具注册表：统一登记、参数校验、自动发现、动态生成提示词。"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .validation import Tool, ToolParam, ToolValidationError, _to_bool, _to_int

DEFAULT_TIMEOUT = 120


class ToolError(RuntimeError):
    """工具执行期错误（与参数校验错误区分开）。"""


class ToolNotFoundError(ToolError):
    pass


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(i) for i in x]
    if hasattr(x, "model_dump"):
        return _jsonable(x.model_dump())
    if hasattr(x, "__dict__"):
        return _jsonable(vars(x))
    return str(x)


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)
    _modules: set[str] = field(default_factory=set)
    _lock: asyncio.Lock | None = None

    # ---------------- 注册 ----------------
    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def register_many(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    # ---------------- 发现（自动导入 tools 包下所有模块） ----------------
    async def discover(self, package: str = "app.tools") -> dict:
        """扫描 package 下所有模块并导入。
        任意模块在 import 时调用 registry.register(...) 即完成登记，
        因此往 tools 目录丢一个新文件，下次调用就会自动被发现，无需改动提示词。
        """
        async with self._lock_or_new():
            pkg = importlib.import_module(package)
            newly: list[str] = []
            for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
                if info.name.startswith("_"):
                    continue
                fq = f"{package}.{info.name}"
                before = set(self._tools)
                if fq not in self._modules:
                    try:
                        importlib.import_module(fq)
                    except Exception as e:  # 单个工具损坏不应拖垮整个发现过程
                        print(f"[discover] 跳过 {fq}: {type(e).__name__}: {e}")
                        continue
                    self._modules.add(fq)
                newly.extend(sorted(set(self._tools) - set(before)))
            return {
                "scanned": sorted(self._modules),
                "registered": self.names(),
                "new": sorted(set(newly)),
            }

    def _lock_or_new(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def reload(self) -> None:
        """清空登记，下次 discover 会重新导入全部模块。"""
        self._tools.clear()
        self._modules.clear()

    # ---------------- 查询 ----------------
    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        t = self._tools.get(name)
        if t is None:
            raise ToolNotFoundError(f"工具 {name!r} 不存在，已注册: {self.names()}")
        return t

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in self.names()]

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.all()]

    # ---------------- 调用 ----------------
    async def call(self, name: str, args: dict | None = None, timeout: float | None = None) -> dict:
        t0 = time.perf_counter()
        if not self._tools:
            await self.discover()
        tool = self.get(name)
        kwargs = tool.validate(args)          # ToolValidationError -> 上层转 400
        limit = timeout or tool.timeout or DEFAULT_TIMEOUT
        try:
            if inspect.iscoroutinefunction(tool.handler):
                result = await asyncio.wait_for(tool.handler(**kwargs), timeout=limit)
            else:
                result = await asyncio.wait_for(asyncio.to_thread(lambda: tool.handler(**kwargs)), timeout=limit)
        except ToolValidationError:
            raise
        except ToolError:
            raise
        except asyncio.TimeoutError as e:
            raise ToolError(f"工具 {name} 执行超时（>{limit:.0f}s）") from e
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            raise ToolError(f"工具 {name} 执行失败: {type(e).__name__}: {e}") from e
        return {
            "tool": name,
            "ok": True,
            "args": kwargs,
            "result": _jsonable(result),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    async def call_many(self, calls: list[dict]) -> list[dict]:
        """并行调用多个工具（asyncio.gather，互不阻塞）。"""

        async def one(c: dict) -> dict:
            try:
                return await self.call(c.get("tool") or c.get("name") or "", c.get("args") or {}, c.get("timeout"))
            except Exception as e:  # noqa: BLE001
                return {
                    "tool": c.get("tool") or c.get("name"),
                    "ok": False,
                    "error": str(e),
                    "args": c.get("args") or {},
                    "elapsed_ms": 0,
                }

        return list(await asyncio.gather(*(one(c) for c in calls)))

    # ---------------- 提示词 ----------------
    def describe(self) -> str:
        """动态生成「可用工具清单」——注册即出现，绝不写死。"""
        lines = [
            "# 可用工具清单",
            "",
            '调用时必须输出一个 JSON 对象：{"tool": "<工具名>", "args": {...}}',
            "参数标注 [必填] 的不可省略；未标注的可省略（省略即用默认值）。",
            "可以并行调用多个互不依赖的工具。",
            "",
        ]
        for t in self.all():
            lines.append(f"## {t.name}")
            lines.append(t.description)
            if not t.params:
                lines.append("    （无参数）")
            for p in t.params:
                extra = ""
                if p.enum:
                    extra += f"，枚举取值 {p.enum}"
                if p.default is not None:
                    extra += f"，默认 {p.default!r}"
                if p.min_value is not None or p.max_value is not None:
                    extra += f"，范围 [{p.min_value}, {p.max_value}]"
                lines.append(
                    f"    - {p.name} ({p.type}, [{p.label}]){extra}: {p.description}"
                )
            lines.append("")
        return "\n".join(lines).strip()


# ---- 全局单例：tools 目录里的模块都 import 这个对象来登记 ----
registry = ToolRegistry()


# ---- 给工具作者的小工具：把简单函数快速包成 Tool ----
def make_tool(name, description, params, handler, tags=None, timeout=DEFAULT_TIMEOUT) -> Tool:
    return Tool(name=name, description=description, params=params or [], handler=handler,
                tags=tags or [], timeout=timeout)


def p(name, type="string", description="", required=False, default=None, enum=None,
      items="string", min_value=None, max_value=None, max_length=None) -> ToolParam:
    """ToolParam 的简写构造器，方便 tools/*.py 里一行定义参数。"""
    return ToolParam(name=name, type=type, description=description, required=required,
                     default=default, enum=list(enum) if enum else None, items=items,
                     min_value=min_value, max_value=max_value, max_length=max_length)


_ = (_to_bool, _to_int)  # 供 tools 模块复用的转换器
