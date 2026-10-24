"""轻量级工具参数定义与校验器（不依赖 pydantic，Tool/ToolParam 即契约）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

_TRUE = {"true", "1", "yes", "y", "on"}
_FALSE = {"false", "0", "no", "n", "off"}
_VALID_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


class ToolValidationError(ValueError):
    """参数校验失败——message 会直接回给前端 / LLM。"""


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    raise ToolValidationError(f"无法将 {v!r} 解析为 boolean")


def _to_int(v: Any) -> int:
    if isinstance(v, bool):
        raise ToolValidationError("boolean 不能转为 integer")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if float(v).is_integer():
            return int(v)
        raise ToolValidationError(f"{v!r} 不是整数")
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError as e:
            raise ToolValidationError(f"{v!r} 不是合法整数") from e
    raise ToolValidationError(f"无法将 {type(v).__name__} 转为 integer")


def _to_float(v: Any) -> float:
    if isinstance(v, bool):
        raise ToolValidationError("boolean 不能转为 number")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError as e:
            raise ToolValidationError(f"{v!r} 不是合法数字") from e
    raise ToolValidationError(f"无法将 {type(v).__name__} 转为 number")


def _to_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [v]  # 单个值容错
    raise ToolValidationError(f"无法将 {type(v).__name__} 转为 array")


def _to_dict(v: Any) -> dict:
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError as e:
            raise ToolValidationError(f"无法将 {v!r} 转为 object（不是合法 JSON）") from e
        if isinstance(parsed, dict):
            return parsed
    raise ToolValidationError(f"无法将 {type(v).__name__} 转为 object")


@dataclass
class ToolParam:
    """一个参数的完整描述——提示词与前端动态表单都由它生成。"""

    name: str
    type: str = "string"                        # string|integer|number|boolean|array|object
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list | None = None
    items: str = "string"                       # array 元素类型
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        if not self.name.replace("_", "").isalnum():
            raise ValueError(f"非法参数名: {self.name!r}")
        if self.type not in _VALID_TYPES:
            raise ValueError(f"参数 {self.name} 类型非法: {self.type!r}")
        if self.required and self.default is not None:
            raise ValueError(f"参数 {self.name} 是必填项，不能同时给默认值")
        if self.type == "array" and self.items not in _VALID_TYPES:
            raise ValueError(f"参数 {self.name} 的 items 类型非法: {self.items!r}")

    @property
    def label(self) -> str:
        return "必填" if self.required else "可选"

    def schema(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "optional": not self.required,
            "default": self.default,
            "enum": list(self.enum) if self.enum else None,
            "items": self.items if self.type == "array" else None,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "max_length": self.max_length,
        }

    def _coerce(self, raw: Any) -> Any:
        t = self.type
        try:
            if t == "string":
                v = str(raw)
            elif t == "integer":
                v = _to_int(raw)
            elif t == "number":
                v = _to_float(raw)
            elif t == "boolean":
                v = _to_bool(raw)
            elif t == "array":
                v = [self._coerce_item(x) for x in _to_list(raw)]
            else:
                v = _to_dict(raw)
        except ToolValidationError as e:
            raise ToolValidationError(f"参数 {self.name!r} 错误: {e}") from e

        if isinstance(v, str):
            if self.min_length is not None and len(v) < self.min_length:
                raise ToolValidationError(f"参数 {self.name!r} 长度至少 {self.min_length}")
            if self.max_length is not None and len(v) > self.max_length:
                raise ToolValidationError(f"参数 {self.name!r} 长度至多 {self.max_length}")
            if not v.strip() and not self.required and self.default is None:
                raise ToolValidationError(f"参数 {self.name!r} 不能为空字符串")
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            if self.min_value is not None and v < self.min_value:
                raise ToolValidationError(f"参数 {self.name!r} 不能小于 {self.min_value}")
            if self.max_value is not None and v > self.max_value:
                raise ToolValidationError(f"参数 {self.name!r} 不能大于 {self.max_value}")
        return v

    def _coerce_item(self, x: Any) -> Any:
        try:
            if self.items == "integer":
                return _to_int(x)
            if self.items == "number":
                return _to_float(x)
            if self.items == "boolean":
                return _to_bool(x)
            if self.items == "object":
                return _to_dict(x)
            return str(x)
        except ToolValidationError as e:
            raise ToolValidationError(f"参数 {self.name!r} 数组元素错误: {e}") from e


@dataclass
class Tool:
    """一个工具的完整定义：元信息 + 参数契约 + 异步 handler。"""

    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    handler: Callable[..., Awaitable[Any]] | None = None
    tags: list[str] = field(default_factory=list)
    timeout: float = 120

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"非法工具名: {self.name!r}")
        if self.handler is None or not callable(self.handler):
            raise ValueError(f"工具 {self.name} 需要可调用的 handler")
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"工具 {self.name} 存在重复参数名: {names}")

    def param(self, name: str) -> ToolParam | None:
        return next((p for p in self.params if p.name == name), None)

    def required_params(self) -> list[str]:
        return [p.name for p in self.params if p.required]

    def optional_params(self) -> list[str]:
        return [p.name for p in self.params if not p.required]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "params": [p.schema() for p in self.params],
        }

    def validate(self, args: dict | None) -> dict:
        """校验 + 类型转换 + 补默认值 -> 可直接 **kwargs 的干净字典。"""
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ToolValidationError(f"{self.name} 的参数必须是 JSON object，收到 {type(args).__name__}")
        if not args and self.required_params():
            raise ToolValidationError(
                f"{self.name} 缺少必填参数 {self.required_params()}"
            )
        unknown = sorted(set(args) - set(p.name for p in self.params))
        if unknown:
            raise ToolValidationError(
                f"{self.name} 收到未知参数 {unknown}，可用参数: {[p.name for p in self.params]}"
            )

        clean: dict[str, Any] = {}
        for p in self.params:
            if p.name in args:
                raw = args[p.name]
                if raw is None or (isinstance(raw, str) and raw == "" and not p.required and p.default is not None):
                    if p.required:
                        raise ToolValidationError(f"{self.name} 缺少必填参数 {p.name!r}（{p.description}）")
                else:
                    clean[p.name] = p._coerce(raw)
            elif p.required:
                raise ToolValidationError(f"{self.name} 缺少必填参数 {p.name!r}（{p.description}）")
            elif p.default is not None:
                clean[p.name] = p._coerce(p.default)

        for p in self.params:
            if p.enum and p.name in clean and clean[p.name] not in p.enum:
                raise ToolValidationError(
                    f"{self.name} 参数 {p.name}={clean[p.name]!r} 不在枚举 {p.enum} 内"
                )
        return clean
