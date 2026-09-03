from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from .models import ToolSpec
from .registry import BaseTool, ToolRegistry
from .repository import ToolSpecRepository

DiscoveryStatus = Literal[
    "registered",
    "already_registered",
    "disabled",
    "ignored",
    "error",
]


@dataclass(frozen=True)
class ToolDiscoveryRecord:
    """The discovery and registration outcome for one Python module."""

    module: str
    path: str | None
    enabled: bool | None
    status: DiscoveryStatus
    tool_name: str | None = None
    version: str | None = None
    generation: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolDiscoveryReport:
    """Immutable report for one package scan."""

    package: str
    records: tuple[ToolDiscoveryRecord, ...]

    @property
    def errors(self) -> tuple[ToolDiscoveryRecord, ...]:
        return tuple(record for record in self.records if record.status == "error")

    @property
    def registered(self) -> tuple[ToolDiscoveryRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.status in {"registered", "already_registered"}
        )

    @property
    def ok(self) -> bool:
        return not self.errors

    def for_tool(self, name: str) -> ToolDiscoveryRecord | None:
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        return next(
            (record for record in self.records if record.tool_name == name),
            None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "ok": self.ok,
            "records": [record.as_dict() for record in self.records],
        }


class ToolDiscoveryError(RuntimeError):
    """Raised after a strict scan if one or more modules could not be loaded."""

    def __init__(self, report: ToolDiscoveryReport) -> None:
        self.report = report
        details = "; ".join(
            f"{record.module}: {record.error}" for record in report.errors
        )
        super().__init__(f"tool discovery failed: {details}")


def discover_tools(
    registry: ToolRegistry,
    *,
    package: str | ModuleType = "tool",
    repository: ToolSpecRepository | None = None,
    replace: bool = False,
    strict: bool = False,
    reload_modules: bool = False,
) -> ToolDiscoveryReport:
    """Discover direct child modules that implement the single-file tool protocol.

    Enabled modules must expose ``TOOL_ENABLED: bool`` and a zero-argument
    ``create_tool()`` factory returning ``BaseTool``. Importing a module executes
    its top-level Python code, so only trusted packages should be scanned.
    """
    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be a ToolRegistry")
    if repository is not None and not isinstance(repository, ToolSpecRepository):
        raise TypeError("repository must be a ToolSpecRepository or None")
    for value, name in (
        (replace, "replace"),
        (strict, "strict"),
        (reload_modules, "reload_modules"),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")

    package_name, package_module, package_error = _load_package(package)
    if package_error is not None:
        report = ToolDiscoveryReport(package_name, (package_error,))
        if strict:
            raise ToolDiscoveryError(report)
        return report

    records: list[ToolDiscoveryRecord] = []
    importlib.invalidate_caches()
    modules = sorted(
        pkgutil.iter_modules(
            package_module.__path__, prefix=f"{package_module.__name__}."
        ),
        key=lambda item: item.name,
    )
    for module_info in modules:
        short_name = module_info.name.rsplit(".", 1)[-1]
        candidate_path = _candidate_path(module_info)
        if short_name.startswith("_") or short_name == "base" or module_info.ispkg:
            records.append(
                ToolDiscoveryRecord(
                    module=module_info.name,
                    path=candidate_path,
                    enabled=None,
                    status="ignored",
                )
            )
            continue

        try:
            module = importlib.import_module(module_info.name)
            if reload_modules and module_info.name in sys.modules:
                module = importlib.reload(module)
        # A plugin is an isolation boundary: report its regular exceptions and
        # continue scanning the remaining independent modules.
        except Exception as exc:  # noqa: BLE001
            records.append(
                _error_record(
                    module_info.name,
                    candidate_path,
                    None,
                    exc,
                    "module import failed",
                )
            )
            continue

        module_path = getattr(module, "__file__", None) or candidate_path
        enabled = getattr(module, "TOOL_ENABLED", None)
        if not isinstance(enabled, bool):
            records.append(
                ToolDiscoveryRecord(
                    module=module.__name__,
                    path=module_path,
                    enabled=None,
                    status="error",
                    error="TOOL_ENABLED must be defined as a boolean",
                )
            )
            continue
        if not enabled:
            records.append(
                ToolDiscoveryRecord(
                    module=module.__name__,
                    path=module_path,
                    enabled=False,
                    status="disabled",
                )
            )
            continue

        factory = getattr(module, "create_tool", None)
        if not callable(factory):
            records.append(
                ToolDiscoveryRecord(
                    module=module.__name__,
                    path=module_path,
                    enabled=True,
                    status="error",
                    error="enabled module must define callable create_tool()",
                )
            )
            continue
        try:
            inspect.signature(factory).bind()
        except (TypeError, ValueError) as exc:
            records.append(
                _error_record(
                    module.__name__,
                    module_path,
                    True,
                    exc,
                    "create_tool must be callable without arguments",
                )
            )
            continue

        tool: object | None = None
        try:
            tool = factory()
            record = _register_discovered_tool(
                registry,
                module,
                module_path,
                tool,
                repository=repository,
                replace=replace,
            )
        # Factories are third-party extension points and can raise any regular
        # exception; one broken tool must not hide healthy sibling tools.
        except Exception as exc:  # noqa: BLE001
            spec = getattr(tool, "spec", None)
            tool_name = spec.name if isinstance(spec, ToolSpec) else None
            version = spec.version if isinstance(spec, ToolSpec) else None
            active = registry.maybe_resolve(tool_name) if tool_name else None
            records.append(
                _error_record(
                    module.__name__,
                    module_path,
                    True,
                    exc,
                    "create or register failed",
                    tool_name=tool_name,
                    version=version,
                    generation=active[1] if active is not None else None,
                )
            )
        else:
            records.append(record)

    report = ToolDiscoveryReport(package_name, tuple(records))
    if strict and report.errors:
        raise ToolDiscoveryError(report)
    return report


def _load_package(
    package: str | ModuleType,
) -> tuple[str, ModuleType | None, ToolDiscoveryRecord | None]:
    if isinstance(package, ModuleType):
        package_name = package.__name__
        package_module = package
    elif isinstance(package, str) and package.strip():
        package_name = package
        try:
            package_module = importlib.import_module(package)
        # Package import is also extension code and belongs in the report.
        except Exception as exc:  # noqa: BLE001
            return (
                package_name,
                None,
                _error_record(
                    package_name,
                    None,
                    None,
                    exc,
                    "package import failed",
                ),
            )
    else:
        raise TypeError("package must be a non-empty module name or ModuleType")
    if not hasattr(package_module, "__path__"):
        return (
            package_name,
            None,
            ToolDiscoveryRecord(
                module=package_name,
                path=getattr(package_module, "__file__", None),
                enabled=None,
                status="error",
                error="tool package must define __path__",
            ),
        )
    return package_name, package_module, None


def _register_discovered_tool(
    registry: ToolRegistry,
    module: ModuleType,
    module_path: str | None,
    tool: object,
    *,
    repository: ToolSpecRepository | None,
    replace: bool,
) -> ToolDiscoveryRecord:
    if not isinstance(tool, BaseTool):
        raise TypeError("create_tool() must return a BaseTool instance")
    spec = getattr(tool, "spec", None)
    if not isinstance(spec, ToolSpec):
        raise TypeError("created tool must define a ToolSpec instance")

    current = registry.maybe_resolve(spec.name)
    if current is not None:
        current_tool, generation = current
        if type(current_tool) is type(tool) and current_tool.spec == spec:
            _save_spec(repository, spec, module, tool)
            return ToolDiscoveryRecord(
                module=module.__name__,
                path=module_path,
                enabled=True,
                status="already_registered",
                tool_name=spec.name,
                version=spec.version,
                generation=generation,
            )
        if not replace:
            raise ValueError(
                f"tool '{spec.name}' has a different active implementation; "
                "pass replace=True to replace it"
            )

    registry.register(tool, replace=current is not None)
    _, generation = registry.resolve(spec.name)
    _save_spec(repository, spec, module, tool)
    return ToolDiscoveryRecord(
        module=module.__name__,
        path=module_path,
        enabled=True,
        status="registered",
        tool_name=spec.name,
        version=spec.version,
        generation=generation,
    )


def _save_spec(
    repository: ToolSpecRepository | None,
    spec: ToolSpec,
    module: ModuleType,
    tool: BaseTool,
) -> None:
    if repository is None:
        return
    implementation_ref = f"{module.__name__}:{type(tool).__qualname__}"
    repository.save(
        spec,
        implementation_ref=implementation_ref,
        replace=True,
    )


def _candidate_path(module_info: pkgutil.ModuleInfo) -> str | None:
    root = getattr(module_info.module_finder, "path", None)
    if not isinstance(root, str):
        return None
    short_name = module_info.name.rsplit(".", 1)[-1]
    if module_info.ispkg:
        return str(Path(root, short_name, "__init__.py"))
    return str(Path(root, f"{short_name}.py"))


def _error_record(
    module: str,
    path: str | None,
    enabled: bool | None,
    error: Exception,
    prefix: str,
    *,
    tool_name: str | None = None,
    version: str | None = None,
    generation: int | None = None,
) -> ToolDiscoveryRecord:
    return ToolDiscoveryRecord(
        module=module,
        path=path,
        enabled=enabled,
        status="error",
        tool_name=tool_name,
        version=version,
        generation=generation,
        error=f"{prefix}: {type(error).__name__}: {error}",
    )
