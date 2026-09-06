"""Flat-file discovery for ``skills/<name>.md`` instruction packages.

Scanning never imports or executes any Python code: a skill is pure
documentation stored as one Markdown file directly inside the skills root.
The file stem is the skill name; the root ``README.md`` is skipped. A
malformed skill produces an error record without hiding healthy siblings,
matching the tool discovery contract. Legacy ``<name>/SKILL.md`` directories
are reported as errors with a migration hint so old layouts cannot silently
disappear.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from constants import (
    DEFAULT_SKILLS_ROOT,
    ENABLED_FIELD,
    LEGACY_SKILL_ENTRY_FILENAME,
    MAX_DESCRIPTION_CHARS,
    MAX_TRIGGER_CHARS,
    MAX_VERSION_CHARS,
    README_FILENAME,
    SKILL_FILE_SUFFIX,
    SKILL_NAME_PATTERN,
)

from .skill_models import SkillSpec

DiscoveryStatus = Literal[
    "registered",
    "already_registered",
    "disabled",
    "ignored",
    "error",
]


@dataclass(frozen=True)
class SkillDiscoveryRecord:
    """The discovery outcome for one candidate skill file."""

    name: str
    path: str | None
    status: DiscoveryStatus
    version: str | None = None
    generation: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillDiscoveryReport:
    """Immutable report for one skills-root scan."""

    root: str
    records: tuple[SkillDiscoveryRecord, ...]

    @property
    def errors(self) -> tuple[SkillDiscoveryRecord, ...]:
        return tuple(record for record in self.records if record.status == "error")

    @property
    def registered(self) -> tuple[SkillDiscoveryRecord, ...]:
        return tuple(
            record for record in self.records if record.status == "registered"
        )

    @property
    def ok(self) -> bool:
        return not self.errors

    def for_skill(self, name: str) -> SkillDiscoveryRecord | None:
        if not isinstance(name, str) or not name:
            raise ValueError("skill name must be a non-empty string")
        return next(
            (record for record in self.records if record.name == name),
            None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "ok": self.ok,
            "records": [record.as_dict() for record in self.records],
        }


class SkillDiscoveryError(RuntimeError):
    """Raised after a strict scan if one or more skills could not be loaded."""

    def __init__(self, report: SkillDiscoveryReport) -> None:
        self.report = report
        details = "; ".join(f"{r.name}: {r.error}" for r in report.errors)
        super().__init__(f"skill discovery failed: {details}")


def discover_skills(
    registry,
    *,
    root: str | Path = DEFAULT_SKILLS_ROOT,
    replace: bool = False,
    strict: bool = False,
) -> SkillDiscoveryReport:
    """Scan ``root`` and register one spec per valid ``<name>.md`` file.

    The file stem becomes the skill name and the whole file (frontmatter
    plus body) is the viewable content. Frontmatter starts with the first
    ``---`` line and ends at the next ``---`` line. Unknown frontmatter keys
    fail validation, mirroring the ``extra="forbid"`` tool contract. An
    optional ``enabled: false`` key skips the skill without an error.
    """

    if root is None or isinstance(root, (str, Path)) is False:
        raise TypeError("root must be a directory path")
    if not isinstance(replace, bool) or not isinstance(strict, bool):
        raise TypeError("replace and strict must be booleans")

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        report = SkillDiscoveryReport(
            str(root_path),
            (
                SkillDiscoveryRecord(
                    name="skills-root",
                    path=str(root_path),
                    status="error",
                    error=f"skills root does not exist: {root_path}",
                ),
            ),
        )
        if strict:
            raise SkillDiscoveryError(report)
        return report

    records: list[SkillDiscoveryRecord] = []
    for entry in sorted(root_path.iterdir(), key=lambda item: item.name):
        if entry.is_dir():
            if (entry / LEGACY_SKILL_ENTRY_FILENAME).is_file():
                records.append(
                    SkillDiscoveryRecord(
                        name=entry.name,
                        path=str(entry),
                        status="error",
                        error=(
                            "legacy directory layout is no longer supported; "
                            f"move {entry.name}/{LEGACY_SKILL_ENTRY_FILENAME} "
                            f"to {entry.name}{SKILL_FILE_SUFFIX}"
                        ),
                    )
                )
            else:
                records.append(
                    SkillDiscoveryRecord(
                        name=entry.name, path=str(entry), status="ignored"
                    )
                )
            continue
        if entry.suffix.lower() != SKILL_FILE_SUFFIX:
            records.append(
                SkillDiscoveryRecord(
                    name=entry.name, path=str(entry), status="ignored"
                )
            )
            continue
        if entry.name == README_FILENAME:
            # The root authoring guide is not a skill; skip silently.
            continue
        name = entry.stem
        try:
            spec = _load_spec(name, entry)
            if not spec:
                records.append(
                    SkillDiscoveryRecord(
                        name=name,
                        path=str(entry),
                        status="disabled",
                    )
                )
                continue
            existing = registry.maybe_resolve(spec.name)
            generation = registry.register(spec, replace=replace)
            status = (
                "already_registered"
                if existing is not None and existing[0] == spec
                else "registered"
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                SkillDiscoveryRecord(
                    name=name,
                    path=str(entry),
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        records.append(
            SkillDiscoveryRecord(
                name=spec.name,
                path=str(entry),
                status=status,
                version=spec.version,
                generation=generation,
            )
        )

    report = SkillDiscoveryReport(str(root_path), tuple(records))
    if strict and report.errors:
        raise SkillDiscoveryError(report)
    return report


def _load_spec(name: str, skill_file: Path) -> SkillSpec | None:
    """Parse one ``<name>.md``; return ``None`` when the skill is disabled."""

    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"file stem '{name}' is not a valid skill name; skill names must "
            "be kebab-case ASCII (lowercase letters, digits, hyphens; max 64 "
            "chars)"
        )
    raw = skill_file.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    fields = _parse_frontmatter(raw, skill_file)
    if fields is None:
        return None  # disabled
    description = fields.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"{skill_file.name} frontmatter must define a non-empty "
            "'description'"
        )
    version = fields.get("version", "1.0.0")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("'version' must be a non-empty string when present")
    triggers_raw = fields.get("triggers", ())
    if isinstance(triggers_raw, str):
        triggers = tuple(
            item.strip() for item in triggers_raw.split(",") if item.strip()
        )
    elif isinstance(triggers_raw, (list, tuple)):
        triggers = tuple(triggers_raw)
    else:
        raise TypeError(
            "'triggers' must be a comma-separated string or a list of strings"
        )
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(
            f"'description' must be at most {MAX_DESCRIPTION_CHARS} characters"
        )
    if len(version) > MAX_VERSION_CHARS:
        raise ValueError(f"'version' must be at most {MAX_VERSION_CHARS} characters")
    for trigger in triggers:
        if not isinstance(trigger, str) or not trigger.strip():
            raise TypeError("every trigger must be a non-empty string")
        if len(trigger) > MAX_TRIGGER_CHARS:
            raise ValueError(
                f"each trigger must be at most {MAX_TRIGGER_CHARS} characters"
            )
    return SkillSpec(
        name=name,
        description=description.strip(),
        version=version.strip(),
        triggers=triggers,
        file_path=str(skill_file),
        content_hash=content_hash,
    )


def _parse_frontmatter(raw: str, skill_file: Path) -> dict[str, Any] | None:
    """Parse the simple ``key: value`` frontmatter block.

    Returns ``None`` when the skill is disabled. Unknown keys raise
    ``ValueError`` so typoed metadata cannot silently disappear.
    """

    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"{skill_file.name} must start with a '---' frontmatter block"
        )
    fields: dict[str, Any] = {}
    end_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise ValueError(
            f"{skill_file.name} frontmatter is not closed with a '---' line"
        )
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(
                f"invalid frontmatter line in {skill_file.name}: {stripped!r}"
            )
        key, _, value = stripped.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == ENABLED_FIELD:
            if value.lower() not in {"true", "false"}:
                raise ValueError("'enabled' must be 'true' or 'false'")
            fields[key] = value.lower() == "true"
        elif key == "description":
            fields[key] = value
        elif key == "version":
            fields[key] = value
        elif key == "triggers":
            fields[key] = value
        else:
            raise ValueError(
                f"unknown frontmatter key {key!r} in {skill_file.name}; allowed "
                "keys: description, version, triggers, enabled"
            )
    if fields.get(ENABLED_FIELD) is False:
        return None
    return fields


def read_skill_content(spec: SkillSpec) -> str:
    """Read the full current skill file content for one skill spec."""

    if spec.file_path is None:
        raise ValueError(f"skill '{spec.name}' has no source file")
    skill_file = Path(spec.file_path)
    if not skill_file.is_file():
        raise FileNotFoundError(
            f"skill '{spec.name}' file does not exist: {skill_file}"
        )
    return skill_file.read_text(encoding="utf-8")
