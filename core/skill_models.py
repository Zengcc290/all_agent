"""Value models for the on-demand skill (instruction package) subsystem.

A skill is a read-only instruction package stored as one Markdown file,
``skills/<name>.md``. Its body is never executed and only reaches the model
after an explicit ``system.skill_catalog`` ``view`` call, mirroring the
catalog-first lazy loading used for tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SKILL_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

MAX_DESCRIPTION_CHARS = 2000
MAX_VERSION_CHARS = 32
MAX_TRIGGER_CHARS = 200
CONTENT_HASH_LENGTH = 64


@dataclass(frozen=True)
class SkillSpec:
    """Static metadata for one discovered skill file.

    ``content_hash`` covers the whole skill file (frontmatter included), so
    any content change is observable for prompt-cache invalidation. An empty
    ``content_hash`` is reserved for synthetic specs that were not loaded
    from a file; discovery always supplies a real SHA-256 hex digest.
    """

    name: str
    description: str
    version: str = "1.0.0"
    triggers: tuple[str, ...] = ()
    file_path: str | None = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not SKILL_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError(
                "skill names must be kebab-case ASCII (lowercase letters, "
                "digits, hyphens; max 64 chars), for example 'paper-review'"
            )
        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or len(self.description) > MAX_DESCRIPTION_CHARS
        ):
            raise ValueError(
                "skill descriptions must be non-empty and at most "
                f"{MAX_DESCRIPTION_CHARS} characters"
            )
        if (
            not isinstance(self.version, str)
            or not self.version.strip()
            or len(self.version) > MAX_VERSION_CHARS
        ):
            raise ValueError(
                "skill versions must be between 1 and "
                f"{MAX_VERSION_CHARS} characters"
            )
        if not isinstance(self.triggers, tuple) or not all(
            isinstance(item, str) and item.strip() and len(item) <= MAX_TRIGGER_CHARS
            for item in self.triggers
        ):
            raise TypeError(
                "triggers must be a tuple of non-empty strings of at most "
                f"{MAX_TRIGGER_CHARS} characters"
            )
        if self.file_path is not None and (
            not isinstance(self.file_path, str) or not self.file_path.strip()
        ):
            raise ValueError("file_path must be a non-empty string or None")
        if not isinstance(self.content_hash, str) or (
            self.content_hash
            and (
                len(self.content_hash) != CONTENT_HASH_LENGTH
                or not re.fullmatch(r"[0-9a-f]+", self.content_hash)
            )
        ):
            raise ValueError(
                "content_hash must be empty or a lowercase SHA-256 hex digest"
            )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly view for catalog output."""

        return {
            "skill_name": self.name,
            "description": self.description,
            "version": self.version,
            "triggers": list(self.triggers),
            "content_hash": self.content_hash,
            "file_path": self.file_path,
        }
