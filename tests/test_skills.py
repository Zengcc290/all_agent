"""Tests for the on-demand skill subsystem (skills/<name>/SKILL.md)."""

import json
from pathlib import Path

import pytest

from core import (
    SkillCatalogInput,
    SkillCatalogOutput,
    SkillCatalogTool,
    SkillDiscoveryError,
    SkillRegistry,
    SkillSpec,
    discover_skills,
)
from core.skill_discovery import read_skill_content


VALID_SKILL_MD = """---
description: Draft and polish weekly reports with a fixed structure.
version: 1.2.0
triggers: weekly report, 周报
enabled: true
---

# Weekly report body

Follow the numbered structure.
"""


def write_skill(root: Path, name: str, content: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(content, encoding="utf-8")
    return directory


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    root.mkdir()
    write_skill(root, "weekly-report", VALID_SKILL_MD)
    write_skill(
        root,
        "paper-review",
        (
            "---\n"
            "description: 回应论文审稿意见时使用;涵盖审稿维度与回复信写法。\n"
            "---\n\n"
            "正文内容。\n"
        ),
    )
    return root


class TestSkillSpec:
    def test_valid_spec_round_trips_summary(self):
        spec = SkillSpec(
            name="paper-review",
            description="Helps with reviews.",
            version="1.0.0",
            triggers=("审稿",),
        )
        summary = spec.summary()
        assert summary["skill_name"] == "paper-review"
        assert summary["triggers"] == ["审稿"]
        assert summary["content_hash"] == ""

    def test_rejects_bad_names(self):
        for bad in ("Paper", "has space", " upper", "", "a" * 65, "-leading"):
            with pytest.raises(ValueError):
                SkillSpec(name=bad, description="d")

    def test_rejects_empty_description_and_bad_triggers(self):
        with pytest.raises(ValueError):
            SkillSpec(name="ok", description="   ")
        with pytest.raises(TypeError):
            SkillSpec(name="ok", description="d", triggers=("", "x"))

    def test_rejects_malformed_content_hash(self):
        with pytest.raises(ValueError):
            SkillSpec(name="ok", description="d", content_hash="xyz")


class TestSkillRegistry:
    def test_register_resolve_and_snapshot_ordering(self):
        registry = SkillRegistry()
        first = SkillSpec(name="a-first", description="d")
        second = SkillSpec(name="b-second", description="d")
        generation = registry.register(first)
        assert generation == 1
        registry.register(second)
        assert list(registry.snapshot()) == ["a-first", "b-second"]
        spec, gen = registry.resolve("a-first")
        assert spec is first and gen == 1

    def test_idempotent_same_spec_keeps_generation(self):
        registry = SkillRegistry()
        spec = SkillSpec(name="stable", description="d")
        first = registry.register(spec)
        again = registry.register(SkillSpec(name="stable", description="d"))
        assert first == again == 1

    def test_conflicting_spec_requires_replace(self):
        registry = SkillRegistry()
        registry.register(SkillSpec(name="s", description="old"))
        with pytest.raises(ValueError, match="replace=True"):
            registry.register(SkillSpec(name="s", description="new"))
        generation = registry.register(SkillSpec(name="s", description="new"), replace=True)
        assert generation == 2
        assert registry.get("s").description == "new"


class TestSkillDiscovery:
    def test_discovers_and_reports_disabled_and_ignored(self, skills_root: Path):
        write_skill(
            skills_root,
            "off-skill",
            "---\ndescription: d\nenabled: false\n---\nbody\n",
        )
        (skills_root / "stray-file.txt").write_text("not a dir", encoding="utf-8")
        (skills_root / "empty-dir").mkdir()
        registry = SkillRegistry()
        report = discover_skills(registry, root=skills_root)
        statuses = {r.name: r.status for r in report.records}
        assert statuses["weekly-report"] == "registered"
        assert statuses["paper-review"] == "registered"
        assert statuses["off-skill"] == "disabled"
        assert statuses["stray-file.txt"] == "ignored"
        # A directory without SKILL.md is a real error, not a silent skip.
        assert statuses["empty-dir"] == "error"
        assert report.errors[0].name == "empty-dir"
        assert not report.ok
        assert set(registry.snapshot()) == {"weekly-report", "paper-review"}

    def test_hash_changes_when_content_changes(self, skills_root: Path):
        registry = SkillRegistry()
        first_report = discover_skills(registry, root=skills_root)
        first_hash = registry.get("weekly-report").content_hash
        assert len(first_hash) == 64
        path = skills_root / "weekly-report" / "SKILL.md"
        path.write_text(VALID_SKILL_MD + "\nextra line\n", encoding="utf-8")
        second_report = discover_skills(registry, root=skills_root, replace=True)
        assert second_report.ok
        second_hash = registry.get("weekly-report").content_hash
        assert first_hash != second_hash

    def test_missing_frontmatter_and_unknown_keys_error(self, skills_root: Path):
        write_skill(skills_root, "no-front", "just body, no frontmatter\n")
        write_skill(
            skills_root,
            "unknown-key",
            "---\ndescription: d\nbogus: x\n---\nbody\n",
        )
        registry = SkillRegistry()
        report = discover_skills(registry, root=skills_root)
        errors = {r.name for r in report.errors}
        assert {"no-front", "unknown-key"} <= errors

    def test_strict_scan_raises_skill_discovery_error(self, skills_root: Path):
        (skills_root / "broken").mkdir()
        registry = SkillRegistry()
        with pytest.raises(SkillDiscoveryError):
            discover_skills(registry, root=skills_root, strict=True)

    def test_read_skill_content_returns_current_body(self, skills_root: Path):
        registry = SkillRegistry()
        discover_skills(registry, root=skills_root)
        content = read_skill_content(registry.get("weekly-report"))
        assert "Follow the numbered structure." in content


class TestSkillCatalogTool:
    def test_list_view_and_unknown_skill(self, skills_root: Path):
        registry = SkillRegistry()
        discover_skills(registry, root=skills_root)
        tool = SkillCatalogTool(registry, root=skills_root)

        listed = tool.execute(SkillCatalogInput(action="list"))
        names = [entry["skill_name"] for entry in listed.skills]
        assert names == ["paper-review", "weekly-report"]
        # Directory entries carry metadata only, never SKILL.md bodies.
        assert all(entry["description"] for entry in listed.skills)
        assert all("SKILL.md" not in json.dumps(entry) for entry in listed.skills)

        viewed = tool.execute(
            SkillCatalogInput(action="view", skill_name="weekly-report")
        )
        assert isinstance(viewed, SkillCatalogOutput)
        assert "Follow the numbered structure." in viewed.content
        assert viewed.version == "1.2.0"
        assert len(viewed.content_hash) == 64

        with pytest.raises(ValueError, match="not registered"):
            tool.execute(SkillCatalogInput(action="view", skill_name="ghost"))

    def test_view_requires_skill_name(self):
        tool = SkillCatalogTool(SkillRegistry())
        with pytest.raises(ValueError, match="skill_name is required"):
            tool.execute(SkillCatalogInput(action="view"))

    def test_read_reference_blocks_traversal_and_missing_files(
        self, skills_root: Path
    ):
        registry = SkillRegistry()
        discover_skills(registry, root=skills_root)
        tool = SkillCatalogTool(registry, root=skills_root)
        references = skills_root / "weekly-report" / "references"
        references.mkdir()
        (references / "guide.md").write_text("reference body", encoding="utf-8")

        result = tool.execute(
            SkillCatalogInput(
                action="read_reference",
                skill_name="weekly-report",
                reference_path="references/guide.md",
            )
        )
        assert result.reference_content == "reference body"
        assert result.reference_path == "references/guide.md"

        for bad in ("../../etc/passwd", "references/missing.md", "."):
            with pytest.raises((ValueError, FileNotFoundError)):
                tool.execute(
                    SkillCatalogInput(
                        action="read_reference",
                        skill_name="weekly-report",
                        reference_path=bad,
                    )
                )


class TestAgentIntegration:
    def _make_agent(self, skills_root: Path, **kwargs):
        from agents.agent import Agent

        class DemoAgent(Agent):
            def run(self, query: str) -> str:
                return query

        return DemoAgent(
            "skill-test",
            llm=None,
            provider_registry=None,
            auto_discover_tools=False,
            skills_root=str(skills_root),
            **kwargs,
        )

    def test_agent_registers_skill_catalog_and_discovers(self, skills_root: Path):
        agent = self._make_agent(skills_root)
        assert "system.skill_catalog" in agent.tools
        assert set(agent.skills.snapshot()) == {"paper-review", "weekly-report"}
        assert agent.skill_discovery_report is not None
        assert agent.skill_discovery_report.ok

    def test_skill_catalog_tool_executes_through_registry(self, skills_root: Path):
        import asyncio

        agent = self._make_agent(skills_root)
        registration = agent.tools.resolve("system.skill_catalog")
        tool, _ = registration
        result = tool.execute(SkillCatalogInput(action="view", skill_name="paper-review"))
        assert "正文内容" in result.content

    def test_system_message_lists_directory_without_content(self, skills_root: Path):
        agent = self._make_agent(skills_root)
        message = agent._with_registered_skill_names()
        assert message is not None
        text = message["content"]
        assert "paper-review" in text and "weekly-report" in text
        assert "审稿" in text  # triggers surface in the directory
        assert "正文内容" not in text and "numbered structure" not in text

    def test_empty_skills_root_omits_message(self, tmp_path: Path):
        agent = self._make_agent(tmp_path / "does-not-exist")
        assert agent._with_registered_skill_names() is None
        assert len(agent.skills) == 0

    def test_tool_names_message_precedes_skill_message(
        self, skills_root: Path
    ):
        agent = self._make_agent(skills_root)
        messages = agent._with_registered_tool_names(
            [{"role": "user", "content": "hi"}]
        )
        assert messages[0]["content"].startswith("All registered tool names:")
        assert messages[1] is not None and "Available skills" in messages[1]["content"]
        assert messages[2] == {"role": "user", "content": "hi"}

    def test_prompt_cache_key_reflects_skill_changes(self, skills_root: Path):
        agent = self._make_agent(skills_root)
        key_before = agent._default_prompt_cache_key("p", "m", mode="native")
        path = skills_root / "weekly-report" / "SKILL.md"
        path.write_text(VALID_SKILL_MD + "\nchanged\n", encoding="utf-8")
        agent.discover_skills(replace=True)
        key_after = agent._default_prompt_cache_key("p", "m", mode="native")
        assert key_before != key_after

        empty_root = skills_root.parent / "empty-skills-root"
        empty_root.mkdir(exist_ok=True)
        empty_agent = self._make_agent(empty_root)
        key_no_skills = empty_agent._default_prompt_cache_key("p", "m", mode="native")
        assert key_no_skills != key_after

    def test_auto_discover_skills_can_be_disabled(self, skills_root: Path):
        agent = self._make_agent(skills_root, auto_discover_skills=False)
        assert len(agent.skills) == 0
        assert agent._with_registered_skill_names() is None
        # The catalog tool itself remains available for a later manual scan.
        assert "system.skill_catalog" in agent.tools
