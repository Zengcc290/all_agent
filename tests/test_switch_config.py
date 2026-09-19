"""本地版 / 云端版配置切换脚本的最小回归集。

覆盖：
① switch_config.local 模板可解析（tomllib），生成的有效配置能加载；
② status 能识别本地版 / 云端版 / 自定义三种来源；
③ 不传 --force 时跳过已存在的生效文件，--force 才覆盖；
④ 无模板时返回非零且不写文件。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import switch_config


@pytest.fixture()
def fake_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把脚本的 CONFIG_DIR 指到临时目录，避免碰真实 config/。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "services.local.toml",
        "services.cloud.toml",
        "provider.local.toml",
        "provider.cloud.toml",
    ):
        source = Path(__file__).resolve().parent.parent / "config" / name
        (config_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(switch_config, "CONFIG_DIR", config_dir)
    return config_dir


def test_templates_are_valid_toml_and_parseable() -> None:
    """模板本身合法，且能过现有加载器（load_services_config / ProviderRegistry）。"""

    from agents.providers import ProviderRegistry
    from core.services_config import load_services_config

    root = Path(__file__).resolve().parent.parent / "config"
    for name in ("services.local.toml", "services.cloud.toml"):
        path = root / name
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        assert isinstance(document, dict)
        services = load_services_config(path)
        assert services.configured

    for name in ("provider.local.toml", "provider.cloud.toml"):
        registry = ProviderRegistry(root / name)
        assert registry.active_profile
        assert registry.profiles


def test_switch_local_copies_templates_to_active_files(fake_config: Path) -> None:
    written, skipped = switch_config.switch("local", force=False)

    assert len(written) == 2
    assert skipped == []
    active_services = fake_config / "services.toml"
    active_provider = fake_config / "provider.toml"
    assert active_services.is_file() and active_provider.is_file()
    assert "本地版" in active_services.read_text(encoding="utf-8")
    assert "本地版" in active_provider.read_text(encoding="utf-8")


def test_switch_skips_existing_without_force_and_overwrites_with_force(fake_config: Path) -> None:
    first_written, _ = switch_config.switch("cloud", force=False)
    assert len(first_written) == 2

    # 已存在 + 无 force → 跳过
    written, skipped = switch_config.switch("local", force=False)
    assert written == []
    assert len(skipped) == 2
    assert "云端版" in (fake_config / "services.toml").read_text(encoding="utf-8")

    # force → 覆盖为本地版
    written, skipped = switch_config.switch("local", force=True)
    assert len(written) == 2
    assert skipped == []
    assert "本地版" in (fake_config / "services.toml").read_text(encoding="utf-8")


def test_status_detects_local_cloud_and_custom(fake_config: Path, tmp_path: Path) -> None:
    switch_config.switch("local", force=True)
    text = switch_config.status()
    assert "本地版" in text

    switch_config.switch("cloud", force=True)
    text = switch_config.status()
    assert "云端版" in text

    # 手改生效文件 → 无法识别来源
    (fake_config / "services.toml").write_text("[embedding]\nprovider = \"hash\"\n", encoding="utf-8")
    text = switch_config.status()
    assert "自定义（无法识别模板来源）" in text


def test_missing_templates_do_not_write(fake_config: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty_config"
    empty.mkdir()
    import switch_config as module

    module.CONFIG_DIR = empty
    with pytest.raises(FileNotFoundError):
        module.switch("local", force=False)
    assert not (empty / "services.toml").exists()
    assert not (empty / "provider.toml").exists()
