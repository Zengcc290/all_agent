"""一键切换本地版 / 云端版配置。

现有代码只读固定文件名 ``config/services.toml`` / ``config/provider.toml``
（gitignored，按需修改），这里不改变加载器契约，只是把选中的模板复制为生效文件：

- 本地版模板：``config/services.local.toml`` + ``config/provider.local.toml``
- 云端版模板：``config/services.cloud.toml`` + ``config/provider.cloud.toml``

用法：
    python scripts/switch_config.py local
    python scripts/switch_config.py cloud
    python scripts/switch_config.py status

复制完成后提示下一步（填密钥 / 填 .env），不会自动覆盖已手工填充的生效文件
（除非加 ``--force``；无有效模板时不写任何文件）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "config"

MODES = {
    "local": {
        "services": "services.local.toml",
        "provider": "provider.local.toml",
        "label": "本地版",
    },
    "cloud": {
        "services": "services.cloud.toml",
        "provider": "provider.cloud.toml",
        "label": "云端版",
    },
}

ACTIVE_FILES = ("services.toml", "provider.toml")


def mode_source(mode: str) -> tuple[Path, Path]:
    files = MODES[mode]
    return CONFIG_DIR / files["services"], CONFIG_DIR / files["provider"]


def switch(mode: str, *, force: bool) -> tuple[list[Path], list[Path]]:
    """复制选中模板到生效文件；返回 (已写, 跳过)。"""

    written: list[Path] = []
    skipped: list[Path] = []
    services_source, provider_source = mode_source(mode)
    for active_name, source in (
        ("services.toml", services_source),
        ("provider.toml", provider_source),
    ):
        active = CONFIG_DIR / active_name
        if not source.is_file():
            raise FileNotFoundError(f"模板不存在：{source}（请检查 config/ 目录）")
        if active.is_file() and not force:
            skipped.append(active)
            continue
        shutil.copyfile(source, active)
        written.append(active)
    return written, skipped


def status() -> str:
    """返回当前生效模式描述（逐文件）。"""

    lines: list[str] = []
    for active_name in ACTIVE_FILES:
        active = CONFIG_DIR / active_name
        if not active.is_file():
            lines.append(f"{active_name}: 缺失（尚未选择模式）")
            continue
        text = active.read_text(encoding="utf-8")
        # 通过文件头注释嗅探来源模板（模板第一行都带“本地版/云端版”字样）。
        if "本地版" in text:
            lines.append(f"{active_name}: 本地版")
        elif "云端版" in text:
            lines.append(f"{active_name}: 云端版")
        else:
            lines.append(f"{active_name}: 自定义（无法识别模板来源）")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="切换启用的配置模式：local / cloud")
    parser.add_argument(
        "mode",
        nargs="?",
        default="status",
        choices=["local", "cloud", "status"],
        help="local=本地版, cloud=云端版, status=查看当前生效模式",
    )
    parser.add_argument("--force", action="store_true", help="覆盖已存在的生效文件")
    args = parser.parse_args(argv)

    if args.mode == "status":
        print(status())
        return 0

    written, skipped = switch(args.mode, force=args.force)

    if skipped:
        print("以下生效文件已存在，跳过（用 --force 覆盖）：")
        for path in skipped:
            print(f"  {path.relative_to(ROOT)}")
    if written:
        label = MODES[args.mode]["label"]
        print(f"已切换到 {label}：")
        for path in written:
            print(f"  {path.relative_to(ROOT)}")
        print(
            "下一步：打开 config/services.toml / config/provider.toml "
            "填入真实密钥（或用 api_key_env 指向 .env），然后重启服务。"
        )
    elif not skipped:
        print(f"没有模板可复制：{MODES[args.mode]['services']} / {MODES[args.mode]['provider']} 缺失。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
