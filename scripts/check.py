"""One-command quality gate: ``ruff check`` then the full ``pytest`` suite.

Run from anywhere:

    python scripts/check.py

Both gates must pass; the first failure stops the run and returns its exit code,
so the script is usable as a pre-commit hook or a CI step. Only the standard
library is used, and the current interpreter (not a hard-coded path) runs both
tools, so it works in any activated virtualenv on Windows or POSIX.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff", ("-m", "ruff", "check", ".")),
    ("pytest", ("-m", "pytest", "-q")),
)


def main() -> int:
    for name, arguments in GATES:
        print(f"==> {name}: {sys.executable} {' '.join(arguments)}", flush=True)
        completed = subprocess.run(
            [sys.executable, *arguments], cwd=ROOT, check=False
        )
        if completed.returncode != 0:
            print(
                f"==> {name} failed with exit code {completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
    print("==> all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
