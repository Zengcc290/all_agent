"""Shared helpers for the P0 filesystem, shell and git tools.

This module is deliberately NOT discoverable: ``core.discovery`` ignores
modules whose names start with an underscore, and this file defines neither
``TOOL_ENABLED`` nor a ``create_tool()`` factory. Every helper here is pure
standard-library code with no network, file-write or process-spawn side effect
at import time.
"""

from __future__ import annotations

import codecs
import locale
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ROOT_ENV = "WORKSPACE_ROOT"
ALLOW_OUTSIDE_ENV = "WORKSPACE_ALLOW_OUTSIDE"

DEFAULT_MAX_OUTPUT_CHARS = 200_000
TRUNCATION_SUFFIX = "\n...[output truncated]"

#: Hard ceiling for a single text file decoded into model context.
MAX_TEXT_FILE_BYTES = 32 * 1024 * 1024


def workspace_root() -> Path:
    """Return the configured workspace root (defaults to the process CWD)."""
    raw = os.getenv(WORKSPACE_ROOT_ENV)
    if raw is not None and raw.strip():
        return Path(raw.strip()).expanduser().resolve()
    return Path.cwd().resolve()


def allow_outside_workspace() -> bool:
    """Return whether tools may touch paths outside the workspace root."""
    raw = os.getenv(ALLOW_OUTSIDE_ENV)
    if raw is None:
        return False
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def resolve_path(
    base_dir: str | Path,
    path: str,
    *,
    allow_outside: bool | None = None,
) -> Path:
    """Resolve ``path`` against ``base_dir`` and enforce workspace containment.

    Relative paths are anchored at ``base_dir``. Absolute paths are allowed
    only when they stay inside the resolved ``base_dir`` unless
    ``allow_outside`` (or the ``WORKSPACE_ALLOW_OUTSIDE`` environment switch)
    permits full access. ``allow_outside`` is a deployment decision made by the
    tool owner, never by the LLM through an input field.
    """
    if not isinstance(base_dir, (str, Path)):
        raise TypeError("base_dir must be a string or pathlib.Path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    base = Path(base_dir).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if allow_outside is None:
        allow_outside = allow_outside_workspace()
    if not allow_outside and not _is_within(base, resolved):
        raise ValueError(
            f"path '{path}' resolves outside the workspace root '{base}'"
        )
    return resolved


def _is_within(base: Path, candidate: Path) -> bool:
    base_text = os.path.normcase(str(base))
    candidate_text = os.path.normcase(str(candidate))
    return candidate_text == base_text or candidate_text.startswith(
        base_text + os.sep
    )


def normalize_encoding(encoding: str) -> str:
    """Canonicalize a codec name and raise a clean error for unknown ones."""
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding must be a non-empty string")
    try:
        return codecs.lookup(encoding.strip()).name
    except LookupError as exc:
        raise ValueError(f"unknown text encoding '{encoding.strip()}'") from exc


def _sanitize_surrogates(text: str) -> str:
    if not any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return text
    return "".join(char if not (0xD800 <= ord(char) <= 0xDFFF) else "\ufffd" for char in text)


def cap_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Cap ``text`` to ``max_chars`` code points without splitting surrogates."""
    text = _sanitize_surrogates(text)
    if len(text) <= max_chars:
        return text, False
    cut = max_chars
    while cut > 0 and 0xD800 <= ord(text[cut - 1]) <= 0xDFFF:
        cut -= 1
    return text[:cut] + TRUNCATION_SUFFIX, True


def read_text_file(path: Path, encoding: str) -> tuple[str, str]:
    """Decode one bounded text file; raise clean errors for binary/oversized."""
    canonical = normalize_encoding(encoding)
    stat_result = path.stat()
    if stat_result.st_size > MAX_TEXT_FILE_BYTES:
        raise ValueError(
            f"file '{path}' is too large to read as text "
            f"({stat_result.st_size} bytes > {MAX_TEXT_FILE_BYTES})"
        )
    raw = path.read_bytes()
    # NUL bytes are a reliable binary marker only for byte-oriented encodings;
    # UTF-16/UTF-32 legitimately contain NUL for every ASCII character.
    if b"\x00" in raw and "16" not in canonical and "32" not in canonical:
        raise ValueError("file appears to be binary (contains NUL bytes)")
    try:
        text = raw.decode(canonical)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"file '{path}' is not valid '{canonical}' text: {exc}"
        ) from exc
    return text, canonical


def atomic_write_text(
    path: Path,
    text: str,
    encoding: str,
    *,
    create_parents: bool,
) -> int:
    """Atomically write ``text`` to ``path`` and return the byte count.

    The parent directory is created when ``create_parents`` is true. Encoding
    happens before any file is touched so a non-encodable payload never leaves
    a partial file behind.
    """
    canonical = normalize_encoding(encoding)
    parent = path.parent
    if parent.exists():
        if not parent.is_dir():
            raise NotADirectoryError(f"parent path is not a directory: '{parent}'")
    elif create_parents:
        parent.mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError(f"parent directory does not exist: '{parent}'")
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"'{path}' is a directory, not a file")
    try:
        data = text.encode(canonical)
    except UnicodeEncodeError as exc:
        raise ValueError(f"text cannot be encoded as '{canonical}'") from exc
    descriptor, tmp_name = tempfile.mkstemp(
        dir=str(parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return len(data)


@dataclass
class ProcOutcome:
    """Normalized result of one subprocess run with bounded output."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_seconds: float = 0.0
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def run_process(
    command: str | list[str],
    *,
    cwd: str | Path | None = None,
    timeout_seconds: float = 60.0,
    env: Mapping[str, str] | None = None,
    shell: bool = False,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ProcOutcome:
    """Run one command with a hard deadline and bounded captured output.

    On timeout the whole process tree is terminated and the partial output is
    still returned with ``timed_out=True``. On Windows a process group is used
    so child processes do not outlive the shell.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise TypeError("timeout_seconds must be a finite positive number")
    if not 0.0 < timeout_seconds < 3600:
        raise ValueError("timeout_seconds must be between 0 and 3600")
    if env is not None and not isinstance(env, Mapping):
        raise TypeError("env must be a mapping of strings to strings")
    if not isinstance(max_output_chars, int) or isinstance(
        max_output_chars, bool
    ) or max_output_chars < 1:
        raise ValueError("max_output_chars must be a positive integer")

    merged_env = dict(os.environ)
    if env is not None:
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("env must map strings to strings")
            merged_env[key] = value

    start_kwargs: dict = {}
    if sys.platform == "win32":
        start_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        start_kwargs["start_new_session"] = True

    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        **start_kwargs,
    )
    started = time.monotonic()
    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(process)
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_bytes, stderr_bytes = process.communicate()
    duration_seconds = round(time.monotonic() - started, 3)
    exit_code = (
        -1
        if timed_out
        else (process.returncode if process.returncode is not None else 0)
    )
    stdout, stdout_truncated = _decode_capped(stdout_bytes, max_output_chars)
    stderr, stderr_truncated = _decode_capped(stderr_bytes, max_output_chars)
    return ProcOutcome(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=duration_seconds,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def git_env() -> dict[str, str]:
    """Base environment for git subprocesses: never hang on interactive prompts.

    Deliberately sets only ``GIT_TERMINAL_PROMPT=0``: overriding GIT_ASKPASS or
    setting GIT_CONFIG_NOSYSTEM breaks credential helpers such as Git Credential
    Manager (registered at the system level on Windows), which then cannot
    authenticate with stored credentials. With prompts disabled, git fails fast
    instead of hanging when no credential is available.
    """
    return {
        "GIT_TERMINAL_PROMPT": "0",
    }


def run_git(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    timeout_seconds: float = 60.0,
) -> ProcOutcome:
    """Run git with terminal prompting disabled so it can never hang."""
    return run_process(
        ["git", *args],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        env=git_env(),
        shell=False,
    )


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/pid", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except OSError:
                pass


def _decode_capped(data: bytes | None, max_chars: int) -> tuple[str, bool]:
    if not data:
        return "", False
    encodings = ["utf-8"]
    preferred = locale.getpreferredencoding(False)
    if preferred and preferred.casefold() != "utf-8":
        encodings.append(preferred)
    text = None
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except (LookupError, UnicodeDecodeError):
            continue
    if text is None:
        text = data.decode("utf-8", errors="replace")
    return cap_text(text, max_chars)