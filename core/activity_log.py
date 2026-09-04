"""Console activity logging for tool registration and ReAct execution."""

from __future__ import annotations

import logging
from collections.abc import Iterable


class _ConsoleLogHandler(logging.Handler):
    """Emit through ``print`` so interactive callers always see progress."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            print(self.format(record), flush=True)
        except Exception:  # noqa: BLE001
            self.handleError(record)


LOGGER = logging.getLogger("all_agent.activity")
if not any(isinstance(handler, _ConsoleLogHandler) for handler in LOGGER.handlers):
    console_handler = _ConsoleLogHandler()
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    )
    LOGGER.addHandler(console_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def log_tool_registration(
    tool_name: str,
    generation: int | None,
    registered_names: Iterable[str],
) -> None:
    """Record one successful registration and the complete active name list."""

    names = _tool_names(registered_names)
    generation_text = str(generation) if generation is not None else "metadata-only"
    LOGGER.info(
        "Tool registered: %s (generation: %s). Current registered tools: %s",
        tool_name,
        generation_text,
        ", ".join(names) or "无",
    )


def log_discovery_summary(package: str, registered_names: Iterable[str]) -> None:
    """Record all tool names successfully handled by one discovery scan."""

    names = _tool_names(registered_names)
    if names:
        LOGGER.info(
            "Tool discovery complete: package %s; successfully registered tools: %s",
            package,
            ", ".join(names),
        )


def log_react_round_started(round_number: int, max_rounds: int) -> None:
    """Display the ReAct round currently in progress."""

    LOGGER.info("ReAct round %d/%d started.", round_number, max_rounds)


def log_model_completed(round_number: int, elapsed_seconds: float) -> None:
    """Display model latency without logging its raw response."""

    LOGGER.info(
        "ReAct round %d model completed in %.2fs.", round_number, elapsed_seconds
    )


def log_react_thought(round_number: int, thought: str) -> None:
    """Display the model's parsed Thought in one readable line."""

    LOGGER.info("ReAct round %d thought: %s", round_number, _clean_text(thought))


def log_tool_call_started(round_number: int, tool_names: Iterable[str]) -> None:
    """Display only the tool names about to be executed."""

    names = _tool_names(tool_names)
    LOGGER.info(
        "ReAct round %d calling tools: %s.",
        round_number,
        ", ".join(names) or "none",
    )


def log_tool_call_completed(
    round_number: int,
    tool_names: Iterable[str],
    elapsed_seconds: float,
) -> None:
    """Display tool latency without logging tool arguments or return values."""

    names = _tool_names(tool_names)
    LOGGER.info(
        "ReAct round %d tools completed in %.2fs: %s.",
        round_number,
        elapsed_seconds,
        ", ".join(names) or "none",
    )


def log_react_final_answer(round_number: int, answer: str) -> None:
    """Display the final response in one readable line."""

    LOGGER.info(
        "ReAct round %d final answer: %s", round_number, _clean_text(answer)
    )


def log_react_parse_issue(round_number: int, issue: str) -> None:
    """Display a concise parse failure so the next round is explainable."""

    LOGGER.info(
        "ReAct round %d parse issue: %s", round_number, _clean_text(issue)
    )


def _tool_names(names: Iterable[str]) -> list[str]:
    return sorted({name for name in names if isinstance(name, str) and name})


def _clean_text(value: str, limit: int = 2000) -> str:
    cleaned = " ".join(str(value).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


__all__ = [
    "LOGGER",
    "log_discovery_summary",
    "log_model_completed",
    "log_react_final_answer",
    "log_react_parse_issue",
    "log_react_round_started",
    "log_react_thought",
    "log_tool_call_completed",
    "log_tool_call_started",
    "log_tool_registration",
]
