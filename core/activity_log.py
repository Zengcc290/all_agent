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
    """Keep tool registration silent in the user-facing activity stream."""


def log_discovery_summary(package: str, registered_names: Iterable[str]) -> None:
    """Keep discovery details out of the user-facing activity stream."""


def log_react_round_started(round_number: int, max_rounds: int | None) -> None:
    """Rounds are internal control flow and are intentionally not displayed."""


def log_model_completed(round_number: int, elapsed_seconds: float) -> None:
    """Display the model round latency in a compact, human-readable form."""

    LOGGER.info("模型思考耗时：第%d轮 %.3f秒", round_number, max(0.0, elapsed_seconds))


def log_model_first_chunk(round_number: int, elapsed_seconds: float) -> None:
    """Display the first streaming chunk latency for the model round."""

    LOGGER.info(
        "模型首字返回：第%d轮 %.3f秒（流式连接已建立）",
        round_number,
        max(0.0, elapsed_seconds),
    )


def log_react_thought(round_number: int, thought: str) -> None:
    """Display only the model's current thought."""

    LOGGER.info("模型思考：%s", _clean_text(thought))


def log_tool_call_started(round_number: int, tool_names: Iterable[str]) -> None:
    """Display only the tool names about to be executed."""

    names = _tool_names(tool_names)
    LOGGER.info("调用工具：%s", ", ".join(names) or "无")


def log_tool_call_completed(
    round_number: int,
    tool_names: Iterable[str],
    elapsed_seconds: float,
) -> None:
    """Display tool latency without dumping potentially large tool results."""

    names = _tool_names(tool_names)
    LOGGER.info(
        "调用工具耗时：第%d轮 %s %.3f秒",
        round_number,
        ", ".join(names) or "无",
        max(0.0, elapsed_seconds),
    )


def log_react_final_answer(round_number: int, answer: str) -> None:
    """The final answer is already returned to the caller; keep it silent."""


def log_react_parse_issue(round_number: int, issue: str) -> None:
    """Show protocol errors so a stuck tool call is diagnosable."""

    LOGGER.warning("工具调用格式问题：第%d轮 %s", round_number, _clean_text(issue))


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
