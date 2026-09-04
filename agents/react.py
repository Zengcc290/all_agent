"""ReAct-style agent built on top of the project's :class:`Agent` runtime.

The normal ``Agent.run_with_tools`` method speaks the OpenAI function-calling
protocol.  ``ReActAgent`` is useful for providers (or local models) which only
return text: it asks the model for ``Thought``/``Action``/``Action Input``
steps, executes actions through the same registry and execution manager, and
feeds a structured ``Observation`` back to the model.

No tool implementation is duplicated here. Registry generations, schemas,
side-effect confirmations, timeouts and output validation are all delegated to
``Agent.execute_tool_calls``. Permission declarations remain compatibility
metadata and are not used as an authorization gate.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from core import (
    ExecutionContext,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpecRepository,
    ToolRegistry,
    discover_tools as discover_tool_modules,
)
from core.activity_log import (
    log_model_completed,
    log_react_final_answer,
    log_react_parse_issue,
    log_react_round_started,
    log_react_thought,
    log_tool_call_completed,
    log_tool_call_started,
    log_tool_registration,
)
from core.registry import BaseTool
from core.parser import parse_openai_tool_calls

from .agent import (
    Agent,
    _field,
    _message_dict,
    _result_json,
    _safe_tool_call_error,
    _safe_tool_name,
)


_FINAL_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*:\s*(.*?)"
    r"(?=^\s*(?:\*\*)?(?:thought|action|action\s+input|observation|"
    r"final\s+answer)(?:\*\*)?\s*:|\Z)",
    re.DOTALL,
)
_ACTION_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?action(?:\*\*)?\s*:\s*([^\r\n]+?)\s*$"
)
_ACTION_INPUT_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?action\s+input(?:\*\*)?\s*:\s*(.*?)"
    r"(?=^\s*(?:\*\*)?(?:thought|action|action\s+input|observation|"
    r"final\s+answer)(?:\*\*)?\s*:|\Z)",
    re.DOTALL,
)
_XML_ANSWER_RE = re.compile(r"(?is)^\s*<answer>\s*(.*?)\s*</answer>\s*$")
_EXPLICIT_FINAL_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*:"
)


@dataclass(frozen=True)
class ParsedReActResponse:
    """The safe-to-process parts of one model response.

    ``arguments`` is ``None`` when the action input was absent or malformed.
    Parsing never executes a tool; callers can turn ``error`` into an
    observation and let the model correct itself.
    """

    raw: str
    thought: str | None = None
    final_answer: str | None = None
    action: str | None = None
    arguments: dict[str, Any] | None = None
    error: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None

    @property
    def has_action(self) -> bool:
        return self.action is not None


def parse_react_response(text: str) -> ParsedReActResponse:
    """Parse a text ReAct response without executing anything.

    The parser accepts the common formats below (marker names are
    case-insensitive and may be bolded)::

        Thought: ...
        Action: namespace.tool
        Action Input: {"value": 1}

        Final Answer: ...

    A response without an action marker is treated as a final answer.  JSON
    action inputs must be objects and are decoded with non-finite constants
    disabled, matching the project's native parser.
    """

    if not isinstance(text, str):
        raise TypeError("ReAct response must be a string")
    raw = text
    final_match = _FINAL_RE.search(text)
    thought = _extract_thought(text)
    action_match = _ACTION_RE.search(text)

    if final_match is not None:
        answer = final_match.group(1).strip()
        # Some models put the answer on the line following ``Final Answer:``.
        if not answer:
            start = final_match.end()
            answer = text[start:].strip()
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            final_answer=answer,
        )

    xml_answer = _XML_ANSWER_RE.fullmatch(text)
    if xml_answer is not None:
        answer = xml_answer.group(1).strip()
        if answer:
            return ParsedReActResponse(
                raw=raw,
                thought=thought,
                final_answer=answer,
            )
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            error="The <answer> tag must contain a non-empty final response",
        )

    if text.strip().casefold() in {"<answer>", "<final_answer>", "[answer]"}:
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            error="The model returned an answer placeholder instead of a final response",
        )

    if action_match is None:
        # Plain text is a valid provider fallback when the model omits the
        # optional protocol markers.
        return ParsedReActResponse(raw=raw, thought=thought, final_answer=text.strip())

    action = action_match.group(1).strip().strip("`").strip()
    if not action:
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            error="Action must contain a non-empty tool name",
        )

    input_match = _ACTION_INPUT_RE.search(text, action_match.end())
    if input_match is None:
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            action=action,
            error="Action Input is required and must be a JSON object",
        )

    payload = _strip_code_fence(input_match.group(1).strip())
    if not payload:
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            action=action,
            error="Action Input is required and must be a JSON object",
        )
    try:
        arguments = json.loads(payload, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            action=action,
            error=f"Action Input is not valid JSON: {exc}",
        )
    if not isinstance(arguments, dict):
        return ParsedReActResponse(
            raw=raw,
            thought=thought,
            action=action,
            error="Action Input must be a JSON object",
        )
    return ParsedReActResponse(
        raw=raw,
        thought=thought,
        action=action,
        arguments=arguments,
    )


def _extract_thought(text: str) -> str | None:
    match = re.search(
        r"(?im)^\s*(?:\*\*)?thought(?:\*\*)?\s*:\s*(.*?)"
        r"(?=^\s*(?:\*\*)?(?:action|action\s+input|observation|"
        r"final\s+answer)(?:\*\*)?\s*:|\Z)",
        text,
        re.DOTALL,
    )
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _strip_code_fence(value: str) -> str:
    lines = value.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    return "\n".join(lines).strip()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


class ReActAgent(Agent):
    """An ``Agent`` variant using a textual ReAct protocol.

    ``run_with_react`` accepts either a user query string or the same list of
    role/content mappings accepted by ``Agent.run_with_tools``.  All inherited
    provider and tool-management methods remain available.
    """

    REACT_INSTRUCTIONS = (
        "You are a ReAct agent. Work in short, explicit steps. When a tool is "
        "needed, respond exactly with:\n"
        "Thought: brief reasoning text\n"
        "Action: exact tool name\n"
        "Action Input: one JSON object\n"
        "Wait for an Observation before taking another action. When the task is "
        "complete, respond with:\nFinal Answer: your final response. Never put an "
        "Observation in your own response. Use only the listed tool names and "
        "JSON object inputs. Do not output placeholder text such as <answer>.\n\n"
        "Use a listed tool directly when its complete schema is available. The "
        "system.tool_catalog tool is optional in that mode."
    )

    CATALOG_FIRST_REACT_INSTRUCTIONS = (
        "Every name in `All registered tool names` is already registered and usable. "
        "That list is a capability index; a name without an Action Input schema in "
        "this request is not unavailable or blocked. Before calling any listed tool "
        "other than system.tool_catalog whose schema has not been provided, first "
        "call system.tool_catalog with `action`: `resolve` and an `intent` describing "
        "the capability you need (for example `web search`, `read a file`, or `send "
        "email`), never the user's actual subject/query. Catalog resolution only "
        "retrieves a tool contract; it is not a registration, permission, or access "
        "request. After its Observation returns the schema, call the resolved tool "
        "with the user's words in its input. Do not claim that a tool is unavailable "
        "merely because a catalog lookup failed; correct the capability intent and "
        "retry. Resolve multiple capabilities together with `limit`: 20 when possible."
    )

    def __init__(
        self,
        name: str,
        *,
        llm: Any | None = None,
        provider_config: str | None = None,
        provider_registry: Any | None = None,
        repository: ToolSpecRepository | None = None,
        auto_discover_tools: bool = True,
        tool_package: str | Any = "tool",
        discovery_strict: bool = False,
        lazy_tools: bool = False,
    ) -> None:
        """Create an agent with eager tool registration by default.

        Set ``lazy_tools=True`` explicitly to persist metadata and defer
        implementation construction until a catalog lookup or direct call.
        """

        if not isinstance(lazy_tools, bool):
            raise TypeError("lazy_tools must be a boolean")
        if repository is None and lazy_tools and auto_discover_tools:
            repository = ToolSpecRepository()
        self.lazy_tools = lazy_tools
        super().__init__(
            name,
            llm=llm,
            provider_config=provider_config,
            provider_registry=provider_registry,
            repository=repository,
            auto_discover_tools=False,
            tool_package=tool_package,
            discovery_strict=discovery_strict,
        )
        # Keep the catalog backed by SQLite even after one implementation has
        # been loaded; otherwise subsequent searches would only see the small
        # set of already-used tools in the runtime registry.
        self.catalog_tool.repository_only = lazy_tools
        if auto_discover_tools:
            self.discover_tools(strict=discovery_strict)

    def discover_tools(
        self,
        *,
        package: str | Any | None = None,
        replace: bool = False,
        strict: bool = False,
        reload_modules: bool = False,
    ):
        """Persist discovered metadata without retaining tool instances."""

        if not self.lazy_tools:
            report = super().discover_tools(
                package=package,
                replace=replace,
                strict=strict,
                reload_modules=reload_modules,
            )
            return report
        if self.repository is None:
            raise RuntimeError("lazy tool discovery requires a ToolSpecRepository")
        selected_package = self.tool_package if package is None else package
        scratch_registry = ToolRegistry()
        report = discover_tool_modules(
            scratch_registry,
            package=selected_package,
            repository=self.repository,
            replace=replace,
            strict=strict,
            reload_modules=reload_modules,
            metadata_only=True,
        )
        # Drop the temporary registry and all factory-created instances. The
        # active registry intentionally retains only system.tool_catalog.
        scratch_registry = None
        self.tool_discovery_report = report
        return report

    def register_tool(self, tool: BaseTool, *, replace: bool = False) -> None:
        """Persist a tool contract and defer construction until first use."""

        if not self.lazy_tools:
            return super().register_tool(tool, replace=replace)
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be a BaseTool instance")
        if self.repository is None:
            raise RuntimeError("lazy tool registration requires a ToolSpecRepository")
        implementation_ref = f"{type(tool).__module__}:{type(tool).__qualname__}"
        self.repository.save(
            tool.spec,
            implementation_ref=implementation_ref,
            replace=replace,
        )
        log_tool_registration(
            tool.spec.name,
            None,
            self.repository.active_tool_names(),
        )

    def is_tool_registered(
        self,
        name: str,
        *,
        version: str | None = None,
        schema_hash: str | None = None,
    ) -> bool:
        if not self.lazy_tools:
            return super().is_tool_registered(
                name, version=version, schema_hash=schema_hash
            )
        if name == self.catalog_tool.spec.name:
            return super().is_tool_registered(
                name, version=version, schema_hash=schema_hash
            )
        if self.repository is None:
            return False
        stored = self.repository.get(name, version)
        return stored is not None and (
            schema_hash is None or stored["schema_hash"] == schema_hash
        )

    def tool_registration_status(self, name: str) -> dict[str, Any]:
        if not self.lazy_tools or name == self.catalog_tool.spec.name:
            return super().tool_registration_status(name)
        if self.repository is None:
            return super().tool_registration_status(name)
        stored = self.repository.get(name)
        if stored is None:
            return super().tool_registration_status(name)
        return {
            "name": name,
            "registered": True,
            "version": stored["version"],
            "schema_hash": stored["schema_hash"],
            "generation": (
                self.tools.registration_status(name)["generation"]
                if name in self.tools
                else None
            ),
            "implementation": stored["implementation_ref"],
        }

    def tool_confirmation_key(self, name: str) -> str:
        """Return a confirmation key without loading a cataloged tool.

        Unloaded tools start at registry generation ``1`` when first loaded;
        once an implementation is active, the registry supplies its exact
        generation-bound key.
        """

        if name in self.tools:
            return self.tools.confirmation_key(name)
        if self.repository is None:
            raise KeyError(f"tool '{name}' is not registered")
        stored = self.repository.get(name)
        if stored is None:
            raise KeyError(f"tool '{name}' is not registered")
        return f"{name}@{stored['version']}#{stored['schema_hash']}:1"

    get_tool_confirmation_key = tool_confirmation_key

    def run(self, query: str, **kwargs: Any) -> str:
        """Synchronous convenience entry point for the ReAct loop."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return _run_sync(self.run_with_react(query, **kwargs))

    async def arun(self, query: str, **kwargs: Any) -> str:
        """Async alias matching common agent APIs."""

        return await self.run_with_react(query, **kwargs)

    async def run_react(
        self,
        messages: str | Sequence[Mapping[str, Any]],
        context: ExecutionContext | None = None,
        *,
        max_rounds: int = 100,
        model: str | None = None,
        temperature: float = 0.7,
        timeout: float = 60,
        tool_names: list[str] | None = None,
        profile_name: str | None = None,
        provider_name: str | None = None,
        use_history: bool = False,
        defer_tool_loading: bool = True,
    ) -> str:
        """Run one textual ReAct conversation to a final answer."""

        return await self.run_with_react(
            messages,
            context,
            max_rounds=max_rounds,
            model=model,
            temperature=temperature,
            timeout=timeout,
            tool_names=tool_names,
            profile_name=profile_name,
            provider_name=provider_name,
            use_history=use_history,
            defer_tool_loading=defer_tool_loading,
        )

    async def run_with_react(
        self,
        messages: str | Sequence[Mapping[str, Any]],
        context: ExecutionContext | None = None,
        *,
        max_rounds: int = 100,
        model: str | None = None,
        temperature: float = 0.7,
        timeout: float = 60,
        tool_names: list[str] | None = None,
        profile_name: str | None = None,
        provider_name: str | None = None,
        use_history: bool = False,
        defer_tool_loading: bool = True,
    ) -> str:
        if isinstance(messages, str):
            if not messages.strip():
                raise ValueError("messages must contain a non-empty query")
            request_messages: list[dict[str, Any]] = [
                {"role": "user", "content": messages}
            ]
        elif isinstance(messages, Sequence) and not isinstance(messages, (bytes, bytearray)):
            if not all(isinstance(message, Mapping) for message in messages):
                raise TypeError("messages must be a sequence of mapping objects")
            request_messages = [dict(message) for message in messages]
        else:
            raise TypeError("messages must be a string or sequence of mappings")

        if (
            isinstance(max_rounds, bool)
            or not isinstance(max_rounds, int)
            or max_rounds < 1
        ):
            raise ValueError("max_rounds must be a positive integer")
        if context is None:
            context = ExecutionContext()
        elif not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext instance")
        if not isinstance(use_history, bool) or not isinstance(
            defer_tool_loading, bool
        ):
            raise TypeError("use_history and defer_tool_loading must be booleans")

        completion_llm, selected_model, history_key = self._completion_target(
            profile_name, model, provider_name=provider_name
        )
        conversation = self._initial_conversation(
            request_messages, use_history=use_history, history_key=history_key
        )
        # A provider can still occasionally copy the user's subject into the
        # catalog ``intent`` despite the explicit protocol instructions. Keep
        # a narrow, read-only recovery hint for common search wording so that
        # such a malformed lookup does not make the model conclude that no
        # search tool exists.
        catalog_intent_hint = self._infer_catalog_intent(request_messages)

        initial_snapshot = self.tools.snapshot()
        # Permission metadata is retained for compatibility and audit output,
        # but it is not an authorization filter in this deployment.
        loaded_tool_schemas: dict[str, dict[str, Any]] = {}
        visible_order = list(initial_snapshot)
        if tool_names is not None:
            if not isinstance(tool_names, (list, tuple)) or not all(
                isinstance(name, str) and name for name in tool_names
            ):
                raise TypeError("tool_names must be a list of non-empty strings")
            requested_order = list(dict.fromkeys(tool_names))
            unknown = {
                name
                for name in requested_order
                if name not in initial_snapshot
                and (self.repository is None or self.repository.get(name) is None)
            }
            if unknown:
                raise ValueError("unknown tool name(s): " + ", ".join(sorted(unknown)))
            loaded_order = (
                [self.catalog_tool.spec.name, *requested_order]
                if self.lazy_tools
                else requested_order
            )
            requested_names: set[str] | None = set(requested_order)
        elif defer_tool_loading or self.lazy_tools:
            loaded_order = [self.catalog_tool.spec.name]
            requested_names = None
        else:
            loaded_order = visible_order
            requested_names = None

        for round_number in range(1, max_rounds + 1):
            log_react_round_started(round_number, max_rounds)
            if self.lazy_tools:
                for name in loaded_order:
                    if name != self.catalog_tool.spec.name and name not in self.tools:
                        self._ensure_tool_loaded(name)
            current_snapshot = self.tools.snapshot()
            registrations = {
                name: current_snapshot[name]
                for name in loaded_order
                if name in current_snapshot
            }
            for name, (tool, _) in registrations.items():
                loaded_tool_schemas.setdefault(name, tool.spec.input_schema)
            conversation_for_request = self._with_tool_instructions(
                conversation,
                registrations,
                loaded_tool_schemas,
                catalog_first=defer_tool_loading or self.lazy_tools,
            )
            options: dict[str, Any] = {
                "model": selected_model,
                "temperature": temperature,
                "timeout": timeout,
                "stream": False,
            }
            model_started_at = time.perf_counter()
            response = await asyncio.to_thread(
                self._complete_text_or_response,
                completion_llm,
                conversation_for_request,
                options,
            )
            log_model_completed(round_number, time.perf_counter() - model_started_at)
            if isinstance(response, str):
                message = None
                content = response
                native_calls = []
            else:
                message = self._response_message(response)
                content = _field(message, "content")
                native_calls = _field(message, "tool_calls") or []
            assistant_message = (
                _message_dict(message)
                if message is not None
                else {"role": "assistant", "content": content or ""}
            )
            if native_calls:
                conversation.append(assistant_message)
                observations = await self._execute_native_calls(
                    native_calls,
                    current_snapshot,
                    registrations,
                    context,
                    round_number=round_number,
                    intent_hint=catalog_intent_hint,
                    loaded_tool_schemas=loaded_tool_schemas,
                )
                conversation.extend(observations)
                if (defer_tool_loading or self.lazy_tools) and requested_names is None:
                    for observation in observations:
                        self._load_catalog_observation(
                            observation,
                            loaded_order,
                            context,
                            loaded_tool_schemas,
                        )
                continue

            if content is None:
                content = ""
            if not isinstance(content, str):
                content = str(content)
            parsed = parse_react_response(content)
            conversation.append({"role": "assistant", "content": content})
            if parsed.thought is not None:
                log_react_thought(round_number, parsed.thought)
            if parsed.is_final:
                if (
                    not _EXPLICIT_FINAL_RE.search(content)
                    and not _XML_ANSWER_RE.fullmatch(content)
                    and self._should_require_tool_action(request_messages, registrations)
                ):
                    protocol_error = (
                        "You returned an unmarked answer before using the available "
                        "tools. This request requires tool evidence. Do not answer "
                        "from memory. First respond with exactly one Thought, Action, "
                        "and Action Input for the most relevant listed tool; wait "
                        "for its Observation before giving Final Answer."
                    )
                    log_react_parse_issue(round_number, protocol_error)
                    conversation.append(
                        {"role": "user", "content": "Observation: " + protocol_error}
                    )
                    continue
                log_react_final_answer(round_number, parsed.final_answer or "")
                self._profile_histories[history_key] = [dict(item) for item in conversation]
                self.history = [dict(item) for item in conversation]
                return parsed.final_answer or ""

            if parsed.error is not None and not parsed.has_action:
                log_react_parse_issue(round_number, parsed.error)
                conversation.append(
                    {
                        "role": "user",
                        "content": "Observation: " + parsed.error,
                    }
                )
                continue

            if not parsed.has_action:
                # A plain text response is already interpreted as a final
                # answer by the parser; this branch is defensive only.
                log_react_final_answer(round_number, content.strip())
                self._profile_histories[history_key] = [dict(item) for item in conversation]
                self.history = [dict(item) for item in conversation]
                return content.strip()

            result = await self._execute_action(
                parsed,
                current_snapshot,
                registrations,
                context,
                call_number=round_number,
                intent_hint=catalog_intent_hint,
                loaded_tool_schemas=loaded_tool_schemas,
            )
            if catalog_intent_hint is not None:
                recovered = await self._recover_catalog_lookup(
                    parsed,
                    result,
                    current_snapshot,
                    registrations,
                    context,
                    call_number=round_number,
                    intent_hint=catalog_intent_hint,
                )
                if recovered is not None:
                    result = recovered
            conversation.append(
                {
                    "role": "user",
                    "content": f"Observation: {_result_json(result)}",
                }
            )
            if (defer_tool_loading or self.lazy_tools) and requested_names is None:
                self._load_catalog_result(
                    result, loaded_order, context, loaded_tool_schemas
                )

        raise RuntimeError("maximum ReAct rounds exceeded")

    @staticmethod
    def _should_require_tool_action(
        request_messages: Sequence[Mapping[str, Any]],
        registrations: Mapping[str, tuple[Any, int]],
    ) -> bool:
        """Decide whether an unmarked plain-text answer should be retried.

        This is intentionally conservative: only requests containing common
        real-time/search/tool-intent terms are gated, so normal small-talk can
        still use the provider's plain-text fallback.
        """

        if not registrations:
            return False
        text = " ".join(
            str(message.get("content", ""))
            for message in request_messages
            if message.get("role") == "user"
        ).casefold()
        return any(
            marker in text
            for marker in (
                "现在",
                "当前时间",
                "今天",
                "实时",
                "查询",
                "搜索",
                "查找",
                "检索",
                "运势",
                "天气",
                "search",
                "look up",
                "current time",
                "latest",
                "today",
            )
        )

    @staticmethod
    def _infer_catalog_intent(
        request_messages: Sequence[Mapping[str, Any]],
    ) -> str | None:
        """Infer a capability hint for recovering an accidental bad lookup.

        This intentionally recognizes only search wording. It never exposes a
        tool list or bypasses the catalog; the retry below still executes
        ``system.tool_catalog`` and applies the normal catalog checks.
        """

        text_parts = [
            str(message.get("content", ""))
            for message in request_messages
            if message.get("role") == "user"
        ]
        text = " ".join(text_parts).casefold()
        if any(
            marker in text
            for marker in (
                "search",
                "搜索",
                "查找",
                "查询",
                "检索",
                "网页",
                "网络资料",
                "web search",
            )
        ):
            return "web search"
        return None

    async def _recover_catalog_lookup(
        self,
        parsed: ParsedReActResponse,
        result: ToolResult,
        current_snapshot: Mapping[str, tuple[Any, int]],
        registrations: Mapping[str, tuple[Any, int]],
        context: ExecutionContext,
        *,
        call_number: int,
        intent_hint: str,
    ) -> ToolResult | None:
        """Retry a failed/empty search catalog lookup using a capability hint."""

        if parsed.action is None or parsed.arguments is None:
            return None
        if self._canonical_action_name(parsed.action, current_snapshot) != self.catalog_tool.spec.name:
            return None
        action = parsed.arguments.get("action")
        if action not in {"resolve", "search"}:
            return None
        empty_success = result.ok and isinstance(result.data, Mapping) and not (
            result.data.get("spec") or result.data.get("candidates")
        )
        failed_lookup = (
            not result.ok
            and result.error is not None
            and result.error.code in {"EXECUTION_ERROR", "INVALID_ARGUMENTS"}
        )
        if not (empty_success or failed_lookup):
            return None
        fallback = ParsedReActResponse(
            raw="",
            action=self.catalog_tool.spec.name,
            arguments={
                "action": "resolve",
                "intent": intent_hint,
                "tool_name": None,
                "version": None,
                "limit": 5,
            },
        )
        recovered = await self._execute_action(
            fallback,
            current_snapshot,
            registrations,
            context,
            call_number=call_number,
        )
        return recovered if recovered.ok else None

    def _initial_conversation(
        self,
        request_messages: list[dict[str, Any]],
        *,
        use_history: bool,
        history_key: str,
    ) -> list[dict[str, Any]]:
        if use_history and self._profile_histories.get(history_key):
            prefix = [dict(item) for item in self._profile_histories[history_key]]
        else:
            prefix = self._configured_prompt_messages()
        return prefix + request_messages

    def _with_tool_instructions(
        self,
        conversation: list[dict[str, Any]],
        registrations: Mapping[str, tuple[Any, int]],
        loaded_tool_schemas: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        catalog_first: bool = False,
    ) -> list[dict[str, Any]]:
        if self.lazy_tools:
            registrations = {
                name: registration
                for name, registration in registrations.items()
                if name == self.catalog_tool.spec.name
            }
        all_tool_names = set(self.tools.snapshot())
        if self.repository is not None:
            all_tool_names.update(self.repository.active_tool_names())
        inventory = ", ".join(sorted(all_tool_names)) or "(none)"
        lines = [
            self.REACT_INSTRUCTIONS,
            "",
            "All registered tool names: " + inventory,
            "",
        ]
        if catalog_first:
            lines.extend((self.CATALOG_FIRST_REACT_INSTRUCTIONS, ""))
        lines.append("Available tools:")
        if not registrations:
            lines.append("(No tools are currently available; answer directly.)")
        else:
            for name, (tool, _) in registrations.items():
                schema = self._strict_react_schema(tool.spec.input_schema)
                lines.append(f"- {name}: {tool.spec.description}")
                lines.append(
                    "  Action Input schema: "
                    + json.dumps(schema, ensure_ascii=False, sort_keys=True)
                )
        resolved_schemas = {
            name: schema
            for name, schema in (loaded_tool_schemas or {}).items()
            if name not in registrations
        }
        if resolved_schemas:
            lines.extend(("", "Loaded tool input schemas:"))
            for name in sorted(resolved_schemas):
                schema = self._strict_react_schema(
                    dict(resolved_schemas[name])
                )
                lines.append(
                    f"- {name}: "
                    + json.dumps(schema, ensure_ascii=False, sort_keys=True)
                )
        instruction = {"role": "system", "content": "\n".join(lines)}
        return [instruction, *conversation]

    def _ensure_tool_loaded(self, name: str) -> None:
        """Load one implementation referenced by the persistent catalog."""

        if name in self.tools:
            return
        if self.repository is None:
            raise RuntimeError("lazy tool loading requires a ToolSpecRepository")
        stored = self.repository.get(name)
        if stored is None:
            return
        implementation_ref = stored.get("implementation_ref")
        if not isinstance(implementation_ref, str) or ":" not in implementation_ref:
            raise RuntimeError(
                f"tool '{name}' has no importable implementation reference"
            )
        module_name, qualname = implementation_ref.split(":", 1)
        if not module_name or not qualname:
            raise RuntimeError(f"tool '{name}' has an invalid implementation reference")
        module = importlib.import_module(module_name)
        target: Any = module
        for component in qualname.split("."):
            target = getattr(target, component)
        factory = getattr(module, "create_tool", None)
        tool = factory() if callable(factory) else target()
        if not isinstance(tool, BaseTool):
            raise TypeError("create_tool() must return a BaseTool instance")
        if (
            tool.spec.name != stored["tool_name"]
            or tool.spec.version != stored["version"]
            or tool.spec.schema_hash != stored["schema_hash"]
        ):
            raise RuntimeError(
                f"tool '{name}' implementation does not match its catalog schema"
            )
        self.tools.register(tool)

    def _load_catalog_result(
        self,
        result: ToolResult,
        loaded_order: list[str],
        context: ExecutionContext,
        loaded_tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Queue schemas returned by a successful catalog resolution."""

        if not result.ok or result.tool_name != self.catalog_tool.spec.name:
            return
        data = result.data if isinstance(result.data, Mapping) else {}
        raw_specs = data.get("specs")
        if not isinstance(raw_specs, list):
            raw_specs = []
        # Keep compatibility with older catalog responses that only have one
        # ``spec`` field.
        raw_spec = data.get("spec")
        if isinstance(raw_spec, Mapping) and raw_spec not in raw_specs:
            raw_specs.insert(0, raw_spec)
        for candidate in raw_specs:
            if not isinstance(candidate, Mapping):
                continue
            tool_name = candidate.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            if self.repository is not None:
                stored = self.repository.get(tool_name, candidate.get("version"))
                if stored is None:
                    continue
            if loaded_tool_schemas is not None:
                input_schema = candidate.get("input_schema")
                if isinstance(input_schema, Mapping):
                    loaded_tool_schemas[tool_name] = dict(input_schema)
            if tool_name not in loaded_order:
                loaded_order.append(tool_name)

    def _load_catalog_observation(
        self,
        observation: Mapping[str, Any],
        loaded_order: list[str],
        context: ExecutionContext,
        loaded_tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Load a catalog result encoded in a native-call observation."""

        content = observation.get("content")
        if not isinstance(content, str) or not content.startswith("Observation: "):
            return
        try:
            payload = json.loads(content[len("Observation: ") :])
            result = ToolResult.model_validate(payload, strict=True)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self._load_catalog_result(
            result, loaded_order, context, loaded_tool_schemas
        )

    @staticmethod
    def _complete_text_or_response(
        completion_llm: Any,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Any:
        """Call either the modern ``complete`` API or the legacy ``think`` API."""

        complete = getattr(completion_llm, "complete", None)
        if callable(complete):
            return complete(messages, **options)
        think = getattr(completion_llm, "think", None)
        if not callable(think):
            raise TypeError("llm must provide a complete() or think() method")
        think_options = {
            key: value
            for key, value in options.items()
            if key in {"temperature", "timeout"}
        }
        # ``think`` returns collected text for the project's LLM wrapper.
        try:
            parameters = inspect.signature(think).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "stream_response_bool" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            think_options["stream_response_bool"] = False
        return think(messages, **think_options)

    @staticmethod
    def _strict_react_schema(schema: dict[str, Any]) -> dict[str, Any]:
        # Reuse Agent's strict schema implementation so ReAct and native
        # function-calling expose exactly the same contract.
        from .agent import _strict_function_schema

        return _strict_function_schema(schema)

    @staticmethod
    def _response_message(response: Any) -> Any:
        choices = _field(response, "choices")
        if not choices:
            raise RuntimeError("LLM response contained no choices")
        message = _field(choices[0], "message")
        if message is None:
            raise RuntimeError("LLM response contained no message")
        return message

    async def _execute_action(
        self,
        parsed: ParsedReActResponse,
        current_snapshot: Mapping[str, tuple[Any, int]],
        registrations: Mapping[str, tuple[Any, int]],
        context: ExecutionContext,
        *,
        call_number: int,
        intent_hint: str | None = None,
        loaded_tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> ToolResult:
        assert parsed.action is not None
        parsed = self._normalize_catalog_action(parsed, context, intent_hint)
        action_name = self._canonical_action_name(parsed.action, current_snapshot)
        call_id = f"react-call-{call_number}"
        if action_name not in current_snapshot:
            # A model may call a known lazy tool directly after seeing the
            # inventory, without first resolving it through the catalog. Load
            # that repository entry on demand so it is not reported as an
            # unregistered tool merely because it was not in this snapshot.
            lazy_name = action_name
            if isinstance(lazy_name, str) and "__" in lazy_name:
                lazy_name = lazy_name.replace("__", ".")
            if self.lazy_tools and self.repository is not None:
                stored = (
                    self.repository.get(lazy_name)
                    if isinstance(lazy_name, str)
                    else None
                )
                if stored is not None:
                    self._ensure_tool_loaded(lazy_name)
                    refreshed = self.tools.snapshot()
                    if lazy_name in refreshed:
                        current_snapshot = refreshed
                        registrations = dict(registrations)
                        registrations[lazy_name] = refreshed[lazy_name]
                        action_name = lazy_name
                        if loaded_tool_schemas is not None:
                            loaded_tool_schemas[lazy_name] = refreshed[
                                lazy_name
                            ][0].spec.input_schema

        if action_name not in current_snapshot:
            return ToolResult(
                call_id=call_id,
                tool_name=_safe_tool_name(action_name),
                ok=False,
                error=ToolError(
                    code="UNKNOWN_TOOL", message=f"tool '{parsed.action}' is not registered"
                ),
            )
        if action_name not in registrations:
            return ToolResult(
                call_id=call_id,
                tool_name=_safe_tool_name(action_name),
                ok=False,
                error=ToolError(
                    code="TOOL_SCHEMA_REQUIRED",
                    message=(
                        f"tool '{_safe_tool_name(action_name)}' is registered and "
                        "usable; first call system.tool_catalog with action "
                        "'resolve' to load its Action Input schema"
                    ),
                ),
            )
        if parsed.error is not None:
            return ToolResult(
                call_id=call_id,
                tool_name=_safe_tool_name(action_name),
                ok=False,
                error=ToolError(code="INVALID_TOOL_CALL", message=parsed.error),
            )
        tool, generation = registrations[action_name]
        call = ToolCall(
            call_id=call_id,
            tool_name=action_name,
            schema_version=tool.spec.version,
            schema_hash=tool.spec.schema_hash,
            registry_generation=generation,
            arguments=parsed.arguments or {},
        )
        log_tool_call_started(call_number, (action_name,))
        tool_started_at = time.perf_counter()
        try:
            batch = await self.execute_tool_calls([call], context)
        finally:
            log_tool_call_completed(
                call_number,
                (action_name,),
                time.perf_counter() - tool_started_at,
            )
        return batch.results[0]

    def _normalize_catalog_action(
        self,
        parsed: ParsedReActResponse,
        context: ExecutionContext,
        intent_hint: str | None,
    ) -> ParsedReActResponse:
        """Correct an unmatched search capability intent before execution."""

        if (
            intent_hint is None
            or parsed.arguments is None
            or self.repository is None
            or self._canonical_action_name(
                parsed.action or "", self.tools.snapshot()
            )
            != self.catalog_tool.spec.name
            or parsed.arguments.get("action") not in {"resolve", "search"}
        ):
            return parsed
        intent = parsed.arguments.get("intent")
        if not isinstance(intent, str) or not intent.strip() or len(intent) > 500:
            return parsed
        records = self.repository.search(intent, 20)
        if records:
            return parsed
        arguments = dict(parsed.arguments)
        arguments.update(action="resolve", intent=intent_hint, limit=20)
        return ParsedReActResponse(
            raw=parsed.raw,
            thought=parsed.thought,
            action=parsed.action,
            arguments=arguments,
        )

    async def _execute_native_calls(
        self,
        native_calls: Sequence[Any],
        current_snapshot: Mapping[str, tuple[Any, int]],
        registrations: Mapping[str, tuple[Any, int]],
        context: ExecutionContext,
        *,
        round_number: int,
        intent_hint: str | None = None,
        loaded_tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        # Native calls are accepted as a compatibility fallback.  They use the
        # exact parser and execution path of Agent.run_with_tools.
        current_snapshot = dict(current_snapshot)
        registrations = dict(registrations)
        # Native providers use dotted names converted to ``__``.  Resolve and
        # load a repository-backed implementation before parsing the call so a
        # direct call to a discovered lazy tool is accepted.
        for native_call in native_calls:
            function = _field(native_call, "function")
            provider_name = _field(function, "name")
            if not isinstance(provider_name, str) or not provider_name.strip():
                continue
            canonical = self._canonical_action_name(provider_name, current_snapshot)
            if canonical in current_snapshot:
                continue
            lazy_name = canonical.replace("__", ".")
            if not self.lazy_tools or self.repository is None:
                continue
            if self.repository.get(lazy_name) is None:
                continue
            self._ensure_tool_loaded(lazy_name)
            current_snapshot = self.tools.snapshot()
            if lazy_name in current_snapshot:
                registrations[lazy_name] = current_snapshot[lazy_name]
                if loaded_tool_schemas is not None:
                    loaded_tool_schemas[lazy_name] = current_snapshot[lazy_name][
                        0
                    ].spec.input_schema
        name_map = {
            self._openai_tool_name(tool.spec.name): name
            for name, (tool, _) in registrations.items()
        }
        calls: list[ToolCall] = []
        positions: list[int] = []
        results: dict[int, ToolResult] = {}
        aliases = {
            self._openai_tool_name(tool.spec.name): name
            for name, (tool, _) in current_snapshot.items()
        }
        for position, native_call in enumerate(native_calls):
            call_id = _field(native_call, "id")
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"react-native-call-{position + 1}"
            function = _field(native_call, "function")
            provider_name = _field(function, "name", "unknown.tool")
            canonical = aliases.get(provider_name, provider_name)
            try:
                call = parse_openai_tool_calls(
                    [native_call], self.tools, name_map, registrations
                )[0]
            except (TypeError, ValueError) as exc:
                code = (
                    "TOOL_SCHEMA_REQUIRED"
                    if isinstance(canonical, str)
                    and canonical in current_snapshot
                    and canonical not in registrations
                    else "INVALID_TOOL_CALL"
                )
                message = (
                    f"tool '{_safe_tool_name(canonical)}' is registered and usable; "
                    "first call system.tool_catalog with action 'resolve' to load "
                    "its Action Input schema"
                    if code == "TOOL_SCHEMA_REQUIRED"
                    else _safe_tool_call_error(exc)
                )
                results[position] = ToolResult(
                    call_id=call_id,
                    tool_name=_safe_tool_name(canonical),
                    ok=False,
                    error=ToolError(code=code, message=message),
                )
                continue
            calls.append(call)
            positions.append(position)
        if calls:
            tool_names = tuple(call.tool_name for call in calls)
            log_tool_call_started(round_number, tool_names)
            tool_started_at = time.perf_counter()
            try:
                batch = await self.execute_tool_calls(calls, context)
            finally:
                log_tool_call_completed(
                    round_number,
                    tool_names,
                    time.perf_counter() - tool_started_at,
                )
            for position, result in zip(positions, batch.results, strict=True):
                results[position] = result
        if intent_hint is not None:
            for position, native_call in enumerate(native_calls):
                result = results[position]
                # Empty catalog searches are recoverable too. Decode only this
                # trusted provider payload; malformed payloads already have an
                # INVALID_TOOL_CALL result and are left untouched.
                function = _field(native_call, "function")
                provider_name = _field(function, "name")
                canonical = (
                    self._canonical_action_name(provider_name, current_snapshot)
                    if isinstance(provider_name, str)
                    else provider_name
                )
                if canonical != self.catalog_tool.spec.name:
                    continue
                raw_arguments = _field(function, "arguments")
                if isinstance(raw_arguments, str):
                    try:
                        raw_arguments = json.loads(raw_arguments)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                if not isinstance(raw_arguments, Mapping):
                    continue
                parsed = ParsedReActResponse(
                    raw="",
                    action=canonical,
                    arguments=dict(raw_arguments),
                )
                recovered = await self._recover_catalog_lookup(
                    parsed,
                    result,
                    current_snapshot,
                    registrations,
                    context,
                    call_number=round_number,
                    intent_hint=intent_hint,
                )
                if recovered is not None:
                    results[position] = recovered
        observations = []
        for position, native_call in enumerate(native_calls):
            call_id = _field(native_call, "id") or f"react-native-call-{position + 1}"
            observations.append(
                {
                    "role": "user",
                    "content": f"Observation: {_result_json(results[position])}",
                }
            )
        return observations

    def _canonical_action_name(
        self,
        action: str,
        snapshot: Mapping[str, tuple[Any, int]],
    ) -> str:
        if action in snapshot:
            return action
        for name in snapshot:
            if self._openai_tool_name(name) == action:
                return name
        return action


# Public spellings used by different versions of the original prototype.
ReactAgent = ReActAgent
ReAct = ReActAgent
React = ReActAgent
react = ReActAgent


def _run_sync(awaitable: Any) -> Any:
    """Run an awaitable from sync code, including when a loop is active."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    # ``asyncio.run`` cannot be nested.  A short-lived worker thread keeps the
    # synchronous API usable from notebooks and async test functions too.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


__all__ = [
    "ParsedReActResponse",
    "ReActAgent",
    "ReAct",
    "ReactAgent",
    "React",
    "parse_react_response",
    "react",
]
