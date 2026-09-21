from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    HISTORY_MAX_MESSAGES,
    OBSERVATION_COMPRESS_THRESHOLD,
    OBSERVATION_PREVIEW_CHARS,
    OBSERVATION_STUB_PREFIX,
    PROMPT_CACHE_KEY_VERSION,
)
from core import (
    ExecutionContext,
    ToolCall,
    ToolCatalogTool,
    ToolDiscoveryReport,
    ToolError,
    ToolExecutionManager,
    ToolLoop,
    ToolRegistry,
    ToolResult,
    ToolSpecRepository,
    parse_openai_tool_calls,
)
from core import discover_tools as discover_tool_modules
from core.activity_log import log_model_completed, log_model_first_chunk
from core.registry import BaseTool

from .llm import LLM, EchoMode
from .message_utils import (
    field as _field,
)
from .message_utils import (
    message_dict as _message_dict,
)
from .message_utils import (
    result_json as _result_json,
)
from .message_utils import (
    safe_tool_call_error as _safe_tool_call_error,
)
from .message_utils import (
    safe_tool_name as _safe_tool_name,
)
from .providers import ProviderRegistry


class Agent(ABC):
    """Agent shell with provider management and centralized tool execution."""

    # ``None`` means "use the active profile's tool_mode". Only a profile that
    # explicitly declares ``tool_mode = "none"`` disables tool support.
    TOOL_MODE_PROTOCOLS: Mapping[str, str | None] = {
        "native_strict": "native",
        "native_loose": "native",
        "text_react": "react",
        "none": None,
    }

    def default_tool_protocol(self) -> str | None:
        """Resolve the protocol declared by the active provider profile.

        Profiles without a ``tool_mode`` key and legacy setups without any
        profile fall back to the native function-calling protocol. The mapping
        covers every value accepted by ProviderProfile validation.
        """

        profile = self.provider_registry.profiles.get(self.active_profile)
        if profile is None:
            return "native"
        protocol = self.TOOL_MODE_PROTOCOLS.get(profile.tool_mode, "native")
        return protocol

    def _save_history(self, history_key: str, conversation: list[dict[str, Any]]) -> None:
        """Persist the conversation with oversized tool payloads compressed.

        The stored history replaces the configured prompt on the next turn, so
        the leading ``system`` block must survive trimming; only old turns are
        dropped once the message count exceeds ``HISTORY_MAX_MESSAGES``.
        """

        compressed = compress_saved_history(conversation)
        compressed = trim_saved_history(compressed)
        self._profile_histories[history_key] = compressed
        self.history = [dict(item) for item in compressed]

    @staticmethod
    def _dispatch_model_call(
        completion_llm: Any,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        round_number: int,
        echo_mode: EchoMode | None = None,
    ) -> Any:
        """Run one model request, preferring a streaming transport.

        Non-streaming requests stall the tool loop: gateways send no bytes
        until the full completion exists, so a long generation trips the read
        timeout and the OpenAI SDK silently retries the identical request.
        Real ``LLM`` clients therefore stream and assemble; injected test
        doubles and legacy ``think``-style wrappers keep their existing
        transports. The returned response shape is identical for both paths.
        """

        model_started_at = time.perf_counter()
        stream_method = getattr(completion_llm, "complete_streaming", None)
        if callable(stream_method) and options.get("stream", False) is False:
            streaming_options = {
                key: value
                for key, value in options.items()
                if key not in {"stream"}
            }
            first_chunk_at: list[float] = []

            def _log_first_chunk() -> None:
                first_chunk_at.append(time.perf_counter())

            response = stream_method(
                messages,
                on_first_chunk=_log_first_chunk,
                echo_mode=echo_mode,
                **streaming_options,
            )
            if first_chunk_at:
                log_model_first_chunk(
                    round_number, first_chunk_at[0] - model_started_at
                )
            log_model_completed(round_number, time.perf_counter() - model_started_at)
            return response
        complete = getattr(completion_llm, "complete", None)
        if callable(complete):
            response = complete(messages, **options)
            log_model_completed(round_number, time.perf_counter() - model_started_at)
            return response
        think = getattr(completion_llm, "think", None)
        if not callable(think):
            raise TypeError("llm must provide a complete() or complete_streaming() method")
        think_options = {
            key: value
            for key, value in options.items()
            if key in {"temperature", "timeout", "prompt_cache_key", "prompt_cache_retention"}
        }
        try:
            parameters = inspect.signature(think).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_var_kw = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "stream_response_bool" in parameters or accepts_var_kw:
            think_options["stream_response_bool"] = False
        if not accepts_var_kw:
            for key in ("prompt_cache_key", "prompt_cache_retention"):
                if key not in parameters:
                    think_options.pop(key, None)
        response = think(messages, **think_options)
        log_model_completed(round_number, time.perf_counter() - model_started_at)
        return response

    async def run_auto(
        self,
        messages: str | list[dict[str, Any]],
        context: ExecutionContext | None = None,
        **kwargs: Any,
    ) -> str:
        """Run one conversation with the protocol declared by ``tool_mode``.

        ``native_strict``/``native_loose`` dispatch to :meth:`run_with_tools`;
        ``text_react`` dispatches to :meth:`run_with_react` (ReActAgent only);
        ``none`` rejects the request. A plain-string query is wrapped for the
        native protocol so callers can stay provider-agnostic.
        """

        protocol = self.default_tool_protocol()
        if protocol is None:
            raise ValueError(
                "the active provider profile disables tool use (tool_mode = 'none')"
            )
        if protocol == "react":
            react_runner = getattr(self, "run_with_react", None)
            if not callable(react_runner):
                raise TypeError("tool_mode 'text_react' requires a ReActAgent instance")
            return await react_runner(messages, context, **kwargs)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return await self.run_with_tools(messages, context, **kwargs)

    def __init__(
        self,
        name: str,
        *,
        llm: LLM | None = None,
        provider_config: str | None = None,
        provider_registry: ProviderRegistry | None = None,
        repository: ToolSpecRepository | None = None,
        auto_discover_tools: bool = True,
        tool_package: str | ModuleType = "tool",
        discovery_strict: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if repository is not None and not isinstance(repository, ToolSpecRepository):
            raise TypeError("repository must be a ToolSpecRepository or None")
        if not isinstance(auto_discover_tools, bool):
            raise TypeError("auto_discover_tools must be a boolean")
        if not isinstance(discovery_strict, bool):
            raise TypeError("discovery_strict must be a boolean")
        if not isinstance(tool_package, ModuleType) and not (
            isinstance(tool_package, str) and tool_package.strip()
        ):
            raise TypeError(
                "tool_package must be a non-empty module name or ModuleType"
            )
        self.name = name
        self.llm = llm
        if provider_registry is not None and not isinstance(
            provider_registry, ProviderRegistry
        ):
            raise TypeError("provider_registry must be a ProviderRegistry or None")
        self.provider_registry = (
            provider_registry
            if provider_registry is not None
            else ProviderRegistry(provider_config)
        )
        self.active_profile = self.provider_registry.active_profile
        self._profile_clients: dict[tuple[str, str], LLM] = {}
        self._profile_histories: dict[str, list[dict[str, Any]]] = {}
        self.repository = repository
        self.tool_package = tool_package
        self.tools = ToolRegistry()
        self.catalog_tool = ToolCatalogTool(self.tools, repository)
        self.tools.register(self.catalog_tool)
        if self.repository is not None:
            self.repository.save(self.catalog_tool.spec, replace=True)
        self.tool_discovery_report: ToolDiscoveryReport | None = None
        if auto_discover_tools:
            self.discover_tools(strict=discovery_strict)
        self.execution_manager = ToolExecutionManager(self.tools)
        # ``providers`` remains a read-only compatibility
        # view for callers of the original prototype. New code should use
        # ``provider_registry`` and ``active_profile``.
        self.role = ["user", "assistant", "system", "tool"]
        self.prompt: dict[str, str] = {}
        self.history: list[dict[str, Any]] = []
        self.max_retries = DEFAULT_MAX_RETRIES
        # Hot-reload cache state. ``_frozen_manifest`` records the
        # prompt-visible fingerprint (schema hash plus description) of the
        # tool set living in the stable prompt prefix; it is captured on the
        # first request and never rebuilt silently afterwards.
        # ``_hot_tools`` buffers tools registered after that capture so they
        # are rendered at the end of the prompt instead of rewriting the
        # cached prefix. ``cache_epoch`` only bumps when the frozen prefix
        # itself must change.
        self._frozen_manifest: dict[str, str] | None = None
        self._hot_tools: dict[str, str] = {}
        self.cache_epoch: int = 0

    @abstractmethod
    def run(self, query: str) -> str:
        raise NotImplementedError

    def register_tool(self, tool: BaseTool, *, replace: bool = False) -> None:
        self.tools.register(tool, replace=replace)
        if self.repository is not None:
            self.repository.save(tool.spec, replace=True)

    def register_hot_tool(self, tool: BaseTool, *, replace: bool = False) -> None:
        """Register a tool for hot-reload rendering instead of prefix edits.

        The tool becomes fully executable immediately; its schema is only
        advertised through the trailing hot-zone block. When no frozen prompt
        exists yet the tool simply becomes part of the first frozen prefix.
        """

        self.register_tool(tool, replace=replace)
        name = tool.spec.name
        fingerprint = self._prompt_fingerprint(tool.spec)
        if self._frozen_manifest is None:
            return
        if self._frozen_manifest.get(name) == fingerprint:
            self._hot_tools.pop(name, None)
            return
        self._hot_tools[name] = fingerprint

    def unregister_tool(self, name: str) -> None:
        """Remove a tool from the runtime registry and the prompt zones.

        Removing a frozen-zone tool invalidates the cached prefix and bumps
        ``cache_epoch``; removing a hot-zone tool is free. When the agent has
        not issued a request yet, the tool is simply unregistered.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        self.tools.unregister(name)
        if self._frozen_manifest is None:
            self._hot_tools.pop(name, None)
            return
        if name in self._frozen_manifest:
            self._frozen_manifest.pop(name)
            self.cache_epoch += 1
            return
        self._hot_tools.pop(name, None)

    @staticmethod
    def _prompt_fingerprint(spec: Any) -> str:
        """Fingerprint what a tool contributes to the prompt text.

        ``schema_hash`` alone misses description-only edits, but the
        description is rendered verbatim into the schema block. Bundle both
        so a re-registration is a no-op only when the rendered prompt would
        be byte-identical.
        """

        return f"{spec.schema_hash}#{spec.description}"

    def _sync_frozen_manifest(self) -> None:
        """Capture the frozen manifest once, before the first request.

        The first request defines the byte-stable prompt prefix: every tool
        registered at that moment is advertised from the frozen system
        message. Later registrations go to the hot zone instead of touching
        that prefix. This deliberately does not rescan the tool package;
        callers decide when to run :meth:`discover_tools`.
        """

        if self._frozen_manifest is not None:
            return
        names = set(self.tools.snapshot())
        if self.repository is not None:
            names.update(self.repository.active_tool_names())
        manifest: dict[str, str] = {}
        for name in names:
            tool = self.tools.maybe_get(name)
            if tool is not None:
                manifest[name] = self._prompt_fingerprint(tool.spec)
            elif self.repository is not None:
                stored = self.repository.get(name)
                if stored is not None and isinstance(stored.get("schema_hash"), str):
                    manifest[name] = (
                        f"{stored['schema_hash']}#{stored.get('description', '')}"
                    )
        self._frozen_manifest = manifest

    def discover_tools(
        self,
        *,
        package: str | ModuleType | None = None,
        replace: bool = False,
        strict: bool = False,
        reload_modules: bool = False,
    ) -> ToolDiscoveryReport:
        """Scan a trusted package and synchronize discovered tools."""
        selected_package = self.tool_package if package is None else package
        report = discover_tool_modules(
            self.tools,
            package=selected_package,
            repository=self.repository,
            replace=replace,
            strict=strict,
            reload_modules=reload_modules,
        )
        self.tool_discovery_report = report
        return report

    def is_tool_registered(
        self,
        name: str,
        *,
        version: str | None = None,
        schema_hash: str | None = None,
    ) -> bool:
        return self.tools.is_registered(
            name,
            version=version,
            schema_hash=schema_hash,
        )

    def tool_registration_status(self, name: str) -> dict[str, Any]:
        return self.tools.registration_status(name)

    async def execute_tool_calls(
        self,
        calls: list[ToolCall],
        context: ExecutionContext | None = None,
    ):
        return await self.execution_manager.execute_batch(calls, context)

    def tool_definitions(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Build OpenAI-compatible definitions from the registered schemas."""
        if names is not None and (
            not isinstance(names, (list, tuple))
            or not all(isinstance(name, str) and name for name in names)
        ):
            raise TypeError("names must be a list of non-empty strings")
        selected = set(names) if names is not None else None
        if selected is not None:
            registered = {spec.name for spec in self.tools.specs()}
            unknown = selected - registered
            if unknown:
                raise ValueError("unknown tool name(s): " + ", ".join(sorted(unknown)))
        ordered_names = (
            [spec.name for spec in self.tools.specs()] if names is None else list(names)
        )
        registrations = self.tools.snapshot(ordered_names)
        definitions, _ = self._definitions_for_registrations(registrations)
        return definitions

    def _definitions_for_registrations(
        self,
        registrations: Mapping[str, tuple[BaseTool, int]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        definitions = []
        aliases: dict[str, str] = {}
        # A canonical lexical order keeps the ``tools`` request field byte-for-
        # byte stable across runs and across agents that discovered modules in
        # a different filesystem order.  Stable ordering is required for
        # OpenAI prefix/KV cache reuse because tool definitions are part of the
        # cached prompt prefix.
        for name in sorted(
            registrations,
            key=lambda item: (item != self.catalog_tool.spec.name, item),
        ):
            tool, _ = registrations[name]
            spec = tool.spec
            alias = self._openai_tool_name(spec.name)
            if len(alias) > 64:
                raise ValueError(
                    f"tool name '{spec.name}' exceeds the provider's 64-character limit"
                )
            previous = aliases.get(alias)
            if previous is not None and previous != spec.name:
                raise ValueError(
                    f"tool names '{previous}' and '{spec.name}' map to the same provider alias"
                )
            aliases[alias] = spec.name
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": alias,
                        "description": spec.model_description,
                        "parameters": _strict_function_schema(spec.input_schema),
                        "strict": True,
                    },
                }
            )
        return definitions, aliases

    @staticmethod
    def _openai_tool_name(name: str) -> str:
        """Use a provider-safe function name while retaining namespaced tool IDs."""
        return name.replace(".", "__")

    async def run_with_tools(
        self,
        messages: list[dict[str, Any]],
        context: ExecutionContext | None = None,
        *,
        max_rounds: int | None = None,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        tool_names: list[str] | None = None,
        profile_name: str | None = None,
        provider_name: str | None = None,
        use_history: bool = True,
        defer_tool_loading: bool = False,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        enable_prompt_cache: bool = True,
        stream_echo: bool = False,
    ) -> str:
        """Run the model/tool protocol until the model emits a final answer."""
        if max_rounds is not None and (
            isinstance(max_rounds, bool)
            or not isinstance(max_rounds, int)
            or max_rounds < 1
        ):
            raise ValueError("max_rounds must be None or a positive integer")
        if not isinstance(messages, (list, tuple)) or not all(
            isinstance(message, Mapping) for message in messages
        ):
            raise TypeError("messages must be a list of mapping objects")
        if context is None:
            context = ExecutionContext()
        elif not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext instance")
        if not isinstance(use_history, bool) or not isinstance(
            defer_tool_loading, bool
        ):
            raise TypeError("use_history and defer_tool_loading must be booleans")
        if not isinstance(enable_prompt_cache, bool):
            raise TypeError("enable_prompt_cache must be a boolean")
        if not isinstance(stream_echo, bool):
            raise TypeError("stream_echo must be a boolean")
        prompt_cache_key = self._validate_prompt_cache_key(prompt_cache_key)
        prompt_cache_retention = self._validate_prompt_cache_retention(
            prompt_cache_retention
        )
        echo_mode: EchoMode | None = "content" if stream_echo else None

        completion_llm, selected_model, history_key = self._completion_target(
            profile_name, model, provider_name=provider_name
        )
        prefix = (
            [dict(item) for item in self._profile_histories.get(history_key, [])]
            if use_history
            else []
        )
        if not prefix:
            prefix = self._configured_prompt_messages()
        conversation = prefix + [dict(message) for message in messages]

        initial_snapshot = self.tools.snapshot()
        # Capture the frozen prefix before the routing key is derived: the
        # first request freezes both the advertised inventory and the epoch,
        # so hot re-registrations never change the native routing key.
        self._sync_frozen_manifest()
        # Permission metadata is retained for compatibility and audit output,
        # but it is not an authorization filter in this deployment.
        visible_order = list(initial_snapshot)
        if tool_names is not None:
            if not isinstance(tool_names, (list, tuple)) or not all(
                isinstance(name, str) and name for name in tool_names
            ):
                raise TypeError("tool_names must be a list of non-empty strings")
            requested_order = list(dict.fromkeys(tool_names))
            unknown = set(requested_order) - set(initial_snapshot)
            if unknown:
                raise ValueError("unknown tool name(s): " + ", ".join(sorted(unknown)))
            requested_names = set(requested_order)
            loaded_order = requested_order
        elif defer_tool_loading:
            requested_names = None
            loaded_order = [self.catalog_tool.spec.name]
        else:
            requested_names = None
            loaded_order = visible_order

        cache_key = None
        if enable_prompt_cache:
            cache_key = prompt_cache_key or self._default_prompt_cache_key(
                history_key,
                selected_model or getattr(completion_llm, "model", None),
                mode="native",
            )

        round_loop = ToolLoop(
            max_rounds,
            safety_limit=ToolLoop.DEFAULT_SAFETY_LIMIT,
        )
        for round_number in round_loop.rounds():
            current_snapshot = self.tools.snapshot()
            registrations = {
                name: current_snapshot[name]
                for name in loaded_order
                if name in current_snapshot
            }
            tool_definitions, name_map = self._definitions_for_registrations(
                registrations
            )
            completion_options: dict[str, Any] = {
                "model": selected_model,
                "temperature": temperature,
                "timeout": timeout,
                "stream": False,
            }
            if cache_key is not None:
                completion_options["prompt_cache_key"] = cache_key
            if prompt_cache_retention is not None:
                completion_options["prompt_cache_retention"] = prompt_cache_retention
            if tool_definitions:
                completion_options["tools"] = tool_definitions
            response = await asyncio.to_thread(
                self._dispatch_model_call,
                completion_llm,
                self._with_registered_tool_names(conversation),
                completion_options,
                round_number=round_number,
                echo_mode=echo_mode,
            )
            choices = _field(response, "choices")
            if not choices:
                raise RuntimeError("LLM response contained no choices")
            message = _field(choices[0], "message")
            if message is None:
                raise RuntimeError("LLM response contained no message")
            native_calls = _field(message, "tool_calls") or []
            assistant_message = _message_dict(message)
            conversation.append(assistant_message)
            if not native_calls:
                self._save_history(history_key, conversation)
                return _field(message, "content") or ""

            calls: list[ToolCall] = []
            positions: list[int] = []
            results_by_position: dict[int, ToolResult] = {}
            all_aliases = {
                self._openai_tool_name(tool.spec.name): name
                for name, (tool, _) in current_snapshot.items()
            }
            for position, native_call in enumerate(native_calls):
                function = _field(native_call, "function")
                provider_tool_name = _field(function, "name", "unknown.tool")
                canonical_name = (
                    all_aliases.get(provider_tool_name, provider_tool_name)
                    if isinstance(provider_tool_name, str)
                    else provider_tool_name
                )
                call_id = _field(native_call, "id")
                if not isinstance(call_id, str) or not call_id.strip():
                    # parse_openai_tool_calls generates a stable fallback for
                    # gateways that omit call IDs.
                    call_id = f"native-call-{position + 1}"
                try:
                    call = parse_openai_tool_calls(
                        [native_call],
                        self.tools,
                        name_map,
                        registrations,
                    )[0]
                except (TypeError, ValueError) as exc:
                    code = (
                        "TOOL_NOT_EXPOSED"
                        if isinstance(canonical_name, str)
                        and canonical_name in current_snapshot
                        and canonical_name not in registrations
                        else "INVALID_TOOL_CALL"
                    )
                    results_by_position[position] = ToolResult(
                        call_id=call_id,
                        tool_name=_safe_tool_name(canonical_name),
                        ok=False,
                        error=ToolError(
                            code=code,
                            message=_safe_tool_call_error(exc),
                        ),
                    )
                    continue
                calls.append(call)
                positions.append(position)

            if calls:
                batch = await self.execute_tool_calls(calls, context)
                for position, result in zip(positions, batch.results, strict=True):
                    results_by_position[position] = result

            for position, native_call in enumerate(native_calls):
                result = results_by_position[position]
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": _field(native_call, "id"),
                        "content": _result_json(result),
                    }
                )
                if defer_tool_loading and requested_names is None:
                    self._load_catalog_result(result, loaded_order, context)
        raise RuntimeError("maximum tool-call rounds exceeded")

    def _load_catalog_result(
        self,
        result: ToolResult,
        loaded_order: list[str],
        context: ExecutionContext,
    ) -> None:
        if not result.ok or result.tool_name != self.catalog_tool.spec.name:
            return
        data = result.data if isinstance(result.data, Mapping) else {}
        raw_specs = data.get("specs")
        if not isinstance(raw_specs, list):
            raw_specs = []
        raw_spec = data.get("spec")
        if isinstance(raw_spec, Mapping) and raw_spec not in raw_specs:
            raw_specs.insert(0, raw_spec)
        for candidate in raw_specs:
            if not isinstance(candidate, Mapping):
                continue
            tool_name = candidate.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            registration = self.tools.maybe_resolve(tool_name)
            if registration is None:
                continue
            if tool_name not in loaded_order:
                loaded_order.append(tool_name)

    def _configured_prompt_messages(self) -> list[dict[str, Any]]:
        messages = [
            {"role": role, "content": self.prompt[role]}
            for role in ("system", "user", "assistant")
            if role in self.prompt
        ]
        if "tool" in self.prompt:
            messages.append(
                {
                    "role": "system",
                    "content": f"Tool-use instructions: {self.prompt['tool']}",
                }
            )
        return messages

    def _with_registered_tool_names(
        self, conversation: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add all known tool names to every model request.

        The inventory intentionally includes repository-only tools used by
        lazy loading, even when their full schemas are not yet supplied in the
        provider's tool definitions.
        """

        names = set(self.tools.snapshot())
        if self.repository is not None:
            names.update(self.repository.active_tool_names())
        inventory = ", ".join(sorted(names)) or "(none)"
        prefix: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "All registered tool names: " + inventory,
            }
        ]
        return [*prefix, *conversation]

    @staticmethod
    def _validate_prompt_cache_key(value: str | None) -> str | None:
        """Validate an OpenAI prompt-cache routing key.

        OpenAI currently limits this key to 64 characters.  Keeping the
        validation in the agent as well as :class:`LLM` makes injected/fake
        clients observe the same contract as the real SDK client.
        """

        if value is None:
            return None
        if not isinstance(value, str) or not value.strip() or len(value) > 64:
            raise ValueError(
                "prompt_cache_key must be a non-empty string of at most 64 characters"
            )
        return value.strip()

    @staticmethod
    def _validate_prompt_cache_retention(value: str | None) -> str | None:
        if value is not None and value not in {"in_memory", "24h"}:
            raise ValueError(
                "prompt_cache_retention must be 'in_memory', '24h', or None"
            )
        return value

    def _default_prompt_cache_key(
        self,
        provider_key: str,
        model: str | None,
        *,
        mode: str,
    ) -> str:
        """Build a deterministic key for the stable prompt prefix.

        The key intentionally excludes user text, tool results, and loaded
        catalog schemas.  Those values belong after the reusable prefix and
        must not fragment the provider's prefix-cache routing.  Hot-reloaded
        tool names are included through the hot-zone roster because their
        rendered block is part of the request tail, but hot reloads must not
        change the routing key: only ``cache_epoch`` (frozen-zone structural
        changes) starts a new cache namespace.
        """

        names = set(self.tools.snapshot())
        if self.repository is not None:
            names.update(self.repository.active_tool_names())
        # Hot-reloaded tools must not change the routing key: their schemas
        # are rendered in the request tail, outside the cached prefix. The
        # frozen manifest only changes together with ``cache_epoch``.
        if self._frozen_manifest is not None:
            names = set(self._frozen_manifest)
        material = {
            "version": PROMPT_CACHE_KEY_VERSION,
            "provider": provider_key,
            "model": model or "",
            "mode": mode,
            "configured_prompt": self._configured_prompt_messages(),
            "tool_names": sorted(names),
            "cache_epoch": self.cache_epoch,
        }
        # ReAct keeps its protocol instructions in class constants.  Include
        # them in the digest when present so a prompt-template deployment
        # change naturally starts a new cache namespace.
        if mode == "react":
            material["react_instructions"] = [
                getattr(self, "REACT_INSTRUCTIONS", ""),
                getattr(self, "CATALOG_FIRST_REACT_INSTRUCTIONS", ""),
            ]
        encoded = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:48]
        return f"{PROMPT_CACHE_KEY_VERSION}-{digest}"

    def _completion_target(
        self,
        profile_name: str | None,
        model: str | None,
        *,
        provider_name: str | None = None,
    ) -> tuple[Any, str | None, str]:
        if profile_name is not None and (
            not isinstance(profile_name, str) or not profile_name.strip()
        ):
            raise ValueError("profile_name must be a non-empty string or None")
        if provider_name is not None:
            if profile_name is not None and profile_name != provider_name:
                raise ValueError("profile_name and provider_name must match when both are set")
            profile_name = provider_name
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string or None")

        selected_provider = profile_name
        selected_model = model

        if selected_provider is None and self.llm is not None:
            return self.llm, selected_model, "__injected__"
        selected_provider = selected_provider or self.active_profile
        profile = self.provider_registry.get(selected_provider)
        selected_model = selected_model or profile.default_model
        if selected_model not in profile.models:
            raise ValueError(
                f"Provider profile '{selected_provider}' does not support model "
                f"'{selected_model}'."
            )
        cache_key = (selected_provider, selected_model)
        client = self._profile_clients.get(cache_key)
        if client is None:
            api_key = self.provider_registry.resolve_api_key(selected_provider)
            client = LLM(
                api_key=api_key,
                base_url=profile.base_url,
                model=selected_model,
                max_retries=self.max_retries,
            )
            self._profile_clients[cache_key] = client
        return client, selected_model, selected_provider

    def set_system_prompt(self, prompt: str) -> None:
        self._set_prompt("system", prompt)

    def _set_prompt(self, role: str, prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.prompt[role] = prompt


def _strict_function_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic object schema to OpenAI strict function form."""
    normalized = copy.deepcopy(schema)
    if normalized.get("type") != "object" or not isinstance(
        normalized.get("properties"), dict
    ):
        raise ValueError("function input schemas must have an object root")

    def normalize(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                normalize(item)
            return
        if not isinstance(value, dict):
            return
        value.pop("default", None)
        properties = value.get("properties")
        if value.get("type") == "object":
            additional = value.get("additionalProperties")
            if additional is not None and additional is not False:
                raise ValueError(
                    "strict function schemas cannot contain arbitrary object keys"
                )
            value["additionalProperties"] = False
            if isinstance(properties, dict):
                value["required"] = list(properties)
        for child in value.values():
            normalize(child)

    normalize(normalized)
    return normalized


def compress_saved_history(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of ``conversation`` with oversized tool payloads stubbed.

    Read tools legitimately return tens of kilobytes per call. Persisting the
    raw payload pins it into every future request of the same profile until
    providers stall or exceed their context window; the stub keeps the tool
    name, the original size, a short preview and a re-query hint instead.
    """

    compressed: list[dict[str, Any]] = []
    for message in conversation:
        content = message.get("content")
        if (
            message.get("role") == "user"
            and isinstance(content, str)
            and content.startswith("Observation: ")
            and len(content) > OBSERVATION_COMPRESS_THRESHOLD
            and not content.startswith(f"Observation: {OBSERVATION_STUB_PREFIX}")
        ):
            payload = content[len("Observation: ") :]
            compressed.append({**message, "content": "Observation: " + _observation_stub(payload)})
        elif (
            message.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > OBSERVATION_COMPRESS_THRESHOLD
        ):
            compressed.append({**message, "content": _observation_stub(content)})
        else:
            compressed.append(dict(message))
    return compressed


def trim_saved_history(
    conversation: list[dict[str, Any]],
    *,
    max_messages: int = HISTORY_MAX_MESSAGES,
) -> list[dict[str, Any]]:
    """Bound the persisted history while keeping the leading ``system`` block.

    Without this cap every turn resends the entire conversation, so token cost
    grows without limit for a long-lived agent. Older turns are dropped from the
    front; an ``Observation`` orphaned by the cut is dropped too, because it
    would otherwise appear without the action that produced it.
    """

    if max_messages < 1 or len(conversation) <= max_messages:
        return conversation

    leading: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for message in conversation:
        if not rest and message.get("role") == "system":
            leading.append(dict(message))
            continue
        rest.append(dict(message))
    if len(rest) <= max_messages:
        return conversation

    kept = rest[-max_messages:]
    while kept and str(kept[0].get("content", "")).startswith("Observation: "):
        kept = kept[1:]
    return leading + kept


def _observation_stub(payload: str) -> str:
    preview = payload[:OBSERVATION_PREVIEW_CHARS]
    return (
        f"{OBSERVATION_STUB_PREFIX} | 原始大小: {len(payload):,} 字符 | "
        f"预览: {preview}... | 如需完整数据请让 AI 重新调用同一工具查询。]"
    )
