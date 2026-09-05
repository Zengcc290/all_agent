from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from core import (
    ExecutionContext,
    ToolCall,
    ToolCatalogTool,
    ToolDiscoveryReport,
    ToolError,
    ToolExecutionManager,
    ToolRegistry,
    ToolResult,
    ToolSpecRepository,
    ToolLoop,
    parse_openai_tool_calls,
)
from core import discover_tools as discover_tool_modules
from core.registry import BaseTool

from .llm import LLM
from .providers import ProviderProfile, ProviderRegistry

MAX_RETRIES = 3
PROMPT_CACHE_KEY_VERSION = "pc-v1"


class Agent(ABC):
    """Agent shell with provider management and centralized tool execution."""

    # An omitted max_rounds should not allow a malfunctioning provider to keep
    # making requests forever.
    UNBOUNDED_ROUND_SAFETY_LIMIT = ToolLoop.DEFAULT_SAFETY_LIMIT

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
        # ``providers`` and ``global_model`` remain read-only compatibility
        # views for callers of the original prototype. New code should use
        # ``provider_registry`` and ``active_profile``.
        self.global_model: str | None = None
        self.role = ["user", "assistant", "system", "tool"]
        self.prompt: dict[str, str] = {}
        self.history: list[dict[str, Any]] = []
        self.max_retries = MAX_RETRIES

    def clear_history(self) -> None:
        self.history.clear()
        self._profile_histories.clear()

    @abstractmethod
    def run(self, query: str) -> str:
        raise NotImplementedError

    def register_tool(self, tool: BaseTool, *, replace: bool = False) -> None:
        self.tools.register(tool, replace=replace)
        if self.repository is not None:
            self.repository.save(tool.spec, replace=True)

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
        temperature: float = 0.7,
        timeout: float = 60,
        tool_names: list[str] | None = None,
        profile_name: str | None = None,
        provider_name: str | None = None,
        use_history: bool = True,
        defer_tool_loading: bool = False,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        enable_prompt_cache: bool = True,
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
        prompt_cache_key = self._validate_prompt_cache_key(prompt_cache_key)
        prompt_cache_retention = self._validate_prompt_cache_retention(
            prompt_cache_retention
        )

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
            safety_limit=self.UNBOUNDED_ROUND_SAFETY_LIMIT,
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
                completion_llm.complete,
                self._with_registered_tool_names(conversation),
                **completion_options,
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
                self._profile_histories[history_key] = [dict(item) for item in conversation]
                self.history = [dict(item) for item in conversation]
                return _field(message, "content") or ""

            calls: list[ToolCall] = []
            positions: list[int] = []
            results_by_position: dict[int, ToolResult] = {}
            all_aliases = {
                self._openai_tool_name(tool.spec.name): name
                for name, (tool, _) in current_snapshot.items()
            }
            for position, native_call in enumerate(native_calls):
                call_id = _field(native_call, "id")
                if (
                    not isinstance(call_id, str)
                    or not call_id.strip()
                    or len(call_id) > 128
                ):
                    raise RuntimeError(
                        "LLM returned a tool call without a usable call ID"
                    )
                function = _field(native_call, "function")
                provider_tool_name = _field(function, "name", "unknown.tool")
                canonical_name = (
                    all_aliases.get(provider_tool_name, provider_tool_name)
                    if isinstance(provider_tool_name, str)
                    else provider_tool_name
                )
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
        return [
            {
                "role": "system",
                "content": "All registered tool names: " + inventory,
            },
            *conversation,
        ]

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
        must not fragment the provider's prefix-cache routing.  Tool names are
        sorted and repository-backed names are included so lazy loading keeps
        the same key across rounds.
        """

        names = set(self.tools.snapshot())
        if self.repository is not None:
            names.update(self.repository.active_tool_names())
        material = {
            "version": PROMPT_CACHE_KEY_VERSION,
            "provider": provider_key,
            "model": model or "",
            "mode": mode,
            "configured_prompt": self._configured_prompt_messages(),
            "tool_names": sorted(names),
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

    @property
    def profiles(self) -> Mapping[str, ProviderProfile]:
        """Configured provider profiles (without resolved API keys)."""

        return self.provider_registry.profiles

    def set_active_profile(self, profile_name: str) -> None:
        """Select the default profile for subsequent requests."""

        profile = self.provider_registry.get(profile_name)
        self.active_profile = profile.name
        self.provider_registry.active_profile = profile.name

    def reload_provider_profiles(self) -> None:
        """Reload TOML and environment-backed credentials.

        Existing clients are discarded so URL/key changes take effect on the
        next request. Conversation history remains isolated per profile.
        """

        self.provider_registry.reload()
        self.active_profile = self.provider_registry.active_profile
        self._profile_clients.clear()

    def profile_info(self, profile_name: str | None = None) -> dict[str, Any]:
        selected = profile_name or self.active_profile
        return self.provider_registry.get(selected).public_info()

    def list_profiles(self) -> list[dict[str, Any]]:
        return [profile.public_info() for profile in self.provider_registry.profiles.values()]

    def add_provider(
        self,
        provider_name: str,
        api_key: str,
        base_url: str,
        default_model: str,
    ) -> None:
        """Deprecated compatibility shim; prefer a TOML profile.

        It deliberately does not probe ``/models``. The old API's network
        probe made valid gateways impossible to configure when that endpoint
        was unavailable.
        """

        self.provider_registry.register_ephemeral(
            provider_name,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
        )

    def detect_models(self, base_url: str, api_key: str) -> list[str]:
        """Deprecated compatibility helper for callers migrating to profiles."""

        from openai import OpenAI

        response = OpenAI(
            api_key=api_key, base_url=base_url, max_retries=self.max_retries
        ).models.list()
        raw_models = _field(response, "data")
        if not isinstance(raw_models, (list, tuple)):
            raise TypeError("provider model response data must be a list")
        return list(dict.fromkeys(str(_field(item, "id", item)).strip() for item in raw_models if str(_field(item, "id", item)).strip()))

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
        if selected_model is None and isinstance(self.global_model, str):
            possible_provider, separator, possible_model = self.global_model.partition(
                ":"
            )
            qualified = (
                separator
                and possible_provider in self.provider_registry.profiles
                and bool(possible_model)
            )
            if selected_provider is None and qualified:
                selected_provider = possible_provider
                selected_model = possible_model
            elif selected_provider is not None and qualified:
                if selected_provider == possible_provider:
                    selected_model = possible_model
            else:
                selected_model = self.global_model

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

    def set_user_prompt(self, prompt: str) -> None:
        self._set_prompt("user", prompt)

    def set_assistant_prompt(self, prompt: str) -> None:
        self._set_prompt("assistant", prompt)

    def set_tool_prompt(self, prompt: str) -> None:
        self._set_prompt("tool", prompt)

    def _set_prompt(self, role: str, prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.prompt[role] = prompt

    def set_global_model(self, model: str | None) -> None:
        """Deprecated compatibility setter; prefer ``set_active_profile`` and ``model=``."""
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string or None")
        self.global_model = model


# Preserve the original public name while offering the conventional class name.
agent = Agent


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


def _safe_tool_name(value: Any) -> str:
    name = value.strip() if isinstance(value, str) else "unknown.tool"
    return (name or "unknown.tool")[:200]


def _safe_tool_call_error(error: Exception) -> str:
    message = str(error) or type(error).__name__
    return message[:1000]


def _message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {
            key: value
            for key in ("role", "content", "tool_calls")
            if (value := getattr(message, key, None)) is not None
        }
    data.setdefault("role", "assistant")
    if "tool_calls" in data and data["tool_calls"] is not None:
        data["tool_calls"] = [_tool_call_dict(item) for item in data["tool_calls"]]
    return data


def _tool_call_dict(item: Any) -> dict[str, Any]:
    """Convert SDK tool-call objects to JSON-compatible assistant messages."""
    function = _field(item, "function")
    result: dict[str, Any] = {
        "id": _field(item, "id"),
        "type": _field(item, "type", "function"),
        "function": {
            "name": _field(function, "name"),
            "arguments": _field(function, "arguments", "{}"),
        },
    }
    return {key: value for key, value in result.items() if value is not None}


def _result_json(result: Any) -> str:
    """Serialize tool results for providers, including permissive Any fields."""
    try:
        payload = result.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        payload = result.model_dump()
    return json.dumps(payload, ensure_ascii=False, default=str)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)
