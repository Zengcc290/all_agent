from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core import ExecutionContext, ToolCall, ToolCatalogTool, ToolExecutionManager, ToolRegistry
from core.registry import BaseTool

from .llm import LLM

MAX_RETRIES = 3


class Agent(ABC):
    """Agent shell with provider management and centralized tool execution."""

    def __init__(self, name: str, *, llm: LLM | None = None) -> None:
        self.name = name
        self.llm = llm
        self.tools = ToolRegistry()
        self.catalog_tool = ToolCatalogTool(self.tools)
        self.tools.register(self.catalog_tool)
        self.execution_manager = ToolExecutionManager(self.tools)
        self.providers: dict[str, dict[str, Any]] = {}
        self.global_model: str | None = None
        self.role = ["user", "assistant", "system", "tool"]
        self.prompt: dict[str, str] = {}
        self.history: list[dict[str, str]] = []
        self.max_retries = MAX_RETRIES

    def clear_history(self) -> None:
        self.history.clear()

    @abstractmethod
    def run(self, query: str) -> str:
        raise NotImplementedError

    def register_tool(self, tool: BaseTool, *, replace: bool = False) -> None:
        self.tools.register(tool, replace=replace)

    async def execute_tool_calls(
        self,
        calls: list[ToolCall],
        context: ExecutionContext | None = None,
    ):
        return await self.execution_manager.execute_batch(calls, context)

    def set_system_prompt(self, prompt: str) -> None:
        self.prompt["system"] = prompt

    def set_user_prompt(self, prompt: str) -> None:
        self.prompt["user"] = prompt

    def set_assistant_prompt(self, prompt: str) -> None:
        self.prompt["assistant"] = prompt

    def set_tool_prompt(self, prompt: str) -> None:
        self.prompt["tool"] = prompt

    def set_global_model(self, model: str | None) -> None:
        self.global_model = model

    def detect_models(self, base_url: str, api_key: str) -> list[str]:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=self.max_retries)
        response = client.models.list()
        return [item.id if hasattr(item, "id") else str(item) for item in response.data]

    def add_provider(
        self,
        provider_name: str,
        api_key: str,
        base_url: str,
        default_model: str | None = None,
    ) -> None:
        if provider_name in self.providers:
            raise ValueError(f"Provider '{provider_name}' already exists.")
        supported_models = self.detect_models(base_url, api_key)
        if default_model is None:
            default_model = supported_models[0] if supported_models else None
        if default_model is not None and default_model not in supported_models:
            raise ValueError(f"Provider '{provider_name}' does not support model '{default_model}'.")
        self.providers[provider_name] = {
            "api_key": api_key,
            "base_url": base_url,
            "all_available_models": supported_models,
            "models": [],
            "default_model": default_model,
        }

    def add_model_to_provider(self, provider_name: str, model_name: str) -> None:
        provider = self._provider(provider_name)
        if model_name not in provider["all_available_models"]:
            raise ValueError(f"Provider '{provider_name}' does not support model '{model_name}'.")
        if model_name not in provider["models"]:
            provider["models"].append(model_name)

    def delete_model_from_provider(self, provider_name: str, model_name: str) -> None:
        provider = self._provider(provider_name)
        if model_name not in provider["models"]:
            raise ValueError(f"Model '{model_name}' not found for provider '{provider_name}'.")
        provider["models"].remove(model_name)

    def delete_provider(self, provider_name: str) -> None:
        self._provider(provider_name)
        del self.providers[provider_name]

    def change_all_provider_available_models(self) -> None:
        for name, provider in self.providers.items():
            models = self.detect_models(provider["base_url"], provider["api_key"])
            provider["all_available_models"] = models
            provider["models"] = [model for model in provider["models"] if model in models]
            if provider["default_model"] not in models:
                provider["default_model"] = models[0] if models else None

    # Backward-compatible spellings used by the original prototype.
    change_all_provider_avalible_models = change_all_provider_available_models

    def change_default_model(self, provider_name: str, default_model: str) -> None:
        provider = self._provider(provider_name)
        if default_model not in provider["all_available_models"]:
            raise ValueError(f"Provider '{provider_name}' does not support model '{default_model}'.")
        provider["default_model"] = default_model

    def get_single_provider_models(self, provider_name: str) -> list[str]:
        return list(self._provider(provider_name)["models"])

    def get_all_providers(self) -> list[str]:
        return list(self.providers)

    def get_single_provider_available_models(self, provider_name: str) -> list[str]:
        return list(self._provider(provider_name)["all_available_models"])

    get_single_provider_avalible_models = get_single_provider_available_models

    def get_all_available_models(self) -> list[str]:
        return [f"{name}:{model}" for name, provider in self.providers.items() for model in provider["all_available_models"]]

    get_all_avalible_models = get_all_available_models

    def list_all_available_models(self) -> list[str]:
        models = self.get_all_available_models()
        for model in models:
            print(model)
        return models

    list_all_avalible_models = list_all_available_models

    def list_single_provider_models(self, provider_name: str) -> list[str]:
        models = self.get_single_provider_models(provider_name)
        for model in models:
            print(model)
        return models

    def list_single_provider_info(self, provider_name: str) -> dict[str, Any]:
        provider = self._provider(provider_name)
        info = {"provider": provider_name, "base_url": provider["base_url"], "default_model": provider["default_model"]}
        print(info)
        return info

    def list_all_providers_info(self) -> list[dict[str, Any]]:
        return [self.list_single_provider_info(name) for name in self.providers]

    def _provider(self, provider_name: str) -> dict[str, Any]:
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found.")
        return self.providers[provider_name]


# Preserve the original public name while offering the conventional class name.
agent = Agent
