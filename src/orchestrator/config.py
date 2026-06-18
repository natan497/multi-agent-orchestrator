"""Environment-driven configuration and the provider factory.

This is the only place that maps env -> config and constructs concrete providers.
Groq is wired by default; the Anthropic provider is constructed only when
``provider="anthropic"`` is selected and ``ANTHROPIC_API_KEY`` is present. Agent code
never imports a concrete provider — it receives one built here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from orchestrator.models import OrchestratorConfig
from providers.base import LLMProvider, ProviderError


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_config(*, dotenv: bool = True) -> OrchestratorConfig:
    """Build an :class:`OrchestratorConfig` from environment variables.

    Reads ``.env`` if present (unless ``dotenv=False``). ``anthropic_enabled`` reflects
    whether a key is available; it does not, by itself, switch the active provider.
    """
    if dotenv:
        load_dotenv()
    return OrchestratorConfig(
        provider=os.getenv("LLM_PROVIDER", "groq").strip().lower() or "groq",
        planner_model=os.getenv("PLANNER_MODEL", "openai/gpt-oss-120b"),
        executor_model=os.getenv("EXECUTOR_MODEL", "llama-3.1-8b-instant"),
        max_iterations=_get_int("MAX_ITERATIONS", 10),
        max_tool_calls=_get_int("MAX_TOOL_CALLS", 20),
        anthropic_enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
    )


def build_provider(model: str, config: OrchestratorConfig) -> LLMProvider:
    """Construct the provider for ``config.provider`` bound to ``model``."""
    provider = config.provider
    if provider == "groq":
        from providers.groq_provider import GroqProvider

        return GroqProvider(model)
    if provider == "anthropic":
        if not config.anthropic_enabled:
            raise ProviderError(
                "provider='anthropic' selected but ANTHROPIC_API_KEY is not set. "
                "Set the key or use the default Groq provider."
            )
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model)
    raise ProviderError(f"Unknown provider {provider!r}. Use 'groq' or 'anthropic'.")


def build_planner(config: OrchestratorConfig) -> LLMProvider:
    return build_provider(config.planner_model, config)


def build_executor(config: OrchestratorConfig) -> LLMProvider:
    return build_provider(config.executor_model, config)
