"""Unit tests for env-driven config and the provider factory."""

import pytest

from orchestrator.config import build_executor, build_planner, build_provider, load_config
from providers.anthropic_provider import AnthropicProvider
from providers.base import ProviderError
from providers.groq_provider import GroqProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "LLM_PROVIDER",
        "PLANNER_MODEL",
        "EXECUTOR_MODEL",
        "MAX_ITERATIONS",
        "MAX_TOOL_CALLS",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_load_config_defaults():
    cfg = load_config(dotenv=False)
    assert cfg.provider == "groq"
    assert cfg.planner_model == "openai/gpt-oss-120b"
    assert cfg.executor_model == "llama-3.1-8b-instant"
    assert cfg.max_iterations == 10
    assert cfg.anthropic_enabled is False


def test_load_config_reads_env(monkeypatch):
    monkeypatch.setenv("PLANNER_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("MAX_ITERATIONS", "3")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = load_config(dotenv=False)
    assert cfg.planner_model == "llama-3.3-70b-versatile"
    assert cfg.max_iterations == 3
    assert cfg.anthropic_enabled is True


def test_invalid_int_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MAX_TOOL_CALLS", "not-a-number")
    cfg = load_config(dotenv=False)
    assert cfg.max_tool_calls == 20


def test_build_provider_defaults_to_groq():
    cfg = load_config(dotenv=False)
    planner = build_planner(cfg)
    executor = build_executor(cfg)
    assert isinstance(planner, GroqProvider)
    assert isinstance(executor, GroqProvider)
    assert planner.model == "openai/gpt-oss-120b"
    assert executor.model == "llama-3.1-8b-instant"


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    cfg = load_config(dotenv=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        build_planner(cfg)


def test_anthropic_built_when_selected_and_keyed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = load_config(dotenv=False)
    provider = build_provider("claude-opus-4-8", cfg)
    assert isinstance(provider, AnthropicProvider)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    cfg = load_config(dotenv=False)
    with pytest.raises(ProviderError, match="Unknown provider"):
        build_planner(cfg)
