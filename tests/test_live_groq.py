"""Opt-in live smoke tests against the real Groq API.

Deselected by default (see pyproject `addopts = -m 'not live'`) and additionally skipped
unless a GROQ_API_KEY is present, so the suite stays green with no key. Run with:

    GROQ_API_KEY=... pytest -m live tests/test_live_groq.py

These spend a tiny number of free-tier tokens. They are the real end-to-end check that the
provider wiring, tool calling, and the orchestration loop work against a live model.
"""

import os

import pytest

from orchestrator.config import load_config
from orchestrator.models import Message
from orchestrator.orchestrator import Orchestrator
from providers.base import ProviderError
from providers.groq_provider import GroqProvider
from tools.builtins.calculator import Calculator
from tools.registry import ToolRegistry

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]


def _executor_model() -> str:
    return load_config().executor_model


def test_groq_chat_smoke():
    """A trivial chat completion returns text and reports token usage."""
    provider = GroqProvider(_executor_model())
    try:
        completion = provider.complete(
            [Message(role="user", content="Reply with exactly the word: pong")]
        )
    except ProviderError as e:
        pytest.skip(f"Groq unavailable: {e}")
    assert completion.text is not None
    assert completion.usage.total_tokens > 0


def test_groq_tool_call_smoke():
    """Offered a calculator, the model issues a tool call (best-effort)."""
    provider = GroqProvider(_executor_model())
    try:
        completion = provider.complete(
            [Message(role="user", content="Use the calculator tool to compute 6 * 7.")],
            tools=[Calculator.to_spec()],
        )
    except ProviderError as e:
        pytest.skip(f"Groq unavailable: {e}")
    # Models occasionally answer directly; accept either, but require a clean response.
    assert completion.text is not None or completion.has_tool_calls
    if completion.has_tool_calls:
        assert completion.tool_calls[0].name == "calculator"


def test_orchestrator_end_to_end_live():
    """The full plan-execute-observe loop solves a simple task end to end."""
    config = load_config()
    registry = ToolRegistry()
    registry.register(Calculator())
    orchestrator = Orchestrator.from_config(config, registry)
    try:
        result = orchestrator.run("What is 6 * 7? Reply with just the number.")
    except ProviderError as e:
        pytest.skip(f"Groq unavailable: {e}")
    assert result.success, result.stop_reason
    assert "42" in (result.final_answer or "")
