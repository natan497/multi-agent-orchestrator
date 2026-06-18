"""Unit tests for the LLMProvider interface contract and normalized errors."""

import pytest

from orchestrator.models import Completion, Message, ToolSpec, Usage
from providers.base import LLMProvider, ProviderError, RateLimitError, ToolFormatError


def test_cannot_instantiate_abstract_provider():
    with pytest.raises(TypeError):
        LLMProvider("some-model")  # type: ignore[abstract]


class _EchoProvider(LLMProvider):
    """Minimal concrete provider used to exercise the interface."""

    def complete(self, messages, tools=None, **opts):
        last = messages[-1].content if messages else ""
        return Completion(text=f"echo: {last}", usage=Usage(input_tokens=1, output_tokens=1))


def test_concrete_provider_carries_model_and_name():
    p = _EchoProvider("openai/gpt-oss-120b")
    assert p.model == "openai/gpt-oss-120b"
    # name is derived from the class: _EchoProvider -> "_echo"
    assert p.name == "_echo"


def test_concrete_provider_complete_returns_normalized_completion():
    p = _EchoProvider("m")
    out = p.complete(
        [Message(role="user", content="hi")],
        tools=[ToolSpec(name="t", description="d")],
    )
    assert out.text == "echo: hi"
    assert out.usage.total_tokens == 2


def test_error_hierarchy():
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(ToolFormatError, ProviderError)


def test_rate_limit_error_carries_retry_after():
    err = RateLimitError("429", retry_after=2.5)
    assert err.retry_after == 2.5
    assert RateLimitError("429").retry_after is None
