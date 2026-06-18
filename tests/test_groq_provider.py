"""Unit tests for GroqProvider: translation, parsing, and rate-limit/backoff.

Fully mocked — no network, no API key. A fake client stands in for the Groq SDK.
"""

from types import SimpleNamespace

import pytest

from orchestrator.models import Message, ToolSpec
from providers.base import ProviderError, RateLimitError, ToolFormatError
from providers.groq_provider import GroqProvider


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeError(Exception):
    """Stand-in for a Groq SDK status error (duck-typed status_code + response)."""

    def __init__(self, status_code, headers=None, name=None, message=None):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})
        if name:
            type(self).__name__ = name


class FakeCompletions:
    def __init__(self, outcomes):
        # outcomes: list of either an exception (raised) or a response (returned)
        self._outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))

    @property
    def calls(self):
        return self.chat.completions.calls


def make_response(*, content=None, tool_calls=None, cached=0, finish="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def make_tool_call(id_, name, arguments):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def provider(outcomes, **kw):
    # sleep stubbed so backoff never actually waits during tests
    return GroqProvider("test-model", client=FakeClient(outcomes), sleep=lambda _: None, **kw)


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #
def test_encodes_messages_and_tools_into_payload():
    p = provider([make_response(content="ok")])
    p.complete(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        tools=[ToolSpec(name="calc", description="adds", parameters={"type": "object"})],
    )
    payload = p.client.calls[0]
    assert payload["model"] == "test-model"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["tools"][0]["function"]["name"] == "calc"


def test_encodes_assistant_tool_calls_and_tool_results():
    from orchestrator.models import ToolCall

    p = provider([make_response(content="ok")])
    call = ToolCall(id="c1", name="calc", arguments={"a": 1})
    p.complete(
        [
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id="c1", name="calc", content="2"),
        ]
    )
    msgs = p.client.calls[0]["messages"]
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert msgs[1] == {"role": "tool", "content": "2", "tool_call_id": "c1", "name": "calc"}


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def test_parses_text_and_usage_including_cached_tokens():
    p = provider([make_response(content="hello", cached=80)])
    out = p.complete([Message(role="user", content="hi")])
    assert out.text == "hello"
    assert out.usage.input_tokens == 100
    assert out.usage.output_tokens == 20
    assert out.usage.cached_tokens == 80
    assert out.finish_reason == "stop"


def test_parses_tool_calls_with_json_arguments():
    resp = make_response(
        tool_calls=[make_tool_call("call_1", "calc", '{"a": 1, "b": 2}')], finish="tool_calls"
    )
    p = provider([resp])
    out = p.complete([Message(role="user", content="add")])
    assert out.has_tool_calls
    assert out.tool_calls[0].name == "calc"
    assert out.tool_calls[0].arguments == {"a": 1, "b": 2}


def test_invalid_tool_arguments_raise_tool_format_error():
    resp = make_response(tool_calls=[make_tool_call("c", "calc", "{not json")])
    p = provider([resp])
    with pytest.raises(ToolFormatError):
        p.complete([Message(role="user", content="x")])


# --------------------------------------------------------------------------- #
# Rate limit / backoff / error normalization
# --------------------------------------------------------------------------- #
def test_429_retries_then_succeeds():
    p = provider([FakeError(429, headers={"retry-after": "0"}), make_response(content="ok")])
    out = p.complete([Message(role="user", content="hi")])
    assert out.text == "ok"
    assert len(p.client.calls) == 2  # retried once


def test_429_exhausts_retries_and_raises_rate_limit_error():
    outcomes = [FakeError(429, headers={"retry-after": "0"}) for _ in range(5)]
    p = provider(outcomes, max_retries=5)
    with pytest.raises(RateLimitError):
        p.complete([Message(role="user", content="hi")])
    assert len(p.client.calls) == 5


def test_transient_5xx_is_retried():
    p = provider([FakeError(503), make_response(content="recovered")])
    out = p.complete([Message(role="user", content="hi")])
    assert out.text == "recovered"
    assert len(p.client.calls) == 2


def test_non_retryable_error_is_not_retried():
    p = provider([FakeError(400)])
    with pytest.raises(ProviderError) as ei:
        p.complete([Message(role="user", content="hi")])
    assert not isinstance(ei.value, RateLimitError)
    assert len(p.client.calls) == 1


def test_tool_use_failed_is_retried():
    # Groq returns a 400 with code 'tool_use_failed' when a model emits a malformed
    # tool call; it's non-deterministic, so we retry rather than abort.
    err = FakeError(
        400,
        message="Failed to call a function ... 'code': 'tool_use_failed', "
        "'failed_generation': '<function=weather>{\"location\": \"Denver\"}}'",
    )
    p = provider([err, make_response(content="recovered")])
    out = p.complete([Message(role="user", content="weather in Denver")])
    assert out.text == "recovered"
    assert len(p.client.calls) == 2


def test_wait_honors_retry_after():
    err = RateLimitError("limited", retry_after=7.0)
    state = SimpleNamespace(outcome=SimpleNamespace(exception=lambda: err), attempt_number=1)
    assert GroqProvider._wait(state) == 7.0


def test_default_client_does_not_override_base_url(monkeypatch):
    """Regression: the native groq SDK already targets /openai/v1, so passing base_url
    would double the path (…/openai/v1/openai/v1/chat/completions -> 404)."""
    import groq

    captured = {}

    def fake_groq(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(groq, "Groq", fake_groq)
    p = GroqProvider("m", api_key="test-key")
    _ = p.client  # triggers lazy construction
    assert "base_url" not in captured
    assert captured.get("api_key") == "test-key"
