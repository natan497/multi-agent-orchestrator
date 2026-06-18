"""Unit tests for the Anthropic stub: construction guard + message translation.

No live API and no `anthropic` package required — a fake client is injected.
"""

from types import SimpleNamespace

import pytest

from orchestrator.models import Message, ToolCall, ToolSpec
from providers.anthropic_provider import AnthropicProvider
from providers.base import ProviderError


def test_requires_key_or_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("claude-opus-4-8")


def test_split_system_extracts_system_prompt():
    system, msgs = AnthropicProvider._split_system(
        [Message(role="system", content="be brief"), Message(role="user", content="hi")]
    )
    assert system == "be brief"
    assert msgs == [{"role": "user", "content": "hi"}]


def test_split_system_translates_tool_turns():
    call = ToolCall(id="t1", name="calc", arguments={"a": 1})
    _, msgs = AnthropicProvider._split_system(
        [
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id="t1", content="2"),
        ]
    )
    assert msgs[0]["content"][0] == {
        "type": "tool_use",
        "id": "t1",
        "name": "calc",
        "input": {"a": 1},
    }
    assert msgs[1]["content"][0]["type"] == "tool_result"
    assert msgs[1]["content"][0]["tool_use_id"] == "t1"


def test_complete_parses_text_and_tool_use_via_fake_client():
    fake_raw = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="u1", name="calc", input={"a": 2}),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=4),
        stop_reason="tool_use",
    )
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return fake_raw

    client = SimpleNamespace(messages=FakeMessages())
    provider = AnthropicProvider("claude-opus-4-8", client=client)
    out = provider.complete(
        [Message(role="system", content="sys"), Message(role="user", content="add")],
        tools=[ToolSpec(name="calc", description="adds", parameters={"type": "object"})],
    )
    assert out.text == "hello"
    assert out.tool_calls[0].arguments == {"a": 2}
    assert out.usage.cached_tokens == 4
    assert captured["system"] == "sys"
    assert captured["tools"][0]["input_schema"] == {"type": "object"}
