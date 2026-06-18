"""End-to-end (mocked) round-trip: a tool registered, offered to the provider,
called by the model, and dispatched back through the registry.

This is the Phase 3 "Done when" check — the calculator round-trips through the Groq
provider — without any network call.
"""

from types import SimpleNamespace

from orchestrator.models import Message
from providers.groq_provider import GroqProvider
from tools.builtins.calculator import Calculator
from tools.registry import ToolRegistry


def _fake_tool_call_response(name, arguments_json):
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=arguments_json))
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10, prompt_tokens_details=None)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_calculator_round_trips_through_provider():
    registry = ToolRegistry()
    registry.register(Calculator())

    client = _FakeClient(_fake_tool_call_response("calculator", '{"expression": "6 * 7"}'))
    provider = GroqProvider("test-model", client=client, sleep=lambda _: None)

    # The registry's specs are what we'd offer the model.
    completion = provider.complete(
        [Message(role="user", content="what is 6 times 7?")],
        tools=registry.specs(),
    )

    # The tool spec was actually serialized into the request.
    assert client.calls[0]["tools"][0]["function"]["name"] == "calculator"

    # The model's tool call dispatches cleanly back through the registry.
    assert completion.has_tool_calls
    result = registry.dispatch(completion.tool_calls[0])
    assert result.ok
    assert result.output == "42"
