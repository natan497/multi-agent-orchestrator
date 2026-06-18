"""Shared test helpers: a scripted provider and Completion builders."""

from __future__ import annotations

from orchestrator.models import Completion, ToolCall, Usage
from providers.base import LLMProvider


class ScriptedProvider(LLMProvider):
    """Returns pre-built Completions in order; records each call's messages/tools."""

    def __init__(self, completions: list[Completion]) -> None:
        super().__init__("scripted")
        self._completions = list(completions)
        self.calls: list[tuple] = []

    def complete(self, messages, tools=None, **opts):
        self.calls.append((messages, tools))
        if not self._completions:
            raise AssertionError("ScriptedProvider ran out of scripted completions")
        return self._completions.pop(0)


def text_completion(text: str, *, tokens: int = 10) -> Completion:
    return Completion(text=text, usage=Usage(input_tokens=tokens, output_tokens=tokens))


def tool_completion(
    name: str, arguments: dict, *, call_id: str = "c1", tokens: int = 10
) -> Completion:
    return Completion(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        usage=Usage(input_tokens=tokens, output_tokens=tokens),
        finish_reason="tool_calls",
    )
