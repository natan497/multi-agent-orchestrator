"""Executor agent: turns the current step into a single tool call.

The executor uses the small, fast model. It receives the step and the registry's tool
specs and is expected to emit one tool call. If the model answers in plain text instead
(no tool needed), that text is returned so the orchestrator can record it as the step's
observation. The system prompt is a constant for prefix caching (SPEC §5).
"""

from __future__ import annotations

from pydantic import BaseModel

from orchestrator.models import Message, Step, ToolCall, ToolSpec, Usage
from providers.base import LLMProvider

EXECUTOR_SYSTEM = (
    "You are the executor in a multi-agent system. You are given one step of a plan and a "
    "set of tools. Call exactly one tool that accomplishes the step, with correct arguments. "
    "Only answer in plain text if no tool is needed for this step."
)


class ExecutionResult(BaseModel):
    tool_call: ToolCall | None = None
    text: str | None = None
    usage: Usage

    @property
    def called_tool(self) -> bool:
        return self.tool_call is not None


class Executor:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def execute(self, goal: str, step: Step, tools: list[ToolSpec]) -> ExecutionResult:
        prompt = (
            f"Overall goal: {goal}\n"
            f"Current step: {step.description}\n"
            + (f"Suggested tool: {step.tool}\n" if step.tool else "")
            + "Call the single most appropriate tool to carry out this step."
        )
        completion = self.provider.complete(
            [
                Message(role="system", content=EXECUTOR_SYSTEM),
                Message(role="user", content=prompt),
            ],
            tools=tools,
        )
        tool_call = completion.tool_calls[0] if completion.tool_calls else None
        return ExecutionResult(tool_call=tool_call, text=completion.text, usage=completion.usage)
