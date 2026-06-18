"""Normalized data models shared across the orchestrator and all LLM providers.

These types are deliberately provider-neutral. Concrete providers translate to and
from their own wire formats internally (see ``providers/base.py``); the orchestrator,
planner, and executor only ever see the models defined here.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# LLM I/O primitives
# --------------------------------------------------------------------------- #
class ToolCall(BaseModel):
    """A model's request to invoke a tool with parsed arguments."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single conversation message in normalized form.

    ``tool_calls`` is set on assistant turns that request tools; ``tool_call_id``
    and ``name`` are set on ``role="tool"`` turns carrying a tool result.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """Provider-neutral description of a callable tool.

    ``parameters`` is a JSON Schema object. The tool registry produces these from
    executable tools, and each provider serializes them to its own tool format.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Usage(BaseModel):
    """Token accounting for a single provider call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class Completion(BaseModel):
    """Normalized result of an ``LLMProvider.complete`` call.

    ``raw`` holds the provider's original response for debugging and is excluded
    from serialization so traces stay clean and provider-agnostic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = Field(default=None, exclude=True, repr=False)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# --------------------------------------------------------------------------- #
# Planning / execution state
# --------------------------------------------------------------------------- #
class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Step(BaseModel):
    """One unit of a plan."""

    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    tool: str | None = None
    result: str | None = None


class Plan(BaseModel):
    """An ordered set of steps the planner produced for a goal."""

    goal: str
    steps: list[Step] = Field(default_factory=list)


class Observation(BaseModel):
    """The outcome of executing a step (running a tool)."""

    step_index: int
    tool_call: ToolCall | None = None
    output: str = ""
    ok: bool = True
    error: str | None = None


class ControlDecision(StrEnum):
    """What the planner decides after observing a step's result."""

    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    DONE = "done"


# --------------------------------------------------------------------------- #
# Run trace
# --------------------------------------------------------------------------- #
class TraceEvent(BaseModel):
    """A single timestamped record of something that happened during a run."""

    seq: int
    kind: str  # e.g. "plan", "tool_call", "observation", "decision", "retry", "error"
    data: dict[str, Any] = Field(default_factory=dict)


class RunTrace(BaseModel):
    """Ordered record of every agent decision and tool I/O for one run."""

    goal: str
    events: list[TraceEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    final_answer: str | None = None

    def add(self, kind: str, **data: Any) -> TraceEvent:
        event = TraceEvent(seq=len(self.events), kind=kind, data=data)
        self.events.append(event)
        return event

    def record_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class OrchestratorConfig(BaseModel):
    """Resolved runtime configuration. Env loading lives in ``config.py`` (Phase 2)."""

    provider: str = "groq"
    planner_model: str = "openai/gpt-oss-120b"
    executor_model: str = "llama-3.1-8b-instant"
    max_iterations: int = 10
    max_tool_calls: int = 20
    anthropic_enabled: bool = False
