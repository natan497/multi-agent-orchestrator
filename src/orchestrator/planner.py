"""Planner agent: decomposes a goal into a plan and decides what to do after each step.

The planner uses the larger model. Its system prompt is a module-level constant so that
Groq's automatic prefix caching applies (cached tokens don't count toward TPM — SPEC §5).
Prompts are kept lean and observations summarized to stay within the per-minute budget.

The planner asks the model for JSON and parses it defensively (tolerating code fences and
surrounding prose). Unparseable output raises :class:`PlannerError`, which the orchestrator
turns into a graceful failed run rather than a crash.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from orchestrator.models import (
    ControlDecision,
    Message,
    Observation,
    Plan,
    Step,
    ToolSpec,
    Usage,
)
from providers.base import LLMProvider

PLANNER_SYSTEM = (
    "You are the planner in a multi-agent system. You break a user's goal into a short, "
    "ordered list of concrete steps that an executor agent can carry out by calling tools, "
    "and after each step you decide whether to continue, retry, re-plan, or finish.\n"
    "Always respond with a single JSON object and nothing else. Keep plans minimal — prefer "
    "the fewest steps that achieve the goal."
)


class PlannerError(Exception):
    """Raised when the planner's output cannot be parsed into a plan/decision."""


class _StepOut(BaseModel):
    description: str
    tool: str | None = None


class _PlanOut(BaseModel):
    steps: list[_StepOut]


class _DecisionOut(BaseModel):
    decision: ControlDecision
    final_answer: str | None = None
    steps: list[_StepOut] | None = None


class PlanResult(BaseModel):
    plan: Plan
    usage: Usage


class DecisionResult(BaseModel):
    decision: ControlDecision
    final_answer: str | None = None
    plan: Plan | None = None  # populated on REPLAN
    usage: Usage


class FinalizeResult(BaseModel):
    final_answer: str
    usage: Usage


class Planner:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def make_plan(self, goal: str, tools: list[ToolSpec]) -> PlanResult:
        prompt = (
            f"Goal: {goal}\n\n"
            f"Available tools:\n{_render_tools(tools)}\n\n"
            'Produce a plan as JSON: {"steps": [{"description": str, "tool": str|null}, ...]}. '
            'Set "tool" to the tool name a step will likely use, or null if none.'
        )
        completion = self.provider.complete(
            [Message(role="system", content=PLANNER_SYSTEM), Message(role="user", content=prompt)]
        )
        parsed = _parse(completion.text, _PlanOut)
        plan = Plan(
            goal=goal,
            steps=[
                Step(index=i, description=s.description, tool=s.tool)
                for i, s in enumerate(parsed.steps)
            ],
        )
        return PlanResult(plan=plan, usage=completion.usage)

    def decide(self, goal: str, plan: Plan, observation: Observation) -> DecisionResult:
        step = plan.steps[observation.step_index]
        outcome = (
            f"output: {observation.output}" if observation.ok else f"error: {observation.error}"
        )
        prompt = (
            f"Goal: {goal}\n\n"
            f"Plan ({len(plan.steps)} steps):\n{_render_steps(plan)}\n\n"
            f"Just executed step {observation.step_index} ({step.description!r}).\n"
            f"Result ({'ok' if observation.ok else 'failed'}): {outcome}\n\n"
            "Decide the next action as JSON: "
            '{"decision": "continue"|"retry"|"replan"|"done", '
            '"final_answer": str|null, "steps": [{"description": str, "tool": str|null}]|null}. '
            'Use "done" with a "final_answer" when the goal is met; "retry" to repeat this '
            'step; "replan" with a new "steps" list to start over; "continue" otherwise.'
        )
        completion = self.provider.complete(
            [Message(role="system", content=PLANNER_SYSTEM), Message(role="user", content=prompt)]
        )
        parsed = _parse(completion.text, _DecisionOut)
        new_plan: Plan | None = None
        if parsed.decision is ControlDecision.REPLAN and parsed.steps:
            new_plan = Plan(
                goal=goal,
                steps=[
                    Step(index=i, description=s.description, tool=s.tool)
                    for i, s in enumerate(parsed.steps)
                ],
            )
        return DecisionResult(
            decision=parsed.decision,
            final_answer=parsed.final_answer,
            plan=new_plan,
            usage=completion.usage,
        )

    def finalize(self, goal: str, observations: list[Observation]) -> FinalizeResult:
        """Synthesize a final answer from observations (used when steps end without 'done')."""
        summary = "\n".join(
            f"- step {o.step_index}: {'ok ' + o.output if o.ok else 'failed ' + (o.error or '')}"
            for o in observations
        )
        prompt = (
            f"Goal: {goal}\n\nResults so far:\n{summary or '(none)'}\n\n"
            'Write the final answer to the goal as JSON: {"final_answer": str}.'
        )
        completion = self.provider.complete(
            [Message(role="system", content=PLANNER_SYSTEM), Message(role="user", content=prompt)]
        )

        class _FinalOut(BaseModel):
            final_answer: str

        parsed = _parse(completion.text, _FinalOut)
        return FinalizeResult(final_answer=parsed.final_answer, usage=completion.usage)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _render_tools(tools: list[ToolSpec]) -> str:
    if not tools:
        return "(no tools available)"
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


def _render_steps(plan: Plan) -> str:
    return "\n".join(
        f"{s.index}. [{s.status}] {s.description}" + (f" (tool: {s.tool})" if s.tool else "")
        for s in plan.steps
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse[T: BaseModel](text: str | None, model: type[T]) -> T:
    obj = _extract_json(text)
    try:
        return model.model_validate(obj)
    except ValidationError as e:
        raise PlannerError(f"planner output failed validation: {e}") from e


def _extract_json(text: str | None) -> dict:
    if not text or not text.strip():
        raise PlannerError("planner returned empty output")
    candidate = text.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    else:
        # Fall back to the outermost {...} span if the model wrapped JSON in prose.
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise PlannerError(f"planner output was not valid JSON: {text!r}") from e
    if not isinstance(obj, dict):
        raise PlannerError(f"planner output was not a JSON object: {text!r}")
    return obj
