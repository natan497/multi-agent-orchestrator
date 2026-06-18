"""Unit tests for the Planner agent (parsing + decision logic)."""

import pytest

from conftest import ScriptedProvider, text_completion
from orchestrator.models import ControlDecision, Observation, Plan, Step, ToolSpec
from orchestrator.planner import Planner, PlannerError


def planner_with(*texts):
    return Planner(ScriptedProvider([text_completion(t) for t in texts]))


def test_make_plan_parses_steps():
    p = planner_with(
        '{"steps": [{"description": "add", "tool": "calculator"}, {"description": "report"}]}'
    )
    res = p.make_plan("do math", [ToolSpec(name="calculator", description="adds")])
    assert [s.description for s in res.plan.steps] == ["add", "report"]
    assert res.plan.steps[0].tool == "calculator"
    assert res.plan.steps[1].tool is None
    assert res.plan.steps[0].index == 0
    assert res.usage.total_tokens == 20


def test_make_plan_tolerates_code_fences():
    p = planner_with('```json\n{"steps": [{"description": "x"}]}\n```')
    res = p.make_plan("g", [])
    assert res.plan.steps[0].description == "x"


def test_make_plan_tolerates_surrounding_prose():
    p = planner_with('Sure! Here is the plan: {"steps": [{"description": "y"}]} Hope that helps.')
    res = p.make_plan("g", [])
    assert res.plan.steps[0].description == "y"


@pytest.mark.parametrize("bad", ["", "   ", "not json at all", "[1, 2, 3]"])
def test_make_plan_raises_on_unparseable_output(bad):
    p = planner_with(bad)
    with pytest.raises(PlannerError):
        p.make_plan("g", [])


def _plan() -> Plan:
    return Plan(goal="g", steps=[Step(index=0, description="s0"), Step(index=1, description="s1")])


def test_decide_continue():
    p = planner_with('{"decision": "continue"}')
    res = p.decide("g", _plan(), Observation(step_index=0, output="ok"))
    assert res.decision is ControlDecision.CONTINUE
    assert res.plan is None


def test_decide_done_with_final_answer():
    p = planner_with('{"decision": "done", "final_answer": "42"}')
    res = p.decide("g", _plan(), Observation(step_index=0, output="42"))
    assert res.decision is ControlDecision.DONE
    assert res.final_answer == "42"


def test_decide_replan_builds_new_plan():
    p = planner_with('{"decision": "replan", "steps": [{"description": "fresh"}]}')
    res = p.decide("g", _plan(), Observation(step_index=0, ok=False, error="boom"))
    assert res.decision is ControlDecision.REPLAN
    assert res.plan is not None
    assert res.plan.steps[0].description == "fresh"


def test_finalize_returns_answer():
    p = planner_with('{"final_answer": "the result is 42"}')
    res = p.finalize("g", [Observation(step_index=0, output="42")])
    assert res.final_answer == "the result is 42"
