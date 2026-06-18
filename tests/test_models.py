"""Unit tests for the normalized data models."""

import json

from orchestrator.models import (
    Completion,
    ControlDecision,
    Message,
    Observation,
    Plan,
    RunTrace,
    Step,
    StepStatus,
    ToolCall,
    ToolSpec,
    Usage,
)


def test_usage_total_and_addition():
    a = Usage(input_tokens=10, output_tokens=5, cached_tokens=4)
    b = Usage(input_tokens=1, output_tokens=2, cached_tokens=3)
    assert a.total_tokens == 15
    combined = a + b
    assert combined.input_tokens == 11
    assert combined.output_tokens == 7
    assert combined.cached_tokens == 7


def test_completion_has_tool_calls_flag():
    empty = Completion(text="hi")
    assert empty.has_tool_calls is False
    withcall = Completion(tool_calls=[ToolCall(id="1", name="calc", arguments={"x": 1})])
    assert withcall.has_tool_calls is True


def test_completion_raw_excluded_from_serialization():
    c = Completion(text="hi", raw={"provider": "internal", "secret": object()})
    dumped = c.model_dump()
    assert "raw" not in dumped
    # raw stays accessible in-process for debugging.
    assert c.raw["provider"] == "internal"


def test_toolspec_default_schema_is_object():
    spec = ToolSpec(name="noop", description="does nothing")
    assert spec.parameters == {"type": "object", "properties": {}}


def test_message_defaults():
    m = Message(role="user", content="hello")
    assert m.tool_calls == []
    assert m.tool_call_id is None


def test_step_status_enum_default():
    step = Step(index=0, description="do thing")
    assert step.status is StepStatus.PENDING


def test_plan_and_observation_roundtrip():
    plan = Plan(goal="g", steps=[Step(index=0, description="s0", tool="calc")])
    assert plan.steps[0].tool == "calc"
    obs = Observation(step_index=0, output="42", ok=True)
    assert obs.ok and obs.error is None


def test_control_decision_values():
    assert {d.value for d in ControlDecision} == {"continue", "retry", "replan", "done"}


def test_runtrace_add_records_sequence_and_usage():
    trace = RunTrace(goal="weather in Denver")
    trace.add("plan", steps=2)
    trace.add("tool_call", name="weather")
    assert [e.seq for e in trace.events] == [0, 1]
    assert trace.events[1].kind == "tool_call"

    trace.record_usage(Usage(input_tokens=100, output_tokens=20, cached_tokens=80))
    trace.record_usage(Usage(input_tokens=50, output_tokens=10))
    assert trace.usage.input_tokens == 150
    assert trace.usage.cached_tokens == 80


def test_runtrace_to_json_is_valid_and_excludes_raw():
    trace = RunTrace(goal="g", final_answer="done")
    trace.add("decision", decision="done")
    parsed = json.loads(trace.to_json())
    assert parsed["goal"] == "g"
    assert parsed["final_answer"] == "done"
    assert parsed["events"][0]["data"]["decision"] == "done"
