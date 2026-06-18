"""Integration tests for the plan-execute-observe loop (fully mocked providers)."""

import json

from conftest import ScriptedProvider, text_completion, tool_completion
from orchestrator.executor import Executor
from orchestrator.models import OrchestratorConfig
from orchestrator.orchestrator import Orchestrator
from orchestrator.planner import Planner
from orchestrator.state import Pacer
from providers.base import ProviderError, RateLimitError
from tools.builtins.calculator import Calculator
from tools.registry import ToolRegistry


def build(planner_texts, executor_completions, *, config=None):
    registry = ToolRegistry()
    registry.register(Calculator())
    planner = Planner(ScriptedProvider([text_completion(t) for t in planner_texts]))
    executor = Executor(ScriptedProvider(executor_completions))
    cfg = config or OrchestratorConfig()
    # Pacer(0) disables real sleeping during tests.
    return Orchestrator(planner, executor, registry, cfg, pacer=Pacer(0))


def test_two_tool_task_runs_end_to_end_with_full_trace():
    planner_texts = [
        '{"steps": [{"description": "compute 2+2", "tool": "calculator"}, '
        '{"description": "multiply by 10", "tool": "calculator"}]}',
        '{"decision": "continue"}',
        '{"decision": "done", "final_answer": "40"}',
    ]
    executor_completions = [
        tool_completion("calculator", {"expression": "2 + 2"}),
        tool_completion("calculator", {"expression": "4 * 10"}),
    ]
    orch = build(planner_texts, executor_completions)
    result = orch.run("compute (2+2) then times 10")

    assert result.success
    assert result.final_answer == "40"
    assert result.tool_calls == 2
    assert result.stop_reason == "done"

    kinds = [e.kind for e in result.trace.events]
    assert kinds.count("tool_call") == 2
    assert kinds.count("observation") == 2
    assert "plan" in kinds and "decision" in kinds
    # The trace serializes cleanly to JSON.
    assert json.loads(result.trace.to_json())["final_answer"] == "40"
    # Tool calls were really dispatched: both observations succeeded.
    observations = [e for e in result.trace.events if e.kind == "observation"]
    assert [o.data["output"] for o in observations] == ["4", "40"]


def test_unknown_tool_is_handled_gracefully():
    planner_texts = [
        '{"steps": [{"description": "do it", "tool": "ghost"}]}',
        '{"decision": "done", "final_answer": "handled"}',
    ]
    executor_completions = [tool_completion("ghost", {})]
    orch = build(planner_texts, executor_completions)
    result = orch.run("call a missing tool")

    assert result.success  # planner recovered; no crash
    obs = [e for e in result.trace.events if e.kind == "observation"][0]
    assert obs.data["ok"] is False
    assert "unknown tool" in obs.data["error"]


def test_max_iterations_guardrail():
    planner_texts = [
        '{"steps": [{"description": "a"}, {"description": "b"}]}',
        '{"decision": "continue"}',  # after step 0
    ]
    executor_completions = [tool_completion("calculator", {"expression": "1+1"})]
    orch = build(planner_texts, executor_completions, config=OrchestratorConfig(max_iterations=1))
    result = orch.run("g")
    assert not result.success
    assert result.stop_reason == "reached max_iterations"


def test_max_tool_calls_guardrail():
    planner_texts = [
        '{"steps": [{"description": "a"}, {"description": "b"}]}',
        '{"decision": "continue"}',
    ]
    executor_completions = [
        tool_completion("calculator", {"expression": "1+1"}),
        tool_completion("calculator", {"expression": "2+2"}),
    ]
    orch = build(planner_texts, executor_completions, config=OrchestratorConfig(max_tool_calls=1))
    result = orch.run("g")
    assert not result.success
    assert result.stop_reason == "reached max_tool_calls"


def test_retry_cap_guardrail():
    planner_texts = [
        '{"steps": [{"description": "flaky"}]}',
        '{"decision": "retry"}',
        '{"decision": "retry"}',
    ]
    executor_completions = [
        tool_completion("calculator", {"expression": "1+1"}),
        tool_completion("calculator", {"expression": "1+1"}),
    ]
    orch = build(
        planner_texts, executor_completions, config=OrchestratorConfig(max_retries_per_step=1)
    )
    result = orch.run("g")
    assert not result.success
    assert "retry limit" in result.stop_reason


def test_executor_provider_error_becomes_failed_observation():
    # A non-rate-limit provider error (e.g. a model's malformed tool call that exhausted
    # retries) should not abort the run: it becomes a failed observation the planner acts on.
    planner_texts = [
        '{"steps": [{"description": "do it", "tool": "weather"}]}',
        '{"decision": "done", "final_answer": "recovered"}',
    ]
    registry = ToolRegistry()
    registry.register(Calculator())
    planner = Planner(ScriptedProvider([text_completion(t) for t in planner_texts]))
    executor = Executor(ScriptedProvider([ProviderError("tool_use_failed")]))
    orch = Orchestrator(planner, executor, registry, OrchestratorConfig(), pacer=Pacer(0))

    result = orch.run("weather please")
    assert result.success
    assert result.final_answer == "recovered"
    obs = [e for e in result.trace.events if e.kind == "error" and e.data["stage"] == "execute"]
    assert obs  # the executor failure was recorded but did not abort the run


def test_executor_rate_limit_aborts_run():
    planner = Planner(ScriptedProvider([text_completion('{"steps": [{"description": "x"}]}')]))
    executor = Executor(ScriptedProvider([RateLimitError("429")]))
    registry = ToolRegistry()
    registry.register(Calculator())
    orch = Orchestrator(planner, executor, registry, OrchestratorConfig(), pacer=Pacer(0))

    result = orch.run("g")
    assert not result.success
    assert result.stop_reason.startswith("rate limited")


def test_planning_failure_ends_run_gracefully():
    orch = build(["this is not valid json"], [])
    result = orch.run("g")
    assert not result.success
    assert result.stop_reason.startswith("planning failed")
    assert result.trace.events[0].kind == "error"


def test_replan_then_finish():
    planner_texts = [
        '{"steps": [{"description": "first"}]}',
        '{"decision": "replan", "steps": [{"description": "better"}]}',
        '{"decision": "done", "final_answer": "ok"}',
    ]
    executor_completions = [
        tool_completion("calculator", {"expression": "1+1"}),
        tool_completion("calculator", {"expression": "2+2"}),
    ]
    orch = build(planner_texts, executor_completions)
    result = orch.run("g")
    assert result.success
    assert result.final_answer == "ok"
    assert any(e.kind == "replan" for e in result.trace.events)


def test_finalize_when_steps_complete_without_done():
    planner_texts = [
        '{"steps": [{"description": "only step"}]}',
        '{"decision": "continue"}',  # advances past the last step
        '{"final_answer": "synthesized from observations"}',  # finalize call
    ]
    executor_completions = [tool_completion("calculator", {"expression": "3+4"})]
    orch = build(planner_texts, executor_completions)
    result = orch.run("g")
    assert result.success
    assert result.stop_reason == "completed_all_steps"
    assert result.final_answer == "synthesized from observations"
