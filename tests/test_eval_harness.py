"""Unit tests for the eval harness (no network, no API key)."""

from examples.eval_harness import EvalCase, answer_contains, run_evals

from orchestrator.models import RunTrace, Usage
from orchestrator.state import RunResult


def make_result(*, success=True, answer="", tool_calls=0, iterations=0, tokens=0) -> RunResult:
    trace = RunTrace(goal="g")
    if tokens:
        trace.record_usage(Usage(input_tokens=tokens, output_tokens=0))
    return RunResult(
        goal="g",
        success=success,
        final_answer=answer,
        tool_calls=tool_calls,
        iterations=iterations,
        trace=trace,
    )


def test_answer_contains_matches_all_needles_case_insensitive():
    check = answer_contains("Denver", "warmer")
    assert check(make_result(answer="It is warmer in DENVER today"))
    assert not check(make_result(answer="It is colder in Tokyo"))


def test_answer_contains_requires_success():
    check = answer_contains("x")
    assert not check(make_result(success=False, answer="x"))
    assert not check(make_result(success=True, answer=None))


def _case(name, check):
    return EvalCase(name=name, goal=name, check=check)


def test_run_evals_reports_pass_fail_and_usage():
    answers = {
        "a": make_result(answer="hit", tokens=100),
        "b": make_result(answer="nope", tokens=50),
    }
    cases = [_case("a", answer_contains("hit")), _case("b", answer_contains("hit"))]
    report = run_evals(lambda goal: answers[goal], cases)

    assert report.total == 2
    assert report.ran == 2
    assert report.passed == 1
    assert not report.all_passed
    assert report.total_usage.input_tokens == 150


def test_run_evals_handles_run_fn_exception():
    def boom(goal):
        raise RuntimeError("kaboom")

    report = run_evals(boom, [_case("a", answer_contains("x"))])
    assert report.results[0].passed is False
    assert "kaboom" in report.results[0].error


def test_run_evals_handles_check_exception():
    def bad_check(result):
        raise ValueError("bad check")

    report = run_evals(lambda g: make_result(answer="x"), [_case("a", bad_check)])
    assert report.results[0].passed is False
    assert "check raised" in report.results[0].error


def test_token_budget_skips_remaining_cases():
    cases = [_case("a", answer_contains("x")), _case("b", answer_contains("x"))]
    report = run_evals(lambda g: make_result(answer="x", tokens=1000), cases, token_budget=500)
    # First case runs (and pushes spend over budget), second is skipped.
    assert report.results[0].skipped is False
    assert report.results[1].skipped is True
    assert report.results[1].stop_reason == "token budget exhausted"
    assert report.ran == 1


def test_delay_paces_between_cases():
    slept: list[float] = []
    cases = [_case("a", answer_contains("x")), _case("b", answer_contains("x"))]
    run_evals(
        lambda g: make_result(answer="x"),
        cases,
        delay_s=2.0,
        sleep=slept.append,
    )
    # One delay between the two cases (not before the first).
    assert slept == [2.0]
