"""A small eval harness: fixed tasks with expected outcomes, plus token/request budgeting.

Each EvalCase pairs a goal with a check over the resulting RunResult. The runner is
decoupled from the orchestrator via a ``run_fn`` callable so it can be unit-tested without
any API key. ``main()`` wires the real Groq-backed orchestrator and prints a rich report.

    python examples/eval_harness.py
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.models import Usage
from orchestrator.state import RunResult

RunFn = Callable[[str], RunResult]
Check = Callable[[RunResult], bool]


# --------------------------------------------------------------------------- #
# Case definitions
# --------------------------------------------------------------------------- #
class EvalCase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    goal: str
    check: Check
    description: str = ""


def answer_contains(*needles: str, case_sensitive: bool = False) -> Check:
    """A check that passes when the run succeeded and the answer contains all needles."""

    def _check(result: RunResult) -> bool:
        if not result.success or not result.final_answer:
            return False
        answer = result.final_answer if case_sensitive else result.final_answer.lower()
        terms = needles if case_sensitive else [n.lower() for n in needles]
        return all(term in answer for term in terms)

    return _check


DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        name="calc_basic",
        goal="What is 1234 * 5678?",
        check=answer_contains("7006652"),
        description="Single calculator call with an exact numeric answer.",
    ),
    EvalCase(
        name="calc_compare",
        goal="Is 17 * 23 greater than 400? Answer yes or no.",
        check=answer_contains("yes"),
        description="Calculator plus a reasoning step (391 < 400 -> no... 17*23=391).",
    ),
    EvalCase(
        name="weather_city",
        goal="What is the current temperature in Denver?",
        check=answer_contains("denver"),
        description="Single weather tool call for a known city.",
    ),
    EvalCase(
        name="weather_compare",
        goal="Is it currently warmer in Denver or in Tokyo?",
        check=lambda r: r.success and r.tool_calls >= 2,
        description="Chains two weather calls and compares (>=2 tool calls).",
    ),
    EvalCase(
        name="wiki_lookup",
        goal="Who was Ada Lovelace? One sentence.",
        check=answer_contains("lovelace"),
        description="Single Wikipedia lookup.",
    ),
    EvalCase(
        name="mixed_chain",
        goal="Look up the year Alan Turing was born, then multiply it by 2.",
        check=lambda r: r.success and r.tool_calls >= 2,
        description="Wikipedia + calculator chained (>=2 tool calls).",
    ),
]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
class CaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    passed: bool
    success: bool
    stop_reason: str = ""
    final_answer: str | None = None
    iterations: int = 0
    tool_calls: int = 0
    elapsed_s: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    skipped: bool = False
    error: str | None = None


class EvalReport(BaseModel):
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ran(self) -> int:
        return sum(1 for r in self.results if not r.skipped)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def all_passed(self) -> bool:
        return self.ran > 0 and all(r.passed for r in self.results if not r.skipped)

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for r in self.results:
            total = total + r.usage
        return total


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_evals(
    run_fn: RunFn,
    cases: list[EvalCase] | None = None,
    *,
    token_budget: int | None = None,
    delay_s: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    on_result: Callable[[CaseResult], None] | None = None,
) -> EvalReport:
    """Run each case via ``run_fn`` and check its outcome.

    Stops spending once cumulative tokens would exceed ``token_budget`` (remaining cases are
    marked skipped) — a stand-in for respecting the free tier's daily/per-minute caps.
    """
    cases = cases or DEFAULT_CASES
    report = EvalReport()
    spent = 0

    for i, case in enumerate(cases):
        if token_budget is not None and spent >= token_budget:
            report.results.append(
                CaseResult(
                    name=case.name,
                    passed=False,
                    success=False,
                    skipped=True,
                    stop_reason="token budget exhausted",
                )
            )
            continue

        if i > 0 and delay_s > 0:
            sleep(delay_s)  # gentle inter-case pacing to stay under per-minute caps

        start = monotonic()
        try:
            result = run_fn(case.goal)
        except Exception as e:  # a crash in one case shouldn't abort the suite
            report.results.append(
                CaseResult(
                    name=case.name, passed=False, success=False, error=f"{type(e).__name__}: {e}"
                )
            )
            continue
        elapsed = monotonic() - start

        try:
            passed = bool(case.check(result))
            check_error = None
        except Exception as e:
            passed, check_error = False, f"check raised: {type(e).__name__}: {e}"

        spent += result.usage.total_tokens
        case_result = CaseResult(
            name=case.name,
            passed=passed,
            success=result.success,
            stop_reason=result.stop_reason,
            final_answer=result.final_answer,
            iterations=result.iterations,
            tool_calls=result.tool_calls,
            elapsed_s=round(elapsed, 3),
            usage=result.usage,
            error=check_error,
        )
        report.results.append(case_result)
        if on_result is not None:
            on_result(case_result)

    return report


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    from rich.console import Console
    from rich.table import Table

    from orchestrator.config import load_config
    from orchestrator.orchestrator import Orchestrator
    from tools.builtins import default_tools
    from tools.registry import ToolRegistry

    console = Console()
    config = load_config()
    if config.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]GROQ_API_KEY is not set.[/] Add it to .env and re-run.")
        return 2

    registry = ToolRegistry()
    registry.register_all(default_tools())
    orchestrator = Orchestrator.from_config(config, registry)

    console.print(f"[dim]Running {len(DEFAULT_CASES)} eval cases…[/]")
    report = run_evals(orchestrator.run, delay_s=2.0)

    table = Table(title="Eval results")
    for col in ("case", "pass", "tools", "iters", "tokens", "time", "answer"):
        table.add_column(col)
    for r in report.results:
        status = "skip" if r.skipped else ("[green]✓[/]" if r.passed else "[red]✗[/]")
        answer = (r.final_answer or r.stop_reason or "")[:50]
        table.add_row(
            r.name,
            status,
            str(r.tool_calls),
            str(r.iterations),
            str(r.usage.total_tokens),
            f"{r.elapsed_s:.1f}s",
            answer,
        )
    console.print(table)
    u = report.total_usage
    console.print(
        f"[bold]{report.passed}/{report.ran} passed[/] · "
        f"{u.input_tokens} in / {u.output_tokens} out tokens ({u.cached_tokens} cached)"
    )
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
