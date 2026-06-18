"""The plan-execute-observe loop.

Ties the planner, executor, and tool registry into a single run with hard guardrails
(max iterations, max tool calls, per-step retry cap), graceful error handling (provider
and parse failures end the run with a partial trace instead of crashing), and TPM-aware
pacing between provider calls. Every decision and tool I/O is appended to a RunTrace.
"""

from __future__ import annotations

from orchestrator.config import build_executor, build_planner
from orchestrator.executor import Executor
from orchestrator.models import (
    ControlDecision,
    Observation,
    OrchestratorConfig,
    Plan,
    RunTrace,
    StepStatus,
)
from orchestrator.planner import Planner, PlannerError
from orchestrator.state import Pacer, RunResult
from providers.base import ProviderError, RateLimitError
from tools.registry import ToolRegistry


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        registry: ToolRegistry,
        config: OrchestratorConfig,
        *,
        pacer: Pacer | None = None,
        pace_tpm: int = 6000,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.registry = registry
        self.config = config
        self.pacer = pacer if pacer is not None else Pacer(pace_tpm)

    @classmethod
    def from_config(
        cls, config: OrchestratorConfig, registry: ToolRegistry, **kwargs
    ) -> Orchestrator:
        """Wire planner + executor providers from config (Groq by default)."""
        return cls(
            Planner(build_planner(config)),
            Executor(build_executor(config)),
            registry,
            config,
            **kwargs,
        )

    def run(self, goal: str) -> RunResult:
        trace = RunTrace(goal=goal)
        specs = self.registry.specs()
        observations: list[Observation] = []
        iterations = 0
        tool_calls = 0
        retries_for_step = 0
        step_index = 0

        # --- Initial plan -------------------------------------------------- #
        try:
            plan_res = self.planner.make_plan(goal, specs)
        except (PlannerError, ProviderError) as e:
            trace.add("error", stage="plan", message=str(e))
            return RunResult(
                goal=goal, success=False, stop_reason=f"planning failed: {e}", trace=trace
            )
        plan = plan_res.plan
        self._account(trace, plan_res.usage)
        trace.add("plan", steps=[s.description for s in plan.steps])

        # --- Execute / observe / decide loop ------------------------------ #
        while step_index < len(plan.steps):
            if iterations >= self.config.max_iterations:
                return self._stopped(trace, goal, "reached max_iterations", iterations, tool_calls)
            iterations += 1
            step = plan.steps[step_index]
            step.status = StepStatus.RUNNING

            # Rate limits are fatal (retrying would just hit the cap again); other provider
            # errors (e.g. a model's malformed tool call) become a failed observation so the
            # planner can react — one bad step shouldn't abort the whole run.
            ex = None
            exec_error: ProviderError | None = None
            try:
                ex = self.executor.execute(goal, step, specs)
            except RateLimitError as e:
                trace.add("error", stage="execute", message=str(e))
                return self._stopped(trace, goal, f"rate limited: {e}", iterations, tool_calls)
            except ProviderError as e:
                trace.add("error", stage="execute", message=str(e))
                exec_error = e

            if exec_error is not None:
                observation = Observation(
                    step_index=step_index, ok=False, error=f"executor failed: {exec_error}"
                )
            else:
                self._account(trace, ex.usage)
                if ex.called_tool:
                    if tool_calls >= self.config.max_tool_calls:
                        return self._stopped(
                            trace, goal, "reached max_tool_calls", iterations, tool_calls
                        )
                    tool_calls += 1
                    trace.add("tool_call", name=ex.tool_call.name, arguments=ex.tool_call.arguments)
                    result = self.registry.dispatch(ex.tool_call)
                    observation = Observation(
                        step_index=step_index,
                        tool_call=ex.tool_call,
                        output=result.output,
                        ok=result.ok,
                        error=result.error,
                    )
                    trace.add("observation", ok=result.ok, output=result.output, error=result.error)
                else:
                    observation = Observation(step_index=step_index, output=ex.text or "", ok=True)
                    trace.add("executor_text", text=ex.text)
            observations.append(observation)

            try:
                decision_res = self.planner.decide(goal, plan, observation)
            except (PlannerError, ProviderError) as e:
                trace.add("error", stage="decide", message=str(e))
                return self._stopped(trace, goal, f"decision failed: {e}", iterations, tool_calls)
            self._account(trace, decision_res.usage)
            trace.add("decision", decision=str(decision_res.decision))

            decision = decision_res.decision
            if decision is ControlDecision.DONE:
                step.status = StepStatus.DONE
                answer = decision_res.final_answer or observation.output
                trace.final_answer = answer
                return RunResult(
                    goal=goal,
                    success=True,
                    final_answer=answer,
                    stop_reason="done",
                    iterations=iterations,
                    tool_calls=tool_calls,
                    trace=trace,
                )
            if decision is ControlDecision.RETRY:
                retries_for_step += 1
                step.status = StepStatus.PENDING
                if retries_for_step > self.config.max_retries_per_step:
                    trace.add("limit", which="max_retries_per_step", step=step_index)
                    return self._stopped(
                        trace,
                        goal,
                        f"step {step_index} exceeded retry limit",
                        iterations,
                        tool_calls,
                    )
                continue  # repeat the same step
            if decision is ControlDecision.REPLAN:
                plan = decision_res.plan or plan
                trace.add("replan", steps=[s.description for s in plan.steps])
                step_index = 0
                retries_for_step = 0
                continue
            # CONTINUE
            step.status = StepStatus.DONE
            step_index += 1
            retries_for_step = 0

        # --- Steps exhausted without an explicit 'done' -> synthesize ------ #
        try:
            fin = self.planner.finalize(goal, observations)
        except (PlannerError, ProviderError) as e:
            trace.add("error", stage="finalize", message=str(e))
            return self._stopped(trace, goal, f"finalize failed: {e}", iterations, tool_calls)
        self._account(trace, fin.usage)
        trace.final_answer = fin.final_answer
        trace.add("final", final_answer=fin.final_answer)
        return RunResult(
            goal=goal,
            success=True,
            final_answer=fin.final_answer,
            stop_reason="completed_all_steps",
            iterations=iterations,
            tool_calls=tool_calls,
            trace=trace,
        )

    # ------------------------------------------------------------------ #
    def _account(self, trace: RunTrace, usage) -> None:
        trace.record_usage(usage)
        self.pacer.spend(usage.total_tokens)

    @staticmethod
    def _stopped(
        trace: RunTrace, goal: str, reason: str, iterations: int, tool_calls: int
    ) -> RunResult:
        return RunResult(
            goal=goal,
            success=False,
            stop_reason=reason,
            iterations=iterations,
            tool_calls=tool_calls,
            trace=trace,
        )


__all__ = ["Orchestrator", "Plan"]
