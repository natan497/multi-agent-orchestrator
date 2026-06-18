"""Unit tests for RunResult and the TPM-aware Pacer."""

from orchestrator.models import RunTrace, Usage
from orchestrator.state import Pacer, RunResult


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds  # advancing time simulates the wait


def make_pacer(tpm, clock):
    return Pacer(tpm, monotonic=clock.monotonic, sleep=clock.sleep)


def test_pacer_does_not_sleep_under_budget():
    clock = FakeClock()
    pacer = make_pacer(1000, clock)
    pacer.spend(500)
    pacer.spend(400)
    assert clock.slept == []


def test_pacer_sleeps_when_budget_exceeded():
    clock = FakeClock()
    pacer = make_pacer(1000, clock)
    clock.t = 10.0  # 10s into the window
    pacer.spend(1000)
    # Should sleep the remainder of the 60s window.
    assert clock.slept == [50.0]


def test_pacer_resets_after_window():
    clock = FakeClock()
    pacer = make_pacer(1000, clock)
    pacer.spend(900)
    clock.t = 61.0  # past the window
    pacer.spend(900)  # new window, should not sleep
    assert clock.slept == []


def test_pacer_disabled_when_tpm_non_positive():
    clock = FakeClock()
    pacer = make_pacer(0, clock)
    pacer.spend(10_000_000)
    assert clock.slept == []


def test_run_result_usage_proxies_trace():
    trace = RunTrace(goal="g")
    trace.record_usage(Usage(input_tokens=5, output_tokens=3))
    result = RunResult(goal="g", success=True, trace=trace)
    assert result.usage.total_tokens == 8
