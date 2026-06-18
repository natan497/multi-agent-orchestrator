"""Run result type and the TPM-aware pacer.

``RunResult`` is the orchestrator's return value. ``Pacer`` implements the token-aware
delay from SPEC §5: it tracks tokens spent in a rolling one-minute window and sleeps just
long enough to avoid tripping the per-minute token cap, rather than using a fixed sleep.
Clock and sleep are injectable so the loop is deterministic under test.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.models import RunTrace, Usage


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    goal: str
    success: bool
    final_answer: str | None = None
    stop_reason: str = ""
    iterations: int = 0
    tool_calls: int = 0
    trace: RunTrace = Field(default_factory=lambda: RunTrace(goal=""))

    @property
    def usage(self) -> Usage:
        return self.trace.usage


class Pacer:
    """Token-aware throttle for a single per-minute token budget.

    Call :meth:`spend` after each provider call with the tokens consumed; it records them
    in the current 60s window and sleeps until the window rolls over if the budget is
    exhausted. ``tpm <= 0`` disables pacing entirely.
    """

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        tpm: int,
        *,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time

        self.tpm = tpm
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._window_start = self._monotonic()
        self._spent = 0

    def spend(self, tokens: int) -> None:
        if self.tpm <= 0:
            return
        now = self._monotonic()
        if now - self._window_start >= self.WINDOW_SECONDS:
            self._window_start = now
            self._spent = 0
        self._spent += max(0, tokens)
        if self._spent >= self.tpm:
            remaining = self.WINDOW_SECONDS - (now - self._window_start)
            if remaining > 0:
                self._sleep(remaining)
            self._window_start = self._monotonic()
            self._spent = 0
