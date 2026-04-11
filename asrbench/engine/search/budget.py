"""
BudgetController — stop condition for the IAMS optimizer.

Two stopping rules are applied in parallel:

    1. Hard cap: total trial count must never exceed `hard_cap`. This is an
       absolute circuit breaker so a user running `asrbench optimize --budget 100`
       can be certain the process terminates no later than 100 trials.

    2. Convergence window: if the best score seen so far has improved by less
       than `convergence_eps` over the last `convergence_window` trials, the
       search has "converged" and should stop early.

The search orchestrator asks the controller two questions per tick:

    budget.can_run()       -> False means "hard cap reached"
    budget.has_converged() -> True  means "stop now, no more improvement"

Either condition being true stops the current layer. Layers are allowed to
continue on to the next layer even after convergence — convergence inside
Layer 2 (local 1D search) just means "stop refining this parameter", not
"stop the whole optimizer". The orchestrator makes that call.

Callers must record each trial's score via `record(score)`. The controller
is oblivious to objective semantics — it assumes the score is a minimization
quantity (as all IAMS-visible scores are; see objective.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BudgetController:
    """
    Hard-cap + convergence-based stop controller.

    Parameters:
        hard_cap: maximum number of trials allowed before can_run() returns False.
        convergence_eps: required absolute improvement in best-so-far score to
            count as "progress". Defaults to 0.005 — the same epsilon used by
            the significance gate, so the two thresholds move together.
        convergence_window: number of consecutive non-improving trials before
            has_converged() returns True. Set to 0 or negative to disable the
            convergence stop entirely (only hard cap applies).

    State:
        history: list of best-so-far scores after each recorded trial.
        raw_history: list of individual trial scores (for debugging/reporting).
    """

    hard_cap: int
    convergence_eps: float = 0.005
    convergence_window: int = 3
    history: list[float] = field(default_factory=list)
    raw_history: list[float] = field(default_factory=list)
    _count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hard_cap <= 0:
            raise ValueError(
                f"BudgetController: hard_cap must be positive, got {self.hard_cap}. "
                "Use a positive integer cap to guarantee termination."
            )
        if self.convergence_eps < 0:
            raise ValueError(
                f"BudgetController: convergence_eps must be >= 0, got {self.convergence_eps}"
            )

    @property
    def runs_used(self) -> int:
        """Total trials recorded via record()."""
        return self._count

    @property
    def remaining(self) -> int:
        """How many more trials can still be run under the hard cap."""
        return max(0, self.hard_cap - self.runs_used)

    def can_run(self, n: int = 1) -> bool:
        """
        Can we afford N more trials without exceeding the hard cap?

        Default N=1 asks "is there room for one more call". Layers that want
        to batch multiple calls (e.g. screening's 2N boundary runs) can pass
        a larger N to check feasibility up-front.
        """
        return self.runs_used + n <= self.hard_cap

    def record(self, score: float) -> None:
        """
        Record a completed trial's score. Updates best-so-far history.

        Called exactly once per evaluated trial. NaN/inf scores are clamped
        to +inf so they never appear as "improvements".
        """
        if math.isnan(score) or math.isinf(score):
            score = math.inf
        self.raw_history.append(score)
        self._count += 1
        current_best = min(self.history[-1], score) if self.history else score
        self.history.append(current_best)

    def best_so_far(self) -> float | None:
        return self.history[-1] if self.history else None

    def has_converged(self) -> bool:
        """
        Has the best-so-far stopped improving?

        Returns True iff the difference between the best-so-far
        `convergence_window + 1` trials ago and the current best is less than
        `convergence_eps`. The +1 ensures we always have one "reference" point
        before the window.

        If convergence_window <= 0, convergence stopping is disabled → always False.
        """
        if self.convergence_window <= 0:
            return False
        needed = self.convergence_window + 1
        if len(self.history) < needed:
            return False
        reference = self.history[-needed]
        current = self.history[-1]
        improvement = reference - current  # positive = improvement under minimization
        return improvement < self.convergence_eps

    def should_stop(self) -> bool:
        """Convenience: can_run() is False OR has_converged() is True."""
        return (not self.can_run()) or self.has_converged()

    def summary(self) -> dict:
        """Structured snapshot for study.json logging."""
        return {
            "hard_cap": self.hard_cap,
            "runs_used": self.runs_used,
            "remaining": self.remaining,
            "convergence_eps": self.convergence_eps,
            "convergence_window": self.convergence_window,
            "best_so_far": self.best_so_far(),
            "has_converged": self.has_converged(),
        }
