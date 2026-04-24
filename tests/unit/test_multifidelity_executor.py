"""
Tests for MultiFidelityTrialExecutor — rung pruning wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from asrbench.engine.search.multifidelity import MultiFidelityTrialExecutor
from asrbench.engine.search.trial import TrialResult


def _make_trial(score: float, fraction: float = 1.0) -> TrialResult:
    return TrialResult(
        config={"x": 1},
        metrics={"wer": score, "wer_ci_lower": score - 0.001, "wer_ci_upper": score + 0.001},
        score=score,
        score_ci=(score - 0.001, score + 0.001),
        phase="local_1d",
        reasoning=f"synthetic @ {fraction:.2f}",
    )


@dataclass
class _FakeExecutor:
    """Minimal inner executor that records fraction calls and returns fixed scores."""

    score_at: dict[float, float] = field(default_factory=dict)
    calls: list[float] = field(default_factory=list)

    @property
    def runs_used(self) -> int:
        return len(self.calls)

    def evaluate(self, config, *, phase="unknown", reasoning=""):
        self.calls.append(1.0)
        return _make_trial(self.score_at.get(1.0, 0.10), 1.0)

    def evaluate_at_fraction(self, config, *, phase="unknown", reasoning="", fraction=1.0):
        self.calls.append(fraction)
        return _make_trial(self.score_at.get(fraction, 0.10), fraction)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prunes_bad_trial_at_first_rung() -> None:
    """A partial score well above the incumbent triggers early pruning."""
    inner = _FakeExecutor(score_at={0.25: 0.25, 0.5: 0.25, 1.0: 0.25})
    mf = MultiFidelityTrialExecutor(inner=inner, rungs=(0.25, 0.5, 1.0), prune_threshold=0.015)
    mf.set_incumbent(0.10)
    trial = mf.evaluate({"x": 1}, phase="local_1d", reasoning="test")
    assert trial.pruned is True
    assert "pruned at 25%" in trial.reasoning
    # Only the first rung was consulted before the wrapper bailed out.
    assert inner.calls == [0.25]


def test_accepts_good_trial_through_all_rungs() -> None:
    """A trial that stays near the incumbent rides all three rungs."""
    inner = _FakeExecutor(score_at={0.25: 0.10, 0.5: 0.09, 1.0: 0.09})
    mf = MultiFidelityTrialExecutor(inner=inner, rungs=(0.25, 0.5, 1.0), prune_threshold=0.015)
    mf.set_incumbent(0.10)
    trial = mf.evaluate({"x": 1}, phase="local_1d", reasoning="")
    assert trial.pruned is False
    assert trial.score == 0.09
    assert inner.calls == [0.25, 0.5, 1.0]


def test_no_incumbent_means_no_pruning_yet() -> None:
    """Before the first full-fidelity success, we can't prune anything."""
    inner = _FakeExecutor(score_at={0.25: 0.50, 0.5: 0.50, 1.0: 0.50})
    mf = MultiFidelityTrialExecutor(inner=inner, rungs=(0.25, 0.5, 1.0))
    # No set_incumbent() call → all rungs run, best seeded by result.
    trial = mf.evaluate({"x": 1})
    assert trial.pruned is False
    assert inner.calls == [0.25, 0.5, 1.0]
    # The first full-fidelity score bootstrapped the best.
    assert mf._best_score == 0.50


def test_updates_best_on_full_fidelity_success() -> None:
    """Subsequent better full-fidelity trials tighten the prune threshold."""
    inner = _FakeExecutor(score_at={0.25: 0.10, 0.5: 0.08, 1.0: 0.08})
    mf = MultiFidelityTrialExecutor(inner=inner, rungs=(0.25, 0.5, 1.0), prune_threshold=0.015)
    mf.set_incumbent(0.15)
    mf.evaluate({"x": 1})
    # After a successful full-fidelity run at 0.08, best drops to 0.08.
    assert mf._best_score == 0.08
