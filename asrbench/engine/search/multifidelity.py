"""
MultiFidelityTrialExecutor — Hyperband-style rung pruning for IAMS trials.

The idea: expensive trials are evaluated at successively larger fractions
of the corpus (25% → 50% → 100%). After each rung the score is compared
against the best full-fidelity trial seen so far plus a safety margin. If
the partial score is already clearly worse than the incumbent, we bail out
and return a pruned result — saving the cost of the remaining rungs.

The wrapper is transparent to search layers: it exposes the same
``evaluate(config, phase, reasoning)`` interface as any TrialExecutor. The
inner executor must provide an ``evaluate_at_fraction(..., fraction=...)``
method — both ``BenchmarkTrialExecutor`` and ``SyntheticTrialExecutor``
satisfy this contract.

Rules:
    - Screening (Layer 1) is gated in the IAMS orchestrator, not here.
      Screening is sensitive to noise so it should NOT be wrapped in a
      multi-fidelity executor; it still needs full-fidelity measurements.
    - Only Layer 2+ consumers should use this wrapper.
    - The wrapper updates its ``_best_score`` only on full-fidelity
      (fraction == 1.0) successes, so partial scores never poison the
      pruning threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from asrbench.engine.search.trial import TrialResult


@runtime_checkable
class FidelityAwareExecutor(Protocol):
    """Narrow protocol for executors that accept a fraction parameter."""

    def evaluate_at_fraction(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
        fraction: float = 1.0,
    ) -> TrialResult: ...

    def evaluate(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
    ) -> TrialResult: ...

    @property
    def runs_used(self) -> int: ...


@dataclass
class MultiFidelityTrialExecutor:
    """
    Wrap a TrialExecutor and run trials across multiple fidelity rungs.

    Each ``evaluate()`` call iterates ``rungs`` in order. After each non-final
    rung, if the partial score is worse than ``self._best_score +
    prune_threshold`` the trial is marked pruned and returned early — the
    caller still gets a TrialResult so the search layer sees a valid score,
    but its ``.pruned`` flag is True and the reasoning mentions where the cut
    happened.

    ``prune_threshold`` defaults to 0.015 (1.5 WER points), chosen so a trial
    that underperforms the incumbent by more than 1-2 WER points at an early
    rung is almost certainly not going to surface as the new best at full
    fidelity.
    """

    inner: FidelityAwareExecutor
    rungs: tuple[float, ...] = (0.25, 0.5, 1.0)
    prune_threshold: float = 0.015
    _best_score: float | None = field(default=None, init=False)

    @property
    def runs_used(self) -> int:
        return self.inner.runs_used

    def set_incumbent(self, score: float) -> None:
        """Seed the pruning threshold from the current search best."""
        if self._best_score is None or score < self._best_score:
            self._best_score = score

    def evaluate(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
    ) -> TrialResult:
        last: TrialResult | None = None
        for rung in self.rungs:
            trial = self.inner.evaluate_at_fraction(
                config, phase=phase, reasoning=reasoning, fraction=rung
            )
            last = trial
            if rung < 1.0 and self._best_score is not None:
                # Prune if the partial score is significantly worse than the
                # incumbent. The trial is returned with pruned=True so the
                # search layer accounts for it in the budget + trial log.
                if trial.score > self._best_score + self.prune_threshold:
                    pct = int(rung * 100)
                    pruned = TrialResult(
                        config=trial.config,
                        metrics=trial.metrics,
                        score=trial.score,
                        score_ci=trial.score_ci,
                        phase=trial.phase,
                        reasoning=(trial.reasoning or reasoning) + f" (pruned at {pct}%)",
                        trial_id=trial.trial_id,
                        pruned=True,
                    )
                    return pruned
        assert last is not None  # rungs is non-empty by construction
        if self._best_score is None or last.score < self._best_score:
            self._best_score = last.score
        return last

    def evaluate_at_fraction(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
        fraction: float = 1.0,
    ) -> TrialResult:
        """Bypass the multi-fidelity ladder when the caller wants a single rung."""
        return self.inner.evaluate_at_fraction(
            config, phase=phase, reasoning=reasoning, fraction=fraction
        )
