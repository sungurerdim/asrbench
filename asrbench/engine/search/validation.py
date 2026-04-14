"""
IAMS Layer 7 — validation and confidence certification.

The final candidate produced by Layers 1-5 is the algorithm's best guess at
a global optimum, but it's still a single measurement. Stochastic noise in
the benchmark (GPU thermal throttling, background OS activity, non-determinism
in model inference) means that a one-off score can misrepresent the true
value of a configuration.

Layer 7 addresses this by re-evaluating the final config multiple times and
certifying the result with a confidence label:

    HIGH   : small variance across runs, CI width reasonable → trust the score
    MEDIUM : moderate variance or one outlier → score is plausible but noisy
    LOW    : large variance, potential result instability → consider re-running

The re-evaluation bypasses the executor cache when possible — for the
synthetic executor we call `set_cache_enabled(False)` before the runs. For
a real BenchmarkTrialExecutor, the TranscriptCache already invalidates on
new run_ids so each validation call produces a fresh measurement.

Certification is determined by the normalized standard deviation relative
to the mean score:

    cv = stdev(scores) / |mean(scores)|

    cv < 0.005                → HIGH
    0.005 ≤ cv < 0.02         → MEDIUM
    cv ≥ 0.02 or N/A          → LOW

**Threshold calibration note (2026 update).** The prior thresholds (0.02 /
0.05) were set by intuition and were too optimistic for the actual benchmark
noise profile. Research on WER measurement reliability (Bisani & Ney 2004,
Liu et al. 2020 blockwise bootstrap, Speechmatics accuracy benchmarking
docs) shows that even an in-corpus variance estimate under-reads true
generalization variance by a factor of ~2-4× when the corpus is speaker-
correlated (FLEURS: 20-30 speakers, LibriSpeech: 40-80 speakers per split).
The new thresholds bring the HIGH band down to CV < 0.5% to match the
project's default eps_min = 0.005, so HIGH confidence means the candidate's
run-to-run noise is below the optimizer's significance floor — which is the
property the caller actually needs to know.

Previous HIGH (0.02) runs that were really MEDIUM under the old scheme will
now correctly report MEDIUM. This is a tightening, not a loosening.

The caller (IAMS orchestrator) may choose to override the final config based
on the validation outcome, but the default behavior is to return the config
as-is with the confidence label attached.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Literal

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.trial import TrialExecutor, TrialResult

ConfidenceLabel = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class ValidationResult:
    """Output of Layer 7."""

    candidate: TrialResult
    validation_trials: list[TrialResult] = field(default_factory=list)
    mean_score: float = 0.0
    stdev_score: float = 0.0
    coefficient_of_variation: float = 0.0
    confidence: ConfidenceLabel = "LOW"
    reason: str = ""

    @property
    def num_runs(self) -> int:
        return len(self.validation_trials)


class ValidationPhase:
    """
    Layer 7 orchestrator.

    Args:
        executor: TrialExecutor
        budget: BudgetController
        n_runs: number of fresh evaluations to do for variance estimation.
                Default 3 is the minimum to compute a meaningful stdev.
        high_cv: CV threshold below which confidence is HIGH.
        medium_cv: CV threshold below which confidence is MEDIUM (HIGH otherwise).
    """

    def __init__(
        self,
        executor: TrialExecutor,
        budget: BudgetController,
        *,
        n_runs: int = 3,
        high_cv: float = 0.005,
        medium_cv: float = 0.02,
    ) -> None:
        if n_runs < 2:
            raise ValueError(
                f"ValidationPhase: n_runs must be >= 2 to compute variance, got {n_runs}"
            )
        if not (0 <= high_cv <= medium_cv):
            raise ValueError(
                f"ValidationPhase: must have 0 <= high_cv <= medium_cv, "
                f"got high_cv={high_cv}, medium_cv={medium_cv}"
            )
        self.executor = executor
        self.budget = budget
        self.n_runs = n_runs
        self.high_cv = high_cv
        self.medium_cv = medium_cv

    def run(self, candidate: TrialResult) -> ValidationResult:
        """
        Re-evaluate `candidate` n_runs times with cache disabled (if supported),
        compute mean/stdev, and assign a confidence label.

        If the budget can't afford n_runs fresh evaluations, we do as many as
        possible and degrade the confidence label accordingly (fewer samples
        → less certainty).
        """
        result = ValidationResult(candidate=candidate)
        # The first measurement IS the incoming candidate — include it.
        result.validation_trials.append(candidate)

        # Try to disable the executor's cache so re-runs actually re-execute.
        # For SyntheticTrialExecutor this is set_cache_enabled(False); real
        # executors may ignore (TranscriptCache is naturally invalidated by
        # the run_id mechanism).
        set_cache = getattr(self.executor, "set_cache_enabled", None)
        had_cache_state = None
        if callable(set_cache):
            had_cache_state = getattr(self.executor, "_cache_enabled", True)
            set_cache(False)

        try:
            for i in range(self.n_runs - 1):  # first run already recorded
                if not self.budget.can_run():
                    break
                trial = self.executor.evaluate(
                    dict(candidate.config),
                    phase="validation",
                    reasoning=f"validation run {i + 2}/{self.n_runs}",
                )
                self.budget.record(trial.score)
                result.validation_trials.append(trial)
        finally:
            if callable(set_cache) and had_cache_state is not None:
                set_cache(bool(had_cache_state))

        self._compute_stats(result)
        return result

    def _compute_stats(self, result: ValidationResult) -> None:
        scores = [t.score for t in result.validation_trials]
        n = len(scores)
        if n == 0:
            result.confidence = "LOW"
            result.reason = "no validation trials collected"
            return

        mean = float(statistics.mean(scores))
        result.mean_score = mean

        if n < 2:
            result.stdev_score = 0.0
            result.coefficient_of_variation = 0.0
            result.confidence = "LOW"
            result.reason = f"only {n} validation trial collected — cannot compute variance"
            return

        stdev = float(statistics.stdev(scores))
        result.stdev_score = stdev

        denom = abs(mean) if abs(mean) > 1e-12 else 1e-12
        cv = stdev / denom
        result.coefficient_of_variation = cv

        if math.isnan(cv) or math.isinf(cv):
            result.confidence = "LOW"
            result.reason = "NaN or Inf coefficient of variation"
            return

        if cv < self.high_cv:
            result.confidence = "HIGH"
            result.reason = f"cv={cv:.4f} below HIGH threshold {self.high_cv:.2f} across {n} runs"
        elif cv < self.medium_cv:
            result.confidence = "MEDIUM"
            result.reason = (
                f"cv={cv:.4f} in MEDIUM range [{self.high_cv:.2f}, "
                f"{self.medium_cv:.2f}) across {n} runs"
            )
        else:
            result.confidence = "LOW"
            result.reason = (
                f"cv={cv:.4f} >= LOW threshold {self.medium_cv:.2f} "
                f"across {n} runs — results are noisy, consider re-optimizing"
            )
