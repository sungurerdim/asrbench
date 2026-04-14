"""Unit tests for ValidationPhase (Layer 7) — confidence certification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.trial import SyntheticTrialExecutor
from asrbench.engine.search.validation import ValidationPhase


def _metrics(wer: float) -> dict[str, float]:
    return {
        "wer": wer,
        "cer": wer * 0.5,
        "mer": wer,
        "wil": wer,
        "rtfx_mean": 20.0,
        "vram_peak_mb": 4000.0,
        "wer_ci_lower": max(0.0, wer - 0.0005),
        "wer_ci_upper": wer + 0.0005,
    }


def stable_landscape(_cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Noiseless: every call returns the same WER. Coefficient of variation = 0."""
    return _metrics(0.10)


# ----------------------------------------------------------------------


class TestValidationInit:
    def test_rejects_n_runs_less_than_2(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=stable_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        with pytest.raises(ValueError, match="n_runs must be >= 2"):
            ValidationPhase(ex, budget, n_runs=1)

    def test_rejects_inverted_cv_thresholds(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=stable_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        with pytest.raises(ValueError, match="high_cv <= medium_cv"):
            ValidationPhase(ex, budget, n_runs=3, high_cv=0.1, medium_cv=0.05)


class TestNoiselessValidation:
    def test_stable_landscape_gives_high_confidence(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=stable_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        result = ValidationPhase(ex, budget, n_runs=3).run(candidate)
        assert result.confidence == "HIGH"
        assert result.coefficient_of_variation == pytest.approx(0.0, abs=1e-9)
        assert result.num_runs == 3

    def test_zero_stdev_for_deterministic_executor(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=stable_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        result = ValidationPhase(ex, budget, n_runs=5).run(candidate)
        assert result.stdev_score == pytest.approx(0.0, abs=1e-9)


class TestNoisyValidation:
    """
    The synthetic executor's noise is config-seeded (reproducible per config),
    so calling it multiple times with the same config yields identical results.
    To test variance handling, we use metric_fn closures with a mutable call
    counter that return different values on each invocation — simulating real
    measurement variability between independent runs.
    """

    def test_high_noise_gives_low_confidence(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        call_count = [0]
        # Noise sequence chosen so stdev/mean ≈ 0.4 (well above 0.05 → LOW)
        noise_seq = [0.05, -0.04, 0.06, -0.03, 0.04]

        def noisy(_cfg):
            idx = call_count[0] % len(noise_seq)
            call_count[0] += 1
            return _metrics(0.10 + noise_seq[idx])

        ex = SyntheticTrialExecutor(metric_fn=noisy, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        result = ValidationPhase(ex, budget, n_runs=5).run(candidate)
        assert result.confidence == "LOW"

    def test_moderate_noise_gives_medium_confidence(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        call_count = [0]
        # Small noise: stdev ≈ 0.003 on mean 0.10 → cv ≈ 0.03 → MEDIUM under
        # LEGACY thresholds (0.02 / 0.05). This test uses the legacy override
        # to document the pre-2026 behavior and lock in that it still works
        # when callers opt into the old band.
        noise_seq = [0.004, -0.003, 0.003, -0.004, 0.002]

        def noisy(_cfg):
            idx = call_count[0] % len(noise_seq)
            call_count[0] += 1
            return _metrics(0.10 + noise_seq[idx])

        ex = SyntheticTrialExecutor(metric_fn=noisy, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        result = ValidationPhase(ex, budget, n_runs=5, high_cv=0.02, medium_cv=0.05).run(candidate)
        assert result.confidence == "MEDIUM"

    def test_default_thresholds_are_noise_floor_calibrated(self) -> None:
        """
        Regression lock for the 2026 threshold recalibration.

        Using the default thresholds (no override), a corpus-noise level that
        would have been "MEDIUM" under the old 0.02/0.05 bands must now report
        "LOW" — the tightened gate demands that HIGH confidence implies the
        run-to-run noise is below the optimizer's significance floor
        (eps_min = 0.005, matching high_cv).
        """
        obj = SingleMetricObjective(metric="wer")
        call_count = [0]
        # Same noise as the legacy test: cv ≈ 3% → used to map to MEDIUM.
        noise_seq = [0.004, -0.003, 0.003, -0.004, 0.002]

        def noisy(_cfg):
            idx = call_count[0] % len(noise_seq)
            call_count[0] += 1
            return _metrics(0.10 + noise_seq[idx])

        ex = SyntheticTrialExecutor(metric_fn=noisy, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        # No high_cv/medium_cv override → uses the new 0.005/0.02 defaults
        result = ValidationPhase(ex, budget, n_runs=5).run(candidate)
        assert result.confidence == "LOW", (
            f"With defaults (high_cv=0.005, medium_cv=0.02), cv≈0.03 must be "
            f"LOW (was MEDIUM under legacy bands). Got {result.confidence} "
            f"with cv={result.coefficient_of_variation:.4f}"
        )

    def test_very_low_noise_still_reaches_high(self) -> None:
        """
        A near-zero-noise run still earns HIGH under the tightened defaults.
        Sanity check that we did not accidentally make HIGH unreachable.
        """
        obj = SingleMetricObjective(metric="wer")
        call_count = [0]
        # cv ≈ 0.002, well below the new high_cv = 0.005 floor
        noise_seq = [0.0002, -0.0001, 0.0003, -0.0002, 0.0001]

        def very_quiet(_cfg):
            idx = call_count[0] % len(noise_seq)
            call_count[0] += 1
            return _metrics(0.10 + noise_seq[idx])

        ex = SyntheticTrialExecutor(metric_fn=very_quiet, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        result = ValidationPhase(ex, budget, n_runs=5).run(candidate)
        assert result.confidence == "HIGH"
        assert result.coefficient_of_variation < 0.005


class TestBudgetAware:
    def test_budget_limits_validation_runs(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=stable_landscape, objective=obj)
        # Only 2 runs affordable — ValidationPhase should cap at that many
        budget = BudgetController(hard_cap=2, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        result = ValidationPhase(ex, budget, n_runs=5).run(candidate)
        # The candidate run (pre-existing) + at most 2 new runs = 3 total
        assert result.num_runs <= 3
        assert budget.runs_used <= 2


class TestCacheDisabled:
    def test_cache_disabled_during_validation(self) -> None:
        """Each validation run must produce a fresh evaluation, not cache hit."""
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=stable_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        # Pre-evaluate the candidate
        candidate = ex.evaluate({"x": 1}, phase="test")
        runs_before = ex.runs_used

        ValidationPhase(ex, budget, n_runs=4).run(candidate)
        # 3 new runs (n_runs - 1, since the candidate counts as the first)
        assert ex.runs_used == runs_before + 3

    def test_cache_state_restored_after_validation(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=stable_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        candidate = ex.evaluate({"x": 1}, phase="test")
        # Cache should be on by default
        assert ex._cache_enabled is True
        ValidationPhase(ex, budget, n_runs=3).run(candidate)
        # Cache should be re-enabled after validation
        assert ex._cache_enabled is True
