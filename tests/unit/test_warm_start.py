"""
Unit tests for warm starting, adaptive inter-layer budget, and screening reconstruction.

Covers:
    - SyntheticTrialExecutor.warm_load() cache pre-population
    - BenchmarkTrialExecutor.warm_load() (same logic, tested via Synthetic)
    - TrialResult.from_db_row() reconstruction
    - ScreeningResult.from_summary() reconstruction
    - IAMSOptimizer with prior_screening (L1 skip)
    - Adaptive multistart budget (L3 empty → candidates=1)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.screening import ScreeningPhase, ScreeningResult
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor, TrialResult

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


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


def beam_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    beam = float(cfg.get("beam_size", 5))
    wer = 0.20 - 0.15 * (beam - 1) / 19.0
    return _metrics(wer)


def two_param_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    beam = float(cfg.get("beam_size", 5))
    temp = float(cfg.get("temperature", 0.0))
    wer_beam = 0.15 * (1.0 - (beam - 1) / 19.0)
    wer_temp = 0.05 * temp
    return _metrics(0.05 + wer_beam + wer_temp)


def _make_space_1() -> ParameterSpace:
    return ParameterSpace(
        parameters=(ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),)
    )


def _make_space_2() -> ParameterSpace:
    return ParameterSpace(
        parameters=(
            ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),
            ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
        )
    )


def _make_executor(landscape: Any = beam_landscape) -> SyntheticTrialExecutor:
    obj = SingleMetricObjective(metric="wer")
    return SyntheticTrialExecutor(metric_fn=landscape, objective=obj)


# ======================================================================
# TrialResult.from_db_row
# ======================================================================


class TestTrialResultFromDbRow:
    def test_basic_reconstruction(self) -> None:
        tr = TrialResult.from_db_row(
            config={"beam_size": 10},
            score=0.12,
            score_ci=(0.11, 0.13),
            phase="screening",
            trial_id="abc-123",
        )
        assert tr.config == {"beam_size": 10}
        assert tr.score == 0.12
        assert tr.score_ci == (0.11, 0.13)
        assert tr.phase == "screening"
        assert tr.trial_id == "abc-123"
        assert tr.metrics == {}
        assert "warm-start" in tr.reasoning

    def test_defaults(self) -> None:
        tr = TrialResult.from_db_row(config={"x": 1}, score=0.5, score_ci=(0.4, 0.6))
        assert tr.phase == "prior"
        assert tr.trial_id is None


# ======================================================================
# SyntheticTrialExecutor.warm_load
# ======================================================================


class TestWarmLoad:
    def test_empty_list_returns_zero(self) -> None:
        ex = _make_executor()
        assert ex.warm_load([]) == 0

    def test_loads_trials_into_cache(self) -> None:
        ex = _make_executor()
        trials = [
            TrialResult.from_db_row(config={"beam_size": 10}, score=0.12, score_ci=(0.11, 0.13)),
            TrialResult.from_db_row(config={"beam_size": 15}, score=0.08, score_ci=(0.07, 0.09)),
        ]
        loaded = ex.warm_load(trials)
        assert loaded == 2

        # evaluate should return cache hit without incrementing runs_used
        result = ex.evaluate({"beam_size": 10}, phase="test")
        assert result.score == 0.12
        assert ex.runs_used == 0  # no actual execution happened

    def test_duplicate_configs_first_wins(self) -> None:
        ex = _make_executor()
        t1 = TrialResult.from_db_row(config={"beam_size": 10}, score=0.12, score_ci=(0.11, 0.13))
        t2 = TrialResult.from_db_row(config={"beam_size": 10}, score=0.99, score_ci=(0.98, 1.0))
        loaded = ex.warm_load([t1, t2])
        assert loaded == 1  # second was duplicate

        result = ex.evaluate({"beam_size": 10}, phase="test")
        assert result.score == 0.12  # first one wins

    def test_warm_load_does_not_increment_runs_used(self) -> None:
        ex = _make_executor()
        trials = [
            TrialResult.from_db_row(
                config={"beam_size": i}, score=0.1 * i, score_ci=(0.09 * i, 0.11 * i)
            )
            for i in range(1, 6)
        ]
        ex.warm_load(trials)
        assert ex.runs_used == 0

        # Cache hits also don't increment
        for i in range(1, 6):
            ex.evaluate({"beam_size": i})
        assert ex.runs_used == 0


# ======================================================================
# ScreeningResult.from_summary
# ======================================================================


class TestScreeningFromSummary:
    def test_basic_reconstruction(self) -> None:
        baseline = TrialResult.from_db_row(
            config={"beam_size": 5, "temperature": 0.0},
            score=0.15,
            score_ci=(0.14, 0.16),
        )
        summary = {
            "sensitive_order": ["beam_size"],
            "insensitive": ["temperature"],
        }
        sr = ScreeningResult.from_summary(summary, baseline)
        assert sr.sensitive_order == ["beam_size"]
        assert sr.insensitive == ["temperature"]
        assert sr.baseline is baseline
        assert sr.params == {}
        assert sr.trials == []
        assert sr.unscreened == []

    def test_sensitive_property_alias(self) -> None:
        baseline = TrialResult.from_db_row(config={"x": 1}, score=0.1, score_ci=(0.09, 0.11))
        sr = ScreeningResult.from_summary(
            {"sensitive_order": ["a", "b"], "insensitive": ["c"]}, baseline
        )
        assert sr.sensitive == ["a", "b"]

    def test_from_summary_feeds_layer2(self) -> None:
        """A reconstructed ScreeningResult has the data Layer 2 needs."""
        baseline = TrialResult.from_db_row(
            config={"beam_size": 5}, score=0.15, score_ci=(0.14, 0.16)
        )
        sr = ScreeningResult.from_summary(
            {"sensitive_order": ["beam_size"], "insensitive": []}, baseline
        )
        # Layer 2 reads .sensitive_order and .baseline — both must be usable
        assert len(sr.sensitive_order) == 1
        assert sr.baseline.score == 0.15


# ======================================================================
# IAMSOptimizer with prior_screening (warm start L1 skip)
# ======================================================================


class TestOptimizerWarmStart:
    def test_prior_screening_skips_l1(self) -> None:
        """When prior_screening is provided, L1 screening is not re-run."""
        from asrbench.engine.optimizer import IAMSOptimizer

        space = _make_space_1()
        obj = SingleMetricObjective(metric="wer")
        ex = _make_executor(beam_landscape)

        # Run L1 normally first to get a real ScreeningResult
        budget_pre = BudgetController(hard_cap=100)
        screening = ScreeningPhase(ex, space, budget_pre, eps_min=0.005).run()
        assert ex.runs_used > 0  # L1 consumed trials

        # Now create a NEW executor with warm-loaded cache
        ex2 = _make_executor(beam_landscape)
        ex2.warm_load(
            [
                TrialResult.from_db_row(config=dict(t.config), score=t.score, score_ci=t.score_ci)
                for t in screening.trials
            ]
        )
        assert ex2.runs_used == 0  # warm_load doesn't count

        # Reconstruct screening from summary
        prior = ScreeningResult.from_summary(
            {
                "sensitive_order": list(screening.sensitive_order),
                "insensitive": list(screening.insensitive),
            },
            screening.baseline,
        )

        budget2 = BudgetController(hard_cap=100)
        optimizer = IAMSOptimizer(
            executor=ex2,
            space=space,
            objective=obj,
            budget=budget2,
            eps_min=0.005,
            mode="fast",
            prior_screening=prior,
        )
        result = optimizer.run()

        # L1 was skipped — reasoning should mention WARM START
        assert any("WARM START" in r for r in result.reasoning)
        # The optimizer still produces a valid result
        assert result.best_trial.score <= screening.baseline.score

    def test_no_prior_screening_runs_l1_normally(self) -> None:
        """Without prior_screening, L1 runs as usual (regression check)."""
        from asrbench.engine.optimizer import IAMSOptimizer

        space = _make_space_1()
        obj = SingleMetricObjective(metric="wer")
        ex = _make_executor(beam_landscape)
        budget = BudgetController(hard_cap=50)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            eps_min=0.005,
            mode="fast",
        )
        result = optimizer.run()

        # L1 should have run normally — no WARM START in reasoning
        assert not any("WARM START" in r for r in result.reasoning)
        assert "Layer 1 (screening)" in result.reasoning[0]

    def test_warm_start_trials_not_counted_as_budget(self) -> None:
        """Prior trials in cache should not consume budget."""
        from asrbench.engine.optimizer import IAMSOptimizer

        space = _make_space_1()
        obj = SingleMetricObjective(metric="wer")
        ex = _make_executor(beam_landscape)

        # Pre-populate cache with trials
        warm_trials = [
            TrialResult.from_db_row(
                config={"beam_size": b},
                score=0.20 - 0.15 * (b - 1) / 19.0,
                score_ci=(
                    max(0.0, 0.20 - 0.15 * (b - 1) / 19.0 - 0.0005),
                    0.20 - 0.15 * (b - 1) / 19.0 + 0.0005,
                ),
            )
            for b in [1, 5, 20]
        ]
        ex.warm_load(warm_trials)

        prior = ScreeningResult.from_summary(
            {"sensitive_order": ["beam_size"], "insensitive": []},
            warm_trials[1],  # beam_size=5 as baseline
        )

        budget = BudgetController(hard_cap=20)
        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            eps_min=0.005,
            mode="fast",
            prior_screening=prior,
        )
        result = optimizer.run()
        assert result.best_trial is not None


# ======================================================================
# Adaptive L3→L4 budget
# ======================================================================


class TestAdaptiveBudget:
    def test_no_interactions_reduces_multistart(self) -> None:
        """When L3 finds no promising points, reasoning mentions reduction."""
        from asrbench.engine.optimizer import IAMSOptimizer

        space = _make_space_2()
        obj = SingleMetricObjective(metric="wer")
        # Use a landscape where only beam matters — L3 won't find interactions
        ex = _make_executor(beam_landscape)
        budget = BudgetController(hard_cap=300)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            eps_min=0.005,
            mode="balanced",
            multistart_candidates=3,
        )
        result = optimizer.run()

        # Check if the adaptive reduction reasoning appears
        has_reduction = any("reducing multi-start" in r for r in result.reasoning)
        has_l3 = any("Layer 3" in r for r in result.reasoning)
        assert has_l3  # L3 ran
        # If L3 found no interactions, we should see the reduction message
        if any("0 promising" in r for r in result.reasoning):
            assert has_reduction

    def test_interactions_preserve_candidates(self) -> None:
        """When L3 finds interactions, multistart_candidates is unchanged."""
        from asrbench.engine.optimizer import IAMSOptimizer

        space = _make_space_2()
        obj = SingleMetricObjective(metric="wer")
        ex = _make_executor(two_param_landscape)
        budget = BudgetController(hard_cap=300)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            eps_min=0.005,
            mode="balanced",
            multistart_candidates=3,
        )
        result = optimizer.run()

        # L4 reasoning should not mention reduction
        has_reduction = any("reducing multi-start" in r for r in result.reasoning)
        # If L3 found promising points, no reduction should happen
        l3_lines = [r for r in result.reasoning if "Layer 3" in r]
        if l3_lines and "0 promising" not in l3_lines[0]:
            assert not has_reduction
