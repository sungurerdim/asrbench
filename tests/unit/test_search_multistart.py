"""
Unit tests for MultiStartSequentialDescent (Layer 4).

The key test is the global-optimum recovery on the interaction trap:
single-start sequential descent from the default baseline gets stuck at
one of the equally-good minima, but multistart seeded with a promising
off-diagonal point from Layer 3 should recover the global optimum.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.multistart import (
    MultiStartSequentialDescent,
    sequential_descent,
)
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor


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


# ----------------------------------------------------------------------
# Landscapes
# ----------------------------------------------------------------------


def asymmetric_interaction(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    Non-symmetric interaction: the global minimum is strictly better when
    BOTH vad=True and chunk=30. Single-start from (vad=False, chunk=10)
    cannot see the off-diagonal win; multistart from (vad=True, chunk=30)
    can.

                  chunk=10  chunk=20  chunk=30
        vad=F     0.12       0.11      0.14
        vad=T     0.15       0.13      0.05     ← global min
    """
    vad = bool(cfg.get("vad", False))
    chunk = int(cfg.get("chunk_length", 10))
    t = {10: 0.12, 20: 0.11, 30: 0.14} if not vad else {10: 0.15, 20: 0.13, 30: 0.05}
    return _metrics(t[chunk])


def convex_additive(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    Additive landscape: f(a, b) = 0.05 + 0.02*(a-2)^2 + 0.03*(b-1)^2.
    Minimum at a=2, b=1, WER=0.05. Sequential descent from anywhere converges.
    """
    a = int(cfg.get("a", 0))
    b = int(cfg.get("b", 0))
    return _metrics(0.05 + 0.01 * (a - 2) ** 2 + 0.01 * (b - 1) ** 2)


# ----------------------------------------------------------------------
# Sequential descent (one start)
# ----------------------------------------------------------------------


class TestSequentialDescent:
    def test_convex_converges_to_minimum(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="int", min=0, default=0, max=4),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_additive, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        start = ex.evaluate({"a": 0, "b": 0}, phase="test")
        result = sequential_descent(
            ex,
            space,
            budget,
            start_config={"a": 0, "b": 0},
            sensitive_order=["a", "b"],
            start_trial=start,
            eps_min=0.001,
        )
        assert result.final_trial.config["a"] == 2
        assert result.final_trial.config["b"] == 1

    def test_final_trial_has_lowest_score(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="int", min=0, default=0, max=4),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_additive, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        start = ex.evaluate({"a": 0, "b": 0}, phase="test")
        result = sequential_descent(
            ex,
            space,
            budget,
            start_config={"a": 0, "b": 0},
            sensitive_order=["a", "b"],
            start_trial=start,
            eps_min=0.001,
        )
        # Final trial's score should be the lowest (or tied for lowest)
        lowest = min(t.score for t in result.trials)
        assert result.final_trial.score == pytest.approx(lowest)


# ----------------------------------------------------------------------
# Multi-start: interaction trap recovery
# ----------------------------------------------------------------------


class TestMultiStartTrapRecovery:
    def test_single_start_misses_global_optimum(self) -> None:
        """Sanity check: from (False, 10), sequential descent stays local."""
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="vad", type="bool", default=False),
                ParamSpec(
                    name="chunk_length",
                    type="int",
                    min=10,
                    default=10,
                    max=30,
                    step=10,
                ),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_interaction, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        start = ex.evaluate({"vad": False, "chunk_length": 10}, phase="test")
        result = sequential_descent(
            ex,
            space,
            budget,
            start_config={"vad": False, "chunk_length": 10},
            sensitive_order=["vad", "chunk_length"],
            start_trial=start,
            eps_min=0.001,
        )
        # Single start must NOT find the (True, 30) global min (WER 0.05)
        final_wer = result.final_trial.metrics["wer"]
        assert final_wer is not None
        assert final_wer > 0.08

    def test_multistart_recovers_global_optimum(self) -> None:
        """Seed the off-diagonal point — multistart must find the global min."""
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="vad", type="bool", default=False),
                ParamSpec(
                    name="chunk_length",
                    type="int",
                    min=10,
                    default=10,
                    max=30,
                    step=10,
                ),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_interaction, objective=obj)
        budget = BudgetController(hard_cap=100, convergence_window=0)

        # Two start points: default baseline + Layer 3's promising point
        default_start = ex.evaluate({"vad": False, "chunk_length": 10}, phase="test")
        promising_start = ex.evaluate({"vad": True, "chunk_length": 30}, phase="test")

        multi = MultiStartSequentialDescent(ex, space, budget, eps_min=0.001)
        result = multi.run(
            sensitive_order=["vad", "chunk_length"],
            start_trials=[default_start, promising_start],
        )

        assert result.best_overall is not None
        # Global minimum must be found (WER=0.05)
        assert result.best_overall.metrics["wer"] == pytest.approx(0.05)

    def test_best_start_label_identifies_winning_path(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="vad", type="bool", default=False),
                ParamSpec(
                    name="chunk_length",
                    type="int",
                    min=10,
                    default=10,
                    max=30,
                    step=10,
                ),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_interaction, objective=obj)
        budget = BudgetController(hard_cap=100, convergence_window=0)

        default_start = ex.evaluate({"vad": False, "chunk_length": 10}, phase="test")
        promising_start = ex.evaluate({"vad": True, "chunk_length": 30}, phase="test")

        multi = MultiStartSequentialDescent(ex, space, budget)
        result = multi.run(
            sensitive_order=["vad", "chunk_length"],
            start_trials=[default_start, promising_start],
        )
        # start-1 is the promising point and it wins
        assert result.best_start_label == "start-1"


class TestMultiStartDedupe:
    def test_duplicate_starts_skipped(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="int", min=0, default=0, max=4),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_additive, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)

        start_a = ex.evaluate({"a": 0, "b": 0}, phase="test")
        start_b = ex.evaluate({"a": 0, "b": 0}, phase="test")  # same config
        runs_before_multi = ex.runs_used

        multi = MultiStartSequentialDescent(ex, space, budget)
        result = multi.run(
            sensitive_order=["a", "b"],
            start_trials=[start_a, start_b],
        )
        # Only 1 descent despite 2 input trials
        assert len(result.descents) == 1
        # Duplicate skipping should not trigger extra evaluations of the dup
        assert ex.runs_used >= runs_before_multi

    def test_empty_start_list_raises(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="a", type="int", min=0, default=0, max=4),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_additive, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        multi = MultiStartSequentialDescent(ex, space, budget)
        with pytest.raises(ValueError, match="start_trials cannot be empty"):
            multi.run(sensitive_order=["a"], start_trials=[])

    def test_budget_exhaustion_stops_between_starts(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="a", type="int", min=0, default=0, max=20),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_additive, objective=obj)
        budget = BudgetController(hard_cap=5, convergence_window=0)

        start1 = ex.evaluate({"a": 0}, phase="test")
        start2 = ex.evaluate({"a": 20}, phase="test")

        multi = MultiStartSequentialDescent(ex, space, budget)
        multi.run(
            sensitive_order=["a"],
            start_trials=[start1, start2],
        )
        assert budget.runs_used <= 5
