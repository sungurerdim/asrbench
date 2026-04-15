"""
Integration test: IAMSOptimizer running on a MultiDatasetTrialExecutor.

The IAMS orchestrator is dataset-agnostic — it sees a single TrialExecutor
protocol. When that executor happens to be a MultiDatasetTrialExecutor, the
7-layer algorithm produces ONE config that maximizes the weighted aggregate
score across all datasets simultaneously.

This verifies the "global config" flow end-to-end: the optimizer converges
on a config that is better on average than the all-defaults baseline across
two synthetic datasets with conflicting optima.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asrbench.engine.optimizer import IAMSOptimizer
from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.multidataset import MultiDatasetTrialExecutor
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor


def _metrics(score: float) -> dict[str, float]:
    return {
        "wer": score,
        "cer": score * 0.5,
        "mer": score,
        "wil": score,
        "rtfx_mean": 20.0,
        "vram_peak_mb": 4000.0,
        "wer_ci_lower": max(0.0, score - 0.0005),
        "wer_ci_upper": score + 0.0005,
    }


def _landscape_a(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Dataset A: WER minimum at beam=3."""
    beam = int(cfg.get("beam_size", 5))
    return _metrics(0.10 + 0.003 * (beam - 3) ** 2)


def _landscape_b(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Dataset B: WER minimum at beam=7 (conflicts with A's optimum at 3)."""
    beam = int(cfg.get("beam_size", 5))
    return _metrics(0.12 + 0.003 * (beam - 7) ** 2)


def test_iams_on_multidataset_finds_compromise_config() -> None:
    """
    With two conflicting landscapes under uniform weights, the optimum is
    the midpoint beam=5 (where the weighted sum is minimized).

    A:   f_A(beam) = 0.10 + 0.003*(beam - 3)^2
    B:   f_B(beam) = 0.12 + 0.003*(beam - 7)^2
    Sum: 0.5*(f_A + f_B) = 0.11 + 0.003*((beam-3)^2 + (beam-7)^2)/2
    Minimum of that is at beam=5.
    """
    objective = SingleMetricObjective(metric="wer")
    inner_a = SyntheticTrialExecutor(metric_fn=_landscape_a, objective=objective)
    inner_b = SyntheticTrialExecutor(metric_fn=_landscape_b, objective=objective)
    multi = MultiDatasetTrialExecutor(executors=[inner_a, inner_b], labels=["A", "B"])

    space = ParameterSpace(
        parameters=(ParamSpec(name="beam_size", type="int", default=1, min=1, max=10),)
    )
    optimizer = IAMSOptimizer(
        executor=multi,  # type: ignore[arg-type]
        space=space,
        objective=objective,
        budget=BudgetController(hard_cap=50, convergence_eps=0.0),
        eps_min=0.0,
        mode="fast",  # Layers 1-2 only for speed
    )
    result = optimizer.run()

    # The global optimum for the weighted sum is at beam=5.
    assert result.best_config["beam_size"] == 5
    # And the best score should be better than the all-defaults baseline.
    baseline_trial = result.screening.baseline
    assert result.best_trial.score <= baseline_trial.score


def test_iams_on_multidataset_respects_weights() -> None:
    """
    Heavy-weight Dataset A (weight 0.9) pulls the optimum toward beam=3.
    Weighted aggregate minimum: d/dx [0.9*(x-3)^2 + 0.1*(x-7)^2] = 0
        → 0.9*(x-3) + 0.1*(x-7) = 0
        → x = 0.9*3 + 0.1*7 = 2.7 + 0.7 = 3.4 → integer rounding → 3
    """
    objective = SingleMetricObjective(metric="wer")
    inner_a = SyntheticTrialExecutor(metric_fn=_landscape_a, objective=objective)
    inner_b = SyntheticTrialExecutor(metric_fn=_landscape_b, objective=objective)
    multi = MultiDatasetTrialExecutor(
        executors=[inner_a, inner_b],
        labels=["A", "B"],
        weights=[0.9, 0.1],
    )

    space = ParameterSpace(
        parameters=(ParamSpec(name="beam_size", type="int", default=1, min=1, max=10),)
    )
    optimizer = IAMSOptimizer(
        executor=multi,  # type: ignore[arg-type]
        space=space,
        objective=objective,
        budget=BudgetController(hard_cap=50, convergence_eps=0.0),
        eps_min=0.0,
        mode="fast",
    )
    result = optimizer.run()

    # Weighted optimum rounds to beam=3 (A's optimum).
    assert result.best_config["beam_size"] == 3
