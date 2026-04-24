"""
Tests for adaptive L2 patience.

Golden section and pattern search both scale their stopping tolerance to the
remaining budget ratio. A tight budget should cause them to bail out sooner
than a loose one on a flat landscape that never improves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.local_1d import golden_section_search
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParamSpec
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


def _flat(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    return _metrics(0.10)


def test_tight_budget_stops_golden_section_sooner() -> None:
    """A near-exhausted budget yields fewer probes on a flat landscape."""
    objective = SingleMetricObjective(metric="wer")
    spec = ParamSpec(name="x", type="float", default=0.5, min=0.0, max=1.0)

    def run(hard_cap: int, already_used: int) -> int:
        executor = SyntheticTrialExecutor(metric_fn=_flat, objective=objective)
        budget = BudgetController(hard_cap=hard_cap, convergence_eps=0.0)
        for _ in range(already_used):
            budget.record(0.10)
        result = golden_section_search(
            executor,
            baseline_config={"x": 0.5},
            param=spec,
            budget=budget,
            eps_min=0.0,
            phase="local_1d",
            max_iterations=8,
        )
        # Count only golden-section trials (excluding quadratic refine, if any).
        return len([t for t in result.trials if "golden-section" in (t.reasoning or "")])

    # Fresh budget (remaining_ratio = 1.0) → patience = 3
    fresh_trials = run(hard_cap=30, already_used=0)
    # Near-depleted budget (remaining_ratio ≈ 0.1) → patience = 1
    tight_trials = run(hard_cap=30, already_used=27)

    # On a flat landscape with patience 1, golden section bails out after
    # the first non-improving iteration — must use fewer probes than the
    # fresh-budget call.
    assert tight_trials < fresh_trials
