"""
Tests for prior-informed screening order.

When the caller passes ``prior_sensitivity_hints`` to ScreeningPhase, the
boundary probes must be issued in descending hint order so a tight budget
still covers the highest-leverage parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.screening import ScreeningPhase
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


def _flat(cfg: Mapping[str, Any]) -> Mapping[str, float]:  # noqa: ARG001
    return _metrics(0.10)


def _make_space() -> ParameterSpace:
    return ParameterSpace(
        parameters=(
            ParamSpec(name="a", type="int", default=1, min=1, max=5),
            ParamSpec(name="b", type="int", default=1, min=1, max=5),
            ParamSpec(name="c", type="int", default=1, min=1, max=5),
            ParamSpec(name="d", type="int", default=1, min=1, max=5),
        )
    )


def test_hints_reorder_screening_probes() -> None:
    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(metric_fn=_flat, objective=objective)
    space = _make_space()
    budget = BudgetController(hard_cap=50, convergence_eps=0.0)
    hints = {"c": 1.0, "a": 0.7, "b": 0.1}  # d is missing → treated as 0
    phase = ScreeningPhase(executor, space, budget, eps_min=0.0, prior_sensitivity_hints=hints)
    result = phase.run()

    # Verify the order of boundary probes. Baseline trial is first; then
    # each parameter's min then max probe. We extract the param name from
    # the reasoning string ("boundary min for 'NAME'"), preserving order.
    probe_params: list[str] = []
    for t in result.trials[1:]:  # skip baseline
        reason = t.reasoning or ""
        # Parse: "boundary min for 'a' (a=1)"
        if " for '" in reason:
            start = reason.find(" for '") + len(" for '")
            end = reason.find("'", start)
            name = reason[start:end]
            if not probe_params or probe_params[-1] != name:
                probe_params.append(name)

    # Expected order: c (hint 1.0) → a (0.7) → b (0.1) → d (0.0)
    assert probe_params == ["c", "a", "b", "d"]


def test_no_hints_preserves_declaration_order() -> None:
    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(metric_fn=_flat, objective=objective)
    space = _make_space()
    budget = BudgetController(hard_cap=50, convergence_eps=0.0)
    phase = ScreeningPhase(executor, space, budget, eps_min=0.0)
    result = phase.run()

    probe_params: list[str] = []
    for t in result.trials[1:]:
        reason = t.reasoning or ""
        if " for '" in reason:
            start = reason.find(" for '") + len(" for '")
            end = reason.find("'", start)
            name = reason[start:end]
            if not probe_params or probe_params[-1] != name:
                probe_params.append(name)

    assert probe_params == ["a", "b", "c", "d"]


def test_budget_constrained_covers_top_hints() -> None:
    """With budget for only ~2 params, the 2 highest hints must be screened."""
    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(metric_fn=_flat, objective=objective)
    space = _make_space()
    # baseline = 1 trial, each param = 2 trials → budget for 2 params = 5
    budget = BudgetController(hard_cap=5, convergence_eps=0.0)
    hints = {"d": 1.0, "b": 0.9, "a": 0.1, "c": 0.0}
    phase = ScreeningPhase(executor, space, budget, eps_min=0.0, prior_sensitivity_hints=hints)
    result = phase.run()

    # Only the top two (d, b) should have full boundary probes; the rest
    # land in unscreened[]. The order inside params reflects completion.
    assert set(result.params.keys()) == {"d", "b"}
    assert "a" in result.unscreened
    assert "c" in result.unscreened
