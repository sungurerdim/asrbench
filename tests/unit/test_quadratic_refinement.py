"""
Tests for _quadratic_refine — curvature-aware extra probe at the end of
golden-section / pattern-search 1D loops.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.local_1d import _quadratic_refine, golden_section_search
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor, TrialResult


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


def _parabolic(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """y = (x - 0.3)^2 + 0.1 — analytic minimum at x=0.3, y=0.1."""
    x = float(cfg["x"])
    return _metrics(0.1 + (x - 0.3) ** 2)


def test_quadratic_refine_finds_analytic_minimum() -> None:
    """Hand-build three equispaced probes and verify the vertex is probed."""
    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(metric_fn=_parabolic, objective=objective)
    space = ParameterSpace(
        parameters=(ParamSpec(name="x", type="float", default=0.5, min=0.0, max=1.0),)
    )
    spec = space.get("x")
    budget = BudgetController(hard_cap=20, convergence_eps=0.0)

    # Seed trials manually — three probes at 0.0, 0.5, 1.0.
    baseline = {"x": 0.5}
    seed_trials: list[TrialResult] = []
    for xv in (0.0, 0.5, 1.0):
        seed_trials.append(executor.evaluate({"x": xv}, phase="seed", reasoning=f"seed x={xv}"))
        budget.record(seed_trials[-1].score)

    refined = _quadratic_refine(seed_trials, baseline, spec, executor, budget, phase="local_1d")
    assert refined is not None
    assert abs(float(refined.config["x"]) - 0.3) < 1e-6
    # And the new probe is a genuine improvement over the seeds.
    assert refined.score < min(t.score for t in seed_trials)


def test_quadratic_refine_skips_concave() -> None:
    """A concave-up fit (a < 0) should produce no refinement probe."""

    def concave(cfg: Mapping[str, Any]) -> Mapping[str, float]:
        x = float(cfg["x"])
        # -x^2 + 0.5 → maximum at 0, not minimum
        return _metrics(0.5 - x * x)

    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(metric_fn=concave, objective=objective)
    spec = ParamSpec(name="x", type="float", default=0.0, min=-1.0, max=1.0)
    budget = BudgetController(hard_cap=10, convergence_eps=0.0)

    trials: list[TrialResult] = []
    for xv in (-1.0, 0.0, 1.0):
        t = executor.evaluate({"x": xv}, phase="seed", reasoning="seed")
        budget.record(t.score)
        trials.append(t)

    refined = _quadratic_refine(trials, {"x": 0.0}, spec, executor, budget, phase="local_1d")
    assert refined is None


def test_quadratic_refine_too_close_to_probe_skips() -> None:
    """Vertex within 1% of an existing probe → skip (no new info)."""
    objective = SingleMetricObjective(metric="wer")
    # Parabola with minimum at ~0.5, which is exactly one of our probes.
    executor = SyntheticTrialExecutor(
        metric_fn=lambda cfg: _metrics(0.1 + (float(cfg["x"]) - 0.5) ** 2),
        objective=objective,
    )
    spec = ParamSpec(name="x", type="float", default=0.5, min=0.0, max=1.0)
    budget = BudgetController(hard_cap=10, convergence_eps=0.0)

    trials: list[TrialResult] = []
    for xv in (0.0, 0.5, 1.0):
        t = executor.evaluate({"x": xv}, phase="seed", reasoning="seed")
        budget.record(t.score)
        trials.append(t)

    refined = _quadratic_refine(trials, {"x": 0.5}, spec, executor, budget, phase="local_1d")
    # Vertex is 0.5, matches the middle probe exactly → skip.
    assert refined is None


def test_golden_section_chains_quadratic_refinement() -> None:
    """
    End-to-end: golden section followed by quadratic refinement.
    The final best should carry the 'local_1d-quad' phase tag if quadratic
    refinement actually fired (or stay at 'local_1d' if it was skipped).
    """
    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(metric_fn=_parabolic, objective=objective)
    spec = ParamSpec(name="x", type="float", default=0.5, min=0.0, max=1.0)
    budget = BudgetController(hard_cap=30, convergence_eps=0.0)

    result = golden_section_search(
        executor,
        baseline_config={"x": 0.5},
        param=spec,
        budget=budget,
        eps_min=0.0,
        phase="local_1d",
    )
    # At least one trial must exist.
    assert result.trials
    # The best score must be near the analytic optimum (0.1).
    assert result.best.score < 0.12
    # Check whether the refinement actually fired — the phase tag is
    # appended with '-quad' when a refinement probe landed in the list.
    refinement_trials = [t for t in result.trials if t.phase == "local_1d-quad"]
    # Not guaranteed to fire (golden section may converge to the vertex
    # already), but if it did, the best MUST reflect it or be no worse.
    if refinement_trials:
        assert any(t.score <= result.best.score for t in refinement_trials)
