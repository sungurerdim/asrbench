"""
Unit tests for DeepAblation (Layer 5).

Tests cover:
    - Leave-one-out detection of toxic parameters
    - Refinement builder reverts all toxic params together
    - Fallback guarantee: never worse than incoming best
    - Leave-two-out pair ablation (enabled)
    - Budget-limited early termination
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.search.ablation import DeepAblation
from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor, TrialResult


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


def _build_incoming(executor: SyntheticTrialExecutor, config: dict[str, Any]) -> TrialResult:
    """Evaluate an explicit config and mark it as 'incoming best' for ablation."""
    return executor.evaluate(config, phase="multistart", reasoning="incoming best")


# ----------------------------------------------------------------------
# Landscapes
# ----------------------------------------------------------------------


def toxic_param_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    Landscape where parameter 'b' is TOXIC but 'a' is legitimately optimal at 2.

    WER(a, b) = 0.05 + 0.02 * (a - 2)^2 + (0.06 if b else 0.0)

    Analytical results relative to incoming best (a=2, b=True, WER=0.11):
        - revert a to default 0: 0.05 + 0.08 + 0.06 = 0.19 (WORSE — a is not toxic)
        - revert b to default False: 0.05 + 0 + 0 = 0.05 (BETTER — b is toxic)
        - revert both: 0.05 + 0.08 + 0 = 0.13 (better than incoming but not optimal)

    Layer 5 must detect that only 'b' is toxic, and the refinement must preserve a=2.
    Because LOO alone would pick `(a=2, b=False)` from the LOO trial list, the
    ablation layer's `final()` should end up at WER=0.05 even though the
    blanket-revert refinement is suboptimal (0.13).
    """
    a = int(cfg.get("a", 0))
    b = bool(cfg.get("b", False))
    base = 0.05 + 0.02 * (a - 2) ** 2
    penalty = 0.06 if b else 0.0
    return _metrics(base + penalty)


def clean_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """No toxic parameters: best config is a=2, b=True."""
    a = int(cfg.get("a", 0))
    b = bool(cfg.get("b", False))
    wer = 0.10 + 0.01 * (a - 2) ** 2 - (0.03 if b else 0.0)
    return _metrics(max(0.0, wer))


def two_toxic_params_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    Both 'b' and 'c' are individually toxic when added to a=2.

    Base a=2, b=False, c=False: WER = 0.05 (good)
    a=2, b=True, c=False:       WER = 0.10 (+0.05)
    a=2, b=False, c=True:       WER = 0.10 (+0.05)
    a=2, b=True, c=True:        WER = 0.16 (+0.11, slightly extra bad)
    """
    a = int(cfg.get("a", 0))
    b = bool(cfg.get("b", False))
    c = bool(cfg.get("c", False))
    base = 0.05 + 0.005 * (a - 2) ** 2
    if b:
        base += 0.05
    if c:
        base += 0.05
    if b and c:
        base += 0.01  # interaction
    return _metrics(base)


# ----------------------------------------------------------------------
# Leave-one-out tests
# ----------------------------------------------------------------------


class TestLeaveOneOut:
    def test_detects_toxic_param(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=toxic_param_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)

        # Incoming best from layers 1-4: a=2, b=True (individually optimal but toxic together)
        incoming = _build_incoming(ex, {"a": 2, "b": True})
        ablation = DeepAblation(ex, space, budget, eps_min=0.005)
        result = ablation.run(incoming, sensitive_order=["a", "b"])

        assert "b" in result.toxic_params
        assert "a" not in result.toxic_params  # a=2 is legitimately optimal

    def test_refinement_and_final_pick_best_overall(self) -> None:
        """
        In this landscape 'b' is toxic but 'a' is not. LOO flags only 'b' as
        toxic. The blanket-revert refinement produces (a=0, b=False) which is
        NOT the optimum — but the `final()` method of AblationResult must fall
        back to the single-parameter LOO trial (a=2, b=False) which IS the
        true minimum among all ablation trials.
        """
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=toxic_param_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True})
        result = DeepAblation(ex, space, budget, eps_min=0.005).run(
            incoming, sensitive_order=["a", "b"]
        )
        # Only 'b' must be flagged toxic — reverting 'a' makes things worse
        assert "b" in result.toxic_params
        assert "a" not in result.toxic_params
        # final() picks the LOO revert-b trial which is the true minimum
        final = result.final()
        assert final.config["a"] == 2
        assert final.config["b"] is False
        assert final.metrics["wer"] == pytest.approx(0.05)

    def test_clean_landscape_no_toxic_detected(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=clean_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True})
        result = DeepAblation(ex, space, budget).run(incoming, sensitive_order=["a", "b"])
        assert result.toxic_params == []
        # Nothing improves the incoming — final should BE the incoming
        assert result.final().score == incoming.score

    def test_skips_params_already_at_default(self) -> None:
        """If a param in incoming_best is already at its default, no LOO happens for it."""
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=clean_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        # a=0 is the default, b=False is the default. No LOO should run.
        incoming = _build_incoming(ex, {"a": 0, "b": False})
        result = DeepAblation(ex, space, budget).run(incoming, sensitive_order=["a", "b"])
        assert result.loo_results == []


# ----------------------------------------------------------------------
# Fallback guarantee
# ----------------------------------------------------------------------


class TestFallbackGuarantee:
    def test_final_never_worse_than_incoming(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=clean_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True})
        result = DeepAblation(ex, space, budget).run(incoming, sensitive_order=["a", "b"])
        assert result.final().score <= incoming.score

    def test_final_strictly_better_when_toxic_detected(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=toxic_param_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True})
        result = DeepAblation(ex, space, budget, eps_min=0.005).run(
            incoming, sensitive_order=["a", "b"]
        )
        assert result.final().score < incoming.score


# ----------------------------------------------------------------------
# Multi-toxic refinement
# ----------------------------------------------------------------------


class TestMultipleToxicParams:
    def test_both_b_and_c_detected_as_toxic(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
                ParamSpec(name="c", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_toxic_params_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True, "c": True})
        result = DeepAblation(ex, space, budget, eps_min=0.005).run(
            incoming, sensitive_order=["a", "b", "c"]
        )
        assert set(result.toxic_params) == {"b", "c"}

    def test_refinement_clears_all_toxic_at_once(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
                ParamSpec(name="c", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_toxic_params_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True, "c": True})
        result = DeepAblation(ex, space, budget, eps_min=0.005).run(
            incoming, sensitive_order=["a", "b", "c"]
        )
        assert result.refined_trial is not None
        assert result.refined_trial.config["b"] is False
        assert result.refined_trial.config["c"] is False
        # Refinement yields WER ~0.05 (base with a=2, b=False, c=False)
        refined_wer = result.refined_trial.metrics["wer"]
        assert refined_wer is not None
        assert refined_wer < 0.06


# ----------------------------------------------------------------------
# Optional leave-two-out
# ----------------------------------------------------------------------


class TestLeaveTwoOut:
    def test_pair_ablation_disabled_by_default(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
                ParamSpec(name="c", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_toxic_params_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True, "c": True})
        result = DeepAblation(ex, space, budget, eps_min=0.005).run(
            incoming, sensitive_order=["a", "b", "c"]
        )
        assert result.pair_trials == []

    def test_pair_ablation_runs_when_enabled(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
                ParamSpec(name="c", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_toxic_params_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True, "c": True})
        result = DeepAblation(
            ex,
            space,
            budget,
            eps_min=0.005,
            enable_pairs=True,
        ).run(incoming, sensitive_order=["a", "b", "c"])
        assert len(result.pair_trials) >= 1


# ----------------------------------------------------------------------
# Budget safety
# ----------------------------------------------------------------------


class TestAblationBudget:
    def test_respects_budget(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=4),
                ParamSpec(name="b", type="bool", default=False),
                ParamSpec(name="c", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_toxic_params_landscape, objective=obj)
        budget = BudgetController(hard_cap=2, convergence_window=0)
        incoming = _build_incoming(ex, {"a": 2, "b": True, "c": True})
        DeepAblation(ex, space, budget).run(incoming, sensitive_order=["a", "b", "c"])
        assert budget.runs_used <= 2
