"""
Unit tests for ScreeningPhase (Layer 1) — OFAT-3 sensitivity analysis.

The tests use SyntheticTrialExecutor with hand-crafted landscape functions so
we can assert exactly which parameters should be judged sensitive and in what
order. No real benchmark is involved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.screening import ScreeningPhase
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor

# ----------------------------------------------------------------------
# Landscape fixtures — deterministic math functions over config dicts
# ----------------------------------------------------------------------


def _make_metrics(wer: float) -> dict[str, float]:
    """Build a metrics dict with tight bootstrap CI so significance gate triggers cleanly."""
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


def flat_landscape(_cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Every config yields the same WER — every parameter should be insensitive."""
    return _make_metrics(0.100)


def beam_only_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    Only beam_size matters. Linear: WER decreases from 0.20 @ beam=1 to 0.05 @ beam=20.
    Other params do nothing.
    """
    beam = float(cfg.get("beam_size", 5))
    wer = 0.20 - 0.15 * (beam - 1) / 19.0  # interpolate 0.20 → 0.05
    return _make_metrics(wer)


def two_sensitive_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    beam_size: high leverage (0.15 spread)
    temperature: lower leverage (0.05 spread)
    vad_filter: no effect
    """
    beam = float(cfg.get("beam_size", 5))
    temp = float(cfg.get("temperature", 0.0))
    wer_beam = 0.15 * (1.0 - (beam - 1) / 19.0)  # 0.15 @ beam=1 → 0.0 @ beam=20
    wer_temp = 0.05 * temp  # 0.0 @ temp=0 → 0.05 @ temp=1
    return _make_metrics(0.05 + wer_beam + wer_temp)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestScreeningBasics:
    def test_baseline_runs_first(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        # Baseline evaluates with all defaults → beam_size=5
        assert result.baseline.config["beam_size"] == 5

    def test_flat_landscape_all_insensitive(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=1, default=5, max=20),
                ParamSpec(name="b", type="float", min=0.0, default=0.0, max=1.0),
                ParamSpec(name="c", type="bool", default=True),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=flat_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        assert result.sensitive_order == []
        assert set(result.insensitive) == {"a", "b", "c"}

    def test_single_sensitive_parameter(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),
                ParamSpec(name="noop", type="bool", default=True),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        assert result.sensitive_order == ["beam_size"]
        assert "noop" in result.insensitive

    def test_sensitivity_ranking_by_leverage(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
                ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),
                ParamSpec(name="vad", type="bool", default=True),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_sensitive_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        # beam has spread 0.15, temperature has spread 0.05 → beam first
        assert result.sensitive_order == ["beam_size", "temperature"]
        assert "vad" in result.insensitive

    def test_trial_count_equals_1_plus_2n(self) -> None:
        # Use three int params so no probe value collides with the default
        # (avoids incidental cache hits and makes the 1+2N accounting exact).
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=1, default=5, max=20),
                ParamSpec(name="b", type="int", min=1, default=10, max=30),
                ParamSpec(name="c", type="int", min=2, default=6, max=15),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_sensitive_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        # 1 baseline + 2*3 boundary = 7 trials
        assert len(result.trials) == 7
        assert ex.runs_used == 7


class TestScreeningBoundaryProbes:
    def test_int_min_max_used(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        ps = result.params["beam_size"]
        assert ps.min_trial.config["beam_size"] == 1
        assert ps.max_trial.config["beam_size"] == 20

    def test_bool_probes_false_and_true(self) -> None:
        space = ParameterSpace(parameters=(ParamSpec(name="vad", type="bool", default=True),))
        obj = SingleMetricObjective(metric="wer")

        def vad_sensitive(cfg):
            return _make_metrics(0.08 if cfg.get("vad") else 0.18)

        ex = SyntheticTrialExecutor(metric_fn=vad_sensitive, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        ps = result.params["vad"]
        assert ps.min_trial.config["vad"] is False
        assert ps.max_trial.config["vad"] is True
        assert ps.sensitive is True

    def test_enum_probes_first_and_last(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(
                    name="lang",
                    type="enum",
                    default="auto",
                    values=("tr", "en", "auto"),
                ),
            )
        )
        obj = SingleMetricObjective(metric="wer")

        def lang_only(cfg):
            mapping = {"tr": 0.20, "en": 0.10, "auto": 0.05}
            return _make_metrics(mapping[cfg["lang"]])

        ex = SyntheticTrialExecutor(metric_fn=lang_only, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        ps = result.params["lang"]
        assert ps.min_trial.config["lang"] == "tr"
        assert ps.max_trial.config["lang"] == "auto"

    def test_degenerate_single_value_enum_marked_insensitive(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="only", type="enum", default="x", values=("x",)),
                ParamSpec(name="beam", type="int", min=1, default=5, max=10),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        assert "only" in result.insensitive


class TestScreeningBudget:
    def test_partial_screening_on_tight_budget(self) -> None:
        """If budget runs out mid-screening, remaining params go to unscreened list."""
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=1, default=5, max=20),
                ParamSpec(name="b", type="int", min=1, default=5, max=20),
                ParamSpec(name="c", type="int", min=1, default=5, max=20),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        # Budget: 1 baseline + 2 boundary = only 'a' can be screened fully, then 1 extra
        budget = BudgetController(hard_cap=3, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        assert "a" in result.params
        # b needs 2 more trials but only 0 remaining → unscreened
        assert "b" in result.unscreened or "c" in result.unscreened

    def test_zero_budget_raises(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="a", type="int", min=1, default=5, max=10),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=1, convergence_window=0)
        # Consume the only run so screening has nothing
        budget.record(0.5)
        with pytest.raises(RuntimeError, match="zero remaining runs"):
            ScreeningPhase(ex, space, budget, eps_min=0.005).run()


class TestScreeningResultHelpers:
    def test_boundary_trials_accessor(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        min_t, max_t = result.boundary_trials("beam_size")
        assert min_t.config["beam_size"] == 1
        assert max_t.config["beam_size"] == 20

    def test_best_overall_returns_lowest_score(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        best = result.best_overall()
        # beam_only_landscape: minimum at beam=20 → WER=0.05
        assert best.config["beam_size"] == 20
        assert best.score == pytest.approx(0.05)

    def test_param_screening_best_probe(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="beam_size", type="int", min=1, default=5, max=20),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=beam_only_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        result = ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        ps = result.params["beam_size"]
        best = ps.best_probe()
        # beam=20 is the minimum in this landscape
        assert best.config["beam_size"] == 20


class TestScreeningBudgetRecording:
    def test_budget_records_every_trial(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=1, default=5, max=10),
                ParamSpec(name="b", type="float", min=0.0, default=0.5, max=1.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=two_sensitive_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        ScreeningPhase(ex, space, budget, eps_min=0.005).run()
        # 1 + 2*2 = 5 trials recorded
        assert budget.runs_used == 5
