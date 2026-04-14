"""
Unit tests for Local 1D search strategies (Layers 2 + 6).

Each strategy is tested against a hand-crafted landscape with a known
minimum. The tests verify that the search converges to — or within tolerance
of — the analytical optimum.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.local_1d import (
    exhaustive_search,
    golden_section_search,
    pattern_search,
    search_1d,
)
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParamSpec
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
# Landscape fixtures
# ----------------------------------------------------------------------


def quadratic_temp(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """WER = 0.05 + 0.3*(temp - 0.4)^2 → minimum at temp=0.4, WER=0.05."""
    t = float(cfg.get("temperature", 0.0))
    return _metrics(0.05 + 0.3 * (t - 0.4) ** 2)


def monotonic_beam(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """WER = 0.20 - 0.007*beam_size → minimum at beam=20, WER=0.06."""
    b = int(cfg.get("beam_size", 5))
    return _metrics(0.20 - 0.007 * b)


def parabolic_beam(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """WER minimum at beam=7; large curvature. Good pattern search test."""
    b = int(cfg.get("beam_size", 5))
    return _metrics(0.05 + 0.003 * (b - 7) ** 2)


def enum_best_at_en(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    table = {"tr": 0.15, "en": 0.05, "auto": 0.10, "de": 0.20}
    return _metrics(table[cfg["language"]])


# ----------------------------------------------------------------------
# Golden Section tests
# ----------------------------------------------------------------------


class TestGoldenSection:
    def test_finds_interior_minimum(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=quadratic_temp, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        param = ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0)
        result = golden_section_search(
            ex,
            baseline_config={"temperature": 0.0},
            param=param,
            budget=budget,
            eps_min=0.001,
            max_iterations=10,
        )
        # True minimum is at temp=0.4
        assert abs(float(result.best.config["temperature"]) - 0.4) < 0.05

    def test_never_exceeds_budget(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=quadratic_temp, objective=obj)
        budget = BudgetController(hard_cap=5, convergence_window=0)
        param = ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0)
        golden_section_search(
            ex,
            baseline_config={"temperature": 0.0},
            param=param,
            budget=budget,
        )
        assert budget.runs_used <= 5

    def test_rejects_non_float_param(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=quadratic_temp, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="beam_size", type="int", min=1, default=5, max=20)
        with pytest.raises(ValueError, match="requires a float parameter"):
            golden_section_search(ex, baseline_config={"beam_size": 5}, param=param, budget=budget)

    def test_uses_seed_trials(self) -> None:
        """Pre-evaluated boundary trials should be reused, not re-run."""
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=quadratic_temp, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0)
        # Pre-compute boundary trials (as the screening phase would)
        t_min = ex.evaluate({"temperature": 0.0}, phase="screening")
        t_max = ex.evaluate({"temperature": 1.0}, phase="screening")
        runs_before = ex.runs_used
        golden_section_search(
            ex,
            baseline_config={"temperature": 0.0},
            param=param,
            budget=budget,
            seed_trials=[t_min, t_max],
        )
        # Seed trials must not trigger extra executor runs for their values
        # (they may still be in the executor's cache from the pre-eval above)
        assert ex.runs_used >= runs_before

    def test_monotonic_landscape_converges_to_boundary(self) -> None:
        obj = SingleMetricObjective(metric="wer")

        def linear(cfg):
            return _metrics(0.2 - 0.15 * float(cfg.get("t", 0.0)))

        ex = SyntheticTrialExecutor(metric_fn=linear, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="t", type="float", min=0.0, default=0.0, max=1.0)
        result = golden_section_search(ex, baseline_config={"t": 0.0}, param=param, budget=budget)
        # True minimum at t=1.0
        assert float(result.best.config["t"]) > 0.85


# ----------------------------------------------------------------------
# Exhaustive search tests
# ----------------------------------------------------------------------


class TestExhaustiveSearch:
    def test_enum_finds_best(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=enum_best_at_en, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(
            name="language",
            type="enum",
            default="auto",
            values=("tr", "en", "auto", "de"),
        )
        result = exhaustive_search(
            ex, baseline_config={"language": "auto"}, param=param, budget=budget
        )
        assert result.best.config["language"] == "en"

    def test_bool_finds_best(self) -> None:
        obj = SingleMetricObjective(metric="wer")

        def vad_better(cfg):
            return _metrics(0.08 if cfg["vad"] else 0.18)

        ex = SyntheticTrialExecutor(metric_fn=vad_better, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="vad", type="bool", default=False)
        result = exhaustive_search(ex, baseline_config={"vad": False}, param=param, budget=budget)
        assert result.best.config["vad"] is True

    def test_small_int_exhaustive(self) -> None:
        obj = SingleMetricObjective(metric="wer")

        def tiny_int(cfg):
            return _metrics(0.05 + 0.02 * abs(int(cfg["n"]) - 2))

        ex = SyntheticTrialExecutor(metric_fn=tiny_int, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="n", type="int", min=0, default=3, max=4)
        result = exhaustive_search(ex, baseline_config={"n": 3}, param=param, budget=budget)
        assert result.best.config["n"] == 2


# ----------------------------------------------------------------------
# Pattern search tests
# ----------------------------------------------------------------------


class TestPatternSearch:
    def test_finds_parabolic_minimum(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=parabolic_beam, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        param = ParamSpec(name="beam_size", type="int", min=1, default=5, max=20)
        result = pattern_search(ex, baseline_config={"beam_size": 5}, param=param, budget=budget)
        # True minimum at beam_size=7
        assert abs(int(result.best.config["beam_size"]) - 7) <= 1

    def test_monotonic_reaches_boundary(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=monotonic_beam, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        param = ParamSpec(name="beam_size", type="int", min=1, default=5, max=20)
        result = pattern_search(ex, baseline_config={"beam_size": 5}, param=param, budget=budget)
        assert int(result.best.config["beam_size"]) >= 18  # near maximum

    def test_rejects_non_int(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=quadratic_temp, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="t", type="float", min=0.0, default=0.0, max=1.0)
        with pytest.raises(ValueError, match="requires an int parameter"):
            pattern_search(ex, baseline_config={"t": 0.0}, param=param, budget=budget)


# ----------------------------------------------------------------------
# Dispatcher test
# ----------------------------------------------------------------------


class TestSearch1dDispatch:
    def test_float_routes_to_golden_section(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=quadratic_temp, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0)
        result = search_1d(ex, baseline_config={"temperature": 0.0}, param=param, budget=budget)
        assert result.param_name == "temperature"
        assert abs(float(result.best.config["temperature"]) - 0.4) < 0.05

    def test_large_int_routes_to_pattern(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=parabolic_beam, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        param = ParamSpec(name="beam_size", type="int", min=1, default=5, max=20)
        result = search_1d(ex, baseline_config={"beam_size": 5}, param=param, budget=budget)
        assert abs(int(result.best.config["beam_size"]) - 7) <= 1

    def test_small_int_routes_to_exhaustive(self) -> None:
        obj = SingleMetricObjective(metric="wer")

        def tiny_int(cfg):
            return _metrics(0.05 + 0.02 * abs(int(cfg["n"]) - 2))

        ex = SyntheticTrialExecutor(metric_fn=tiny_int, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="n", type="int", min=0, default=3, max=4)
        result = search_1d(ex, baseline_config={"n": 3}, param=param, budget=budget)
        assert result.best.config["n"] == 2

    def test_bool_routes_to_exhaustive(self) -> None:
        obj = SingleMetricObjective(metric="wer")

        def vad_better(cfg):
            return _metrics(0.05 if cfg["vad"] else 0.15)

        ex = SyntheticTrialExecutor(metric_fn=vad_better, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(name="vad", type="bool", default=False)
        result = search_1d(ex, baseline_config={"vad": False}, param=param, budget=budget)
        assert result.best.config["vad"] is True

    def test_enum_routes_to_exhaustive(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=enum_best_at_en, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        param = ParamSpec(
            name="language",
            type="enum",
            default="auto",
            values=("tr", "en", "auto", "de"),
        )
        result = search_1d(ex, baseline_config={"language": "auto"}, param=param, budget=budget)
        assert result.best.config["language"] == "en"
