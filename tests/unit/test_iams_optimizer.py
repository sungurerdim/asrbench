"""
End-to-end tests for IAMSOptimizer — the 7-layer orchestrator.

These tests run the full pipeline against synthetic landscapes and verify that
the orchestrator finds the global optimum on landscapes where naive coordinate
descent would fail. The synthetic executor is deterministic, so every test is
reproducible — run it twice, get the same trial count.

The key test is the interaction-trap landscape: a 2-parameter problem where
one axis of coordinate descent from the default baseline converges to a
non-global minimum. IAMSOptimizer in "maximum" mode must recover the global
via Layer 3's pairwise grid + Layer 4's multi-start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.optimizer import IAMSOptimizer
from asrbench.engine.search.budget import BudgetController
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


def convex_5param(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    5-parameter convex landscape: quadratic in each axis, perfectly additive.
    Unique global minimum at beam_size=7, temperature=0.3, patience=1.0,
    vad=True, best_of=5. WER at minimum = 0.05.
    """
    beam = int(cfg.get("beam_size", 5))
    temp = float(cfg.get("temperature", 0.0))
    patience = float(cfg.get("patience", 0.5))
    vad = bool(cfg.get("vad", False))
    best_of = int(cfg.get("best_of", 1))

    wer = 0.05
    wer += 0.003 * (beam - 7) ** 2
    wer += 0.02 * (temp - 0.3) ** 2
    wer += 0.01 * (patience - 1.0) ** 2
    wer += 0.0 if vad else 0.04
    wer += 0.002 * (best_of - 5) ** 2
    return _metrics(wer)


def asymmetric_interaction_trap(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    The canonical vad × chunk_length interaction trap — extended with a
    no-op parameter 'noise' to verify insensitive-parameter detection.

    vad × chunk table:
                   chunk=10  chunk=20  chunk=30
        vad=False    0.12      0.11      0.14
        vad=True     0.15      0.13      0.05    ← global minimum

    'noise' is a float in [0, 1] that does not affect WER at all. The
    screening phase must mark it as insensitive and NOT include it in
    the sensitive_order, so Layer 3 focuses on the real interaction.
    """
    vad = bool(cfg.get("vad", False))
    chunk = int(cfg.get("chunk_length", 10))
    t = {10: 0.12, 20: 0.11, 30: 0.14} if not vad else {10: 0.15, 20: 0.13, 30: 0.05}
    return _metrics(t[chunk])


def flat_landscape(_cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Everything returns the same WER — no sensitive parameters."""
    return _metrics(0.10)


# ----------------------------------------------------------------------
# Basic orchestration tests
# ----------------------------------------------------------------------


class TestIAMSBasics:
    def test_convex_landscape_finds_global_minimum(self) -> None:
        # Defaults are deliberately offset from BOTH min and max so OFAT-3
        # screening sees distinct scores on all three probes. Putting default
        # equal to min makes baseline↔min a no-op and can mask a parameter's
        # sensitivity when the optimum is in the interior of the range.
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="beam_size", type="int", min=1, default=3, max=15),
                ParamSpec(name="temperature", type="float", min=0.0, default=0.1, max=1.0),
                ParamSpec(name="patience", type="float", min=0.0, default=0.5, max=2.0),
                ParamSpec(name="vad", type="bool", default=False),
                ParamSpec(name="best_of", type="int", min=1, default=3, max=10),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_5param, objective=obj)
        budget = BudgetController(hard_cap=300, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            eps_min=0.002,
            mode="maximum",
        )
        result = optimizer.run()

        # Global minimum: WER = 0.05. Allow 0.02 tolerance because continuous
        # float parameters (temperature, patience) converge to within ~5% of
        # their optimum under golden section with finite budget, contributing
        # ~0.005-0.015 residual.
        assert result.best_trial.metrics["wer"] == pytest.approx(0.05, abs=0.02)
        # Discrete parameters should lock onto their optima under pattern search
        assert abs(result.best_config["beam_size"] - 7) <= 1
        assert result.best_config["vad"] is True
        assert abs(int(result.best_config["best_of"]) - 5) <= 1
        # Continuous parameters should be in the right neighborhood
        assert abs(float(result.best_config["temperature"]) - 0.3) < 0.2
        assert abs(float(result.best_config["patience"]) - 1.0) < 0.3

    def test_fast_mode_runs_only_layer_1_and_2(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="beam_size", type="int", min=1, default=1, max=15),
                ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_5param, objective=obj)
        budget = BudgetController(hard_cap=100, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="fast",
        )
        result = optimizer.run()
        assert result.pairwise is None
        assert result.multistart is None
        assert result.ablation is None
        assert result.validation is None

    def test_balanced_mode_runs_layers_1_to_5(self) -> None:
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
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_interaction_trap, objective=obj)
        budget = BudgetController(hard_cap=200, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="balanced",
            eps_min=0.005,
        )
        result = optimizer.run()
        assert result.pairwise is not None
        assert result.multistart is not None
        assert result.ablation is not None
        assert result.refined_trial is None
        assert result.validation is None

    def test_maximum_mode_runs_all_layers(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="beam_size", type="int", min=1, default=1, max=15),
                ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_5param, objective=obj)
        budget = BudgetController(hard_cap=300, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="maximum",
        )
        result = optimizer.run()
        assert result.pairwise is not None
        assert result.multistart is not None
        assert result.ablation is not None
        assert result.refined_trial is not None
        assert result.validation is not None
        assert result.validation.confidence in ("HIGH", "MEDIUM", "LOW")


# ----------------------------------------------------------------------
# THE killer test: interaction trap global recovery
# ----------------------------------------------------------------------


class TestInteractionTrapRecovery:
    def test_maximum_mode_escapes_interaction_trap(self) -> None:
        """
        The IAMS orchestrator must find the global minimum (WER=0.05) on the
        vad × chunk_length interaction trap, even though a pure coordinate
        descent from the default baseline converges to a non-global point.

        This is the core correctness claim of the IAMS design.
        """
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
                # Insensitive red-herring param — forces the orchestrator to
                # correctly filter it out during screening.
                ParamSpec(name="noise", type="float", min=0.0, default=0.0, max=1.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_interaction_trap, objective=obj)
        budget = BudgetController(hard_cap=200, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="maximum",
            eps_min=0.003,
        )
        result = optimizer.run()

        # Global minimum: WER = 0.05 at (vad=True, chunk_length=30)
        assert result.best_trial.metrics["wer"] == pytest.approx(0.05, abs=0.001)
        assert result.best_config["vad"] is True
        assert result.best_config["chunk_length"] == 30
        # 'noise' must be flagged insensitive
        assert "noise" in result.insensitive_params

    def test_coordinate_descent_alone_misses_optimum(self) -> None:
        """
        Sanity check: with 'fast' mode (no Layer 3 / Layer 4), the orchestrator
        should NOT find the global minimum — demonstrating that the later
        layers are load-bearing, not decorative.
        """
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
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_interaction_trap, objective=obj)
        budget = BudgetController(hard_cap=100, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="fast",
            eps_min=0.003,
        )
        result = optimizer.run()

        # Fast mode ought to get stuck at a local minimum (WER > 0.05).
        # Interaction trap: the default (False, 10) = 0.12 is the only
        # "downhill move" endpoint for pure coordinate descent from defaults.
        wer = result.best_trial.metrics["wer"]
        assert wer is not None
        assert wer > 0.08


# ----------------------------------------------------------------------
# Degenerate and edge cases
# ----------------------------------------------------------------------


class TestDegenerateCases:
    def test_flat_landscape_stops_early(self) -> None:
        """If no parameter is sensitive, IAMS should stop after Layer 1."""
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=1, default=5, max=20),
                ParamSpec(name="b", type="bool", default=False),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=flat_landscape, objective=obj)
        budget = BudgetController(hard_cap=100, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="maximum",
        )
        result = optimizer.run()
        # Only Layer 1 ran: no Layer 2+ artifacts
        assert result.pairwise is None
        assert result.multistart is None
        assert result.ablation is None
        assert result.refined_trial is None
        assert result.validation is None
        assert set(result.insensitive_params) == {"a", "b"}

    def test_budget_hard_cap_never_exceeded(self) -> None:
        """The hard budget cap must be respected throughout all layers."""
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="beam_size", type="int", min=1, default=1, max=15),
                ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
                ParamSpec(name="patience", type="float", min=0.0, default=0.0, max=2.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=convex_5param, objective=obj)
        budget = BudgetController(hard_cap=30, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="maximum",
        )
        optimizer.run()
        assert budget.runs_used <= 30

    def test_invalid_mode_rejected(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="a", type="int", min=1, default=5, max=10),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=flat_landscape, objective=obj)
        budget = BudgetController(hard_cap=20, convergence_window=0)
        with pytest.raises(ValueError, match="unknown mode"):
            IAMSOptimizer(
                executor=ex,
                space=space,
                objective=obj,
                budget=budget,
                mode="thorough",  # type: ignore[arg-type]
            )


# ----------------------------------------------------------------------
# Fallback guarantee
# ----------------------------------------------------------------------


class TestFallbackGuarantee:
    def test_best_trial_is_global_minimum_of_all_trials(self) -> None:
        """
        The fallback invariant: result.best_trial.score must equal the
        minimum score observed across ALL evaluated trials, regardless of
        which layer produced it.
        """
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
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_interaction_trap, objective=obj)
        budget = BudgetController(hard_cap=200, convergence_window=0)

        optimizer = IAMSOptimizer(
            executor=ex,
            space=space,
            objective=obj,
            budget=budget,
            mode="maximum",
            eps_min=0.003,
        )
        result = optimizer.run()
        min_observed = min(t.score for t in result.all_trials)
        assert result.best_trial.score == pytest.approx(min_observed)
