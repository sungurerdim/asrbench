"""
Unit tests for PairwiseGridScan (Layer 3) — interaction detection.

The killer test is the classic (vad, chunk_length) interaction trap:

                   chunk
               low    mid    high
    vad=false  8%    9%    11%    ← OFAT descent gets stuck here
    vad=true  11%   10%    8%    ← true global minimum (off-diagonal)

A parameter search that looks at each axis independently can see the upper-left
minimum `(vad=false, chunk=low) = 8%` and the bottom-right minimum
`(vad=true, chunk=high) = 8%` as two equally good options — neither axis's
1D search discovers that the bottom-right combination is the global optimum
because it requires BOTH settings together. Layer 3 must mark the bottom-right
cell as a promising off-diagonal point.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.pairwise_grid import PairwiseGridScan
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
# Landscape fixtures
# ----------------------------------------------------------------------


def interaction_trap(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    The vad × chunk_length interaction trap.

    vad ∈ {False, True}, chunk_length ∈ {10, 20, 30}.

    Table (WER):
                chunk=10  chunk=20  chunk=30
        vad=F      0.08     0.09     0.11
        vad=T      0.11     0.10     0.08

    Two global minima at (F, 10) and (T, 30), both with WER=0.08.
    An OFAT search starting from the default (vad=F, chunk=10) would pick
    vad=F, chunk=10 and never see that (T, 30) is equally good.
    """
    vad = bool(cfg.get("vad", False))
    chunk = int(cfg.get("chunk_length", 10))
    table = {10: 0.08, 20: 0.09, 30: 0.11} if not vad else {10: 0.11, 20: 0.1, 30: 0.08}
    return _metrics(table[chunk])


def additive_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """
    Perfectly additive: WER = 0.05 + 0.02*a + 0.03*b.
    No interaction. Layer 3 should find near-zero interaction score.
    """
    a = float(cfg.get("temperature", 0.0))
    b = float(cfg.get("patience", 0.0))
    return _metrics(0.05 + 0.02 * a + 0.03 * b)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestInteractionDetection:
    def test_detects_vad_chunk_interaction(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="vad", type="bool", default=False),
                ParamSpec(name="chunk_length", type="int", min=10, default=10, max=30, step=10),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=interaction_trap, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=2,
            eps_min=0.005,
            baseline_config={"vad": False, "chunk_length": 10},
            baseline_score=0.08,
        )
        result = scan.run(sensitive_params=["vad", "chunk_length"])
        assert len(result.grids) == 1
        grid = result.grids[0]
        # Non-trivial interaction detected — exact value depends on grid spacing
        assert grid.interaction_score > 0.01

    def test_finds_off_diagonal_global_minimum(self) -> None:
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
        ex = SyntheticTrialExecutor(metric_fn=interaction_trap, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=2,
            eps_min=0.001,
            baseline_config={"vad": False, "chunk_length": 10},
            baseline_score=0.08,
        )
        result = scan.run(sensitive_params=["vad", "chunk_length"])
        grid = result.grids[0]
        # The cell (vad=True, chunk=30) must be visible in the trials
        tied = [
            t for t in grid.trials if t.config["vad"] is True and t.config["chunk_length"] == 30
        ]
        assert len(tied) == 1
        assert tied[0].metrics["wer"] == pytest.approx(0.08)

    def test_promising_points_include_off_diagonal_win(self) -> None:
        """
        Use a landscape where the off-diagonal (vad=True, chunk=30) is strictly
        BETTER than the axis-best, so the significance gate fires.
        """

        def asymmetric_trap(cfg):
            vad = bool(cfg.get("vad", False))
            chunk = int(cfg.get("chunk_length", 10))
            # off-diagonal (vad=True, chunk=30) is strictly the best point
            t = {10: 0.12, 20: 0.11, 30: 0.05} if vad else {10: 0.09, 20: 0.10, 30: 0.12}
            return _metrics(t[chunk])

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
        ex = SyntheticTrialExecutor(metric_fn=asymmetric_trap, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=2,
            eps_min=0.005,
            baseline_config={"vad": False, "chunk_length": 10},
            baseline_score=0.09,
        )
        result = scan.run(sensitive_params=["vad", "chunk_length"])
        promising = result.promising_points()
        # At least one promising point should exist
        assert len(promising) >= 1
        # The promising point should be off-diagonal (not at base_a or base_b)
        for t in promising:
            assert not (t.config["vad"] is False and t.config["chunk_length"] == 10)


class TestAdditiveLandscape:
    def test_no_interaction_detected(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
                ParamSpec(name="patience", type="float", min=0.0, default=0.0, max=1.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=additive_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=2,
            eps_min=0.005,
            baseline_config={"temperature": 0.0, "patience": 0.0},
            baseline_score=0.05,
        )
        result = scan.run(sensitive_params=["temperature", "patience"])
        grid = result.grids[0]
        # Additive landscape: interaction deviation should be near zero
        assert grid.interaction_score < 0.001

    def test_additive_no_promising_points(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
                ParamSpec(name="patience", type="float", min=0.0, default=0.0, max=1.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=additive_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=2,
            eps_min=0.005,
            baseline_config={"temperature": 0.0, "patience": 0.0},
            baseline_score=0.05,
        )
        result = scan.run(sensitive_params=["temperature", "patience"])
        promising = result.promising_points()
        # Additive: no off-diagonal cell beats the axis minima
        assert promising == []


class TestGridInputs:
    def test_rejects_top_k_below_2(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="a", type="int", min=1, default=5, max=10),)
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=additive_landscape, objective=obj)
        budget = BudgetController(hard_cap=50, convergence_window=0)
        with pytest.raises(ValueError, match="top_k must be >= 2"):
            PairwiseGridScan(ex, space, budget, top_k=1)

    def test_respects_top_k_limit(self) -> None:
        """Only the first top_k sensitive params are used to form pairs."""
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=5, max=10),
                ParamSpec(name="b", type="int", min=0, default=5, max=10),
                ParamSpec(name="c", type="int", min=0, default=5, max=10),
                ParamSpec(name="d", type="int", min=0, default=5, max=10),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=additive_landscape, objective=obj)
        budget = BudgetController(hard_cap=200, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=3,
            baseline_config=space.defaults(),
            baseline_score=0.05,
        )
        result = scan.run(sensitive_params=["a", "b", "c", "d"])
        # top_k=3 → pairs from {a, b, c} = 3 pairs (not including d)
        assert len(result.grids) == 3
        pair_names = {(g.param_a, g.param_b) for g in result.grids}
        assert ("a", "b") in pair_names
        assert ("a", "c") in pair_names
        assert ("b", "c") in pair_names

    def test_budget_exhaustion_stops_early(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=5, max=10),
                ParamSpec(name="b", type="int", min=0, default=5, max=10),
                ParamSpec(name="c", type="int", min=0, default=5, max=10),
            )
        )
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=additive_landscape, objective=obj)
        # 3×3 grid per pair = 9 trials; budget caps at 5
        budget = BudgetController(hard_cap=5, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=3,
            baseline_config=space.defaults(),
            baseline_score=0.05,
        )
        scan.run(sensitive_params=["a", "b", "c"])
        assert budget.runs_used <= 5

    def test_pair_by_interaction_sorted(self) -> None:
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
                ParamSpec(name="temperature", type="float", min=0.0, default=0.0, max=1.0),
            )
        )
        obj = SingleMetricObjective(metric="wer")

        def mixed(cfg):
            # vad×chunk has an interaction; temp is additive
            base = interaction_trap(cfg)["wer"]
            return _metrics(base + 0.01 * float(cfg.get("temperature", 0.0)))

        ex = SyntheticTrialExecutor(metric_fn=mixed, objective=obj)
        budget = BudgetController(hard_cap=200, convergence_window=0)
        scan = PairwiseGridScan(
            ex,
            space,
            budget,
            top_k=3,
            eps_min=0.005,
            baseline_config=space.defaults(),
            baseline_score=0.08,
        )
        result = scan.run(sensitive_params=["vad", "chunk_length", "temperature"])
        sorted_grids = result.pair_by_interaction()
        # The (vad, chunk_length) pair should have the highest interaction
        assert sorted_grids[0].param_a == "vad" and sorted_grids[0].param_b == "chunk_length"
