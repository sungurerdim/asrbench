"""
Tests for MultiDatasetTrialExecutor — "one config across N datasets" wrapper.

Verifies that:
    1. Aggregate score is the weighted mean of component scores.
    2. Uniform weights default when the caller omits them.
    3. CI shrinks under independent-variance aggregation.
    4. Per-dataset metrics are preserved under the label prefix.
    5. Multi-fidelity forwarding (evaluate_at_fraction) works.
    6. Propagates pruned=True if any inner component was pruned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from asrbench.engine.search.multidataset import MultiDatasetTrialExecutor
from asrbench.engine.search.trial import TrialResult


@dataclass
class _FixedExecutor:
    """Minimal executor that returns a pre-built TrialResult regardless of config."""

    score: float
    ci_half: float = 0.001
    pruned: bool = False
    extra_metrics: dict[str, float] = field(default_factory=dict)
    supports_fraction: bool = True
    _calls: int = field(default=0, init=False)

    @property
    def runs_used(self) -> int:
        return self._calls

    def _build(self, config: Mapping[str, Any], phase: str, reasoning: str) -> TrialResult:
        self._calls += 1
        metrics: dict[str, float | None] = {
            "wer": self.score,
            "cer": self.score * 0.5,
            "rtfx_mean": 20.0,
            "vram_peak_mb": 4000.0,
            "wer_ci_lower": max(0.0, self.score - self.ci_half),
            "wer_ci_upper": self.score + self.ci_half,
        }
        metrics.update(self.extra_metrics)
        return TrialResult(
            config=dict(config),
            metrics=metrics,
            score=self.score,
            score_ci=(
                max(0.0, self.score - self.ci_half),
                self.score + self.ci_half,
            ),
            phase=phase,
            reasoning=reasoning,
            pruned=self.pruned,
        )

    def evaluate(self, config, *, phase="unknown", reasoning=""):
        return self._build(config, phase, reasoning)

    def evaluate_at_fraction(self, config, *, phase="unknown", reasoning="", fraction=1.0):
        if not self.supports_fraction:
            raise AttributeError("does not support fractions")
        return self._build(config, phase, reasoning)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_score_is_weighted_mean() -> None:
    mf = MultiDatasetTrialExecutor(
        executors=[_FixedExecutor(score=0.10), _FixedExecutor(score=0.20)],
        weights=[0.25, 0.75],
    )
    trial = mf.evaluate({"x": 1}, phase="local_1d", reasoning="test")
    # 0.25*0.10 + 0.75*0.20 = 0.025 + 0.15 = 0.175
    assert trial.score == pytest.approx(0.175)


def test_uniform_weights_default() -> None:
    mf = MultiDatasetTrialExecutor(
        executors=[_FixedExecutor(score=0.10), _FixedExecutor(score=0.30)],
    )
    trial = mf.evaluate({"x": 1})
    assert trial.score == pytest.approx(0.20)  # (0.10 + 0.30) / 2


def test_weights_are_normalized() -> None:
    mf = MultiDatasetTrialExecutor(
        executors=[_FixedExecutor(score=0.10), _FixedExecutor(score=0.30)],
        weights=[1, 3],  # should normalize to [0.25, 0.75]
    )
    trial = mf.evaluate({"x": 1})
    assert trial.score == pytest.approx(0.25)  # 0.25*0.10 + 0.75*0.30


def test_construction_rejects_empty_executor_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiDatasetTrialExecutor(executors=[])


def test_construction_rejects_weight_length_mismatch() -> None:
    with pytest.raises(ValueError, match="weights length"):
        MultiDatasetTrialExecutor(
            executors=[_FixedExecutor(score=0.1), _FixedExecutor(score=0.2)],
            weights=[1.0, 1.0, 1.0],
        )


def test_construction_rejects_non_positive_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        MultiDatasetTrialExecutor(executors=[_FixedExecutor(score=0.1)], weights=[0.0])


# ---------------------------------------------------------------------------
# CI aggregation
# ---------------------------------------------------------------------------


def test_aggregate_ci_shrinks_under_independence() -> None:
    """N independent datasets should yield a tighter CI than any one alone."""
    executors = [
        _FixedExecutor(score=0.10, ci_half=0.005),
        _FixedExecutor(score=0.10, ci_half=0.005),
        _FixedExecutor(score=0.10, ci_half=0.005),
    ]
    mf = MultiDatasetTrialExecutor(executors=executors)
    trial = mf.evaluate({"x": 1})
    combined_half = (trial.score_ci[1] - trial.score_ci[0]) / 2
    # Under uniform weights and equal component CIs, the combined half-width
    # should be strictly smaller than the per-dataset half (0.005).
    assert combined_half < 0.005


# ---------------------------------------------------------------------------
# Per-dataset metric breakdown
# ---------------------------------------------------------------------------


def test_per_dataset_metrics_are_prefixed_with_label() -> None:
    mf = MultiDatasetTrialExecutor(
        executors=[
            _FixedExecutor(score=0.10, extra_metrics={"rtfx_mean": 15.0}),
            _FixedExecutor(score=0.20, extra_metrics={"rtfx_mean": 25.0}),
        ],
        labels=["clean", "noisy"],
    )
    trial = mf.evaluate({"x": 1})
    assert "clean.wer" in trial.metrics
    assert "noisy.wer" in trial.metrics
    assert trial.metrics["clean.rtfx_mean"] == 15.0
    assert trial.metrics["noisy.rtfx_mean"] == 25.0
    # Aggregate "wer" is still surfaced at top level for Objective.score.
    assert trial.metrics["wer"] == pytest.approx(0.15)


def test_breakdown_appears_in_reasoning() -> None:
    mf = MultiDatasetTrialExecutor(
        executors=[_FixedExecutor(score=0.10), _FixedExecutor(score=0.20)],
        labels=["tr", "en"],
    )
    trial = mf.evaluate({"x": 1}, reasoning="L2 sweep")
    assert "tr=0.1000" in trial.reasoning
    assert "en=0.2000" in trial.reasoning
    assert "aggregate=0.1500" in trial.reasoning


# ---------------------------------------------------------------------------
# Multi-fidelity forwarding
# ---------------------------------------------------------------------------


def test_evaluate_at_fraction_forwards_to_inner_when_supported() -> None:
    executors = [
        _FixedExecutor(score=0.10, supports_fraction=True),
        _FixedExecutor(score=0.20, supports_fraction=True),
    ]
    mf = MultiDatasetTrialExecutor(executors=executors)
    trial = mf.evaluate_at_fraction({"x": 1}, fraction=0.25)
    assert trial.score == pytest.approx(0.15)
    # Each inner executor saw exactly one call.
    assert executors[0].runs_used == 1
    assert executors[1].runs_used == 1


def test_evaluate_at_fraction_falls_back_when_inner_lacks_support() -> None:
    # One inner executor explicitly lacks evaluate_at_fraction support.
    class _NoFraction:
        runs_used = 0

        def evaluate(self, config, *, phase="unknown", reasoning=""):
            self.runs_used += 1
            return TrialResult(
                config=dict(config),
                metrics={"wer": 0.3, "wer_ci_lower": 0.29, "wer_ci_upper": 0.31},
                score=0.3,
                score_ci=(0.29, 0.31),
                phase=phase,
                reasoning=reasoning,
            )

    mf = MultiDatasetTrialExecutor(
        executors=[_FixedExecutor(score=0.1), _NoFraction()],
    )
    trial = mf.evaluate_at_fraction({"x": 1}, fraction=0.5)
    assert trial.score == pytest.approx(0.2)  # (0.1 + 0.3) / 2


# ---------------------------------------------------------------------------
# Pruned propagation
# ---------------------------------------------------------------------------


def test_pruned_propagates_when_any_inner_pruned() -> None:
    mf = MultiDatasetTrialExecutor(
        executors=[
            _FixedExecutor(score=0.10, pruned=False),
            _FixedExecutor(score=0.40, pruned=True),
        ],
    )
    trial = mf.evaluate({"x": 1})
    assert trial.pruned is True


def test_pruned_false_when_no_inner_pruned() -> None:
    mf = MultiDatasetTrialExecutor(
        executors=[
            _FixedExecutor(score=0.10, pruned=False),
            _FixedExecutor(score=0.15, pruned=False),
        ],
    )
    trial = mf.evaluate({"x": 1})
    assert trial.pruned is False
