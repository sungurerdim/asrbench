"""Unit tests for Trial abstractions — TrialResult identity + SyntheticTrialExecutor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.trial import (
    SyntheticTrialExecutor,
    TrialExecutor,
    TrialResult,
)


def _quadratic_wer(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Simple 1D quadratic landscape: minimum at beam_size=7."""
    beam = float(cfg.get("beam_size", 7))
    wer = 0.1 + 0.005 * (beam - 7) ** 2
    return {
        "wer": wer,
        "cer": wer * 0.5,
        "mer": wer,
        "wil": wer,
        "rtfx_mean": 20.0,
        "vram_peak_mb": 4000.0,
        "wer_ci_lower": max(0.0, wer - 0.002),
        "wer_ci_upper": wer + 0.002,
    }


class TestTrialResult:
    def test_config_key_order_independent(self) -> None:
        t1 = TrialResult(
            config={"a": 1, "b": 2},
            metrics={"wer": 0.1},
            score=0.1,
            score_ci=(0.09, 0.11),
        )
        t2 = TrialResult(
            config={"b": 2, "a": 1},  # different insertion order
            metrics={"wer": 0.1},
            score=0.1,
            score_ci=(0.09, 0.11),
        )
        assert t1.config_key() == t2.config_key()

    def test_config_key_differs_for_different_configs(self) -> None:
        t1 = TrialResult(config={"a": 1}, metrics={"wer": 0.1}, score=0.1, score_ci=(0.1, 0.1))
        t2 = TrialResult(config={"a": 2}, metrics={"wer": 0.1}, score=0.1, score_ci=(0.1, 0.1))
        assert t1.config_key() != t2.config_key()

    def test_with_phase_preserves_data(self) -> None:
        t = TrialResult(
            config={"a": 1},
            metrics={"wer": 0.1},
            score=0.1,
            score_ci=(0.1, 0.1),
            phase="screening",
            reasoning="baseline",
        )
        t2 = t.with_phase("local_1d", "golden section iteration 3")
        assert t2.config == t.config
        assert t2.metrics == t.metrics
        assert t2.score == t.score
        assert t2.phase == "local_1d"
        assert t2.reasoning == "golden section iteration 3"
        assert t.phase == "screening"  # immutable: original unchanged


class TestSyntheticTrialExecutor:
    def test_deterministic_without_noise(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        t1 = ex.evaluate({"beam_size": 5})
        t2 = ex.evaluate({"beam_size": 5})
        assert t1.score == t2.score
        assert t1.config_key() == t2.config_key()

    def test_score_reflects_objective_direction(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        optimal = ex.evaluate({"beam_size": 7})
        worse = ex.evaluate({"beam_size": 3})
        assert optimal.score < worse.score
        # At beam=7 the quadratic is exactly minimized to 0.1
        assert optimal.score == pytest.approx(0.1)

    def test_protocol_conformance(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        # Runtime protocol check
        assert isinstance(ex, TrialExecutor)

    def test_runs_used_increments(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        assert ex.runs_used == 0
        ex.evaluate({"beam_size": 1})
        ex.evaluate({"beam_size": 2})
        ex.evaluate({"beam_size": 3})
        assert ex.runs_used == 3

    def test_cache_prevents_duplicate_runs(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        ex.evaluate({"beam_size": 5})
        ex.evaluate({"beam_size": 5})  # cache hit
        ex.evaluate({"beam_size": 5})  # cache hit
        # Only one actual run should have occurred
        assert ex.runs_used == 1

    def test_cache_can_be_disabled(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        ex.set_cache_enabled(False)
        ex.evaluate({"beam_size": 5})
        ex.evaluate({"beam_size": 5})
        ex.evaluate({"beam_size": 5})
        assert ex.runs_used == 3

    def test_phase_and_reasoning_propagated(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        t = ex.evaluate({"beam_size": 5}, phase="screening", reasoning="boundary min")
        assert t.phase == "screening"
        assert t.reasoning == "boundary min"

    def test_cached_result_uses_new_phase_reasoning(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        t1 = ex.evaluate({"beam_size": 5}, phase="screening", reasoning="first call")
        t2 = ex.evaluate({"beam_size": 5}, phase="local_1d", reasoning="second call")
        # Same numeric result, but the phase/reasoning should be updated
        assert t1.score == t2.score
        assert t2.phase == "local_1d"
        assert t2.reasoning == "second call"
        # Only one underlying run
        assert ex.runs_used == 1

    def test_noise_is_reproducible_per_config(self) -> None:
        """Same config + same seed must produce identical noisy results."""
        obj = SingleMetricObjective(metric="wer")
        ex1 = SyntheticTrialExecutor(
            metric_fn=_quadratic_wer, objective=obj, noise_std=0.01, seed=123
        )
        ex2 = SyntheticTrialExecutor(
            metric_fn=_quadratic_wer, objective=obj, noise_std=0.01, seed=123
        )
        t1 = ex1.evaluate({"beam_size": 5})
        t2 = ex2.evaluate({"beam_size": 5})
        assert t1.score == t2.score

    def test_noise_different_between_configs(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(
            metric_fn=_quadratic_wer, objective=obj, noise_std=0.05, seed=42
        )
        t_a = ex.evaluate({"beam_size": 5})
        t_b = ex.evaluate({"beam_size": 9})
        # Both have same raw quadratic value (5-7)^2 == (9-7)^2, but noise differs
        raw_a = 0.1 + 0.005 * 4
        raw_b = 0.1 + 0.005 * 4
        assert raw_a == raw_b
        assert t_a.score != t_b.score  # different noise seeds → different results

    def test_wer_clipped_at_zero_under_noise(self) -> None:
        """Heavy negative noise must not produce negative WER (physical constraint)."""
        obj = SingleMetricObjective(metric="wer")

        def zero_wer(_cfg: Mapping[str, Any]) -> Mapping[str, float]:
            del _cfg
            return {
                "wer": 0.001,
                "cer": 0.001,
                "mer": 0.001,
                "wil": 0.001,
                "rtfx_mean": 10.0,
                "vram_peak_mb": 1000.0,
                "wer_ci_lower": 0.0,
                "wer_ci_upper": 0.002,
            }

        ex = SyntheticTrialExecutor(metric_fn=zero_wer, objective=obj, noise_std=0.5, seed=7)
        for beam in range(1, 10):
            t = ex.evaluate({"beam_size": beam})
            wer_val = t.metrics["wer"]
            assert wer_val is not None
            assert wer_val >= 0.0

    def test_ci_included_in_result(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        t = ex.evaluate({"beam_size": 7})
        lo, hi = t.score_ci
        assert lo <= t.score <= hi

    def test_trial_id_is_set(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        t = ex.evaluate({"beam_size": 5})
        assert t.trial_id is not None
        assert t.trial_id.startswith("synthetic-")


class TestSyntheticExecutorLandscapes:
    """Verify the executor correctly routes metrics through the objective."""

    def test_weighted_objective_routing(self) -> None:
        from asrbench.engine.search.objective import WeightedObjective

        obj = WeightedObjective(weights={"wer": 1.0, "rtfx": -0.01})
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        t = ex.evaluate({"beam_size": 7})
        # score = 1.0 * 0.1 + (-0.01) * 20.0 = 0.1 - 0.2 = -0.1
        assert t.score == pytest.approx(-0.1)

    def test_maximize_objective_routing(self) -> None:
        obj = SingleMetricObjective(metric="rtfx")
        ex = SyntheticTrialExecutor(metric_fn=_quadratic_wer, objective=obj)
        t = ex.evaluate({"beam_size": 5})
        # RTFx is constant 20.0 in _quadratic_wer → internal score is -20.0
        assert t.score == pytest.approx(-20.0)
