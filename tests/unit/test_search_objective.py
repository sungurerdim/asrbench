"""Unit tests for Objective — SingleMetric and Weighted scoring + CI propagation."""

from __future__ import annotations

import pytest

from asrbench.engine.search.objective import (
    SingleMetricObjective,
    WeightedObjective,
)


def _metrics(
    wer: float = 0.1,
    cer: float = 0.05,
    rtfx_mean: float = 20.0,
    vram_peak_mb: float = 4000.0,
    wer_ci_lower: float | None = None,
    wer_ci_upper: float | None = None,
) -> dict:
    """Build a metrics dict with sensible defaults for tests."""
    return {
        "wer": wer,
        "cer": cer,
        "mer": wer,  # close enough for tests
        "wil": wer,
        "rtfx_mean": rtfx_mean,
        "vram_peak_mb": vram_peak_mb,
        "wer_ci_lower": wer_ci_lower,
        "wer_ci_upper": wer_ci_upper,
    }


class TestSingleMetricObjective:
    def test_wer_default_minimize(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        assert obj.direction == "minimize"
        assert obj.score(_metrics(wer=0.15)) == pytest.approx(0.15)

    def test_rtfx_default_maximize(self) -> None:
        obj = SingleMetricObjective(metric="rtfx")
        assert obj.direction == "maximize"
        # Internally minimize -rtfx
        assert obj.score(_metrics(rtfx_mean=25.0)) == pytest.approx(-25.0)

    def test_vram_default_minimize(self) -> None:
        obj = SingleMetricObjective(metric="vram")
        assert obj.score(_metrics(vram_peak_mb=3500.0)) == pytest.approx(3500.0)

    def test_explicit_direction_override(self) -> None:
        obj = SingleMetricObjective(metric="wer", direction="maximize")
        assert obj.score(_metrics(wer=0.2)) == pytest.approx(-0.2)

    def test_lower_wer_is_better_minimize(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        s_good = obj.score(_metrics(wer=0.05))
        s_bad = obj.score(_metrics(wer=0.20))
        assert s_good < s_bad, "Minimize: lower WER must give lower score"

    def test_higher_rtfx_is_better_maximize(self) -> None:
        obj = SingleMetricObjective(metric="rtfx")
        s_good = obj.score(_metrics(rtfx_mean=30.0))
        s_bad = obj.score(_metrics(rtfx_mean=10.0))
        assert s_good < s_bad, "Maximize: higher RTFx must give lower score internally"

    def test_unknown_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            SingleMetricObjective(metric="ppl")

    def test_bad_direction_rejected(self) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            SingleMetricObjective(metric="wer", direction="up")  # type: ignore[arg-type]

    def test_missing_metric_in_dict_raises(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        with pytest.raises(KeyError, match="not present in trial metrics"):
            obj.score({"cer": 0.1})

    def test_none_metric_value_raises(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        m = _metrics()
        m["wer"] = None
        with pytest.raises(ValueError, match="is None in trial metrics"):
            obj.score(m)

    def test_ci_wer_minimize_uses_bootstrap_bounds(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        m = _metrics(wer=0.15, wer_ci_lower=0.12, wer_ci_upper=0.18)
        lo, hi = obj.score_ci(m)
        assert lo == pytest.approx(0.12)
        assert hi == pytest.approx(0.18)

    def test_ci_wer_maximize_flips_interval(self) -> None:
        # Unusual but supported: maximize WER
        obj = SingleMetricObjective(metric="wer", direction="maximize")
        m = _metrics(wer=0.15, wer_ci_lower=0.12, wer_ci_upper=0.18)
        lo, hi = obj.score_ci(m)
        # score = -0.15, interval becomes [-0.18, -0.12]
        assert lo == pytest.approx(-0.18)
        assert hi == pytest.approx(-0.12)
        assert lo <= obj.score(m) <= hi

    def test_ci_non_wer_metric_is_degenerate(self) -> None:
        obj = SingleMetricObjective(metric="rtfx")
        m = _metrics(rtfx_mean=25.0)
        lo, hi = obj.score_ci(m)
        assert lo == hi == obj.score(m)

    def test_ci_wer_without_bounds_degenerate(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        m = _metrics(wer=0.1, wer_ci_lower=None, wer_ci_upper=None)
        lo, hi = obj.score_ci(m)
        assert lo == hi == 0.1

    def test_describe(self) -> None:
        assert "minimize" in SingleMetricObjective(metric="wer").describe()
        assert "maximize" in SingleMetricObjective(metric="rtfx").describe()


class TestWeightedObjective:
    def test_basic_positive_weights(self) -> None:
        obj = WeightedObjective(weights={"wer": 1.0, "vram": 0.0001})
        score = obj.score(_metrics(wer=0.1, vram_peak_mb=4000.0))
        # 1.0 * 0.1 + 0.0001 * 4000 = 0.1 + 0.4 = 0.5
        assert score == pytest.approx(0.5)

    def test_negative_weight_maximizes(self) -> None:
        obj = WeightedObjective(weights={"wer": 1.0, "rtfx": -0.01})
        m_slow = _metrics(wer=0.1, rtfx_mean=5.0)
        m_fast = _metrics(wer=0.1, rtfx_mean=50.0)
        # slow: 0.1 - 0.01*5 = 0.05;  fast: 0.1 - 0.01*50 = -0.4
        assert obj.score(m_slow) > obj.score(m_fast)

    def test_empty_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            WeightedObjective(weights={})

    def test_unknown_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            WeightedObjective(weights={"bleu": 1.0})

    def test_non_numeric_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            WeightedObjective(weights={"wer": "1.0"})  # type: ignore[dict-item]

    def test_zero_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="is 0, which makes"):
            WeightedObjective(weights={"wer": 1.0, "cer": 0.0})

    def test_bool_weight_rejected(self) -> None:
        # Guard against Python bool-is-int trap
        with pytest.raises(ValueError, match="must be numeric"):
            WeightedObjective(weights={"wer": True})  # type: ignore[dict-item]

    def test_ci_propagates_wer_half_width(self) -> None:
        obj = WeightedObjective(weights={"wer": 2.0, "rtfx": -0.01})
        m = _metrics(wer=0.1, rtfx_mean=20.0, wer_ci_lower=0.08, wer_ci_upper=0.12)
        score = obj.score(m)
        lo, hi = obj.score_ci(m)
        # Only WER has CI. half_width = |2.0| * (0.12 - 0.08) / 2 = 0.04
        assert lo == pytest.approx(score - 0.04)
        assert hi == pytest.approx(score + 0.04)
        assert lo <= score <= hi

    def test_ci_without_wer_weight_degenerate(self) -> None:
        obj = WeightedObjective(weights={"rtfx": -0.1})
        m = _metrics(rtfx_mean=25.0)
        lo, hi = obj.score_ci(m)
        assert lo == hi == obj.score(m)

    def test_ci_wer_weight_but_no_bounds_degenerate(self) -> None:
        obj = WeightedObjective(weights={"wer": 1.0})
        m = _metrics(wer=0.1, wer_ci_lower=None, wer_ci_upper=None)
        lo, hi = obj.score_ci(m)
        assert lo == hi == obj.score(m)

    def test_describe_contains_all_terms(self) -> None:
        obj = WeightedObjective(weights={"wer": 1.0, "rtfx": -0.1})
        desc = obj.describe()
        assert "wer" in desc
        assert "rtfx" in desc
        assert "-" in desc  # the negative sign for rtfx


class TestScoreComparisonSemantics:
    """Invariant checks across both objectives: lower score = better config."""

    def test_single_metric_lower_is_better(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        worse = _metrics(wer=0.20)
        better = _metrics(wer=0.05)
        assert obj.score(better) < obj.score(worse)

    def test_weighted_lower_is_better(self) -> None:
        obj = WeightedObjective(weights={"wer": 1.0, "rtfx": -0.05})
        worse = _metrics(wer=0.20, rtfx_mean=5.0)  # high wer, slow → 0.2 - 0.25 = -0.05
        better = _metrics(wer=0.05, rtfx_mean=50.0)  # low wer, fast → 0.05 - 2.5 = -2.45
        assert obj.score(better) < obj.score(worse)
