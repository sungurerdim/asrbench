"""Unit tests for the (CI ∧ eps) significance gate."""

from __future__ import annotations

import pytest

from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.significance import (
    ci_overlap,
    compare,
    is_improvement,
    is_sensitive,
    sensitivity_score,
)
from asrbench.engine.search.trial import TrialResult


def _trial(score: float, ci_half: float = 0.001, config: dict | None = None) -> TrialResult:
    """Build a TrialResult with a symmetric CI for testing."""
    return TrialResult(
        config=config or {"x": 1},
        metrics={"wer": score},
        score=score,
        score_ci=(score - ci_half, score + ci_half),
    )


class TestCiOverlap:
    def test_disjoint_intervals(self) -> None:
        assert ci_overlap((0.0, 0.1), (0.2, 0.3)) is False

    def test_touching_intervals_overlap(self) -> None:
        # closed intervals: touching counts as overlap
        assert ci_overlap((0.0, 0.1), (0.1, 0.2)) is True

    def test_nested_intervals(self) -> None:
        assert ci_overlap((0.0, 1.0), (0.3, 0.7)) is True

    def test_identical_intervals(self) -> None:
        assert ci_overlap((0.0, 0.1), (0.0, 0.1)) is True

    def test_swapped_bounds_normalized(self) -> None:
        # Caller passes (hi, lo) — function must be forgiving
        assert ci_overlap((0.1, 0.0), (0.2, 0.3)) is False

    def test_point_inside_interval(self) -> None:
        assert ci_overlap((0.5, 0.5), (0.3, 0.7)) is True

    def test_point_outside_interval(self) -> None:
        assert ci_overlap((0.1, 0.1), (0.3, 0.7)) is False


class TestCompare:
    def test_both_gates_pass_a_better(self) -> None:
        a = _trial(score=0.05, ci_half=0.001)
        b = _trial(score=0.15, ci_half=0.001)
        v = compare(a, b, eps_min=0.005)
        assert v.significant is True
        assert v.a_better is True
        assert v.b_better is False
        assert "significant" in v.reason

    def test_both_gates_pass_b_better(self) -> None:
        a = _trial(score=0.15, ci_half=0.001)
        b = _trial(score=0.05, ci_half=0.001)
        v = compare(a, b, eps_min=0.005)
        assert v.significant is True
        assert v.b_better is True
        assert v.a_better is False

    def test_eps_gate_fails(self) -> None:
        # Tiny CI, but absolute difference below epsilon
        a = _trial(score=0.100, ci_half=0.0001)
        b = _trial(score=0.102, ci_half=0.0001)
        v = compare(a, b, eps_min=0.005)
        assert v.significant is False
        assert v.equivalent is True
        assert "would pass ε gate alone" not in v.reason  # eps gate actually failed
        assert "< ε" in v.reason

    def test_ci_gate_fails(self) -> None:
        # Large difference but overlapping CIs
        a = _trial(score=0.10, ci_half=0.05)  # [0.05, 0.15]
        b = _trial(score=0.13, ci_half=0.05)  # [0.08, 0.18]
        v = compare(a, b, eps_min=0.005)
        assert v.significant is False
        assert "CIs overlap" in v.reason

    def test_both_gates_fail(self) -> None:
        a = _trial(score=0.10, ci_half=0.02)
        b = _trial(score=0.101, ci_half=0.02)
        v = compare(a, b, eps_min=0.005)
        assert v.significant is False
        assert "CIs overlap" in v.reason
        assert "< ε" in v.reason

    def test_delta_sign(self) -> None:
        a = _trial(score=0.05)
        b = _trial(score=0.15)
        v = compare(a, b, eps_min=0.005)
        assert v.delta < 0

    def test_no_false_positive_on_tiny_noise(self) -> None:
        """Realistic scenario: near-identical configs with realistic bootstrap noise."""
        a = _trial(score=0.101, ci_half=0.008)  # [0.093, 0.109]
        b = _trial(score=0.100, ci_half=0.008)  # [0.092, 0.108]
        v = compare(a, b, eps_min=0.005)
        assert v.significant is False


class TestIsImprovement:
    def test_significantly_better_candidate(self) -> None:
        candidate = _trial(score=0.05)
        incumbent = _trial(score=0.15)
        assert is_improvement(candidate, incumbent, eps_min=0.005) is True

    def test_significantly_worse_candidate(self) -> None:
        candidate = _trial(score=0.15)
        incumbent = _trial(score=0.05)
        assert is_improvement(candidate, incumbent, eps_min=0.005) is False

    def test_equivalent_returns_false(self) -> None:
        candidate = _trial(score=0.100)
        incumbent = _trial(score=0.101)
        assert is_improvement(candidate, incumbent, eps_min=0.005) is False


class TestIsSensitive:
    def test_flat_landscape_insensitive(self) -> None:
        """If all three probes give essentially the same score, skip the param."""
        baseline = _trial(score=0.100)
        min_t = _trial(score=0.101)
        max_t = _trial(score=0.100)
        assert is_sensitive(baseline, min_t, max_t, eps_min=0.005) is False

    def test_monotonic_sensitive(self) -> None:
        """Classic 'bigger beam is better' — baseline vs max clearly differs."""
        baseline = _trial(score=0.12)
        min_t = _trial(score=0.15)
        max_t = _trial(score=0.08)
        assert is_sensitive(baseline, min_t, max_t, eps_min=0.005) is True

    def test_only_boundary_vs_boundary_significant(self) -> None:
        """min↔max differs but both are close to baseline individually."""
        baseline = _trial(score=0.10, ci_half=0.004)
        min_t = _trial(score=0.08, ci_half=0.001)
        max_t = _trial(score=0.12, ci_half=0.001)
        # baseline↔min overlap; baseline↔max overlap; but min↔max does not
        assert is_sensitive(baseline, min_t, max_t, eps_min=0.005) is True

    def test_u_shape_sensitive(self) -> None:
        """Middle-best pattern: baseline is best, both boundaries worse."""
        baseline = _trial(score=0.08)
        min_t = _trial(score=0.15)
        max_t = _trial(score=0.15)
        assert is_sensitive(baseline, min_t, max_t, eps_min=0.005) is True

    def test_inverted_u_sensitive(self) -> None:
        """Middle-worst pattern: both boundaries better than default."""
        baseline = _trial(score=0.15)
        min_t = _trial(score=0.08)
        max_t = _trial(score=0.08)
        assert is_sensitive(baseline, min_t, max_t, eps_min=0.005) is True


class TestSensitivityScore:
    def test_spread_is_max_minus_min(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        baseline = _trial(score=0.10)
        min_t = _trial(score=0.15)
        max_t = _trial(score=0.05)
        s = sensitivity_score(baseline, min_t, max_t, obj)
        assert s == pytest.approx(0.10)

    def test_zero_for_flat(self) -> None:
        obj = SingleMetricObjective(metric="wer")
        baseline = _trial(score=0.1)
        min_t = _trial(score=0.1)
        max_t = _trial(score=0.1)
        assert sensitivity_score(baseline, min_t, max_t, obj) == pytest.approx(0.0)

    def test_ordering_reflects_real_leverage(self) -> None:
        """Layer 2 uses this to pick which param to optimize first."""
        obj = SingleMetricObjective(metric="wer")
        # Param A: huge spread
        s_a = sensitivity_score(_trial(0.10), _trial(0.20), _trial(0.01), obj)
        # Param B: small spread
        s_b = sensitivity_score(_trial(0.10), _trial(0.11), _trial(0.09), obj)
        assert s_a > s_b
