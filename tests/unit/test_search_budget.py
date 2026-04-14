"""Unit tests for BudgetController — hard cap + convergence-based stopping."""

from __future__ import annotations

import math

import pytest

from asrbench.engine.search.budget import BudgetController


class TestBudgetInit:
    def test_basic_init(self) -> None:
        b = BudgetController(hard_cap=50)
        assert b.runs_used == 0
        assert b.remaining == 50
        assert b.best_so_far() is None

    def test_rejects_zero_cap(self) -> None:
        with pytest.raises(ValueError, match="hard_cap must be positive"):
            BudgetController(hard_cap=0)

    def test_rejects_negative_cap(self) -> None:
        with pytest.raises(ValueError, match="hard_cap must be positive"):
            BudgetController(hard_cap=-5)

    def test_rejects_negative_eps(self) -> None:
        with pytest.raises(ValueError, match="convergence_eps must be >= 0"):
            BudgetController(hard_cap=10, convergence_eps=-0.001)


class TestCanRun:
    def test_can_run_single_under_cap(self) -> None:
        b = BudgetController(hard_cap=3)
        assert b.can_run() is True
        b.record(0.1)
        assert b.can_run() is True
        b.record(0.1)
        assert b.can_run() is True
        b.record(0.1)
        assert b.can_run() is False

    def test_can_run_batch(self) -> None:
        b = BudgetController(hard_cap=10)
        b.record(0.1)
        b.record(0.1)
        assert b.can_run(n=8) is True
        assert b.can_run(n=9) is False  # 2 + 9 = 11 > 10

    def test_remaining_counts_down(self) -> None:
        b = BudgetController(hard_cap=5)
        assert b.remaining == 5
        b.record(0.1)
        assert b.remaining == 4
        for _ in range(4):
            b.record(0.1)
        assert b.remaining == 0

    def test_remaining_never_negative(self) -> None:
        # Record more than cap (caller misuse) — remaining stays at 0, not negative
        b = BudgetController(hard_cap=3)
        for _ in range(5):
            b.record(0.1)
        assert b.remaining == 0


class TestRecord:
    def test_best_so_far_updates(self) -> None:
        b = BudgetController(hard_cap=10)
        b.record(0.2)
        assert b.best_so_far() == 0.2
        b.record(0.15)
        assert b.best_so_far() == 0.15
        b.record(0.18)  # worse
        assert b.best_so_far() == 0.15  # unchanged

    def test_history_reflects_best_not_raw(self) -> None:
        b = BudgetController(hard_cap=10)
        b.record(0.3)
        b.record(0.1)
        b.record(0.5)
        assert b.history == [0.3, 0.1, 0.1]
        assert b.raw_history == [0.3, 0.1, 0.5]

    def test_nan_score_clamped_to_inf(self) -> None:
        b = BudgetController(hard_cap=10)
        b.record(0.1)
        b.record(math.nan)
        assert b.best_so_far() == 0.1  # NaN never improves
        b.record(0.05)
        assert b.best_so_far() == 0.05

    def test_inf_score_clamped_to_inf(self) -> None:
        b = BudgetController(hard_cap=10)
        b.record(math.inf)
        b.record(0.3)
        assert b.best_so_far() == 0.3


class TestHasConverged:
    def test_not_enough_history(self) -> None:
        b = BudgetController(hard_cap=50, convergence_window=3)
        assert b.has_converged() is False  # empty
        for _ in range(3):
            b.record(0.1)
        # Need window+1 = 4 points; only 3 recorded
        assert b.has_converged() is False

    def test_converges_on_plateau(self) -> None:
        b = BudgetController(hard_cap=50, convergence_window=3, convergence_eps=0.005)
        for _ in range(4):
            b.record(0.100)
        assert b.has_converged() is True

    def test_does_not_converge_with_real_improvement(self) -> None:
        b = BudgetController(hard_cap=50, convergence_window=3, convergence_eps=0.005)
        # Each trial improves by 0.01 — well above eps
        for i in range(5):
            b.record(0.10 - 0.01 * i)
        assert b.has_converged() is False

    def test_converges_when_improvement_below_eps(self) -> None:
        b = BudgetController(hard_cap=50, convergence_window=3, convergence_eps=0.005)
        # Improvements of 0.001 each — below eps
        b.record(0.110)
        b.record(0.109)
        b.record(0.108)
        b.record(0.107)
        assert b.has_converged() is True

    def test_detects_worsening_as_no_progress(self) -> None:
        b = BudgetController(hard_cap=50, convergence_window=3, convergence_eps=0.005)
        b.record(0.10)
        b.record(0.12)
        b.record(0.15)
        b.record(0.20)
        # Best stays at 0.10, no improvement → converged
        assert b.has_converged() is True

    def test_window_zero_disables_convergence(self) -> None:
        b = BudgetController(hard_cap=50, convergence_window=0)
        for _ in range(10):
            b.record(0.1)
        assert b.has_converged() is False

    def test_window_negative_disables_convergence(self) -> None:
        b = BudgetController(hard_cap=50, convergence_window=-1)
        for _ in range(10):
            b.record(0.1)
        assert b.has_converged() is False


class TestShouldStop:
    def test_stops_on_cap(self) -> None:
        b = BudgetController(hard_cap=3, convergence_window=100)
        for _ in range(3):
            b.record(0.1)
        assert b.should_stop() is True  # cap reached

    def test_stops_on_convergence(self) -> None:
        b = BudgetController(hard_cap=100, convergence_window=2, convergence_eps=0.005)
        for _ in range(5):
            b.record(0.1)
        assert b.should_stop() is True  # converged

    def test_continues_on_progress_under_cap(self) -> None:
        b = BudgetController(hard_cap=100, convergence_window=2, convergence_eps=0.005)
        b.record(0.2)
        b.record(0.15)
        b.record(0.10)
        assert b.should_stop() is False


class TestSummary:
    def test_summary_contains_all_fields(self) -> None:
        b = BudgetController(hard_cap=20, convergence_eps=0.01, convergence_window=5)
        b.record(0.15)
        b.record(0.12)
        s = b.summary()
        assert s["hard_cap"] == 20
        assert s["runs_used"] == 2
        assert s["remaining"] == 18
        assert s["convergence_eps"] == 0.01
        assert s["convergence_window"] == 5
        assert s["best_so_far"] == 0.12
        assert s["has_converged"] is False
