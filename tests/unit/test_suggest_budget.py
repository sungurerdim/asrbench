"""Tests for BudgetController.suggest — space-size-driven budget auto-sizing."""

from __future__ import annotations

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.space import ParameterSpace, ParamSpec


def _make_space(n_params: int) -> ParameterSpace:
    specs = [ParamSpec(name=f"p{i}", type="int", default=1, min=1, max=5) for i in range(n_params)]
    return ParameterSpace(parameters=tuple(specs))


def test_suggest_coarse_covers_screening_and_early_l2() -> None:
    # N=20 → L1: 1 + 2*20 = 41, L2 coarse: max(15, 10+10) = 20 → 61
    space = _make_space(n_params=20)
    budget = BudgetController.suggest(space, phase="coarse")
    assert budget == 61


def test_suggest_coarse_warm_start_skips_l1() -> None:
    # N=20, warm start → L1 dropped → just L2 coarse (20)
    space = _make_space(n_params=20)
    budget = BudgetController.suggest(space, phase="coarse", warm_start=True)
    assert budget == 20


def test_suggest_fine_floored_at_30() -> None:
    # Tiny space → fine budget floored at 30 so L3-L7 have room
    tiny = _make_space(n_params=2)
    assert BudgetController.suggest(tiny, phase="fine") == 30


def test_suggest_fine_scales_with_space_size() -> None:
    # N=25 → max(30, 25+20) = 45
    space = _make_space(n_params=25)
    assert BudgetController.suggest(space, phase="fine") == 45


def test_suggest_coarse_tiny_space_still_has_margin() -> None:
    # N=1 → L1 = 1+2 = 3, L2 coarse floored at 15 → 18
    tiny = _make_space(n_params=1)
    assert BudgetController.suggest(tiny, phase="coarse") == 18
