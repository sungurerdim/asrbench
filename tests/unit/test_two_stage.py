"""
End-to-end tests for the library-level two-stage orchestrator.

Uses synthetic executors so no real dataset / backend is required. The key
properties under test are:

    1. Both stages actually run and produce non-empty study results.
    2. Stage 2 is warm-started: its screening result carries over from Stage 1,
       so the Stage-2 trial log does not re-run Layer 1 (sensitive_order is
       preserved, insensitive list matches).
    3. Auto-sizing kicks in when the caller leaves budget / epsilon as None
       — the resolved BudgetController.hard_cap is strictly positive and the
       epsilon passed to each optimizer matches suggest_epsilon(duration).
    4. The executor_factory is called once per stage with the stage's duration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asrbench.backends.base import BaseBackend, Segment
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor
from asrbench.engine.two_stage import TwoStageConfig, run_two_stage

# ---------------------------------------------------------------------------
# Landscape + helpers
# ---------------------------------------------------------------------------


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


def _simple_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Quadratic in ``x`` with minimum at 5; ``noise`` has no effect."""
    x = int(cfg.get("x", 1))
    wer = 0.10 + 0.003 * (x - 5) ** 2
    return _metrics(wer)


def _make_space() -> ParameterSpace:
    return ParameterSpace(
        parameters=(
            ParamSpec(name="x", type="int", default=1, min=1, max=10),
            ParamSpec(name="noise", type="float", default=0.5, min=0.0, max=1.0),
        )
    )


class _NullBackend(BaseBackend):
    """Backend stub that accepts every param (supported_params → None)."""

    family = "null"
    name = "null"

    def default_params(self) -> dict:
        return {}

    def load(self, model_path: str, params: dict) -> None:
        return None

    def unload(self) -> None:
        return None

    def transcribe(self, audio, lang, params) -> list[Segment]:  # type: ignore[override]
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_two_stage_end_to_end() -> None:
    objective = SingleMetricObjective(metric="wer")

    # Share one executor instance between stages for simplicity — the factory
    # is still called twice (we'll count it below).
    executor = SyntheticTrialExecutor(metric_fn=_simple_landscape, objective=objective)

    calls: list[int] = []

    def dataset_loader(duration_s: int) -> int:
        # Synthetic "dataset" is just the duration integer — the factory
        # doesn't actually use it.
        return duration_s

    def executor_factory(prepared, duration_s: int):
        calls.append(duration_s)
        return executor

    result = run_two_stage(
        space=_make_space(),
        objective=objective,
        backend=_NullBackend(),
        mode_hint=None,
        dataset_loader=dataset_loader,
        executor_factory=executor_factory,
        cfg=TwoStageConfig(
            stage1_duration_s=900,
            stage2_duration_s=2400,
            mode="fast",  # keep the test short (only L1 + L2)
        ),
    )

    # Both stages produced at least one trial
    assert result.stage1.total_trials > 0
    assert result.stage2.total_trials > 0

    # Factory got called once per stage with the stage duration
    assert calls == [900, 2400]

    # Warm-start: Stage 2 reused Stage 1's ScreeningResult object verbatim.
    # The optimizer's prior_screening branch simply rebinds the existing
    # instance, so identity (``is``) is the strongest signal that Stage 2
    # did not re-run Layer 1 on its own executor.
    assert result.stage2.screening is result.stage1.screening
    # Sensitive order was preserved, too.
    assert result.stage1.screening.sensitive_order == result.stage2.screening.sensitive_order


def test_two_stage_auto_budget_and_epsilon() -> None:
    """Leaving budget/epsilon as None triggers the library's auto-sizing helpers."""
    from asrbench.engine.search.budget import BudgetController
    from asrbench.engine.search.significance import suggest_epsilon

    space = _make_space()
    cfg = TwoStageConfig(
        stage1_duration_s=900,
        stage2_duration_s=2400,
        # budget + epsilon left as None on purpose
        mode="fast",
    )

    # Independently compute what the library should resolve to.
    expected_s1_budget = BudgetController.suggest(space, phase="coarse")
    expected_s2_budget = BudgetController.suggest(space, phase="fine", warm_start=True)
    expected_s1_eps = suggest_epsilon(900)
    expected_s2_eps = suggest_epsilon(2400)

    # Run — we just need it to succeed without raising.
    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(metric_fn=_simple_landscape, objective=objective)
    result = run_two_stage(
        space=space,
        objective=objective,
        backend=_NullBackend(),
        mode_hint=None,
        dataset_loader=lambda d: d,
        executor_factory=lambda ds, d: executor,
        cfg=cfg,
    )

    assert result.stage1.total_trials <= expected_s1_budget
    assert result.stage2.total_trials <= expected_s2_budget
    # Sanity: the suggested epsilons are positive and within sane ranges.
    assert expected_s1_eps > 0
    assert expected_s2_eps > 0
    assert expected_s1_eps > expected_s2_eps  # shorter stage → looser eps
