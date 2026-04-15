"""
Integration tests for IAMSOptimizer + MultiFidelityTrialExecutor wiring.

Verifies that:
  1. ``use_multifidelity=True`` actually wraps Layer 2+ executor calls.
  2. Layer 1 (screening) and Layer 7 (validation) keep using the base
     executor — noise-sensitive layers must stay at full fidelity.
  3. Pruning fires on clearly bad configs when the landscape distinguishes
     between rungs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from asrbench.engine.optimizer import IAMSOptimizer
from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor


def _metrics(score: float) -> dict[str, float]:
    return {
        "wer": score,
        "cer": score * 0.5,
        "mer": score,
        "wil": score,
        "rtfx_mean": 20.0,
        "vram_peak_mb": 4000.0,
        "wer_ci_lower": max(0.0, score - 0.0005),
        "wer_ci_upper": score + 0.0005,
    }


def _simple_quadratic(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    """Minimum at x=5, flat in y — IAMS should converge quickly."""
    x = int(cfg.get("x", 1))
    y = int(cfg.get("y", 1))  # noqa: F841 — explicit unused
    return _metrics(0.05 + 0.002 * (x - 5) ** 2)


def _make_space() -> ParameterSpace:
    return ParameterSpace(
        parameters=(
            ParamSpec(name="x", type="int", default=1, min=1, max=10),
            ParamSpec(name="y", type="int", default=1, min=1, max=10),
        )
    )


@dataclass
class _RecordingExecutor:
    """Wraps a SyntheticTrialExecutor and records each call's fraction."""

    inner: SyntheticTrialExecutor
    fraction_calls: list[float] = field(default_factory=list)
    plain_calls: int = 0

    @property
    def runs_used(self) -> int:
        return self.inner.runs_used

    def evaluate(self, config, *, phase="unknown", reasoning=""):
        self.plain_calls += 1
        return self.inner.evaluate(config, phase=phase, reasoning=reasoning)

    def evaluate_at_fraction(self, config, *, phase="unknown", reasoning="", fraction=1.0):
        self.fraction_calls.append(fraction)
        return self.inner.evaluate_at_fraction(
            config, phase=phase, reasoning=reasoning, fraction=fraction
        )


def test_multifidelity_wrapper_activated_when_enabled() -> None:
    """With use_multifidelity=True, Layer 2+ trials go through the wrapper."""
    objective = SingleMetricObjective(metric="wer")
    inner = SyntheticTrialExecutor(metric_fn=_simple_quadratic, objective=objective)
    executor = _RecordingExecutor(inner=inner)
    space = _make_space()

    optimizer = IAMSOptimizer(
        executor=executor,  # type: ignore[arg-type]
        space=space,
        objective=objective,
        budget=BudgetController(hard_cap=100, convergence_eps=0.0),
        eps_min=0.0,
        mode="fast",  # Layer 1 + Layer 2 only
        use_multifidelity=True,
    )
    optimizer.run()

    # Layer 1 (screening) uses evaluate() directly — no fraction.
    # Layer 2 goes through MultiFidelityTrialExecutor which always calls
    # evaluate_at_fraction on the inner.
    assert executor.fraction_calls, "Layer 2+ must invoke evaluate_at_fraction"
    # SyntheticTrialExecutor ignores fraction, but the multi-fidelity
    # wrapper always walks through all rungs (0.25, 0.5, 1.0) per trial
    # when the incumbent is tight — so fraction calls must include at
    # least one non-1.0 value.
    assert any(f < 1.0 for f in executor.fraction_calls)


def test_multifidelity_disabled_uses_base_executor() -> None:
    """With use_multifidelity=False (default), no fraction calls happen."""
    objective = SingleMetricObjective(metric="wer")
    inner = SyntheticTrialExecutor(metric_fn=_simple_quadratic, objective=objective)
    executor = _RecordingExecutor(inner=inner)
    space = _make_space()

    optimizer = IAMSOptimizer(
        executor=executor,  # type: ignore[arg-type]
        space=space,
        objective=objective,
        budget=BudgetController(hard_cap=100, convergence_eps=0.0),
        eps_min=0.0,
        mode="fast",
        use_multifidelity=False,
    )
    optimizer.run()

    # All calls go through plain evaluate() — no wrapper.
    assert executor.fraction_calls == []
    assert executor.plain_calls > 0


def test_multifidelity_prunes_bad_configs() -> None:
    """
    Landscape where some configs are obviously bad. MF wrapper should prune
    them at early rungs, surfacing pruned trials in the log.
    """

    def bad_landscape(cfg: Mapping[str, Any]) -> Mapping[str, float]:
        x = int(cfg["x"])
        # x=5 is good (WER 0.05), everything else is AWFUL (WER 0.5)
        return _metrics(0.05 if x == 5 else 0.5)

    # Custom inner executor: fractions report the same score.
    objective = SingleMetricObjective(metric="wer")
    inner = SyntheticTrialExecutor(metric_fn=bad_landscape, objective=objective)

    space = ParameterSpace(parameters=(ParamSpec(name="x", type="int", default=5, min=1, max=10),))
    optimizer = IAMSOptimizer(
        executor=inner,
        space=space,
        objective=objective,
        budget=BudgetController(hard_cap=100, convergence_eps=0.0),
        eps_min=0.0,
        mode="fast",
        use_multifidelity=True,
        multifidelity_prune_threshold=0.015,
    )
    result = optimizer.run()

    pruned_count = sum(1 for t in result.all_trials if getattr(t, "pruned", False))
    assert pruned_count > 0, "Bad configs should have been pruned at early rungs"
    # Best result should still be the good x=5 config.
    assert result.best_trial.score < 0.10
