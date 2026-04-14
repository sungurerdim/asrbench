"""
IAMS Layer 1 — independent OFAT-3 screening.

Screens every parameter at three points (min, default, max) against a fixed
baseline of defaults. Produces:

    1. A baseline trial (all parameters at default)
    2. Two boundary trials per parameter (min and max), with every OTHER
       parameter held at its default
    3. A sensitivity verdict per parameter (sensitive or insensitive)
    4. A sensitivity score per parameter (max - min of the 3 probes)
    5. A sorted list of sensitive parameters in descending leverage order

Total trials: 1 + 2N for N parameters. The baseline is run first and cached
against further phases via the TrialExecutor's internal config-level cache —
so if Layer 2 or Layer 4 later asks for the same default config, no extra run
happens.

Decision semantics:
    - Sensitivity requires is_sensitive() from significance.py, which itself
      demands (CI non-overlap AND |Δ| ≥ eps_min) in at least one of the three
      pairwise comparisons (baseline↔min, baseline↔max, min↔max).
    - A parameter judged insensitive is NEVER re-examined by subsequent layers.
      This is the screening phase's contract: when in doubt, keep it — the
      AND-gate is conservative enough that "false negatives" (real effects
      missed) are vanishingly rare in practice.

Interaction with budget:
    The screening phase consumes 1 + 2N trials unconditionally. If the budget
    controller has fewer than 1 + 2N trials available, screening aborts with
    a partial result and flags the remaining parameters as "unknown sensitivity"
    so downstream layers know to treat them cautiously. This respects the
    user's hard budget cap even when the space is underspecified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.significance import (
    is_sensitive as _is_sensitive,
)
from asrbench.engine.search.significance import (
    sensitivity_score as _sensitivity_score,
)
from asrbench.engine.search.space import ParameterSpace
from asrbench.engine.search.trial import TrialExecutor, TrialResult


@dataclass
class ParamScreening:
    """Per-parameter screening outcome."""

    name: str
    baseline: TrialResult
    min_trial: TrialResult
    max_trial: TrialResult
    sensitive: bool
    sensitivity: float
    screened: bool = True  # False if budget ran out before this parameter

    def best_probe(self) -> TrialResult:
        """Return the lowest-scoring probe among {baseline, min, max}."""
        return min(
            (self.baseline, self.min_trial, self.max_trial),
            key=lambda t: t.score,
        )


@dataclass
class ScreeningResult:
    """Full output of the screening phase, consumed by Layer 2."""

    baseline: TrialResult
    params: dict[str, ParamScreening] = field(default_factory=dict)
    sensitive_order: list[str] = field(default_factory=list)
    insensitive: list[str] = field(default_factory=list)
    unscreened: list[str] = field(default_factory=list)  # budget ran out
    trials: list[TrialResult] = field(default_factory=list)

    @classmethod
    def from_summary(cls, summary: dict, baseline: TrialResult) -> ScreeningResult:
        """Reconstruct a minimal ScreeningResult from persisted summary (warm start)."""
        return cls(
            baseline=baseline,
            params={},
            sensitive_order=list(summary["sensitive_order"]),
            insensitive=list(summary["insensitive"]),
            unscreened=[],
            trials=[],
        )

    @property
    def sensitive(self) -> list[str]:
        """Alias: the list of parameter names the rest of IAMS should optimize."""
        return list(self.sensitive_order)

    def boundary_trials(self, name: str) -> tuple[TrialResult, TrialResult]:
        """Return (min_trial, max_trial) for a parameter — reused by Layer 2."""
        ps = self.params[name]
        return (ps.min_trial, ps.max_trial)

    def best_overall(self) -> TrialResult:
        """Lowest-scoring trial seen in screening (for fallback guarantee)."""
        return min(self.trials, key=lambda t: t.score)


class ScreeningPhase:
    """
    Layer 1 orchestrator.

    Dependencies are injected (executor, space, budget) so the phase is fully
    unit-testable with a SyntheticTrialExecutor.

    Usage:
        phase = ScreeningPhase(executor, space, budget, eps_min=0.005)
        result = phase.run()

    After run():
        result.sensitive_order gives the descending-sensitivity parameter list
        result.params[name]    has the full per-parameter screening record
        result.baseline        is the all-defaults trial used as reference
    """

    def __init__(
        self,
        executor: TrialExecutor,
        space: ParameterSpace,
        budget: BudgetController,
        eps_min: float = 0.005,
    ) -> None:
        self.executor = executor
        self.space = space
        self.budget = budget
        self.eps_min = eps_min

    def run(self) -> ScreeningResult:
        # === Baseline ===
        baseline_config = self.space.defaults()
        if not self.budget.can_run():
            raise RuntimeError(
                "BudgetController has zero remaining runs at the start of "
                "screening. Increase --budget to at least 1 + 2N."
            )
        baseline = self.executor.evaluate(
            baseline_config,
            phase="screening",
            reasoning="baseline (all parameters at default)",
        )
        self.budget.record(baseline.score)

        result = ScreeningResult(baseline=baseline)
        result.trials.append(baseline)

        # === Boundary probes per parameter ===
        for spec in self.space.parameters:
            # Budget check: need 2 more runs for this parameter
            if not self.budget.can_run(n=2):
                result.unscreened.append(spec.name)
                continue

            # Skip parameters with degenerate ranges (e.g. min == max == default).
            # They can't be screened at three distinct points, and the optimizer
            # should treat them as fixed.
            min_val, max_val = self._probe_points(spec)
            if min_val == spec.default and max_val == spec.default:
                # Degenerate: reuse the baseline trial for both ends so downstream
                # data structures stay consistent.
                degenerate = ParamScreening(
                    name=spec.name,
                    baseline=baseline,
                    min_trial=baseline,
                    max_trial=baseline,
                    sensitive=False,
                    sensitivity=0.0,
                )
                result.params[spec.name] = degenerate
                result.insensitive.append(spec.name)
                continue

            min_config = dict(baseline_config)
            min_config[spec.name] = min_val
            min_trial = self.executor.evaluate(
                min_config,
                phase="screening",
                reasoning=f"boundary min for '{spec.name}' ({spec.name}={min_val})",
            )
            self.budget.record(min_trial.score)
            result.trials.append(min_trial)

            max_config = dict(baseline_config)
            max_config[spec.name] = max_val
            max_trial = self.executor.evaluate(
                max_config,
                phase="screening",
                reasoning=f"boundary max for '{spec.name}' ({spec.name}={max_val})",
            )
            self.budget.record(max_trial.score)
            result.trials.append(max_trial)

            sensitive = _is_sensitive(baseline, min_trial, max_trial, eps_min=self.eps_min)
            sensitivity = _sensitivity_score(baseline, min_trial, max_trial, None)  # type: ignore[arg-type]

            ps = ParamScreening(
                name=spec.name,
                baseline=baseline,
                min_trial=min_trial,
                max_trial=max_trial,
                sensitive=sensitive,
                sensitivity=sensitivity,
            )
            result.params[spec.name] = ps
            if sensitive:
                result.sensitive_order.append(spec.name)
            else:
                result.insensitive.append(spec.name)

        # Sort sensitive params by sensitivity score descending — Layer 2
        # optimizes the highest-leverage parameter first.
        result.sensitive_order.sort(key=lambda n: result.params[n].sensitivity, reverse=True)
        return result

    @staticmethod
    def _probe_points(spec: Any) -> tuple[Any, Any]:
        """
        Pick the two boundary probe points for a parameter.

        For float/int: literal min and max from the ParamSpec.
        For bool: False and True.
        For enum: first and last value in the declared list. If the list has
                  only one element, returns (default, default) — a degenerate
                  case flagged upstream.
        """
        if spec.type in ("float", "int"):
            return spec.min, spec.max
        if spec.type == "bool":
            return False, True
        if spec.type == "enum":
            assert spec.values is not None
            if len(spec.values) == 1:
                return spec.default, spec.default
            return spec.values[0], spec.values[-1]
        raise ValueError(f"Unknown param type {spec.type!r} in screening")
