"""
IAMS Layer 4 — multi-start sequential coordinate descent.

Coordinate descent is a local method: it converges to whatever basin it
starts in. When the landscape has multiple basins (classic interaction
landscapes do), a single run from the default baseline will miss the
global optimum. Layer 4 fixes this by running Layer 2's hybrid-sequential
descent from multiple starting points in parallel, then picking the best
result among all paths.

Starting points are assembled from:

    1. The "hybrid candidate" — the result of Layer 2 applied to the
       default baseline. This is always included (matches the non-multistart
       case).
    2. The "promising points" — off-diagonal cells from Layer 3's pairwise
       grids that were judged significantly below the univariate bound.
       These represent "landscape evidence" that a non-default configuration
       might dominate.
    3. (Optional) User-supplied overrides — future extensibility.

For each starting point, the multistart runs a SequentialCoordinateDescent:

    - Walk through sensitive parameters in descending sensitivity order.
    - For each parameter, 1D-search over its domain using search_1d().
    - After each parameter's search, update the running baseline to its
      newly-optimized value.
    - The running baseline is what the next parameter's search uses.

This is the "hybrid" from our earlier design discussion: independent screening
gives us the sensitivity order (removing the sequential-sort bias), then
sequential descent harvests additive improvements.

Deduplication:
    Two different starting points may converge to the same local minimum.
    The multistart detects this via TrialResult.config_key() and skips
    duplicate starts early — saves budget without affecting accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.local_1d import LocalSearchResult, search_1d
from asrbench.engine.search.significance import is_improvement
from asrbench.engine.search.space import ParameterSpace
from asrbench.engine.search.trial import TrialExecutor, TrialResult


@dataclass
class SequentialDescentResult:
    """Result of one start point's hybrid sequential descent."""

    start_label: str
    start_config: dict[str, Any]
    final_trial: TrialResult
    per_param: list[LocalSearchResult] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)


@dataclass
class MultiStartResult:
    """Aggregated result over all starting points."""

    descents: list[SequentialDescentResult] = field(default_factory=list)
    best_overall: TrialResult | None = None
    best_start_label: str = ""

    @property
    def all_trials(self) -> list[TrialResult]:
        out: list[TrialResult] = []
        for d in self.descents:
            out.extend(d.trials)
        return out


def sequential_descent(
    executor: TrialExecutor,
    space: ParameterSpace,
    budget: BudgetController,
    *,
    start_config: dict[str, Any],
    sensitive_order: list[str],
    start_trial: TrialResult,
    eps_min: float = 0.005,
    start_label: str = "",
    phase: str = "local_1d",
) -> SequentialDescentResult:
    """
    Run one pass of hybrid sequential coordinate descent.

    Walks through `sensitive_order`, optimizes each parameter with search_1d,
    and updates the running baseline config after each parameter so the next
    one benefits from the improved context.

    This IS Layer 2 of IAMS — multistart just calls it multiple times from
    different start configs and picks the best end state.
    """
    result = SequentialDescentResult(
        start_label=start_label,
        start_config=dict(start_config),
        final_trial=start_trial,
    )
    result.trials.append(start_trial)

    current_config = dict(start_config)
    current_best = start_trial

    for param_name in sensitive_order:
        if budget.should_stop():
            break
        spec = space.get(param_name)
        local = search_1d(
            executor,
            baseline_config=current_config,
            param=spec,
            budget=budget,
            eps_min=eps_min,
            phase=phase,
        )
        result.per_param.append(local)
        result.trials.extend(local.trials)

        # Update baseline if significantly better
        if is_improvement(local.best, current_best, eps_min=eps_min):
            current_best = local.best
            current_config = dict(local.best.config)
        elif local.best.score < current_best.score:
            # Not significant but still monotone — accept anyway so subsequent
            # coordinates search from the lower-score context.
            current_best = local.best
            current_config = dict(local.best.config)

    result.final_trial = current_best
    return result


class MultiStartSequentialDescent:
    """
    Layer 4 orchestrator.

    Given a list of start configs (with pre-evaluated trial results), run
    sequential coordinate descent from each and return the best final config
    across all starts.

    Dedupe strategy:
        Start configs that share the same config_key() as a previously-seen
        start are skipped entirely. This handles the common case where
        Layer 3's promising points coincide with the hybrid candidate.
    """

    def __init__(
        self,
        executor: TrialExecutor,
        space: ParameterSpace,
        budget: BudgetController,
        *,
        eps_min: float = 0.005,
    ) -> None:
        self.executor = executor
        self.space = space
        self.budget = budget
        self.eps_min = eps_min

    def run(
        self,
        sensitive_order: list[str],
        start_trials: list[TrialResult],
    ) -> MultiStartResult:
        """
        Run descent from every start trial in the list.

        `start_trials` are already-evaluated TrialResults. The caller is
        responsible for ranking or filtering them (e.g. picking top-M
        promising points from Layer 3). They are processed in list order,
        with duplicates skipped.
        """
        if not start_trials:
            raise ValueError(
                "MultiStartSequentialDescent: start_trials cannot be empty. "
                "Provide at least the hybrid candidate trial."
            )

        result = MultiStartResult()
        seen_keys: set[str] = set()

        for i, start in enumerate(start_trials):
            key = start.config_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if self.budget.should_stop():
                break

            label = f"start-{i}"
            descent = sequential_descent(
                self.executor,
                self.space,
                self.budget,
                start_config=dict(start.config),
                sensitive_order=sensitive_order,
                start_trial=start,
                eps_min=self.eps_min,
                start_label=label,
                phase="multistart",
            )
            result.descents.append(descent)

            if result.best_overall is None or descent.final_trial.score < result.best_overall.score:
                result.best_overall = descent.final_trial
                result.best_start_label = label

        return result
