"""
IAMS Layer 5 — deep ablation for 2+-way interaction detection.

After the multistart in Layer 4 produces a "best overall" candidate, Layer 5
asks a sharper question: are there any parameters in that candidate whose
presence is actually HARMING the score?

This happens when the coordinate-descent process assigns a value to a
parameter based on its univariate behavior, but a higher-order interaction
with OTHER parameters makes that value a net loss in the full combination.
Removing (= resetting to default) such "toxic" parameters improves the
final config.

Three levels of ablation, progressively expensive:

    1. Leave-one-out (always run): N trials, each replacing one parameter
       with its default. Any parameter whose removal improves the score is
       flagged as potentially toxic. Produces a "toxicity ranking".
    2. Leave-two-out (optional, triggered by top-K toxic pairs): K*(K-1)/2
       trials on the most toxic pair combinations. Catches two-way interaction
       effects that LOO misses.
    3. Leave-three-out (optional, cap-guarded): top-J triple combinations.
       Rarely necessary; enabled only when budget is abundant.

Refinement builder:
    Once toxicity is measured, the refinement builder constructs an alternative
    config by zeroing out (resetting to default) every parameter deemed toxic.
    The refined trial is evaluated and compared to the Layer 4 best — whichever
    has the lower score becomes the layer's output.

Fallback guarantee:
    The ablation layer NEVER makes things worse. If none of the ablations
    beats the incoming best, it simply returns the incoming best unchanged.
    The final output of IAMS always respects the invariant "result.score ≤
    all_trials.min(score)".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.significance import is_improvement
from asrbench.engine.search.space import ParameterSpace
from asrbench.engine.search.trial import TrialExecutor, TrialResult


@dataclass
class ParamToxicity:
    """Per-parameter outcome of leave-one-out ablation."""

    name: str
    original_value: Any
    ablated_trial: TrialResult
    toxicity: float  # incoming_best.score - ablated_trial.score; >0 means toxic
    is_toxic: bool


@dataclass
class AblationResult:
    """Full output of Layer 5."""

    incoming_best: TrialResult
    loo_results: list[ParamToxicity] = field(default_factory=list)
    loo_trials: list[TrialResult] = field(default_factory=list)
    pair_trials: list[TrialResult] = field(default_factory=list)
    refined_trial: TrialResult | None = None
    best: TrialResult | None = None
    toxic_params: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def all_trials(self) -> list[TrialResult]:
        out = list(self.loo_trials)
        out.extend(self.pair_trials)
        if self.refined_trial is not None:
            out.append(self.refined_trial)
        return out

    def final(self) -> TrialResult:
        """
        The lowest-scoring config seen during ablation, or the incoming best
        if nothing ablation-related beat it.
        """
        candidates = [self.incoming_best]
        candidates.extend(self.loo_trials)
        candidates.extend(self.pair_trials)
        if self.refined_trial is not None:
            candidates.append(self.refined_trial)
        return min(candidates, key=lambda t: t.score)


class DeepAblation:
    """
    Layer 5 orchestrator.

    Args:
        executor, space, budget: standard dependency injection
        eps_min: significance threshold for is_toxic classification
        enable_pairs: run leave-two-out (default False, off by default)
        max_pair_combinations: cap on the number of pair ablations to run
    """

    def __init__(
        self,
        executor: TrialExecutor,
        space: ParameterSpace,
        budget: BudgetController,
        *,
        eps_min: float = 0.005,
        enable_pairs: bool = False,
        max_pair_combinations: int = 6,
    ) -> None:
        self.executor = executor
        self.space = space
        self.budget = budget
        self.eps_min = eps_min
        self.enable_pairs = enable_pairs
        self.max_pair_combinations = max_pair_combinations

    def run(
        self,
        incoming_best: TrialResult,
        sensitive_order: list[str],
    ) -> AblationResult:
        """
        Run leave-one-out ablation over sensitive parameters, then optionally
        leave-two-out on the most toxic pairs, then build a refined config.
        """
        result = AblationResult(incoming_best=incoming_best)
        defaults = self.space.defaults()

        # --- Leave-one-out ---
        for name in sensitive_order:
            if not self.budget.can_run():
                break
            if name not in incoming_best.config:
                continue
            original = incoming_best.config[name]
            default_val = defaults[name]
            if original == default_val:
                # Parameter is already at its default → no-op, skip
                continue

            ablated_config = dict(incoming_best.config)
            ablated_config[name] = default_val
            ablated = self.executor.evaluate(
                ablated_config,
                phase="ablation",
                reasoning=(
                    f"leave-one-out: revert '{name}' from {original!r} to default {default_val!r}"
                ),
            )
            self.budget.record(ablated.score)
            result.loo_trials.append(ablated)

            # Toxicity > 0 means ablation improved the score
            toxicity = incoming_best.score - ablated.score
            toxic = is_improvement(ablated, incoming_best, eps_min=self.eps_min)

            result.loo_results.append(
                ParamToxicity(
                    name=name,
                    original_value=original,
                    ablated_trial=ablated,
                    toxicity=toxicity,
                    is_toxic=toxic,
                )
            )
            if toxic:
                result.toxic_params.append(name)

        # --- Leave-two-out (optional) ---
        if self.enable_pairs and len(result.toxic_params) >= 2:
            toxic_sorted = sorted(
                (r for r in result.loo_results if r.is_toxic),
                key=lambda r: -r.toxicity,
            )
            top_toxic = [r.name for r in toxic_sorted[: self.max_pair_combinations]]
            pairs_done = 0
            for a, b in combinations(top_toxic, 2):
                if pairs_done >= self.max_pair_combinations:
                    break
                if not self.budget.can_run():
                    break
                ablated_config = dict(incoming_best.config)
                ablated_config[a] = defaults[a]
                ablated_config[b] = defaults[b]
                trial = self.executor.evaluate(
                    ablated_config,
                    phase="ablation",
                    reasoning=f"leave-two-out: revert '{a}' and '{b}' to defaults",
                )
                self.budget.record(trial.score)
                result.pair_trials.append(trial)
                pairs_done += 1

        # --- Refinement: revert all toxic parameters at once ---
        if result.toxic_params and self.budget.can_run():
            refined_config = dict(incoming_best.config)
            for name in result.toxic_params:
                refined_config[name] = defaults[name]
            refined = self.executor.evaluate(
                refined_config,
                phase="refinement",
                reasoning=(
                    "refinement: revert all toxic parameters "
                    f"({', '.join(result.toxic_params)}) to defaults simultaneously"
                ),
            )
            self.budget.record(refined.score)
            result.refined_trial = refined

        # --- Pick the best ---
        result.best = result.final()
        if result.best is incoming_best:
            result.reason = "incoming best retained (no ablation improved the score)"
        elif result.best is result.refined_trial:
            result.reason = (
                f"refined config wins by reverting {len(result.toxic_params)} "
                f"toxic params: {', '.join(result.toxic_params)}"
            )
        elif result.best in result.pair_trials:
            result.reason = "pair ablation wins"
        else:
            result.reason = "single leave-one-out ablation wins"

        return result
