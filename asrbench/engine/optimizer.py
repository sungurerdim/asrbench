"""
IAMS (Interaction-Aware Multi-Start) parameter optimizer orchestrator.

Runs the 7-layer algorithm end-to-end:

    Layer 1: Screening       → identify sensitive parameters
    Layer 2: Hybrid Sequential → first candidate via coordinate descent
    Layer 3: Pairwise 2D Grid → detect interactions, generate promising points
    Layer 4: Multi-Start     → escape local minima from multiple start points
    Layer 5: Deep Ablation   → detect toxic parameters, refine config
    Layer 6: High-Res Refine → tight-tolerance 1D search on top-sensitive params
    Layer 7: Validation      → variance-based confidence certification

Three accuracy modes:

    "fast"     : Layers 1-2 only (~ 6N runs)    — quick screening + coord descent
    "balanced" : Layers 1-5                     — adds interaction detection + ablation
    "maximum"  : Layers 1-7 (default, "correct >> fast")

Output:
    IAMSStudyResult — full trial log, best config, confidence label, per-layer
    sub-results. Designed to round-trip to JSON so the orchestrator can write
    a study.json file for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from asrbench.engine.search.ablation import AblationResult, DeepAblation
from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.local_1d import search_1d
from asrbench.engine.search.multistart import (
    MultiStartResult,
    MultiStartSequentialDescent,
    sequential_descent,
)
from asrbench.engine.search.objective import Objective
from asrbench.engine.search.pairwise_grid import (
    PairwiseGridResult,
    PairwiseGridScan,
)
from asrbench.engine.search.screening import ScreeningPhase, ScreeningResult
from asrbench.engine.search.space import ParameterSpace
from asrbench.engine.search.trial import TrialExecutor, TrialResult
from asrbench.engine.search.validation import ValidationPhase, ValidationResult

AccuracyMode = Literal["fast", "balanced", "maximum"]


@dataclass
class IAMSStudyResult:
    """
    Complete output of one IAMS optimization run.

    Fields:
        best_config: the lowest-scoring configuration found (fallback guarantee)
        best_trial: the TrialResult carrying best_config
        screening: Layer 1 result
        layer2_trial: the Layer 2 (hybrid sequential) candidate
        pairwise: Layer 3 result (None in "fast" mode)
        multistart: Layer 4 result (None in "fast" mode)
        ablation: Layer 5 result (None in "fast" mode)
        refined_trial: Layer 6 result (None in "fast"/"balanced" modes)
        validation: Layer 7 result (None in "fast"/"balanced" modes)
        all_trials: flat list of every trial evaluated across all layers
        insensitive_params: parameters fixed at default (Layer 1 verdict)
        mode: accuracy mode the study was run in
        total_trials: total number of trials across all layers
    """

    best_config: dict
    best_trial: TrialResult
    screening: ScreeningResult
    layer2_trial: TrialResult
    pairwise: PairwiseGridResult | None = None
    multistart: MultiStartResult | None = None
    ablation: AblationResult | None = None
    refined_trial: TrialResult | None = None
    validation: ValidationResult | None = None
    all_trials: list[TrialResult] = field(default_factory=list)
    insensitive_params: list[str] = field(default_factory=list)
    mode: AccuracyMode = "maximum"
    total_trials: int = 0
    reasoning: list[str] = field(default_factory=list)


class IAMSOptimizer:
    """
    Run a full IAMS optimization study against a ParameterSpace and Objective.

    Usage:
        optimizer = IAMSOptimizer(
            executor=synthetic_or_real_executor,
            space=ParameterSpace.from_yaml("space.yaml"),
            objective=SingleMetricObjective(metric="wer"),
            budget=BudgetController(hard_cap=200, convergence_eps=0.005),
            eps_min=0.005,
            mode="maximum",
        )
        result = optimizer.run()
        print(result.best_config)
    """

    def __init__(
        self,
        *,
        executor: TrialExecutor,
        space: ParameterSpace,
        objective: Objective,
        budget: BudgetController,
        eps_min: float = 0.005,
        mode: AccuracyMode = "maximum",
        top_k_pairs: int = 4,
        multistart_candidates: int = 3,
        validation_runs: int = 3,
        enable_deep_ablation: bool = False,
        prior_screening: ScreeningResult | None = None,
    ) -> None:
        if mode not in ("fast", "balanced", "maximum"):
            raise ValueError(
                f"IAMSOptimizer: unknown mode {mode!r}. Use 'fast', 'balanced', or 'maximum'."
            )
        self.executor = executor
        self.space = space
        self.objective = objective
        self.budget = budget
        self.eps_min = eps_min
        self.mode: AccuracyMode = mode
        self.top_k_pairs = top_k_pairs
        self.multistart_candidates = multistart_candidates
        self.validation_runs = validation_runs
        self.enable_deep_ablation = enable_deep_ablation
        self.prior_screening = prior_screening

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> IAMSStudyResult:
        reasoning: list[str] = []
        # Collect trial lists by reference — merge once at the end.
        trial_batches: list[list[TrialResult]] = []
        best = None  # type: TrialResult | None

        # ---- Layer 1: Screening ----
        if self.prior_screening is not None:
            screening = self.prior_screening
            reasoning.append(
                f"Layer 1 (screening): WARM START — reusing prior screening "
                f"({len(screening.sensitive_order)} sensitive, "
                f"insensitive: {screening.insensitive})"
            )
        else:
            screening = ScreeningPhase(
                self.executor, self.space, self.budget, eps_min=self.eps_min
            ).run()
            reasoning.append(
                f"Layer 1 (screening): {len(screening.sensitive_order)} sensitive "
                f"parameters in {len(screening.trials)} trials "
                f"(insensitive: {screening.insensitive})"
            )
        trial_batches.append(screening.trials)
        best = screening.baseline

        # If nothing is sensitive, the baseline is already the answer.
        if not screening.sensitive_order:
            reasoning.append("No sensitive parameters detected — baseline is optimal. Done.")
            best = screening.best_overall()
            return self._build_result(
                best,
                screening,
                reasoning,
                trial_batches,
                layer2_trial=best,
                insensitive_params=list(screening.insensitive),
            )

        # ---- Layer 2: Hybrid sequential descent ----
        baseline_config = dict(screening.baseline.config)
        layer2 = sequential_descent(
            self.executor,
            self.space,
            self.budget,
            start_config=baseline_config,
            sensitive_order=screening.sensitive_order,
            start_trial=screening.baseline,
            eps_min=self.eps_min,
            start_label="layer2-hybrid",
            phase="local_1d",
        )
        trial_batches.append(layer2.trials)
        best = self._update_best(best, layer2.final_trial)
        reasoning.append(
            f"Layer 2 (hybrid sequential): descent finished with score "
            f"{layer2.final_trial.score:.4f} after {len(layer2.trials)} trials"
        )

        if self.mode == "fast":
            reasoning.append("Mode 'fast' — stopping after Layer 2")
            return self._build_result(
                best,
                screening,
                reasoning,
                trial_batches,
                layer2_trial=layer2.final_trial,
                insensitive_params=list(screening.insensitive),
            )

        # ---- Layer 3: Pairwise 2D grid ----
        pairwise_scan = PairwiseGridScan(
            self.executor,
            self.space,
            self.budget,
            top_k=self.top_k_pairs,
            eps_min=self.eps_min,
            baseline_config=dict(best.config),
            baseline_score=best.score,
        )
        pairwise = pairwise_scan.run(sensitive_params=screening.sensitive_order)
        trial_batches.append(pairwise.all_trials)
        pair_best = pairwise.best_overall()
        if pair_best is not None:
            best = self._update_best(best, pair_best)
        promising = pairwise.promising_points()
        reasoning.append(
            f"Layer 3 (pairwise): scanned {len(pairwise.grids)} pairs, "
            f"found {len(promising)} promising off-diagonal points"
        )

        # ---- Layer 4: Multi-start ----
        effective_candidates = self.multistart_candidates
        if not promising:
            effective_candidates = 1
            reasoning.append("Layer 3 found no interactions — reducing multi-start to 1 candidate")
        promising_sorted = sorted(promising, key=lambda t: t.score)[:effective_candidates]
        start_trials = [layer2.final_trial] + promising_sorted
        multi = MultiStartSequentialDescent(
            self.executor, self.space, self.budget, eps_min=self.eps_min
        ).run(
            sensitive_order=screening.sensitive_order,
            start_trials=start_trials,
        )
        trial_batches.append(multi.all_trials)
        if multi.best_overall is not None:
            best = self._update_best(best, multi.best_overall)
        reasoning.append(
            f"Layer 4 (multistart): {len(multi.descents)} descents, winner={multi.best_start_label}"
        )

        # ---- Layer 5: Deep ablation ----
        ablation = DeepAblation(
            self.executor,
            self.space,
            self.budget,
            eps_min=self.eps_min,
            enable_pairs=self.enable_deep_ablation,
        ).run(
            incoming_best=best,
            sensitive_order=screening.sensitive_order,
        )
        trial_batches.append(ablation.all_trials)
        if ablation.best is not None:
            best = self._update_best(best, ablation.best)
        reasoning.append(
            f"Layer 5 (ablation): toxic_params={ablation.toxic_params}, reason={ablation.reason}"
        )

        if self.mode == "balanced":
            reasoning.append("Mode 'balanced' — stopping after Layer 5")
            return self._build_result(
                best,
                screening,
                reasoning,
                trial_batches,
                layer2_trial=layer2.final_trial,
                insensitive_params=list(screening.insensitive),
                pairwise=pairwise,
                multistart=multi,
                ablation=ablation,
            )

        # ---- Layer 6: High-res refinement ----
        refine_eps = self.eps_min / 2.0
        refine_baseline = dict(best.config)
        refine_best = best
        for name in screening.sensitive_order[: self.top_k_pairs]:
            if self.budget.should_stop():
                break
            spec = self.space.get(name)
            local = search_1d(
                self.executor,
                baseline_config=refine_baseline,
                param=spec,
                budget=self.budget,
                eps_min=refine_eps,
                phase="refinement",
            )
            trial_batches.append(local.trials)
            if local.best.score < refine_best.score:
                refine_best = local.best
                refine_baseline = dict(local.best.config)
        best = self._update_best(best, refine_best)
        reasoning.append(f"Layer 6 (refinement): final refined score {refine_best.score:.4f}")

        # ---- Layer 7: Validation ----
        validation = None
        if self.budget.can_run():
            validation = ValidationPhase(
                self.executor, self.budget, n_runs=self.validation_runs
            ).run(best)
            trial_batches.append(validation.validation_trials)
            reasoning.append(
                f"Layer 7 (validation): confidence={validation.confidence}, "
                f"mean={validation.mean_score:.4f}, cv={validation.coefficient_of_variation:.4f}"
            )
        else:
            reasoning.append("Layer 7 (validation): skipped, budget exhausted")

        return self._build_result(
            best,
            screening,
            reasoning,
            trial_batches,
            layer2_trial=layer2.final_trial,
            insensitive_params=list(screening.insensitive),
            pairwise=pairwise,
            multistart=multi,
            ablation=ablation,
            refined_trial=refine_best,
            validation=validation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        best: TrialResult,
        screening: ScreeningResult,
        reasoning: list[str],
        trial_batches: list[list[TrialResult]],
        *,
        layer2_trial: TrialResult | None = None,
        insensitive_params: list[str] | None = None,
        pairwise: PairwiseGridResult | None = None,
        multistart: MultiStartResult | None = None,
        ablation: AblationResult | None = None,
        refined_trial: TrialResult | None = None,
        validation: ValidationResult | None = None,
    ) -> IAMSStudyResult:
        """Assemble the final result, merging trial batches once."""
        all_trials: list[TrialResult] = []
        for batch in trial_batches:
            all_trials.extend(batch)
        return IAMSStudyResult(
            best_config=dict(best.config),
            best_trial=best,
            screening=screening,
            layer2_trial=layer2_trial or best,
            pairwise=pairwise,
            multistart=multistart,
            ablation=ablation,
            refined_trial=refined_trial,
            validation=validation,
            all_trials=all_trials,
            insensitive_params=insensitive_params or [],
            mode=self.mode,
            total_trials=len(all_trials),
            reasoning=reasoning,
        )

    @staticmethod
    def _update_best(current: TrialResult, candidate: TrialResult) -> TrialResult:
        """
        Global fallback guarantee: every best-update compares raw scores and
        returns the lower one. IAMS cannot silently drop an already-observed
        better candidate, regardless of what any intermediate layer reports.
        """
        return candidate if candidate.score < current.score else current
