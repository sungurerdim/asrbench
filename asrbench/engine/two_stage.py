"""
Two-stage IAMS optimization orchestrator.

This module encapsulates the "coarse → fine" / Hyperband-inspired optimization
flow that previously lived inside ``optimize_matrix.py`` as a script-level
loop. Keeping it in the library means the same code path is exercised by:

    - ``optimize_matrix.py`` (CLI matrix runner)
    - ``asrbench/api/optimization.py`` (REST + Svelte UI)
    - Notebook / programmatic callers that want to reuse the flow

Flow:

    Stage 1 (coarse)
        Short dataset (e.g. 900 s), loose epsilon, larger trial budget.
        Runs the full 7-layer IAMS with no warm start. Cheap screening +
        initial exploration.

    Stage 2 (fine)
        Longer dataset (e.g. 2400 s), tight epsilon, smaller budget.
        Warm-started from Stage 1: screening metadata (sensitive_order,
        insensitive) is reused so Stage 2 skips L1 entirely and spends its
        entire budget on refinement + validation.

Budget and epsilon default to auto-sized values derived from the space and
the stage duration (see ``BudgetController.suggest`` / ``suggest_epsilon``),
so the caller only has to supply the corpus durations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from asrbench.engine.optimizer import IAMSOptimizer, IAMSStudyResult
from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import Objective
from asrbench.engine.search.significance import suggest_epsilon
from asrbench.engine.search.space import ParameterSpace
from asrbench.engine.search.trial import TrialExecutor

if TYPE_CHECKING:
    from asrbench.backends.base import BaseBackend

ExecutorFactory = Callable[[Any, int], TrialExecutor]
DatasetLoader = Callable[[int], Any]


@dataclass(frozen=True)
class TwoStageConfig:
    """
    User-tunable knobs for the two-stage flow.

    All durations are in seconds. Leaving ``stage*_budget`` / ``stage*_epsilon``
    as None delegates sizing to the library helpers — recommended unless the
    caller has a specific reason to override.
    """

    stage1_duration_s: int = 900
    stage2_duration_s: int = 2400
    stage1_budget: int | None = None
    stage2_budget: int | None = None
    stage1_epsilon: float | None = None
    stage2_epsilon: float | None = None
    mode: str = "maximum"
    top_k_pairs: int = 4
    multistart_candidates: int = 3
    validation_runs: int = 3
    enable_deep_ablation: bool = False
    use_multifidelity: bool = False
    multifidelity_rungs: tuple[float, ...] = (0.25, 0.5, 1.0)
    multifidelity_prune_threshold: float = 0.015


@dataclass
class TwoStageResult:
    """Outcome of a complete two-stage run — both stages plus the final best."""

    stage1: IAMSStudyResult
    stage2: IAMSStudyResult

    @property
    def best_trial(self):
        """Final best — always the Stage-2 winner (measured on the bigger corpus)."""
        return self.stage2.best_trial

    @property
    def best_config(self) -> dict:
        return dict(self.stage2.best_config)


def _resolve_budget(cfg: TwoStageConfig, space: ParameterSpace) -> tuple[int, int]:
    """Return (stage1_budget, stage2_budget), auto-sizing any that were left None."""
    s1 = cfg.stage1_budget
    if s1 is None:
        s1 = BudgetController.suggest(space, phase="coarse")
    s2 = cfg.stage2_budget
    if s2 is None:
        s2 = BudgetController.suggest(space, phase="fine", warm_start=True)
    return int(s1), int(s2)


def _resolve_epsilon(cfg: TwoStageConfig) -> tuple[float, float]:
    """Return (stage1_epsilon, stage2_epsilon), auto-calibrating from durations."""
    s1 = cfg.stage1_epsilon
    if s1 is None:
        s1 = suggest_epsilon(cfg.stage1_duration_s)
    s2 = cfg.stage2_epsilon
    if s2 is None:
        s2 = suggest_epsilon(cfg.stage2_duration_s)
    return float(s1), float(s2)


def run_two_stage(
    *,
    space: ParameterSpace,
    objective: Objective,
    backend: BaseBackend,
    mode_hint: dict | None,
    dataset_loader: DatasetLoader,
    executor_factory: ExecutorFactory,
    cfg: TwoStageConfig = TwoStageConfig(),
) -> TwoStageResult:
    """
    Run a coarse Stage 1 followed by a warm-started fine Stage 2.

    Parameters:
        space: Parameter search space shared by both stages. The optimizer
            applies backend filtering once per stage.
        objective: Scoring function; same instance is passed to both stages.
        backend: Loaded backend. The caller owns load()/unload() — this
            function does not touch the backend lifecycle.
        mode_hint: Forwarded to IAMSOptimizer (e.g. ``{"batch_size": 5}``).
        dataset_loader: ``(duration_s) -> PreparedDataset``. Called once per
            stage with the stage's ``stage*_duration_s`` value. The caller is
            responsible for any caching / reuse policy.
        executor_factory: ``(prepared_dataset, duration_s) -> TrialExecutor``.
            Built per stage so each stage can carry its own BenchmarkEngine
            state and, e.g., per-dataset cache key context.
        cfg: Duration / budget / epsilon knobs. See ``TwoStageConfig``.

    Returns:
        ``TwoStageResult`` with both per-stage ``IAMSStudyResult`` objects.
    """
    s1_budget, s2_budget = _resolve_budget(cfg, space)
    s1_eps, s2_eps = _resolve_epsilon(cfg)

    # --- Stage 1 --------------------------------------------------------------
    s1_dataset = dataset_loader(cfg.stage1_duration_s)
    s1_executor = executor_factory(s1_dataset, cfg.stage1_duration_s)
    s1_budget_ctrl = BudgetController(
        hard_cap=s1_budget,
        convergence_eps=s1_eps,
    )
    s1_optimizer = IAMSOptimizer(
        executor=s1_executor,
        space=space,
        objective=objective,
        budget=s1_budget_ctrl,
        eps_min=s1_eps,
        mode=cfg.mode,  # type: ignore[arg-type]
        top_k_pairs=cfg.top_k_pairs,
        multistart_candidates=cfg.multistart_candidates,
        validation_runs=cfg.validation_runs,
        enable_deep_ablation=cfg.enable_deep_ablation,
        backend=backend,
        mode_hint=mode_hint,
        use_multifidelity=cfg.use_multifidelity,
        multifidelity_rungs=cfg.multifidelity_rungs,
        multifidelity_prune_threshold=cfg.multifidelity_prune_threshold,
    )
    s1_result = s1_optimizer.run()

    # --- Stage 2 — warm-started from Stage 1's screening ----------------------
    s2_dataset = dataset_loader(cfg.stage2_duration_s)
    s2_executor = executor_factory(s2_dataset, cfg.stage2_duration_s)
    s2_budget_ctrl = BudgetController(
        hard_cap=s2_budget,
        convergence_eps=s2_eps,
    )
    s2_optimizer = IAMSOptimizer(
        executor=s2_executor,
        space=space,
        objective=objective,
        budget=s2_budget_ctrl,
        eps_min=s2_eps,
        mode=cfg.mode,  # type: ignore[arg-type]
        top_k_pairs=cfg.top_k_pairs,
        multistart_candidates=cfg.multistart_candidates,
        validation_runs=cfg.validation_runs,
        enable_deep_ablation=cfg.enable_deep_ablation,
        prior_screening=s1_result.screening,
        backend=backend,
        mode_hint=mode_hint,
        use_multifidelity=cfg.use_multifidelity,
        multifidelity_rungs=cfg.multifidelity_rungs,
        multifidelity_prune_threshold=cfg.multifidelity_prune_threshold,
    )
    s2_result = s2_optimizer.run()

    return TwoStageResult(stage1=s1_result, stage2=s2_result)
