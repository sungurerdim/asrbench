"""Single-study background task + cross-runner helpers.

The other two runner entry points — ``run_two_stage`` and
``run_global_config`` — live in ``_optimization_multistage`` because the
three shapes together exceed the 500-line-per-file guideline. The
helpers in this module (``build_objective``, ``_resolve_backend``,
``_mode_hint_from_space``) are imported from the multistage module too.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from asrbench.db import get_conn

from ._optimization_models import (
    ObjectiveConfig,
    OptimizeStartRequest,
)
from ._optimization_persistence import (
    finalize_stage_failure,
    finalize_stage_success,
    warm_start_result,
)

if TYPE_CHECKING:
    from asrbench.engine.search.objective import Objective

logger = logging.getLogger(__name__)

__all__ = [
    "build_objective",
    "run_single_study",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def build_objective(cfg: ObjectiveConfig) -> Objective:
    """Convert ObjectiveConfig into a concrete Objective instance."""
    from asrbench.engine.search.objective import (
        SingleMetricObjective,
        WeightedObjective,
    )

    if cfg.type == "single":
        if not cfg.metric:
            raise ValueError("single objective requires 'metric'")
        return SingleMetricObjective(
            metric=cfg.metric,
            direction=cfg.direction or None,  # type: ignore[arg-type]
        )
    if cfg.type == "weighted":
        if not cfg.weights:
            raise ValueError("weighted objective requires non-empty 'weights'")
        return WeightedObjective(weights=cfg.weights)
    raise ValueError(f"Unknown objective type {cfg.type!r}")


def _resolve_backend(backend_name: str) -> Any:
    """Load and instantiate a backend class by its registry name."""
    from asrbench.backends import load_backends

    backends = load_backends()
    if backend_name not in backends:
        raise RuntimeError(
            f"Backend '{backend_name}' is not installed. "
            f"Install it: pip install 'asrbench[{backend_name}]'"
        )
    return backends[backend_name]()


def _mode_hint_from_space(space: Any) -> dict[str, Any]:
    """Extract the batch-size hint the optimizer passes to the backend."""
    hint: dict[str, Any] = {}
    try:
        hint["batch_size"] = int(space.defaults().get("batch_size", 0))
    except (TypeError, ValueError):
        hint["batch_size"] = 0
    return hint


# ---------------------------------------------------------------------------
# Single study
# ---------------------------------------------------------------------------


async def run_single_study(
    *,
    study_id: str,
    req: OptimizeStartRequest,
    backend_name: str,
    model_local_path: str,
) -> None:
    """Background task for ``POST /optimize/start``.

    Exceptions are caught, logged with the full traceback, and surfaced
    on the study row via ``finalize_stage_failure``. The /start handler
    has already returned, so there is no HTTP path to propagate to.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        result = await _execute_single(
            cur,
            study_id=study_id,
            req=req,
            backend_name=backend_name,
            model_local_path=model_local_path,
        )
        _persist_single_success(cur, study_id=study_id, result=result)
        logger.info(
            "Study %s completed: best_score=%.4f, %d trials",
            study_id,
            result.best_trial.score,
            result.total_trials,
        )
    except Exception as exc:
        logger.error("Study %s failed: %s", study_id, exc, exc_info=True)
        finalize_stage_failure(cur, study_ids=(study_id,), exc=exc)


async def _execute_single(
    cur: Any,
    *,
    study_id: str,
    req: OptimizeStartRequest,
    backend_name: str,
    model_local_path: str,
) -> Any:
    """Prepare dataset, load backend, run IAMS optimiser."""
    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager
    from asrbench.engine.benchmark import BenchmarkEngine
    from asrbench.engine.optimizer import IAMSOptimizer
    from asrbench.engine.search.benchmark_executor import BenchmarkTrialExecutor
    from asrbench.engine.search.budget import BudgetController
    from asrbench.engine.search.space import ParameterSpace

    conn = get_conn()
    logger.info("Study %s: preparing dataset %s", study_id, req.dataset_id)
    config = get_config()
    prepared = DatasetManager(config, conn).prepare(req.dataset_id)
    logger.info(
        "Study %s: dataset ready (%d segments, %.1fs)",
        study_id,
        len(prepared.segments),
        prepared.duration_s,
    )

    space = ParameterSpace.from_dict(req.space)
    backend_instance = _resolve_backend(backend_name)

    logger.debug("Study %s: loading model %s", study_id, model_local_path)
    backend_instance.load(model_local_path, space.defaults())
    logger.debug("Study %s: model loaded, starting optimizer", study_id)

    try:
        engine = BenchmarkEngine(
            conn,
            cache_dir=config.storage.cache_dir,
            segment_timeout_s=config.limits.segment_timeout_s,
        )
        objective = build_objective(req.objective)
        budget = BudgetController(
            hard_cap=req.budget.hard_cap,
            convergence_eps=req.budget.convergence_eps,
            convergence_window=req.budget.convergence_window,
        )

        executor = BenchmarkTrialExecutor(
            engine=engine,
            conn=conn,
            study_id=study_id,
            model_id=req.model_id,
            backend_name=backend_name,
            model_local_path=model_local_path,
            dataset=prepared,
            objective=objective,
            backend=backend_instance,
            lang=req.lang,
        )

        prior_screening = (
            warm_start_result(cur, executor, req.prior_study_id, space)
            if req.prior_study_id
            else None
        )

        optimizer = IAMSOptimizer(
            executor=executor,
            space=space,
            objective=objective,
            budget=budget,
            eps_min=req.eps_min,
            mode=req.mode,  # type: ignore[arg-type]
            top_k_pairs=req.top_k_pairs,
            multistart_candidates=req.multistart_candidates,
            validation_runs=req.validation_runs,
            enable_deep_ablation=req.enable_deep_ablation,
            prior_screening=prior_screening,
            backend=backend_instance,
            mode_hint=_mode_hint_from_space(space),
        )
        return optimizer.run()
    finally:
        backend_instance.unload()


def _persist_single_success(cur: Any, *, study_id: str, result: Any) -> None:
    """Reuse the stage-success writer for the single-study shape.

    IAMSOptimizer.run()'s return type mirrors a TwoStageResult stage
    closely enough that ``finalize_stage_success`` accepts it directly;
    keeping the reuse avoids two near-identical UPDATE statements.
    """
    finalize_stage_success(cur, study_id=study_id, stage_result=result)


# ---------------------------------------------------------------------------
# Two-stage
