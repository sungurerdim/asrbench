"""Two-stage and global-config background tasks.

Split out of ``_optimization_runners`` so every file in the /optimize
facade stays under the project's 500-line-per-file guideline. The
single-study path lives in ``_optimization_runners`` — shape-wise it
shares little with the two-stage orchestrator, so keeping it separate
also reads clearly.
"""

from __future__ import annotations

import logging
from typing import Any

from asrbench.db import get_conn

from ._optimization_models import (
    GlobalConfigStartRequest,
    TwoStageStartRequest,
)
from ._optimization_persistence import (
    finalize_stage_failure,
    finalize_stage_success,
)
from ._optimization_runners import (
    _mode_hint_from_space,
    _resolve_backend,
    build_objective,
)

logger = logging.getLogger(__name__)

__all__ = ["run_two_stage", "run_global_config"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _two_stage_cfg(req: TwoStageStartRequest | GlobalConfigStartRequest) -> Any:
    """Build a TwoStageConfig from the request's stage-level fields."""
    from asrbench.engine.two_stage import TwoStageConfig

    return TwoStageConfig(
        stage1_duration_s=req.stage1_duration_s,
        stage2_duration_s=req.stage2_duration_s,
        stage1_budget=req.stage1_budget,
        stage2_budget=req.stage2_budget,
        stage1_epsilon=req.stage1_epsilon,
        stage2_epsilon=req.stage2_epsilon,
        mode=req.mode,
        top_k_pairs=req.top_k_pairs,
        multistart_candidates=req.multistart_candidates,
        validation_runs=req.validation_runs,
        enable_deep_ablation=req.enable_deep_ablation,
        use_multifidelity=req.use_multifidelity,
    )


def _prepare_shared(
    *,
    req: TwoStageStartRequest | GlobalConfigStartRequest,
    backend_name: str,
    model_local_path: str,
    log_prefix: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Boot everything the library orchestrator needs.

    Returns ``(conn, engine, backend_instance, space, objective, dm)``.
    The caller owns ``backend_instance.unload()`` lifecycle and closes
    over ``conn`` / ``engine`` / ``dm`` from the executor factory.
    """
    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager
    from asrbench.engine.benchmark import BenchmarkEngine
    from asrbench.engine.search.space import ParameterSpace

    conn = get_conn()
    config = get_config()
    dm = DatasetManager(config, conn)

    backend_instance = _resolve_backend(backend_name)
    space = ParameterSpace.from_dict(req.space)
    objective = build_objective(req.objective)

    logger.info("%s: loading model %s", log_prefix, model_local_path)
    backend_instance.load(model_local_path, space.defaults())

    engine = BenchmarkEngine(conn, cache_dir=config.storage.cache_dir)
    return conn, engine, backend_instance, space, objective, dm


# ---------------------------------------------------------------------------
# Two-stage (one dataset)
# ---------------------------------------------------------------------------


async def run_two_stage(
    *,
    stage1_study_id: str,
    stage2_study_id: str,
    req: TwoStageStartRequest,
    backend_name: str,
    model_local_path: str,
) -> None:
    """Background task for ``POST /optimize/two-stage``."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        result = await _execute_two_stage(
            stage1_study_id=stage1_study_id,
            stage2_study_id=stage2_study_id,
            req=req,
            backend_name=backend_name,
            model_local_path=model_local_path,
        )
        finalize_stage_success(cur, study_id=stage1_study_id, stage_result=result.stage1)
        finalize_stage_success(cur, study_id=stage2_study_id, stage_result=result.stage2)
        logger.info(
            "Two-stage complete: stage1=%s (%d trials), stage2=%s (%d trials)",
            stage1_study_id,
            result.stage1.total_trials,
            stage2_study_id,
            result.stage2.total_trials,
        )
    except Exception as exc:
        logger.error(
            "Two-stage %s/%s failed: %s",
            stage1_study_id,
            stage2_study_id,
            exc,
            exc_info=True,
        )
        finalize_stage_failure(cur, study_ids=(stage1_study_id, stage2_study_id), exc=exc)


async def _execute_two_stage(
    *,
    stage1_study_id: str,
    stage2_study_id: str,
    req: TwoStageStartRequest,
    backend_name: str,
    model_local_path: str,
) -> Any:
    """Run the library's ``run_two_stage`` orchestrator."""
    from asrbench.engine.search.benchmark_executor import BenchmarkTrialExecutor
    from asrbench.engine.two_stage import run_two_stage as run_two_stage_lib

    conn, engine, backend_instance, space, objective, dm = _prepare_shared(
        req=req,
        backend_name=backend_name,
        model_local_path=model_local_path,
        log_prefix=f"Two-stage {stage1_study_id}/{stage2_study_id}",
    )

    try:
        stage_ids = [stage1_study_id, stage2_study_id]
        factory_calls = {"n": 0}

        def dataset_loader(duration_s: int):  # type: ignore[no-untyped-def]
            current = stage_ids[min(factory_calls["n"], 1)]
            logger.info(
                "Two-stage %s: preparing dataset %s (cap=%ds)",
                current,
                req.dataset_id,
                duration_s,
            )
            return dm.prepare(req.dataset_id, max_duration_s=float(duration_s))

        def executor_factory(prepared, duration_s):  # type: ignore[no-untyped-def]
            _ = duration_s
            current = stage_ids[min(factory_calls["n"], 1)]
            factory_calls["n"] += 1
            return BenchmarkTrialExecutor(
                engine=engine,
                conn=conn,
                study_id=current,
                model_id=req.model_id,
                backend_name=backend_name,
                model_local_path=model_local_path,
                dataset=prepared,
                objective=objective,
                backend=backend_instance,
                lang=req.lang,
            )

        return run_two_stage_lib(
            space=space,
            objective=objective,
            backend=backend_instance,
            mode_hint=_mode_hint_from_space(space),
            dataset_loader=dataset_loader,
            executor_factory=executor_factory,
            cfg=_two_stage_cfg(req),
        )
    finally:
        backend_instance.unload()


# ---------------------------------------------------------------------------
# Global-config (N datasets)
# ---------------------------------------------------------------------------


async def run_global_config(
    *,
    stage1_study_id: str,
    stage2_study_id: str,
    req: GlobalConfigStartRequest,
    backend_name: str,
    model_local_path: str,
) -> None:
    """Background task for ``POST /optimize/global-config``."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        result = await _execute_global(
            stage1_study_id=stage1_study_id,
            stage2_study_id=stage2_study_id,
            req=req,
            backend_name=backend_name,
            model_local_path=model_local_path,
        )
        _persist_global_success(
            cur,
            stage1_study_id=stage1_study_id,
            stage2_study_id=stage2_study_id,
            req=req,
            result=result,
        )
        logger.info(
            "Global-config complete: stage1=%s, stage2=%s, N_datasets=%d",
            stage1_study_id,
            stage2_study_id,
            len(req.datasets),
        )
    except Exception as exc:
        logger.error(
            "Global-config %s/%s failed: %s",
            stage1_study_id,
            stage2_study_id,
            exc,
            exc_info=True,
        )
        finalize_stage_failure(cur, study_ids=(stage1_study_id, stage2_study_id), exc=exc)


async def _execute_global(
    *,
    stage1_study_id: str,
    stage2_study_id: str,
    req: GlobalConfigStartRequest,
    backend_name: str,
    model_local_path: str,
) -> Any:
    """Run the library's two-stage orchestrator over an N-dataset executor."""
    from asrbench.engine.search.benchmark_executor import BenchmarkTrialExecutor
    from asrbench.engine.search.multidataset import MultiDatasetTrialExecutor
    from asrbench.engine.two_stage import run_two_stage as run_two_stage_lib

    conn, engine, backend_instance, space, objective, dm = _prepare_shared(
        req=req,
        backend_name=backend_name,
        model_local_path=model_local_path,
        log_prefix=(
            f"Global-config {stage1_study_id}/{stage2_study_id} (N={len(req.datasets)} datasets)"
        ),
    )

    try:
        stage_ids = [stage1_study_id, stage2_study_id]
        factory_calls = {"n": 0}
        weights = [ds.weight for ds in req.datasets]
        labels = [f"{ds.dataset_id[:8]}_{ds.lang}" for ds in req.datasets]

        def dataset_loader(duration_s: int):  # type: ignore[no-untyped-def]
            current = stage_ids[min(factory_calls["n"], 1)]
            logger.info(
                "Global-config %s: preparing %d datasets at cap=%ds",
                current,
                len(req.datasets),
                duration_s,
            )
            return [
                dm.prepare(ds.dataset_id, max_duration_s=float(duration_s)) for ds in req.datasets
            ]

        def executor_factory(prepared_bundle, duration_s):  # type: ignore[no-untyped-def]
            _ = duration_s
            current = stage_ids[min(factory_calls["n"], 1)]
            factory_calls["n"] += 1
            inner = [
                BenchmarkTrialExecutor(
                    engine=engine,
                    conn=conn,
                    study_id=current,
                    model_id=req.model_id,
                    backend_name=backend_name,
                    model_local_path=model_local_path,
                    dataset=prepared,
                    objective=objective,
                    backend=backend_instance,
                    lang=ds_spec.lang,
                )
                for prepared, ds_spec in zip(prepared_bundle, req.datasets)
            ]
            return MultiDatasetTrialExecutor(executors=inner, weights=weights, labels=labels)

        return run_two_stage_lib(
            space=space,
            objective=objective,
            backend=backend_instance,
            mode_hint=_mode_hint_from_space(space),
            dataset_loader=dataset_loader,
            executor_factory=executor_factory,
            cfg=_two_stage_cfg(req),
        )
    finally:
        backend_instance.unload()


def _persist_global_success(
    cur: Any,
    *,
    stage1_study_id: str,
    stage2_study_id: str,
    req: GlobalConfigStartRequest,
    result: Any,
) -> None:
    """Persist both stages with a "fleet summary" prepended to reasoning."""
    fleet_summary = [
        {"dataset_id": ds.dataset_id, "lang": ds.lang, "weight": ds.weight} for ds in req.datasets
    ]
    header = f"[global-config over {len(req.datasets)} datasets: {fleet_summary}]"
    for stage_id, stage_result in (
        (stage1_study_id, result.stage1),
        (stage2_study_id, result.stage2),
    ):
        reasoning = [header, *stage_result.reasoning]
        finalize_stage_success(
            cur,
            study_id=stage_id,
            stage_result=stage_result,
            reasoning=reasoning,
        )
