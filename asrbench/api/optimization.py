"""REST API for the IAMS parameter optimizer.

Endpoints:

    POST /optimize/start          — single-dataset IAMS study
    POST /optimize/two-stage      — coarse→fine two-stage run
    POST /optimize/global-config  — two-stage run aggregated over N datasets
    GET  /optimize/               — list studies (most recent 100)
    GET  /optimize/{id}           — study metadata + best config
    GET  /optimize/{id}/trials    — paginated trial log
    POST /optimize/{id}/cancel    — force-mark a stuck study as cancelled

The module is a thin HTTP facade. Pydantic models, DB persistence, and
the background IAMS runners live in private siblings so each file stays
under the project's 500-line-per-file guideline:

    _optimization_models        — Pydantic request/response classes
    _optimization_persistence   — DuckDB read/write helpers
    _optimization_runners       — single-study background task + shared helpers
    _optimization_multistage    — two-stage / global-config background tasks
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from asrbench.db import get_conn

from ._optimization_models import (
    GlobalConfigStartRequest,
    GlobalConfigStartResponse,
    OptimizeStartRequest,
    OptimizeStartResponse,
    StudyResponse,
    TrialResponse,
    TwoStageStartRequest,
    TwoStageStartResponse,
)
from ._optimization_multistage import run_global_config, run_two_stage
from ._optimization_persistence import (
    insert_global_config_pair,
    insert_single_study,
    insert_two_stage_pair,
)
from ._optimization_persistence import (
    list_studies as _list_studies_db,
)
from ._optimization_persistence import (
    list_trials as _list_trials_db,
)
from ._optimization_persistence import (
    load_study as _load_study_db,
)
from ._optimization_runners import build_objective, run_single_study

# Re-exports kept so tests / call sites can keep ``from asrbench.api.optimization import X``.
__all__ = [
    "GlobalConfigStartRequest",
    "GlobalConfigStartResponse",
    "OptimizeStartRequest",
    "OptimizeStartResponse",
    "StudyResponse",
    "TrialResponse",
    "TwoStageStartRequest",
    "TwoStageStartResponse",
    "router",
]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimize", tags=["optimize"])


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _fetch_model_or_404(cur: Any, model_id: str) -> tuple[str, str]:
    """Return (backend_name, model_local_path) or raise 404."""
    row = cur.execute(
        "SELECT model_id, backend, default_params, local_path FROM models WHERE model_id = ?",
        [model_id],
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Register it first via POST /models.",
        )
    return str(row[1]), str(row[3])


def _require_dataset(cur: Any, dataset_id: str) -> None:
    row = cur.execute(
        "SELECT dataset_id FROM datasets WHERE dataset_id = ?", [dataset_id]
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Add it via POST /datasets/fetch.",
        )


def _require_datasets(cur: Any, dataset_ids: list[str]) -> None:
    missing: list[str] = []
    for dsid in dataset_ids:
        row = cur.execute("SELECT dataset_id FROM datasets WHERE dataset_id = ?", [dsid]).fetchone()
        if not row:
            missing.append(dsid)
    if missing:
        raise HTTPException(status_code=404, detail=f"Datasets not found: {missing}")


def _require_no_running_study(cur: Any) -> None:
    row = cur.execute(
        "SELECT study_id FROM optimization_studies WHERE status = 'running' LIMIT 1"
    ).fetchone()
    if row:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An optimization study is already in progress ({row[0]}). "
                "Only one concurrent study is supported."
            ),
        )


def _validate_space_and_objective(space: dict, objective: Any) -> None:
    from asrbench.engine.search.space import ParameterSpace

    try:
        ParameterSpace.from_dict(space)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid parameter space: {exc}") from exc
    try:
        build_objective(objective)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid objective: {exc}") from exc


def _require_prior_study_completed(cur: Any, prior_study_id: str | None) -> None:
    if not prior_study_id:
        return
    row = cur.execute(
        "SELECT status FROM optimization_studies WHERE study_id = ?",
        [prior_study_id],
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Prior study '{prior_study_id}' not found.")
    if str(row[0]) != "completed":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Prior study '{prior_study_id}' has status '{row[0]}' — "
                "only completed studies can be used for warm start."
            ),
        )


# ---------------------------------------------------------------------------
# Endpoints — start*
# ---------------------------------------------------------------------------


@router.post("/start", response_model=OptimizeStartResponse, status_code=202)
async def start_study(
    req: OptimizeStartRequest, background_tasks: BackgroundTasks
) -> OptimizeStartResponse:
    """Create a single-dataset optimization study and kick off IAMS."""
    conn = get_conn()
    cur = conn.cursor()

    backend_name, model_local_path = _fetch_model_or_404(cur, req.model_id)
    _require_dataset(cur, req.dataset_id)
    _require_no_running_study(cur)
    _validate_space_and_objective(req.space, req.objective)
    _require_prior_study_completed(cur, req.prior_study_id)

    try:
        study_id = insert_single_study(cur, req=req)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(
        run_single_study,
        study_id=study_id,
        req=req,
        backend_name=backend_name,
        model_local_path=model_local_path,
    )

    return OptimizeStartResponse(
        study_id=study_id,
        status="running",
        mode=req.mode,
        hard_cap=req.budget.hard_cap,
    )


@router.post("/two-stage", response_model=TwoStageStartResponse, status_code=202)
async def start_two_stage(
    req: TwoStageStartRequest, background_tasks: BackgroundTasks
) -> TwoStageStartResponse:
    """Start a coarse→fine two-stage optimization run.

    Creates two ``optimization_studies`` rows (one per stage) and runs
    both in sequence via the library's ``run_two_stage`` orchestrator.
    Budget/epsilon default to ``None`` — the library sizes them from the
    space and the stage duration when left unset.
    """
    conn = get_conn()
    cur = conn.cursor()

    backend_name, model_local_path = _fetch_model_or_404(cur, req.model_id)
    _require_dataset(cur, req.dataset_id)
    _require_no_running_study(cur)
    _validate_space_and_objective(req.space, req.objective)

    try:
        stage1, stage2 = insert_two_stage_pair(cur, req=req)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(
        run_two_stage,
        stage1_study_id=stage1,
        stage2_study_id=stage2,
        req=req,
        backend_name=backend_name,
        model_local_path=model_local_path,
    )

    return TwoStageStartResponse(
        stage1_study_id=stage1,
        stage2_study_id=stage2,
        status="running",
        mode=req.mode,
    )


@router.post("/global-config", response_model=GlobalConfigStartResponse, status_code=202)
async def start_global_config(
    req: GlobalConfigStartRequest, background_tasks: BackgroundTasks
) -> GlobalConfigStartResponse:
    """Start a two-stage IAMS run aggregated across N datasets.

    Every trial evaluates the candidate config on ALL listed datasets,
    weights the per-dataset scores, and returns one aggregate. IAMS's
    7-layer algorithm then produces a single global config that
    minimises the weighted mean across the fleet — use this when
    deploying to a product with a single shared preset.
    """
    conn = get_conn()
    cur = conn.cursor()

    backend_name, model_local_path = _fetch_model_or_404(cur, req.model_id)
    _require_datasets(cur, [ds.dataset_id for ds in req.datasets])
    _require_no_running_study(cur)
    _validate_space_and_objective(req.space, req.objective)

    try:
        stage1, stage2 = insert_global_config_pair(cur, req=req)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(
        run_global_config,
        stage1_study_id=stage1,
        stage2_study_id=stage2,
        req=req,
        backend_name=backend_name,
        model_local_path=model_local_path,
    )

    return GlobalConfigStartResponse(
        stage1_study_id=stage1,
        stage2_study_id=stage2,
        status="running",
        mode=req.mode,
        dataset_count=len(req.datasets),
    )


# ---------------------------------------------------------------------------
# Endpoints — read / cancel
# ---------------------------------------------------------------------------


@router.get("/", response_model=list[StudyResponse])
async def list_studies(status: str | None = None) -> list[StudyResponse]:
    """List optimization studies, optionally filtered by status."""
    return _list_studies_db(get_conn().cursor(), status=status)


@router.post("/{study_id}/cancel", response_model=StudyResponse)
async def cancel_study(study_id: str) -> StudyResponse:
    """Force-cancel a stuck running or failed study.

    Marks status as 'cancelled' so a new study can start. Safe to call
    on a study whose background task has already died.
    """
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT status FROM optimization_studies WHERE study_id = ?", [study_id]
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found.")
    if str(row[0]) not in ("running", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Study '{study_id}' has status '{row[0]}' and cannot be cancelled.",
        )
    cur.execute(
        "UPDATE optimization_studies SET status = 'cancelled', finished_at = now() "
        "WHERE study_id = ?",
        [study_id],
    )
    return await get_study(study_id)


@router.get("/{study_id}", response_model=StudyResponse)
async def get_study(study_id: str) -> StudyResponse:
    """Return a single study's metadata and final result (if completed)."""
    study = _load_study_db(get_conn().cursor(), study_id)
    if study is None:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found.")
    return study


@router.get("/{study_id}/trials", response_model=list[TrialResponse])
async def list_trials(
    study_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, le=1000),
    phase: str | None = None,
) -> list[TrialResponse]:
    """Paginated access to the full trial log for audit / UI replay."""
    cur = get_conn().cursor()
    if not cur.execute(
        "SELECT 1 FROM optimization_studies WHERE study_id = ?", [study_id]
    ).fetchone():
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found.")
    return _list_trials_db(cur, study_id=study_id, page=page, page_size=page_size, phase=phase)
