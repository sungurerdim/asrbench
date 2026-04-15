"""
REST API for the IAMS parameter optimizer.

Endpoints:

    POST /optimize/start            — create a study and kick off the 7-layer
                                      algorithm in a background task
    GET  /optimize/{study_id}       — status + best config + summary metrics
    GET  /optimize/{study_id}/trials — paginated trial log (for the UI / audit)

A study is a long-running process. The /start endpoint returns immediately
with a study_id; the actual optimization runs in a FastAPI BackgroundTasks
coroutine. The /{study_id} endpoint can be polled to track progress.

This module deliberately does NOT embed IAMS algorithm logic. It is a thin
facade that:

    1. Validates inputs (space YAML dict, objective config, budget)
    2. Creates an optimization_studies row
    3. Constructs a BenchmarkTrialExecutor and an IAMSOptimizer
    4. Runs the optimizer and persists per-trial results as it goes
    5. Finalizes the study row with best config and confidence label

Concurrency:
    Only one optimization study may run at a time (enforced via a check on
    optimization_studies.status). This matches the existing runs constraint.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from asrbench.db import get_conn

if TYPE_CHECKING:
    from asrbench.engine.search.objective import Objective

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimize", tags=["optimize"])


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class ObjectiveConfig(BaseModel):
    """
    Objective function payload.

    - For a single metric: {"type": "single", "metric": "wer",
                            "direction": "minimize" | "maximize" | null}
    - For weighted: {"type": "weighted", "weights": {"wer": 1.0, "rtfx": -0.1}}
    """

    type: str = Field(..., pattern="^(single|weighted)$")
    metric: str | None = None
    direction: str | None = None
    weights: dict[str, float] | None = None


class BudgetConfig(BaseModel):
    hard_cap: int = Field(..., gt=0)
    convergence_eps: float = Field(0.005, ge=0)
    convergence_window: int = Field(3, ge=0)


class OptimizeStartRequest(BaseModel):
    model_id: str
    dataset_id: str
    lang: str = "en"
    space: dict[str, Any]
    objective: ObjectiveConfig
    mode: str = Field("maximum", pattern="^(fast|balanced|maximum)$")
    budget: BudgetConfig
    eps_min: float = 0.005
    top_k_pairs: int = Field(4, ge=2)
    multistart_candidates: int = Field(3, ge=1)
    validation_runs: int = Field(3, ge=2)
    enable_deep_ablation: bool = False
    prior_study_id: str | None = Field(
        None, description="Resume from a completed study's cache + screening"
    )


class OptimizeStartResponse(BaseModel):
    study_id: str
    status: str
    mode: str
    hard_cap: int


class TwoStageStartRequest(BaseModel):
    """
    Kick off a two-stage coarse→fine IAMS run.

    The request carries the same fields as ``OptimizeStartRequest`` PLUS
    two durations. Budget/epsilon are optional — if omitted the library's
    auto-sizing helpers size them from the space and stage durations.
    """

    model_id: str
    dataset_id: str
    lang: str = "en"
    space: dict[str, Any]
    objective: ObjectiveConfig
    mode: str = Field("maximum", pattern="^(fast|balanced|maximum)$")
    stage1_duration_s: int = Field(900, gt=0)
    stage2_duration_s: int = Field(2400, gt=0)
    stage1_budget: int | None = Field(None, ge=1)
    stage2_budget: int | None = Field(None, ge=1)
    stage1_epsilon: float | None = Field(None, ge=0)
    stage2_epsilon: float | None = Field(None, ge=0)
    top_k_pairs: int = Field(4, ge=2)
    multistart_candidates: int = Field(3, ge=1)
    validation_runs: int = Field(3, ge=2)
    enable_deep_ablation: bool = False
    use_multifidelity: bool = Field(
        False,
        description=(
            "Enable Hyperband-style rung pruning for Layer 2+ trials. "
            "Cheap configs get evaluated at 25%/50%/100% of the corpus; "
            "clearly-worse partial scores short-circuit the trial. "
            "Layer 1 screening and Layer 7 validation stay at full fidelity."
        ),
    )


class TwoStageStartResponse(BaseModel):
    stage1_study_id: str
    stage2_study_id: str
    status: str
    mode: str


class GlobalDatasetSpec(BaseModel):
    """One dataset slot in a global-config run."""

    dataset_id: str
    lang: str = "en"
    weight: float = Field(1.0, gt=0)


class GlobalConfigStartRequest(BaseModel):
    """
    Kick off a two-stage IAMS run over N datasets simultaneously.

    All datasets are evaluated by every IAMS trial; their scores are combined
    via ``MultiDatasetTrialExecutor`` (variance-weighted CI, weighted mean
    score) so the optimizer produces ONE config that minimizes the aggregate
    objective across the whole fleet. Use this when deploying to a product
    with a single shared preset.
    """

    model_id: str
    datasets: list[GlobalDatasetSpec] = Field(..., min_length=1)
    space: dict[str, Any]
    objective: ObjectiveConfig
    mode: str = Field("maximum", pattern="^(fast|balanced|maximum)$")
    stage1_duration_s: int = Field(900, gt=0)
    stage2_duration_s: int = Field(2400, gt=0)
    stage1_budget: int | None = Field(None, ge=1)
    stage2_budget: int | None = Field(None, ge=1)
    stage1_epsilon: float | None = Field(None, ge=0)
    stage2_epsilon: float | None = Field(None, ge=0)
    top_k_pairs: int = Field(4, ge=2)
    multistart_candidates: int = Field(3, ge=1)
    validation_runs: int = Field(3, ge=2)
    enable_deep_ablation: bool = False
    use_multifidelity: bool = False


class GlobalConfigStartResponse(BaseModel):
    stage1_study_id: str
    stage2_study_id: str
    status: str
    mode: str
    dataset_count: int


class TrialResponse(BaseModel):
    trial_id: str
    run_id: str | None
    phase: str
    config: dict
    score: float | None
    score_ci_lower: float | None
    score_ci_upper: float | None
    reasoning: str | None
    created_at: str


class StudyResponse(BaseModel):
    study_id: str
    model_id: str
    dataset_id: str
    lang: str
    mode: str
    status: str
    eps_min: float
    best_run_id: str | None
    best_score: float | None
    best_config: dict | None
    confidence: str | None
    total_trials: int | None
    reasoning: list[str] | None
    started_at: str | None
    finished_at: str | None
    created_at: str
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start", response_model=OptimizeStartResponse, status_code=202)
async def start_study(
    req: OptimizeStartRequest, background_tasks: BackgroundTasks
) -> OptimizeStartResponse:
    """
    Create an optimization study and run it in a background task.

    Validates the model, dataset, and parameter space up-front so the caller
    gets immediate feedback on bad input. Actual optimization (L1-L7) runs
    asynchronously — poll GET /optimize/{study_id} for progress.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Validate model
    model_row = cur.execute(
        "SELECT model_id, backend, default_params, local_path FROM models WHERE model_id = ?",
        [req.model_id],
    ).fetchone()
    if not model_row:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model_id}' not found. Register it first via POST /models.",
        )
    backend_name = str(model_row[1])
    model_local_path = str(model_row[3])

    # Validate dataset
    dataset_row = cur.execute(
        "SELECT dataset_id, lang FROM datasets WHERE dataset_id = ?",
        [req.dataset_id],
    ).fetchone()
    if not dataset_row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{req.dataset_id}' not found. Add it via POST /datasets/fetch.",
        )

    # Enforce single concurrent study
    running = cur.execute(
        "SELECT study_id FROM optimization_studies WHERE status = 'running' LIMIT 1"
    ).fetchone()
    if running:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An optimization study is already in progress ({running[0]}). "
                "Only one concurrent study is supported."
            ),
        )

    # Parse and validate the parameter space up-front (fail fast on bad YAML)
    try:
        from asrbench.engine.search.space import ParameterSpace

        ParameterSpace.from_dict(req.space)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid parameter space: {exc}",
        ) from exc

    # Validate objective config
    try:
        _build_objective(req.objective)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid objective: {exc}") from exc

    # Validate prior study if provided
    if req.prior_study_id:
        prior_row = cur.execute(
            "SELECT status FROM optimization_studies WHERE study_id = ?",
            [req.prior_study_id],
        ).fetchone()
        if not prior_row:
            raise HTTPException(
                status_code=404,
                detail=f"Prior study '{req.prior_study_id}' not found.",
            )
        if str(prior_row[0]) != "completed":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Prior study '{req.prior_study_id}' has status "
                    f"'{prior_row[0]}' — only completed studies can be used for warm start."
                ),
            )

    # Insert the study row
    cur.execute(
        """
        INSERT INTO optimization_studies
            (model_id, dataset_id, lang, space, objective, budget,
             mode, eps_min, prior_study_id, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', now())
        RETURNING study_id
        """,
        [
            req.model_id,
            req.dataset_id,
            req.lang,
            json.dumps(req.space),
            json.dumps(req.objective.model_dump()),
            json.dumps(req.budget.model_dump()),
            req.mode,
            req.eps_min,
            req.prior_study_id,
        ],
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to insert study row.")
    study_id = str(row[0])

    # Kick off the background task
    background_tasks.add_task(
        _run_study_background,
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


@router.post(
    "/two-stage",
    response_model=TwoStageStartResponse,
    status_code=202,
)
async def start_two_stage(
    req: TwoStageStartRequest, background_tasks: BackgroundTasks
) -> TwoStageStartResponse:
    """
    Start a coarse→fine two-stage optimization run.

    Creates two ``optimization_studies`` rows (one per stage) and runs both
    in sequence inside a single background task. The Stage-2 study is
    warm-started from Stage-1's screening result via the library's
    ``run_two_stage`` orchestrator, which is the same code path CLI and
    notebook callers use.

    Budget and epsilon default to ``None`` in the request — the library then
    sizes them from the space (BudgetController.suggest) and the stage
    duration (suggest_epsilon). Set them explicitly to override.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Validate model + dataset the same way /start does.
    model_row = cur.execute(
        "SELECT model_id, backend, default_params, local_path FROM models WHERE model_id = ?",
        [req.model_id],
    ).fetchone()
    if not model_row:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model_id}' not found. Register it first via POST /models.",
        )
    backend_name = str(model_row[1])
    model_local_path = str(model_row[3])

    dataset_row = cur.execute(
        "SELECT dataset_id, lang FROM datasets WHERE dataset_id = ?",
        [req.dataset_id],
    ).fetchone()
    if not dataset_row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{req.dataset_id}' not found. Add it via POST /datasets/fetch.",
        )

    running = cur.execute(
        "SELECT study_id FROM optimization_studies WHERE status = 'running' LIMIT 1"
    ).fetchone()
    if running:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An optimization study is already in progress ({running[0]}). "
                "Only one concurrent study is supported."
            ),
        )

    try:
        from asrbench.engine.search.space import ParameterSpace

        ParameterSpace.from_dict(req.space)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid parameter space: {exc}") from exc

    try:
        _build_objective(req.objective)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid objective: {exc}") from exc

    # Insert two placeholder study rows — one per stage. Budget gets the
    # placeholder 0 when the caller left it as None (the library will resolve
    # the real value before either stage starts).
    def _insert_stage(tag: str, budget_val: int | None, eps_val: float | None) -> str:
        cur.execute(
            """
            INSERT INTO optimization_studies
                (model_id, dataset_id, lang, space, objective, budget,
                 mode, eps_min, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', now())
            RETURNING study_id
            """,
            [
                req.model_id,
                req.dataset_id,
                req.lang,
                json.dumps(req.space),
                json.dumps(req.objective.model_dump()),
                json.dumps({"hard_cap": budget_val or 0, "convergence_eps": eps_val or 0.0}),
                req.mode,
                eps_val or 0.0,
                # NB: tag not persisted — kept in the reasoning field when the
                # background task writes results back.
            ],
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail=f"Failed to insert {tag} study row.")
        return str(row[0])

    stage1_study_id = _insert_stage("stage1", req.stage1_budget, req.stage1_epsilon)
    stage2_study_id = _insert_stage("stage2", req.stage2_budget, req.stage2_epsilon)

    background_tasks.add_task(
        _run_two_stage_background,
        stage1_study_id=stage1_study_id,
        stage2_study_id=stage2_study_id,
        req=req,
        backend_name=backend_name,
        model_local_path=model_local_path,
    )

    return TwoStageStartResponse(
        stage1_study_id=stage1_study_id,
        stage2_study_id=stage2_study_id,
        status="running",
        mode=req.mode,
    )


@router.post(
    "/global-config",
    response_model=GlobalConfigStartResponse,
    status_code=202,
)
async def start_global_config(
    req: GlobalConfigStartRequest, background_tasks: BackgroundTasks
) -> GlobalConfigStartResponse:
    """
    Start a two-stage IAMS run across N datasets with aggregate scoring.

    Unlike ``/optimize/two-stage`` (one dataset, one optimal config), this
    endpoint drives a ``MultiDatasetTrialExecutor``: every trial evaluates
    the candidate config on ALL listed datasets, weights the per-dataset
    scores, and returns one aggregate. IAMS's 7-layer algorithm then
    produces a single global config that minimizes the weighted mean.

    Use when a downstream product ships one shared preset to every user
    — e.g. a mobile ASR app tuned across clean/noisy/multilingual
    corpora simultaneously.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Validate model
    model_row = cur.execute(
        "SELECT model_id, backend, default_params, local_path FROM models WHERE model_id = ?",
        [req.model_id],
    ).fetchone()
    if not model_row:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model_id}' not found.",
        )
    backend_name = str(model_row[1])
    model_local_path = str(model_row[3])

    # Validate every dataset_id up-front — fail fast on bad input.
    missing: list[str] = []
    for ds in req.datasets:
        row = cur.execute(
            "SELECT dataset_id FROM datasets WHERE dataset_id = ?",
            [ds.dataset_id],
        ).fetchone()
        if not row:
            missing.append(ds.dataset_id)
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Datasets not found: {missing}",
        )

    running = cur.execute(
        "SELECT study_id FROM optimization_studies WHERE status = 'running' LIMIT 1"
    ).fetchone()
    if running:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An optimization study is already in progress ({running[0]}). "
                "Only one concurrent study is supported."
            ),
        )

    try:
        from asrbench.engine.search.space import ParameterSpace

        ParameterSpace.from_dict(req.space)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid parameter space: {exc}") from exc

    try:
        _build_objective(req.objective)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid objective: {exc}") from exc

    # Two study rows — mirror the two-stage endpoint. The dataset_id column
    # gets the first dataset's id as a stand-in (the "primary" dataset);
    # the full fleet is captured in the study's reasoning blob.
    primary_dataset_id = req.datasets[0].dataset_id
    primary_lang = req.datasets[0].lang

    def _insert_stage(budget_val: int | None, eps_val: float | None) -> str:
        cur.execute(
            """
            INSERT INTO optimization_studies
                (model_id, dataset_id, lang, space, objective, budget,
                 mode, eps_min, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', now())
            RETURNING study_id
            """,
            [
                req.model_id,
                primary_dataset_id,
                primary_lang,
                json.dumps(req.space),
                json.dumps(req.objective.model_dump()),
                json.dumps({"hard_cap": budget_val or 0, "convergence_eps": eps_val or 0.0}),
                req.mode,
                eps_val or 0.0,
            ],
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="Failed to insert study row.")
        return str(row[0])

    stage1_study_id = _insert_stage(req.stage1_budget, req.stage1_epsilon)
    stage2_study_id = _insert_stage(req.stage2_budget, req.stage2_epsilon)

    background_tasks.add_task(
        _run_global_config_background,
        stage1_study_id=stage1_study_id,
        stage2_study_id=stage2_study_id,
        req=req,
        backend_name=backend_name,
        model_local_path=model_local_path,
    )

    return GlobalConfigStartResponse(
        stage1_study_id=stage1_study_id,
        stage2_study_id=stage2_study_id,
        status="running",
        mode=req.mode,
        dataset_count=len(req.datasets),
    )


@router.get("/", response_model=list[StudyResponse])
async def list_studies(status: str | None = None) -> list[StudyResponse]:
    """List optimization studies, optionally filtered by status."""
    conn = get_conn()
    cur = conn.cursor()
    query = """
        SELECT study_id, model_id, dataset_id, lang, mode, status, eps_min,
               best_run_id, best_score, best_config, confidence, total_trials, reasoning,
               started_at, finished_at, created_at, error_message
        FROM optimization_studies
    """
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT 100"
    rows = cur.execute(query, params).fetchall()
    return [
        StudyResponse(
            study_id=str(r[0]),
            model_id=str(r[1]),
            dataset_id=str(r[2]),
            lang=str(r[3]),
            mode=str(r[4]),
            status=str(r[5]),
            eps_min=float(r[6]),
            best_run_id=str(r[7]) if r[7] else None,
            best_score=r[8],
            best_config=json.loads(r[9]) if r[9] else None,
            confidence=str(r[10]) if r[10] else None,
            total_trials=r[11],
            reasoning=json.loads(r[12]) if r[12] else None,
            started_at=str(r[13]) if r[13] else None,
            finished_at=str(r[14]) if r[14] else None,
            created_at=str(r[15]),
            error_message=str(r[16]) if r[16] else None,
        )
        for r in rows
    ]


@router.post("/{study_id}/cancel", response_model=StudyResponse)
async def cancel_study(study_id: str) -> StudyResponse:
    """
    Force-cancel a stuck running study.

    Marks status as 'cancelled' so a new study can start.
    Safe to call on a study whose background task has already died.
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
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT study_id, model_id, dataset_id, lang, mode, status, eps_min,
               best_run_id, best_score, best_config, confidence, total_trials, reasoning,
               started_at, finished_at, created_at, error_message
        FROM optimization_studies WHERE study_id = ?
        """,
        [study_id],
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found.")
    return StudyResponse(
        study_id=str(row[0]),
        model_id=str(row[1]),
        dataset_id=str(row[2]),
        lang=str(row[3]),
        mode=str(row[4]),
        status=str(row[5]),
        eps_min=float(row[6]),
        best_run_id=str(row[7]) if row[7] else None,
        best_score=row[8],
        best_config=json.loads(row[9]) if row[9] else None,
        confidence=str(row[10]) if row[10] else None,
        total_trials=row[11],
        reasoning=json.loads(row[12]) if row[12] else None,
        started_at=str(row[13]) if row[13] else None,
        finished_at=str(row[14]) if row[14] else None,
        created_at=str(row[15]),
        error_message=str(row[16]) if row[16] else None,
    )


@router.get("/{study_id}/trials", response_model=list[TrialResponse])
async def list_trials(
    study_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, le=1000),
    phase: str | None = None,
) -> list[TrialResponse]:
    """Paginated access to the full trial log for audit / UI replay."""
    conn = get_conn()
    cur = conn.cursor()
    if not cur.execute(
        "SELECT 1 FROM optimization_studies WHERE study_id = ?", [study_id]
    ).fetchone():
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found.")

    offset = (page - 1) * page_size
    conditions = ["study_id = ?"]
    params: list[Any] = [study_id]
    if phase:
        conditions.append("phase = ?")
        params.append(phase)
    where = " AND ".join(conditions)
    params.extend([page_size, offset])

    rows = cur.execute(
        f"""
        SELECT trial_id, run_id, phase, config, score,
               score_ci_lower, score_ci_upper, reasoning, created_at
        FROM optimization_trials
        WHERE {where}
        ORDER BY created_at
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()

    return [
        TrialResponse(
            trial_id=str(r[0]),
            run_id=str(r[1]) if r[1] else None,
            phase=str(r[2]),
            config=json.loads(r[3]) if isinstance(r[3], str) else (r[3] or {}),
            score=r[4],
            score_ci_lower=r[5],
            score_ci_upper=r[6],
            reasoning=str(r[7]) if r[7] else None,
            created_at=str(r[8]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_objective(cfg: ObjectiveConfig) -> Objective:
    """Convert ObjectiveConfig into a concrete Objective instance."""
    from asrbench.engine.search.objective import (
        SingleMetricObjective,
        WeightedObjective,
    )

    if cfg.type == "single":
        if not cfg.metric:
            raise ValueError("single objective requires 'metric'")
        direction = cfg.direction or None
        return SingleMetricObjective(metric=cfg.metric, direction=direction)  # type: ignore[arg-type]
    if cfg.type == "weighted":
        if not cfg.weights:
            raise ValueError("weighted objective requires non-empty 'weights'")
        return WeightedObjective(weights=cfg.weights)
    raise ValueError(f"Unknown objective type {cfg.type!r}")


async def _run_study_background(
    *,
    study_id: str,
    req: OptimizeStartRequest,
    backend_name: str,
    model_local_path: str,
) -> None:
    """
    Background task: prepare dataset, load backend, run IAMS, persist results.

    Any exception here is logged and the study row is marked 'failed'. The
    caller (FastAPI BackgroundTasks) does not propagate exceptions to the
    HTTP response — the original /start call has already returned.
    """
    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager
    from asrbench.engine.benchmark import BenchmarkEngine
    from asrbench.engine.optimizer import IAMSOptimizer
    from asrbench.engine.search.benchmark_executor import BenchmarkTrialExecutor
    from asrbench.engine.search.budget import BudgetController
    from asrbench.engine.search.space import ParameterSpace

    conn = get_conn()
    cur = conn.cursor()
    try:
        # Build dependencies
        logger.info("Study %s: preparing dataset %s", study_id, req.dataset_id)
        config = get_config()
        dm = DatasetManager(config, conn)
        prepared = dm.prepare(req.dataset_id)
        logger.info(
            "Study %s: dataset ready (%d segments, %.1fs)",
            study_id,
            len(prepared.segments),
            prepared.duration_s,
        )

        from asrbench.backends import load_backends

        backends = load_backends()
        if backend_name not in backends:
            raise RuntimeError(
                f"Backend '{backend_name}' is not installed. "
                f"Install it: pip install 'asrbench[{backend_name}]'"
            )
        backend_cls = backends[backend_name]
        backend_instance = backend_cls()

        # Load the model once; the executor reuses the same backend across trials
        space = ParameterSpace.from_dict(req.space)
        logger.info("Study %s: loading model %s", study_id, model_local_path)
        backend_instance.load(model_local_path, space.defaults())
        logger.info("Study %s: model loaded, starting optimizer", study_id)

        try:
            engine = BenchmarkEngine(conn, cache_dir=config.storage.cache_dir)
            objective = _build_objective(req.objective)
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

            # Warm start: load prior study's trials + screening
            prior_screening = None
            if req.prior_study_id:
                prior_screening = _warm_start(cur, executor, req.prior_study_id, space)

            # Build a mode hint so the backend can advertise which params it
            # actually honors (batched vs sequential, for instance). The space
            # default for batch_size is the single source of truth — callers
            # that want batched mode declare it there.
            mode_hint: dict[str, Any] = {}
            try:
                mode_hint["batch_size"] = int(space.defaults().get("batch_size", 0))
            except (TypeError, ValueError):
                mode_hint["batch_size"] = 0

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
                mode_hint=mode_hint,
            )
            result = optimizer.run()
        finally:
            backend_instance.unload()

        # Persist final result + screening summary to the study row
        screening_json = json.dumps(
            {
                "sensitive_order": list(result.screening.sensitive_order),
                "insensitive": list(result.screening.insensitive),
            }
        )
        cur.execute(
            """
            UPDATE optimization_studies
            SET status = 'completed',
                best_run_id = ?,
                best_score = ?,
                best_config = ?,
                confidence = ?,
                total_trials = ?,
                reasoning = ?,
                screening_result = ?,
                finished_at = now()
            WHERE study_id = ?
            """,
            [
                result.best_trial.trial_id,
                result.best_trial.score,
                json.dumps(dict(result.best_config)),
                result.validation.confidence if result.validation else None,
                result.total_trials,
                json.dumps(list(result.reasoning)),
                screening_json,
                study_id,
            ],
        )
        logger.info(
            "Study %s completed: best_score=%.4f, %d trials",
            study_id,
            result.best_trial.score,
            result.total_trials,
        )
    except Exception as exc:
        import traceback

        err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("Study %s failed: %s", study_id, exc, exc_info=True)
        cur.execute(
            "UPDATE optimization_studies SET status = 'failed', finished_at = now(), "
            "error_message = ? WHERE study_id = ?",
            [err_msg[:4000], study_id],
        )


async def _run_two_stage_background(
    *,
    stage1_study_id: str,
    stage2_study_id: str,
    req: TwoStageStartRequest,
    backend_name: str,
    model_local_path: str,
) -> None:
    """
    Background task: run the library's ``run_two_stage`` orchestrator and
    persist each stage's result to its own ``optimization_studies`` row.

    Both stage rows were inserted with ``status='running'`` by the endpoint
    handler. On success both get marked ``completed``; on failure the stage
    that blew up (or both, if the error came before Stage 1) is marked
    ``failed`` with the exception text.
    """
    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager
    from asrbench.engine.benchmark import BenchmarkEngine
    from asrbench.engine.search.benchmark_executor import BenchmarkTrialExecutor
    from asrbench.engine.search.space import ParameterSpace
    from asrbench.engine.two_stage import TwoStageConfig, run_two_stage

    conn = get_conn()
    cur = conn.cursor()
    current_stage_id = stage1_study_id  # for error reporting
    try:
        config = get_config()
        dm = DatasetManager(config, conn)

        from asrbench.backends import load_backends

        backends = load_backends()
        if backend_name not in backends:
            raise RuntimeError(
                f"Backend '{backend_name}' is not installed. "
                f"Install it: pip install 'asrbench[{backend_name}]'"
            )
        backend_cls = backends[backend_name]
        backend_instance = backend_cls()

        space = ParameterSpace.from_dict(req.space)
        objective = _build_objective(req.objective)

        logger.info(
            "Two-stage %s/%s: loading model %s",
            stage1_study_id,
            stage2_study_id,
            model_local_path,
        )
        backend_instance.load(model_local_path, space.defaults())

        try:
            engine = BenchmarkEngine(conn, cache_dir=config.storage.cache_dir)

            # run_two_stage calls dataset_loader and executor_factory twice
            # — once per stage, in order. A simple factory-call counter is
            # enough to bind each stage's trial rows to the right study_id.
            stage_ids = [stage1_study_id, stage2_study_id]
            factory_call_count = {"n": 0}

            def dataset_loader(duration_s: int):  # type: ignore[no-untyped-def]
                current = stage_ids[min(factory_call_count["n"], 1)]
                logger.info(
                    "Two-stage %s: preparing dataset %s (cap=%ds)",
                    current,
                    req.dataset_id,
                    duration_s,
                )
                return dm.prepare(req.dataset_id, max_duration_s=float(duration_s))

            def executor_factory(prepared, duration_s):  # type: ignore[no-untyped-def]
                _ = duration_s
                current = stage_ids[min(factory_call_count["n"], 1)]
                factory_call_count["n"] += 1
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

            mode_hint: dict[str, Any] = {}
            try:
                mode_hint["batch_size"] = int(space.defaults().get("batch_size", 0))
            except (TypeError, ValueError):
                mode_hint["batch_size"] = 0

            cfg = TwoStageConfig(
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

            current_stage_id = stage1_study_id
            result = run_two_stage(
                space=space,
                objective=objective,
                backend=backend_instance,
                mode_hint=mode_hint,
                dataset_loader=dataset_loader,
                executor_factory=executor_factory,
                cfg=cfg,
            )
        finally:
            backend_instance.unload()

        # Persist both stage rows.
        for stage_id, stage_result in (
            (stage1_study_id, result.stage1),
            (stage2_study_id, result.stage2),
        ):
            screening_json = json.dumps(
                {
                    "sensitive_order": list(stage_result.screening.sensitive_order),
                    "insensitive": list(stage_result.screening.insensitive),
                }
            )
            cur.execute(
                """
                UPDATE optimization_studies
                SET status = 'completed',
                    best_run_id = ?,
                    best_score = ?,
                    best_config = ?,
                    confidence = ?,
                    total_trials = ?,
                    reasoning = ?,
                    screening_result = ?,
                    finished_at = now()
                WHERE study_id = ?
                """,
                [
                    stage_result.best_trial.trial_id,
                    stage_result.best_trial.score,
                    json.dumps(dict(stage_result.best_config)),
                    stage_result.validation.confidence if stage_result.validation else None,
                    stage_result.total_trials,
                    json.dumps(list(stage_result.reasoning)),
                    screening_json,
                    stage_id,
                ],
            )
        logger.info(
            "Two-stage complete: stage1=%s (%d trials), stage2=%s (%d trials)",
            stage1_study_id,
            result.stage1.total_trials,
            stage2_study_id,
            result.stage2.total_trials,
        )
    except Exception as exc:
        import traceback

        err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error(
            "Two-stage %s failed at stage %s: %s",
            stage2_study_id,
            current_stage_id,
            exc,
            exc_info=True,
        )
        # Mark any still-running stage rows as failed.
        for sid in (stage1_study_id, stage2_study_id):
            row = cur.execute(
                "SELECT status FROM optimization_studies WHERE study_id = ?",
                [sid],
            ).fetchone()
            if row and str(row[0]) == "running":
                cur.execute(
                    "UPDATE optimization_studies SET status = 'failed', "
                    "finished_at = now(), error_message = ? WHERE study_id = ?",
                    [err_msg[:4000], sid],
                )


async def _run_global_config_background(
    *,
    stage1_study_id: str,
    stage2_study_id: str,
    req: GlobalConfigStartRequest,
    backend_name: str,
    model_local_path: str,
) -> None:
    """
    Background task: run library's ``run_two_stage`` over a
    ``MultiDatasetTrialExecutor`` wrapping N prepared datasets.

    Each logical IAMS trial hits every dataset and aggregates scores; the
    7-layer algorithm produces ONE config optimized for the fleet. Per-
    dataset trial rows still go to ``optimization_trials`` (one per
    dataset per trial), so the audit trail shows component breakdowns.
    """
    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager
    from asrbench.engine.benchmark import BenchmarkEngine
    from asrbench.engine.search.benchmark_executor import BenchmarkTrialExecutor
    from asrbench.engine.search.multidataset import MultiDatasetTrialExecutor
    from asrbench.engine.search.space import ParameterSpace
    from asrbench.engine.two_stage import TwoStageConfig, run_two_stage

    conn = get_conn()
    cur = conn.cursor()
    current_stage_id = stage1_study_id
    try:
        config = get_config()
        dm = DatasetManager(config, conn)

        from asrbench.backends import load_backends

        backends = load_backends()
        if backend_name not in backends:
            raise RuntimeError(
                f"Backend '{backend_name}' is not installed. "
                f"Install it: pip install 'asrbench[{backend_name}]'"
            )
        backend_cls = backends[backend_name]
        backend_instance = backend_cls()

        space = ParameterSpace.from_dict(req.space)
        objective = _build_objective(req.objective)

        logger.info(
            "Global-config %s/%s: loading model %s (N=%d datasets)",
            stage1_study_id,
            stage2_study_id,
            model_local_path,
            len(req.datasets),
        )
        backend_instance.load(model_local_path, space.defaults())

        try:
            engine = BenchmarkEngine(conn, cache_dir=config.storage.cache_dir)

            # Stage-binding counter: run_two_stage calls dataset_loader /
            # executor_factory twice — once per stage — in order.
            stage_ids = [stage1_study_id, stage2_study_id]
            factory_call_count = {"n": 0}
            weights = [ds.weight for ds in req.datasets]
            labels = [f"{ds.dataset_id[:8]}_{ds.lang}" for ds in req.datasets]

            def dataset_loader(duration_s: int):  # type: ignore[no-untyped-def]
                current = stage_ids[min(factory_call_count["n"], 1)]
                logger.info(
                    "Global-config %s: preparing %d datasets at cap=%ds",
                    current,
                    len(req.datasets),
                    duration_s,
                )
                prepared = [
                    dm.prepare(ds.dataset_id, max_duration_s=float(duration_s))
                    for ds in req.datasets
                ]
                return prepared  # opaque bundle — factory closes over it

            def executor_factory(prepared_bundle, duration_s):  # type: ignore[no-untyped-def]
                _ = duration_s
                current = stage_ids[min(factory_call_count["n"], 1)]
                factory_call_count["n"] += 1
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
                return MultiDatasetTrialExecutor(
                    executors=inner,
                    weights=weights,
                    labels=labels,
                )

            mode_hint: dict[str, Any] = {}
            try:
                mode_hint["batch_size"] = int(space.defaults().get("batch_size", 0))
            except (TypeError, ValueError):
                mode_hint["batch_size"] = 0

            cfg = TwoStageConfig(
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

            current_stage_id = stage1_study_id
            result = run_two_stage(
                space=space,
                objective=objective,
                backend=backend_instance,
                mode_hint=mode_hint,
                dataset_loader=dataset_loader,
                executor_factory=executor_factory,
                cfg=cfg,
            )
        finally:
            backend_instance.unload()

        # Persist both stage rows. ``reasoning`` gets a "datasets=[...]"
        # header so audit consumers can see which fleet this global
        # config was tuned against.
        dataset_fleet_summary = [
            {"dataset_id": ds.dataset_id, "lang": ds.lang, "weight": ds.weight}
            for ds in req.datasets
        ]
        for stage_id, stage_result in (
            (stage1_study_id, result.stage1),
            (stage2_study_id, result.stage2),
        ):
            screening_json = json.dumps(
                {
                    "sensitive_order": list(stage_result.screening.sensitive_order),
                    "insensitive": list(stage_result.screening.insensitive),
                }
            )
            reasoning_with_fleet = [
                f"[global-config over {len(req.datasets)} datasets: {dataset_fleet_summary}]",
                *stage_result.reasoning,
            ]
            cur.execute(
                """
                UPDATE optimization_studies
                SET status = 'completed',
                    best_run_id = ?,
                    best_score = ?,
                    best_config = ?,
                    confidence = ?,
                    total_trials = ?,
                    reasoning = ?,
                    screening_result = ?,
                    finished_at = now()
                WHERE study_id = ?
                """,
                [
                    stage_result.best_trial.trial_id,
                    stage_result.best_trial.score,
                    json.dumps(dict(stage_result.best_config)),
                    stage_result.validation.confidence if stage_result.validation else None,
                    stage_result.total_trials,
                    json.dumps(reasoning_with_fleet),
                    screening_json,
                    stage_id,
                ],
            )
        logger.info(
            "Global-config complete: stage1=%s, stage2=%s, N_datasets=%d",
            stage1_study_id,
            stage2_study_id,
            len(req.datasets),
        )
    except Exception as exc:
        import traceback

        err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error(
            "Global-config %s failed at stage %s: %s",
            stage2_study_id,
            current_stage_id,
            exc,
            exc_info=True,
        )
        for sid in (stage1_study_id, stage2_study_id):
            row = cur.execute(
                "SELECT status FROM optimization_studies WHERE study_id = ?",
                [sid],
            ).fetchone()
            if row and str(row[0]) == "running":
                cur.execute(
                    "UPDATE optimization_studies SET status = 'failed', "
                    "finished_at = now(), error_message = ? WHERE study_id = ?",
                    [err_msg[:4000], sid],
                )


def _warm_start(
    cur: Any,
    executor: Any,
    prior_study_id: str,
    space: Any,
) -> Any:
    """
    Import prior-study knowledge into the current optimizer.

    Two distinct things are carried over:

    1. **Screening metadata** (sensitive_order, insensitive) — always valid to
       reuse because it describes which parameters are worth touching, not
       their exact scores. Passed back via IAMSOptimizer(prior_screening=...).

    2. **Raw trial scores** — only valid when the prior study was measured
       under the SAME (model, dataset, language) context. A 15-min Stage-1
       score is NOT a drop-in replacement for a 60-min Stage-2 score; blindly
       caching it would silently return stale numbers on the first cache hit.
       We therefore thread context through `warm_load(source_*=...)` and let
       the executor refuse mismatches.
    """
    from asrbench.engine.search.screening import ScreeningResult
    from asrbench.engine.search.trial import (
        TrialResult,
        canonical_config_repr,
    )

    # Load screening summary + prior context
    row = cur.execute(
        """
        SELECT screening_result, model_id, dataset_id, lang
        FROM optimization_studies
        WHERE study_id = ?
        """,
        [prior_study_id],
    ).fetchone()
    if not row:
        logger.warning("Prior study %s not found — warm start skipped", prior_study_id)
        return None
    screening_json = json.loads(row[0]) if row[0] else None
    prior_model_id = str(row[1]) if row[1] else None
    prior_dataset_id = str(row[2]) if row[2] else None
    prior_lang = str(row[3]) if row[3] else None
    if screening_json is None:
        logger.warning(
            "Prior study %s has no screening_result — warm start skipped",
            prior_study_id,
        )
        return None

    # Load all trials from the prior study
    trial_rows = cur.execute(
        """
        SELECT config, score, score_ci_lower, score_ci_upper, phase, trial_id
        FROM optimization_trials
        WHERE study_id = ?
        ORDER BY created_at
        """,
        [prior_study_id],
    ).fetchall()

    prior_trials = []
    for tr in trial_rows:
        config = json.loads(tr[0]) if isinstance(tr[0], str) else (tr[0] or {})
        prior_trials.append(
            TrialResult.from_db_row(
                config=config,
                score=float(tr[1]),
                score_ci=(float(tr[2]), float(tr[3])),
                phase=str(tr[4]) if tr[4] else "prior",
                trial_id=str(tr[5]) if tr[5] else None,
            )
        )

    # Context-guarded warm load: the executor refuses (returns 0) whenever
    # source context differs. Screening metadata below is still passed
    # through in that case.
    loaded = executor.warm_load(
        prior_trials,
        source_model_id=prior_model_id,
        source_dataset_id=prior_dataset_id,
        source_lang=prior_lang,
    )
    if loaded == 0 and prior_trials:
        logger.info(
            "Warm start from study %s: score cache NOT reused "
            "(context mismatch or refused) — screening metadata reused only.",
            prior_study_id,
        )
    else:
        logger.info(
            "Warm start from study %s: %d trials loaded into cache (of %d total)",
            prior_study_id,
            loaded,
            len(prior_trials),
        )

    # Build a baseline stand-in from defaults for ScreeningResult.from_summary.
    # Match on canonical config repr (deterministic, order-independent) so the
    # lookup is stable regardless of PYTHONHASHSEED.
    baseline_config = space.defaults()
    baseline_repr = canonical_config_repr(baseline_config)
    baseline = None
    for t in prior_trials:
        if canonical_config_repr(t.config) == baseline_repr:
            baseline = t
            break
    if baseline is None and prior_trials:
        baseline = min(prior_trials, key=lambda t: t.score)
    if baseline is None:
        logger.warning("No usable baseline in prior study %s", prior_study_id)
        return None

    return ScreeningResult.from_summary(screening_json, baseline)
