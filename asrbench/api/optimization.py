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
