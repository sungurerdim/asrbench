"""Benchmark run endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from asrbench.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


class RunStartRequest(BaseModel):
    model_id: str
    dataset_id: str
    lang: str = "en"
    params: dict | None = None
    label: str | None = None


class RunStartResponse(BaseModel):
    run_id: str
    status: str


class SegmentResponse(BaseModel):
    offset_s: float
    duration_s: float
    ref_text: str
    hyp_text: str


class AggregateResponse(BaseModel):
    wer_mean: float | None = None
    cer_mean: float | None = None
    mer_mean: float | None = None
    rtfx_mean: float | None = None
    rtfx_p95: float | None = None
    vram_peak_mb: float | None = None
    wall_time_s: float | None = None
    word_count: int | None = None
    wer_ci_lower: float | None = None
    wer_ci_upper: float | None = None


class RunResponse(BaseModel):
    run_id: str
    model_id: str
    backend: str
    lang: str
    status: str
    label: str | None = None
    params: dict | None = None
    aggregate: AggregateResponse | None = None


class CompareResponse(BaseModel):
    runs: list[RunResponse]


@router.post("/start", response_model=RunStartResponse, status_code=202)
async def start_run(req: RunStartRequest, background_tasks: BackgroundTasks) -> RunStartResponse:
    """Start a benchmark run in the background."""
    conn = get_conn()
    cur = conn.cursor()

    # Validate model
    model_row = cur.execute(
        "SELECT model_id, backend, default_params, local_path, family "
        "FROM models WHERE model_id = ?",
        [req.model_id],
    ).fetchone()
    if not model_row:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model_id}' not found.",
        )
    backend_name = str(model_row[1])
    default_params = json.loads(model_row[2]) if model_row[2] else {}
    model_local_path = str(model_row[3])
    model_family = str(model_row[4]) if model_row[4] else None

    # Validate dataset
    dataset_row = cur.execute(
        "SELECT dataset_id FROM datasets WHERE dataset_id = ?",
        [req.dataset_id],
    ).fetchone()
    if not dataset_row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{req.dataset_id}' not found.",
        )

    # Merge params
    params = {**default_params, **(req.params or {})}
    params_json = json.dumps(params)

    # Insert run row
    cur.execute(
        "INSERT INTO runs (model_id, backend, params, dataset_id, lang, status, label) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?) RETURNING run_id",
        [req.model_id, backend_name, params_json, req.dataset_id, req.lang, req.label],
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to insert run row.")
    run_id = str(row[0])

    background_tasks.add_task(
        _run_background,
        run_id=run_id,
        model_id=req.model_id,
        backend_name=backend_name,
        model_local_path=model_local_path,
        model_family=model_family,
        dataset_id=req.dataset_id,
        lang=req.lang,
        params=params,
    )

    return RunStartResponse(run_id=run_id, status="pending")


@router.get("", response_model=list[RunResponse])
async def list_runs(
    status: str | None = Query(default=None),
) -> list[RunResponse]:
    """List all runs, optionally filtered by status."""
    conn = get_conn()
    cur = conn.cursor()

    if status:
        rows = cur.execute(
            "SELECT run_id, model_id, backend, lang, status, params, label "
            "FROM runs WHERE status = ? ORDER BY run_id",
            [status],
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT run_id, model_id, backend, lang, status, params, label "
            "FROM runs ORDER BY run_id"
        ).fetchall()

    return [_run_from_row(r) for r in rows]


@router.get("/compare", response_model=CompareResponse)
async def compare_runs(
    ids: str = Query(..., description="Comma-separated run IDs"),
) -> CompareResponse:
    """Compare metrics of two or more runs side by side."""
    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]
    if len(run_ids) < 2:
        raise HTTPException(
            status_code=422,
            detail="Provide at least 2 comma-separated run IDs.",
        )

    conn = get_conn()
    cur = conn.cursor()
    results: list[RunResponse] = []

    for rid in run_ids:
        row = cur.execute(
            "SELECT run_id, model_id, backend, lang, status, params, label "
            "FROM runs WHERE run_id = ?",
            [rid],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Run '{rid}' not found.")
        run_resp = _run_from_row(row)
        run_resp.aggregate = _get_aggregate(cur, str(row[0]))
        results.append(run_resp)

    return CompareResponse(runs=results)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Return a single run with aggregate metrics."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT run_id, model_id, backend, lang, status, params, label FROM runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    run_resp = _run_from_row(row)
    run_resp.aggregate = _get_aggregate(cur, run_id)
    return run_resp


@router.get("/{run_id}/segments", response_model=list[SegmentResponse])
async def get_segments(
    run_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, le=1000),
) -> list[SegmentResponse]:
    """Paginated segment results for a run."""
    conn = get_conn()
    cur = conn.cursor()

    if not cur.execute("SELECT 1 FROM runs WHERE run_id = ?", [run_id]).fetchone():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    offset = (page - 1) * page_size
    rows = cur.execute(
        "SELECT offset_s, duration_s, ref_text, hyp_text FROM segments "
        "WHERE run_id = ? ORDER BY offset_s LIMIT ? OFFSET ?",
        [run_id, page_size, offset],
    ).fetchall()

    return [
        SegmentResponse(
            offset_s=float(r[0]),
            duration_s=float(r[1]),
            ref_text=str(r[2] or ""),
            hyp_text=str(r[3] or ""),
        )
        for r in rows
    ]


def _run_from_row(row: Any) -> RunResponse:
    """Build a RunResponse from a DB row tuple."""
    params_raw = row[5]
    params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
    return RunResponse(
        run_id=str(row[0]),
        model_id=str(row[1]),
        backend=str(row[2]),
        lang=str(row[3]),
        status=str(row[4]),
        params=params,
        label=str(row[6]) if row[6] is not None else None,
    )


def _get_aggregate(cur: Any, run_id: str) -> AggregateResponse | None:
    """Fetch aggregate metrics for a run."""
    agg = cur.execute(
        "SELECT wer_mean, cer_mean, mer_mean, rtfx_mean, rtfx_p95, "
        "vram_peak_mb, wall_time_s, word_count, wer_ci_lower, wer_ci_upper "
        "FROM aggregates WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if not agg:
        return None
    return AggregateResponse(
        wer_mean=agg[0],
        cer_mean=agg[1],
        mer_mean=agg[2],
        rtfx_mean=agg[3],
        rtfx_p95=agg[4],
        vram_peak_mb=agg[5],
        wall_time_s=agg[6],
        word_count=agg[7],
        wer_ci_lower=agg[8],
        wer_ci_upper=agg[9],
    )


async def _run_background(
    *,
    run_id: str,
    model_id: str,
    backend_name: str,
    model_local_path: str,
    model_family: str | None,
    dataset_id: str,
    lang: str,
    params: dict,
) -> None:
    """Background task: load backend if needed, prepare dataset, run benchmark."""
    from asrbench.api.models import get_loaded_backend
    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager
    from asrbench.engine.benchmark import BenchmarkEngine

    conn = get_conn()
    cur = conn.cursor()

    try:
        config = get_config()

        # Use loaded model from pool, or load fresh
        loaded = get_loaded_backend(model_id)
        auto_loaded = False
        if loaded:
            backend_instance, _ = loaded
        else:
            from asrbench.backends import load_backends

            backends = load_backends()
            if backend_name not in backends:
                raise RuntimeError(f"Backend '{backend_name}' is not installed.")
            backend_instance = backends[backend_name]()
            backend_instance.load(model_local_path, params)
            auto_loaded = True

        try:
            dm = DatasetManager(config, conn)
            prepared = dm.prepare(dataset_id)

            engine = BenchmarkEngine(conn, cache_dir=config.storage.cache_dir)
            await engine.run(
                run_id=run_id,
                backend=backend_instance,
                dataset=prepared,
                params=params,
                model_family=model_family,
                model_local_path=model_local_path,
            )
            logger.info("Run %s completed", run_id)
        finally:
            if auto_loaded:
                backend_instance.unload()

    except Exception as exc:
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
        cur.execute(
            "UPDATE runs SET status = 'failed' WHERE run_id = ?",
            [run_id],
        )
