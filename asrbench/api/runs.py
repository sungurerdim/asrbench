"""Benchmark run endpoints."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from asrbench.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

_RETRYABLE_STATUSES: frozenset[str] = frozenset({"failed", "cancelled"})


def request_cancel(run_id: str) -> None:
    """Flag a run as cancellation-requested in the DB.

    The flag lives on the ``runs`` row itself (``cancel_requested`` column)
    so an externally-triggered cancel survives across worker restarts and
    is visible to any process that shares the DB file — not just the
    FastAPI process that received the request.
    """
    get_conn().cursor().execute(
        "UPDATE runs SET cancel_requested = true WHERE run_id = ?",
        [run_id],
    )


def is_cancel_requested(run_id: str) -> bool:
    """Return True iff the row's ``cancel_requested`` flag is set.

    Benchmark engine polls this between segments; the check is cheap
    (indexed primary-key lookup on a tiny table).
    """
    row = (
        get_conn()
        .cursor()
        .execute("SELECT cancel_requested FROM runs WHERE run_id = ?", [run_id])
        .fetchone()
    )
    return bool(row and row[0])


def clear_cancel(run_id: str) -> None:
    """Reset the cancel flag once a run reaches a terminal state.

    Keeps the semantics from the old in-memory set: retry picks up a
    fresh flag regardless of whether the previous attempt was cancelled.
    """
    get_conn().cursor().execute(
        "UPDATE runs SET cancel_requested = false WHERE run_id = ?",
        [run_id],
    )


class RunStartRequest(BaseModel):
    model_id: str
    dataset_id: str
    lang: str = "en"
    mode: str = "model_compare"
    params: dict | None = None
    matrix: dict[str, list[Any]] | None = None
    label: str | None = None


class RunStartResponse(BaseModel):
    run_id: str
    status: str


class SegmentResponse(BaseModel):
    offset_s: float
    duration_s: float
    ref_text: str
    hyp_text: str
    wer: float | None = None
    rtfx: float | None = None


class AggregateResponse(BaseModel):
    wer_mean: float | None = None
    cer_mean: float | None = None
    mer_mean: float | None = None
    wil_mean: float | None = None
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
    runs: list[dict[str, Any]]
    params_diff: list[str]
    params_same: list[str]
    wilcoxon_p: float | None = None


class CancelResponse(BaseModel):
    run_id: str
    status: str


class RetryResponse(BaseModel):
    original_run_id: str
    new_run_id: str
    status: str


@router.post("/start", response_model=RunStartResponse, status_code=202)
async def start_run(req: RunStartRequest, background_tasks: BackgroundTasks) -> RunStartResponse:
    """Start a benchmark run in the background."""
    from asrbench.config import get_config
    from asrbench.engine.matrix import MatrixBuilder

    conn = get_conn()
    cur = conn.cursor()

    # Enforce max_concurrent_runs by counting rows that are actively in-flight.
    limits = get_config().limits
    in_flight_row = cur.execute(
        "SELECT count(*) FROM runs WHERE status IN ('pending', 'running')"
    ).fetchone()
    in_flight = int(in_flight_row[0]) if in_flight_row else 0
    if in_flight >= limits.max_concurrent_runs:
        raise HTTPException(
            status_code=409,
            detail=(
                f"At least one benchmark is already running "
                f"(max_concurrent_runs={limits.max_concurrent_runs}). "
                "Wait for it to finish or cancel it first."
            ),
        )

    # Validate matrix early — empty value lists are a common client bug.
    if req.mode == "param_compare" and req.matrix is not None:
        try:
            MatrixBuilder().build_matrix(req.matrix, req.params or {}, req.mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    dataset_row = cur.execute(
        "SELECT dataset_id FROM datasets WHERE dataset_id = ?",
        [req.dataset_id],
    ).fetchone()
    if not dataset_row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{req.dataset_id}' not found.",
        )

    params = {**default_params, **(req.params or {})}
    params_json = json.dumps(params)

    cur.execute(
        "INSERT INTO runs (model_id, backend, params, dataset_id, lang, mode, status, label) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?) RETURNING run_id",
        [
            req.model_id,
            backend_name,
            params_json,
            req.dataset_id,
            req.lang,
            req.mode,
            req.label,
        ],
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
    lang: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=1000),
) -> list[RunResponse]:
    """List runs with optional status/lang filters and a result cap."""
    conn = get_conn()
    cur = conn.cursor()

    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if lang:
        where.append("lang = ?")
        params.append(lang)

    sql = "SELECT run_id, model_id, backend, lang, status, params, label FROM runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY run_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    rows = cur.execute(sql, params).fetchall()
    return [_run_from_row(r) for r in rows]


@router.get("/compare", response_model=CompareResponse)
async def compare_runs(
    ids: str = Query(..., description="Comma-separated run IDs"),
) -> CompareResponse:
    """Compare metrics of two or more runs side by side."""
    from asrbench.engine.compare import CompareEngine, CompareInput

    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]
    if len(run_ids) < 2:
        raise HTTPException(
            status_code=422,
            detail="Provide at least 2 comma-separated run IDs.",
        )

    conn = get_conn()
    cur = conn.cursor()

    inputs: list[CompareInput] = []
    missing: list[str] = []
    for rid in run_ids:
        row = cur.execute(
            "SELECT run_id, model_id, backend, lang, status, params, label "
            "FROM runs WHERE run_id = ?",
            [rid],
        ).fetchone()
        if not row:
            missing.append(rid)
            continue
        agg = _get_aggregate(cur, str(row[0]))
        inputs.append(
            CompareInput(
                run_id=str(row[0]),
                params=json.loads(row[5]) if isinstance(row[5], str) else (row[5] or {}),
                aggregate=agg.model_dump() if agg else {},
            )
        )

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Runs not found: {', '.join(missing)}.",
        )

    result = CompareEngine().compare(inputs)
    return CompareResponse(
        runs=result.runs,
        params_diff=result.params_diff,
        params_same=result.params_same,
        wilcoxon_p=result.wilcoxon_p,
    )


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


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: str) -> Response:
    """Delete a run and all its segments/aggregates. Rejects in-flight runs."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT status FROM runs WHERE run_id = ?", [run_id]).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    if str(row[0]) == "running":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run '{run_id}' is currently running. Cancel it with "
                "POST /runs/{id}/cancel before deleting."
            ),
        )

    cur.execute("DELETE FROM segments WHERE run_id = ?", [run_id])
    cur.execute("DELETE FROM aggregates WHERE run_id = ?", [run_id])
    cur.execute("DELETE FROM optimization_trials WHERE run_id = ?", [run_id])
    cur.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
    clear_cancel(run_id)
    return Response(status_code=204)


@router.post("/{run_id}/cancel", response_model=CancelResponse)
async def cancel_run(run_id: str) -> CancelResponse:
    """Request cancellation of an in-progress run."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT status FROM runs WHERE run_id = ?", [run_id]).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    status = str(row[0])
    if status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' is not running (status={status!r}); nothing to cancel.",
        )

    request_cancel(run_id)
    return CancelResponse(run_id=run_id, status="cancellation_requested")


@router.post("/{run_id}/retry", response_model=RetryResponse, status_code=202)
async def retry_run(run_id: str, background_tasks: BackgroundTasks) -> RetryResponse:
    """Re-run a failed or cancelled run with the same params; creates a new run row."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT run_id, model_id, backend, params, dataset_id, lang, mode, label, status "
        "FROM runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    status = str(row[8])
    if status not in _RETRYABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Only failed or cancelled runs can be retried (status={status!r}). "
                "Use /runs/start for a fresh run."
            ),
        )

    model_id = str(row[1])
    backend_name = str(row[2])
    params_json = row[3] if isinstance(row[3], str) else json.dumps(row[3] or {})
    params = json.loads(params_json) if isinstance(params_json, str) else {}
    dataset_id = str(row[4])
    lang = str(row[5])
    mode = str(row[6]) if row[6] else "model_compare"
    label = str(row[7]) if row[7] is not None else None

    model_row = cur.execute(
        "SELECT local_path, family FROM models WHERE model_id = ?",
        [model_id],
    ).fetchone()
    if not model_row:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' for original run no longer exists.",
        )
    model_local_path = str(model_row[0])
    model_family = str(model_row[1]) if model_row[1] else None

    cur.execute(
        "INSERT INTO runs (model_id, backend, params, dataset_id, lang, mode, status, label) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?) RETURNING run_id",
        [model_id, backend_name, params_json, dataset_id, lang, mode, label],
    )
    new_row = cur.fetchone()
    if new_row is None:
        raise HTTPException(status_code=500, detail="Failed to insert retry run row.")
    new_run_id = str(new_row[0])

    background_tasks.add_task(
        _run_background,
        run_id=new_run_id,
        model_id=model_id,
        backend_name=backend_name,
        model_local_path=model_local_path,
        model_family=model_family,
        dataset_id=dataset_id,
        lang=lang,
        params=params,
    )

    return RetryResponse(
        original_run_id=run_id,
        new_run_id=new_run_id,
        status="pending",
    )


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


@router.get("/{run_id}/export")
async def export_run(
    run_id: str,
    fmt: str = Query(default="json", description="json | csv"),
) -> Response:
    """Export a run's full results as JSON or CSV."""
    if fmt not in {"json", "csv"}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported export format '{fmt}'. "
                "Use fmt=json or fmt=csv (PDF export pending — see [pdf] extra)."
            ),
        )

    conn = get_conn()
    cur = conn.cursor()
    run_row = cur.execute(
        "SELECT run_id, model_id, backend, lang, status, params, label FROM runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if not run_row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    run_resp = _run_from_row(run_row)
    run_resp.aggregate = _get_aggregate(cur, run_id)

    seg_rows = cur.execute(
        "SELECT offset_s, duration_s, ref_text, hyp_text FROM segments "
        "WHERE run_id = ? ORDER BY offset_s",
        [run_id],
    ).fetchall()

    if fmt == "json":
        payload = {
            **run_resp.model_dump(),
            "segments": [
                {
                    "offset_s": float(r[0]),
                    "duration_s": float(r[1]),
                    "ref_text": str(r[2] or ""),
                    "hyp_text": str(r[3] or ""),
                }
                for r in seg_rows
            ],
        }
        return Response(
            content=json.dumps(payload, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="run-{run_id}.json"'},
        )

    # CSV path
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["segment_id", "offset_s", "duration_s", "ref_text", "hyp_text"])
    for idx, r in enumerate(seg_rows):
        writer.writerow(
            [
                idx,
                float(r[0]),
                float(r[1]),
                str(r[2] or ""),
                str(r[3] or ""),
            ]
        )
    csv_bytes = buf.getvalue().encode("utf-8")
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="run-{run_id}.csv"'},
    )


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
        "SELECT wer_mean, cer_mean, mer_mean, wil_mean, rtfx_mean, rtfx_p95, "
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
        wil_mean=agg[3],
        rtfx_mean=agg[4],
        rtfx_p95=agg[5],
        vram_peak_mb=agg[6],
        wall_time_s=agg[7],
        word_count=agg[8],
        wer_ci_lower=agg[9],
        wer_ci_upper=agg[10],
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
    from asrbench.engine.benchmark import BenchmarkEngine, RunCancelled
    from asrbench.engine.errors import sanitize_error

    conn = get_conn()
    cur = conn.cursor()

    try:
        config = get_config()

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

    except RunCancelled:
        logger.info("Run %s cancelled by request", run_id)
        cur.execute(
            "UPDATE runs SET status = 'cancelled', error_message = NULL WHERE run_id = ?",
            [run_id],
        )
    except Exception as exc:
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
        cur.execute(
            "UPDATE runs SET status = 'failed', error_message = ? WHERE run_id = ?",
            [sanitize_error(exc), run_id],
        )
    finally:
        clear_cancel(run_id)
