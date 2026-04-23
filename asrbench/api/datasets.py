"""Dataset CRUD endpoints."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from pydantic import BaseModel

from asrbench.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])

_SUPPORTED_SOURCES: frozenset[str] = frozenset(
    {
        "common_voice",
        "fleurs",
        "yodas",
        "ted_lium",
        "librispeech",
        "earnings22",
        "mediaspeech",
        "custom",
    }
)


class FetchRequest(BaseModel):
    source: str
    lang: str = "en"
    split: str = "test"
    local_path: str | None = None
    max_duration_s: float | None = None


class FetchResponse(BaseModel):
    dataset_id: str
    name: str
    status: str
    stream_url: str


class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    lang: str
    split: str
    source: str
    duration_s: float | None = None
    verified: bool = False


@router.post("/fetch", response_model=FetchResponse, status_code=202)
async def fetch_dataset(req: FetchRequest, background_tasks: BackgroundTasks) -> FetchResponse:
    """Start fetching a dataset in the background."""
    import uuid

    if req.source not in _SUPPORTED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Source '{req.source}' is not supported. "
                f"Valid sources: {', '.join(sorted(_SUPPORTED_SOURCES))}."
            ),
        )

    conn = get_conn()
    cur = conn.cursor()

    # Build a unique name that includes duration when capped
    if req.max_duration_s and req.max_duration_s > 0:
        dur_min = int(req.max_duration_s / 60)
        name = f"{req.source}_{req.lang}_{req.split}_{dur_min}m"
    else:
        name = f"{req.source}_{req.lang}_{req.split}"

    existing = cur.execute(
        "SELECT dataset_id, name FROM datasets WHERE name = ?",
        [name],
    ).fetchone()

    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Dataset named '{name}' already exists. "
                "Delete it first or pick a different source/lang/split."
            ),
        )

    dataset_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO datasets "
        "(dataset_id, name, source, lang, split, local_path, max_duration_s, verified) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, false)",
        [dataset_id, name, req.source, req.lang, req.split, req.local_path, req.max_duration_s],
    )

    background_tasks.add_task(
        _fetch_background,
        dataset_id=dataset_id,
        source=req.source,
        lang=req.lang,
        split=req.split,
        local_path=req.local_path,
        max_duration_s=req.max_duration_s,
    )

    return FetchResponse(
        dataset_id=dataset_id,
        name=name,
        status="downloading",
        stream_url=f"/ws/logs?dataset_id={dataset_id}",
    )


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(
    lang: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> list[DatasetResponse]:
    """List all registered datasets, optionally filtered by language and/or source."""
    conn = get_conn()
    cur = conn.cursor()

    where: list[str] = []
    params: list[str] = []
    if lang:
        where.append("lang = ?")
        params.append(lang)
    if source:
        where.append("source = ?")
        params.append(source)

    sql = "SELECT dataset_id, name, lang, split, source, local_path, verified FROM datasets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY name"

    rows = cur.execute(sql, params).fetchall()

    return [
        DatasetResponse(
            dataset_id=str(r[0]),
            name=str(r[1]),
            lang=str(r[2]),
            split=str(r[3]),
            source=str(r[4]),
            verified=bool(r[6]),
        )
        for r in rows
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str) -> DatasetResponse:
    """Return a single dataset's metadata."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT dataset_id, name, lang, split, source, local_path, verified "
        "FROM datasets WHERE dataset_id = ?",
        [dataset_id],
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    return DatasetResponse(
        dataset_id=str(row[0]),
        name=str(row[1]),
        lang=str(row[2]),
        split=str(row[3]),
        source=str(row[4]),
        verified=bool(row[6]),
    )


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    delete_files: bool = Query(default=False),
) -> Response:
    """Delete a dataset row and, optionally, its cached audio files.

    Returns 409 when at least one run references the dataset — delete or
    retry those runs first to avoid orphaning them.
    """
    conn = get_conn()
    cur = conn.cursor()

    row = cur.execute(
        "SELECT dataset_id, local_path FROM datasets WHERE dataset_id = ?",
        [dataset_id],
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    ref_count = cur.execute(
        "SELECT count(*) FROM runs WHERE dataset_id = ?",
        [dataset_id],
    ).fetchone()
    if ref_count is not None and int(ref_count[0]) > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot delete dataset '{dataset_id}': {int(ref_count[0])} run(s) "
                "reference it. Delete or retry those runs first."
            ),
        )

    if delete_files:
        local_path = row[1]
        if local_path:
            path = Path(str(local_path))
            if path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove dataset files at %s: %s", path, exc)

    cur.execute("DELETE FROM datasets WHERE dataset_id = ?", [dataset_id])
    return Response(status_code=204)


async def _fetch_background(
    *,
    dataset_id: str,
    source: str,
    lang: str,
    split: str,
    local_path: str | None,
    max_duration_s: float | None,
) -> None:
    """Background task: download and cache the dataset, mark as verified on success."""
    import asyncio

    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager
    from asrbench.engine.events import get_event_bus

    conn = get_conn()
    bus = get_event_bus()
    topic = f"datasets:{dataset_id}"

    await bus.publish(topic, {"type": "fetch_start", "dataset_id": dataset_id})

    try:
        config = get_config()
        dm = DatasetManager(config, conn)

        def _do_fetch() -> None:
            if local_path:
                dm._fetch_local(local_path, lang)
            else:
                prepared = dm._fetch_hf(source, lang, split, max_duration_s)
                cache_key = dm._cache.cache_key(source, lang, split, max_duration_s)
                dm._cache.save(cache_key, prepared)

        # DatasetManager does blocking I/O — run it off the event loop so
        # concurrent WS broadcasts and other requests keep responding.
        await asyncio.to_thread(_do_fetch)

        conn.cursor().execute(
            "UPDATE datasets SET verified = true WHERE dataset_id = ?",
            [dataset_id],
        )
        logger.info("Dataset %s fetch complete", dataset_id)
        await bus.publish(
            topic,
            {"type": "verified", "dataset_id": dataset_id, "verified": True},
        )
    except Exception as exc:
        logger.error("Dataset %s fetch failed: %s", dataset_id, exc)
        await bus.publish(
            topic,
            {"type": "error", "dataset_id": dataset_id, "error": str(exc)},
        )
