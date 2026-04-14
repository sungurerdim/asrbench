"""Dataset CRUD endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from asrbench.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


class FetchRequest(BaseModel):
    source: str
    lang: str = "en"
    split: str = "test"
    local_path: str | None = None
    max_duration_s: float | None = None


class FetchResponse(BaseModel):
    dataset_id: str
    name: str
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

    conn = get_conn()
    cur = conn.cursor()

    # Build a unique name that includes duration when capped
    if req.max_duration_s and req.max_duration_s > 0:
        dur_min = int(req.max_duration_s / 60)
        name = f"{req.source}_{req.lang}_{req.split}_{dur_min}m"
    else:
        name = f"{req.source}_{req.lang}_{req.split}"

    # Idempotent: name encodes source+lang+split+duration
    existing = cur.execute(
        "SELECT dataset_id, name FROM datasets WHERE name = ?",
        [name],
    ).fetchone()

    if existing:
        dataset_id = str(existing[0])
        return FetchResponse(
            dataset_id=dataset_id,
            name=name,
            stream_url=f"/ws/datasets/{dataset_id}",
        )

    # Insert a pending row
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
        stream_url=f"/ws/datasets/{dataset_id}",
    )


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(
    lang: str | None = Query(default=None),
) -> list[DatasetResponse]:
    """List all registered datasets, optionally filtered by language."""
    conn = get_conn()
    cur = conn.cursor()

    if lang:
        rows = cur.execute(
            "SELECT dataset_id, name, lang, split, source, local_path, verified "
            "FROM datasets WHERE lang = ? ORDER BY name",
            [lang],
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT dataset_id, name, lang, split, source, local_path, verified "
            "FROM datasets ORDER BY name"
        ).fetchall()

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


def _fetch_background(
    *,
    dataset_id: str,
    source: str,
    lang: str,
    split: str,
    local_path: str | None,
    max_duration_s: float | None,
) -> None:
    """Background task: download and cache the dataset, mark as verified on success."""
    from asrbench.config import get_config
    from asrbench.data.dataset_manager import DatasetManager

    conn = get_conn()
    try:
        config = get_config()
        dm = DatasetManager(config, conn)

        if local_path:
            dm._fetch_local(local_path, lang)
        else:
            prepared = dm._fetch_hf(source, lang, split, max_duration_s)
            cache_key = dm._cache.cache_key(source, lang, split, max_duration_s)
            dm._cache.save(cache_key, prepared)

        conn.cursor().execute(
            "UPDATE datasets SET verified = true WHERE dataset_id = ?",
            [dataset_id],
        )
        logger.info("Dataset %s fetch complete", dataset_id)
    except Exception as exc:
        logger.error("Dataset %s fetch failed: %s", dataset_id, exc)
