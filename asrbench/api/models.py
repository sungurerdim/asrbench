"""Model CRUD and load/unload endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from asrbench.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# In-memory model pool: model_id -> (backend_instance, model_path)
_loaded_models: dict[str, tuple[Any, str]] = {}


class RegisterRequest(BaseModel):
    family: str
    name: str
    backend: str
    local_path: str
    default_params: dict | None = None


class RegisterResponse(BaseModel):
    model_id: str
    name: str


class ModelResponse(BaseModel):
    model_id: str
    family: str
    name: str
    backend: str
    local_path: str
    default_params: dict | None = None
    loaded: bool = False


class LoadResponse(BaseModel):
    model_id: str
    vram_used_mb: float
    vram_total_mb: float


class UnloadResponse(BaseModel):
    model_id: str


@router.get("", response_model=list[ModelResponse])
async def list_models() -> list[ModelResponse]:
    """List all registered models."""
    conn = get_conn()
    rows = (
        conn.cursor()
        .execute(
            "SELECT model_id, family, name, backend, local_path, default_params "
            "FROM models ORDER BY name"
        )
        .fetchall()
    )

    return [
        ModelResponse(
            model_id=str(r[0]),
            family=str(r[1]),
            name=str(r[2]),
            backend=str(r[3]),
            local_path=str(r[4]),
            default_params=json.loads(r[5]) if r[5] else None,
            loaded=str(r[0]) in _loaded_models,
        )
        for r in rows
    ]


@router.post("", response_model=RegisterResponse, status_code=201)
async def register_model(req: RegisterRequest) -> RegisterResponse:
    """Register a new model. Idempotent — returns existing if name + backend match."""
    conn = get_conn()
    cur = conn.cursor()

    # Idempotent check
    existing = cur.execute(
        "SELECT model_id, name FROM models WHERE name = ? AND backend = ?",
        [req.name, req.backend],
    ).fetchone()
    if existing:
        return RegisterResponse(model_id=str(existing[0]), name=str(existing[1]))

    # Validate backend is known
    from asrbench.backends import load_backends

    backends = load_backends()
    if req.backend not in backends:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown backend '{req.backend}'. "
                f"Available: {', '.join(sorted(backends)) or 'none installed'}. "
                f"Install with: pip install asrbench[{req.backend}]"
            ),
        )

    params_json = json.dumps(req.default_params) if req.default_params else None
    cur.execute(
        "INSERT INTO models (family, name, backend, local_path, default_params) "
        "VALUES (?, ?, ?, ?, ?) RETURNING model_id",
        [req.family, req.name, req.backend, req.local_path, params_json],
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to insert model row.")
    return RegisterResponse(model_id=str(row[0]), name=req.name)


@router.post("/{model_id}/load", response_model=LoadResponse)
async def load_model(model_id: str) -> LoadResponse:
    """Load a model into GPU/CPU memory."""
    if model_id in _loaded_models:
        vram = _get_vram()
        return LoadResponse(
            model_id=model_id,
            vram_used_mb=vram[0],
            vram_total_mb=vram[1],
        )

    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT backend, local_path, default_params FROM models WHERE model_id = ?",
        [model_id],
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Register it first via POST /models.",
        )

    backend_name = str(row[0])
    local_path = str(row[1])
    default_params = json.loads(row[2]) if row[2] else {}

    from asrbench.backends import load_backends

    backends = load_backends()
    if backend_name not in backends:
        raise HTTPException(
            status_code=422,
            detail=f"Backend '{backend_name}' is not installed.",
        )

    backend_cls = backends[backend_name]
    backend_instance = backend_cls()

    try:
        backend_instance.load(local_path, default_params)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {exc}",
        ) from exc

    _loaded_models[model_id] = (backend_instance, local_path)
    vram = _get_vram()

    logger.info("Model %s loaded (backend=%s)", model_id, backend_name)
    return LoadResponse(
        model_id=model_id,
        vram_used_mb=vram[0],
        vram_total_mb=vram[1],
    )


@router.post("/{model_id}/unload", response_model=UnloadResponse)
async def unload_model(model_id: str) -> UnloadResponse:
    """Unload a model from memory."""
    entry = _loaded_models.pop(model_id, None)
    if entry is not None:
        backend_instance, _ = entry
        try:
            backend_instance.unload()
        except Exception as exc:
            logger.warning("Error unloading model %s: %s", model_id, exc)
    return UnloadResponse(model_id=model_id)


def get_loaded_backend(model_id: str) -> tuple[Any, str] | None:
    """Return (backend_instance, model_path) if model is loaded, else None."""
    return _loaded_models.get(model_id)


def _get_vram() -> tuple[float, float]:
    """Return (used_mb, total_mb) for the first GPU, or (0, 0) if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return float(mem.used) / 1024 / 1024, float(mem.total) / 1024 / 1024
    except Exception:
        return 0.0, 0.0
