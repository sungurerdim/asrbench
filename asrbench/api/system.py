"""System health and VRAM monitoring endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from asrbench import __version__
from asrbench.engine.vram import get_vram_monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


class HealthResponse(BaseModel):
    status: str
    version: str


class GPUInfo(BaseModel):
    index: int
    name: str
    vram_used_mb: float
    vram_total_mb: float
    vram_free_mb: float


class VRAMResponse(BaseModel):
    gpu_available: bool
    gpus: list[GPUInfo]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — always returns ok."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/vram", response_model=VRAMResponse)
async def vram() -> VRAMResponse:
    """Report GPU VRAM usage. Returns empty list if no GPU is available."""
    monitor = get_vram_monitor()
    snap = monitor.snapshot()
    if not snap.available:
        return VRAMResponse(gpu_available=False, gpus=[])

    name = "GPU 0"
    try:
        raw = monitor._pynvml.nvmlDeviceGetName(monitor._handle)  # type: ignore[union-attr]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        name = str(raw)
    except Exception as exc:
        logger.debug("nvmlDeviceGetName failed: %s", exc)

    return VRAMResponse(
        gpu_available=True,
        gpus=[
            GPUInfo(
                index=0,
                name=name,
                vram_used_mb=snap.used_mb,
                vram_total_mb=snap.total_mb,
                vram_free_mb=snap.free_mb,
            )
        ],
    )
