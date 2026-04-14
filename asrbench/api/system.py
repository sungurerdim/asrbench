"""System health and VRAM monitoring endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

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
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/vram", response_model=VRAMResponse)
async def vram() -> VRAMResponse:
    """Report GPU VRAM usage. Returns empty list if no GPU is available."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[GPUInfo] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    vram_used_mb=float(mem.used) / 1024 / 1024,
                    vram_total_mb=float(mem.total) / 1024 / 1024,
                    vram_free_mb=float(mem.free) / 1024 / 1024,
                )
            )
        pynvml.nvmlShutdown()
        return VRAMResponse(gpu_available=True, gpus=gpus)
    except Exception as exc:
        logger.debug("VRAM query failed (no GPU?): %s", exc)
        return VRAMResponse(gpu_available=False, gpus=[])
