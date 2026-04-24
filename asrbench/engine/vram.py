"""VRAM monitoring wrapper around pynvml with graceful fallback.

Uses a single lazily-initialized process-wide handle so repeated calls (from
API endpoints, the benchmark loop, and the live /ws/vram stream) reuse the
same NVML context instead of calling ``nvmlInit``/``nvmlShutdown`` per query.

When pynvml is not installed or no GPU is visible, all methods return
unavailable-shaped snapshots and never raise — callers can treat VRAM as a
best-effort signal rather than a hard dependency.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ResourceExhausted",
    "VRAMMonitor",
    "VRAMSnapshot",
    "get_vram_monitor",
]


class ResourceExhausted(RuntimeError):
    """Raised before a backend load that cannot fit in free VRAM.

    The benchmark engine catches this in place of a generic RuntimeError
    so the failing run is marked with a precise error message instead of
    a mid-load CUDA OOM that would leave the process in an uncertain
    state.
    """


@dataclass(frozen=True)
class VRAMSnapshot:
    """A single VRAM reading for one GPU, or an unavailable placeholder."""

    available: bool
    used_mb: float
    total_mb: float

    @property
    def pct(self) -> float:
        if not self.available or self.total_mb <= 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100.0

    @property
    def free_mb(self) -> float:
        if not self.available:
            return 0.0
        return max(0.0, self.total_mb - self.used_mb)


_UNAVAILABLE = VRAMSnapshot(available=False, used_mb=0.0, total_mb=0.0)


class VRAMMonitor:
    """Lightweight wrapper around pynvml with peak tracking for benchmark runs.

    The monitor holds a single NVML handle after first successful init and
    reuses it for every ``snapshot()``. ``peak_mb`` tracks the max ``used_mb``
    observed since the last ``reset_peak()`` — BenchmarkEngine calls
    ``reset_peak()`` at run start and ``peak_mb`` at run end to write the
    ``aggregates.vram_peak_mb`` column.
    """

    def __init__(self) -> None:
        self._pynvml: Any = None
        self._handle: Any = None
        self._init_attempted = False
        self._lock = threading.Lock()
        self._peak_mb = 0.0

    def _ensure_handle(self) -> None:
        """Initialize pynvml exactly once per process. Safe to call repeatedly."""
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            logger.debug("VRAMMonitor initialized with NVML handle")
        except Exception as exc:
            logger.debug("VRAMMonitor: NVML unavailable (%s) — falling back", exc)
            self._pynvml = None
            self._handle = None

    def snapshot(self) -> VRAMSnapshot:
        """Return the current VRAM reading. Never raises."""
        self._ensure_handle()
        if self._handle is None or self._pynvml is None:
            return _UNAVAILABLE
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_mb = float(info.used) / 1024 / 1024
            total_mb = float(info.total) / 1024 / 1024
            snap = VRAMSnapshot(available=True, used_mb=used_mb, total_mb=total_mb)
            with self._lock:
                if used_mb > self._peak_mb:
                    self._peak_mb = used_mb
            return snap
        except Exception as exc:
            logger.debug("VRAMMonitor.snapshot failed: %s", exc)
            return _UNAVAILABLE

    def warn_threshold_pct(self, threshold: float) -> bool:
        """Return True if the current GPU usage meets or exceeds ``threshold`` percent."""
        snap = self.snapshot()
        if not snap.available:
            return False
        return snap.pct >= threshold

    def can_accommodate(
        self,
        required_mb: float,
        *,
        safety_margin_pct: float = 10.0,
    ) -> bool:
        """Return True when the GPU has enough free VRAM to load a model.

        ``required_mb`` is the caller's best-effort estimate of the model
        memory (backend-specific — e.g. whisper's param table × bytes per
        element). ``safety_margin_pct`` defaults to 10 %, giving the
        backend's own working buffers room to breathe before triggering
        a real OOM.

        When NVML is unavailable (CPU-only or no NVIDIA driver), returns
        True so the CPU path still runs. A caller that wants a hard
        guarantee should also check ``snapshot().available``.
        """
        if required_mb <= 0:
            return True
        snap = self.snapshot()
        if not snap.available:
            return True
        margin = required_mb * (safety_margin_pct / 100.0)
        return snap.free_mb >= (required_mb + margin)

    def require_capacity(
        self,
        required_mb: float,
        *,
        model_label: str = "model",
        safety_margin_pct: float = 10.0,
    ) -> None:
        """Raise :class:`ResourceExhausted` when ``can_accommodate`` says no.

        Gives the caller a one-line guard: ``monitor.require_capacity(...)``
        replaces a manual snapshot-then-compare block. The exception
        message includes the free/required/margin numbers so the study
        row's ``error_message`` carries a usable diagnostic.
        """
        if required_mb <= 0:
            return
        snap = self.snapshot()
        if not snap.available:
            return
        margin = required_mb * (safety_margin_pct / 100.0)
        needed = required_mb + margin
        if snap.free_mb < needed:
            raise ResourceExhausted(
                f"Not enough VRAM to load {model_label}: "
                f"need ~{needed:.0f} MB (estimate {required_mb:.0f} MB + "
                f"{safety_margin_pct:.0f}% safety margin), "
                f"have {snap.free_mb:.0f} MB free of {snap.total_mb:.0f} MB."
            )

    def reset_peak(self) -> None:
        """Clear the peak counter. Call at the start of a benchmark run."""
        with self._lock:
            self._peak_mb = 0.0

    @property
    def peak_mb(self) -> float:
        """Peak ``used_mb`` observed since the last ``reset_peak()``."""
        with self._lock:
            return self._peak_mb


_monitor: VRAMMonitor | None = None
_monitor_lock = threading.Lock()


def get_vram_monitor() -> VRAMMonitor:
    """Return the process-wide VRAMMonitor singleton."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = VRAMMonitor()
    return _monitor
