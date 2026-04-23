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

__all__ = ["VRAMMonitor", "VRAMSnapshot", "get_vram_monitor"]


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
