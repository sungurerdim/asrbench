"""Unit tests for VRAMMonitor — graceful fallback behavior.

Skipped: VRAMMonitor / VRAMSnapshot were replaced by the lightweight pynvml
wrapper in ``engine/vram.py``; the class-based API no longer exists. Tests
retained as historical reference until the successor coverage is audited.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch  # noqa: F401

import pytest

pytest.skip(
    "VRAMMonitor class was removed; see module docstring.",
    allow_module_level=True,
)

from asrbench.engine.vram import (  # type: ignore[attr-defined]  # noqa: E402
    VRAMMonitor,
    VRAMSnapshot,
)


class TestVRAMMonitorFallback:
    def test_snapshot_returns_unavailable_when_no_gpu(self) -> None:
        """When pynvml init fails, snapshot() returns available=False."""
        with patch("builtins.__import__", side_effect=ImportError("No pynvml")):
            monitor = VRAMMonitor()

        snap = monitor.snapshot()
        assert snap.available is False
        assert snap.used_mb == 0.0
        assert snap.total_mb == 0.0
        assert snap.pct == 0.0

    def test_snapshot_never_raises(self) -> None:
        """snapshot() must never raise regardless of GPU state."""
        monitor = VRAMMonitor()
        # Force handle to None to simulate no GPU
        monitor._handle = None
        snap = monitor.snapshot()
        assert isinstance(snap, VRAMSnapshot)

    def test_warn_threshold_false_when_unavailable(self) -> None:
        """warn_threshold_pct() returns False when GPU is unavailable."""
        monitor = VRAMMonitor()
        monitor._handle = None
        assert monitor.warn_threshold_pct(threshold=0.0) is False

    def test_warn_threshold_with_mock_gpu(self) -> None:
        """warn_threshold_pct() returns True when usage exceeds threshold."""
        monitor = VRAMMonitor()

        mock_pynvml = MagicMock()
        mock_info = MagicMock()
        mock_info.used = 9 * 1024**3  # 9 GB used
        mock_info.total = 10 * 1024**3  # 10 GB total
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info
        monitor._handle = MagicMock()
        monitor._pynvml = mock_pynvml

        assert monitor.warn_threshold_pct(threshold=85.0) is True
        assert monitor.warn_threshold_pct(threshold=95.0) is False

    def test_snapshot_with_mock_gpu(self) -> None:
        """snapshot() returns correct values when GPU is available."""
        monitor = VRAMMonitor()

        mock_pynvml = MagicMock()
        mock_info = MagicMock()
        mock_info.used = 2 * 1024**3  # 2 GB
        mock_info.total = 8 * 1024**3  # 8 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info
        monitor._handle = MagicMock()
        monitor._pynvml = mock_pynvml

        snap = monitor.snapshot()
        assert snap.available is True
        assert snap.used_mb == pytest.approx(2048.0)
        assert snap.total_mb == pytest.approx(8192.0)
        assert snap.pct == pytest.approx(25.0)
