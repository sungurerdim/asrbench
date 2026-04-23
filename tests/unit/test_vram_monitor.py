"""Unit tests for VRAMMonitor — graceful fallback and peak tracking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from asrbench.engine.vram import VRAMMonitor, VRAMSnapshot, get_vram_monitor


def _force_unavailable(monitor: VRAMMonitor) -> None:
    """Bypass pynvml lookup so snapshot() returns the unavailable placeholder."""
    monitor._init_attempted = True
    monitor._pynvml = None
    monitor._handle = None


def _force_mock(monitor: VRAMMonitor, used_bytes: int, total_bytes: int) -> None:
    """Inject a fake pynvml + handle reporting the given memory numbers."""
    mock_pynvml = MagicMock()
    mock_info = MagicMock()
    mock_info.used = used_bytes
    mock_info.total = total_bytes
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info
    monitor._init_attempted = True
    monitor._pynvml = mock_pynvml
    monitor._handle = MagicMock()


class TestFallback:
    def test_snapshot_returns_unavailable_when_no_gpu(self) -> None:
        monitor = VRAMMonitor()
        _force_unavailable(monitor)

        snap = monitor.snapshot()
        assert snap.available is False
        assert snap.used_mb == 0.0
        assert snap.total_mb == 0.0
        assert snap.pct == 0.0
        assert snap.free_mb == 0.0

    def test_snapshot_never_raises_when_nvml_errors(self) -> None:
        """A transient NVML failure returns the unavailable snapshot, not an exception."""
        monitor = VRAMMonitor()
        _force_mock(monitor, used_bytes=0, total_bytes=0)
        monitor._pynvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("nvml gone")

        snap = monitor.snapshot()
        assert isinstance(snap, VRAMSnapshot)
        assert snap.available is False

    def test_warn_threshold_false_when_unavailable(self) -> None:
        monitor = VRAMMonitor()
        _force_unavailable(monitor)
        assert monitor.warn_threshold_pct(threshold=0.0) is False


class TestMockGPU:
    def test_snapshot_reports_used_and_total(self) -> None:
        monitor = VRAMMonitor()
        _force_mock(monitor, used_bytes=2 * 1024**3, total_bytes=8 * 1024**3)

        snap = monitor.snapshot()
        assert snap.available is True
        assert snap.used_mb == pytest.approx(2048.0)
        assert snap.total_mb == pytest.approx(8192.0)
        assert snap.pct == pytest.approx(25.0)
        assert snap.free_mb == pytest.approx(6144.0)

    def test_warn_threshold_compares_pct(self) -> None:
        monitor = VRAMMonitor()
        _force_mock(monitor, used_bytes=9 * 1024**3, total_bytes=10 * 1024**3)

        assert monitor.warn_threshold_pct(threshold=85.0) is True
        assert monitor.warn_threshold_pct(threshold=95.0) is False


class TestPeakTracking:
    def test_peak_reflects_max_observed_usage(self) -> None:
        monitor = VRAMMonitor()
        _force_mock(monitor, used_bytes=1 * 1024**3, total_bytes=8 * 1024**3)
        monitor.snapshot()

        # Simulate a spike
        monitor._pynvml.nvmlDeviceGetMemoryInfo.return_value.used = 4 * 1024**3
        monitor.snapshot()

        # Then a drop — peak should stick at the spike
        monitor._pynvml.nvmlDeviceGetMemoryInfo.return_value.used = 2 * 1024**3
        monitor.snapshot()

        assert monitor.peak_mb == pytest.approx(4096.0)

    def test_reset_peak_clears_counter(self) -> None:
        monitor = VRAMMonitor()
        _force_mock(monitor, used_bytes=3 * 1024**3, total_bytes=8 * 1024**3)
        monitor.snapshot()
        assert monitor.peak_mb > 0

        monitor.reset_peak()
        assert monitor.peak_mb == 0.0


class TestSingleton:
    def test_get_vram_monitor_returns_same_instance(self) -> None:
        a = get_vram_monitor()
        b = get_vram_monitor()
        assert a is b
