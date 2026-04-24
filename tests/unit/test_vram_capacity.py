"""Unit tests for VRAMMonitor capacity helpers (Faz 4.3)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from asrbench.engine.vram import ResourceExhausted, VRAMMonitor, VRAMSnapshot


def _patch_snapshot(monitor: VRAMMonitor, snap: VRAMSnapshot) -> None:
    """Override ``VRAMMonitor.snapshot`` so tests don't depend on real NVML."""
    monitor.snapshot = lambda: snap  # type: ignore[method-assign]


class TestCanAccommodate:
    def test_ok_with_headroom(self) -> None:
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=True, used_mb=2000, total_mb=16000))
        # free_mb = 14000, need 8800 (8000 + 10% margin) → fits
        assert m.can_accommodate(8000) is True

    def test_denies_when_free_below_need(self) -> None:
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=True, used_mb=10000, total_mb=12000))
        # free_mb = 2000, need 5500 → does not fit
        assert m.can_accommodate(5000) is False

    def test_zero_required_is_always_ok(self) -> None:
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=True, used_mb=12000, total_mb=12000))
        assert m.can_accommodate(0) is True

    def test_returns_true_when_nvml_unavailable(self) -> None:
        """CPU-only or driver missing → optimistic path runs."""
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=False, used_mb=0, total_mb=0))
        assert m.can_accommodate(9999) is True

    def test_custom_safety_margin(self) -> None:
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=True, used_mb=0, total_mb=10000))
        # free_mb = 10000, need 10000 exactly with 0% margin
        assert m.can_accommodate(10000, safety_margin_pct=0.0) is True
        # With the default 10% margin, need 11000 → fails
        assert m.can_accommodate(10000) is False


class TestRequireCapacity:
    def test_ok_does_not_raise(self) -> None:
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=True, used_mb=0, total_mb=16000))
        m.require_capacity(8000, model_label="whisper-large-v3")

    def test_fails_with_diagnostic_message(self) -> None:
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=True, used_mb=14000, total_mb=16000))
        with pytest.raises(ResourceExhausted) as excinfo:
            m.require_capacity(4000, model_label="whisper-large-v3")
        msg = str(excinfo.value)
        assert "whisper-large-v3" in msg
        assert "free" in msg.lower()
        assert "4400 MB" in msg  # 4000 + 10% margin

    def test_unavailable_nvml_silently_allows(self) -> None:
        """Missing NVML cannot block CPU runs."""
        m = VRAMMonitor()
        _patch_snapshot(m, VRAMSnapshot(available=False, used_mb=0, total_mb=0))
        m.require_capacity(100_000)


class TestSnapshotHelpers:
    def test_pct_free_accessors(self) -> None:
        snap = VRAMSnapshot(available=True, used_mb=4000, total_mb=16000)
        assert snap.pct == pytest.approx(25.0)
        assert snap.free_mb == pytest.approx(12000)

    def test_unavailable_snapshot_defaults(self) -> None:
        snap = VRAMSnapshot(available=False, used_mb=0, total_mb=0)
        assert snap.pct == 0.0
        assert snap.free_mb == 0.0
        # Replace on a frozen dataclass still works
        other = replace(snap, available=True, used_mb=10, total_mb=20)
        assert other.pct == pytest.approx(50.0)
