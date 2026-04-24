"""Integration tests for /system/* endpoints (Faz 9)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestSystemHealth:
    def test_health_returns_ok_and_version(self, app_client: TestClient) -> None:
        from asrbench import __version__

        resp = app_client.get("/system/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__


class TestSystemVram:
    def test_vram_empty_when_unavailable(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No NVML → gpu_available=false, gpus=[]."""
        import asrbench.api.system as system_mod
        from asrbench.engine.vram import VRAMMonitor, VRAMSnapshot

        fake = VRAMMonitor()
        fake.snapshot = lambda: VRAMSnapshot(  # type: ignore[method-assign]
            available=False, used_mb=0, total_mb=0
        )
        monkeypatch.setattr(system_mod, "get_vram_monitor", lambda: fake)

        resp = app_client.get("/system/vram")
        assert resp.status_code == 200
        body = resp.json()
        assert body["gpu_available"] is False
        assert body["gpus"] == []

    def test_vram_reports_snapshot(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asrbench.api.system as system_mod
        from asrbench.engine.vram import VRAMMonitor, VRAMSnapshot

        fake = VRAMMonitor()
        fake.snapshot = lambda: VRAMSnapshot(  # type: ignore[method-assign]
            available=True, used_mb=4000, total_mb=16_000
        )
        # monitor._pynvml is None in this fake so the name-lookup branch
        # falls through and we get the default "GPU 0".
        monkeypatch.setattr(system_mod, "get_vram_monitor", lambda: fake)

        resp = app_client.get("/system/vram")
        assert resp.status_code == 200
        body = resp.json()
        assert body["gpu_available"] is True
        assert len(body["gpus"]) == 1
        gpu = body["gpus"][0]
        assert gpu["vram_used_mb"] == 4000
        assert gpu["vram_total_mb"] == 16_000
        assert gpu["vram_free_mb"] == 12_000
