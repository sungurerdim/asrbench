"""Integration tests for /models endpoints (Faz 9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _allowed_path(tmp_path: Path) -> str:
    p = tmp_path / ".asrbench" / "cache" / "models" / "tiny"
    return str(p)


class TestRegisterModel:
    def test_register_then_listed(self, app_client: TestClient, tmp_path: Path) -> None:
        resp = app_client.post(
            "/models",
            json={
                "family": "whisper",
                "name": "tiny",
                "backend": "faster-whisper",
                "local_path": _allowed_path(tmp_path),
            },
        )
        assert resp.status_code == 201, resp.text
        model_id = resp.json()["model_id"]

        listing = app_client.get("/models")
        assert listing.status_code == 200
        names = [m["name"] for m in listing.json()]
        assert "tiny" in names
        assert any(m["model_id"] == model_id for m in listing.json())

    def test_register_is_idempotent(self, app_client: TestClient, tmp_path: Path) -> None:
        payload = {
            "family": "whisper",
            "name": "tiny",
            "backend": "faster-whisper",
            "local_path": _allowed_path(tmp_path),
        }
        first = app_client.post("/models", json=payload).json()
        second = app_client.post("/models", json=payload).json()
        assert first["model_id"] == second["model_id"]

    def test_unknown_backend_returns_422(self, app_client: TestClient, tmp_path: Path) -> None:
        resp = app_client.post(
            "/models",
            json={
                "family": "whisper",
                "name": "tiny",
                "backend": "nonexistent-backend-xyz",
                "local_path": _allowed_path(tmp_path),
            },
        )
        assert resp.status_code == 422
        assert "Unknown backend" in resp.json()["detail"]

    def test_path_outside_whitelist_rejected(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/models",
            json={
                "family": "whisper",
                "name": "tiny",
                "backend": "faster-whisper",
                "local_path": "/etc/passwd",
            },
        )
        assert resp.status_code == 422  # pydantic ValidationError → 422


class TestLoadUnloadModel:
    def test_load_unknown_model_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.post("/models/00000000-0000-0000-0000-000000000000/load")
        assert resp.status_code == 404

    def test_load_backend_failure_returns_500(
        self,
        app_client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When backend.load raises, the API must surface a 500."""
        registered = app_client.post(
            "/models",
            json={
                "family": "whisper",
                "name": "tiny-fail",
                "backend": "faster-whisper",
                "local_path": _allowed_path(tmp_path),
            },
        )
        model_id = registered.json()["model_id"]

        class _BrokenBackend:
            family = "whisper"
            name = "faster-whisper"

            def load(self, *_a, **_kw):
                raise RuntimeError("model file missing")

            def unload(self):
                return None

            def default_params(self):
                return {}

            def supported_params(self, *, mode_hint=None):
                return None

        # `load_backends` is imported inside the handler; patch the
        # source module instead of api.models.
        import asrbench.backends as backends_mod

        monkeypatch.setattr(
            backends_mod, "load_backends", lambda: {"faster-whisper": _BrokenBackend}
        )

        resp = app_client.post(f"/models/{model_id}/load")
        assert resp.status_code == 500
        assert "model file missing" in resp.json()["detail"]
