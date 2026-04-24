"""Integration tests for the API key auth middleware (Faz 1.2)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from asrbench.middleware.auth import AuthMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    return app


def test_loopback_passes_without_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """TestClient reports as ``testclient``, treated as loopback."""
    monkeypatch.delenv("ASRBENCH_API_KEY", raising=False)
    client = TestClient(_build_app())
    resp = client.get("/ping")
    assert resp.status_code == 200


def test_non_loopback_without_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ASRBENCH_API_KEY is unset, any remote client gets 401.

    The plain TestClient has loopback_hosts-friendly address, so we patch
    the loopback set to force the auth check to run.
    """
    import asrbench.middleware.auth as auth_mod

    monkeypatch.setattr(auth_mod, "LOOPBACK_HOSTS", frozenset({"127.0.0.1"}))
    monkeypatch.delenv("ASRBENCH_API_KEY", raising=False)
    client = TestClient(_build_app())
    resp = client.get("/ping")
    assert resp.status_code == 401
    assert "not configured" in resp.json()["detail"].lower()


def test_non_loopback_missing_header_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import asrbench.middleware.auth as auth_mod

    monkeypatch.setattr(auth_mod, "LOOPBACK_HOSTS", frozenset({"127.0.0.1"}))
    monkeypatch.setenv("ASRBENCH_API_KEY", "super-secret-token-value")
    client = TestClient(_build_app())
    resp = client.get("/ping")
    assert resp.status_code == 401
    assert "x-api-key" in resp.json()["detail"].lower()


def test_non_loopback_wrong_header_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import asrbench.middleware.auth as auth_mod

    monkeypatch.setattr(auth_mod, "LOOPBACK_HOSTS", frozenset({"127.0.0.1"}))
    monkeypatch.setenv("ASRBENCH_API_KEY", "super-secret-token-value")
    client = TestClient(_build_app())
    resp = client.get("/ping", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_non_loopback_correct_header_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import asrbench.middleware.auth as auth_mod

    monkeypatch.setattr(auth_mod, "LOOPBACK_HOSTS", frozenset({"127.0.0.1"}))
    monkeypatch.setenv("ASRBENCH_API_KEY", "super-secret-token-value")
    client = TestClient(_build_app())
    resp = client.get("/ping", headers={"X-API-Key": "super-secret-token-value"})
    assert resp.status_code == 200


def test_empty_api_key_env_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leading/trailing whitespace and empty strings must not count as valid."""
    from asrbench.middleware.auth import get_api_key

    monkeypatch.setenv("ASRBENCH_API_KEY", "   ")
    assert get_api_key() is None
    monkeypatch.setenv("ASRBENCH_API_KEY", "")
    assert get_api_key() is None
    monkeypatch.setenv("ASRBENCH_API_KEY", "real-value")
    assert get_api_key() == "real-value"
