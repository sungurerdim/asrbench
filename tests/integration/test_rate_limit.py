"""Integration tests for the tightened rate-limit exemption list.

Faz 1.1 narrowed exemptions from "any path under /runs/" to specifically
GET polling endpoints. Mutating POSTs (start, register, fetch) must now
be rate-limited so a single misbehaving UI cannot flood expensive work.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from asrbench.middleware.rate_limit import RateLimitMiddleware


def _build_app(rpm: int = 2, burst: int = 2) -> FastAPI:
    """Minimal FastAPI app wrapping RateLimitMiddleware for tight-loop tests."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm, burst=burst)

    @app.get("/runs/status/{run_id}")
    async def runs_status(run_id: str) -> dict:
        return {"ok": True}

    @app.post("/runs/start")
    async def runs_start() -> dict:
        return {"ok": True}

    @app.get("/optimize/{study_id}")
    async def optimize_get(study_id: str) -> dict:
        return {"ok": True}

    @app.post("/optimize/start")
    async def optimize_start() -> dict:
        return {"ok": True}

    @app.get("/system/health")
    async def system_health() -> dict:
        return {"status": "ok"}

    @app.post("/datasets/fetch")
    async def datasets_fetch() -> dict:
        return {"ok": True}

    @app.post("/models/register")
    async def models_register() -> dict:
        return {"ok": True}

    return app


def test_get_runs_polling_is_exempt() -> None:
    """GET /runs/<id> must survive far more requests than the bucket holds."""
    client = TestClient(_build_app(rpm=2, burst=2))
    statuses = [client.get("/runs/status/abc").status_code for _ in range(50)]
    assert all(s == 200 for s in statuses), f"unexpected 429 in polling path: {statuses}"


def test_get_optimize_polling_is_exempt() -> None:
    client = TestClient(_build_app(rpm=2, burst=2))
    statuses = [client.get("/optimize/study-1").status_code for _ in range(50)]
    assert all(s == 200 for s in statuses)


def test_system_health_is_exempt() -> None:
    client = TestClient(_build_app(rpm=2, burst=2))
    statuses = [client.get("/system/health").status_code for _ in range(50)]
    assert all(s == 200 for s in statuses)


def test_post_runs_start_is_rate_limited() -> None:
    """POST /runs/start must hit 429 once the burst is exhausted."""
    client = TestClient(_build_app(rpm=2, burst=2))
    statuses = [client.post("/runs/start").status_code for _ in range(6)]
    assert 429 in statuses, f"POST /runs/start never throttled: {statuses}"


def test_post_optimize_start_is_rate_limited() -> None:
    client = TestClient(_build_app(rpm=2, burst=2))
    statuses = [client.post("/optimize/start").status_code for _ in range(6)]
    assert 429 in statuses


def test_post_datasets_fetch_is_rate_limited() -> None:
    client = TestClient(_build_app(rpm=2, burst=2))
    statuses = [client.post("/datasets/fetch").status_code for _ in range(6)]
    assert 429 in statuses


def test_post_models_register_is_rate_limited() -> None:
    client = TestClient(_build_app(rpm=2, burst=2))
    statuses = [client.post("/models/register").status_code for _ in range(6)]
    assert 429 in statuses


def test_throttle_response_schema() -> None:
    """The 429 body must include a human-readable retry hint."""
    client = TestClient(_build_app(rpm=2, burst=2))
    # Burn the bucket
    for _ in range(5):
        client.post("/runs/start")
    resp = client.post("/runs/start")
    assert resp.status_code == 429
    body = resp.json()
    assert "detail" in body
    assert "Rate limit" in body["detail"]
