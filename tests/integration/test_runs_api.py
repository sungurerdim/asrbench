"""Integration tests for the /runs API endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import (
    insert_aggregate,
    insert_dataset,
    insert_model,
    insert_run,
)

# ---------------------------------------------------------------------------
# List runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_returns_list(self, app_client: TestClient) -> None:
        resp = app_client.get("/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_inserted_run(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        insert_run(model_id, dataset_id)

        resp = app_client.get("/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model_id"] == model_id
        assert data[0]["status"] == "pending"

    def test_filter_by_status(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        insert_run(model_id, dataset_id, status="pending")
        insert_run(model_id, dataset_id, status="failed")

        resp = app_client.get("/runs?status=failed")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"

    def test_filter_by_lang(self, app_client: TestClient) -> None:
        model_id = insert_model()
        ds_en = insert_dataset(lang="en")
        ds_tr = insert_dataset(lang="tr")
        insert_run(model_id, ds_en, lang="en")
        insert_run(model_id, ds_tr, lang="tr")

        resp = app_client.get("/runs?lang=tr")
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["lang"] == "tr"

    def test_limit_respected(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        for _ in range(5):
            insert_run(model_id, dataset_id)

        resp = app_client.get("/runs?limit=3")
        assert len(resp.json()) == 3


# ---------------------------------------------------------------------------
# Get single run
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_get_existing_run(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id)

        resp = app_client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["model_id"] == model_id
        assert data["aggregate"] is None

    def test_get_run_with_aggregate(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="completed")
        insert_aggregate(run_id)

        resp = app_client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        agg = resp.json()["aggregate"]
        assert agg is not None
        assert pytest.approx(agg["wer_mean"], abs=1e-6) == 0.15
        assert pytest.approx(agg["rtfx_mean"], abs=1e-6) == 5.3

    def test_get_nonexistent_run_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.get("/runs/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete run
# ---------------------------------------------------------------------------


class TestDeleteRun:
    def test_delete_pending_run(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="pending")

        resp = app_client.delete(f"/runs/{run_id}")
        assert resp.status_code == 204

        assert app_client.get(f"/runs/{run_id}").status_code == 404

    def test_delete_running_run_rejected(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="running")

        resp = app_client.delete(f"/runs/{run_id}")
        assert resp.status_code == 409

    def test_delete_nonexistent_run_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.delete("/runs/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_delete_cascades_segments(self, app_client: TestClient) -> None:
        """Segments belonging to a deleted run must also be removed."""
        from asrbench.db import get_conn

        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="completed")

        # Insert a segment directly
        get_conn().cursor().execute(
            "INSERT INTO segments (run_id, offset_s, duration_s, ref_text, hyp_text) "
            "VALUES (?, 0.0, 5.0, 'hello', 'hello')",
            [run_id],
        )

        app_client.delete(f"/runs/{run_id}")

        remaining = (
            get_conn()
            .cursor()
            .execute("SELECT COUNT(*) FROM segments WHERE run_id = ?", [run_id])
            .fetchone()
        )
        assert remaining is not None
        assert remaining[0] == 0


# ---------------------------------------------------------------------------
# Cancel run
# ---------------------------------------------------------------------------


class TestCancelRun:
    def test_cancel_nonexistent_run_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.post("/runs/00000000-0000-0000-0000-000000000000/cancel")
        assert resp.status_code == 404

    def test_cancel_non_running_run_returns_409(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="pending")

        resp = app_client.post(f"/runs/{run_id}/cancel")
        assert resp.status_code == 409
        assert "not running" in resp.json()["detail"]

    def test_cancel_running_run(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="running")

        resp = app_client.post(f"/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancellation_requested"


# ---------------------------------------------------------------------------
# Retry run
# ---------------------------------------------------------------------------


class TestRetryRun:
    def test_retry_pending_run_rejected(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="pending")

        resp = app_client.post(f"/runs/{run_id}/retry")
        assert resp.status_code == 409

    def test_retry_failed_run_creates_new_run(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="failed")

        resp = app_client.post(f"/runs/{run_id}/retry")
        assert resp.status_code == 202
        body = resp.json()
        assert body["original_run_id"] == run_id
        assert body["new_run_id"] != run_id

    def test_retry_cancelled_run_creates_new_run(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="cancelled")

        resp = app_client.post(f"/runs/{run_id}/retry")
        assert resp.status_code == 202
        assert resp.json()["original_run_id"] == run_id

    def test_retry_nonexistent_run_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.post("/runs/00000000-0000-0000-0000-000000000000/retry")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


class TestGetSegments:
    def test_empty_segments(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id)

        resp = app_client.get(f"/runs/{run_id}/segments")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_pagination(self, app_client: TestClient) -> None:
        from asrbench.db import get_conn

        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="completed")

        # Insert 5 segments
        cur = get_conn().cursor()
        for i in range(5):
            cur.execute(
                "INSERT INTO segments (run_id, offset_s, duration_s, ref_text, hyp_text) "
                "VALUES (?, ?, 2.0, 'hello world', 'hello world')",
                [run_id, float(i * 2)],
            )

        page1 = app_client.get(f"/runs/{run_id}/segments?page=1&page_size=3").json()
        page2 = app_client.get(f"/runs/{run_id}/segments?page=2&page_size=3").json()
        assert len(page1) == 3
        assert len(page2) == 2

    def test_segments_not_found_run_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.get("/runs/00000000-0000-0000-0000-000000000000/segments")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Start run (validates model/dataset existence)
# ---------------------------------------------------------------------------


class TestStartRun:
    def test_start_run_model_not_found(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/runs/start",
            json={
                "mode": "model_compare",
                "model_id": "00000000-0000-0000-0000-000000000000",
                "dataset_id": "00000000-0000-0000-0000-000000000001",
                "lang": "en",
            },
        )
        assert resp.status_code == 404
        assert "model" in resp.json()["detail"].lower()

    def test_start_run_dataset_not_found(self, app_client: TestClient) -> None:
        model_id = insert_model()
        resp = app_client.post(
            "/runs/start",
            json={
                "mode": "model_compare",
                "model_id": model_id,
                "dataset_id": "00000000-0000-0000-0000-000000000000",
                "lang": "en",
            },
        )
        assert resp.status_code == 404
        assert "dataset" in resp.json()["detail"].lower()

    def test_start_run_blocks_when_run_is_running(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        insert_run(model_id, dataset_id, status="running")

        resp = app_client.post(
            "/runs/start",
            json={
                "mode": "model_compare",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "lang": "en",
            },
        )
        assert resp.status_code == 409

    def test_start_run_invalid_matrix_raises_422(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()

        resp = app_client.post(
            "/runs/start",
            json={
                "mode": "param_compare",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "lang": "en",
                "matrix": {"beam_size": []},  # empty list → ValueError
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Compare runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_compare_two_completed_runs(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id_a = insert_run(model_id, dataset_id, status="completed")
        run_id_b = insert_run(model_id, dataset_id, status="completed")
        insert_aggregate(run_id_a)
        insert_aggregate(run_id_b)

        resp = app_client.get(f"/runs/compare?ids={run_id_a},{run_id_b}")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["runs"]) == 2
        assert "params_diff" in body
        assert "params_same" in body

    def test_compare_missing_run_returns_400(self, app_client: TestClient) -> None:
        resp = app_client.get(
            "/runs/compare?ids=00000000-0000-0000-0000-000000000000,"
            "00000000-0000-0000-0000-000000000001"
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Export run
# ---------------------------------------------------------------------------


class TestExportRun:
    def test_export_json(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="completed")

        resp = app_client.get(f"/runs/{run_id}/export?fmt=json")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        payload = json.loads(resp.content)
        assert payload["run_id"] == run_id
        assert "segments" in payload

    def test_export_csv(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="completed")

        resp = app_client.get(f"/runs/{run_id}/export?fmt=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "segment_id" in resp.text

    def test_export_invalid_format_returns_400(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id)

        resp = app_client.get(f"/runs/{run_id}/export?fmt=xlsx")
        assert resp.status_code == 400

    def test_export_nonexistent_run_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.get("/runs/00000000-0000-0000-0000-000000000000/export?fmt=json")
        assert resp.status_code == 404
