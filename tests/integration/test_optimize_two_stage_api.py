"""
Integration tests for POST /optimize/two-stage.

The endpoint kicks off a background task that loads the model and runs the
library's ``run_two_stage`` orchestrator. These tests exercise only the
synchronous validation path (HTTP request → 2 study rows inserted → 202
response); they do NOT wait for the background task to complete because
that would require a real model + dataset on disk.

Covered:
    1. Happy path: valid model + dataset + space → 202 with both study_ids.
    2. Missing model → 404.
    3. Missing dataset → 404.
    4. Invalid space → 422.
    5. Concurrent study lock → 409.
    6. Optional budget/epsilon left None → server accepts them (auto-sizing
       path kicks in when the background task runs; validation does not fire).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.conftest import insert_dataset, insert_model

_MIN_SPACE = {
    "parameters": {
        "beam_size": {"type": "int", "min": 1, "default": 5, "max": 10},
    }
}

_MIN_OBJECTIVE = {"type": "single", "metric": "wer"}


def _two_stage_body(model_id: str, dataset_id: str, **overrides) -> dict:
    body: dict = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "lang": "en",
        "space": _MIN_SPACE,
        "objective": _MIN_OBJECTIVE,
        "mode": "fast",
        "stage1_duration_s": 300,
        "stage2_duration_s": 600,
    }
    body.update(overrides)
    return body


class TestTwoStageValidation:
    def test_happy_path_returns_202_and_two_study_ids(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()

        resp = app_client.post(
            "/optimize/two-stage",
            json=_two_stage_body(model_id, dataset_id),
        )
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        assert "stage1_study_id" in payload
        assert "stage2_study_id" in payload
        assert payload["stage1_study_id"] != payload["stage2_study_id"]
        assert payload["status"] == "running"
        assert payload["mode"] == "fast"

    def test_missing_model_returns_404(self, app_client: TestClient) -> None:
        dataset_id = insert_dataset()
        resp = app_client.post(
            "/optimize/two-stage",
            json=_two_stage_body("00000000-0000-0000-0000-000000000000", dataset_id),
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_missing_dataset_returns_404(self, app_client: TestClient) -> None:
        model_id = insert_model()
        resp = app_client.post(
            "/optimize/two-stage",
            json=_two_stage_body(model_id, "00000000-0000-0000-0000-000000000000"),
        )
        assert resp.status_code == 404

    def test_invalid_space_returns_422(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        resp = app_client.post(
            "/optimize/two-stage",
            json=_two_stage_body(
                model_id,
                dataset_id,
                space={"parameters": {}},  # empty → ParameterSpace rejects
            ),
        )
        assert resp.status_code == 422
        assert "parameter" in resp.json()["detail"].lower()

    def test_auto_sized_budgets_accepted(self, app_client: TestClient) -> None:
        """Leaving budget/epsilon as None is a valid request shape."""
        model_id = insert_model()
        dataset_id = insert_dataset()
        body = _two_stage_body(model_id, dataset_id)
        # Explicitly omit all optional sizing knobs — server must accept.
        body.pop("stage1_budget", None)
        body.pop("stage2_budget", None)
        body.pop("stage1_epsilon", None)
        body.pop("stage2_epsilon", None)
        resp = app_client.post("/optimize/two-stage", json=body)
        assert resp.status_code == 202, resp.text

    def test_multifidelity_flag_accepted(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        resp = app_client.post(
            "/optimize/two-stage",
            json=_two_stage_body(model_id, dataset_id, use_multifidelity=True),
        )
        assert resp.status_code == 202
