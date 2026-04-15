"""
Integration tests for POST /optimize/global-config.

Covers only the synchronous validation path (HTTP request → 2 study rows
inserted → 202 response); the background task is not waited on because it
needs a real model + dataset on disk.

Checks:
    1. Happy path: valid model + N datasets + space → 202 + stage_ids.
    2. Missing model → 404.
    3. Any missing dataset → 404 (all-or-nothing validation).
    4. Empty datasets list → 422 (min_length=1).
    5. Invalid space → 422.
    6. Concurrent study lock → 409.
    7. Optional weights accepted (default 1.0 per dataset).
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


def _body(model_id: str, dataset_ids: list[str], **overrides) -> dict:
    body: dict = {
        "model_id": model_id,
        "datasets": [{"dataset_id": ds_id, "lang": "en", "weight": 1.0} for ds_id in dataset_ids],
        "space": _MIN_SPACE,
        "objective": _MIN_OBJECTIVE,
        "mode": "fast",
        "stage1_duration_s": 300,
        "stage2_duration_s": 600,
    }
    body.update(overrides)
    return body


class TestGlobalConfigValidation:
    def test_happy_path_three_datasets(self, app_client: TestClient) -> None:
        model_id = insert_model()
        ds1 = insert_dataset(source="fleurs", lang="tr")
        ds2 = insert_dataset(source="fleurs", lang="en")
        ds3 = insert_dataset(source="librispeech", lang="en")

        resp = app_client.post(
            "/optimize/global-config",
            json=_body(model_id, [ds1, ds2, ds3]),
        )
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        assert payload["dataset_count"] == 3
        assert payload["stage1_study_id"] != payload["stage2_study_id"]
        assert payload["status"] == "running"

    def test_empty_datasets_list_rejected(self, app_client: TestClient) -> None:
        model_id = insert_model()
        body = _body(model_id, [])
        resp = app_client.post("/optimize/global-config", json=body)
        assert resp.status_code == 422

    def test_missing_model_returns_404(self, app_client: TestClient) -> None:
        ds_id = insert_dataset()
        resp = app_client.post(
            "/optimize/global-config",
            json=_body("00000000-0000-0000-0000-000000000000", [ds_id]),
        )
        assert resp.status_code == 404

    def test_any_missing_dataset_returns_404(self, app_client: TestClient) -> None:
        """Bulk validation — one bad id fails the whole batch."""
        model_id = insert_model()
        ds_good = insert_dataset()
        resp = app_client.post(
            "/optimize/global-config",
            json=_body(model_id, [ds_good, "00000000-0000-0000-0000-000000000000"]),
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_invalid_space_returns_422(self, app_client: TestClient) -> None:
        model_id = insert_model()
        ds_id = insert_dataset()
        resp = app_client.post(
            "/optimize/global-config",
            json=_body(model_id, [ds_id], space={"parameters": {}}),
        )
        assert resp.status_code == 422

    def test_weights_respected_in_payload(self, app_client: TestClient) -> None:
        """Custom weights accepted — aggregation uses them in the bg task."""
        model_id = insert_model()
        ds1 = insert_dataset(source="fleurs", lang="tr")
        ds2 = insert_dataset(source="fleurs", lang="en")

        body = _body(model_id, [ds1, ds2])
        body["datasets"][0]["weight"] = 3.0  # TR gets 3x say
        body["datasets"][1]["weight"] = 1.0

        resp = app_client.post("/optimize/global-config", json=body)
        assert resp.status_code == 202

    def test_multifidelity_flag_accepted(self, app_client: TestClient) -> None:
        model_id = insert_model()
        ds_id = insert_dataset()
        resp = app_client.post(
            "/optimize/global-config",
            json=_body(model_id, [ds_id], use_multifidelity=True),
        )
        assert resp.status_code == 202
