"""Integration tests for the /datasets API endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.integration.conftest import insert_dataset, insert_model, insert_run

# ---------------------------------------------------------------------------
# List datasets
# ---------------------------------------------------------------------------


class TestListDatasets:
    def test_empty_returns_list(self, app_client: TestClient) -> None:
        resp = app_client.get("/datasets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_inserted_dataset(self, app_client: TestClient) -> None:
        dataset_id = insert_dataset(lang="en")
        resp = app_client.get("/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["dataset_id"] == dataset_id
        assert data[0]["lang"] == "en"
        assert data[0]["verified"] is True

    def test_filter_by_lang(self, app_client: TestClient) -> None:
        insert_dataset(lang="en")
        insert_dataset(lang="tr")

        resp = app_client.get("/datasets?lang=tr")
        datasets = resp.json()
        assert len(datasets) == 1
        assert datasets[0]["lang"] == "tr"

    def test_filter_by_source(self, app_client: TestClient) -> None:
        insert_dataset(source="custom", lang="en")
        insert_dataset(source="fleurs", lang="en")

        resp = app_client.get("/datasets?source=fleurs")
        datasets = resp.json()
        assert len(datasets) == 1
        assert datasets[0]["source"] == "fleurs"

    def test_filter_combined(self, app_client: TestClient) -> None:
        insert_dataset(source="custom", lang="en")
        insert_dataset(source="custom", lang="tr")
        insert_dataset(source="fleurs", lang="tr")

        resp = app_client.get("/datasets?lang=tr&source=custom")
        datasets = resp.json()
        assert len(datasets) == 1
        assert datasets[0]["lang"] == "tr"
        assert datasets[0]["source"] == "custom"


# ---------------------------------------------------------------------------
# Fetch dataset
# ---------------------------------------------------------------------------


class TestFetchDataset:
    def test_invalid_source_returns_400(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/datasets/fetch",
            json={"source": "unsupported_source", "lang": "en", "split": "test"},
        )
        assert resp.status_code == 400
        assert "not supported" in resp.json()["detail"]

    def test_duplicate_dataset_returns_409(self, app_client: TestClient) -> None:
        # Insert a dataset with the name that would be created by the fetch request
        insert_dataset(source="custom", lang="en")

        resp = app_client.post(
            "/datasets/fetch",
            json={"source": "custom", "lang": "en", "split": "test"},
        )
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_valid_fetch_returns_202(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/datasets/fetch",
            json={"source": "common_voice", "lang": "de", "split": "test"},
        )
        # Background download is triggered; response is immediate
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "downloading"
        assert body["name"] == "common_voice_de_test"
        assert "/ws/logs" in body["stream_url"]

    def test_all_valid_sources_accepted(self, app_client: TestClient) -> None:
        valid_sources = ["common_voice", "fleurs", "yodas", "ted_lium", "custom"]
        for i, source in enumerate(valid_sources):
            resp = app_client.post(
                "/datasets/fetch",
                json={"source": source, "lang": f"x{i}", "split": "test"},
            )
            # Should not return 400 (might return 202 or 409 if duplicate)
            assert resp.status_code != 400, f"Source {source!r} incorrectly rejected"


# ---------------------------------------------------------------------------
# Delete dataset
# ---------------------------------------------------------------------------


class TestDeleteDataset:
    def test_delete_nonexistent_dataset_returns_404(self, app_client: TestClient) -> None:
        resp = app_client.delete("/datasets/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_delete_dataset_with_runs_returns_409(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        insert_run(model_id, dataset_id)

        resp = app_client.delete(f"/datasets/{dataset_id}")
        assert resp.status_code == 409
        assert "run" in resp.json()["detail"].lower()

    def test_delete_dataset_no_runs(self, app_client: TestClient) -> None:
        dataset_id = insert_dataset()

        resp = app_client.delete(f"/datasets/{dataset_id}")
        assert resp.status_code == 204

        assert app_client.get("/datasets").json() == []

    def test_delete_with_delete_files_flag_inside_whitelist(
        self, app_client: TestClient, tmp_path: Path
    ) -> None:
        """delete_files=true with a path under the whitelist must succeed."""
        allowed = str(tmp_path / ".asrbench" / "cache" / "custom-ds")
        dataset_id = insert_dataset(local_path=allowed)

        resp = app_client.delete(f"/datasets/{dataset_id}?delete_files=true")
        # Succeeds even if the cache dir does not exist — we just unregister.
        assert resp.status_code == 204, resp.text

    def test_delete_with_delete_files_flag_outside_whitelist_rejected(
        self, app_client: TestClient
    ) -> None:
        """A dataset row pointing outside the whitelist must not be unlinked.

        The row itself can still be deleted without ``delete_files=true``.
        """
        dataset_id = insert_dataset(local_path="/nonexistent/path")

        resp = app_client.delete(f"/datasets/{dataset_id}?delete_files=true")
        assert resp.status_code == 400
        assert "allowed roots" in resp.json()["detail"].lower()

        # Unregistering without delete_files must still succeed.
        resp2 = app_client.delete(f"/datasets/{dataset_id}")
        assert resp2.status_code == 204
