"""Integration tests for startup stale-run recovery + graceful shutdown (Faz 3)."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient


def _fresh_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fresh FastAPI app rooted in ``tmp_path``.

    Clears both the config cache and the DB singleton so each call
    lands on a clean lifespan sequence.
    """
    from asrbench.config import get_config
    from asrbench.db import reset

    get_config.cache_clear()
    reset()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    from asrbench.main import create_app

    return create_app()


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / ".asrbench" / "benchmark.db"


def _seed_via_direct_conn(db_file: Path, *, run_status: str) -> tuple[str, str]:
    """Write a run row directly to the DB, bypassing the lifespan shutdown.

    Simulates the state left behind by a hard process kill — the row
    stays in ``status=running`` because no graceful-shutdown hook got a
    chance to downgrade it.
    """
    conn = duckdb.connect(str(db_file))
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO models (family, name, backend, local_path, default_params) "
            "VALUES (?, ?, ?, ?, ?) RETURNING model_id",
            ["whisper", "recovery-test", "faster-whisper", "/tmp/mm", json.dumps({})],
        )
        model_id = str(cur.fetchone()[0])
        cur.execute(
            "INSERT INTO datasets (name, source, lang, split, local_path, verified) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING dataset_id",
            ["recov-ds", "custom", "en", "test", "/tmp/ds", True],
        )
        dataset_id = str(cur.fetchone()[0])
        cur.execute(
            "INSERT INTO runs (model_id, backend, params, dataset_id, lang, mode, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING run_id",
            [
                model_id,
                "faster-whisper",
                json.dumps({}),
                dataset_id,
                "en",
                "model_compare",
                run_status,
            ],
        )
        run_id = str(cur.fetchone()[0])
    finally:
        conn.close()
    return run_id, dataset_id


class TestStartupRecovery:
    """A hard kill must not leave pending/running rows forever in-flight."""

    def test_running_runs_marked_failed_on_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Phase 1: boot once to create the schema, then close cleanly.
        app1 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app1):
            pass  # lifespan runs init_db + clean shutdown

        # Phase 2: simulate a hard kill — write a "running" row with no
        # process alive, then boot a fresh app and observe recovery.
        run_id, _ = _seed_via_direct_conn(_db_path(tmp_path), run_status="running")

        app2 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app2) as client:
            resp = client.get(f"/runs/{run_id}")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["status"] == "failed"

    def test_pending_runs_marked_failed_on_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app1 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app1):
            pass

        run_id, _ = _seed_via_direct_conn(_db_path(tmp_path), run_status="pending")

        app2 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app2) as client:
            resp = client.get(f"/runs/{run_id}")
            assert resp.json()["status"] == "failed"

    def test_pending_studies_marked_failed_on_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app1 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app1):
            pass

        # Seed a stale study directly.
        conn = duckdb.connect(str(_db_path(tmp_path)))
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO models (family, name, backend, local_path, default_params) "
                "VALUES (?, ?, ?, ?, ?) RETURNING model_id",
                ["whisper", "m", "faster-whisper", "/tmp/m", json.dumps({})],
            )
            model_id = str(cur.fetchone()[0])
            cur.execute(
                "INSERT INTO datasets (name, source, lang, split, local_path, verified) "
                "VALUES (?, ?, ?, ?, ?, ?) RETURNING dataset_id",
                ["ds", "custom", "en", "test", "/tmp/ds", True],
            )
            dataset_id = str(cur.fetchone()[0])
            cur.execute(
                "INSERT INTO optimization_studies "
                "(model_id, dataset_id, lang, mode, eps_min, status) "
                "VALUES (?, ?, 'en', 'screening', 0.005, 'pending') "
                "RETURNING study_id",
                [model_id, dataset_id],
            )
            study_id = str(cur.fetchone()[0])
        finally:
            conn.close()

        app2 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app2) as client:
            resp = client.get(f"/optimize/{study_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "failed"

    def test_completed_runs_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app1 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app1):
            pass

        run_completed, _ = _seed_via_direct_conn(_db_path(tmp_path), run_status="completed")
        run_failed, _ = _seed_via_direct_conn(_db_path(tmp_path), run_status="failed")
        run_cancelled, _ = _seed_via_direct_conn(_db_path(tmp_path), run_status="cancelled")

        app2 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app2) as client:
            assert client.get(f"/runs/{run_completed}").json()["status"] == "completed"
            assert client.get(f"/runs/{run_failed}").json()["status"] == "failed"
            assert client.get(f"/runs/{run_cancelled}").json()["status"] == "cancelled"


class TestGracefulShutdown:
    """Shutdown must never leave rows stuck as ``running``."""

    def test_shutdown_force_cancels_running_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force the grace window to zero so the fallback path runs immediately."""
        import asrbench.main as main_mod

        monkeypatch.setattr(main_mod, "_SHUTDOWN_GRACE_SECONDS", 0.0)

        # Boot once to create the schema.
        app1 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app1):
            pass

        # Seed a running row via a direct connection (NO lifespan — we do
        # not want startup-recovery to grab it before shutdown does).
        run_id, _ = _seed_via_direct_conn(_db_path(tmp_path), run_status="running")

        # Second boot: manually drive the lifespan so startup recovery flips
        # the row to "failed", then the graceful-shutdown branch runs as the
        # TestClient exits.
        app2 = _fresh_app(tmp_path, monkeypatch)
        with TestClient(app2):
            # Re-insert a running row after startup recovery ran, so the
            # shutdown path has something to drain.
            run_id, _ = _seed_via_direct_conn(_db_path(tmp_path), run_status="running")

        # After lifespan shutdown — row should be cancelled, not running.
        conn = duckdb.connect(str(_db_path(tmp_path)))
        try:
            row = conn.execute(
                "SELECT status, cancel_requested, error_message FROM runs WHERE run_id = ?",
                [run_id],
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "cancelled"
        assert bool(row[1]) is True
        assert "grace period" in (row[2] or "")


class TestCancelFlagPersistedToDb:
    """POST /runs/<id>/cancel writes cancel_requested=TRUE on the row."""

    def test_cancel_endpoint_sets_db_flag(self, app_client: TestClient) -> None:
        from tests.integration.conftest import insert_dataset, insert_model, insert_run

        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="running")

        resp = app_client.post(f"/runs/{run_id}/cancel")
        assert resp.status_code == 200

        from asrbench.db import get_conn

        row = (
            get_conn()
            .cursor()
            .execute("SELECT cancel_requested FROM runs WHERE run_id = ?", [run_id])
            .fetchone()
        )
        assert row is not None
        assert bool(row[0]) is True

    def test_cancel_helpers_roundtrip(self, app_client: TestClient) -> None:
        """Direct helpers mirror the endpoint behaviour."""
        from asrbench.api.runs import clear_cancel, is_cancel_requested, request_cancel
        from tests.integration.conftest import insert_dataset, insert_model, insert_run

        _ = app_client  # fixture already initialised the DB via lifespan
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="running")

        assert is_cancel_requested(run_id) is False
        request_cancel(run_id)
        assert is_cancel_requested(run_id) is True
        clear_cancel(run_id)
        assert is_cancel_requested(run_id) is False


class TestNewConnectionHelper:
    """``db.new_connection`` must open an independent handle to the same file."""

    def test_new_connection_sees_singleton_writes(self, app_client: TestClient) -> None:
        from tests.integration.conftest import insert_dataset, insert_model, insert_run

        _ = app_client
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="pending")

        from asrbench.db import new_connection

        fresh = new_connection()
        try:
            row = (
                fresh.cursor()
                .execute("SELECT status FROM runs WHERE run_id = ?", [run_id])
                .fetchone()
            )
        finally:
            fresh.close()
        assert row is not None
        assert row[0] == "pending"
