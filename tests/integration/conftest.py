"""Integration test fixtures — real DB via lifespan, no mocked I/O."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """
    Provide a FastAPI TestClient backed by a real DuckDB in a temp directory.

    The lifespan runs normally (init_db + VRAMMonitor). After startup,
    ``asrbench.db.get_conn()`` points to the in-temp-dir database so
    test helpers can insert seed data via the same connection.
    """
    from asrbench.config import get_config

    # Fresh config per test — otherwise lru_cache leaks paths between tests.
    get_config.cache_clear()

    # Redirect home so config writes to tmp_path, not the real ~/.asrbench.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows compat

    from asrbench.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    get_config.cache_clear()


# ---------------------------------------------------------------------------
# Seed helpers — insert minimal records into the live DB connection
# ---------------------------------------------------------------------------


def insert_model(backend: str = "faster-whisper", local_path: str = "/tmp/model") -> str:
    """Insert a model row; returns model_id."""
    from asrbench.db import get_conn

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO models (family, name, backend, local_path, default_params) "
        "VALUES (?, ?, ?, ?, ?) RETURNING model_id",
        ["whisper", "test-model", backend, local_path, json.dumps({"beam_size": 5})],
    )
    row = cur.fetchone()
    assert row is not None
    return str(row[0])


def insert_dataset(
    source: str = "custom",
    lang: str = "en",
    local_path: str = "/tmp/ds",
) -> str:
    """Insert a dataset row; returns dataset_id."""
    from asrbench.db import get_conn

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (name, source, lang, split, local_path, verified) "
        "VALUES (?, ?, ?, ?, ?, ?) RETURNING dataset_id",
        [f"{source}_{lang}_test", source, lang, "test", local_path, True],
    )
    row = cur.fetchone()
    assert row is not None
    return str(row[0])


def insert_run(
    model_id: str,
    dataset_id: str,
    status: str = "pending",
    lang: str = "en",
    mode: str = "model_compare",
) -> str:
    """Insert a run row; returns run_id."""
    from asrbench.db import get_conn

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (model_id, backend, params, dataset_id, lang, mode, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING run_id",
        [model_id, "faster-whisper", json.dumps({"beam_size": 5}), dataset_id, lang, mode, status],
    )
    row = cur.fetchone()
    assert row is not None
    return str(row[0])


def insert_aggregate(run_id: str) -> None:
    """Insert a minimal aggregate row for a completed run."""
    from asrbench.db import get_conn

    conn = get_conn()
    conn.cursor().execute(
        "INSERT INTO aggregates (run_id, wer_mean, cer_mean, mer_mean, rtfx_mean, "
        "rtfx_p95, vram_peak_mb, wall_time_s, word_count, data_leakage_warning) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [run_id, 0.15, 0.08, 0.12, 5.3, 6.1, 2048.0, 30.0, 500, False],
    )
