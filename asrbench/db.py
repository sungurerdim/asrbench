"""DuckDB connection manager and schema DDL for asrbench."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection, creating the DB file if needed."""
    global _conn
    if _conn is not None:
        return _conn

    from asrbench.config import get_config

    config = get_config()
    _conn = connect(config.storage.db_path)
    return _conn


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection at *db_path* and ensure the schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    init_db(conn)
    return conn


def reset() -> None:
    """Close the singleton connection and clear the reference (for tests)."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if not exist. Idempotent."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            name       VARCHAR NOT NULL,
            source     VARCHAR NOT NULL,
            lang       VARCHAR NOT NULL,
            split      VARCHAR NOT NULL DEFAULT 'test',
            local_path VARCHAR,
            max_duration_s DOUBLE,
            verified   BOOLEAN DEFAULT false
        )
    """)

    # Idempotent migration for existing databases
    try:
        conn.execute("ALTER TABLE datasets ADD COLUMN max_duration_s DOUBLE")
    except duckdb.CatalogException:
        pass  # column already exists

    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id       UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            family         VARCHAR NOT NULL,
            name           VARCHAR NOT NULL,
            backend        VARCHAR NOT NULL,
            local_path     VARCHAR NOT NULL,
            default_params JSON
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id     UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            model_id   UUID REFERENCES models(model_id),
            backend    VARCHAR NOT NULL,
            params     JSON,
            dataset_id UUID REFERENCES datasets(dataset_id),
            lang       VARCHAR NOT NULL DEFAULT 'en',
            mode       VARCHAR NOT NULL DEFAULT 'model_compare',
            status     VARCHAR NOT NULL DEFAULT 'pending',
            label      VARCHAR
        )
    """)

    # Idempotent migration for existing databases
    try:
        conn.execute("ALTER TABLE runs ADD COLUMN label VARCHAR")
    except duckdb.CatalogException:
        pass  # column already exists

    conn.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            run_id     UUID REFERENCES runs(run_id),
            offset_s   DOUBLE,
            duration_s DOUBLE,
            ref_text   VARCHAR,
            hyp_text   VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS aggregates (
            run_id              UUID REFERENCES runs(run_id),
            wer_mean            DOUBLE,
            cer_mean            DOUBLE,
            mer_mean            DOUBLE,
            wil_mean            DOUBLE,
            rtfx_mean           DOUBLE,
            rtfx_p95            DOUBLE,
            vram_peak_mb        DOUBLE,
            wall_time_s         DOUBLE,
            word_count          INTEGER,
            data_leakage_warning BOOLEAN DEFAULT false,
            wer_ci_lower        DOUBLE,
            wer_ci_upper        DOUBLE
        )
    """)

    # Idempotent migration: wil_mean added in v0.2 for correct WIL aggregation.
    try:
        conn.execute("ALTER TABLE aggregates ADD COLUMN wil_mean DOUBLE")
    except duckdb.CatalogException:
        pass  # column already exists

    conn.execute("""
        CREATE TABLE IF NOT EXISTS optimization_studies (
            study_id    UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            model_id    UUID REFERENCES models(model_id),
            dataset_id  UUID REFERENCES datasets(dataset_id),
            lang        VARCHAR NOT NULL DEFAULT 'en',
            space       JSON,
            objective   JSON,
            budget      JSON,
            mode        VARCHAR,
            eps_min     DOUBLE,
            status      VARCHAR NOT NULL DEFAULT 'pending',
            best_run_id UUID,
            best_score  DOUBLE,
            best_config JSON,
            confidence  DOUBLE,
            total_trials INTEGER DEFAULT 0,
            reasoning   JSON,
            prior_study_id UUID,
            screening_result JSON,
            started_at  TIMESTAMP,
            finished_at TIMESTAMP,
            created_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # Idempotent migration for existing databases
    for col, col_type in [
        ("prior_study_id", "UUID"),
        ("screening_result", "JSON"),
        ("error_message", "VARCHAR"),
    ]:
        try:
            conn.execute(f"ALTER TABLE optimization_studies ADD COLUMN {col} {col_type}")
        except duckdb.CatalogException:
            pass  # column already exists

    conn.execute("""
        CREATE TABLE IF NOT EXISTS optimization_trials (
            trial_id     UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            study_id     UUID REFERENCES optimization_studies(study_id),
            run_id       UUID REFERENCES runs(run_id),
            phase        VARCHAR,
            config       JSON,
            score        DOUBLE,
            score_ci_lower DOUBLE,
            score_ci_upper DOUBLE,
            reasoning    VARCHAR,
            created_at   TIMESTAMP DEFAULT current_timestamp
        )
    """)

    logger.debug("Database schema initialized")
