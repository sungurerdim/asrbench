"""
BenchmarkTrialExecutor — the production TrialExecutor for IAMS.

Wraps BenchmarkEngine so every IAMS trial becomes a full benchmark run against
a real dataset and model. Each call to `evaluate()`:

    1. Inserts a new row into the `runs` table with the trial's config.
    2. Kicks off BenchmarkEngine.run() synchronously (via asyncio.run).
    3. Reads back the aggregate row, extracts metrics, and computes the score
       through the injected Objective.
    4. Inserts an `optimization_trials` row linking the trial to the run.
    5. Returns the TrialResult for the search layer to consume.

Config-level caching:
    The executor keeps an in-process dict mapping config_key → TrialResult so
    repeated evaluations of the same config (e.g., Layer 2 revisiting the
    baseline after Layer 3) return instantly without re-running the benchmark.
    This is a memory cache only — it does NOT persist across process restarts.
    The transcript_cache layer inside BenchmarkEngine provides persistent
    caching at the audio-segment level.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from asrbench.engine.search.objective import Objective
from asrbench.engine.search.trial import TrialResult

if TYPE_CHECKING:
    import duckdb  # type: ignore[import-untyped]

    from asrbench.data.dataset_manager import PreparedDataset
    from asrbench.engine.benchmark import BenchmarkEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTrialExecutor:
    """
    Production trial executor wrapping BenchmarkEngine.

    Construction inputs:
        engine       : a BenchmarkEngine instance (with cache_dir wired up)
        conn         : DuckDB connection (engine uses cur = conn.cursor())
        study_id     : UUID of the enclosing optimization_studies row
        model_id     : target model UUID
        backend_name : backend plugin name (e.g. "faster_whisper")
        model_local_path: path passed to BaseBackend.load()
        dataset      : PreparedDataset ready for evaluation
        objective    : Objective instance producing score() + score_ci()
        backend      : already-loaded BaseBackend instance (caller manages
                       load/unload lifecycle to avoid reloading the model
                       per trial)

    The executor does NOT manage backend loading — that is the caller's
    responsibility. A typical usage pattern:

        backend.load(model_local_path, default_params)
        try:
            executor = BenchmarkTrialExecutor(engine=..., backend=..., ...)
            optimizer = IAMSOptimizer(executor=executor, ...)
            result = optimizer.run()
        finally:
            backend.unload()
    """

    engine: BenchmarkEngine
    conn: duckdb.DuckDBPyConnection
    study_id: str
    model_id: str
    backend_name: str
    model_local_path: str
    dataset: PreparedDataset
    objective: Objective
    backend: Any  # BaseBackend — not type-annotated to avoid circular import
    lang: str = "en"
    mode_tag: str = "param_compare"

    _cache: dict[str, TrialResult] = field(default_factory=dict, init=False)
    _runs_used: int = field(default=0, init=False)

    @property
    def runs_used(self) -> int:
        return self._runs_used

    def set_cache_enabled(self, enabled: bool) -> None:
        """
        Validation layer calls this to force fresh runs. When disabled, all
        incoming configs miss the cache and a new benchmark run is executed.
        """
        if not enabled:
            self._cache.clear()
        self._cache_enabled = enabled

    def evaluate(
        self,
        config: Mapping[str, Any],
        *,
        phase: str = "unknown",
        reasoning: str = "",
    ) -> TrialResult:
        """
        Execute a benchmark run for this config and return the corresponding
        TrialResult.

        This call is SYNCHRONOUS from the caller's perspective — internally
        it drives an asyncio event loop because BenchmarkEngine.run() is an
        async coroutine. A fresh loop is created per call so the executor can
        be used from both sync and async contexts.
        """
        key = self._config_key(config)
        if key in self._cache:
            cached = self._cache[key]
            self._persist_trial(cached, phase, reasoning, cached.trial_id)
            return cached.with_phase(phase, reasoning)

        # Serialize config once for both DB writes
        config_json = json.dumps(dict(config))

        # Insert a new row in `runs`
        run_id = self._create_run_row(config_json)

        # Run the benchmark synchronously
        try:
            asyncio.run(
                self.engine.run(
                    run_id=run_id,
                    backend=self.backend,
                    dataset=self.dataset,
                    params=dict(config),
                    model_family=None,
                    model_local_path=self.model_local_path,
                )
            )
        except Exception as exc:
            logger.error("BenchmarkTrialExecutor: run %s failed: %s", run_id, exc)
            raise

        # Read back aggregate
        metrics = self._read_aggregate(run_id)
        score = self.objective.score(metrics)
        score_ci = self.objective.score_ci(metrics)

        trial = TrialResult(
            config=dict(config),
            metrics=metrics,
            score=score,
            score_ci=score_ci,
            phase=phase,
            reasoning=reasoning,
            trial_id=run_id,
        )

        self._runs_used += 1
        self._cache[key] = trial
        self._persist_trial(trial, phase, reasoning, run_id, config_json)
        return trial

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_run_row(self, config_json: str) -> str:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (model_id, backend, params, dataset_id, lang, mode)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING run_id
            """,
            [
                self.model_id,
                self.backend_name,
                config_json,
                self.dataset.dataset_id,
                self.lang,
                self.mode_tag,
            ],
        )
        row = cur.fetchone()
        assert row is not None
        return str(row[0])

    def _read_aggregate(self, run_id: str) -> dict[str, float | None]:
        cur = self.conn.cursor()
        row = cur.execute(
            """
            SELECT wer_mean, cer_mean, mer_mean, rtfx_mean, rtfx_p95, vram_peak_mb,
                   wall_time_s, wer_ci_lower, wer_ci_upper
            FROM aggregates WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"BenchmarkTrialExecutor: run {run_id} produced no aggregate row. "
                "The benchmark may have failed mid-run — check runs.status."
            )
        return {
            "wer": row[0],
            "cer": row[1],
            "mer": row[2],
            "wil": row[2],  # WIL not tracked separately in aggregates today
            "rtfx_mean": row[3],
            "rtfx_p95": row[4],
            "vram_peak_mb": row[5],
            "wall_time_s": row[6],
            "wer_ci_lower": row[7],
            "wer_ci_upper": row[8],
        }

    def _persist_trial(
        self,
        trial: TrialResult,
        phase: str,
        reasoning: str,
        run_id: str | None,
        config_json: str | None = None,
    ) -> None:
        """Write the trial into optimization_trials so the study is auditable."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO optimization_trials
                (study_id, run_id, phase, config, score,
                 score_ci_lower, score_ci_upper, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                self.study_id,
                run_id,
                phase,
                config_json or json.dumps(dict(trial.config)),
                trial.score,
                trial.score_ci[0],
                trial.score_ci[1],
                reasoning,
            ],
        )

    @staticmethod
    def _config_key(config: Mapping[str, Any]) -> str:
        return str(hash(tuple(sorted(config.items()))))
