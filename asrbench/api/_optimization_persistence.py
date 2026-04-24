"""DB helpers for the /optimize API.

Every read/write against ``optimization_studies`` /
``optimization_trials`` lives here so the background-task and endpoint
modules can focus on orchestration. No FastAPI, no Pydantic — just
DuckDB cursors and plain dicts.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from asrbench.engine.errors import sanitize_error

from ._optimization_models import StudyResponse, TrialResponse

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

__all__ = [
    "insert_single_study",
    "insert_two_stage_pair",
    "insert_global_config_pair",
    "load_study",
    "list_studies",
    "list_trials",
    "finalize_stage_success",
    "finalize_stage_failure",
    "warm_start_result",
]


# ---------------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------------


def insert_single_study(
    cur: duckdb.DuckDBPyConnection,
    *,
    req: Any,
) -> str:
    """Insert a single ``running`` optimization_studies row and return its id."""
    cur.execute(
        """
        INSERT INTO optimization_studies
            (model_id, dataset_id, lang, space, objective, budget,
             mode, eps_min, prior_study_id, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', now())
        RETURNING study_id
        """,
        [
            req.model_id,
            req.dataset_id,
            req.lang,
            json.dumps(req.space),
            json.dumps(req.objective.model_dump()),
            json.dumps(req.budget.model_dump()),
            req.mode,
            req.eps_min,
            req.prior_study_id,
        ],
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Failed to insert study row.")
    return str(row[0])


def _insert_stage_row(
    cur: duckdb.DuckDBPyConnection,
    *,
    model_id: str,
    dataset_id: str,
    lang: str,
    space_json: str,
    objective_json: str,
    mode: str,
    budget_val: int | None,
    eps_val: float | None,
) -> str:
    """Insert a single stage row used by two-stage / global-config.

    Budget / epsilon come in as optional — the caller either pinned them
    in the request or left them for ``run_two_stage`` to size from the
    space and duration.
    """
    cur.execute(
        """
        INSERT INTO optimization_studies
            (model_id, dataset_id, lang, space, objective, budget,
             mode, eps_min, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', now())
        RETURNING study_id
        """,
        [
            model_id,
            dataset_id,
            lang,
            space_json,
            objective_json,
            json.dumps({"hard_cap": budget_val or 0, "convergence_eps": eps_val or 0.0}),
            mode,
            eps_val or 0.0,
        ],
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Failed to insert stage row.")
    return str(row[0])


def insert_two_stage_pair(
    cur: duckdb.DuckDBPyConnection,
    *,
    req: Any,
) -> tuple[str, str]:
    """Insert two running study rows for a two-stage request."""
    space_json = json.dumps(req.space)
    objective_json = json.dumps(req.objective.model_dump())
    s1 = _insert_stage_row(
        cur,
        model_id=req.model_id,
        dataset_id=req.dataset_id,
        lang=req.lang,
        space_json=space_json,
        objective_json=objective_json,
        mode=req.mode,
        budget_val=req.stage1_budget,
        eps_val=req.stage1_epsilon,
    )
    s2 = _insert_stage_row(
        cur,
        model_id=req.model_id,
        dataset_id=req.dataset_id,
        lang=req.lang,
        space_json=space_json,
        objective_json=objective_json,
        mode=req.mode,
        budget_val=req.stage2_budget,
        eps_val=req.stage2_epsilon,
    )
    return s1, s2


def insert_global_config_pair(
    cur: duckdb.DuckDBPyConnection,
    *,
    req: Any,
) -> tuple[str, str]:
    """Insert two running study rows for a global-config request.

    The first dataset in the bundle stands in for the primary ``dataset_id``
    / ``lang`` — the full fleet is captured when ``finalize_stage_success``
    writes the reasoning blob.
    """
    space_json = json.dumps(req.space)
    objective_json = json.dumps(req.objective.model_dump())
    primary = req.datasets[0]
    s1 = _insert_stage_row(
        cur,
        model_id=req.model_id,
        dataset_id=primary.dataset_id,
        lang=primary.lang,
        space_json=space_json,
        objective_json=objective_json,
        mode=req.mode,
        budget_val=req.stage1_budget,
        eps_val=req.stage1_epsilon,
    )
    s2 = _insert_stage_row(
        cur,
        model_id=req.model_id,
        dataset_id=primary.dataset_id,
        lang=primary.lang,
        space_json=space_json,
        objective_json=objective_json,
        mode=req.mode,
        budget_val=req.stage2_budget,
        eps_val=req.stage2_epsilon,
    )
    return s1, s2


# ---------------------------------------------------------------------------
# SELECT / list
# ---------------------------------------------------------------------------


_STUDY_COLUMNS = (
    "study_id, model_id, dataset_id, lang, mode, status, eps_min, "
    "best_run_id, best_score, best_config, confidence, total_trials, reasoning, "
    "started_at, finished_at, created_at, error_message"
)


def _row_to_study(row: Any) -> StudyResponse:
    """Turn a DuckDB row from ``_STUDY_COLUMNS`` into a StudyResponse."""
    return StudyResponse(
        study_id=str(row[0]),
        model_id=str(row[1]),
        dataset_id=str(row[2]),
        lang=str(row[3]),
        mode=str(row[4]),
        status=str(row[5]),
        eps_min=float(row[6]),
        best_run_id=str(row[7]) if row[7] else None,
        best_score=row[8],
        best_config=json.loads(row[9]) if row[9] else None,
        confidence=str(row[10]) if row[10] else None,
        total_trials=row[11],
        reasoning=json.loads(row[12]) if row[12] else None,
        started_at=str(row[13]) if row[13] else None,
        finished_at=str(row[14]) if row[14] else None,
        created_at=str(row[15]),
        error_message=str(row[16]) if row[16] else None,
    )


def load_study(cur: duckdb.DuckDBPyConnection, study_id: str) -> StudyResponse | None:
    """Return the study row or None when the id is unknown."""
    row = cur.execute(
        f"SELECT {_STUDY_COLUMNS} FROM optimization_studies WHERE study_id = ?",
        [study_id],
    ).fetchone()
    return _row_to_study(row) if row else None


def list_studies(
    cur: duckdb.DuckDBPyConnection, *, status: str | None = None
) -> list[StudyResponse]:
    """List up to 100 most-recent studies, optionally filtered by status."""
    query = f"SELECT {_STUDY_COLUMNS} FROM optimization_studies"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT 100"
    rows = cur.execute(query, params).fetchall()
    return [_row_to_study(r) for r in rows]


def list_trials(
    cur: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
    page: int,
    page_size: int,
    phase: str | None,
) -> list[TrialResponse]:
    """Paginated trial list for a study. Caller must verify the study exists."""
    offset = (page - 1) * page_size
    conditions = ["study_id = ?"]
    params: list[Any] = [study_id]
    if phase:
        conditions.append("phase = ?")
        params.append(phase)
    where = " AND ".join(conditions)
    params.extend([page_size, offset])

    rows = cur.execute(
        f"""
        SELECT trial_id, run_id, phase, config, score,
               score_ci_lower, score_ci_upper, reasoning, created_at
        FROM optimization_trials
        WHERE {where}
        ORDER BY created_at
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()

    return [
        TrialResponse(
            trial_id=str(r[0]),
            run_id=str(r[1]) if r[1] else None,
            phase=str(r[2]),
            config=json.loads(r[3]) if isinstance(r[3], str) else (r[3] or {}),
            score=r[4],
            score_ci_lower=r[5],
            score_ci_upper=r[6],
            reasoning=str(r[7]) if r[7] else None,
            created_at=str(r[8]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# UPDATE / finalize
# ---------------------------------------------------------------------------


def _screening_to_json(screening: Any) -> str:
    return json.dumps(
        {
            "sensitive_order": list(screening.sensitive_order),
            "insensitive": list(screening.insensitive),
        }
    )


def finalize_stage_success(
    cur: duckdb.DuckDBPyConnection,
    *,
    study_id: str,
    stage_result: Any,
    reasoning: list[str] | None = None,
) -> None:
    """Write the completed stage's best config + screening to its row.

    ``reasoning`` defaults to the library-produced list; pass a custom
    list (e.g. prepended with a global-config header) to override.
    """
    cur.execute(
        """
        UPDATE optimization_studies
        SET status = 'completed',
            best_run_id = ?,
            best_score = ?,
            best_config = ?,
            confidence = ?,
            total_trials = ?,
            reasoning = ?,
            screening_result = ?,
            finished_at = now()
        WHERE study_id = ?
        """,
        [
            stage_result.best_trial.trial_id,
            stage_result.best_trial.score,
            json.dumps(dict(stage_result.best_config)),
            stage_result.validation.confidence if stage_result.validation else None,
            stage_result.total_trials,
            json.dumps(list(reasoning if reasoning is not None else stage_result.reasoning)),
            _screening_to_json(stage_result.screening),
            study_id,
        ],
    )


def finalize_stage_failure(
    cur: duckdb.DuckDBPyConnection,
    *,
    study_ids: tuple[str, ...],
    exc: BaseException,
) -> None:
    """Mark any still-``running`` row in ``study_ids`` as failed."""
    err_msg = sanitize_error(exc)
    for sid in study_ids:
        row = cur.execute(
            "SELECT status FROM optimization_studies WHERE study_id = ?",
            [sid],
        ).fetchone()
        if row and str(row[0]) == "running":
            cur.execute(
                "UPDATE optimization_studies SET status = 'failed', "
                "finished_at = now(), error_message = ? WHERE study_id = ?",
                [err_msg, sid],
            )


# ---------------------------------------------------------------------------
# Warm-start — reads prior study's trials + screening
# ---------------------------------------------------------------------------


def warm_start_result(
    cur: duckdb.DuckDBPyConnection,
    executor: Any,
    prior_study_id: str,
    space: Any,
) -> Any:
    """Import prior-study knowledge into the current optimizer.

    Two distinct things are carried over:

    1. **Screening metadata** (sensitive_order, insensitive) — always valid
       to reuse because it describes which parameters are worth touching,
       not their exact scores. Passed back via
       ``IAMSOptimizer(prior_screening=...)``.

    2. **Raw trial scores** — only valid when the prior study was measured
       under the SAME (model, dataset, language) context. The executor
       refuses a context mismatch and the current run re-measures from
       scratch.
    """
    from asrbench.engine.search.screening import ScreeningResult

    row = cur.execute(
        "SELECT screening_result, model_id, dataset_id, lang "
        "FROM optimization_studies WHERE study_id = ?",
        [prior_study_id],
    ).fetchone()
    if not row:
        logger.warning("Prior study %s not found — warm start skipped", prior_study_id)
        return None
    screening_json = json.loads(row[0]) if row[0] else None
    if screening_json is None:
        logger.warning(
            "Prior study %s has no screening_result — warm start skipped",
            prior_study_id,
        )
        return None
    prior_model_id = str(row[1]) if row[1] else None
    prior_dataset_id = str(row[2]) if row[2] else None
    prior_lang = str(row[3]) if row[3] else None

    prior_trials = _load_prior_trials(cur, prior_study_id)

    loaded = executor.warm_load(
        prior_trials,
        source_model_id=prior_model_id,
        source_dataset_id=prior_dataset_id,
        source_lang=prior_lang,
    )
    if loaded == 0 and prior_trials:
        logger.info(
            "Warm start from study %s: score cache NOT reused "
            "(context mismatch or refused) — screening metadata reused only.",
            prior_study_id,
        )
    else:
        logger.info(
            "Warm start from study %s: %d trials loaded into cache (of %d total)",
            prior_study_id,
            loaded,
            len(prior_trials),
        )

    baseline = _select_baseline_trial(prior_trials, space)
    if baseline is None:
        logger.warning("No usable baseline in prior study %s", prior_study_id)
        return None

    return ScreeningResult.from_summary(screening_json, baseline)


def _load_prior_trials(cur: duckdb.DuckDBPyConnection, prior_study_id: str) -> list[Any]:
    """Read every trial of the prior study in insertion order."""
    from asrbench.engine.search.trial import TrialResult

    trial_rows = cur.execute(
        """
        SELECT config, score, score_ci_lower, score_ci_upper, phase, trial_id
        FROM optimization_trials
        WHERE study_id = ?
        ORDER BY created_at
        """,
        [prior_study_id],
    ).fetchall()

    prior_trials = []
    for tr in trial_rows:
        config = json.loads(tr[0]) if isinstance(tr[0], str) else (tr[0] or {})
        prior_trials.append(
            TrialResult.from_db_row(
                config=config,
                score=float(tr[1]),
                score_ci=(float(tr[2]), float(tr[3])),
                phase=str(tr[4]) if tr[4] else "prior",
                trial_id=str(tr[5]) if tr[5] else None,
            )
        )
    return prior_trials


def _select_baseline_trial(prior_trials: list[Any], space: Any) -> Any | None:
    """Pick the trial that matches the space's defaults — or the best-scoring.

    Match on canonical config repr (deterministic, order-independent) so
    the lookup is stable regardless of PYTHONHASHSEED.
    """
    from asrbench.engine.search.trial import canonical_config_repr

    baseline_repr = canonical_config_repr(space.defaults())
    for t in prior_trials:
        if canonical_config_repr(t.config) == baseline_repr:
            return t
    if prior_trials:
        return min(prior_trials, key=lambda t: t.score)
    return None
