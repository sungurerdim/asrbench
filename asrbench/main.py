"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from asrbench import __version__
from asrbench.api import datasets, models, optimization, runs, system, ws
from asrbench.config import get_config
from asrbench.middleware.auth import LOOPBACK_HOSTS, AuthMiddleware, get_api_key
from asrbench.middleware.rate_limit import RateLimitMiddleware

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"

_STALE_STATUSES: tuple[str, ...] = ("pending", "running")
"""Row states that are only valid while a process is alive.

Any row still in one of these statuses on startup belongs to a previous
process that died without finalising it (Ctrl-C, OOM kill, host crash).
The startup recovery job rewrites them to ``failed`` so the UI does not
show a forever-spinning job and new jobs can start.
"""

_SHUTDOWN_GRACE_SECONDS: float = 30.0
"""Maximum time the shutdown hook waits for running jobs to observe the
cancel flag before force-marking them ``cancelled``. 30 s is long enough
for a segment-granularity benchmark to reach the next poll point but
short enough that ``uvicorn`` does not lose patience and kill the
process mid-SQL."""


def _recover_stale_rows(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Mark orphaned ``pending`` / ``running`` rows as ``failed``.

    Returns counts per table so the lifespan can emit a single warning
    summarising the cleanup. Each UPDATE is idempotent — running it on
    a cold DB is a no-op.
    """
    cur = conn.cursor()
    recovered: dict[str, int] = {}

    for table in ("runs", "optimization_studies"):
        result = cur.execute(
            f"UPDATE {table} "
            "SET status = 'failed', "
            "    error_message = 'startup recovery: previous process did not finalise this job' "
            f"WHERE status IN ({','.join('?' * len(_STALE_STATUSES))})",
            list(_STALE_STATUSES),
        )
        # DuckDB's UPDATE returns a cursor whose ``fetchone()`` yields
        # ``(rowcount,)`` — or ``None`` when the executor returns early.
        # Guard the indexing so a driver oddity never crashes startup.
        row = result.fetchone()
        try:
            recovered[table] = int(row[0]) if row else 0
        except (TypeError, ValueError):
            recovered[table] = 0

    return recovered


_GRACE_ELAPSED_RUN_MSG = "shutdown: grace period elapsed before the run observed the cancel flag"
_GRACE_ELAPSED_STUDY_MSG = (
    "shutdown: grace period elapsed before the study observed the cancel flag"
)


async def _graceful_shutdown(conn: duckdb.DuckDBPyConnection) -> None:
    """Signal cancellation to live jobs and wait up to 30 s.

    Uses the ``cancel_requested`` column so a benchmark/optimiser loop
    observes the request on its next segment boundary and exits cleanly
    (rows end up ``status='cancelled'`` instead of ``failed``). If the
    window elapses while rows are still ``running``, flip them to
    ``cancelled`` anyway so the next startup's recovery job does not
    mis-label them as ``failed``.
    """
    cur = conn.cursor()

    cur.execute("UPDATE runs SET cancel_requested = true WHERE status = 'running'")
    cur.execute("UPDATE optimization_studies SET cancel_requested = true WHERE status = 'running'")

    deadline = asyncio.get_event_loop().time() + _SHUTDOWN_GRACE_SECONDS
    while asyncio.get_event_loop().time() < deadline:
        remaining = cur.execute(
            "SELECT "
            "  (SELECT count(*) FROM runs WHERE status = 'running') "
            "+ (SELECT count(*) FROM optimization_studies WHERE status = 'running')"
        ).fetchone()
        if remaining is None or int(remaining[0]) == 0:
            return
        await asyncio.sleep(0.25)

    cur.execute(
        "UPDATE runs SET status = 'cancelled', error_message = ? WHERE status = 'running'",
        [_GRACE_ELAPSED_RUN_MSG],
    )
    cur.execute(
        "UPDATE optimization_studies SET status = 'cancelled', finished_at = now(), "
        "error_message = ? WHERE status = 'running'",
        [_GRACE_ELAPSED_STUDY_MSG],
    )
    logger.warning(
        "Shutdown grace window (%.0fs) elapsed — force-cancelled any still-running jobs",
        _SHUTDOWN_GRACE_SECONDS,
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Startup: init DB + clean stale rows. Shutdown: drain + close DB."""
    from asrbench.db import get_conn, reset

    cfg = get_config()
    if cfg.server.host not in LOOPBACK_HOSTS:
        if get_api_key() is None:
            logger.error(
                "Server is bound to %s but ASRBENCH_API_KEY is not set. "
                "Every remote request will be rejected with 401. Set the "
                "env var and restart, or re-bind to 127.0.0.1.",
                cfg.server.host,
            )
        else:
            logger.warning(
                "ASRbench is reachable at %s. AuthMiddleware will reject any "
                "non-loopback client that does not present X-API-Key.",
                cfg.server.host,
            )

    logger.info("ASRbench %s starting up — initializing database", __version__)
    conn = get_conn()

    recovered = _recover_stale_rows(conn)
    total_recovered = sum(recovered.values())
    if total_recovered:
        logger.warning(
            "Startup recovery: %d runs + %d studies left in pending/running state "
            "by a previous process were marked failed.",
            recovered.get("runs", 0),
            recovered.get("optimization_studies", 0),
        )

    yield

    logger.info(
        "ASRbench shutting down — draining active jobs (grace: %ds)", int(_SHUTDOWN_GRACE_SECONDS)
    )
    try:
        await _graceful_shutdown(conn)
    except Exception as exc:
        logger.warning("Graceful shutdown drain failed: %s", exc, exc_info=True)
    reset()


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""
    app = FastAPI(
        title="ASRbench",
        version=__version__,
        description="Fully local, multi-backend ASR benchmarking platform",
        lifespan=_lifespan,
    )

    # FastAPI wraps middleware in reverse registration order — the newest
    # call becomes the outermost wrapper. Request order is therefore:
    # AuthMiddleware (last added) → RateLimitMiddleware → CORSMiddleware
    # (first added) → handler. Auth must be outermost so unauthenticated
    # remote clients never burn rate-limit tokens.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_LOCAL_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*", "X-API-Key"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

    app.include_router(system.router)
    app.include_router(datasets.router)
    app.include_router(models.router)
    app.include_router(runs.router)
    app.include_router(optimization.router)
    app.include_router(ws.router)

    from asrbench.api.metrics import install_metrics

    install_metrics(app)

    _mount_ui_if_present(app)

    return app


def _ui_static_dir() -> Path:
    """Resolve the on-disk directory that Vite builds into.

    Kept as a helper so tests can monkey-patch it to a tmp dir with
    a handcrafted index.html.
    """
    return Path(__file__).resolve().parent / "static"


def _mount_ui_if_present(app: FastAPI) -> None:
    """Serve the built Svelte UI at ``/`` when the bundle exists.

    ``pip install asrbench`` ships the wheel with ``asrbench/static/``
    populated by the CI build step. Mounting happens AFTER every API
    router is registered so path-matching still prefers real endpoints
    (e.g. ``GET /system/health``) over a 404 from StaticFiles.

    When the directory is empty (fresh checkout that has not run
    ``npm run build`` yet), no mount is added so the dev experience
    stays unchanged — developers hit the Vite dev server at :5173 via
    the ``--dev`` flag and forget about the static mount entirely.
    """
    static_dir = _ui_static_dir()
    if not static_dir.is_dir():
        return
    if not (static_dir / "index.html").is_file():
        logger.info("UI bundle missing at %s — serve API only", static_dir)
        return

    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
