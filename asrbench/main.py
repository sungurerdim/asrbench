"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from asrbench import __version__
from asrbench.api import datasets, models, optimization, runs, system, ws
from asrbench.config import get_config
from asrbench.middleware.auth import LOOPBACK_HOSTS, AuthMiddleware, get_api_key
from asrbench.middleware.rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)

_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    """Startup: init DB. Shutdown: close DB connection."""
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
    get_conn()  # triggers init_db via connect()
    yield
    logger.info("ASRbench shutting down")
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

    return app
