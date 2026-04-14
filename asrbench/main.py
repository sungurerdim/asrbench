"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from asrbench.api import datasets, models, optimization, runs, system, ws
from asrbench.middleware.rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup: init DB. Shutdown: close DB connection."""
    from asrbench.db import get_conn, reset

    logger.info("ASRbench starting up — initializing database")
    get_conn()  # triggers init_db via connect()
    yield
    logger.info("ASRbench shutting down")
    reset()


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""
    app = FastAPI(
        title="ASRbench",
        version="0.1.0",
        description="Fully local, multi-backend ASR benchmarking platform",
        lifespan=_lifespan,
    )

    app.include_router(system.router)
    app.include_router(datasets.router)
    app.include_router(models.router)
    app.include_router(runs.router)
    app.include_router(optimization.router)
    app.include_router(ws.router)

    app.add_middleware(RateLimitMiddleware)

    return app
