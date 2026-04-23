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
from asrbench.middleware.rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    """Startup: init DB. Shutdown: close DB connection."""
    from asrbench.db import get_conn, reset

    cfg = get_config()
    if cfg.server.host not in _LOOPBACK_HOSTS:
        logger.warning(
            "Binding to %s exposes ASRbench to the network. This is a single-user "
            "local tool with no authentication; set server.host to 127.0.0.1 in "
            "~/.asrbench/config.toml unless you understand the risk.",
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

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_LOCAL_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)

    app.include_router(system.router)
    app.include_router(datasets.router)
    app.include_router(models.router)
    app.include_router(runs.router)
    app.include_router(optimization.router)
    app.include_router(ws.router)

    return app
