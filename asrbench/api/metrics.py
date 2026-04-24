"""Prometheus metrics endpoint — opt-in via the [observability] extra.

`prometheus-fastapi-instrumentator` ships request-level metrics
(request counter, latency histogram, in-flight gauge) out of the box.
On top of that we register a handful of ASRbench-specific gauges and
counters so an Ops dashboard can track benchmark throughput, VRAM
pressure, and per-backend error rates.

The module is tolerant of missing deps: when the `observability`
extra is not installed, :func:`install_metrics` logs a warning and
returns without touching the app. That keeps the base install light.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

__all__ = [
    "BACKEND_ERRORS_TOTAL",
    "RUN_DURATION_SECONDS",
    "RUN_TOTAL",
    "VRAM_USED_MB",
    "install_metrics",
]

# Metric handles. Exposed at module level so background tasks can import
# them without having to own a reference to the instrumentator. We
# initialise them lazily inside install_metrics(); before that call they
# are no-op placeholders so imports work even when prometheus_client is
# missing.


class _NoOpMetric:
    """Stand-in that swallows every counter/gauge op when prom is absent."""

    def inc(self, *_args, **_kwargs) -> None:
        return None

    def set(self, *_args, **_kwargs) -> None:
        return None

    def observe(self, *_args, **_kwargs) -> None:
        return None

    def labels(self, *_args, **_kwargs) -> _NoOpMetric:
        return self


RUN_DURATION_SECONDS: object = _NoOpMetric()
RUN_TOTAL: object = _NoOpMetric()
VRAM_USED_MB: object = _NoOpMetric()
BACKEND_ERRORS_TOTAL: object = _NoOpMetric()

# Process-wide install guard. Prometheus's default registry lives in
# ``prometheus_client.REGISTRY`` and raises on duplicate registration;
# the FastAPI instrumentator registers its own HTTP metrics on every
# ``.instrument()`` call. In tests that rebuild the app per case we
# silently skip re-installation — the first app in the process owns
# the /metrics endpoint; subsequent rebuilds still work for every
# other route, they just don't re-expose /metrics.
_INSTALLED_ONCE: bool = False


def install_metrics(app: FastAPI) -> bool:
    """Attach Prometheus-compatible metrics to *app*.

    Returns True when the instrumentator was wired up, False when the
    optional dependency is missing (no error — just skipped). Safe to
    call twice: the guards below bail out cleanly if metrics already
    exist on the app.
    """
    global RUN_DURATION_SECONDS, RUN_TOTAL, VRAM_USED_MB, BACKEND_ERRORS_TOTAL
    global _INSTALLED_ONCE

    try:
        from prometheus_client import Counter, Gauge, Histogram
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        logger.info(
            "Prometheus metrics skipped — install `pip install "
            "'asrbench[observability]'` to enable /metrics."
        )
        return False

    if getattr(app.state, "_metrics_installed", False):
        return True
    app.state._metrics_installed = True

    # Prometheus's default registry is process-global, so creating the
    # same metric twice (test teardown, second create_app()) raises
    # "Duplicated timeseries". Guard by checking the module-level
    # placeholders and skip re-registration once real metrics exist.
    if isinstance(RUN_DURATION_SECONDS, _NoOpMetric):
        RUN_DURATION_SECONDS = Histogram(
            "asrbench_run_duration_seconds",
            "Wall-clock duration of completed benchmark runs.",
            buckets=(5, 15, 60, 300, 1_200, 3_600, 7_200, 14_400),
        )
        RUN_TOTAL = Counter(
            "asrbench_run_total",
            "Count of finalised benchmark runs by terminal status.",
            labelnames=("status",),
        )
        VRAM_USED_MB = Gauge(
            "asrbench_vram_used_mb",
            "Current NVIDIA VRAM usage in MiB (0 when no NVML handle).",
        )
        BACKEND_ERRORS_TOTAL = Counter(
            "asrbench_backend_errors_total",
            "Backend-level transcription errors by backend name.",
            labelnames=("backend",),
        )

    if _INSTALLED_ONCE:
        # A previous install registered every metric against the
        # process-wide prometheus_client.REGISTRY; rebuilding the app
        # would trip "Duplicated timeseries". Skip instrumentation on
        # the new app so tests still work. In production the app is
        # built exactly once per process so this branch never fires.
        logger.debug("Prometheus metrics already installed in this process; skipping.")
        return True

    Instrumentator(
        should_instrument_requests_inprogress=True,
        inprogress_labels=True,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        should_gzip=True,
    )
    _INSTALLED_ONCE = True
    logger.info("Prometheus metrics installed at /metrics")
    return True
