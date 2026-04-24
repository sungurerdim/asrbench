"""Integration test for the Prometheus metrics endpoint (Faz 11)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestMetricsModule:
    """The install_metrics helper contract is stable regardless of
    whether the optional dep is present.

    We do NOT assert anything about ``/metrics`` over a live test
    client: Prometheus's default registry is process-wide and
    ``install_metrics`` guards against re-registration, so in a
    pytest session that rebuilds the FastAPI app for every fixture
    only the first app owns /metrics. The check below targets the
    install helper's behaviour instead.
    """

    def test_install_metrics_noop_without_optional_dep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing optional dep → install_metrics returns False quietly."""
        import sys

        from fastapi import FastAPI

        # Reset the install guard so this test's result is independent
        # of other tests that may have installed metrics first.
        import asrbench.api.metrics as metrics_mod

        monkeypatch.setattr(metrics_mod, "_INSTALLED_ONCE", False)
        monkeypatch.setitem(sys.modules, "prometheus_fastapi_instrumentator", None)

        app = FastAPI()
        installed = metrics_mod.install_metrics(app)
        assert installed is False
        with TestClient(app) as client:
            assert client.get("/metrics").status_code == 404

    def test_noop_metric_interface(self) -> None:
        """The placeholder handles absorb inc/set/observe/labels calls."""
        from asrbench.api.metrics import _NoOpMetric

        noop = _NoOpMetric()
        # None of these can raise; they must also be chainable via labels.
        noop.inc()
        noop.set(123)
        noop.observe(0.5)
        assert noop.labels(backend="x") is noop
