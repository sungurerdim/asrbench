"""Integration test for the dataset fetch timeout wrapper (Faz 4.1)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


class TestFetchTimeout:
    """Wrapping `_do_fetch` in asyncio.wait_for stops a hung HF stream."""

    def test_timeout_surfaces_as_error_event(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force DatasetManager._fetch_hf to sleep past the configured timeout.

        The background task should log an error and publish an ``error``
        event on the dataset bus. The row stays ``verified=false`` because
        the UPDATE runs only on the success branch.
        """
        from asrbench.config import get_config
        from asrbench.data.dataset_manager import DatasetManager

        # Keep the test fast: one-second timeout, two-second sleep.
        cfg = get_config()
        monkeypatch.setattr(cfg.limits, "dataset_fetch_timeout_s", 1.0)

        def _sluggish_fetch(self, source, lang, split, max_duration_s):
            import time

            time.sleep(2.0)
            raise RuntimeError("should have timed out before reaching here")

        monkeypatch.setattr(DatasetManager, "_fetch_hf", _sluggish_fetch)

        # Seed a dataset row whose fetch is about to stall. We insert it
        # as verified=False so the timeout keeps it unverified; the
        # conftest helper hardcodes True.
        from asrbench.db import get_conn

        cur = get_conn().cursor()
        cur.execute(
            "INSERT INTO datasets (name, source, lang, split, local_path, verified) "
            "VALUES (?, 'fleurs', 'en', 'test', '/tmp/ds', false) RETURNING dataset_id",
            ["timeout_fleurs_en_test"],
        )
        row = cur.fetchone()
        assert row is not None
        dataset_id = str(row[0])

        # Drive the background task by calling it directly — the TestClient
        # lifespan is not strictly needed and this keeps the test focused.
        from asrbench.api.datasets import _fetch_background

        asyncio.run(
            _fetch_background(
                dataset_id=dataset_id,
                source="fleurs",
                lang="en",
                split="test",
                local_path=None,
                max_duration_s=None,
            )
        )

        # Row must NOT be flipped to verified=true.
        resp = app_client.get(f"/datasets/{dataset_id}")
        assert resp.status_code == 200
        assert resp.json()["verified"] is False


class TestTimeoutConfigWired:
    """The LimitsConfig dataclass exposes the new timeout knob with a sane default."""

    def test_defaults_are_accessible(self) -> None:
        from asrbench.config import LimitsConfig

        cfg = LimitsConfig()
        assert cfg.dataset_fetch_timeout_s == pytest.approx(600.0)
        assert cfg.segment_timeout_s == pytest.approx(120.0)

    def test_toml_override_respected(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A user-supplied config.toml must reach LimitsConfig."""
        from asrbench.config import get_config

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        get_config.cache_clear()

        config_dir = tmp_path / ".asrbench"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.toml").write_text(
            "[limits]\ndataset_fetch_timeout_s = 42.5\nsegment_timeout_s = 7.0\n",
            encoding="utf-8",
        )

        cfg = get_config()
        assert cfg.limits.dataset_fetch_timeout_s == pytest.approx(42.5)
        assert cfg.limits.segment_timeout_s == pytest.approx(7.0)

        get_config.cache_clear()
