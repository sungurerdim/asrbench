"""Integration tests for the Svelte UI static mount (Faz 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import insert_dataset


def _boot_with_static_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, static_dir: Path | None):
    """Build a FastAPI app whose ``_ui_static_dir`` points at ``static_dir``.

    Pass ``None`` to emulate a fresh checkout with no UI bundle.
    """
    from asrbench.config import get_config
    from asrbench.db import reset

    get_config.cache_clear()
    reset()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    import asrbench.main as main_mod

    probe = static_dir or (tmp_path / "no-such-ui-dir")
    monkeypatch.setattr(main_mod, "_ui_static_dir", lambda: probe)

    return main_mod.create_app()


def _write_fake_bundle(static_dir: Path) -> None:
    """Create an index.html + assets/ under ``static_dir``."""
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text(
        "<!doctype html><html><body>"
        "<div id='app'></div>"
        "<script src='/assets/app.js'></script>"
        "</body></html>",
        encoding="utf-8",
    )
    assets = static_dir / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "app.js").write_text("console.log('hello asrbench');\n", encoding="utf-8")
    (assets / "style.css").write_text("body{margin:0}", encoding="utf-8")


class TestStaticMount:
    def test_root_serves_index_html(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        static_dir = tmp_path / "static"
        _write_fake_bundle(static_dir)

        app = _boot_with_static_dir(tmp_path, monkeypatch, static_dir)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")
            assert "<div id='app'></div>" in resp.text

    def test_assets_directory_is_served(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        static_dir = tmp_path / "static"
        _write_fake_bundle(static_dir)

        app = _boot_with_static_dir(tmp_path, monkeypatch, static_dir)
        with TestClient(app) as client:
            resp = client.get("/assets/app.js")
            assert resp.status_code == 200
            assert resp.text.strip() == "console.log('hello asrbench');"

            css = client.get("/assets/style.css")
            assert css.status_code == 200
            assert "margin:0" in css.text

    def test_api_routes_take_priority_over_static(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static mount must NOT shadow registered API endpoints."""
        static_dir = tmp_path / "static"
        _write_fake_bundle(static_dir)

        app = _boot_with_static_dir(tmp_path, monkeypatch, static_dir)
        with TestClient(app) as client:
            # /system/health is a real route; must not fall through to
            # StaticFiles and return 404 or the index.html.
            resp = client.get("/system/health")
            assert resp.status_code == 200
            assert resp.json().get("status") == "ok"

            # Inserting a dataset then listing must go through the JSON API.
            insert_dataset()
            listed = client.get("/datasets")
            assert listed.status_code == 200
            assert isinstance(listed.json(), list)

    def test_no_mount_when_bundle_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no bundle on disk the mount must be skipped entirely.

        The API must keep responding (health check) and ``/`` must
        return 404 — there is simply no UI to serve.
        """
        app = _boot_with_static_dir(tmp_path, monkeypatch, static_dir=None)
        with TestClient(app) as client:
            assert client.get("/system/health").status_code == 200
            assert client.get("/").status_code == 404

    def test_directory_without_index_html_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A static dir that exists but is empty must not be mounted.

        Guards against partial CI artefacts where the dir was created
        by a previous tool but the bundle was never produced.
        """
        empty = tmp_path / "empty-static"
        empty.mkdir()

        app = _boot_with_static_dir(tmp_path, monkeypatch, empty)
        with TestClient(app) as client:
            assert client.get("/system/health").status_code == 200
            assert client.get("/").status_code == 404
