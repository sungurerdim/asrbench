"""Smoke tests for the ``asrbench config`` CLI."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from asrbench.cli.config_cmd import app


@pytest.fixture
def redirected_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Redirect Path.home() so config lands in tmp_path and tests stay isolated."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from asrbench.config import get_config

    get_config.cache_clear()
    yield tmp_path
    get_config.cache_clear()


def test_path_prints_config_location(redirected_home: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["path"])
    assert result.exit_code == 0
    assert ".asrbench" in result.stdout


def test_init_creates_default_config(redirected_home: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    expected_path = redirected_home / ".asrbench" / "config.toml"
    assert expected_path.exists()
    body = expected_path.read_text(encoding="utf-8")
    assert "[server]" in body
    assert "[limits]" in body


def test_init_refuses_overwrite_without_force(redirected_home: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])  # create once
    result = runner.invoke(app, ["init"])  # again, without --force
    assert result.exit_code == 0
    assert "already exists" in result.stdout


def test_init_force_overwrites(redirected_home: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    path = redirected_home / ".asrbench" / "config.toml"
    path.write_text('[server]\nhost = "0.0.0.0"\n', encoding="utf-8")
    result = runner.invoke(app, ["init", "--force"])
    assert result.exit_code == 0
    reloaded = path.read_text(encoding="utf-8")
    assert 'host = "127.0.0.1"' in reloaded


def test_set_updates_scalar(redirected_home: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["set", "server.port", "9000"])
    assert result.exit_code == 0

    path = redirected_home / ".asrbench" / "config.toml"
    body = path.read_text(encoding="utf-8")
    assert "port = 9000" in body


def test_set_rejects_non_dotted_key(redirected_home: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["set", "port", "9000"])
    assert result.exit_code == 2


def test_show_errors_without_config(redirected_home: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 1
