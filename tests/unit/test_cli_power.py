"""Smoke tests for the ``asrbench power`` CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from asrbench.cli.power import app


def test_power_suggest_prints_eps() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["suggest", "--duration", "2400"])
    assert result.exit_code == 0
    assert "Suggested epsilon: 0.008" in result.stdout
    assert "6000 words" in result.stdout


def test_power_suggest_custom_wpm() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["suggest", "--duration", "900", "--wpm", "120", "--base-wer", "0.15"]
    )
    assert result.exit_code == 0
    assert "Suggested epsilon:" in result.stdout
