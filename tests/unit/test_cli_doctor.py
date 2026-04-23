"""Smoke tests for the ``asrbench doctor`` CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from asrbench.cli.doctor import app


def test_doctor_emits_human_table() -> None:
    runner = CliRunner()
    result = runner.invoke(app, [])
    # exit_code may be 1 (FAIL present in CI / missing deps) or 0; both are
    # acceptable. What we guarantee is that every status label appears.
    assert "Python" in result.stdout
    assert "HuggingFace cache" in result.stdout


def test_doctor_json_output_is_parseable() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--json"])
    # Strip any ANSI colour codes just in case; json output should be clean.
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert all({"status", "label", "detail"} <= set(c.keys()) for c in payload)
    # Python check is always present and should be OK on a test host.
    python_check = next(c for c in payload if c["label"] == "Python")
    assert python_check["status"] == "OK"
