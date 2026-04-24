"""Unit tests for `asrbench` top-level CLI and `optimize` (Faz 9)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestTopLevelApp:
    def test_version_flag_prints_version_and_exits(self, runner: CliRunner) -> None:
        from asrbench import __version__
        from asrbench.cli.app import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_root_app_registers_every_subcommand(self) -> None:
        """Smoke: the Typer app exposes every subcommand group.

        Introspect the registered ``typer`` groups rather than invoking
        the help screen — the rendered help depends on click internals
        that shift between versions.
        """
        from asrbench.cli.app import app

        group_names = {group.name for group in app.registered_groups}
        command_names = {cmd.name for cmd in app.registered_commands}
        expected_groups = {
            "config",
            "datasets",
            "models",
            "optimize",
            "power",
            "run",
            "compare",
            "doctor",
        }
        assert expected_groups.issubset(group_names)
        assert "serve" in command_names


class TestOptimizeCLI:
    def _mock_httpx(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        response: httpx.Response | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        class _FakeClient:
            def __init__(self, *_a, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def post(self, *_a, **_kw):
                if raise_exc:
                    raise raise_exc
                return response

        monkeypatch.setattr(httpx, "Client", _FakeClient)

    @staticmethod
    def _write_space(tmp_path: Path) -> Path:
        space = tmp_path / "space.yaml"
        space.write_text(
            "parameters:\n  beam_size:\n    type: int\n    min: 1\n    max: 10\n    default: 5\n",
            encoding="utf-8",
        )
        return space

    def test_missing_space_file_exits_2(self, tmp_path: Path, runner: CliRunner) -> None:
        from asrbench.cli.optimize import app

        result = runner.invoke(
            app,
            [
                "--model",
                "m-1",
                "--dataset",
                "d-1",
                "--space",
                str(tmp_path / "missing.yaml"),
            ],
        )
        assert result.exit_code == 2
        assert "space file not found" in result.output

    def test_invalid_mode_exits_2(self, tmp_path: Path, runner: CliRunner) -> None:
        from asrbench.cli.optimize import app

        space = self._write_space(tmp_path)
        result = runner.invoke(
            app,
            [
                "--model",
                "m-1",
                "--dataset",
                "d-1",
                "--space",
                str(space),
                "--mode",
                "bogus",
            ],
        )
        assert result.exit_code == 2
        assert "invalid --mode" in result.output

    def test_happy_path_single_metric(
        self, tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from asrbench.cli.optimize import app

        space = self._write_space(tmp_path)
        self._mock_httpx(
            monkeypatch,
            response=httpx.Response(
                status_code=202,
                json={"study_id": "s-1", "status": "running", "mode": "maximum", "hard_cap": 50},
                request=httpx.Request("POST", "http://t"),
            ),
        )
        result = runner.invoke(
            app,
            [
                "--model",
                "m-1",
                "--dataset",
                "d-1",
                "--space",
                str(space),
                "--budget",
                "50",
            ],
        )
        assert result.exit_code == 0
        assert "Study started: s-1" in result.output

    def test_weighted_objective_without_weights_exits_2(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        from asrbench.cli.optimize import app

        space = self._write_space(tmp_path)
        result = runner.invoke(
            app,
            [
                "--model",
                "m-1",
                "--dataset",
                "d-1",
                "--space",
                str(space),
                "--objective",
                "weighted",
            ],
        )
        assert result.exit_code == 2
        assert "--weights" in result.output

    def test_connection_error_exits_1(
        self, tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from asrbench.cli.optimize import app

        space = self._write_space(tmp_path)
        self._mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("offline"))
        result = runner.invoke(
            app,
            [
                "--model",
                "m-1",
                "--dataset",
                "d-1",
                "--space",
                str(space),
            ],
        )
        assert result.exit_code == 1
        assert "cannot reach asrbench" in result.output
