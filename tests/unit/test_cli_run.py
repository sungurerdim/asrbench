"""Unit tests for `asrbench run` CLI (Faz 9)."""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from asrbench.cli.run import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _mock_httpx(monkeypatch: pytest.MonkeyPatch, *, response=None, raise_exc=None):
    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def get(self, *_a, **_kw):
            if raise_exc:
                raise raise_exc
            return response

        def post(self, *_a, **_kw):
            if raise_exc:
                raise raise_exc
            return response

        def delete(self, *_a, **_kw):
            if raise_exc:
                raise raise_exc
            return response

    monkeypatch.setattr(httpx, "Client", _FakeClient)


def _resp(status_code: int, body) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("GET", "http://test"),
    )


class TestRunStart:
    def test_happy_path(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(
            monkeypatch,
            response=_resp(202, {"run_id": "r-1", "status": "pending"}),
        )
        result = runner.invoke(app, ["start", "--model", "m-1", "--dataset", "d-1", "--lang", "en"])
        assert result.exit_code == 0
        assert "r-1" in result.output

    def test_invalid_params_json_exits_2(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bad --params JSON exits with code 2 before hitting the server."""
        _mock_httpx(monkeypatch, response=_resp(200, {}))
        result = runner.invoke(
            app,
            [
                "start",
                "--model",
                "m-1",
                "--dataset",
                "d-1",
                "--params",
                "{not json}",
            ],
        )
        assert result.exit_code == 2
        assert "invalid --params JSON" in result.output

    def test_connection_error(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("offline"))
        result = runner.invoke(app, ["start", "--model", "m-1", "--dataset", "d-1"])
        assert result.exit_code == 1


class TestRunStatus:
    def test_status_renders_payload(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "run_id": "r-1",
            "status": "completed",
            "lang": "en",
            "aggregate": {
                "wer_mean": 0.12,
                "cer_mean": 0.08,
                "rtfx_mean": 5.3,
                "wall_time_s": 30.0,
            },
        }
        _mock_httpx(monkeypatch, response=_resp(200, payload))
        result = runner.invoke(app, ["status", "r-1"])
        assert result.exit_code == 0
        assert "completed" in result.output

    def test_status_connection_error(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("offline"))
        result = runner.invoke(app, ["status", "r-1"])
        assert result.exit_code == 1


class TestRunList:
    def test_empty(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(monkeypatch, response=_resp(200, []))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

    def test_connection_error(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("offline"))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 1
