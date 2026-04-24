"""Unit tests for `asrbench models` CLI (Faz 9)."""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from asrbench.cli.models import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _mock_httpx(monkeypatch: pytest.MonkeyPatch, *, response=None, raise_exc=None):
    class _FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, *_args, **_kwargs):
            if raise_exc:
                raise raise_exc
            return response

        def post(self, *_args, **_kwargs):
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


class TestModelsList:
    def test_empty_table(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(monkeypatch, response=_resp(200, []))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No models" in result.output

    def test_renders_rows(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(
            monkeypatch,
            response=_resp(
                200,
                [
                    {
                        "model_id": "m-1",
                        "name": "fw-large-v3",
                        "backend": "faster-whisper",
                        "local_path": "/models/fw",
                    }
                ],
            ),
        )
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "fw-large-v3" in result.output

    def test_connection_error_exits_nonzero(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("no server"))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 1


class TestModelsRegister:
    def test_happy_path(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(monkeypatch, response=_resp(201, {"model_id": "m-1", "name": "fw-large-v3"}))
        result = runner.invoke(
            app,
            [
                "register",
                "--family",
                "whisper",
                "--name",
                "fw-large-v3",
                "--backend",
                "faster-whisper",
                "--path",
                "/tmp/fw",
            ],
        )
        assert result.exit_code == 0
        assert "Registered: m-1" in result.output

    def test_http_error_exits_nonzero(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        err_resp = httpx.Response(
            status_code=422,
            text="validation",
            request=httpx.Request("POST", "http://test"),
        )
        exc = httpx.HTTPStatusError("x", request=err_resp.request, response=err_resp)
        _mock_httpx(monkeypatch, raise_exc=exc)
        result = runner.invoke(
            app,
            [
                "register",
                "--family",
                "whisper",
                "--name",
                "x",
                "--backend",
                "fw",
                "--path",
                "/bad",
            ],
        )
        assert result.exit_code == 1


class TestModelsLoadUnload:
    def test_load_happy_path(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(
            monkeypatch,
            response=_resp(
                200,
                {"model_id": "m-1", "vram_used_mb": 3000, "vram_total_mb": 16000},
            ),
        )
        result = runner.invoke(app, ["load", "m-1"])
        assert result.exit_code == 0
        assert "Loaded: m-1" in result.output

    def test_unload_happy_path(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_httpx(monkeypatch, response=_resp(200, {"model_id": "m-1"}))
        result = runner.invoke(app, ["unload", "m-1"])
        assert result.exit_code == 0
        assert "Unloaded: m-1" in result.output

    def test_unload_connection_error_exits_nonzero(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("no server"))
        result = runner.invoke(app, ["unload", "m-1"])
        assert result.exit_code == 1
