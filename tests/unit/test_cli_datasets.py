"""Unit tests for `asrbench datasets` CLI (Faz 9)."""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from asrbench.cli.datasets import app


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


def _make_response(status_code: int, body) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("GET", "http://test"),
    )


class TestDatasetsList:
    def test_prints_nothing_found_when_empty(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_httpx(monkeypatch, response=_make_response(200, []))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No datasets" in result.output

    def test_renders_dataset_table(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = [
            {
                "dataset_id": "ds-1",
                "name": "fleurs_en_test",
                "lang": "en",
                "split": "test",
                "duration_s": 3600,
                "verified": True,
            }
        ]
        _mock_httpx(monkeypatch, response=_make_response(200, payload))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "fleurs_en_test" in result.output

    def test_connection_error_exits_nonzero(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("no server"))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 1
        assert "Cannot connect" in result.output


class TestDatasetsFetch:
    def test_happy_path_prints_stream_url(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "dataset_id": "ds-2",
            "name": "common_voice_tr_test",
            "status": "downloading",
            "stream_url": "/ws/datasets/ds-2",
        }
        _mock_httpx(monkeypatch, response=_make_response(202, payload))
        result = runner.invoke(app, ["fetch", "common_voice", "--lang", "tr"])
        assert result.exit_code == 0
        assert "common_voice_tr_test" in result.output
        assert "/ws/datasets/ds-2" in result.output

    def test_http_error_exits_nonzero(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        err_resp = httpx.Response(
            status_code=422,
            text='{"detail": "bad source"}',
            request=httpx.Request("POST", "http://test"),
        )
        exc = httpx.HTTPStatusError("boom", request=err_resp.request, response=err_resp)
        _mock_httpx(monkeypatch, raise_exc=exc)
        result = runner.invoke(app, ["fetch", "not_a_source"])
        assert result.exit_code == 1
        assert "422" in result.output

    def test_connection_error_exits_nonzero(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_httpx(monkeypatch, raise_exc=httpx.ConnectError("no server"))
        result = runner.invoke(app, ["fetch", "fleurs"])
        assert result.exit_code == 1
        assert "Cannot connect" in result.output
