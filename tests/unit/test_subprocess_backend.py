"""Unit tests for SubprocessBackend — contract-level coverage without spawning.

The real subprocess path is exercised in Phase-7 integration smoke tests that
actually spin up a worker; here we focus on the pure-python contract surface:

- ``family`` is derived from the resolved backend class's entry point metadata
  instead of being silently empty (the plan's subprocess_backend family fix).
- ``transcribe`` before ``load`` raises a clear RuntimeError.
- the audio-bytes wire format stays float32 and the response shape is
  materialised into :class:`Segment` objects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from asrbench.backends.base import BaseBackend, Segment
from asrbench.backends.subprocess_backend import (
    SubprocessBackend,
    _lookup_backend_family,
)


class _FakeBackend(BaseBackend):
    """Stand-in backend with a known family for entry-point lookup tests."""

    family = "fake-family"
    name = "fake"

    def default_params(self) -> dict:
        return {}

    def load(self, model_path: str, params: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        return []


class TestFamilyLookup:
    def test_family_read_from_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SubprocessBackend.family mirrors the resolved class's family attribute."""

        def fake_resolve(_name: str) -> type[BaseBackend]:
            return _FakeBackend

        monkeypatch.setattr(
            "asrbench.backends.subprocess_backend._resolve_backend_cls",
            fake_resolve,
        )
        backend = SubprocessBackend("any-name")
        assert backend.family == "fake-family"
        assert backend.name == "subprocess:any-name"

    def test_family_falls_back_to_empty_on_resolution_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing entry points degrade gracefully — family is empty, not an exception."""

        def raising_resolve(_name: str) -> type[BaseBackend]:
            raise RuntimeError("no such backend")

        monkeypatch.setattr(
            "asrbench.backends.subprocess_backend._resolve_backend_cls",
            raising_resolve,
        )
        assert _lookup_backend_family("missing") == ""

    def test_lookup_returns_string_for_non_string_family(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _QuirkyBackend(_FakeBackend):
            family = 42  # type: ignore[assignment]

        monkeypatch.setattr(
            "asrbench.backends.subprocess_backend._resolve_backend_cls",
            lambda _name: _QuirkyBackend,
        )
        result = _lookup_backend_family("quirky")
        assert isinstance(result, str)
        assert result == "42"


class TestTranscribeContract:
    def test_transcribe_before_load_raises(self) -> None:
        backend = SubprocessBackend.__new__(SubprocessBackend)
        backend._parent_conn = None
        backend._proc = None
        backend._backend_name = "anything"
        with pytest.raises(RuntimeError, match="called before load"):
            backend.transcribe(np.zeros(16, dtype=np.float32), "en", {})

    def test_transcribe_deserializes_worker_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ('result', [{...}, ...]) worker response materializes into Segment objects."""
        monkeypatch.setattr(
            "asrbench.backends.subprocess_backend._resolve_backend_cls",
            lambda _name: _FakeBackend,
        )
        backend = SubprocessBackend("fake")
        # Skip the real spawn; inject fakes so _send/_recv round-trip succeeds.
        fake_conn = MagicMock()
        fake_conn.recv.return_value = (
            "result",
            [
                {
                    "offset_s": 0.0,
                    "duration_s": 1.5,
                    "ref_text": "",
                    "hyp_text": "hello",
                },
                {
                    "offset_s": 1.5,
                    "duration_s": 1.5,
                    "ref_text": "",
                    "hyp_text": "world",
                },
            ],
        )
        backend._parent_conn = fake_conn
        backend._proc = SimpleNamespace(is_alive=lambda: True)

        audio = np.ones(32, dtype=np.float32) * 0.5
        segments = backend.transcribe(audio, "en", {"beam_size": 5})

        assert len(segments) == 2
        assert all(isinstance(s, Segment) for s in segments)
        assert segments[0].hyp_text == "hello"
        assert segments[1].offset_s == pytest.approx(1.5)

        # First send was ("transcribe", audio_bytes, lang, params).
        sent: Any = fake_conn.send.call_args.args[0]
        assert sent[0] == "transcribe"
        assert sent[2] == "en"
        # Audio must be serialised as float32 bytes so the worker can
        # np.frombuffer without surprises.
        assert isinstance(sent[1], (bytes, bytearray))
        assert len(sent[1]) == audio.nbytes
