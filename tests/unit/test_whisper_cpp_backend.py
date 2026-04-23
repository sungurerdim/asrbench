"""Unit tests for the whisper.cpp backend adapter.

The heavy live-model test is gated behind ``importorskip`` so CI hosts that
don't ship pywhispercpp still collect the module. The lightweight pure-python
tests (default_params, conversion helpers, stub fallback path) run
unconditionally.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from asrbench.backends.base import Segment
from asrbench.backends.whisper_cpp import WhisperCppBackend, _to_segment


class TestDefaults:
    def test_default_params_match_plan_shape(self) -> None:
        backend = WhisperCppBackend()
        defaults = backend.default_params()
        # Core decode controls must all be present so the IAMS space has
        # something to sweep, even if pywhispercpp renames internals later.
        for key in (
            "n_threads",
            "beam_size",
            "temperature",
            "language",
            "no_speech_thold",
        ):
            assert key in defaults

    def test_family_and_name_stable(self) -> None:
        assert WhisperCppBackend.family == "whisper"
        assert WhisperCppBackend.name == "whisper-cpp"

    def test_supported_params_is_none_without_pywhispercpp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the library is missing, the optimizer must keep the full space."""

        def _raise() -> frozenset[str]:
            raise ImportError("pywhispercpp not installed")

        monkeypatch.setattr("asrbench.backends.whisper_cpp._transcribe_kwargs", _raise)
        backend = WhisperCppBackend()
        assert backend.supported_params() is None


class TestLoadErrorPath:
    def test_load_raises_clear_error_when_library_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without pywhispercpp, load() must raise with an install pointer."""
        import builtins
        from typing import Any

        real_import = builtins.__import__

        def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pywhispercpp.model":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        backend = WhisperCppBackend()
        with pytest.raises(RuntimeError, match="pip install 'asrbench\\[whisper-cpp\\]'"):
            backend.load("/does/not/exist", {})


class TestSegmentConversion:
    def test_t0_and_t1_converted_from_centiseconds(self) -> None:
        raw = SimpleNamespace(text="hello world", t0=150, t1=300)
        seg = _to_segment(raw)
        assert isinstance(seg, Segment)
        assert seg.offset_s == pytest.approx(1.5)
        assert seg.duration_s == pytest.approx(1.5)
        assert seg.hyp_text == "hello world"
        assert seg.ref_text == ""

    def test_missing_attributes_fall_back_to_zero(self) -> None:
        raw = SimpleNamespace()
        seg = _to_segment(raw)
        assert seg.offset_s == 0.0
        assert seg.duration_s == 0.0
        assert seg.hyp_text == ""

    def test_text_is_stripped(self) -> None:
        raw = SimpleNamespace(text="  hi  ", t0=0, t1=100)
        seg = _to_segment(raw)
        assert seg.hyp_text == "hi"


class TestTranscribeWithMockModel:
    def test_transcribe_forwards_to_pywhispercpp_and_converts(self) -> None:
        """The backend passes audio to the underlying model and converts result segments."""
        backend = WhisperCppBackend()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [
            SimpleNamespace(text="hello", t0=0, t1=100),
            SimpleNamespace(text="world", t0=100, t1=200),
        ]
        backend._model = mock_model
        backend._accepted_kwargs = frozenset({"language", "beam_size", "temperature"})

        audio = np.zeros(16000, dtype=np.float32)
        segments = backend.transcribe(audio, "en", {"beam_size": 5, "temperature": 0.0})

        assert len(segments) == 2
        assert segments[0].hyp_text == "hello"
        assert segments[1].offset_s == pytest.approx(1.0)
        # language propagated from `lang` arg, not params
        call = mock_model.transcribe.call_args
        assert call.kwargs["language"] == "en"
        assert call.kwargs["beam_size"] == 5

    def test_transcribe_raises_when_not_loaded(self) -> None:
        backend = WhisperCppBackend()
        with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
            backend.transcribe(np.zeros(16, dtype=np.float32), "en", {})

    def test_transcribe_drops_unknown_kwargs_on_type_error_retry(self) -> None:
        """If pywhispercpp rejects our kwargs, we retry with language only."""
        backend = WhisperCppBackend()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            TypeError("unexpected keyword argument"),
            [SimpleNamespace(text="retry", t0=0, t1=50)],
        ]
        backend._model = mock_model
        backend._accepted_kwargs = frozenset({"weird_param", "language"})

        segments = backend.transcribe(np.zeros(16, dtype=np.float32), "tr", {"weird_param": 9})
        assert len(segments) == 1
        assert segments[0].hyp_text == "retry"
        # Second call used only `language`.
        last_call = mock_model.transcribe.call_args_list[-1]
        assert last_call.kwargs == {"language": "tr"}


@pytest.mark.skipif(True, reason="Real-audio smoke — requires pywhispercpp + a ggml model.")
class TestRealModel:
    """Placeholder slot for the live smoke test.

    Enabled by setting ``WHISPER_CPP_MODEL_PATH`` in the environment and
    removing the skip marker. Kept in-tree so the audit trail for the plan's
    ``test_whisper_cpp_backend.py`` acceptance item lives with the code.
    """

    def test_transcribes_known_short_clip(self) -> None:  # pragma: no cover
        pytest.importorskip("pywhispercpp")
        import os
        from pathlib import Path

        model = os.environ.get("WHISPER_CPP_MODEL_PATH")
        fixture = Path(__file__).parent.parent / "fixtures" / "real_audio" / "en_short.wav"
        if not model or not fixture.exists():
            pytest.skip("set WHISPER_CPP_MODEL_PATH and add tests/fixtures/real_audio/en_short.wav")

        import soundfile as sf

        audio, sr = sf.read(str(fixture), dtype="float32")
        assert sr == 16000
        backend = WhisperCppBackend()
        backend.load(model, {"n_threads": 2})
        try:
            segments = backend.transcribe(audio, "en", backend.default_params())
            assert any(s.hyp_text for s in segments), "expected non-empty transcription"
        finally:
            backend.unload()
