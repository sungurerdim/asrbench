"""Unit tests for FasterWhisperBackend — contract compliance and param surface."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

fw = pytest.importorskip("faster_whisper", reason="faster-whisper not installed")

# The two imports below are intentionally placed after importorskip so the
# module still loads cleanly in environments without the faster-whisper
# extra installed — importorskip raises pytest.skip before these run.
from asrbench.backends.base import Segment  # noqa: E402
from asrbench.backends.faster_whisper import FasterWhisperBackend  # noqa: E402


class TestDefaultParams:
    def test_returns_dict(self) -> None:
        backend = FasterWhisperBackend()
        params = backend.default_params()
        assert isinstance(params, dict)

    def test_expected_keys(self) -> None:
        backend = FasterWhisperBackend()
        expected = {
            # Beam / decoding
            "beam_size",
            "best_of",
            "patience",
            "length_penalty",
            "temperature",
            # Repetition / output constraints
            "repetition_penalty",
            "no_repeat_ngram_size",
            "suppress_blank",
            "without_timestamps",
            "max_initial_timestamp",
            "max_new_tokens",
            # Reliability thresholds
            "compression_ratio_threshold",
            "log_prob_threshold",
            "no_speech_threshold",
            "hallucination_silence_threshold",
            # Context / chunking
            "condition_on_previous_text",
            "prompt_reset_on_temperature",
            "chunk_length",
            # VAD
            "vad_filter",
            "vad_threshold",
            "vad_min_speech_duration_ms",
            "vad_max_speech_duration_s",
            "vad_min_silence_duration_ms",
            "vad_speech_pad_ms",
            # Inference mode
            "batch_size",
        }
        assert set(backend.default_params().keys()) == expected

    def test_value_types(self) -> None:
        backend = FasterWhisperBackend()
        params = backend.default_params()
        assert isinstance(params["beam_size"], int)
        assert isinstance(params["temperature"], float)
        assert isinstance(params["patience"], float)
        assert isinstance(params["repetition_penalty"], float)
        assert isinstance(params["no_repeat_ngram_size"], int)
        assert isinstance(params["compression_ratio_threshold"], float)
        assert isinstance(params["log_prob_threshold"], float)
        assert isinstance(params["no_speech_threshold"], float)
        assert isinstance(params["condition_on_previous_text"], bool)
        assert isinstance(params["vad_filter"], bool)

    def test_default_values(self) -> None:
        backend = FasterWhisperBackend()
        params = backend.default_params()
        assert params["beam_size"] == 5
        assert params["temperature"] == 0.0
        assert params["patience"] == 1.0
        assert params["vad_filter"] is False
        assert params["condition_on_previous_text"] is True


class TestClassAttributes:
    def test_family(self) -> None:
        assert FasterWhisperBackend.family == "whisper"

    def test_name(self) -> None:
        assert FasterWhisperBackend.name == "faster-whisper"


class TestTranscribeWithoutLoad:
    def test_raises_runtime_error(self) -> None:
        backend = FasterWhisperBackend()
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            backend.transcribe(audio, "en", {})


class TestUnloadIdempotent:
    def test_unload_without_load(self) -> None:
        backend = FasterWhisperBackend()
        backend.unload()  # should not raise

    def test_unload_twice(self) -> None:
        backend = FasterWhisperBackend()
        backend.unload()
        backend.unload()  # second call should not raise


class TestLoadErrors:
    def test_invalid_model_path(self) -> None:
        backend = FasterWhisperBackend()
        with pytest.raises(RuntimeError, match="Failed to load"):
            backend.load("/nonexistent/model/path", {})


class TestTranscribeIntegration:
    """Tests that exercise transcribe with a mocked WhisperModel."""

    def test_returns_segments(self) -> None:
        backend = FasterWhisperBackend()

        mock_seg = SimpleNamespace(start=0.0, end=1.5, text=" Hello world ")
        mock_info = SimpleNamespace(language="en", language_probability=0.99)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        backend._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = backend.transcribe(audio, "en", {"beam_size": 5})

        assert len(result) == 1
        assert isinstance(result[0], Segment)
        assert result[0].hyp_text == "Hello world"
        assert result[0].offset_s == 0.0
        assert result[0].duration_s == 1.5
        assert result[0].ref_text == ""

    def test_filters_load_params_from_transcribe(self) -> None:
        backend = FasterWhisperBackend()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], SimpleNamespace())

        backend._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        backend.transcribe(audio, "en", {"beam_size": 5, "device": "cuda", "compute_type": "int8"})

        call_kwargs = mock_model.transcribe.call_args[1]
        assert "device" not in call_kwargs
        assert "compute_type" not in call_kwargs
        assert call_kwargs["beam_size"] == 5
