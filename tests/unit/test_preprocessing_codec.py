"""Unit tests for codec simulation preprocessing (Faz 9)."""

from __future__ import annotations

import shutil
from unittest.mock import patch

import numpy as np
import pytest

from asrbench.preprocessing.codec import simulate_codec


class TestSimulateCodec:
    def test_format_none_returns_input_unchanged(self) -> None:
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        out = simulate_codec(audio, "none")
        assert out is audio  # short-circuits

    def test_unknown_format_raises_valueerror(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown codec"):
            simulate_codec(audio, "mp3_9999")

    def test_missing_ffmpeg_raises_runtimeerror(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        with patch.object(shutil, "which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                simulate_codec(audio, "opus_64k")

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
    def test_opus_roundtrip_preserves_length_order(self) -> None:
        """Opus roundtrip yields comparable output length.

        Codec sim is lossy so content changes, but the length should
        stay within ±20 % of the input.
        """
        rng = np.random.default_rng(seed=42)
        audio = rng.standard_normal(16_000).astype(np.float32) * 0.1
        out = simulate_codec(audio, "opus_64k", sr=16_000)
        assert out.dtype == np.float32
        assert abs(len(out) - len(audio)) <= len(audio) * 0.2

    def test_all_known_formats_recognised(self) -> None:
        """Every registered bitrate in _CODEC_BITRATES is accepted by the
        format validator."""
        from asrbench.preprocessing.codec import _CODEC_BITRATES

        audio = np.zeros(100, dtype=np.float32)
        with patch.object(shutil, "which", return_value=None):
            # Should pass format-validator but fail on the missing ffmpeg.
            for fmt in _CODEC_BITRATES:
                with pytest.raises(RuntimeError, match="ffmpeg not found"):
                    simulate_codec(audio, fmt)
