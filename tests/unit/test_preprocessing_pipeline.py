"""Tests for asrbench.preprocessing — individual steps and pipeline orchestrator."""

from __future__ import annotations

import numpy as np

from asrbench.preprocessing.filters import apply_drc, apply_highpass
from asrbench.preprocessing.pipeline import PreprocessingPipeline
from asrbench.preprocessing.resample import simulate_resample
from asrbench.preprocessing.vad import vad_trim

SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq: float, duration_s: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate a mono float32 sine wave."""
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float = 1.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


# ---------------------------------------------------------------------------
# Pipeline — identity / no-op defaults
# ---------------------------------------------------------------------------


class TestPipelineIdentity:
    """With default (no-op) params the pipeline must return audio unchanged."""

    def test_default_params_identity(self) -> None:
        audio = _sine(440.0)
        result = PreprocessingPipeline.apply(audio, PreprocessingPipeline.default_params(), SR)
        np.testing.assert_array_equal(result, audio)

    def test_empty_params_identity(self) -> None:
        audio = _sine(440.0)
        result = PreprocessingPipeline.apply(audio, {}, SR)
        np.testing.assert_array_equal(result, audio)

    def test_default_params_keys(self) -> None:
        defaults = PreprocessingPipeline.default_params()
        expected_keys = {
            "format",
            "sample_rate",
            "lufs_target",
            "highpass_hz",
            "drc_ratio",
            "noise_reduce",
            "vad_trim",
        }
        assert set(defaults.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Highpass filter
# ---------------------------------------------------------------------------


class TestHighpass:
    def test_noop_when_zero(self) -> None:
        audio = _sine(100.0)
        result = apply_highpass(audio, 0, SR)
        np.testing.assert_array_equal(result, audio)

    def test_attenuates_low_freq(self) -> None:
        low = _sine(50.0, duration_s=0.5)
        filtered = apply_highpass(low, 200, SR)
        # Energy should drop significantly
        assert np.sqrt(np.mean(filtered**2)) < 0.1 * np.sqrt(np.mean(low**2))

    def test_preserves_high_freq(self) -> None:
        high = _sine(4000.0, duration_s=0.5)
        filtered = apply_highpass(high, 200, SR)
        # RMS should stay within 90% of original
        rms_orig = np.sqrt(np.mean(high**2))
        rms_filt = np.sqrt(np.mean(filtered**2))
        assert rms_filt > 0.9 * rms_orig


# ---------------------------------------------------------------------------
# DRC
# ---------------------------------------------------------------------------


class TestDRC:
    def test_noop_when_ratio_one(self) -> None:
        audio = _sine(440.0)
        result = apply_drc(audio, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_compresses_dynamic_range(self) -> None:
        loud = _sine(440.0, duration_s=0.5) * 0.9
        quiet = _sine(440.0, duration_s=0.5) * 0.01
        audio = np.concatenate([loud, quiet]).astype(np.float32)

        compressed = apply_drc(audio, ratio=4.0, threshold_db=-20.0)

        rms_loud_before = np.sqrt(np.mean(loud**2))
        rms_loud_after = np.sqrt(np.mean(compressed[: len(loud)] ** 2))
        # Loud part should be attenuated
        assert rms_loud_after < rms_loud_before


# ---------------------------------------------------------------------------
# Resample simulation
# ---------------------------------------------------------------------------


class TestResample:
    def test_noop_same_rate(self) -> None:
        audio = _sine(440.0)
        result = simulate_resample(audio, SR, SR)
        np.testing.assert_array_equal(result, audio)

    def test_roundtrip_preserves_shape(self) -> None:
        audio = _sine(440.0, duration_s=0.5)
        result = simulate_resample(audio, 8000, SR)
        # Length may differ by a few samples due to resampling rounding
        assert abs(len(result) - len(audio)) <= 2

    def test_bandwidth_limitation(self) -> None:
        """7 kHz content should be removed after 8 kHz roundtrip (Nyquist 4 kHz)."""
        high = _sine(7000.0, duration_s=0.5)
        result = simulate_resample(high, 8000, SR)
        # Most energy gone after bandwidth limiting
        assert np.sqrt(np.mean(result**2)) < 0.1 * np.sqrt(np.mean(high**2))


# ---------------------------------------------------------------------------
# VAD trim
# ---------------------------------------------------------------------------


class TestVADTrim:
    def test_removes_leading_trailing_silence(self) -> None:
        speech = _sine(440.0, duration_s=0.5) * 0.5
        padded = np.concatenate(
            [
                _silence(1.0),
                speech,
                _silence(1.0),
            ]
        ).astype(np.float32)

        trimmed = vad_trim(padded, SR)
        # Should be much shorter — at least silence removed
        assert len(trimmed) < len(padded) * 0.7

    def test_short_segment_unchanged(self) -> None:
        short = _sine(440.0, duration_s=0.01)  # 160 samples at 16kHz
        result = vad_trim(short, SR)
        np.testing.assert_array_equal(result, short)

    def test_all_silence_unchanged(self) -> None:
        silence = _silence(1.0)
        result = vad_trim(silence, SR)
        np.testing.assert_array_equal(result, silence)


# ---------------------------------------------------------------------------
# Pipeline — with active steps
# ---------------------------------------------------------------------------


class TestPipelineActive:
    def test_highpass_via_pipeline(self) -> None:
        low = _sine(50.0, duration_s=0.5)
        params = {"highpass_hz": 200}
        result = PreprocessingPipeline.apply(low, params, SR)
        assert np.sqrt(np.mean(result**2)) < 0.1 * np.sqrt(np.mean(low**2))

    def test_vad_trim_via_pipeline(self) -> None:
        speech = _sine(440.0, duration_s=0.3) * 0.5
        padded = np.concatenate([_silence(0.5), speech, _silence(0.5)]).astype(np.float32)
        params = {"vad_trim": True}
        result = PreprocessingPipeline.apply(padded, params, SR)
        assert len(result) < len(padded) * 0.8

    def test_multiple_steps_compose(self) -> None:
        """Highpass + DRC should both take effect."""
        audio = _sine(50.0, duration_s=0.5) * 0.9
        params = {"highpass_hz": 200, "drc_ratio": 4.0}
        result = PreprocessingPipeline.apply(audio, params, SR)
        # Low freq attenuated → almost zero
        assert np.sqrt(np.mean(result**2)) < 0.1 * np.sqrt(np.mean(audio**2))
