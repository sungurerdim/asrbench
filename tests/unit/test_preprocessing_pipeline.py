"""Tests for asrbench.preprocessing — individual steps and pipeline orchestrator."""

from __future__ import annotations

import numpy as np

from asrbench.preprocessing.filters import (
    apply_drc,
    apply_highpass,
    apply_limiter,
    apply_lowpass,
    apply_notch,
    apply_preemphasis,
)
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
            "backend",
            "format",
            "sample_rate",
            "notch_hz",
            "highpass_hz",
            "lowpass_hz",
            "lufs_target",
            "lufs_lra",
            "loudnorm_linear",
            "drc_ratio",
            "limiter_ceiling_db",
            "noise_reduce",
            "preemph_coef",
            "vad_trim",
            "silence_threshold_db",
            "silence_min_duration_s",
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
# Lowpass filter
# ---------------------------------------------------------------------------


class TestLowpass:
    def test_noop_when_zero(self) -> None:
        audio = _sine(440.0)
        result = apply_lowpass(audio, 0, SR)
        np.testing.assert_array_equal(result, audio)

    def test_noop_when_at_or_above_nyquist(self) -> None:
        audio = _sine(1000.0)
        # 8000 Hz at 16 kHz sample rate = Nyquist, treated as identity
        result = apply_lowpass(audio, 8000, SR)
        np.testing.assert_array_equal(result, audio)

    def test_attenuates_high_freq(self) -> None:
        high = _sine(7800.0, duration_s=0.5)
        filtered = apply_lowpass(high, 4000, SR)
        assert np.sqrt(np.mean(filtered**2)) < 0.1 * np.sqrt(np.mean(high**2))

    def test_preserves_low_freq(self) -> None:
        low = _sine(500.0, duration_s=0.5)
        filtered = apply_lowpass(low, 4000, SR)
        rms_orig = np.sqrt(np.mean(low**2))
        rms_filt = np.sqrt(np.mean(filtered**2))
        assert rms_filt > 0.9 * rms_orig

    def test_preserves_dtype(self) -> None:
        audio = _sine(500.0)
        result = apply_lowpass(audio, 4000, SR)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Notch filter
# ---------------------------------------------------------------------------


class TestNotch:
    def test_noop_when_zero(self) -> None:
        audio = _sine(440.0)
        result = apply_notch(audio, 0, SR)
        np.testing.assert_array_equal(result, audio)

    def test_removes_target_frequency(self) -> None:
        # Pure 50 Hz hum — should be suppressed
        hum = _sine(50.0, duration_s=1.0)
        filtered = apply_notch(hum, 50, SR, quality=30.0)
        # After transient, RMS should drop dramatically
        steady = filtered[SR // 2 :]  # skip first 0.5 s transient
        assert np.sqrt(np.mean(steady**2)) < 0.2 * np.sqrt(np.mean(hum**2))

    def test_preserves_offband_content(self) -> None:
        speech_band = _sine(1000.0, duration_s=0.5)
        filtered = apply_notch(speech_band, 50, SR, quality=30.0)
        rms_orig = np.sqrt(np.mean(speech_band**2))
        rms_filt = np.sqrt(np.mean(filtered**2))
        assert rms_filt > 0.95 * rms_orig

    def test_preserves_dtype(self) -> None:
        audio = _sine(440.0)
        result = apply_notch(audio, 50, SR)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Pre-emphasis
# ---------------------------------------------------------------------------


class TestPreemphasis:
    def test_noop_when_zero_coef(self) -> None:
        audio = _sine(440.0)
        result = apply_preemphasis(audio, 0.0)
        np.testing.assert_array_equal(result, audio)

    def test_first_sample_unchanged(self) -> None:
        audio = _sine(440.0, duration_s=0.1)
        result = apply_preemphasis(audio, 0.97)
        assert result[0] == audio[0]

    def test_boosts_high_frequency_energy(self) -> None:
        """High-freq sine should end up relatively louder than low-freq sine after pre-emph."""
        low = _sine(200.0, duration_s=0.5)
        high = _sine(4000.0, duration_s=0.5)
        low_p = apply_preemphasis(low, 0.97)
        high_p = apply_preemphasis(high, 0.97)
        # Ratio (high/low) should increase vs. unfiltered
        ratio_before = np.sqrt(np.mean(high**2)) / np.sqrt(np.mean(low**2))
        ratio_after = np.sqrt(np.mean(high_p**2)) / np.sqrt(np.mean(low_p**2))
        assert ratio_after > ratio_before

    def test_preserves_length_and_dtype(self) -> None:
        audio = _sine(440.0, duration_s=0.3)
        result = apply_preemphasis(audio, 0.97)
        assert len(result) == len(audio)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Peak limiter (alimiter equivalent)
# ---------------------------------------------------------------------------


class TestLimiter:
    def test_noop_when_zero_ceiling(self) -> None:
        audio = _sine(440.0) * 0.95
        result = apply_limiter(audio, 0.0)
        np.testing.assert_array_equal(result, audio)

    def test_noop_when_positive_ceiling(self) -> None:
        audio = _sine(440.0) * 0.95
        result = apply_limiter(audio, 0.5)  # above 0 dBFS
        np.testing.assert_array_equal(result, audio)

    def test_limits_peaks_below_ceiling(self) -> None:
        # Loud signal: peak = 0.95 (≈ -0.45 dBFS)
        loud = _sine(440.0, duration_s=0.2) * 0.95
        ceiling_db = -3.0
        ceiling_lin = 10.0 ** (ceiling_db / 20.0)  # ≈ 0.708
        limited = apply_limiter(loud, ceiling_db)
        # With tanh soft-limit, peaks stay strictly below the linear ceiling
        assert np.max(np.abs(limited)) <= ceiling_lin + 1e-6

    def test_preserves_quiet_content(self) -> None:
        """Samples well below the ceiling should be near-linear."""
        # Very quiet signal — tanh is ~linear for small arguments
        quiet = _sine(440.0, duration_s=0.2) * 0.05
        limited = apply_limiter(quiet, -1.0)
        # Correlation near 1.0, RMS change < 1%
        rms_orig = np.sqrt(np.mean(quiet**2))
        rms_lim = np.sqrt(np.mean(limited**2))
        assert abs(rms_lim - rms_orig) / rms_orig < 0.01

    def test_preserves_dtype(self) -> None:
        audio = _sine(440.0) * 0.95
        result = apply_limiter(audio, -1.0)
        assert result.dtype == np.float32


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

    def test_notch_via_pipeline(self) -> None:
        hum = _sine(50.0, duration_s=1.0)
        params = {"notch_hz": 50}
        result = PreprocessingPipeline.apply(hum, params, SR)
        steady = result[SR // 2 :]
        assert np.sqrt(np.mean(steady**2)) < 0.2 * np.sqrt(np.mean(hum**2))

    def test_lowpass_via_pipeline(self) -> None:
        high = _sine(7800.0, duration_s=0.5)
        params = {"lowpass_hz": 4000}
        result = PreprocessingPipeline.apply(high, params, SR)
        assert np.sqrt(np.mean(result**2)) < 0.1 * np.sqrt(np.mean(high**2))

    def test_limiter_via_pipeline(self) -> None:
        loud = _sine(440.0, duration_s=0.2) * 0.95
        params = {"limiter_ceiling_db": -3.0}
        result = PreprocessingPipeline.apply(loud, params, SR)
        ceiling_lin = 10.0 ** (-3.0 / 20.0)
        assert np.max(np.abs(result)) <= ceiling_lin + 1e-6

    def test_preemph_via_pipeline(self) -> None:
        audio = _sine(440.0, duration_s=0.3)
        params = {"preemph_coef": 0.97}
        result = PreprocessingPipeline.apply(audio, params, SR)
        # Length preserved, first sample identical
        assert len(result) == len(audio)
        assert result[0] == audio[0]
        # Mid-signal should differ from input
        assert not np.allclose(result[100:200], audio[100:200])

    def test_full_chain_all_steps_active(self) -> None:
        """All 11 pipeline steps active simultaneously should not crash or corrupt shape/dtype.

        LUFS normalization is gated on pyloudnorm being installed (optional dep).
        """
        import importlib.util

        has_pyln = importlib.util.find_spec("pyloudnorm") is not None
        has_noisereduce = importlib.util.find_spec("noisereduce") is not None

        speech = _sine(1000.0, duration_s=0.5) * 0.8
        padded = np.concatenate([_silence(0.3), speech, _silence(0.3)]).astype(np.float32)
        params = {
            "format": "none",  # simulate_codec supports 'none' as no-op
            "sample_rate": 16_000,
            "notch_hz": 50,
            "highpass_hz": 80,
            "lowpass_hz": 7500,
            "lufs_target": -16.0 if has_pyln else None,
            "drc_ratio": 2.0,
            "limiter_ceiling_db": -1.0,
            "noise_reduce": 0.2 if has_noisereduce else 0.0,
            "preemph_coef": 0.97,
            "vad_trim": True,
        }
        result = PreprocessingPipeline.apply(padded, params, SR)
        # VAD trim may shorten; dtype must stay float32; no NaN/Inf
        # (limiter ceiling is tested separately in test_limiter_via_pipeline — pre-emph
        #  and NR downstream of the limiter can re-inflate peaks, so asserting the
        #  ceiling on the full-chain output is not meaningful here.)
        assert result.dtype == np.float32
        assert len(result) > 0
        assert np.all(np.isfinite(result))
