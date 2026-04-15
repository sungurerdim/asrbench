"""
Tests for the Tier 1 + Tier 2 preprocessing extensions:

    - silence_threshold_db / silence_min_duration_s  (vad_trim knobs)
    - lufs_lra / loudnorm_linear                     (normalize_lufs knobs)

Verifies that the pipeline forwards the new keys, that the filter
implementations honor them, and that default values preserve the
backward-compatible behavior of the original bool-only / single-lufs API.
"""

from __future__ import annotations

import numpy as np
import pytest

from asrbench.preprocessing.loudness import normalize_lufs
from asrbench.preprocessing.pipeline import PreprocessingPipeline
from asrbench.preprocessing.vad import vad_trim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal_with_silence_gap(sr: int = 16_000) -> np.ndarray:
    """2 s tone, 2 s silence, 2 s tone → 6 s total."""
    t = np.arange(0, 2.0, 1.0 / sr, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    silence = np.zeros(2 * sr, dtype=np.float32)
    return np.concatenate([tone, silence, tone])


def _make_tone(duration_s: float = 1.0, sr: int = 16_000, amp: float = 0.3) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# vad_trim — Tier 1
# ---------------------------------------------------------------------------


class TestVadTrimKnobs:
    def test_default_args_preserve_existing_behavior(self) -> None:
        """No kwargs → same behavior as the pre-refactor vad_trim."""
        signal = _make_signal_with_silence_gap()
        trimmed = vad_trim(signal)
        # Some silence inside should be collapsed → output < input.
        assert len(trimmed) < len(signal)

    def test_conservative_threshold_keeps_more_audio(self) -> None:
        """Threshold -60 dBFS classifies fewer frames as silence."""
        signal = _make_signal_with_silence_gap()
        aggressive = vad_trim(signal, threshold_dbfs=-30.0, max_silence_s=0.1)
        conservative = vad_trim(signal, threshold_dbfs=-60.0, max_silence_s=0.1)
        assert len(conservative) >= len(aggressive)

    def test_long_max_silence_keeps_more_silence(self) -> None:
        """Larger max_silence_s means less collapsing of interior gaps."""
        signal = _make_signal_with_silence_gap()
        short_silence = vad_trim(signal, threshold_dbfs=-40.0, max_silence_s=0.1)
        long_silence = vad_trim(signal, threshold_dbfs=-40.0, max_silence_s=1.5)
        assert len(long_silence) >= len(short_silence)

    def test_pipeline_forwards_silence_params(self) -> None:
        """PreprocessingPipeline.apply passes silence knobs to vad_trim."""
        signal = _make_signal_with_silence_gap()
        # Run pipeline twice with different silence settings; results must differ.
        aggressive_params = {
            "vad_trim": True,
            "silence_threshold_db": -30.0,
            "silence_min_duration_s": 0.1,
        }
        conservative_params = {
            "vad_trim": True,
            "silence_threshold_db": -55.0,
            "silence_min_duration_s": 1.5,
        }
        out_aggressive = PreprocessingPipeline.apply(signal, aggressive_params)
        out_conservative = PreprocessingPipeline.apply(signal, conservative_params)
        assert len(out_aggressive) != len(out_conservative)

    def test_vad_trim_false_ignores_silence_params(self) -> None:
        """When vad_trim=false, silence params are no-ops (audio unchanged)."""
        signal = _make_signal_with_silence_gap()
        params = {
            "vad_trim": False,
            "silence_threshold_db": -30.0,
            "silence_min_duration_s": 0.1,
        }
        out = PreprocessingPipeline.apply(signal, params)
        np.testing.assert_array_equal(out, signal)


# ---------------------------------------------------------------------------
# normalize_lufs — Tier 2
# ---------------------------------------------------------------------------


class TestLoudnormKnobs:
    def test_linear_and_dynamic_modes_both_reach_target(self) -> None:
        """Both modes should land near the target integrated loudness."""
        pyln = pytest.importorskip("pyloudnorm")
        audio = _make_tone(duration_s=2.0, amp=0.1)
        target = -16.0

        linear_out = normalize_lufs(audio, target, linear=True)
        dynamic_out = normalize_lufs(audio, target, linear=False, lra=7.0)

        meter = pyln.Meter(16_000)
        assert meter.integrated_loudness(linear_out) == pytest.approx(target, abs=0.5)
        # Dynamic mode's soft compression may shift the integrated LUFS
        # slightly; allow a wider tolerance.
        assert meter.integrated_loudness(dynamic_out) == pytest.approx(target, abs=1.5)

    def test_dynamic_mode_attenuates_transients(self) -> None:
        """Dynamic mode should reduce peaks more than linear mode."""
        pytest.importorskip("pyloudnorm")
        sr = 16_000
        # Steady tone with a big transient burst in the middle.
        base = _make_tone(duration_s=2.0, amp=0.05, sr=sr)
        burst = np.zeros_like(base)
        mid = len(burst) // 2
        burst[mid : mid + sr // 10] = 0.9
        audio = base + burst

        target = -16.0
        linear_out = normalize_lufs(audio, target, linear=True)
        dynamic_out = normalize_lufs(audio, target, linear=False, lra=5.0)

        linear_peak = float(np.max(np.abs(linear_out)))
        dynamic_peak = float(np.max(np.abs(dynamic_out)))
        # Dynamic mode tames the transient; peak should be lower.
        assert dynamic_peak <= linear_peak + 1e-6

    def test_pipeline_forwards_lufs_params(self) -> None:
        """PreprocessingPipeline.apply passes lra + linear to normalize_lufs."""
        pytest.importorskip("pyloudnorm")
        audio = _make_tone(duration_s=2.0, amp=0.1)
        linear_params = {
            "lufs_target": -16.0,
            "lufs_lra": 7.0,
            "loudnorm_linear": True,
        }
        dynamic_params = {
            "lufs_target": -16.0,
            "lufs_lra": 5.0,
            "loudnorm_linear": False,
        }
        out_linear = PreprocessingPipeline.apply(audio, linear_params)
        out_dynamic = PreprocessingPipeline.apply(audio, dynamic_params)
        # Two distinct modes must produce distinct output.
        assert not np.allclose(out_linear, out_dynamic)


# ---------------------------------------------------------------------------
# default_params() contract
# ---------------------------------------------------------------------------


def test_default_params_include_new_keys() -> None:
    """default_params must expose every new knob so IAMS can overlay them."""
    defaults = PreprocessingPipeline.default_params()
    assert "silence_threshold_db" in defaults
    assert "silence_min_duration_s" in defaults
    assert "lufs_lra" in defaults
    assert "loudnorm_linear" in defaults
    # Backward-compat: the no-op defaults keep the pipeline as identity when
    # combined with their gating flags.
    assert defaults["vad_trim"] is False
    assert defaults["lufs_target"] is None
    assert defaults["loudnorm_linear"] is False
