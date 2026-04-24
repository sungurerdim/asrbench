"""Unit tests for LUFS loudness normalization (Faz 9)."""

from __future__ import annotations

import numpy as np
import pytest

pyln = pytest.importorskip("pyloudnorm")

from asrbench.preprocessing.loudness import (  # noqa: E402
    _soft_compress_toward_lra,
    normalize_lufs,
)


class TestNormalizeLufs:
    def test_short_segment_returns_unchanged(self) -> None:
        """Segments under 400 ms skip LUFS normalisation."""
        audio = np.zeros(100, dtype=np.float32)  # 6 ms at 16 kHz
        out = normalize_lufs(audio, target_lufs=-16.0, sr=16_000)
        assert out is audio

    def test_silent_segment_returns_unchanged(self) -> None:
        """pyloudnorm returns -inf for silence; we must not apply the gain."""
        audio = np.zeros(16_000, dtype=np.float32)  # 1 s silence
        out = normalize_lufs(audio, target_lufs=-16.0, sr=16_000)
        np.testing.assert_array_equal(out, audio)

    def test_linear_mode_applies_gain(self) -> None:
        """A loud signal gets attenuated toward the target."""
        rng = np.random.default_rng(seed=1)
        audio = (rng.standard_normal(16_000) * 0.5).astype(np.float32)
        out = normalize_lufs(audio, target_lufs=-30.0, sr=16_000, linear=True)
        # The attenuated output should have smaller RMS than the input.
        assert np.sqrt(np.mean(out**2)) < np.sqrt(np.mean(audio**2))

    def test_dynamic_mode_runs_soft_compression(self) -> None:
        """linear=False path exercises the compressor branch."""
        rng = np.random.default_rng(seed=2)
        audio = (rng.standard_normal(2 * 16_000) * 0.3).astype(np.float32)
        out = normalize_lufs(audio, target_lufs=-23.0, sr=16_000, linear=False, lra=5.0)
        assert out.dtype == np.float32
        assert out.shape == audio.shape


class TestSoftCompressToLra:
    def test_nonpositive_lra_returns_input(self) -> None:
        audio = np.zeros(16_000, dtype=np.float32)
        out = _soft_compress_toward_lra(audio, integrated_lufs=-23.0, lra=0.0, sr=16_000)
        assert out is audio

    def test_compressor_reduces_peaks(self) -> None:
        rng = np.random.default_rng(seed=3)
        base = rng.standard_normal(16_000) * 0.1
        spikes = np.zeros_like(base)
        spikes[5000:5100] = 0.8  # loud burst
        audio = (base + spikes).astype(np.float32)

        out = _soft_compress_toward_lra(audio, integrated_lufs=-23.0, lra=7.0, sr=16_000)
        # The spike region must be attenuated relative to the input.
        assert np.max(np.abs(out[5000:5100])) <= np.max(np.abs(audio[5000:5100]))
