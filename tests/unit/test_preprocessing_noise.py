"""Unit tests for spectral noise reduction (Faz 9)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("noisereduce")

from asrbench.preprocessing.noise import reduce_noise


class TestReduceNoise:
    def test_zero_strength_returns_input(self) -> None:
        """strength == 0.0 short-circuits and returns the input unchanged."""
        audio = np.zeros(8_000, dtype=np.float32)
        out = reduce_noise(audio, strength=0.0, sr=16_000)
        assert out is audio

    def test_positive_strength_runs(self) -> None:
        """Non-zero strength invokes noisereduce and returns same-shape output."""
        rng = np.random.default_rng(seed=4)
        audio = (rng.standard_normal(16_000) * 0.1).astype(np.float32)
        out = reduce_noise(audio, strength=0.5, sr=16_000)
        assert out.shape == audio.shape
        assert out.dtype == np.float32
