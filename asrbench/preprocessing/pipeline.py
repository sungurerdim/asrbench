"""Preprocessing orchestrator — applies all steps in a fixed, signal-theory-correct order."""

from __future__ import annotations

from typing import Any

import numpy as np

from asrbench.preprocessing.codec import simulate_codec
from asrbench.preprocessing.filters import apply_drc, apply_highpass
from asrbench.preprocessing.loudness import normalize_lufs
from asrbench.preprocessing.noise import reduce_noise
from asrbench.preprocessing.resample import simulate_resample
from asrbench.preprocessing.vad import vad_trim


class PreprocessingPipeline:
    """Stateless orchestrator that applies preprocessing steps in a fixed order.

    Order (signal-processing convention):
        1. Codec simulation  — lossy artifact injection
        2. Resample simulation — bandwidth limitation
        3. LUFS normalization — consistent loudness
        4. Highpass filter    — rumble / hum removal
        5. DRC               — dynamic range control
        6. Noise reduction    — spectral denoising
        7. VAD trim           — silence removal

    With default (no-op) parameters every step is an identity transform.
    """

    @staticmethod
    def apply(
        audio: np.ndarray,
        params: dict[str, Any],
        sr: int = 16_000,
    ) -> np.ndarray:
        """Run the full preprocessing chain on *audio*.

        Parameters
        ----------
        audio:
            Mono float32 waveform.
        params:
            Preprocessing parameters (without the ``preprocess.`` prefix).
            Missing keys fall back to no-op defaults.
        sr:
            Sample rate of *audio*.

        Returns
        -------
        np.ndarray
            Processed float32 waveform.
        """
        # 1. Codec simulation
        fmt = params.get("format", "none")
        if fmt != "none":
            audio = simulate_codec(audio, fmt, sr)

        # 2. Resample simulation
        target_sr = params.get("sample_rate", sr)
        if target_sr != sr:
            audio = simulate_resample(audio, target_sr, sr)

        # 3. LUFS normalization
        lufs_target = params.get("lufs_target")
        if lufs_target is not None:
            audio = normalize_lufs(audio, lufs_target, sr)

        # 4. Highpass filter
        highpass_hz = params.get("highpass_hz", 0)
        if highpass_hz > 0:
            audio = apply_highpass(audio, highpass_hz, sr)

        # 5. Dynamic range compression
        drc_ratio = params.get("drc_ratio", 1.0)
        if drc_ratio > 1.0:
            audio = apply_drc(audio, drc_ratio)

        # 6. Noise reduction
        noise_strength = params.get("noise_reduce", 0.0)
        if noise_strength > 0.0:
            audio = reduce_noise(audio, noise_strength, sr)

        # 7. VAD trim
        if params.get("vad_trim", False):
            audio = vad_trim(audio, sr)

        return audio

    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return no-op default parameters (identity transform)."""
        return {
            "format": "none",
            "sample_rate": 16_000,
            "lufs_target": None,
            "highpass_hz": 0,
            "drc_ratio": 1.0,
            "noise_reduce": 0.0,
            "vad_trim": False,
        }
