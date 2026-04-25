"""Spectral noise reduction wrapper."""

from __future__ import annotations

import numpy as np


def reduce_noise(audio: np.ndarray, strength: float, sr: int = 16_000) -> np.ndarray:
    """Apply spectral-gating noise reduction.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    strength:
        Aggressiveness in ``(0.0, 1.0]``.  ``0.0`` disables noise
        reduction (no-op).
    sr:
        Sample rate of *audio*.

    Returns
    -------
    np.ndarray
        Denoised float32 waveform.
    """
    if strength <= 0.0:
        return audio

    import noisereduce as nr

    denoised: np.ndarray = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=np.clip(strength, 0.0, 1.0),
        stationary=True,
    ).astype(np.float32)
    return denoised
