"""LUFS loudness normalization (EBU R128)."""

from __future__ import annotations

import numpy as np

# Minimum segment length (seconds) for reliable LUFS measurement.
_MIN_DURATION_S = 0.4


def normalize_lufs(audio: np.ndarray, target_lufs: float, sr: int = 16_000) -> np.ndarray:
    """Normalize *audio* to *target_lufs* integrated loudness.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    target_lufs:
        Target integrated loudness in LUFS (e.g. ``-16.0``).
    sr:
        Sample rate of *audio*.

    Returns
    -------
    np.ndarray
        Loudness-normalized float32 waveform.  Returned unchanged when
        the segment is too short or effectively silent.
    """
    duration_s = len(audio) / sr
    if duration_s < _MIN_DURATION_S:
        return audio

    import pyloudnorm as pyln  # type: ignore[import-not-found]

    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)

    # pyloudnorm returns -inf for silence
    if not np.isfinite(current_lufs):
        return audio

    return pyln.normalize.loudness(audio, current_lufs, target_lufs)
