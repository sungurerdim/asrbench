"""Sample rate roundtrip simulation — bandwidth limitation via resample."""

from __future__ import annotations

import numpy as np
import soxr


def simulate_resample(audio: np.ndarray, target_sr: int, native_sr: int = 16_000) -> np.ndarray:
    """Resample audio down to *target_sr* and back up to *native_sr*.

    This simulates the bandwidth limitation caused by recording or
    transmitting at a lower sample rate (e.g. 8 kHz telephone).

    Parameters
    ----------
    audio:
        Mono float32 waveform at *native_sr*.
    target_sr:
        Intermediate sample rate.  If equal to *native_sr*, returns
        *audio* unchanged (no-op).
    native_sr:
        Original sample rate of *audio*.

    Returns
    -------
    np.ndarray
        Float32 waveform at *native_sr* after the roundtrip.
    """
    if target_sr == native_sr:
        return audio

    down = soxr.resample(audio, native_sr, target_sr)
    result: np.ndarray = soxr.resample(down, target_sr, native_sr)
    return result
