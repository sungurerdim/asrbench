"""Highpass filter and dynamic range compression."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def apply_highpass(audio: np.ndarray, cutoff_hz: int, sr: int = 16_000) -> np.ndarray:
    """Apply a 4th-order Butterworth highpass filter.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    cutoff_hz:
        Cutoff frequency in Hz.  ``0`` disables the filter (no-op).
    sr:
        Sample rate of *audio*.

    Returns
    -------
    np.ndarray
        Filtered float32 waveform.
    """
    if cutoff_hz <= 0:
        return audio

    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        return np.zeros_like(audio)

    sos = butter(4, cutoff_hz / nyquist, btype="high", output="sos")
    # ``sosfilt`` has an overloaded type: when ``zi`` is omitted (our case)
    # it returns a single ndarray, but scipy's stubs still advertise the
    # tuple variant, so we wrap in ``np.asarray`` to narrow for pyright.
    filtered = np.asarray(sosfilt(sos, audio))
    return filtered.astype(np.float32)


def apply_drc(audio: np.ndarray, ratio: float, threshold_db: float = -20.0) -> np.ndarray:
    """Simple envelope-follower dynamic range compressor.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    ratio:
        Compression ratio (e.g. ``4.0`` means 4:1).
        ``1.0`` disables compression (no-op).
    threshold_db:
        Threshold in dBFS above which compression is applied.

    Returns
    -------
    np.ndarray
        Compressed float32 waveform.
    """
    if ratio <= 1.0:
        return audio

    threshold_lin = 10.0 ** (threshold_db / 20.0)
    envelope = np.abs(audio)

    gain = np.ones_like(audio)
    above = envelope > threshold_lin
    if not np.any(above):
        return audio

    overshoot = envelope[above] / threshold_lin
    compressed_overshoot = overshoot ** (1.0 / ratio)
    gain[above] = compressed_overshoot / overshoot

    return (audio * gain).astype(np.float32)
