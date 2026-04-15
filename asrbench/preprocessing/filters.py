"""Frequency-domain filters, compressor, and peak limiter."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos


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


def apply_lowpass(audio: np.ndarray, cutoff_hz: int, sr: int = 16_000) -> np.ndarray:
    """Apply a 4th-order Butterworth lowpass filter.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    cutoff_hz:
        Cutoff frequency in Hz.  ``0`` disables the filter (no-op).
        A cutoff at or above Nyquist is treated as no-op (identity).
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
        return audio  # identity: no content above Nyquist to remove

    sos = butter(4, cutoff_hz / nyquist, btype="low", output="sos")
    filtered = np.asarray(sosfilt(sos, audio))
    return filtered.astype(np.float32)


def apply_notch(
    audio: np.ndarray,
    freq_hz: int,
    sr: int = 16_000,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply an IIR notch filter to suppress a single narrow frequency band.

    Typical ASR use is mains hum removal (50 Hz in EU/TR, 60 Hz in US/JP).

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    freq_hz:
        Center frequency to notch out (Hz). ``0`` disables the filter.
    sr:
        Sample rate of *audio*.
    quality:
        Quality factor (Q). Higher = narrower notch. 30 is a safe default
        that removes hum without affecting neighboring speech content.

    Returns
    -------
    np.ndarray
        Filtered float32 waveform.
    """
    if freq_hz <= 0:
        return audio

    nyquist = sr / 2.0
    if freq_hz >= nyquist:
        return audio

    b, a = iirnotch(freq_hz, quality, sr)
    sos = tf2sos(b, a)
    filtered = np.asarray(sosfilt(sos, audio))
    return filtered.astype(np.float32)


def apply_preemphasis(audio: np.ndarray, coef: float) -> np.ndarray:
    """Apply a first-order FIR pre-emphasis filter: ``y[n] = x[n] − α·x[n−1]``.

    Classical ASR feature-extraction front-end (Kaldi/HTK/ESPnet). Boosts
    high frequencies to compensate for the −6 dB/oct glottal source roll-off.
    Whisper skips this in its log-Mel extractor, so applying it upstream is
    an independent knob IAMS can probe.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    coef:
        Pre-emphasis coefficient α ∈ [0, 1). ``0.0`` disables the filter.
        Standard values: 0.95–0.97.

    Returns
    -------
    np.ndarray
        Filtered float32 waveform (same length as input).
    """
    if coef <= 0.0:
        return audio

    out = np.empty_like(audio)
    out[0] = audio[0]
    out[1:] = audio[1:] - coef * audio[:-1]
    return out.astype(np.float32)


def apply_limiter(audio: np.ndarray, ceiling_db: float) -> np.ndarray:
    """Soft tanh-based peak limiter (alimiter equivalent).

    Uses ``y = C·tanh(x / C)`` where C = 10^(ceiling_db / 20). Samples well
    below the ceiling pass through almost linearly (tanh is ≈ identity near
    zero); samples approaching the ceiling are smoothly saturated without
    introducing hard-clip harmonics.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    ceiling_db:
        Output ceiling in dBFS (e.g. ``-1.0`` for a −1 dB ceiling).
        ``0.0`` or positive values disable the limiter (no-op).

    Returns
    -------
    np.ndarray
        Limited float32 waveform.
    """
    if ceiling_db >= 0.0:
        return audio

    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    return (ceiling_lin * np.tanh(audio / ceiling_lin)).astype(np.float32)


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
