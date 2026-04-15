"""LUFS loudness normalization (EBU R128)."""

from __future__ import annotations

import numpy as np

# Minimum segment length (seconds) for reliable LUFS measurement.
_MIN_DURATION_S = 0.4


def normalize_lufs(
    audio: np.ndarray,
    target_lufs: float,
    sr: int = 16_000,
    *,
    lra: float = 7.0,
    linear: bool = False,
) -> np.ndarray:
    """Normalize *audio* to *target_lufs* integrated loudness.

    Mirrors the ``loudnorm`` FFmpeg filter. ``lra`` maps to FFmpeg's target
    Loudness Range, ``linear`` toggles linear (single-pass, stats-based) vs
    dynamic (two-pass compressor) mode — same semantics as the ``linear=true``
    flag on FFmpeg loudnorm.

    pyloudnorm is a single-pass gain application; it does not implement the
    full two-pass dynamic compressor FFmpeg uses. Therefore:

    - ``linear=True`` → pyloudnorm's native mode (pure gain to reach the
      target loudness). Transients preserved.
    - ``linear=False`` → approximate dynamic mode by soft-compressing the
      signal around the target LRA before applying the gain. This is an
      approximation; for byte-identical mobile parity use the FFmpeg
      preprocessing backend instead.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    target_lufs:
        Target integrated loudness in LUFS (e.g. ``-16.0``).
    sr:
        Sample rate of *audio*.
    lra:
        Target loudness range in LU. Only consulted when ``linear=False``.
        Default ``7.0`` (FFmpeg loudnorm default).
    linear:
        When True, apply a single-pass gain shift and leave dynamics alone.
        When False, soft-compress toward the ``lra`` target first. Default
        False to mirror FFmpeg ``loudnorm`` out-of-the-box behavior.

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

    if not linear:
        # Dynamic mode approximation: soft-compress transients whose short-term
        # LUFS stray more than lra/2 above the integrated level before the gain
        # step. This is a cheap pre-shaping; true two-pass loudnorm needs the
        # FFmpeg backend.
        audio = _soft_compress_toward_lra(audio, current_lufs, lra, sr)

    return pyln.normalize.loudness(audio, current_lufs, target_lufs)


def _soft_compress_toward_lra(
    audio: np.ndarray, integrated_lufs: float, lra: float, sr: int
) -> np.ndarray:
    """
    Tame transients that stick out more than ``lra/2`` LU above the integrated
    loudness. Uses a per-sample smooth envelope follower with the same 400 ms
    window EBU R128 short-term measurements use.

    Not a true two-pass loudnorm — for that use the FFmpeg backend. This is a
    best-effort scipy-only approximation that keeps WER parity close enough for
    IAMS screening on typical spoken content.
    """
    if lra <= 0:
        return audio

    # Envelope: 400 ms RMS window
    win = max(1, int(0.4 * sr))
    squared = audio.astype(np.float64) ** 2
    # Running mean via cumulative sum for O(n)
    cs = np.cumsum(squared)
    cs = np.concatenate(([0.0], cs))
    envelope = (cs[win:] - cs[:-win]) / win
    envelope = np.sqrt(np.maximum(envelope, 1e-20))
    envelope_full = np.interp(
        np.arange(len(audio)), np.linspace(0, len(audio) - 1, len(envelope)), envelope
    )

    # Convert envelope to approximate short-term LUFS (dBFS + 3)
    env_lufs = 20.0 * np.log10(envelope_full + 1e-10) + 3.0
    ceiling = integrated_lufs + lra / 2.0
    over = env_lufs - ceiling
    # Linear-in-dB soft knee above the ceiling: each dB over reduced to 0.5 dB.
    gain_db = np.where(over > 0, -0.5 * over, 0.0)
    gain_lin = 10.0 ** (gain_db / 20.0)
    return (audio * gain_lin.astype(np.float32)).astype(np.float32)
