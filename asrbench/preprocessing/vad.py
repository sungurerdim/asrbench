"""Energy-based voice activity detection and silence trimming."""

from __future__ import annotations

import numpy as np

# Defaults — tuned for typical ASR segments.
_THRESHOLD_DBFS = -40.0
_MAX_SILENCE_S = 0.5


def vad_trim(
    audio: np.ndarray,
    sr: int = 16_000,
    *,
    threshold_dbfs: float = _THRESHOLD_DBFS,
    max_silence_s: float = _MAX_SILENCE_S,
) -> np.ndarray:
    """Remove leading/trailing silence and collapse long internal pauses.

    Uses a simple energy-based approach with no external model dependency.
    Mirrors the knobs of FFmpeg ``silenceremove`` so ``threshold_dbfs`` /
    ``max_silence_s`` map to ``silenceremove_threshold_db`` /
    ``silenceremove_duration_s`` in mobile deployments.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    sr:
        Sample rate of *audio*.
    threshold_dbfs:
        Frame RMS level (dBFS) below which a frame is classified as
        silence. Lower (more negative) = more conservative trim. Default
        ``-40`` dBFS.
    max_silence_s:
        Maximum contiguous silence (seconds) kept inside a trimmed segment.
        Silence runs longer than this are collapsed. Default ``0.5`` s.

    Returns
    -------
    np.ndarray
        Trimmed float32 waveform.  If the entire signal is below the
        energy threshold the original *audio* is returned unchanged.
    """
    frame_len = int(sr * 0.02)  # 20 ms frames
    if len(audio) < frame_len:
        return audio

    threshold_lin = 10.0 ** (threshold_dbfs / 20.0)
    max_silence_frames = max(1, int(max_silence_s / 0.02))

    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    is_speech = rms > threshold_lin

    if not np.any(is_speech):
        return audio

    # Find first / last speech frame
    speech_indices = np.nonzero(is_speech)[0]
    first_speech = int(speech_indices[0])
    last_speech = int(speech_indices[-1])

    # Walk through the interior and collapse long silence gaps
    kept: list[np.ndarray] = []
    silence_run = 0

    for i in range(first_speech, last_speech + 1):
        if is_speech[i]:
            silence_run = 0
            kept.append(frames[i])
        else:
            silence_run += 1
            if silence_run <= max_silence_frames:
                kept.append(frames[i])

    # Include any partial tail after the last full frame
    tail_start = (last_speech + 1) * frame_len
    if tail_start < len(audio):
        remaining = audio[tail_start:]
        remaining_rms = np.sqrt(np.mean(remaining**2))
        if remaining_rms > threshold_lin:
            kept.append(remaining)

    if not kept:
        return audio

    return np.concatenate(kept)
