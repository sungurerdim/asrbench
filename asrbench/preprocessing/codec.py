"""Codec simulation — encode/decode roundtrip to inject lossy artifacts."""

from __future__ import annotations

import shutil
import struct
import subprocess

import numpy as np

_CODEC_BITRATES: dict[str, int] = {
    "opus_64k": 64_000,
    "opus_32k": 32_000,
    "opus_24k": 24_000,
}


def simulate_codec(audio: np.ndarray, format_tag: str, sr: int = 16_000) -> np.ndarray:
    """Encode audio to a lossy codec and decode back to float32.

    Parameters
    ----------
    audio:
        Mono float32 waveform, shape ``(n_samples,)``.
    format_tag:
        One of ``"none"``, ``"opus_64k"``, ``"opus_32k"``, ``"opus_24k"``.
    sr:
        Sample rate of *audio*.

    Returns
    -------
    np.ndarray
        Roundtripped float32 waveform at the original sample rate.

    Raises
    ------
    RuntimeError
        If *ffmpeg* is not found on ``PATH``.
    ValueError
        If *format_tag* is not recognised.
    """
    if format_tag == "none":
        return audio

    bitrate = _CODEC_BITRATES.get(format_tag)
    if bitrate is None:
        raise ValueError(
            f"Unknown codec format_tag '{format_tag}'. "
            f"Expected one of: none, {', '.join(sorted(_CODEC_BITRATES))}."
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to use codec simulation.")

    pcm_in = audio.astype(np.float32).tobytes()

    # Encode to Opus in OGG container
    encode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "pipe:",
            "-c:a",
            "libopus",
            "-b:a",
            str(bitrate),
            "-f",
            "ogg",
            "pipe:",
        ],
        input=pcm_in,
        capture_output=True,
        check=True,
    )

    # Decode back to f32le
    decode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:",
            "-f",
            "f32le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "pipe:",
        ],
        input=encode.stdout,
        capture_output=True,
        check=True,
    )

    n_samples = len(decode.stdout) // struct.calcsize("f")
    return np.frombuffer(decode.stdout, dtype=np.float32, count=n_samples).copy()
