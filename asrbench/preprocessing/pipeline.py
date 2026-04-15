"""Preprocessing orchestrator — applies all steps in a fixed, signal-theory-correct order."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from asrbench.preprocessing.codec import simulate_codec
from asrbench.preprocessing.filters import (
    apply_drc,
    apply_highpass,
    apply_limiter,
    apply_lowpass,
    apply_notch,
    apply_preemphasis,
)
from asrbench.preprocessing.loudness import normalize_lufs
from asrbench.preprocessing.noise import reduce_noise
from asrbench.preprocessing.resample import simulate_resample
from asrbench.preprocessing.vad import vad_trim

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Stateless orchestrator that applies preprocessing steps in a fixed order.

    Order (signal-processing convention):
         1. Codec simulation   — lossy artifact injection
         2. Resample simulation — bandwidth limitation
         3. Notch filter        — mains hum removal (50/60 Hz)
         4. Highpass filter     — rumble / DC removal
         5. Lowpass filter      — HF noise / anti-alias shaping
         6. LUFS normalization  — consistent loudness
         7. DRC (compressor)    — dynamic range control
         8. Peak limiter        — brick-wall clip guard
         9. Noise reduction     — spectral denoising
        10. Pre-emphasis        — feature-extract-style HF boost
        11. VAD trim            — silence removal

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
            Missing keys fall back to no-op defaults. The optional
            ``backend`` key (``"scipy"`` default, ``"ffmpeg"``) selects the
            filter implementation; use ``"ffmpeg"`` for byte-identical
            mobile deployment parity.
        sr:
            Sample rate of *audio*.

        Returns
        -------
        np.ndarray
            Processed float32 waveform.
        """
        backend = str(params.get("backend", "scipy")).lower()
        if backend == "ffmpeg":
            try:
                from asrbench.preprocessing.ffmpeg_pipeline import (
                    FFmpegNotAvailable,
                    FFmpegPreprocessingPipeline,
                )

                return FFmpegPreprocessingPipeline.apply(audio, params, sr)
            except FFmpegNotAvailable as exc:
                logger.warning(
                    "preprocessing backend=ffmpeg requested but ffmpeg is not "
                    "available (%s); falling back to scipy.",
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                # Any runtime ffmpeg failure (bad filter chain, broken pipe)
                # degrades gracefully. The fallback keeps the benchmark
                # moving; the log line is the audit trail.
                logger.warning(
                    "ffmpeg preprocessing failed (%s); falling back to scipy for this segment.",
                    exc,
                )

        # 1. Codec simulation
        fmt = params.get("format", "none")
        if fmt != "none":
            audio = simulate_codec(audio, fmt, sr)

        # 2. Resample simulation
        target_sr = params.get("sample_rate", sr)
        if target_sr != sr:
            audio = simulate_resample(audio, target_sr, sr)

        # 3. Notch filter (mains hum)
        notch_hz = int(params.get("notch_hz", 0))
        if notch_hz > 0:
            audio = apply_notch(audio, notch_hz, sr)

        # 4. Highpass filter
        highpass_hz = params.get("highpass_hz", 0)
        if highpass_hz > 0:
            audio = apply_highpass(audio, highpass_hz, sr)

        # 5. Lowpass filter
        lowpass_hz = int(params.get("lowpass_hz", 0))
        if lowpass_hz > 0:
            audio = apply_lowpass(audio, lowpass_hz, sr)

        # 6. LUFS normalization
        lufs_target = params.get("lufs_target")
        if lufs_target is not None:
            lufs_lra = float(params.get("lufs_lra", 7.0))
            loudnorm_linear = bool(params.get("loudnorm_linear", False))
            audio = normalize_lufs(
                audio,
                lufs_target,
                sr,
                lra=lufs_lra,
                linear=loudnorm_linear,
            )

        # 7. Dynamic range compression
        drc_ratio = params.get("drc_ratio", 1.0)
        if drc_ratio > 1.0:
            audio = apply_drc(audio, drc_ratio)

        # 8. Peak limiter (alimiter equivalent)
        limiter_ceiling_db = float(params.get("limiter_ceiling_db", 0.0))
        if limiter_ceiling_db < 0.0:
            audio = apply_limiter(audio, limiter_ceiling_db)

        # 9. Noise reduction
        noise_strength = params.get("noise_reduce", 0.0)
        if noise_strength > 0.0:
            audio = reduce_noise(audio, noise_strength, sr)

        # 10. Pre-emphasis (classical ASR HF boost)
        preemph_coef = float(params.get("preemph_coef", 0.0))
        if preemph_coef > 0.0:
            audio = apply_preemphasis(audio, preemph_coef)

        # 11. VAD trim
        if params.get("vad_trim", False):
            threshold_dbfs = float(params.get("silence_threshold_db", -40.0))
            max_silence_s = float(params.get("silence_min_duration_s", 0.5))
            audio = vad_trim(
                audio,
                sr,
                threshold_dbfs=threshold_dbfs,
                max_silence_s=max_silence_s,
            )

        return audio

    @staticmethod
    def default_params() -> dict[str, Any]:
        """Return no-op default parameters (identity transform)."""
        return {
            "backend": "scipy",
            "format": "none",
            "sample_rate": 16_000,
            "notch_hz": 0,
            "highpass_hz": 0,
            "lowpass_hz": 0,
            "lufs_target": None,
            "lufs_lra": 7.0,
            "loudnorm_linear": False,
            "drc_ratio": 1.0,
            "limiter_ceiling_db": 0.0,
            "noise_reduce": 0.0,
            "preemph_coef": 0.0,
            "vad_trim": False,
            "silence_threshold_db": -40.0,
            "silence_min_duration_s": 0.5,
        }
