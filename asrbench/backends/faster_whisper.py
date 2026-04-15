"""Faster-whisper backend adapter using CTranslate2."""

from __future__ import annotations

import functools
import gc
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from asrbench.backends.base import BaseBackend, Segment

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Params consumed by load(), not by transcribe()
_LOAD_PARAMS: frozenset[str] = frozenset({"device", "compute_type"})

# batch_size = 0 → sequential (WhisperModel.transcribe)
# batch_size > 0 → BatchedInferencePipeline.transcribe(batch_size=N)
# Extracted before building transcribe_params to determine routing.
_BATCH_SIZE_KEY = "batch_size"

# Flat vad_* keys → vad_parameters dict keys.
# These are passed to transcribe() as vad_parameters={"threshold": ..., ...}
# when vad_filter=True and any vad_* override is present.
_VAD_PARAM_MAP: dict[str, str] = {
    "vad_threshold": "threshold",
    "vad_min_speech_duration_ms": "min_speech_duration_ms",
    "vad_max_speech_duration_s": "max_speech_duration_s",
    "vad_min_silence_duration_ms": "min_silence_duration_ms",
    "vad_speech_pad_ms": "speech_pad_ms",
}

# Params that use a sentinel value to mean "disabled / None".
# In the IAMS space these are expressed as numbers; 0 (or 0.0) → None in the API.
#   hallucination_silence_threshold: 0.0 → None
#   max_new_tokens:                  0   → None
#   chunk_length:                    0   → None
_SENTINEL_NONE: dict[str, int | float] = {
    "hallucination_silence_threshold": 0.0,
    "max_new_tokens": 0,
    "chunk_length": 0,
}

# Params that BatchedInferencePipeline.transcribe() accepts at the signature
# level but silently no-ops at runtime. Optimizing them wastes trials because
# every probed value produces the same WER. Kept as the single source of truth
# for batched-mode pruning — replaces the previous v3 YAML split.
_BATCHED_IGNORED_PARAMS: frozenset[str] = frozenset(
    {
        "without_timestamps",
        "vad_filter",
        "vad_threshold",
        "vad_min_speech_duration_ms",
        "vad_max_speech_duration_s",
        "vad_min_silence_duration_ms",
        "vad_speech_pad_ms",
        "condition_on_previous_text",
        "prompt_reset_on_temperature",
        "hallucination_silence_threshold",
        "chunk_length",
    }
)


@functools.cache
def _whisper_transcribe_params() -> frozenset[str]:
    """Return accepted parameter names for WhisperModel.transcribe() (cached)."""
    import inspect

    from faster_whisper import WhisperModel

    return frozenset(inspect.signature(WhisperModel.transcribe).parameters.keys())


@functools.cache
def _batched_transcribe_params() -> frozenset[str]:
    """Return accepted parameter names for BatchedInferencePipeline.transcribe() (cached)."""
    import inspect

    from faster_whisper import BatchedInferencePipeline

    return frozenset(inspect.signature(BatchedInferencePipeline.transcribe).parameters.keys())


class FasterWhisperBackend(BaseBackend):
    """
    Backend adapter for faster-whisper (CTranslate2-based Whisper inference).

    Preconditions:
    - ``pip install asrbench[faster-whisper]`` must be installed.
    - load() must be called before transcribe().

    Side effects:
    - load() allocates GPU memory via CTranslate2.
    - unload() releases it and calls torch.cuda.empty_cache() if torch is available.
    """

    family = "whisper"
    name = "faster-whisper"

    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def supported_params(self, *, mode_hint: dict | None = None) -> set[str] | None:
        """
        Restrict the IAMS parameter space to what faster-whisper honors
        in the current inference mode.

        - Sequential (``batch_size == 0``): return None → optimizer keeps the
          full space; signature-level filtering in transcribe() is enough.
        - Batched (``batch_size > 0``): return BatchedInferencePipeline.transcribe()
          signature MINUS the runtime no-ops listed in _BATCHED_IGNORED_PARAMS.
          The optimizer then skips those params entirely, avoiding wasted trials.
        """
        batch_size = int((mode_hint or {}).get("batch_size", 0))
        if batch_size <= 0:
            return None
        try:
            accepted = set(_batched_transcribe_params())
        except ImportError:
            # faster-whisper not installed — no filtering (optimizer will fail
            # elsewhere on first transcribe, which is the right signal).
            return None
        accepted -= _BATCHED_IGNORED_PARAMS
        # batch_size itself must stay — it's how the space tells the backend
        # to enter batched mode in the first place.
        accepted.add(_BATCH_SIZE_KEY)
        # Load-time params belong to the backend, not to transcribe(); keep
        # them in the space so load() can still consume them.
        accepted.update(_LOAD_PARAMS)
        return accepted

    def default_params(self) -> dict:
        return {
            # ── Beam / decoding ────────────────────────────────────────────
            "beam_size": 5,
            "best_of": 5,  # candidates when temperature > 0
            "patience": 1.0,
            "length_penalty": 1.0,
            "temperature": 0.0,  # 0.0 = greedy; list for fallback cascade
            # ── Repetition / output constraints ───────────────────────────
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "suppress_blank": True,
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "max_new_tokens": 0,  # 0 = no limit (sentinel → None)
            # ── Reliability thresholds ─────────────────────────────────────
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "hallucination_silence_threshold": 0.0,  # 0.0 = disabled (sentinel → None)
            # ── Context / chunking ─────────────────────────────────────────
            "condition_on_previous_text": True,
            "prompt_reset_on_temperature": 0.5,
            "chunk_length": 0,  # 0 = model default (sentinel → None)
            # ── VAD (Silero) ───────────────────────────────────────────────
            "vad_filter": False,
            # Sub-params below only used when vad_filter=True.
            # Assembled into vad_parameters={...} dict by transcribe().
            "vad_threshold": 0.5,  # speech probability threshold
            "vad_min_speech_duration_ms": 250,  # ignore speech shorter than this
            "vad_max_speech_duration_s": 30.0,  # split segments longer than this
            "vad_min_silence_duration_ms": 2000,  # silence gap needed to split
            "vad_speech_pad_ms": 400,  # padding around speech segments
            # ── Inference mode ─────────────────────────────────────────────
            # 0 = sequential (WhisperModel.transcribe)
            # N > 0 = BatchedInferencePipeline.transcribe(batch_size=N)
            "batch_size": 0,
        }

    def load(self, model_path: str, params: dict) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install with: pip install asrbench[faster-whisper]"
            )

        device = params.get("device", "cuda")
        compute_type = params.get("compute_type", "float16")
        try:
            self._model = WhisperModel(model_path, device=device, compute_type=compute_type)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load faster-whisper model from '{model_path}': {exc}"
            ) from exc

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() before transcribe().")

        # Extract inference-mode param before building transcribe_params.
        # 0 = sequential (WhisperModel), N > 0 = BatchedInferencePipeline(batch_size=N).
        batch_size: int = int(params.get(_BATCH_SIZE_KEY, 0))

        transcribe_params: dict[str, Any] = {}
        vad_overrides: dict[str, Any] = {}

        for k, v in params.items():
            if k in _LOAD_PARAMS or k == _BATCH_SIZE_KEY:
                continue
            if k in _VAD_PARAM_MAP:
                vad_overrides[_VAD_PARAM_MAP[k]] = v
            elif k in _SENTINEL_NONE:
                # Convert sentinel → None (disabled); keep real values as-is
                transcribe_params[k] = None if v == _SENTINEL_NONE[k] else v
            else:
                transcribe_params[k] = v

        # Assemble vad_parameters when VAD is enabled and any sub-param was set
        if transcribe_params.get("vad_filter") and vad_overrides:
            transcribe_params["vad_parameters"] = vad_overrides

        if batch_size > 0:
            from faster_whisper import BatchedInferencePipeline

            accepted = _batched_transcribe_params()
            dropped = set(transcribe_params) - accepted
            if dropped:
                logger.warning(
                    "faster_whisper: dropping unsupported batched params: %s", sorted(dropped)
                )
            batched_params = {k: v for k, v in transcribe_params.items() if k in accepted}
            pipeline = BatchedInferencePipeline(model=self._model)
            segments_iter, _ = pipeline.transcribe(
                audio, language=lang, batch_size=batch_size, **batched_params
            )
        else:
            accepted = _whisper_transcribe_params()
            dropped = set(transcribe_params) - accepted
            if dropped:
                logger.warning(
                    "faster_whisper: dropping unsupported sequential params: %s", sorted(dropped)
                )
            seq_params = {k: v for k, v in transcribe_params.items() if k in accepted}
            segments_iter, _ = self._model.transcribe(audio, language=lang, **seq_params)

        result: list[Segment] = []
        for seg in segments_iter:
            result.append(
                Segment(
                    offset_s=seg.start,
                    duration_s=seg.end - seg.start,
                    ref_text="",
                    hyp_text=seg.text.strip(),
                )
            )
        return result
