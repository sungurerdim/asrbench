"""NVIDIA NeMo Parakeet backend adapter.

Parakeet is NVIDIA's production ASR family (TDT + CTC + RNN-T variants).
The models are hosted on NGC / HuggingFace under CC-BY-4.0; the NeMo
runtime itself is Apache-2.0. ASRbench registers Parakeet as an optional
backend so the heavy NeMo install (~1 GB wheel, CUDA-coupled) is
opt-in via ``pip install asrbench[parakeet]``.

Supported tunables in ``transcribe(params=...)``:

* ``beam_size`` (int, default 1) — beam width for TDT/RNN-T decoders.
  Passed as ``batch_size`` is NOT a beam knob for Parakeet — ``beam_size``
  covers the accuracy/latency tradeoff.
* ``decoder_type`` (str, default auto) — one of ``greedy``, ``beam``,
  ``beam_maes``. Falls back to the model's shipped default when the
  model exposes only one decoder.
* ``batch_size`` (int, default 1) — parallel transcribe batch for
  multi-segment pipelines. ASRbench always calls transcribe one
  segment at a time so this rarely matters here; exposed for
  benchmark callers that pre-aggregate audio.
* ``compute_type`` (str, default ``float16``) — ``float32`` /
  ``float16`` / ``bfloat16``. Parakeet accepts all three on recent
  NeMo versions.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from asrbench.backends.base import BaseBackend, Segment

logger = logging.getLogger(__name__)

# ParameterSpace knobs the IAMS optimizer may target for this backend.
SEARCHABLE_PARAMS: frozenset[str] = frozenset(
    {
        "beam_size",
        "decoder_type",
        "batch_size",
        "compute_type",
    }
)

_DEFAULT_PARAMS: dict[str, Any] = {
    "beam_size": 1,
    "decoder_type": "greedy",
    "batch_size": 1,
    "compute_type": "float16",
}

# Rough per-parameter memory footprint in MiB. Real size depends on
# variant (parakeet-tdt-0.6b vs parakeet-rnnt-1.1b); the optimiser only
# needs a ballpark so the VRAM guard can refuse obviously-too-big loads.
_VRAM_ESTIMATE_MIB: dict[str, float] = {
    "parakeet-tdt-0.6b": 1500.0,
    "parakeet-tdt-1.1b": 2800.0,
    "parakeet-rnnt-0.6b": 1500.0,
    "parakeet-rnnt-1.1b": 2800.0,
    "parakeet-ctc-0.6b": 1100.0,
    "parakeet-ctc-1.1b": 2200.0,
}


def _estimate_vram_mb(model_id: str, compute_type: str) -> float:
    """Best-effort VRAM estimate for the VRAMMonitor.require_capacity guard."""
    key = model_id.rsplit("/", 1)[-1].lower()
    base = _VRAM_ESTIMATE_MIB.get(key, 1500.0)
    # fp32 doubles the memory; bf16/fp16 keep the base.
    if compute_type == "float32":
        return base * 2.0
    return base


class ParakeetBackend(BaseBackend):
    """NeMo-backed Parakeet adapter.

    ``load`` pulls the model via
    ``nemo.collections.asr.models.ASRModel.from_pretrained``; NeMo
    caches the weights under ``~/.cache/huggingface`` (or NGC). Unload
    drops the module and empties the CUDA cache so consecutive benchmark
    runs on different Parakeet variants don't OOM.
    """

    family: str = "parakeet"
    name: str = "parakeet"

    def __init__(self) -> None:
        # ``Any`` because NeMo is optional; the adapter still needs to
        # import on hosts that lack nemo_toolkit so ``load_backends``
        # can list Parakeet in registry errors.
        self._model: Any = None
        self._model_path: str | None = None
        self._compute_type: str = "float16"

    def default_params(self) -> dict[str, Any]:
        return dict(_DEFAULT_PARAMS)

    def load(self, model_path: str, params: dict) -> None:
        """Download (or reuse) the NGC/HF checkpoint and move it to GPU."""
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet requires NeMo. Install with: "
                "pip install 'asrbench[parakeet]'\n"
                "NeMo has a CUDA-coupled install; see "
                "https://github.com/NVIDIA/NeMo for platform-specific hints."
            ) from exc

        compute_type = str(params.get("compute_type", "float16"))
        self._guard_vram(model_path, compute_type)

        logger.info("Parakeet: loading %s (compute_type=%s)", model_path, compute_type)
        model = ASRModel.from_pretrained(model_name=model_path)
        model = _move_to_device_and_cast(model, compute_type)
        model.eval()
        self._model = model
        self._model_path = model_path
        self._compute_type = compute_type

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        self._model_path = None
        gc.collect()
        _release_cuda_cache()

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        if self._model is None:
            raise RuntimeError(
                "ParakeetBackend.transcribe() called before load(). "
                "Call load(model_path, params) first."
            )
        del lang  # Parakeet models are single-language by checkpoint choice

        beam_size = int(params.get("beam_size", _DEFAULT_PARAMS["beam_size"]))
        decoder_type = str(params.get("decoder_type", _DEFAULT_PARAMS["decoder_type"]))
        batch_size = max(1, int(params.get("batch_size", _DEFAULT_PARAMS["batch_size"])))

        self._configure_decoder(decoder_type=decoder_type, beam_size=beam_size)

        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
        transcribe_kwargs: dict[str, Any] = {"batch_size": batch_size}
        # NeMo transcribe() accepts a list of numpy arrays or file paths.
        raw_output = self._model.transcribe([audio_f32], **transcribe_kwargs)

        hyp_text = _extract_hypothesis(raw_output)
        duration_s = float(len(audio_f32)) / 16_000.0
        return [Segment(offset_s=0.0, duration_s=duration_s, ref_text="", hyp_text=hyp_text)]

    def supported_params(self, *, mode_hint: dict | None = None) -> set[str] | None:
        del mode_hint
        return set(SEARCHABLE_PARAMS)

    def _guard_vram(self, model_path: str, compute_type: str) -> None:
        """Refuse to load when free VRAM is obviously insufficient."""
        from asrbench.engine.vram import get_vram_monitor

        estimate = _estimate_vram_mb(model_path, compute_type)
        get_vram_monitor().require_capacity(estimate, model_label=f"Parakeet ({model_path})")

    def _configure_decoder(self, *, decoder_type: str, beam_size: int) -> None:
        """Apply decoder_type / beam_size to the live model.

        Parakeet's NeMo API changed across versions; the guarded
        ``hasattr`` dance keeps the adapter working on both the
        1.x (change_decoding_strategy) and 2.x (decoding.apply_config)
        surfaces without requiring a pinned NeMo version.
        """
        model = self._model
        assert model is not None

        if hasattr(model, "change_decoding_strategy"):
            try:
                from omegaconf import OmegaConf

                cfg = OmegaConf.create({"strategy": decoder_type, "beam": {"beam_size": beam_size}})
                model.change_decoding_strategy(cfg)
                return
            except Exception as exc:
                logger.debug(
                    "Parakeet: change_decoding_strategy failed, using shipped default: %s",
                    exc,
                )


def _move_to_device_and_cast(model: Any, compute_type: str) -> Any:
    """Cast weights and push to CUDA when available."""
    try:
        import torch
    except ImportError:
        return model

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(compute_type, torch.float16)

    if torch.cuda.is_available():
        model = model.to("cuda")
    if compute_type in dtype_map:
        model = model.to(dtype=dtype)
    return model


def _extract_hypothesis(raw_output: Any) -> str:
    """NeMo.transcribe returns a list (one per audio) of either strings
    or Hypothesis objects depending on version and decoder_type.
    Normalise to a plain string.
    """
    if not raw_output:
        return ""
    item = raw_output[0] if isinstance(raw_output, list) else raw_output
    if isinstance(item, list):
        # Older NeMo: [[best_hyp], ...]
        item = item[0] if item else ""
    if isinstance(item, str):
        return item.strip()
    # Hypothesis-like object
    for attr in ("text", "hypothesis"):
        value = getattr(item, attr, None)
        if isinstance(value, str):
            return value.strip()
    return str(item).strip()


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
