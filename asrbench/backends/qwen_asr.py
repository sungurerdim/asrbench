"""Qwen-Audio ASR backend adapter via HuggingFace Transformers.

**License warning**: Qwen2-Audio is released under the Qwen Community
License, which restricts commercial use beyond a threshold of monthly
active users. ASRbench ships this backend as an OPTIONAL extra so the
license decision sits with the installing user. Install with
``pip install 'asrbench[qwen]'`` and read
https://github.com/QwenLM/Qwen2-Audio/blob/main/LICENSE before using
it in production.

Prompting approach
------------------
Qwen2-Audio is a multimodal instruction-tuned LLM, not a pure CTC/RNN-T
model. Transcription happens via a fixed chat template:

    system: "You are a speech transcription assistant."
    user  : <|audio_bos|>{audio}<|audio_eos|> "Transcribe the audio.
            Output ONLY the transcript, no commentary."

The assistant's generated text (with special tokens stripped) is the
hypothesis. Output is deterministic (``do_sample=False``,
``temperature=0``) by default; the IAMS optimiser can flip these for
decoder-temperature experiments.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from asrbench.backends.base import BaseBackend, Segment

logger = logging.getLogger(__name__)

SEARCHABLE_PARAMS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "max_new_tokens",
        "do_sample",
        "compute_type",
    }
)

_DEFAULT_PARAMS: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "do_sample": False,
    "compute_type": "bfloat16",
}

_SAMPLE_RATE = 16_000

# Rough per-model VRAM estimates in MiB (fp16). Used only by the
# ResourceExhausted guard — not authoritative for planning.
_VRAM_ESTIMATE_MIB: dict[str, float] = {
    "Qwen/Qwen2-Audio-7B": 16_000.0,
    "Qwen/Qwen2-Audio-7B-Instruct": 16_000.0,
}

_SYSTEM_PROMPT = (
    "You are a precise speech transcription assistant. "
    "Output ONLY the verbatim transcript of the audio you are given. "
    "Do not add commentary, translations, or formatting."
)
_USER_PROMPT = "Transcribe this audio."


def _estimate_vram_mb(model_id: str, compute_type: str) -> float:
    base = _VRAM_ESTIMATE_MIB.get(model_id, 16_000.0)
    if compute_type == "float32":
        return base * 2.0
    return base


class QwenASRBackend(BaseBackend):
    """Qwen2-Audio adapter wired through ``transformers``."""

    family: str = "qwen"
    name: str = "qwen_asr"

    def __init__(self) -> None:
        # Use ``Any`` for the model / processor handles because the real
        # types are only resolvable once ``transformers`` is installed;
        # the adapter must still import (and its non-runtime methods type
        # cleanly) without the optional extra.
        self._model: Any = None
        self._processor: Any = None
        self._model_path: str | None = None
        self._compute_type: str = "bfloat16"

    def default_params(self) -> dict[str, Any]:
        return dict(_DEFAULT_PARAMS)

    def load(self, model_path: str, params: dict) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen-Audio requires the 'transformers' extra. Install with: "
                "pip install 'asrbench[qwen]'. This pulls transformers, "
                "torch, and accelerate."
            ) from exc

        compute_type = str(params.get("compute_type", "bfloat16"))
        self._guard_vram(model_path, compute_type)

        logger.info("Qwen-Audio: loading %s (compute_type=%s)", model_path, compute_type)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=_resolve_torch_dtype(compute_type),
            trust_remote_code=True,
        )
        model.eval()

        self._model = model
        self._processor = processor
        self._model_path = model_path
        self._compute_type = compute_type

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._model_path = None
        gc.collect()
        _release_cuda_cache()

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        if self._model is None or self._processor is None:
            raise RuntimeError(
                "QwenASRBackend.transcribe() called before load(). "
                "Call load(model_path, params) first."
            )
        del lang  # Qwen-Audio reads language cues from the audio + prompt

        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
        inputs = self._build_inputs(audio_f32)
        generation_kwargs = self._generation_kwargs(params)

        import torch

        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_kwargs)

        # The prompt tokens live at the beginning of generated; strip them.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_len:]
        text = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        duration_s = float(len(audio_f32)) / float(_SAMPLE_RATE)
        return [Segment(offset_s=0.0, duration_s=duration_s, ref_text="", hyp_text=text.strip())]

    def supported_params(self, *, mode_hint: dict | None = None) -> set[str] | None:
        del mode_hint
        return set(SEARCHABLE_PARAMS)

    def _guard_vram(self, model_path: str, compute_type: str) -> None:
        from asrbench.engine.vram import get_vram_monitor

        estimate = _estimate_vram_mb(model_path, compute_type)
        get_vram_monitor().require_capacity(estimate, model_label=f"Qwen-Audio ({model_path})")

    def _build_inputs(self, audio: np.ndarray) -> dict[str, Any]:
        """Render the chat template and let the processor produce tensors."""
        assert self._processor is not None
        conversation = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "local"},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        raw_inputs = self._processor(
            text=text,
            audios=[audio],
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs: dict[str, Any] = dict(raw_inputs)
        try:
            import torch

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        except ImportError:
            pass
        return inputs

    def _generation_kwargs(self, params: dict) -> dict[str, Any]:
        """Map ASRbench params onto transformers' generate() kwargs."""
        do_sample = bool(params.get("do_sample", _DEFAULT_PARAMS["do_sample"]))
        temperature = float(params.get("temperature", _DEFAULT_PARAMS["temperature"]))
        top_p = float(params.get("top_p", _DEFAULT_PARAMS["top_p"]))
        max_new_tokens = int(params.get("max_new_tokens", _DEFAULT_PARAMS["max_new_tokens"]))

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        # Transformers warns when temperature/top_p are passed with
        # do_sample=False; only include them when sampling is enabled.
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        return kwargs


def _resolve_torch_dtype(compute_type: str) -> Any:
    try:
        import torch
    except ImportError:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(compute_type, torch.bfloat16)


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
