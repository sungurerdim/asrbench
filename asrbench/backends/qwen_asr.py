"""Qwen-Audio backend stub — requires transformers + torch."""

from __future__ import annotations

import numpy as np

from asrbench.backends.base import BaseBackend, Segment


class QwenASRBackend(BaseBackend):
    """Stub for Qwen-Audio via HuggingFace Transformers."""

    family = "qwen"
    name = "qwen-asr"

    def default_params(self) -> dict:
        return {}

    def load(self, model_path: str, params: dict) -> None:
        raise RuntimeError(
            "Qwen-Audio backend is not installed. Install with: pip install asrbench[qwen-asr]"
        )

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        raise RuntimeError(
            "Qwen-Audio backend is not installed. Install with: pip install asrbench[qwen-asr]"
        )
