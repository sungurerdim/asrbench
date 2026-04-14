"""Whisper.cpp backend stub — requires pywhispercpp."""

from __future__ import annotations

import numpy as np

from asrbench.backends.base import BaseBackend, Segment


class WhisperCppBackend(BaseBackend):
    """Stub for whisper.cpp via pywhispercpp. Install with ``pip install asrbench[whisper-cpp]``."""

    family = "whisper"
    name = "whisper-cpp"

    def default_params(self) -> dict:
        return {
            "beam_size": 5,
            "temperature": 0.0,
            "language": "en",
        }

    def load(self, model_path: str, params: dict) -> None:
        raise RuntimeError(
            "whisper.cpp backend is not installed. Install with: pip install asrbench[whisper-cpp]"
        )

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        raise RuntimeError(
            "whisper.cpp backend is not installed. Install with: pip install asrbench[whisper-cpp]"
        )
