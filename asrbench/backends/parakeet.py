"""NVIDIA Parakeet backend stub — requires NeMo toolkit."""

from __future__ import annotations

import numpy as np

from asrbench.backends.base import BaseBackend, Segment


class ParakeetBackend(BaseBackend):
    """Stub for NVIDIA Parakeet via NeMo. Install with ``pip install asrbench[parakeet]``."""

    family = "parakeet"
    name = "parakeet"

    def default_params(self) -> dict:
        return {}

    def load(self, model_path: str, params: dict) -> None:
        raise RuntimeError(
            "Parakeet backend is not installed. Install with: pip install asrbench[parakeet]"
        )

    def unload(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        raise RuntimeError(
            "Parakeet backend is not installed. Install with: pip install asrbench[parakeet]"
        )
