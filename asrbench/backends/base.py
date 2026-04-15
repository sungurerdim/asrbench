"""Base class for all ASRbench backend adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    """A single transcription segment with timing and metric fields."""

    offset_s: float
    duration_s: float
    ref_text: str
    hyp_text: str


class BaseBackend(ABC):
    """
    Contract for all ASR backend adapters.

    Preconditions:
    - load() must be called before transcribe().
    - audio must be float32 mono at 16 kHz.
    - lang must be an ISO 639-1 code (e.g. "en", "tr").
    - params must not contain unknown keys — validated by the caller.

    Side effects:
    - load() allocates GPU/CPU memory.
    - unload() releases it; no-op if not currently loaded.
    """

    family: str  # e.g. "whisper"
    name: str  # e.g. "faster-whisper"

    @abstractmethod
    def default_params(self) -> dict:
        """Return backend-specific default transcription parameters."""
        ...

    @abstractmethod
    def load(self, model_path: str, params: dict) -> None:
        """
        Load model into memory.

        Raises:
            RuntimeError: if model_path does not exist.
            MemoryError:  if insufficient VRAM or RAM.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory. No-op if not currently loaded."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        """
        Transcribe an audio array.

        Returns a list of Segment with hyp_text populated; ref_text is empty string.

        Raises:
            RuntimeError: if model is not loaded.
        """
        ...

    def supported_params(self, *, mode_hint: dict | None = None) -> set[str] | None:
        """
        Return the subset of parameter names the backend actually honors
        in the given runtime mode, or None to opt out of filtering.

        Default None = "accept everything" — existing behavior (signature-level
        filtering happens inside transcribe()). Override in a subclass when the
        backend silently ignores a subset of params in a particular mode (e.g.
        faster-whisper batched mode drops VAD / without_timestamps / etc.).

        The IAMSOptimizer uses this return value to restrict the ParameterSpace
        before Layer 1 screening — no trial budget is spent probing params the
        backend will ignore at runtime.

        Parameters that start with ``preprocess.`` are always kept by the
        optimizer regardless of this return value — they live outside the
        backend surface.
        """
        _ = mode_hint  # default impl ignores mode; subclasses consume it
        return None
