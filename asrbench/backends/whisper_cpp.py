"""Whisper.cpp backend adapter via `pywhispercpp`.

pywhispercpp is a lightweight Cython wrapper around the upstream whisper.cpp
ggml runtime. It is the ASRbench backend of choice when no NVIDIA GPU is
available — whisper.cpp uses CPU (AVX/NEON) or Metal on Apple Silicon, so
benchmarks stay meaningful in GPU-less environments.

The public surface matches faster-whisper closely so parameter matrices and
IAMS studies can be shared between the two backends. Parameters that
whisper.cpp does not honour (VAD sub-params, batch_size, etc.) are accepted
silently — filtering happens in ``supported_params``.
"""

from __future__ import annotations

import gc
import inspect
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from asrbench.backends.base import BaseBackend, Segment

if TYPE_CHECKING:
    from pywhispercpp.model import Model  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Parameters consumed during ``load`` rather than ``transcribe``.
_LOAD_PARAMS: frozenset[str] = frozenset({"n_threads", "device"})

# Keys accepted at the module default level — mirrors faster-whisper where it
# makes sense so a shared matrix.json can swap backends without edits.
_DEFAULT_N_THREADS = 4


class WhisperCppBackend(BaseBackend):
    """`whisper.cpp` via pywhispercpp (CPU-first; Metal on Apple Silicon)."""

    family = "whisper"
    name = "whisper-cpp"

    def __init__(self) -> None:
        self._model: Model | None = None
        self._n_threads: int = _DEFAULT_N_THREADS
        self._accepted_kwargs: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    # BaseBackend contract
    # ------------------------------------------------------------------

    def default_params(self) -> dict:
        return {
            # Load-time
            "n_threads": _DEFAULT_N_THREADS,
            # Decoding
            "beam_size": 5,
            "temperature": 0.0,
            "language": "en",
            "translate": False,
            # Reliability thresholds (names match whisper.cpp public options)
            "no_speech_thold": 0.6,
            "logprob_thold": -1.0,
            "entropy_thold": 2.4,
            # Output shape
            "single_segment": False,
            "suppress_blank": True,
        }

    def supported_params(self, *, mode_hint: dict | None = None) -> set[str] | None:
        """Return the subset of params pywhispercpp actually honours.

        Discovered once at import time rather than hard-coded so the plugin
        keeps working when pywhispercpp adds or renames options. Returns
        None when the library is not installed — the optimizer then keeps
        the full space and trips with a clear error the first time load()
        runs.
        """
        _ = mode_hint
        try:
            kwargs = _transcribe_kwargs()
        except ImportError:
            return None
        accepted = set(kwargs) | set(_LOAD_PARAMS)
        accepted.add("language")
        accepted.add("beam_size")
        accepted.add("temperature")
        return accepted

    def load(self, model_path: str, params: dict) -> None:
        try:
            from pywhispercpp.model import Model  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "whisper.cpp backend is not installed. "
                "Install with: pip install 'asrbench[whisper-cpp]'"
            ) from exc

        self._n_threads = int(params.get("n_threads", _DEFAULT_N_THREADS))
        try:
            self._model = Model(model_path, n_threads=self._n_threads)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load whisper.cpp model from '{model_path}': {exc}"
            ) from exc

        try:
            self._accepted_kwargs = frozenset(_transcribe_kwargs())
        except ImportError:
            self._accepted_kwargs = frozenset()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

    def transcribe(self, audio: np.ndarray, lang: str, params: dict) -> list[Segment]:
        if self._model is None:
            raise RuntimeError("WhisperCppBackend.load() must be called before transcribe().")

        audio = np.ascontiguousarray(audio, dtype=np.float32)

        forwarded: dict[str, Any] = {"language": lang}
        for key, value in params.items():
            if key in _LOAD_PARAMS:
                continue
            if self._accepted_kwargs and key not in self._accepted_kwargs:
                continue
            forwarded[key] = value

        try:
            raw_segments = self._model.transcribe(audio, **forwarded)
        except TypeError as exc:
            # Older / newer pywhispercpp releases reject unknown kwargs. Retry
            # with only the keys we know for certain are accepted.
            logger.debug(
                "WhisperCpp transcribe rejected kwargs (%s); retrying with language only.", exc
            )
            raw_segments = self._model.transcribe(audio, language=lang)

        return [_to_segment(seg) for seg in raw_segments]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _transcribe_kwargs() -> frozenset[str]:
    """Inspect ``pywhispercpp.Model.transcribe`` for its accepted kwargs.

    Cached at module level via ``functools.cache`` so the inspection cost is
    paid once per process. Raises ``ImportError`` if pywhispercpp is missing.
    """
    return _cached_transcribe_kwargs()


def _to_segment(raw: Any) -> Segment:
    """Convert a pywhispercpp segment into an ASRbench :class:`Segment`.

    pywhispercpp returns ``Segment`` objects with ``t0`` / ``t1`` in units of
    10 ms (as exposed by the underlying whisper.cpp C API). ``text`` is the
    already-decoded string, with no leading whitespace.
    """
    text = str(getattr(raw, "text", "")).strip()
    t0 = float(getattr(raw, "t0", 0.0)) / 100.0
    t1 = float(getattr(raw, "t1", 0.0)) / 100.0
    duration = max(0.0, t1 - t0)
    return Segment(offset_s=t0, duration_s=duration, ref_text="", hyp_text=text)


def _cached_transcribe_kwargs() -> frozenset[str]:
    global _TRANSCRIBE_KWARGS_CACHE
    if _TRANSCRIBE_KWARGS_CACHE is not None:
        return _TRANSCRIBE_KWARGS_CACHE
    from pywhispercpp.model import Model  # type: ignore[import-not-found]

    sig = inspect.signature(Model.transcribe)
    names = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name not in ("self", "media", "audio", "audio_input")
    }
    _TRANSCRIBE_KWARGS_CACHE = frozenset(names)
    return _TRANSCRIBE_KWARGS_CACHE


_TRANSCRIBE_KWARGS_CACHE: frozenset[str] | None = None
