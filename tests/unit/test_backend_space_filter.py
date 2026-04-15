"""
Tests for Faz 1: backend-aware ParameterSpace filtering.

Covers three layers:

    1. ParameterSpace.restrict_to() — keeps allowed names, always keeps
       ``preprocess.*``, short-circuits to a no-op if the filter would empty
       the space.

    2. BaseBackend.supported_params() default → None (opt-in). Backends that
       don't override it must not change optimizer behavior.

    3. FasterWhisperBackend.supported_params(mode_hint={"batch_size": N}):
       returns a set that EXCLUDES the batched-mode no-ops
       (without_timestamps, vad_*, condition_on_previous_text, etc.) when
       batch_size > 0, and None in sequential mode.

    4. IAMSOptimizer construction with a backend + mode_hint actually shrinks
       the space passed to Layer 1 — the whole point of the filter.
"""

from __future__ import annotations

import pytest

from asrbench.backends.base import BaseBackend, Segment
from asrbench.backends.faster_whisper import FasterWhisperBackend
from asrbench.engine.optimizer import IAMSOptimizer
from asrbench.engine.search.budget import BudgetController
from asrbench.engine.search.objective import SingleMetricObjective
from asrbench.engine.search.space import ParameterSpace, ParamSpec
from asrbench.engine.search.trial import SyntheticTrialExecutor

# ---------------------------------------------------------------------------
# restrict_to()
# ---------------------------------------------------------------------------


def _make_space(*names: str) -> ParameterSpace:
    specs = []
    for n in names:
        specs.append(ParamSpec(name=n, type="int", default=1, min=1, max=5))
    return ParameterSpace(parameters=tuple(specs))


def test_restrict_to_keeps_only_allowed() -> None:
    space = _make_space("beam_size", "vad_filter", "temperature")
    filtered = space.restrict_to({"beam_size", "temperature"})
    assert [p.name for p in filtered.parameters] == ["beam_size", "temperature"]


def test_restrict_to_always_keeps_preprocess_namespace() -> None:
    space = _make_space("beam_size", "preprocess.denoise", "vad_filter")
    filtered = space.restrict_to({"beam_size"})  # preprocess.* not in allowed
    names = {p.name for p in filtered.parameters}
    assert "beam_size" in names
    assert "preprocess.denoise" in names
    assert "vad_filter" not in names


def test_restrict_to_empty_result_falls_back_to_original_space() -> None:
    space = _make_space("beam_size", "vad_filter")
    # The allowed set has zero overlap and no preprocess.* → fallback no-op.
    filtered = space.restrict_to({"nonexistent_param"})
    assert filtered is space  # exact identity — no new ParameterSpace built


# ---------------------------------------------------------------------------
# BaseBackend default
# ---------------------------------------------------------------------------


class _StubBackend(BaseBackend):
    family = "stub"
    name = "stub"

    def default_params(self) -> dict:
        return {}

    def load(self, model_path: str, params: dict) -> None:  # noqa: ARG002
        return None

    def unload(self) -> None:
        return None

    def transcribe(self, audio, lang, params) -> list[Segment]:  # type: ignore[override]  # noqa: ARG002
        return []


def test_base_backend_supported_params_defaults_to_none() -> None:
    b = _StubBackend()
    assert b.supported_params() is None
    assert b.supported_params(mode_hint={"batch_size": 0}) is None
    assert b.supported_params(mode_hint={"batch_size": 5}) is None


# ---------------------------------------------------------------------------
# FasterWhisperBackend batched filter
# ---------------------------------------------------------------------------


def test_faster_whisper_sequential_returns_none() -> None:
    backend = FasterWhisperBackend()
    assert backend.supported_params(mode_hint={"batch_size": 0}) is None
    assert backend.supported_params() is None


def test_faster_whisper_batched_excludes_known_noops() -> None:
    pytest.importorskip("faster_whisper")
    backend = FasterWhisperBackend()
    supported = backend.supported_params(mode_hint={"batch_size": 5})
    assert supported is not None

    # Runtime no-ops must NOT be in the supported set.
    ignored = {
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
    assert supported.isdisjoint(ignored)

    # Core decoding knobs must STILL be present (signature-level accepted).
    assert "beam_size" in supported
    # batch_size routing key itself must stay so the space can declare batched mode.
    assert "batch_size" in supported


# ---------------------------------------------------------------------------
# IAMSOptimizer wires backend through restrict_to
# ---------------------------------------------------------------------------


class _FakeFilteringBackend(BaseBackend):
    """Trivial backend whose supported_params drops 'drop_me'."""

    family = "fake"
    name = "fake-filter"

    def default_params(self) -> dict:
        return {}

    def load(self, model_path: str, params: dict) -> None:  # noqa: ARG002
        return None

    def unload(self) -> None:
        return None

    def transcribe(self, audio, lang, params) -> list[Segment]:  # type: ignore[override]  # noqa: ARG002
        return []

    def supported_params(self, *, mode_hint: dict | None = None) -> set[str] | None:  # noqa: ARG002
        return {"keep_me", "preprocess.foo"}


def _metrics(wer: float) -> dict[str, float]:
    return {
        "wer": wer,
        "cer": wer,
        "mer": wer,
        "wil": wer,
        "rtfx_mean": 10.0,
        "vram_peak_mb": 1000.0,
        "wer_ci_lower": max(0.0, wer - 0.001),
        "wer_ci_upper": wer + 0.001,
    }


def test_optimizer_filters_space_via_backend() -> None:
    space = ParameterSpace(
        parameters=(
            ParamSpec(name="keep_me", type="int", default=1, min=1, max=4),
            ParamSpec(name="drop_me", type="int", default=1, min=1, max=4),
            ParamSpec(name="preprocess.foo", type="int", default=1, min=1, max=4),
        )
    )

    # Synthetic flat landscape — we don't care about the optimum, only the
    # final space the optimizer kept after filtering.
    objective = SingleMetricObjective(metric="wer")
    executor = SyntheticTrialExecutor(
        metric_fn=lambda cfg: _metrics(0.1),  # noqa: ARG005
        objective=objective,
    )
    optimizer = IAMSOptimizer(
        executor=executor,
        space=space,
        objective=objective,
        budget=BudgetController(hard_cap=50, convergence_eps=0.0),
        eps_min=0.0,
        mode="fast",
        backend=_FakeFilteringBackend(),
        mode_hint={"batch_size": 5},
    )
    kept = {p.name for p in optimizer.space.parameters}
    assert "keep_me" in kept
    assert "preprocess.foo" in kept
    assert "drop_me" not in kept
