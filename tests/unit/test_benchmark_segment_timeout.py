"""Unit tests for BenchmarkEngine per-segment timeout (Faz 4.2)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from asrbench.engine.benchmark import BenchmarkEngine


class _SluggishBackend:
    """Backend double whose transcribe() blocks longer than the timeout."""

    family = "test"
    name = "test-sluggish"

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s
        self.calls = 0

    def transcribe(self, audio, lang, params):  # noqa: ARG002
        self.calls += 1
        time.sleep(self.sleep_s)
        from asrbench.backends.base import Segment

        return [Segment(offset_s=0.0, duration_s=0.1, ref_text="", hyp_text="ok")]


class _FastBackend:
    family = "test"
    name = "test-fast"

    def transcribe(self, audio, lang, params):  # noqa: ARG002
        from asrbench.backends.base import Segment

        return [Segment(offset_s=0.0, duration_s=0.1, ref_text="", hyp_text="fast")]


async def test_transcribe_with_timeout_hits_budget(tmp_path) -> None:
    """A backend that blocks beyond the budget must raise TimeoutError."""
    engine = BenchmarkEngine(conn=None, cache_dir=tmp_path, segment_timeout_s=0.2)  # type: ignore[arg-type]
    audio = np.zeros(16_000, dtype=np.float32)
    backend = _SluggishBackend(sleep_s=0.5)

    with pytest.raises(TimeoutError, match="per-segment timeout"):
        await engine._transcribe_with_timeout(backend, audio, "en", {})

    assert backend.calls == 1  # the slow call was actually started


async def test_transcribe_with_timeout_ok_below_budget(tmp_path) -> None:
    engine = BenchmarkEngine(conn=None, cache_dir=tmp_path, segment_timeout_s=2.0)  # type: ignore[arg-type]
    audio = np.zeros(16_000, dtype=np.float32)
    backend = _FastBackend()

    result = await engine._transcribe_with_timeout(backend, audio, "en", {})
    assert len(result) == 1
    assert result[0].hyp_text == "fast"


async def test_none_timeout_disables_the_guard(tmp_path) -> None:
    """Legacy behaviour: segment_timeout_s=None means "wait forever"."""
    engine = BenchmarkEngine(conn=None, cache_dir=tmp_path, segment_timeout_s=None)  # type: ignore[arg-type]
    audio = np.zeros(16_000, dtype=np.float32)
    backend = _SluggishBackend(sleep_s=0.3)  # 0.3 s is short; no timeout

    result = await engine._transcribe_with_timeout(backend, audio, "en", {})
    assert len(result) == 1


async def test_zero_or_negative_timeout_also_disables(tmp_path) -> None:
    engine = BenchmarkEngine(conn=None, cache_dir=tmp_path, segment_timeout_s=0.0)  # type: ignore[arg-type]
    audio = np.zeros(16_000, dtype=np.float32)
    backend = _SluggishBackend(sleep_s=0.2)

    result = await engine._transcribe_with_timeout(backend, audio, "en", {})
    assert len(result) == 1


def test_default_timeout_is_none(tmp_path) -> None:
    """Unless the caller passes one, BenchmarkEngine stays timeout-free."""
    engine = BenchmarkEngine(conn=None, cache_dir=tmp_path)  # type: ignore[arg-type]
    assert engine._segment_timeout_s is None
