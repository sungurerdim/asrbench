"""
Tests for the FFmpeg preprocessing backend.

These cover:
    1. ``build_filter_chain`` — pure string construction, no subprocess.
    2. ``is_ffmpeg_available`` — binary probing.
    3. Pipeline dispatch: ``PreprocessingPipeline.apply(backend="ffmpeg")``
       routes to the FFmpeg path and falls back to scipy on failure.
    4. End-to-end ffmpeg subprocess execution (skipped when ffmpeg is
       not on PATH).
"""

from __future__ import annotations

import numpy as np
import pytest

from asrbench.preprocessing.ffmpeg_pipeline import (
    FFmpegNotAvailable,
    build_filter_chain,
    is_ffmpeg_available,
    run_ffmpeg_chain,
)
from asrbench.preprocessing.pipeline import PreprocessingPipeline

# ---------------------------------------------------------------------------
# build_filter_chain — pure string construction
# ---------------------------------------------------------------------------


class TestBuildFilterChain:
    def test_empty_params_returns_empty_string(self) -> None:
        assert build_filter_chain({}, sr=16_000) == ""

    def test_default_params_returns_empty_string(self) -> None:
        """All-no-op defaults should not emit a single filter."""
        defaults = PreprocessingPipeline.default_params()
        assert build_filter_chain(defaults, sr=16_000) == ""

    def test_highpass_emits_filter(self) -> None:
        chain = build_filter_chain({"highpass_hz": 80}, sr=16_000)
        assert chain == "highpass=f=80"

    def test_lufs_target_emits_loudnorm(self) -> None:
        chain = build_filter_chain(
            {"lufs_target": -16.0, "lufs_lra": 7.0, "loudnorm_linear": False},
            sr=16_000,
        )
        assert "loudnorm=" in chain
        assert "I=-16.0" in chain
        assert "LRA=7.0" in chain
        assert "linear=false" in chain

    def test_lufs_linear_mode_flag(self) -> None:
        chain = build_filter_chain({"lufs_target": -14.0, "loudnorm_linear": True}, sr=16_000)
        assert "linear=true" in chain

    def test_vad_trim_emits_silenceremove_with_threshold_and_duration(self) -> None:
        chain = build_filter_chain(
            {
                "vad_trim": True,
                "silence_threshold_db": -50.0,
                "silence_min_duration_s": 1.0,
            },
            sr=16_000,
        )
        assert "silenceremove" in chain
        assert "start_threshold=-50.0dB" in chain
        assert "stop_threshold=-50.0dB" in chain
        assert "start_duration=1.0" in chain

    def test_limiter_converts_db_to_linear(self) -> None:
        chain = build_filter_chain({"limiter_ceiling_db": -1.0}, sr=16_000)
        # -1 dB ≈ 0.8913 linear
        assert "alimiter=limit=0.89" in chain

    def test_notch_filter_uses_bandreject(self) -> None:
        chain = build_filter_chain({"notch_hz": 50}, sr=16_000)
        assert "bandreject=f=50" in chain

    def test_full_chain_order_matches_scipy(self) -> None:
        """The filter order should match PreprocessingPipeline's scipy order."""
        params = {
            "sample_rate": 16_000,
            "notch_hz": 50,
            "highpass_hz": 80,
            "lowpass_hz": 7000,
            "lufs_target": -16.0,
            "drc_ratio": 2.0,
            "limiter_ceiling_db": -1.0,
            "vad_trim": True,
        }
        chain = build_filter_chain(params, sr=16_000)
        parts = chain.split(",")
        order = [p.split("=")[0] for p in parts]
        expected_order = [
            "bandreject",
            "highpass",
            "lowpass",
            "loudnorm",
            "acompressor",
            "alimiter",
            "silenceremove",
        ]
        assert order == expected_order


# ---------------------------------------------------------------------------
# Binary probing
# ---------------------------------------------------------------------------


def test_is_ffmpeg_available_returns_bool() -> None:
    result = is_ffmpeg_available()
    assert isinstance(result, bool)


def test_run_ffmpeg_chain_empty_chain_is_noop() -> None:
    audio = np.ones(1000, dtype=np.float32) * 0.1
    out = run_ffmpeg_chain(audio, "", sr=16_000)
    np.testing.assert_array_equal(out, audio)


def test_run_ffmpeg_chain_raises_when_binary_missing(monkeypatch) -> None:
    """Force ``shutil.which('ffmpeg')`` to return None and expect the error."""
    import asrbench.preprocessing.ffmpeg_pipeline as fp

    monkeypatch.setattr(fp, "shutil", type("_S", (), {"which": staticmethod(lambda _: None)}))
    with pytest.raises(FFmpegNotAvailable):
        run_ffmpeg_chain(np.zeros(100, dtype=np.float32), "highpass=f=80", sr=16_000)


# ---------------------------------------------------------------------------
# Pipeline dispatch + graceful fallback
# ---------------------------------------------------------------------------


class TestPipelineBackendSwitch:
    def test_scipy_is_default(self) -> None:
        """Without explicit backend, scipy path handles the call."""
        audio = np.ones(1600, dtype=np.float32) * 0.05
        params = {"highpass_hz": 80}
        # Identity-enough sanity check: output shape preserved, no subprocess.
        out = PreprocessingPipeline.apply(audio, params, sr=16_000)
        assert out.shape == audio.shape

    def test_ffmpeg_backend_falls_back_when_unavailable(self, monkeypatch) -> None:
        """
        If the caller asks for backend=ffmpeg but ffmpeg is not on PATH, the
        pipeline must log a warning and fall through to scipy. No exception
        should surface to the benchmark loop — we'd rather run a slightly
        different pipeline than crash the whole study.
        """
        import asrbench.preprocessing.ffmpeg_pipeline as fp

        monkeypatch.setattr(fp, "shutil", type("_S", (), {"which": staticmethod(lambda _: None)}))

        audio = np.ones(1600, dtype=np.float32) * 0.05
        params = {"backend": "ffmpeg", "highpass_hz": 80}
        # Must not raise.
        out = PreprocessingPipeline.apply(audio, params, sr=16_000)
        assert out.shape == audio.shape

    @pytest.mark.skipif(not is_ffmpeg_available(), reason="ffmpeg binary not on PATH")
    def test_ffmpeg_backend_end_to_end_highpass(self) -> None:
        """
        Real subprocess smoke: feed a DC-biased signal through a highpass
        and verify the output is roughly zero-mean.
        """
        sr = 16_000
        # 1 second DC-biased sine — highpass should kill the DC component.
        t = np.arange(sr, dtype=np.float32) / sr
        audio = (0.5 + 0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        params = {"backend": "ffmpeg", "highpass_hz": 100}
        out = PreprocessingPipeline.apply(audio, params, sr=sr)
        # DC removed → mean near zero
        assert abs(float(np.mean(out))) < 0.05
