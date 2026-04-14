"""Tests for BenchmarkEngine._split_params and preprocessing integration."""

from __future__ import annotations

from asrbench.engine.benchmark import BenchmarkEngine


class TestSplitParams:
    """Verify _split_params separates backend and preprocessing params correctly."""

    def test_no_preprocess_params(self) -> None:
        params = {"beam_size": 5, "temperature": 0.0}
        backend, preprocess = BenchmarkEngine._split_params(params)
        assert backend == {"beam_size": 5, "temperature": 0.0}
        assert preprocess == {}

    def test_only_preprocess_params(self) -> None:
        params = {
            "preprocess.lufs_target": -16.0,
            "preprocess.highpass_hz": 80,
        }
        backend, preprocess = BenchmarkEngine._split_params(params)
        assert backend == {}
        assert preprocess == {"lufs_target": -16.0, "highpass_hz": 80}

    def test_mixed_params(self) -> None:
        params = {
            "beam_size": 5,
            "temperature": 0.0,
            "preprocess.lufs_target": -16.0,
            "preprocess.highpass_hz": 80,
            "preprocess.vad_trim": True,
        }
        backend, preprocess = BenchmarkEngine._split_params(params)
        assert backend == {"beam_size": 5, "temperature": 0.0}
        assert preprocess == {
            "lufs_target": -16.0,
            "highpass_hz": 80,
            "vad_trim": True,
        }

    def test_empty_params(self) -> None:
        backend, preprocess = BenchmarkEngine._split_params({})
        assert backend == {}
        assert preprocess == {}

    def test_prefix_stripping_correct(self) -> None:
        """Ensure only the 'preprocess.' prefix is stripped, not partial matches."""
        params = {
            "preprocess.noise_reduce": 0.5,
            "preprocessor_version": "1.0",
        }
        backend, preprocess = BenchmarkEngine._split_params(params)
        assert backend == {"preprocessor_version": "1.0"}
        assert preprocess == {"noise_reduce": 0.5}
