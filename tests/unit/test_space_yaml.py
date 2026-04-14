"""Unit tests for faster_whisper_full.yaml — IAMS space validation and cross-checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from asrbench.engine.search.space import ParameterSpace

SPACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "asrbench"
    / "data"
    / "spaces"
    / "faster_whisper_full.yaml"
)


class TestSpaceLoads:
    def test_yaml_loads_without_error(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        assert len(space.parameters) > 0

    def test_parameter_count(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        assert len(space.parameters) == 17


class TestParameterRanges:
    @pytest.fixture()
    def space(self) -> ParameterSpace:
        return ParameterSpace.from_yaml(SPACE_PATH)

    def test_numeric_min_lt_max(self, space: ParameterSpace) -> None:
        for p in space.parameters:
            if p.type in ("float", "int"):
                assert p.min < p.max, f"{p.name}: min ({p.min}) must be < max ({p.max})"

    def test_default_in_range(self, space: ParameterSpace) -> None:
        for p in space.parameters:
            if p.type in ("float", "int"):
                assert p.min <= p.default <= p.max, (
                    f"{p.name}: default ({p.default}) not in [{p.min}, {p.max}]"
                )

    def test_stepped_grid_alignment(self, space: ParameterSpace) -> None:
        for p in space.parameters:
            if p.step is not None:
                span = float(p.max) - float(p.min)
                steps = span / float(p.step)
                assert abs(steps - round(steps)) < 1e-9, (
                    f"{p.name}: (max-min)/step = {steps}, expected integer grid alignment"
                )


class TestPreprocessParams:
    def test_preprocess_params_exist(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        preprocess_names = [n for n in space.names if n.startswith("preprocess.")]
        assert len(preprocess_names) == 7

    def test_preprocess_param_names(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        preprocess_names = sorted(n for n in space.names if n.startswith("preprocess."))
        expected = sorted(
            [
                "preprocess.format",
                "preprocess.sample_rate",
                "preprocess.lufs_target",
                "preprocess.highpass_hz",
                "preprocess.drc_ratio",
                "preprocess.noise_reduce",
                "preprocess.vad_trim",
            ]
        )
        assert preprocess_names == expected


class TestBackendParamCrossCheck:
    def test_backend_defaults_match_space(self) -> None:
        """Backend default_params() keys must match non-preprocess space param names."""
        from asrbench.backends.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        backend_keys = set(backend.default_params().keys())

        space = ParameterSpace.from_yaml(SPACE_PATH)
        space_backend_keys = {n for n in space.names if not n.startswith("preprocess.")}

        assert backend_keys == space_backend_keys, (
            f"Mismatch between backend default_params() and space YAML.\n"
            f"  Only in backend: {backend_keys - space_backend_keys}\n"
            f"  Only in space:   {space_backend_keys - backend_keys}"
        )

    def test_backend_default_values_match_space(self) -> None:
        """Backend default values must equal space YAML defaults for each parameter."""
        from asrbench.backends.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        backend_defaults = backend.default_params()

        space = ParameterSpace.from_yaml(SPACE_PATH)
        space_defaults = space.defaults()

        for key, backend_val in backend_defaults.items():
            space_val = space_defaults[key]
            assert backend_val == space_val, (
                f"{key}: backend default={backend_val!r}, space default={space_val!r}"
            )


class TestEnumParams:
    def test_format_values(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        fmt = space.get("preprocess.format")
        assert fmt.type == "enum"
        assert fmt.values is not None
        assert set(fmt.values) == {"none", "wav", "flac", "ogg"}

    def test_sample_rate_values(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        sr = space.get("preprocess.sample_rate")
        assert sr.type == "enum"
        assert sr.values is not None
        assert set(sr.values) == {8000, 16000, 22050}
