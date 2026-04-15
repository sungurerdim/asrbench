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
        # 10 backend + 16 preprocessing
        #   7 original (format, sample_rate, lufs_target, highpass_hz,
        #               drc_ratio, noise_reduce, vad_trim)
        # + 4 v3 shaping (notch, lowpass, limiter_ceiling, preemph)
        # + 2 loudnorm extensions (lufs_lra, loudnorm_linear)
        # + 2 silence extensions (silence_threshold_db, silence_min_duration_s)
        # + 1 backend selector (scipy|ffmpeg)
        assert len(space.parameters) == 26


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
        assert len(preprocess_names) == 16

    def test_preprocess_param_names(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        preprocess_names = sorted(n for n in space.names if n.startswith("preprocess."))
        expected = sorted(
            [
                "preprocess.backend",
                "preprocess.format",
                "preprocess.sample_rate",
                "preprocess.lufs_target",
                "preprocess.lufs_lra",
                "preprocess.loudnorm_linear",
                "preprocess.highpass_hz",
                "preprocess.drc_ratio",
                "preprocess.noise_reduce",
                "preprocess.vad_trim",
                "preprocess.silence_threshold_db",
                "preprocess.silence_min_duration_s",
                "preprocess.notch_hz",
                "preprocess.lowpass_hz",
                "preprocess.limiter_ceiling_db",
                "preprocess.preemph_coef",
            ]
        )
        assert preprocess_names == expected


class TestBackendParamCrossCheck:
    def test_space_backend_keys_are_subset_of_backend_defaults(self) -> None:
        """
        Every backend parameter listed in the space YAML must be a known
        ``faster_whisper.default_params()`` key.

        The reverse direction (backend ⊆ space) is NOT asserted: the backend
        exposes a superset of tuning knobs, and a curated IAMS space is
        expected to pick a minimal, high-leverage subset. Requiring the full
        set would force every new backend knob into every YAML.
        """
        from asrbench.backends.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        backend_keys = set(backend.default_params().keys())

        space = ParameterSpace.from_yaml(SPACE_PATH)
        space_backend_keys = {n for n in space.names if not n.startswith("preprocess.")}

        missing_from_backend = space_backend_keys - backend_keys
        assert not missing_from_backend, (
            f"Space YAML references parameters that are NOT in "
            f"FasterWhisperBackend.default_params(): {sorted(missing_from_backend)}"
        )

    def test_backend_default_values_match_space_intersection(self) -> None:
        """
        For each parameter that appears in BOTH the backend and the space,
        the default values must agree — otherwise an IAMS screening probe
        at "the default" means different things in code vs. config.
        """
        from asrbench.backends.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
        backend_defaults = backend.default_params()

        space = ParameterSpace.from_yaml(SPACE_PATH)
        space_defaults = space.defaults()

        for key, space_val in space_defaults.items():
            if key.startswith("preprocess."):
                continue
            if key not in backend_defaults:
                continue  # handled by test above; skip the value check here
            backend_val = backend_defaults[key]
            assert backend_val == space_val, (
                f"{key}: backend default={backend_val!r}, space default={space_val!r}"
            )


class TestEnumParams:
    def test_format_values(self) -> None:
        # v3 shaping space replaced raw-container variants (wav/flac/ogg)
        # with opus bitrate ladders so the encoder artifact sweep is
        # meaningful. "none" stays as the no-codec baseline.
        space = ParameterSpace.from_yaml(SPACE_PATH)
        fmt = space.get("preprocess.format")
        assert fmt.type == "enum"
        assert fmt.values is not None
        assert set(fmt.values) == {"none", "opus_64k", "opus_32k", "opus_24k"}

    def test_sample_rate_values(self) -> None:
        space = ParameterSpace.from_yaml(SPACE_PATH)
        sr = space.get("preprocess.sample_rate")
        assert sr.type == "enum"
        assert sr.values is not None
        assert set(sr.values) == {8000, 16000, 22050}
