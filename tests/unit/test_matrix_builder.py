"""Unit tests for MatrixBuilder — cartesian product and baseline ordering."""

from __future__ import annotations

import pytest

from asrbench.engine.matrix import MatrixBuilder


@pytest.fixture
def builder() -> MatrixBuilder:
    return MatrixBuilder()


DEFAULT_PARAMS = {"beam_size": 5, "compute_type": "float16", "language": None}


class TestBuildMatrix:
    def test_single_param_single_value(self, builder: MatrixBuilder) -> None:
        runs = builder.build_matrix({"beam_size": [5]}, DEFAULT_PARAMS, "param_compare")
        assert len(runs) == 1
        assert runs[0].is_baseline is True
        assert runs[0].params["beam_size"] == 5

    def test_single_param_multiple_values(self, builder: MatrixBuilder) -> None:
        runs = builder.build_matrix({"beam_size": [1, 2, 4, 8]}, DEFAULT_PARAMS, "param_compare")
        assert len(runs) == 4
        assert runs[0].is_baseline is True, "First run must be baseline"
        assert runs[0].params["beam_size"] == 1, "Baseline uses first value"

    def test_cartesian_product(self, builder: MatrixBuilder) -> None:
        runs = builder.build_matrix(
            {"beam_size": [1, 4], "compute_type": ["float16", "int8"]},
            DEFAULT_PARAMS,
            "param_compare",
        )
        assert len(runs) == 4  # 2 × 2
        param_combos = {(r.params["beam_size"], r.params["compute_type"]) for r in runs}
        assert param_combos == {(1, "float16"), (1, "int8"), (4, "float16"), (4, "int8")}

    def test_baseline_is_first(self, builder: MatrixBuilder) -> None:
        runs = builder.build_matrix({"beam_size": [1, 2, 4]}, DEFAULT_PARAMS, "param_compare")
        assert runs[0].is_baseline is True
        for run in runs[1:]:
            assert run.is_baseline is False

    def test_only_one_baseline(self, builder: MatrixBuilder) -> None:
        runs = builder.build_matrix(
            {"beam_size": [1, 2, 4], "compute_type": ["float16", "int8"]},
            DEFAULT_PARAMS,
            "param_compare",
        )
        baselines = [r for r in runs if r.is_baseline]
        assert len(baselines) == 1

    def test_default_params_merged(self, builder: MatrixBuilder) -> None:
        runs = builder.build_matrix(
            {"beam_size": [1]},
            {"beam_size": 5, "compute_type": "float16"},
            "param_compare",
        )
        assert runs[0].params["compute_type"] == "float16", "default_params must be merged"

    def test_empty_matrix_raises(self, builder: MatrixBuilder) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            builder.build_matrix({}, DEFAULT_PARAMS, "param_compare")

    def test_empty_value_list_raises(self, builder: MatrixBuilder) -> None:
        with pytest.raises(ValueError, match="'beam_size' has no values"):
            builder.build_matrix({"beam_size": []}, DEFAULT_PARAMS, "param_compare")
