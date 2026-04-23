"""Unit tests for CompareEngine — params diff/same, deltas, Wilcoxon."""

from __future__ import annotations

import pytest

from asrbench.engine.compare import CompareEngine, CompareInput


@pytest.fixture
def engine() -> CompareEngine:
    return CompareEngine()


def _run(
    run_id: str,
    *,
    params: dict | None = None,
    wer: float | None = None,
    cer: float | None = None,
    rtfx: float | None = None,
    segment_wers: list[float] | None = None,
) -> CompareInput:
    return CompareInput(
        run_id=run_id,
        params=params or {},
        aggregate={
            "wer_mean": wer,
            "cer_mean": cer,
            "rtfx_mean": rtfx,
        },
        segment_wers=segment_wers,
    )


class TestValidation:
    def test_requires_at_least_two_runs(self, engine: CompareEngine) -> None:
        with pytest.raises(ValueError, match="at least 2 runs"):
            engine.compare([_run("a")])

    def test_baseline_id_must_match_an_input(self, engine: CompareEngine) -> None:
        with pytest.raises(ValueError, match="baseline_run_id"):
            engine.compare(
                [_run("a"), _run("b")],
                baseline_run_id="does-not-exist",
            )


class TestParamsDiff:
    def test_identical_params_are_in_params_same(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", params={"beam_size": 5, "vad_filter": True}),
                _run("b", params={"beam_size": 5, "vad_filter": True}),
            ]
        )
        assert sorted(result.params_same) == ["beam_size", "vad_filter"]
        assert result.params_diff == []

    def test_differing_params_go_to_params_diff(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", params={"beam_size": 5, "compute_type": "float16"}),
                _run("b", params={"beam_size": 1, "compute_type": "float16"}),
            ]
        )
        assert result.params_diff == ["beam_size"]
        assert result.params_same == ["compute_type"]

    def test_key_missing_from_one_counts_as_diff(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", params={"beam_size": 5}),
                _run("b", params={"beam_size": 5, "vad_filter": True}),
            ]
        )
        assert result.params_diff == ["vad_filter"]
        assert result.params_same == ["beam_size"]


class TestDeltas:
    def test_first_run_is_baseline_by_default(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", wer=0.20, cer=0.10, rtfx=5.0),
                _run("b", wer=0.15, cer=0.08, rtfx=6.0),
            ]
        )
        assert result.runs[0]["is_baseline"] is True
        assert result.runs[0]["delta_wer_mean"] == pytest.approx(0.0)
        assert result.runs[1]["delta_wer_mean"] == pytest.approx(-0.05)
        assert result.runs[1]["delta_cer_mean"] == pytest.approx(-0.02)
        assert result.runs[1]["delta_rtfx_mean"] == pytest.approx(1.0)

    def test_explicit_baseline_id(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", wer=0.10),
                _run("b", wer=0.20),
                _run("c", wer=0.30),
            ],
            baseline_run_id="b",
        )
        assert result.runs[0]["is_baseline"] is False
        assert result.runs[1]["is_baseline"] is True
        assert result.runs[0]["delta_wer_mean"] == pytest.approx(-0.10)
        assert result.runs[2]["delta_wer_mean"] == pytest.approx(0.10)

    def test_missing_metric_returns_none_delta(self, engine: CompareEngine) -> None:
        result = engine.compare([_run("a", wer=0.15), _run("b")])  # b has no wer
        assert result.runs[1]["delta_wer_mean"] is None


class TestWilcoxon:
    def test_none_when_only_one_run_has_segments(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", segment_wers=[0.1, 0.2, 0.1]),
                _run("b", segment_wers=None),
            ]
        )
        assert result.wilcoxon_p is None

    def test_none_when_more_than_two_runs(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", segment_wers=[0.1] * 20),
                _run("b", segment_wers=[0.2] * 20),
                _run("c", segment_wers=[0.3] * 20),
            ]
        )
        assert result.wilcoxon_p is None

    def test_wilcoxon_rejects_null_when_runs_clearly_differ(self, engine: CompareEngine) -> None:
        pytest.importorskip("scipy")
        # Run B consistently worse than A — paired test should find a
        # highly significant difference.
        rng = [0.1 + 0.001 * i for i in range(30)]
        a = rng
        b = [x + 0.05 for x in rng]
        result = engine.compare(
            [
                _run("a", segment_wers=a),
                _run("b", segment_wers=b),
            ]
        )
        assert result.wilcoxon_p is not None
        assert result.wilcoxon_p < 0.05

    def test_wilcoxon_none_when_lengths_differ(self, engine: CompareEngine) -> None:
        result = engine.compare(
            [
                _run("a", segment_wers=[0.1, 0.2]),
                _run("b", segment_wers=[0.1, 0.2, 0.3]),
            ]
        )
        assert result.wilcoxon_p is None
