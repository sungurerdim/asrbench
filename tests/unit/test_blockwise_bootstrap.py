"""
Regression tests for blockwise (per-speaker) bootstrap WER CI.

Liu et al. 2020 (Interspeech), "Statistical Testing on ASR Performance via
Blockwise Bootstrap", shows that naive per-segment bootstrap under-estimates
the confidence interval width whenever segments are correlated within speakers
— which is always true in practice for read-speech corpora where a single
speaker contributes many consecutive utterances. The blockwise variant
resamples SPEAKERS with replacement (concatenating all their segments) so
the inflation introduced by within-speaker correlation is preserved in the
bootstrap distribution.

These tests lock in two things:

1. **Behavior switch**: WEREngine.compute must route through the blockwise
   path when speaker_ids carries >= 2 distinct labels, and through the
   per-segment path otherwise. The ci_method field in the result dict
   advertises which path ran so callers can log / audit it.

2. **Correctness of inflation**: on a synthetic corpus with strong within-
   speaker correlation, the blockwise CI must be *wider* than the per-
   segment CI for the same (refs, hyps). A narrower blockwise CI would
   indicate a bug in the sampling logic.
"""

from __future__ import annotations

import pytest

from asrbench.engine.wer import WEREngine


@pytest.fixture
def engine() -> WEREngine:
    return WEREngine()


def _make_correlated_corpus() -> tuple[list[str], list[str], list[str | None]]:
    """
    Build a synthetic EN corpus where errors cluster heavily inside two
    speakers out of four — the classic speaker-correlation scenario.

    Speaker A: 15 clean utterances (WER = 0) — "hello world"
    Speaker B: 15 utterances with 100% WER — "goodbye moon" hypothesized as "x y"
    Speaker C: 15 clean utterances (WER = 0) — "the quick brown fox"
    Speaker D: 15 utterances with 100% WER — "pack my box" hypothesized as "a b c"

    True corpus WER: ~50%. Per-segment bootstrap mixes individual
    utterances so every draw gets a smooth WER near the mean. Blockwise
    bootstrap draws entire speakers: a draw with 4 copies of speaker A
    has WER=0, a draw with 4 copies of speaker B has WER~1.0, giving the
    distribution a much wider spread.
    """
    refs: list[str] = []
    hyps: list[str] = []
    speaker_ids: list[str | None] = []

    # Speaker A — clean
    for _ in range(15):
        refs.append("hello world")
        hyps.append("hello world")
        speaker_ids.append("A")
    # Speaker B — all errors
    for _ in range(15):
        refs.append("goodbye moon")
        hyps.append("x y")
        speaker_ids.append("B")
    # Speaker C — clean
    for _ in range(15):
        refs.append("the quick brown fox")
        hyps.append("the quick brown fox")
        speaker_ids.append("C")
    # Speaker D — all errors
    for _ in range(15):
        refs.append("pack my box")
        hyps.append("a b c")
        speaker_ids.append("D")

    return refs, hyps, speaker_ids


class TestMethodSelection:
    def test_per_segment_when_speaker_ids_none(self, engine: WEREngine) -> None:
        refs, hyps, _ = _make_correlated_corpus()
        result = engine.compute(refs, hyps, lang="en", speaker_ids=None)
        assert result["wer_ci_method"] == "per_segment"

    def test_per_segment_when_all_speakers_none(self, engine: WEREngine) -> None:
        refs, hyps, _ = _make_correlated_corpus()
        all_none: list[str | None] = [None] * len(refs)
        result = engine.compute(refs, hyps, lang="en", speaker_ids=all_none)
        assert result["wer_ci_method"] == "per_segment"

    def test_per_segment_when_only_one_distinct_speaker(self, engine: WEREngine) -> None:
        refs, hyps, _ = _make_correlated_corpus()
        same_speaker: list[str | None] = ["solo" for _ in range(len(refs))]
        result = engine.compute(refs, hyps, lang="en", speaker_ids=same_speaker)
        # Only one distinct speaker → blockwise has nothing to resample,
        # we fall back to per-segment for a non-degenerate CI.
        assert result["wer_ci_method"] == "per_segment"

    def test_blockwise_when_two_or_more_speakers(self, engine: WEREngine) -> None:
        refs, hyps, speaker_ids = _make_correlated_corpus()
        result = engine.compute(refs, hyps, lang="en", speaker_ids=speaker_ids)
        assert result["wer_ci_method"] == "blockwise_speaker"

    def test_speaker_ids_length_mismatch_raises(self, engine: WEREngine) -> None:
        refs = ["a b c", "d e f"]
        hyps = ["a b c", "d e f"]
        with pytest.raises(ValueError, match="speaker_ids length"):
            engine.compute(refs, hyps, lang="en", speaker_ids=["one"])


class TestInflationOnCorrelatedData:
    def test_blockwise_ci_is_wider_than_per_segment(self, engine: WEREngine) -> None:
        """
        On a corpus where errors cluster within speakers, the blockwise CI
        must be meaningfully wider than the per-segment CI — this is the
        whole point of Liu et al. 2020.
        """
        refs, hyps, speaker_ids = _make_correlated_corpus()

        per_seg = engine.compute(refs, hyps, lang="en", speaker_ids=None)
        blockwise = engine.compute(refs, hyps, lang="en", speaker_ids=speaker_ids)

        per_seg_width = per_seg["wer_ci_upper"] - per_seg["wer_ci_lower"]
        blockwise_width = blockwise["wer_ci_upper"] - blockwise["wer_ci_lower"]

        # Both methods agree on the point estimate
        assert per_seg["wer"] == pytest.approx(blockwise["wer"], rel=1e-9)

        # But blockwise CI is wider. Concretely: on this corpus per-segment
        # width is typically around 0.12-0.18, blockwise is around 0.6-0.9.
        # We use a 2× floor so the test is not flaky under a different seed.
        assert blockwise_width > per_seg_width * 2.0, (
            f"Blockwise CI should be clearly wider than per-segment on "
            f"speaker-correlated data. Got per_seg_width={per_seg_width:.4f}, "
            f"blockwise_width={blockwise_width:.4f}"
        )

    def test_blockwise_ci_covers_true_wer(self, engine: WEREngine) -> None:
        """Sanity: the blockwise 95% CI should contain the corpus WER estimate."""
        refs, hyps, speaker_ids = _make_correlated_corpus()
        result = engine.compute(refs, hyps, lang="en", speaker_ids=speaker_ids)
        assert result["wer_ci_lower"] <= result["wer"] <= result["wer_ci_upper"]


class TestBackwardCompatibility:
    def test_default_call_still_works_without_speaker_arg(self, engine: WEREngine) -> None:
        """Existing call sites that don't pass speaker_ids must keep working."""
        result = engine.compute(
            ["hello world"] * 20,
            ["hello word"] * 20,
            lang="en",
        )
        assert "wer_ci_lower" in result
        assert "wer_ci_upper" in result
        assert result["wer_ci_method"] == "per_segment"
