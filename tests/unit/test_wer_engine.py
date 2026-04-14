"""Unit tests for WEREngine — normalization and metric computation."""

from __future__ import annotations

import pytest

from asrbench.engine.wer import WEREngine, _normalize_arabic, _normalize_text


@pytest.fixture
def engine() -> WEREngine:
    return WEREngine()


class TestEnglishNormalization:
    def test_basic_wer_zero(self, engine: WEREngine) -> None:
        result = engine.compute(["hello world"], ["hello world"], lang="en")
        assert result["wer"] == pytest.approx(0.0)
        assert result["cer"] == pytest.approx(0.0)

    def test_basic_wer_nonzero(self, engine: WEREngine) -> None:
        # "hello world" vs "hello earth" — 1 substitution out of 2 words = 0.5 WER
        result = engine.compute(["hello world"], ["hello earth"], lang="en")
        assert result["wer"] == pytest.approx(0.5)

    def test_normalization_applied_symmetrically(self, engine: WEREngine) -> None:
        # Punctuation and case should be stripped by EN normalizer
        result = engine.compute(
            ["Hello, World!"],
            ["hello world"],
            lang="en",
        )
        assert result["wer"] == pytest.approx(0.0), (
            "EN normalization must strip punctuation and lowercase symmetrically"
        )

    def test_empty_hypothesis(self, engine: WEREngine) -> None:
        result = engine.compute(["hello world"], [""], lang="en")
        assert result["wer"] > 0

    def test_corpus_level_uses_all_pairs(self, engine: WEREngine) -> None:
        refs = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
        ]
        hyps = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
        ]
        result = engine.compute(refs, hyps, lang="en")
        assert result["wer"] == pytest.approx(0.0)

    def test_data_leakage_detection(self, engine: WEREngine) -> None:
        result = engine.compute(
            ["test sentence"],
            ["test sentence"],
            lang="en",
            model_family="whisper",
            dataset_source="librispeech",
        )
        assert result["data_leakage_warning"] is True

    def test_no_leakage_for_custom_dataset(self, engine: WEREngine) -> None:
        result = engine.compute(
            ["test sentence"],
            ["test sentence"],
            lang="en",
            model_family="whisper",
            dataset_source="common_voice",
        )
        assert result["data_leakage_warning"] is False


class TestTurkishNormalization:
    def test_turkish_lowercase(self, engine: WEREngine) -> None:
        result = engine.compute(
            ["Merhaba Dünya"],
            ["merhaba dünya"],
            lang="tr",
        )
        # After normalization both should be lowercase and match
        assert result["wer"] == pytest.approx(0.0)

    def test_turkish_dotless_i_pipeline_order(self, engine: WEREngine) -> None:
        # A1 regression: "IRAK" must normalize to "ırak" (dotless-ı), NOT "irak".
        # Before the fix, BasicTextNormalizer.lower() ran first, turning I→i, so
        # _turkish_lower() never had a chance to produce ı.
        # Normalization is now a module-level function (_normalize_text) so the
        # same cache survives across WEREngine re-instantiation; tests call it
        # directly rather than a bound method that no longer exists.
        del engine  # unused — kept for fixture symmetry with other tests
        result = _normalize_text("IRAK", "tr")
        assert result == "ırak", (
            f"Turkish I must lower to ı, not i. Got: {result!r}. "
            "Check that _turkish_lower() runs BEFORE BasicTextNormalizer."
        )

    def test_turkish_dotted_capital_i(self, engine: WEREngine) -> None:
        # İ (U+0130, dotted capital I) must lower to i, not ı.
        del engine
        result = _normalize_text("İSTANBUL", "tr")
        assert result == "istanbul", f"İ must lower to i. Got: {result!r}"

    def test_turkish_wer_zero_with_dotless_i(self, engine: WEREngine) -> None:
        # End-to-end: ref "IRAK" and hyp "ırak" must match after normalization.
        result = engine.compute(["IRAK'a gittim"], ["ırak'a gittim"], lang="tr")
        assert result["wer"] == pytest.approx(0.0)


class TestDefaultPipeline:
    def test_non_english_no_contraction_expansion(self, engine: WEREngine) -> None:
        # A2 regression: non-EN languages must NOT use EnglishTextNormalizer.
        # EnglishTextNormalizer would expand "won't" → "will not", raising WER.
        result = engine.compute(["won't"], ["won't"], lang="de")
        assert result["wer"] == pytest.approx(0.0), (
            "German (and other non-EN) text must not be passed through "
            "EnglishTextNormalizer. Contraction expansion inflates WER."
        )

    def test_non_english_won_t_not_expanded(self, engine: WEREngine) -> None:
        # Verify directly that "won't" is NOT transformed by the default pipeline.
        del engine
        normalized = _normalize_text("won't", "de")
        assert "will not" not in normalized, (
            f"Default pipeline must not expand contractions. Got: {normalized!r}"
        )


class TestArabicNormalization:
    def test_diacritics_removed(self, engine: WEREngine) -> None:
        # A3: harakat must be stripped
        del engine
        result = _normalize_text("بِسْمِ اللَّهِ", "ar")
        # No diacritic code points (U+064B–U+065F, U+0670) should remain
        for ch in result:
            assert "\u064b" > ch or ch > "\u065f", (
                f"Diacritic U+{ord(ch):04X} found in Arabic normalized output: {result!r}"
            )

    def test_arabic_wer_zero_with_diacritics(self, engine: WEREngine) -> None:
        # Ref with diacritics vs hyp without must score 0 WER after normalization
        result = engine.compute(["بِسْمِ اللَّهِ"], ["بسم الله"], lang="ar")
        assert result["wer"] == pytest.approx(0.0)

    def test_normalize_arabic_helper_alef(self) -> None:
        # Alef variants (أ U+0623, إ U+0625, آ U+0622) must normalize to bare alef (ا U+0627).
        # Note: "أهلاً" also contains tanwin (ً U+064B), a diacritic that is removed too.
        assert _normalize_arabic("أهلاً") == "اهلا"
        assert _normalize_arabic("إسلام") == "اسلام"
        assert _normalize_arabic("آمين") == "امين"

    def test_normalize_arabic_helper_tatweel(self) -> None:
        assert _normalize_arabic("مـرحـبا") == "مرحبا"

    def test_normalize_arabic_helper_alef_maqsura(self) -> None:
        assert _normalize_arabic("على") == "علي"


class TestBootstrapCI:
    def test_ci_present_in_result(self, engine: WEREngine) -> None:
        result = engine.compute(
            ["the quick brown fox", "hello world"],
            ["the quick brown fox", "hello earth"],
            lang="en",
        )
        assert "wer_ci_lower" in result
        assert "wer_ci_upper" in result

    def test_ci_bounds_valid(self, engine: WEREngine) -> None:
        # wer_ci_lower <= wer <= wer_ci_upper
        refs = ["the quick brown fox jumps over the lazy dog"] * 10
        hyps = ["the quick brown fox jumps over the lazy cat"] * 10
        result = engine.compute(refs, hyps, lang="en")
        wer = result["wer"]
        assert result["wer_ci_lower"] <= wer + 1e-9, (
            f"CI lower ({result['wer_ci_lower']:.4f}) must be <= WER ({wer:.4f})"
        )
        assert result["wer_ci_upper"] >= wer - 1e-9, (
            f"CI upper ({result['wer_ci_upper']:.4f}) must be >= WER ({wer:.4f})"
        )

    def test_ci_single_pair_degenerate(self, engine: WEREngine) -> None:
        # Single pair falls back to (0.0, 1.0) — must not raise
        result = engine.compute(["hello"], ["hello"], lang="en")
        assert result["wer_ci_lower"] == pytest.approx(0.0)
        assert result["wer_ci_upper"] == pytest.approx(1.0)


class TestEdgeCases:
    def test_raises_on_empty_input(self, engine: WEREngine) -> None:
        with pytest.raises(ValueError, match="empty input"):
            engine.compute([], [], lang="en")

    def test_raises_on_length_mismatch(self, engine: WEREngine) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            engine.compute(["ref1", "ref2"], ["hyp1"], lang="en")

    def test_unicode_text(self, engine: WEREngine) -> None:
        result = engine.compute(
            ["café au lait"],
            ["cafe au lait"],
            lang="en",
        )
        # Should not raise — unicode handled gracefully
        assert "wer" in result

    def test_punctuation_only_ref(self, engine: WEREngine) -> None:
        # After normalization, punctuation-only becomes empty
        result = engine.compute(["..."], ["..."], lang="en")
        assert "wer" in result

    def test_all_metrics_present(self, engine: WEREngine) -> None:
        result = engine.compute(
            ["the cat sat on the mat"],
            ["the cat sat on the hat"],
            lang="en",
        )
        for key in (
            "wer",
            "cer",
            "mer",
            "wil",
            "wer_ci_lower",
            "wer_ci_upper",
            "wilcoxon_p",
            "data_leakage_warning",
        ):
            assert key in result

    def test_wilcoxon_none_for_small_sample(self, engine: WEREngine) -> None:
        result = engine.compute(["hello world"], ["hello world"], lang="en")
        assert result["wilcoxon_p"] is None
