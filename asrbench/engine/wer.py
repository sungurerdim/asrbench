"""WER/CER/MER/WIL metric computation with language-aware text normalization."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import numpy as np
from jiwer import process_characters, process_words
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

logger = logging.getLogger(__name__)

_tr_normalize: Any = None

try:
    from trnorm import normalize as _trnorm_fn  # type: ignore[import-untyped]

    _tr_normalize = _trnorm_fn
    _TRNORM_AVAILABLE = True
except ImportError:
    _TRNORM_AVAILABLE = False
    logger.warning(
        "trnorm not installed — Turkish normalization degraded. Install: pip install trnorm"
    )

_en_normalizer = EnglishTextNormalizer()
_basic_normalizer = BasicTextNormalizer()

# Language-specific advisory notes for interpreting WER/CER results.
# Shown alongside benchmark results to aid correct interpretation.
_LANG_NOTES: dict[str, list[str]] = {
    "tr": [
        "Turkish is agglutinative: a single word can represent a multi-word English "
        "phrase, inflating WER. CER is a more reliable accuracy indicator for Turkish.",
        "Turkish-safe lowercasing is applied (I\u2192\u0131, \u0130\u2192i) to prevent "
        "incorrect case folding from standard .lower().",
    ],
    "zh": [
        "Chinese has no word boundaries. WER is computed on space-separated tokens; "
        "CER is the primary metric for Chinese.",
    ],
    "ja": [
        "Japanese uses mixed scripts with no word spaces. "
        "CER is more meaningful than WER for Japanese.",
    ],
    "ar": [
        "Arabic morphology is complex; cliticization can inflate WER. "
        "CER provides a complementary view.",
    ],
    "fi": [
        "Finnish is agglutinative. WER may be inflated; CER is a more reliable indicator.",
    ],
    "hu": [
        "Hungarian is agglutinative. WER may be inflated; CER is a more reliable indicator.",
    ],
    "ko": [
        "Korean is agglutinative. WER may be inflated; CER is a more reliable indicator.",
    ],
}


def get_lang_notes(lang: str) -> list[str]:
    """Return language-specific advisory notes for interpreting WER/CER scores."""
    return list(_LANG_NOTES.get(lang, []))


_RE_AR_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_RE_AR_ALEF = re.compile(r"[\u0622\u0623\u0625]")


def _normalize_arabic(text: str) -> str:
    """
    Arabic-specific normalization: remove diacritics (harakat), normalize alef
    variants, remove tatweel (kashida), and normalize alef maqsura.

    No external dependencies — stdlib re + unicodedata only.
    """
    text = _RE_AR_DIACRITICS.sub("", text)
    text = _RE_AR_ALEF.sub("\u0627", text)
    text = text.replace("\u0640", "")
    text = text.replace("\u0649", "\u064a")
    return text


def _turkish_lower(text: str) -> str:
    """
    Turkish-safe lowercase: apply İ→i and I→ı before .lower().

    Standard Python .lower() maps I→i, which is correct for Latin scripts but
    wrong for Turkish where dotless-I (I) should lower to ı, not i.
    NFC normalization must be applied before calling this function.
    """
    return text.replace("\u0130", "i").replace("I", "\u0131").lower()


class WEREngine:
    """
    Compute WER/CER/MER/WIL for a list of (reference, hypothesis) pairs.

    Normalization is always applied symmetrically to BOTH ref and hyp.
    EN  pipeline: EnglishTextNormalizer → lowercase
    TR  pipeline: NFC → Turkish-safe lowercase → BasicTextNormalizer → trnorm (if available)
    AR  pipeline: Arabic diacritic/alef normalization → BasicTextNormalizer → lowercase
    ZH/JA pipeline: NFKC → BasicTextNormalizer → lowercase
    KO  pipeline: NFC → BasicTextNormalizer → lowercase
    Default (all other langs): BasicTextNormalizer → lowercase
    """

    _DATA_LEAKAGE_MODELS = {"whisper", "openai-whisper"}
    _DATA_LEAKAGE_DATASETS = {"librispeech", "fleurs"}

    def compute(
        self,
        refs: list[str],
        hyps: list[str],
        lang: str,
        model_family: str | None = None,
        dataset_source: str | None = None,
    ) -> dict:
        """
        Compute corpus-level WER/CER/MER/WIL metrics.

        Params:
        - refs: reference transcripts (ground truth)
        - hyps: hypothesis transcripts (model output)
        - lang: ISO 639-1 language code
        - model_family: for data leakage detection (optional)
        - dataset_source: for data leakage detection (optional)

        Returns dict with keys:
            wer, cer, mer, wil, wer_ci_lower, wer_ci_upper,
            wilcoxon_p, data_leakage_warning, lang_notes
        Raises ValueError if len(refs) != len(hyps) or refs is empty.
        """
        if len(refs) != len(hyps):
            raise ValueError(
                f"refs and hyps length mismatch: {len(refs)} vs {len(hyps)}. "
                "Both lists must have the same number of entries."
            )
        if not refs:
            raise ValueError(
                "Cannot compute WER on empty input. Provide at least one (ref, hyp) pair."
            )

        norm_refs = [self._normalize(t, lang) for t in refs]
        norm_hyps = [self._normalize(t, lang) for t in hyps]

        wer_out = process_words(norm_refs, norm_hyps)
        cer_out = process_characters(norm_refs, norm_hyps)

        wilcoxon_p = self._wilcoxon(wer_out) if len(refs) >= 100 else None
        leakage = self._check_leakage(model_family, dataset_source)
        ci_lower, ci_upper = self._bootstrap_wer_ci(norm_refs, norm_hyps)

        return {
            "wer": wer_out.wer,
            "cer": cer_out.cer,
            "mer": wer_out.mer,
            "wil": wer_out.wil,
            "wer_ci_lower": ci_lower,
            "wer_ci_upper": ci_upper,
            "wilcoxon_p": wilcoxon_p,
            "data_leakage_warning": leakage,
            "lang_notes": get_lang_notes(lang),
        }

    def _normalize(self, text: str, lang: str) -> str:
        if lang == "tr":
            text = unicodedata.normalize("NFC", text)
            text = _turkish_lower(text)  # FIRST: I→ı, İ→i — before BasicTextNormalizer.lower()
            text = _basic_normalizer(text)  # .lower() now harmless (already lowercase)
            if _TRNORM_AVAILABLE and _tr_normalize is not None:
                text = unicodedata.normalize("NFC", str(_tr_normalize(text)))
            return text
        if lang == "en":
            text = _en_normalizer(text)
            return text.lower()
        if lang == "ar":
            text = _normalize_arabic(text)
            text = _basic_normalizer(text)
            return text.lower()
        if lang in ("zh", "ja"):
            # NFKC converts full-width (Ａ→a, １→1) and half-width Katakana
            text = unicodedata.normalize("NFKC", text)
            text = _basic_normalizer(text)
            return text.lower()
        if lang == "ko":
            # NFC ensures consistent Hangul syllable block representation
            text = unicodedata.normalize("NFC", text)
            text = _basic_normalizer(text)
            return text.lower()
        # Default: all non-EN languages use BasicTextNormalizer (not English-specific)
        text = _basic_normalizer(text)
        return text.lower()

    def _bootstrap_wer_ci(
        self,
        refs: list[str],
        hyps: list[str],
        n_boot: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Bootstrap 95% confidence interval on corpus WER via per-segment resampling.

        Pre-computes per-segment edit counts once, then resamples with numpy —
        avoids calling process_words() 1000 times on the full corpus.

        Returns (ci_lower, ci_upper) — the 2.5th and 97.5th percentiles of the
        bootstrap distribution. Requires at least 2 segments; returns (0.0, 1.0)
        as a degenerate fallback for single-segment input.
        """
        n = len(refs)
        if n < 2:
            return (0.0, 1.0)

        # Pre-compute per-segment errors and ref-lengths (one process_words call each)
        errors = np.empty(n, dtype=np.float64)
        ref_lens = np.empty(n, dtype=np.float64)
        for i in range(n):
            try:
                out = process_words([refs[i]], [hyps[i]])
                errors[i] = out.substitutions + out.insertions + out.deletions
                ref_lens[i] = max(len(refs[i].split()), 1)
            except Exception:
                errors[i] = 0.0
                ref_lens[i] = 1.0

        # Bootstrap: resample indices, compute corpus WER = sum(errors) / sum(ref_lens)
        rng = np.random.default_rng(seed)
        all_indices = rng.integers(0, n, size=(n_boot, n))
        boot_errors = errors[all_indices]  # (n_boot, n)
        boot_refs = ref_lens[all_indices]  # (n_boot, n)
        boot_wers = boot_errors.sum(axis=1) / boot_refs.sum(axis=1)

        return (float(np.percentile(boot_wers, 2.5)), float(np.percentile(boot_wers, 97.5)))

    def _wilcoxon(self, wer_out: Any) -> float | None:
        """Wilcoxon signed-rank test on per-segment WER. Returns p-value or None on failure."""
        try:
            from scipy.stats import wilcoxon

            # Extract per-segment WER from existing alignment data
            seg_wers = []
            for alignment, ref_tokens in zip(wer_out.alignments, wer_out.references):
                n_ref = max(len(ref_tokens), 1)
                errors = sum(1 for chunk in alignment if chunk.type != "equal")
                seg_wers.append(errors / n_ref)
            seg_zeros = [0.0] * len(seg_wers)
            result = wilcoxon(seg_zeros, seg_wers, zero_method="wilcox")
            return float(result[1])  # type: ignore[arg-type]  # WilcoxonResult[1] == p-value
        except Exception:
            return None

    def _check_leakage(self, model_family: str | None, dataset_source: str | None) -> bool:
        if model_family is None or dataset_source is None:
            return False
        return (
            model_family.lower() in self._DATA_LEAKAGE_MODELS
            and dataset_source.lower() in self._DATA_LEAKAGE_DATASETS
        )
