"""WER/CER/MER/WIL metric computation with language-aware text normalization."""

from __future__ import annotations

import functools
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


# ---------------------------------------------------------------------------
# Module-level text normalization cache.
#
# Was previously a bound lru_cache on `WEREngine._normalize`, which made the
# cache instance-scoped: every new WEREngine() instance rebuilt the cache from
# scratch. Since a fresh WEREngine is created per BenchmarkEngine construction
# (i.e. once per /optimize/start background task), this caused every trial in
# a study to redo the exact same reference normalization — the 3600 dataset
# references were normalized 1× per trial rather than 1× total.
#
# Hoisting the cache to module scope makes it survive engine re-construction:
# references are normalized the first time they appear in any study and then
# hit the cache for every subsequent trial regardless of which WEREngine
# instance did the lookup. Hypothesis strings still miss (each trial produces
# fresh text) — that is unavoidable, but no longer the hot path.
#
# Size chosen to comfortably hold ~65k references + ~200k hypotheses before
# any LRU eviction kicks in (~25 MB peak at ~100 bytes/string).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=262144)
def _normalize_text(text: str, lang: str) -> str:
    """
    Language-aware text normalization with a module-level LRU cache.

    The core of WEREngine's per-segment preprocessing; see the WEREngine
    docstring for the per-language pipeline. Pure function: no side effects,
    deterministic for a given (text, lang). Safe to call concurrently — the
    underlying CPython GIL plus functools.lru_cache's own locking make the
    cache thread-safe for simple get/put usage.
    """
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
        speaker_ids: list[str | None] | None = None,
    ) -> dict:
        """
        Compute corpus-level WER/CER/MER/WIL metrics.

        Params:
        - refs: reference transcripts (ground truth)
        - hyps: hypothesis transcripts (model output)
        - lang: ISO 639-1 language code
        - model_family: for data leakage detection (optional)
        - dataset_source: for data leakage detection (optional)
        - speaker_ids: optional per-segment speaker label list, aligned with
                       refs/hyps. When at least 2 distinct non-None speakers
                       are present, the bootstrap CI switches to per-speaker
                       block resampling (Liu et al. 2020), which produces
                       correctly-sized intervals under speaker correlation.
                       Set to None for datasets without speaker labels
                       (FLEURS, earnings22, mediaspeech) — the classic
                       per-segment bootstrap is used as a fallback.

        Returns dict with keys:
            wer, cer, mer, wil, wer_ci_lower, wer_ci_upper, wer_ci_method,
            data_leakage_warning, lang_notes
        Raises ValueError if len(refs) != len(hyps) or refs is empty.

        Note: pairwise significance testing (Wilcoxon signed-rank on
        per-segment WER distributions) lives in ``CompareEngine`` so that
        two concrete runs can be compared against each other. The earlier
        single-run ``wilcoxon_p`` field compared a run's per-segment WER
        against a list of zeros, which is not a meaningful test and has
        been removed.
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
        if speaker_ids is not None and len(speaker_ids) != len(refs):
            raise ValueError(
                f"speaker_ids length ({len(speaker_ids)}) must match refs/hyps "
                f"length ({len(refs)}) when provided."
            )

        norm_refs = [_normalize_text(t, lang) for t in refs]
        norm_hyps = [_normalize_text(t, lang) for t in hyps]

        wer_out = process_words(norm_refs, norm_hyps)
        cer_out = process_characters(norm_refs, norm_hyps)

        leakage = self._check_leakage(model_family, dataset_source)
        ci_lower, ci_upper, ci_method = self._bootstrap_wer_ci(wer_out, speaker_ids=speaker_ids)

        return {
            "wer": wer_out.wer,
            "cer": cer_out.cer,
            "mer": wer_out.mer,
            "wil": wer_out.wil,
            "wer_ci_lower": ci_lower,
            "wer_ci_upper": ci_upper,
            "wer_ci_method": ci_method,
            "data_leakage_warning": leakage,
            "lang_notes": get_lang_notes(lang),
        }

    def _bootstrap_wer_ci(
        self,
        wer_out: Any,
        n_boot: int = 1000,
        seed: int = 42,
        speaker_ids: list[str | None] | None = None,
    ) -> tuple[float, float, str]:
        """
        Bootstrap 95% confidence interval on corpus WER.

        Two sampling strategies:

        **Per-segment (classic)** — `speaker_ids` is None or carries fewer
        than 2 distinct labels. Each bootstrap draw resamples segment indices
        with replacement from [0, n), and corpus WER is recomputed as
        `sum(errors) / sum(ref_lens)` across the resampled indices. This is
        the standard approach and is what Bisani & Ney 2004 (ICASSP) describe.

        **Blockwise per-speaker** — `speaker_ids` carries ≥ 2 distinct
        non-None labels. Segments are grouped by speaker; each bootstrap
        draw resamples SPEAKERS (not segments) with replacement, then
        concatenates all segments of every drawn speaker. This captures the
        within-speaker correlation that the per-segment variant ignores and
        produces wider — more honest — CIs on speaker-heterogeneous corpora.
        Method described in Liu et al. 2020, "Statistical Testing on ASR
        Performance via Blockwise Bootstrap" (Interspeech).

        Returns `(ci_lower, ci_upper, method)` where method is
        "per_segment" | "blockwise_speaker" so downstream callers can log
        which CI width they are comparing. Degenerate fallback `(0.0, 1.0,
        "degenerate")` for fewer than 2 segments.
        """
        n = len(wer_out.alignments)
        if n < 2:
            return (0.0, 1.0, "degenerate")
        if n < 10:
            # Bootstrap on fewer than ~10 segments is degenerate: the resample
            # distribution has too few distinct orderings to approximate the
            # sampling distribution of the corpus WER, and the returned CI
            # tends to collapse onto a handful of error-count ratios. We still
            # return a number (the optimizer's significance gate needs a
            # concrete interval) but warn loudly so the caller knows rankings
            # built on this run are below the noise floor.
            logger.warning(
                "_bootstrap_wer_ci: only %d segments — CI is unreliable. "
                "Community norm for publication-quality WER is >=1h audio / "
                ">=8000 words (Speechmatics accuracy benchmarking, Open ASR "
                "Leaderboard 2024). Consider a larger max_duration_s.",
                n,
            )

        # Extract per-segment errors and ref-lengths from existing alignment.
        # Tight int32 arrays + tuple-compare on the alignment chunk type avoid
        # per-chunk attribute lookups dominating the loop for large corpora.
        errors = np.empty(n, dtype=np.int32)
        ref_lens = np.empty(n, dtype=np.int32)
        alignments = wer_out.alignments
        references = wer_out.references
        for i in range(n):
            seg_errors = 0
            for chunk in alignments[i]:
                ctype = chunk.type
                if ctype == "substitute" or ctype == "delete":
                    seg_errors += chunk.ref_end_idx - chunk.ref_start_idx
                elif ctype == "insert":
                    seg_errors += chunk.hyp_end_idx - chunk.hyp_start_idx
            errors[i] = seg_errors
            ref_lens[i] = max(len(references[i]), 1)

        # Decide sampling mode: blockwise if we have >=2 distinct speakers.
        use_blockwise = False
        if speaker_ids is not None and len(speaker_ids) == n:
            distinct = {s for s in speaker_ids if s is not None}
            if len(distinct) >= 2:
                use_blockwise = True

        rng = np.random.default_rng(seed)
        errors_f = errors.astype(np.float64)
        ref_lens_f = ref_lens.astype(np.float64)

        if use_blockwise:
            # Group segment indices by speaker. Segments with speaker_id=None
            # are treated as their own singleton "block" so we never silently
            # drop them — matches the per-segment behavior for unlabeled rows.
            assert speaker_ids is not None  # narrowed by use_blockwise guard
            groups: dict[str, list[int]] = {}
            for i, sid in enumerate(speaker_ids):
                key = sid if sid is not None else f"__unk_{i}"
                groups.setdefault(key, []).append(i)
            block_keys = list(groups.keys())
            block_indices = [np.array(groups[k], dtype=np.int64) for k in block_keys]
            n_blocks = len(block_keys)

            # Precompute per-block sums so each bootstrap draw is an O(n_blocks)
            # lookup rather than an O(n) resegment. This keeps blockwise as
            # fast as the per-segment path for n_boot=1000.
            block_err_sums = np.array(
                [errors_f[ix].sum() for ix in block_indices], dtype=np.float64
            )
            block_len_sums = np.array(
                [ref_lens_f[ix].sum() for ix in block_indices], dtype=np.float64
            )

            block_draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks), dtype=np.int32)
            boot_err_sums = block_err_sums[block_draws].sum(axis=1)
            boot_len_sums = block_len_sums[block_draws].sum(axis=1)
            # Avoid division by zero if a draw somehow yields zero refs.
            boot_len_sums = np.where(boot_len_sums < 1e-9, 1.0, boot_len_sums)
            boot_wers = boot_err_sums / boot_len_sums

            lo, hi = np.percentile(boot_wers, [2.5, 97.5])
            return (float(lo), float(hi), "blockwise_speaker")

        # Per-segment fallback (classic Bisani & Ney 2004).
        all_indices = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
        boot_wers = errors_f[all_indices].sum(axis=1) / ref_lens_f[all_indices].sum(axis=1)

        lo, hi = np.percentile(boot_wers, [2.5, 97.5])
        return (float(lo), float(hi), "per_segment")

    def _check_leakage(self, model_family: str | None, dataset_source: str | None) -> bool:
        if model_family is None or dataset_source is None:
            return False
        return (
            model_family.lower() in self._DATA_LEAKAGE_MODELS
            and dataset_source.lower() in self._DATA_LEAKAGE_DATASETS
        )
