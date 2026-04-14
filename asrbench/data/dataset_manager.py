"""Dataset manager — fetch, resample, segment, and cache datasets for benchmarking."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

logger = logging.getLogger(__name__)

_TARGET_SR = 16_000
_SILENCE_GAP_S = 0.3

# Speaking-rate heuristic used to convert duration → approximate word count
# for the reliability warning. 150 wpm = ~2.5 words/sec is a corpus-average
# figure across EN/TR/DE read speech (covers FLEURS, LibriSpeech, CommonVoice).
# It is intentionally conservative: agglutinative languages (tr, fi, hu) emit
# fewer surface words per second, so the real figure is usually lower and the
# warning triggers slightly earlier for them — which is the correct behavior
# because those languages ALSO have less published WER research to lean on.
_APPROX_WORDS_PER_SECOND = 2.5

# Empirical reliability thresholds drawn from:
#   - Speechmatics accuracy benchmarking docs: ">= 10,000 words for
#     statistically meaningful WER comparison" at 0.5% delta resolution
#   - Bisani & Ney 2004 (ICASSP): bootstrap CI half-width scales as 1/sqrt(N)
#   - Open ASR Leaderboard 2024: all published benchmarks use >= 1 h audio
# 8000 is the "warning" floor (below this, eps_min=0.005 is under-resolved);
# 4000 is the "critical" floor (below this, any ranking decision is noise).
_WORDS_RELIABLE_FLOOR = 8000
_WORDS_CRITICAL_FLOOR = 4000

# HuggingFace source mapping — extensible dict.
# {lang} and {region} placeholders are resolved at fetch time.
_HF_SOURCE_MAP: dict[str, dict[str, str]] = {
    "librispeech": {
        "repo": "librispeech_asr",
        "config": "clean",
        "audio_col": "audio",
        "text_col": "text",
        # LibriSpeech exposes speaker_id as a top-level column; using it
        # turns on the blockwise bootstrap in WEREngine (Liu et al. 2020).
        "speaker_col": "speaker_id",
    },
    "fleurs": {
        "repo": "google/fleurs",
        "config": "{lang}_{region}",
        "audio_col": "audio",
        "text_col": "raw_transcription",
        # FLEURS does not publish a speaker_id column — each test split has
        # ~20-30 unique speakers but the HuggingFace schema only carries
        # gender + sample path. We leave speaker_col absent so the bootstrap
        # falls back to per-segment sampling for FLEURS (noted as a known
        # limitation in the research report on dataset undersizing).
    },
    "common_voice": {
        "repo": "mozilla-foundation/common_voice_17_0",
        "config": "{lang}",
        "audio_col": "audio",
        "text_col": "sentence",
        "speaker_col": "client_id",  # Common Voice hashes speaker as client_id
        "note": "Gated dataset — requires HF terms acceptance and may be unavailable.",
    },
    "earnings22": {
        "repo": "distil-whisper/earnings22",
        "config": "chunked",
        "audio_col": "audio",
        "text_col": "transcription",
    },
    "mediaspeech": {
        "repo": "ymoslem/MediaSpeech",
        "config": "{lang}",
        "audio_col": "audio",
        "text_col": "text",
    },
}

# ISO 639-1 → Fleurs region suffix (used when config contains {region})
_LANG_REGION: dict[str, str] = {
    "en": "us",
    "tr": "tr",
    "de": "de",
    "fr": "fr",
    "es": "419",
    "it": "it",
    "pt": "br",
    "nl": "nl",
    "pl": "pl",
    "ru": "ru",
    "ar": "eg",
    "zh": "cn",
    "ja": "jp",
    "ko": "kr",
    "hi": "in",
}


@dataclass
class DatasetSegment:
    """A single transcription segment."""

    idx: int
    audio: np.ndarray  # float32, mono, 16 kHz
    ref_text: str
    offset_s: float  # start position in concatenated audio
    duration_s: float
    # Optional speaker identifier used by the blockwise bootstrap CI in
    # WEREngine. When available (LibriSpeech, CommonVoice), the WER CI is
    # computed by resampling per-speaker blocks rather than per-segment,
    # which corrects the CI under-estimation documented by
    # Liu et al. 2020 (Interspeech). None means "unknown / unique per
    # segment" — the bootstrap falls back to the classic per-segment path.
    speaker_id: str | None = None


@dataclass
class PreparedDataset:
    """Ready-to-consume data package for BenchmarkEngine / BenchmarkTrialExecutor."""

    dataset_id: str  # UUID from datasets table
    source: str  # "fleurs", "librispeech", "common_voice", "local", ...
    lang: str  # ISO 639-1
    split: str  # "test", "train", "dev"
    segments: list[DatasetSegment]
    audio: np.ndarray  # concatenated float32 16 kHz — for single-pass backends
    duration_s: float  # total duration
    sample_rate: int = _TARGET_SR


def _warn_if_undersized(prepared: PreparedDataset) -> None:
    """
    Emit a one-line reliability warning when a dataset is too small for a
    0.5%-resolution WER comparison.

    Why this matters:
        The IAMS optimizer's significance gate uses eps_min = 0.005 by default
        (see search/significance.py). That epsilon is only meaningful when the
        bootstrap CI half-width is comfortably below it, which requires on the
        order of 10,000 words (Speechmatics docs; Bisani & Ney 2004). Below
        ~8000 words the gate starts firing on measurement noise; below ~4000
        words it is essentially reading tea leaves.

    We intentionally warn rather than refuse:
        - unit tests and quick smoke runs still need to work on tiny datasets
        - the decision whether to trust the result belongs to the caller
        - forcing this at load time would break every test fixture in the tree

    The estimate uses a rough words-per-second constant; agglutinative
    languages will understate the true figure (less harmful: the warning
    fires slightly earlier) and very dense English read speech may overstate
    it (more harmful: warning fires slightly later). For a production run on
    full FLEURS or LibriSpeech test splits the result is unchanged because
    both are well above the reliable floor.
    """
    approx_words = int(prepared.duration_s * _APPROX_WORDS_PER_SECOND)
    if approx_words >= _WORDS_RELIABLE_FLOOR:
        return
    severity = "warning" if approx_words >= _WORDS_CRITICAL_FLOOR else "critical"
    level = logger.warning if severity == "warning" else logger.error
    level(
        "Dataset %s (%s/%s, %.0fs audio) yields ~%d words — below the %d-word "
        "reliable floor for WER comparisons at 0.5%% resolution. "
        "Bootstrap CI half-width will exceed eps_min; optimizer ranking decisions "
        "may be dominated by measurement noise. Increase max_duration_s or accept "
        "that Stage 1 epsilon must be raised to match the achievable resolution. "
        "(Speechmatics accuracy benchmarking; Bisani & Ney 2004, ICASSP.) "
        "[severity=%s]",
        prepared.dataset_id[:8] if prepared.dataset_id else "?",
        prepared.source,
        prepared.lang,
        prepared.duration_s,
        approx_words,
        _WORDS_RELIABLE_FLOOR,
        severity,
    )


class DatasetManager:
    """
    Fetch, prepare, cache, and register datasets for benchmark runs.

    Two entry points:
    - ``prepare(dataset_id)`` — load from DB row + cache, fetch on miss
    - ``fetch_and_register(source, lang, split)`` — download, register in DB, return id
    """

    def __init__(self, config: Any, conn: Any) -> None:
        from asrbench.data.audio_cache import AudioCache

        self._conn = conn
        self._cache = AudioCache(config.storage.cache_dir)

    def prepare(self, dataset_id: str, *, max_duration_s: float | None = None) -> PreparedDataset:  # noqa: E501
        """
        Prepare a dataset for benchmarking.

        Reads the dataset row from DB, checks the audio cache, and fetches on miss.
        If *max_duration_s* is not provided, the value stored in the DB row is used.

        Raises:
            ValueError: if dataset_id not found in DB.
        """
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT source, lang, split, local_path, max_duration_s "
            "FROM datasets WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()

        if row is None:
            raise ValueError(
                f"Dataset '{dataset_id}' not found in DB. "
                "Register it first with fetch_and_register()."
            )

        source, lang, split, local_path = row[0], row[1], row[2], row[3]
        # Use DB-stored max_duration_s unless caller overrides
        if max_duration_s is None and row[4] is not None:
            max_duration_s = float(row[4])

        cache_key = self._cache.cache_key(source, lang, split, max_duration_s)
        cached = self._cache.load(cache_key)
        if cached is not None:
            cached.dataset_id = dataset_id
            return cached

        if local_path:
            prepared = self._fetch_local(local_path, lang)
        else:
            prepared = self._fetch_hf(source, lang, split, max_duration_s)

        prepared.dataset_id = dataset_id
        self._cache.save(cache_key, prepared)
        _warn_if_undersized(prepared)
        return prepared

    def fetch_and_register(
        self,
        source: str,
        lang: str,
        split: str,
        *,
        local_path: str | None = None,
        max_duration_s: float | None = None,
    ) -> str:
        """
        Download (or read local), register in DB, return dataset_id.

        This is the entry point for CLI/API fetch commands.
        """
        if local_path:
            prepared = self._fetch_local(local_path, lang)
        else:
            prepared = self._fetch_hf(source, lang, split, max_duration_s)

        dataset_id = str(uuid.uuid4())
        prepared.dataset_id = dataset_id

        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO datasets (dataset_id, name, source, lang, split, local_path, verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [dataset_id, f"{source}_{lang}_{split}", source, lang, split, local_path, True],
        )

        cache_key = self._cache.cache_key(source, lang, split, max_duration_s)
        self._cache.save(cache_key, prepared)

        return dataset_id

    def _fetch_hf(
        self,
        source: str,
        lang: str,
        split: str,
        max_duration_s: float | None,
    ) -> PreparedDataset:
        """Fetch from HuggingFace datasets in streaming mode."""
        from datasets import Audio, load_dataset

        mapping = _HF_SOURCE_MAP.get(source)
        if mapping is None:
            raise ValueError(
                f"Unknown HuggingFace source '{source}'. "
                f"Available: {', '.join(sorted(_HF_SOURCE_MAP))}. "
                "For custom datasets, use local_path instead."
            )

        region = _LANG_REGION.get(lang, lang)
        config_name = mapping["config"].format(lang=lang, region=region)
        audio_col = mapping["audio_col"]
        text_col = mapping["text_col"]
        speaker_col = mapping.get("speaker_col")  # None for datasets without speaker labels

        ds = load_dataset(
            mapping["repo"], config_name, split=split, streaming=True, trust_remote_code=False
        )
        ds = ds.cast_column(audio_col, Audio(decode=False))

        silence_gap = np.zeros(int(_SILENCE_GAP_S * _TARGET_SR), dtype=np.float32)

        parts: list[np.ndarray] = []
        segments: list[DatasetSegment] = []
        total_duration = 0.0
        idx = 0

        for sample in ds:
            row: dict[str, Any] = dict(sample)  # type: ignore[arg-type]
            audio_data: dict[str, Any] = row[audio_col]
            text = row.get(text_col, "") or ""

            raw_bytes = audio_data.get("bytes")
            if raw_bytes is None:
                continue

            try:
                audio_arr, sr = sf.read(BytesIO(raw_bytes), dtype="float32")
            except Exception as exc:
                logger.warning("Skipping sample %d — decode failed: %s", idx, exc)
                continue

            # Ensure mono
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)

            # Resample to 16 kHz if needed
            if sr != _TARGET_SR:
                audio_arr = soxr.resample(audio_arr, sr, _TARGET_SR)

            seg_duration = len(audio_arr) / _TARGET_SR
            offset = total_duration

            speaker_id: str | None = None
            if speaker_col is not None:
                raw_spk = row.get(speaker_col)
                if raw_spk is not None:
                    speaker_id = str(raw_spk)

            segments.append(
                DatasetSegment(
                    idx=idx,
                    audio=audio_arr,
                    ref_text=str(text).strip(),
                    offset_s=offset,
                    duration_s=seg_duration,
                    speaker_id=speaker_id,
                )
            )

            parts.append(audio_arr)
            parts.append(silence_gap)
            total_duration += seg_duration + _SILENCE_GAP_S
            idx += 1

            if max_duration_s is not None and total_duration >= max_duration_s:
                break

        if not segments:
            raise RuntimeError(
                f"No valid audio segments found in {source}/{config_name}/{split}. "
                "Check that the dataset exists and contains audio data."
            )

        # Remove trailing silence gap
        if parts:
            parts.pop()
            total_duration -= _SILENCE_GAP_S

        concat_audio = np.concatenate(parts).astype(np.float32)

        return PreparedDataset(
            dataset_id="",
            source=source,
            lang=lang,
            split=split,
            segments=segments,
            audio=concat_audio,
            duration_s=total_duration,
        )

    def _fetch_local(self, local_path: str, lang: str) -> PreparedDataset:
        """
        Load audio from a local path.

        - Directory: reads all .wav/.flac files; optional .txt sidecar for ref_text.
        - Single file: entire audio becomes one segment.
        """
        path = Path(local_path)
        silence_gap = np.zeros(int(_SILENCE_GAP_S * _TARGET_SR), dtype=np.float32)

        if path.is_file():
            audio_arr, sr = sf.read(str(path), dtype="float32")
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            if sr != _TARGET_SR:
                audio_arr = soxr.resample(audio_arr, sr, _TARGET_SR)

            txt_path = path.with_suffix(".txt")
            ref_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""

            seg = DatasetSegment(
                idx=0,
                audio=audio_arr,
                ref_text=ref_text,
                offset_s=0.0,
                duration_s=len(audio_arr) / _TARGET_SR,
            )

            return PreparedDataset(
                dataset_id="",
                source="local",
                lang=lang,
                split="test",
                segments=[seg],
                audio=audio_arr,
                duration_s=seg.duration_s,
            )

        if not path.is_dir():
            raise FileNotFoundError(
                f"Local path '{local_path}' does not exist. "
                "Provide a directory of .wav/.flac files or a single audio file."
            )

        audio_files = sorted(f for f in path.iterdir() if f.suffix.lower() in (".wav", ".flac"))

        if not audio_files:
            raise RuntimeError(
                f"No .wav or .flac files found in '{local_path}'. "
                "The directory must contain audio files."
            )

        parts: list[np.ndarray] = []
        segments: list[DatasetSegment] = []
        total_duration = 0.0

        for idx, audio_file in enumerate(audio_files):
            audio_arr, sr = sf.read(str(audio_file), dtype="float32")
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            if sr != _TARGET_SR:
                audio_arr = soxr.resample(audio_arr, sr, _TARGET_SR)

            txt_path = audio_file.with_suffix(".txt")
            ref_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""

            seg_duration = len(audio_arr) / _TARGET_SR

            segments.append(
                DatasetSegment(
                    idx=idx,
                    audio=audio_arr,
                    ref_text=ref_text,
                    offset_s=total_duration,
                    duration_s=seg_duration,
                )
            )

            parts.append(audio_arr)
            parts.append(silence_gap)
            total_duration += seg_duration + _SILENCE_GAP_S

        # Remove trailing silence
        if parts:
            parts.pop()
            total_duration -= _SILENCE_GAP_S

        concat_audio = np.concatenate(parts).astype(np.float32)

        return PreparedDataset(
            dataset_id="",
            source="local",
            lang=lang,
            split="test",
            segments=segments,
            audio=concat_audio,
            duration_s=total_duration,
        )
