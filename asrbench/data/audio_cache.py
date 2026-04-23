"""Disk-level audio cache — FLAC + segments JSON read/write for prepared datasets."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf

if TYPE_CHECKING:
    from asrbench.data.dataset_manager import PreparedDataset

logger = logging.getLogger(__name__)


class AudioCache:
    """
    Persistent cache for PreparedDataset objects on disk.

    Layout per entry::

        {cache_dir}/prepared_datasets/{hash16}/
            audio.flac          — concatenated float32 mono 16 kHz
            segments.json       — [{idx, ref_text, offset_s, duration_s}, ...]
            meta.json           — {source, lang, split, duration_s, sample_rate, ...}
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "prepared_datasets"
        self._dir.mkdir(parents=True, exist_ok=True)

    def cache_key(
        self,
        source: str,
        lang: str,
        split: str,
        duration_min: float | None,
        extra: str = "",
    ) -> str:
        """SHA-256[:16] of source|lang|split|duration|extra."""
        payload = f"{source}|{lang}|{split}|{duration_min}|{extra}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def exists(self, key: str) -> bool:
        entry = self._dir / key
        return (entry / "audio.flac").exists() and (entry / "segments.json").exists()

    def load(self, key: str) -> PreparedDataset | None:
        """Read {key}/audio.flac + segments.json → PreparedDataset or None."""
        from asrbench.data.dataset_manager import DatasetSegment, PreparedDataset

        entry = self._dir / key
        audio_path = entry / "audio.flac"
        segments_path = entry / "segments.json"
        meta_path = entry / "meta.json"

        if not audio_path.exists() or not segments_path.exists():
            return None

        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            raw_segs = json.loads(segments_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Corrupt audio cache entry %s — treating as miss: %s", key, exc)
            return None

        segments: list[DatasetSegment] = []
        for s in raw_segs:
            offset_samples = int(s["offset_s"] * sr)
            dur_samples = int(s["duration_s"] * sr)
            end_sample = min(offset_samples + dur_samples, len(audio))
            seg_audio = audio[offset_samples:end_sample]

            segments.append(
                DatasetSegment(
                    idx=s["idx"],
                    audio=seg_audio,
                    ref_text=s["ref_text"],
                    offset_s=s["offset_s"],
                    duration_s=s["duration_s"],
                    speaker_id=s.get("speaker_id"),
                )
            )

        return PreparedDataset(
            dataset_id=meta.get("dataset_id", ""),
            source=meta.get("source", ""),
            lang=meta.get("lang", ""),
            split=meta.get("split", ""),
            segments=segments,
            audio=audio,
            duration_s=float(len(audio)) / sr,
            sample_rate=int(sr),
        )

    def save(self, key: str, dataset: PreparedDataset) -> None:
        """Write audio.flac + segments.json + meta.json."""
        entry = self._dir / key
        entry.mkdir(parents=True, exist_ok=True)

        try:
            sf.write(str(entry / "audio.flac"), dataset.audio, dataset.sample_rate)

            seg_data = [
                {
                    "idx": s.idx,
                    "ref_text": s.ref_text,
                    "offset_s": s.offset_s,
                    "duration_s": s.duration_s,
                    "speaker_id": s.speaker_id,
                }
                for s in dataset.segments
            ]
            (entry / "segments.json").write_text(
                json.dumps(seg_data, ensure_ascii=False), encoding="utf-8"
            )

            meta = {
                "dataset_id": dataset.dataset_id,
                "source": dataset.source,
                "lang": dataset.lang,
                "split": dataset.split,
                "duration_s": dataset.duration_s,
                "sample_rate": dataset.sample_rate,
                "num_segments": len(dataset.segments),
                "created_at": datetime.now(UTC).isoformat(),
            }
            (entry / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write audio cache %s: %s", key, exc)
