"""Unit tests for the on-disk PreparedDataset audio cache (Faz 9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asrbench.data.audio_cache import AudioCache


def _make_prepared(tmp_path: Path, n_segments: int = 2):
    """Build a minimal PreparedDataset for round-tripping tests."""
    from asrbench.data.dataset_manager import DatasetSegment, PreparedDataset

    sr = 16_000
    segs = []
    total = 0
    per_seg_samples = sr  # 1 s each
    audio_parts = []
    for i in range(n_segments):
        seg_audio = np.full(per_seg_samples, 0.1 * (i + 1), dtype=np.float32)
        segs.append(
            DatasetSegment(
                idx=i,
                audio=seg_audio,
                ref_text=f"segment {i}",
                offset_s=float(total) / sr,
                duration_s=float(per_seg_samples) / sr,
                speaker_id=f"spk-{i}",
            )
        )
        audio_parts.append(seg_audio)
        total += per_seg_samples

    full = np.concatenate(audio_parts)
    return PreparedDataset(
        dataset_id="ds-abc",
        source="custom",
        lang="en",
        split="test",
        segments=segs,
        audio=full,
        duration_s=float(total) / sr,
        sample_rate=sr,
    )


class TestCacheKey:
    def test_key_is_deterministic(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path)
        a = cache.cache_key("fleurs", "en", "test", 600.0)
        b = cache.cache_key("fleurs", "en", "test", 600.0)
        assert a == b

    def test_key_changes_with_inputs(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path)
        assert cache.cache_key("fleurs", "en", "test", 600.0) != cache.cache_key(
            "fleurs", "tr", "test", 600.0
        )


class TestSaveLoadRoundtrip:
    def test_save_then_load_returns_equivalent_dataset(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path)
        prepared = _make_prepared(tmp_path, n_segments=3)
        key = "roundtrip-key"

        cache.save(key, prepared)
        assert cache.exists(key) is True

        loaded = cache.load(key)
        assert loaded is not None
        assert loaded.source == "custom"
        assert loaded.lang == "en"
        assert loaded.split == "test"
        assert loaded.sample_rate == 16_000
        assert len(loaded.segments) == 3
        # Each segment's ref_text round-trips.
        assert [s.ref_text for s in loaded.segments] == [
            "segment 0",
            "segment 1",
            "segment 2",
        ]
        # speaker_id round-trips too.
        assert [s.speaker_id for s in loaded.segments] == ["spk-0", "spk-1", "spk-2"]

    def test_load_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path)
        assert cache.load("does-not-exist") is None
        assert cache.exists("does-not-exist") is False

    def test_load_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        cache = AudioCache(tmp_path)
        prepared = _make_prepared(tmp_path, n_segments=1)
        key = "corrupt-key"
        cache.save(key, prepared)

        # Corrupt segments.json — load must fail-safe to None.
        (tmp_path / "prepared_datasets" / key / "segments.json").write_text(
            "{not valid", encoding="utf-8"
        )
        assert cache.load(key) is None


class TestSaveFailurePath:
    def test_save_swallows_oserror(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failing sf.write is logged but must not raise."""
        import asrbench.data.audio_cache as cache_mod

        def _fail_write(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr(cache_mod.sf, "write", _fail_write)

        cache = AudioCache(tmp_path)
        prepared = _make_prepared(tmp_path, n_segments=1)
        cache.save("no-disk", prepared)  # should not raise
