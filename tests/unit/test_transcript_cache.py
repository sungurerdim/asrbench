"""Unit tests for the per-segment transcript cache (Faz 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from asrbench.engine.transcript_cache import TranscriptCache


class TestKeyDeterminism:
    def test_same_inputs_same_key(self, tmp_path: Path) -> None:
        cache = TranscriptCache(tmp_path)
        a = cache.key("/m", {"beam": 5}, "ds-1", 0, "en")
        b = cache.key("/m", {"beam": 5}, "ds-1", 0, "en")
        assert a == b

    def test_param_diff_changes_key(self, tmp_path: Path) -> None:
        cache = TranscriptCache(tmp_path)
        a = cache.key("/m", {"beam": 5}, "ds-1", 0, "en")
        b = cache.key("/m", {"beam": 7}, "ds-1", 0, "en")
        assert a != b

    def test_segment_index_changes_key(self, tmp_path: Path) -> None:
        cache = TranscriptCache(tmp_path)
        a = cache.key("/m", {}, "ds-1", 0, "en")
        b = cache.key("/m", {}, "ds-1", 1, "en")
        assert a != b

    def test_lang_changes_key(self, tmp_path: Path) -> None:
        cache = TranscriptCache(tmp_path)
        a = cache.key("/m", {}, "ds-1", 0, "en")
        b = cache.key("/m", {}, "ds-1", 0, "tr")
        assert a != b


class TestLoadSave:
    def test_load_miss_returns_none(self, tmp_path: Path) -> None:
        cache = TranscriptCache(tmp_path)
        assert cache.load("no-such-key") is None

    def test_save_then_load(self, tmp_path: Path) -> None:
        cache = TranscriptCache(tmp_path)
        cache.save("abc123", "hello world", 1.25)
        hit = cache.load("abc123")
        assert hit == {"hyp_text": "hello world", "elapsed_s": 1.25}

    def test_corrupt_entry_returns_none(self, tmp_path: Path) -> None:
        cache = TranscriptCache(tmp_path)
        cache.save("bad", "ok", 0.5)
        (tmp_path / "hyp_cache" / "bad.json").write_text("{not json", encoding="utf-8")
        assert cache.load("bad") is None


class TestSaveFailure:
    def test_oserror_is_swallowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Save failure only warns; harmless — next run re-transcribes."""
        from pathlib import Path as PathCls

        real_write = PathCls.write_text

        def _fail(self, *_a, **_kw):
            if self.name.endswith(".json"):
                raise OSError("disk full")
            return real_write(self, *_a, **_kw)

        monkeypatch.setattr(PathCls, "write_text", _fail)
        cache = TranscriptCache(tmp_path)
        cache.save("xyz", "should not raise", 0.1)
