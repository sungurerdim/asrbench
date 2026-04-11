"""Hypothesis transcript cache — avoids re-transcribing identical combinations.

Cache key: SHA-256 of (model_local_path, params, dataset_id, segment_idx, lang).
Uses dataset_id + segment_idx instead of raw audio bytes — the dataset checksum
already guarantees audio integrity, and index-based keys avoid hashing large arrays.

Design invariant: WER/CER are NEVER cached. Only hyp_text and elapsed_s are stored.
This ensures metric formula changes are always reflected on the next run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptCache:
    """
    Per-segment transcript cache keyed by a deterministic hash of
    (model_local_path, params, dataset_id, segment_idx, lang).

    Usage:
        key  = cache.key(model_path, params, dataset_id, idx, lang)
        hit  = cache.load(key)          # dict or None
        if hit is None:
            hyp, elapsed = transcribe(...)
            cache.save(key, hyp, elapsed)
        else:
            hyp, elapsed = hit["hyp_text"], hit["elapsed_s"]

    Storage: {cache_dir}/hyp_cache/{hash16}.json
    Entry:   {"hyp_text": "...", "elapsed_s": 1.23}
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "hyp_cache"
        self._dir.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        model_local_path: str,
        params: dict,
        dataset_id: str,
        segment_idx: int,
        lang: str,
    ) -> str:
        """
        Return a 16-char hex cache key for this combination.

        Any change to model path, params, dataset, segment position, or language
        produces a different key — guaranteeing no stale cache hits.

        Uses a deterministic string representation (stable across process restarts)
        hashed with SHA-256 for filesystem-safe fixed-width keys.
        """
        # Build a canonical string directly — avoids json.dumps overhead
        params_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        payload = f"{model_local_path}|{params_str}|{dataset_id}|{segment_idx}|{lang}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def load(self, cache_key: str) -> dict | None:
        """
        Return {"hyp_text": str, "elapsed_s": float} on hit, None on miss.

        A corrupt or unreadable entry is treated as a miss and logged as a warning.
        """
        path = self._dir / f"{cache_key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupt transcript cache entry %s — treating as miss: %s",
                cache_key,
                exc,
            )
            return None

    def save(self, cache_key: str, hyp_text: str, elapsed_s: float) -> None:
        """
        Persist a transcript to the cache.

        Write failures are logged as warnings and silently ignored — a failed
        save is harmless: the next run will simply transcribe again.
        """
        path = self._dir / f"{cache_key}.json"
        try:
            path.write_text(
                json.dumps(
                    {"hyp_text": hyp_text, "elapsed_s": elapsed_s},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to write transcript cache %s: %s", cache_key, exc)
