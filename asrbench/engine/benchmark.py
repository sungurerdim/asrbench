"""BenchmarkEngine — per-segment transcription, WER computation, and DB persistence."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from asrbench.engine.transcript_cache import TranscriptCache
from asrbench.engine.wer import WEREngine
from asrbench.preprocessing.pipeline import PreprocessingPipeline

if TYPE_CHECKING:
    import duckdb

    from asrbench.backends.base import BaseBackend
    from asrbench.data.dataset_manager import PreparedDataset

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """
    Run orchestrator: transcribe every segment, compute WER, persist results.

    Pipeline per run:
        1. For each segment — cache check → backend.transcribe() on miss → cache save
        2. Collect (ref, hyp) pairs → WEREngine.compute()
        3. Compute RTFx = dataset.duration_s / wall_time
        4. INSERT into segments + aggregates tables
        5. UPDATE runs.status = 'completed'
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, *, cache_dir: Path) -> None:
        self._conn = conn
        self._cache = TranscriptCache(cache_dir)
        self._wer = WEREngine()

    @staticmethod
    def _split_params(params: dict) -> tuple[dict, dict]:
        """Separate backend params from ``preprocess.*`` params.

        Returns ``(backend_params, preprocess_params)`` where keys in the
        second dict have the ``preprocess.`` prefix stripped.
        """
        backend_params: dict = {}
        preprocess_params: dict = {}
        for k, v in params.items():
            if k.startswith("preprocess."):
                preprocess_params[k.removeprefix("preprocess.")] = v
            else:
                backend_params[k] = v
        return backend_params, preprocess_params

    async def run(
        self,
        run_id: str,
        backend: BaseBackend,
        dataset: PreparedDataset,
        params: dict,
        model_family: str | None,
        model_local_path: str,
    ) -> None:
        """
        Execute a complete benchmark run.

        Raises:
            RuntimeError: if the backend fails during transcription.
        """
        cur = self._conn.cursor()
        cur.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", [run_id])

        refs: list[str] = []
        hyps: list[str] = []
        seg_elapsed: list[float] = []

        backend_params, preprocess_params = self._split_params(params)

        wall_start = time.perf_counter()

        try:
            for seg in dataset.segments:
                cache_key = self._cache.key(
                    model_local_path, params, dataset.dataset_id, seg.idx, dataset.lang
                )
                cached = self._cache.load(cache_key)

                if cached is not None:
                    hyp_text = cached["hyp_text"]
                    elapsed = cached["elapsed_s"]
                else:
                    processed_audio = PreprocessingPipeline.apply(
                        seg.audio, preprocess_params, dataset.sample_rate
                    )
                    t0 = time.perf_counter()
                    result_segs = backend.transcribe(processed_audio, dataset.lang, backend_params)
                    elapsed = time.perf_counter() - t0

                    hyp_text = " ".join(s.hyp_text for s in result_segs).strip()
                    self._cache.save(cache_key, hyp_text, elapsed)

                refs.append(seg.ref_text)
                hyps.append(hyp_text)
                seg_elapsed.append(elapsed)

                cur.execute(
                    "INSERT INTO segments (run_id, offset_s, duration_s, ref_text, hyp_text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [run_id, seg.offset_s, seg.duration_s, seg.ref_text, hyp_text],
                )

            wall_time = time.perf_counter() - wall_start

            # Collect optional per-segment speaker labels for the WER CI's
            # blockwise bootstrap path. Datasets that publish speaker_id
            # (LibriSpeech, CommonVoice) set this; FLEURS leaves it None so
            # WEREngine falls back to per-segment sampling. See
            # dataset_manager._fetch_hf and wer._bootstrap_wer_ci.
            speaker_ids: list[str | None] | None = [seg.speaker_id for seg in dataset.segments]
            if speaker_ids is not None and not any(sid is not None for sid in speaker_ids):
                speaker_ids = None  # short-circuit for the fallback path

            metrics = self._wer.compute(
                refs,
                hyps,
                dataset.lang,
                model_family=model_family,
                dataset_source=dataset.source,
                speaker_ids=speaker_ids,
            )

            rtfx_mean = dataset.duration_s / wall_time if wall_time > 0 else 0.0

            # RTFx p95: use per-segment RTFx values
            seg_rtfx = np.array(
                [
                    seg.duration_s / e if e > 0 else 0.0
                    for seg, e in zip(dataset.segments, seg_elapsed)
                ]
            )
            rtfx_p95 = float(np.percentile(seg_rtfx, 5)) if len(seg_rtfx) > 0 else 0.0

            word_count = sum(len(r.split()) for r in refs)

            cur.execute(
                "INSERT INTO aggregates "
                "(run_id, wer_mean, cer_mean, mer_mean, rtfx_mean, rtfx_p95, "
                "vram_peak_mb, wall_time_s, word_count, data_leakage_warning, "
                "wer_ci_lower, wer_ci_upper) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    metrics["wer"],
                    metrics["cer"],
                    metrics["mer"],
                    rtfx_mean,
                    rtfx_p95,
                    None,  # vram_peak_mb — populated by VRAMMonitor if available
                    wall_time,
                    word_count,
                    metrics["data_leakage_warning"],
                    metrics["wer_ci_lower"],
                    metrics["wer_ci_upper"],
                ],
            )

            cur.execute("UPDATE runs SET status = 'completed' WHERE run_id = ?", [run_id])

        except Exception:
            cur.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", [run_id])
            raise
