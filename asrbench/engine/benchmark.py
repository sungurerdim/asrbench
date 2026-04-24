"""BenchmarkEngine — per-segment transcription, WER computation, and DB persistence."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from asrbench.engine.events import get_event_bus
from asrbench.engine.transcript_cache import TranscriptCache
from asrbench.engine.vram import get_vram_monitor
from asrbench.engine.wer import WEREngine
from asrbench.preprocessing.pipeline import PreprocessingPipeline

if TYPE_CHECKING:
    import duckdb

    from asrbench.backends.base import BaseBackend
    from asrbench.data.dataset_manager import PreparedDataset

logger = logging.getLogger(__name__)


class RunCancelled(RuntimeError):
    """Raised by the benchmark loop when the run's cancel flag is set.

    Distinct from a generic RuntimeError so the outer background task
    can write ``status='cancelled'`` instead of ``status='failed'``.
    """

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id} was cancelled by request.")
        self.run_id = run_id


def _is_cancel_requested(conn: duckdb.DuckDBPyConnection, run_id: str) -> bool:
    """Check the row's ``cancel_requested`` flag.

    The engine shares a connection with the API, so a cursor-level read
    sees the latest committed value even when the cancel is issued from
    a different async task. Fails closed on any DB error: if we cannot
    read the flag we assume no cancel.
    """
    try:
        row = (
            conn.cursor()
            .execute("SELECT cancel_requested FROM runs WHERE run_id = ?", [run_id])
            .fetchone()
        )
    except Exception:
        return False
    return bool(row and row[0])


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

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        cache_dir: Path,
        segment_timeout_s: float | None = None,
    ) -> None:
        self._conn = conn
        self._cache = TranscriptCache(cache_dir)
        self._wer = WEREngine()
        # Per-segment backend.transcribe() timeout. ``None`` keeps the
        # legacy "wait forever" behaviour; the API / CLI wires the value
        # from config.limits.segment_timeout_s (default 120 s). A stuck
        # worker used to pin the event loop until the user killed the
        # whole process.
        self._segment_timeout_s = segment_timeout_s

    async def _transcribe_with_timeout(
        self,
        backend: BaseBackend,
        audio: np.ndarray,
        lang: str,
        params: dict,
    ) -> list:
        """Run ``backend.transcribe`` off the event loop, optionally time-capped.

        Always delegates to ``asyncio.to_thread`` so the event loop keeps
        serving WebSocket pushes while a long transcription runs. When
        ``self._segment_timeout_s`` is set, the call is further wrapped in
        ``asyncio.wait_for``; on expiry the future is cancelled and
        TimeoutError propagates up to the run's outer except clause.
        """

        def _call() -> list:
            return backend.transcribe(audio, lang, params)

        if self._segment_timeout_s is None or self._segment_timeout_s <= 0:
            return await asyncio.to_thread(_call)
        try:
            return await asyncio.wait_for(asyncio.to_thread(_call), timeout=self._segment_timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"backend.transcribe exceeded {self._segment_timeout_s:.0f}s "
                "per-segment timeout. Raise limits.segment_timeout_s in "
                "~/.asrbench/config.toml, or investigate why the backend "
                "is stuck — this usually means a deadlocked CUDA kernel "
                "or a child process that stopped consuming its input pipe."
            ) from exc

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
        segment_fraction: float = 1.0,
    ) -> None:
        """
        Execute a complete benchmark run.

        ``segment_fraction`` (default 1.0) is a multi-fidelity hook: when set
        to a value < 1, only the first ``ceil(N * segment_fraction)`` segments
        are processed. This lets Hyperband-style rung pruning run a trial on a
        cheap fraction of the corpus, score it, and decide whether to
        advance to a fuller fraction. Caller opt-in — production runs keep
        the default of 1.0 so nothing changes for non-pruned evaluations.

        Raises:
            RuntimeError: if the backend fails during transcription.
        """
        cur = self._conn.cursor()
        cur.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", [run_id])

        bus = get_event_bus()
        topic = f"runs:{run_id}"
        await bus.publish(topic, {"type": "status", "run_id": run_id, "status": "running"})

        refs: list[str] = []
        hyps: list[str] = []
        seg_elapsed: list[float] = []

        backend_params, preprocess_params = self._split_params(params)

        # Multi-fidelity slice: deterministic prefix so different rungs
        # measure the same audio in the same order (a key assumption the
        # promotion decision relies on).
        segments_all = dataset.segments
        if segment_fraction < 1.0:
            import math as _math

            n_keep = max(1, _math.ceil(len(segments_all) * segment_fraction))
            segments = segments_all[:n_keep]
        else:
            segments = segments_all
        duration_s = sum(seg.duration_s for seg in segments)

        wall_start = time.perf_counter()

        vram_monitor = get_vram_monitor()
        vram_monitor.reset_peak()
        # Take an initial reading so the peak includes pre-run allocations
        # (model weights already resident in VRAM when the run starts).
        vram_monitor.snapshot()

        try:
            total_segments = len(segments)
            for seg_index, seg in enumerate(segments, start=1):
                if _is_cancel_requested(self._conn, run_id):
                    raise RunCancelled(run_id)

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
                    result_segs = await self._transcribe_with_timeout(
                        backend, processed_audio, dataset.lang, backend_params
                    )
                    elapsed = time.perf_counter() - t0

                    hyp_text = " ".join(s.hyp_text for s in result_segs).strip()
                    self._cache.save(cache_key, hyp_text, elapsed)
                    vram_monitor.snapshot()

                refs.append(seg.ref_text)
                hyps.append(hyp_text)
                seg_elapsed.append(elapsed)

                cur.execute(
                    "INSERT INTO segments (run_id, offset_s, duration_s, ref_text, hyp_text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [run_id, seg.offset_s, seg.duration_s, seg.ref_text, hyp_text],
                )

                await bus.publish(
                    topic,
                    {
                        "type": "segment_done",
                        "run_id": run_id,
                        "segments_done": seg_index,
                        "total_segments": total_segments,
                        "elapsed_s": elapsed,
                    },
                )

            wall_time = time.perf_counter() - wall_start

            # Collect optional per-segment speaker labels for the WER CI's
            # blockwise bootstrap path. Datasets that publish speaker_id
            # (LibriSpeech, CommonVoice) set this; FLEURS leaves it None so
            # WEREngine falls back to per-segment sampling. See
            # dataset_manager._fetch_hf and wer._bootstrap_wer_ci.
            speaker_ids: list[str | None] | None = [seg.speaker_id for seg in segments]
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

            rtfx_mean = duration_s / wall_time if wall_time > 0 else 0.0

            # RTFx p95: use per-segment RTFx values
            seg_rtfx = np.array(
                [
                    seg.duration_s / e if e > 0 else 0.0
                    for seg, e in zip(segments, seg_elapsed, strict=False)
                ]
            )
            rtfx_p95 = float(np.percentile(seg_rtfx, 5)) if len(seg_rtfx) > 0 else 0.0

            word_count = sum(len(r.split()) for r in refs)

            vram_peak_mb = vram_monitor.peak_mb if vram_monitor.peak_mb > 0 else None

            cur.execute(
                "INSERT INTO aggregates "
                "(run_id, wer_mean, cer_mean, mer_mean, wil_mean, "
                "rtfx_mean, rtfx_p95, vram_peak_mb, wall_time_s, word_count, "
                "data_leakage_warning, wer_ci_lower, wer_ci_upper) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    metrics["wer"],
                    metrics["cer"],
                    metrics["mer"],
                    metrics["wil"],
                    rtfx_mean,
                    rtfx_p95,
                    vram_peak_mb,
                    wall_time,
                    word_count,
                    metrics["data_leakage_warning"],
                    metrics["wer_ci_lower"],
                    metrics["wer_ci_upper"],
                ],
            )

            cur.execute("UPDATE runs SET status = 'completed' WHERE run_id = ?", [run_id])
            await bus.publish(
                topic,
                {
                    "type": "complete",
                    "run_id": run_id,
                    "status": "completed",
                    "wer_mean": metrics["wer"],
                    "cer_mean": metrics["cer"],
                    "rtfx_mean": rtfx_mean,
                    "wall_time_s": wall_time,
                    "vram_peak_mb": vram_peak_mb,
                },
            )

        except Exception as exc:
            cur.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", [run_id])
            await bus.publish(
                topic,
                {
                    "type": "error",
                    "run_id": run_id,
                    "status": "failed",
                    "error": str(exc),
                },
            )
            raise
