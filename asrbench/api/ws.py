"""WebSocket progress streaming — EventBus-driven, no polling."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asrbench.db import get_conn
from asrbench.engine.events import get_event_bus
from asrbench.engine.vram import get_vram_monitor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

_HEARTBEAT_INTERVAL_S = 30.0
_VRAM_SNAPSHOT_INTERVAL_S = 0.5
_TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_STUDY_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


async def _safe_send(ws: WebSocket, event: dict[str, Any]) -> bool:
    """Send ``event`` to ``ws`` and return False if the connection has dropped."""
    try:
        await ws.send_json(event)
        return True
    except Exception as exc:
        logger.debug("WebSocket send failed, dropping connection: %s", exc)
        return False


async def _subscribe_loop(
    ws: WebSocket,
    topic: str,
    *,
    initial: dict[str, Any] | None = None,
    terminal_check: Callable[[dict[str, Any]], bool] | None = None,
) -> None:
    """Forward every event on ``topic`` to ``ws`` until the client disconnects.

    A periodic heartbeat with type="heartbeat" is sent whenever no event
    arrives within ``_HEARTBEAT_INTERVAL_S``, which keeps intermediaries
    (reverse proxies, dev-tunnel daemons) from closing the socket on idle.
    If ``terminal_check`` returns True for an event, the loop cleanly exits
    so the client doesn't sit on a dead stream.
    """
    bus = get_event_bus()
    if initial is not None and not await _safe_send(ws, initial):
        return

    async with bus.subscribe(topic) as queue:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL_S)
            except asyncio.TimeoutError:
                if not await _safe_send(ws, {"type": "heartbeat", "topic": topic}):
                    return
                continue

            if not await _safe_send(ws, event):
                return
            if terminal_check is not None and terminal_check(event):
                return


def _run_terminal(event: dict[str, Any]) -> bool:
    return str(event.get("status", "")) in _TERMINAL_RUN_STATUSES


def _study_terminal(event: dict[str, Any]) -> bool:
    return str(event.get("status", "")) in _TERMINAL_STUDY_STATUSES


@router.websocket("/ws/runs/{run_id}")
async def ws_run_progress(ws: WebSocket, run_id: str) -> None:
    """Stream benchmark run progress from the event bus."""
    await ws.accept()
    conn = get_conn()
    row = conn.cursor().execute("SELECT status FROM runs WHERE run_id = ?", [run_id]).fetchone()
    if not row:
        await _safe_send(ws, {"type": "error", "error": f"Run '{run_id}' not found."})
        await ws.close()
        return

    initial = {"type": "snapshot", "run_id": run_id, "status": str(row[0])}
    if str(row[0]) in _TERMINAL_RUN_STATUSES:
        await _safe_send(ws, initial)
        await ws.close()
        return

    try:
        await _subscribe_loop(
            ws,
            topic=f"runs:{run_id}",
            initial=initial,
            terminal_check=_run_terminal,
        )
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/optimize/{study_id}")
async def ws_optimize_progress(ws: WebSocket, study_id: str) -> None:
    """Stream optimization study progress from the event bus."""
    await ws.accept()
    conn = get_conn()
    row = (
        conn.cursor()
        .execute(
            "SELECT status, total_trials, best_score FROM optimization_studies WHERE study_id = ?",
            [study_id],
        )
        .fetchone()
    )
    if not row:
        await _safe_send(ws, {"type": "error", "error": f"Study '{study_id}' not found."})
        await ws.close()
        return

    initial = {
        "type": "snapshot",
        "study_id": study_id,
        "status": str(row[0]),
        "total_trials": row[1] or 0,
        "best_score": row[2],
    }
    if str(row[0]) in _TERMINAL_STUDY_STATUSES:
        await _safe_send(ws, initial)
        await ws.close()
        return

    try:
        await _subscribe_loop(
            ws,
            topic=f"optimize:{study_id}",
            initial=initial,
            terminal_check=_study_terminal,
        )
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/datasets/{dataset_id}")
async def ws_dataset_progress(ws: WebSocket, dataset_id: str) -> None:
    """Stream dataset fetch progress from the event bus."""
    await ws.accept()
    conn = get_conn()
    row = (
        conn.cursor()
        .execute(
            "SELECT dataset_id, verified FROM datasets WHERE dataset_id = ?",
            [dataset_id],
        )
        .fetchone()
    )
    if not row:
        await _safe_send(ws, {"type": "error", "error": f"Dataset '{dataset_id}' not found."})
        await ws.close()
        return

    initial = {
        "type": "snapshot",
        "dataset_id": dataset_id,
        "verified": bool(row[1]),
    }

    def _verified_terminal(event: dict[str, Any]) -> bool:
        return bool(event.get("verified"))

    try:
        await _subscribe_loop(
            ws,
            topic=f"datasets:{dataset_id}",
            initial=initial,
            terminal_check=_verified_terminal,
        )
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/vram")
async def ws_vram(ws: WebSocket) -> None:
    """Stream periodic VRAM snapshots to any connected dashboard client.

    The sampling task is shared: a single snapshot per tick is taken from
    the VRAMMonitor singleton and broadcast to every subscriber via the
    event bus. A disconnected client simply stops receiving events; it
    does not stop the sampler for other clients.
    """
    await ws.accept()
    sampler = _ensure_vram_sampler()
    try:
        await _subscribe_loop(ws, topic="vram")
    except WebSocketDisconnect:
        pass
    finally:
        sampler.release()


@router.websocket("/ws/activity")
async def ws_activity(ws: WebSocket) -> None:
    """Stream structured activity log lines from the event bus."""
    await ws.accept()
    try:
        await _subscribe_loop(ws, topic="activity")
    except WebSocketDisconnect:
        pass


# Legacy alias — early UI code subscribed to /ws/logs for the same
# activity stream. Keep the name so existing clients keep working while
# the UI migrates.
@router.websocket("/ws/logs")
async def ws_logs(ws: WebSocket) -> None:
    await ws.accept()
    try:
        await _subscribe_loop(ws, topic="activity")
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# VRAM sampler — reference-counted background task
# ---------------------------------------------------------------------------


class _VRAMSampler:
    """Single shared background task that publishes VRAM snapshots."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._ref_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            self._ref_count += 1
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._loop())

    def release(self) -> None:
        self._ref_count -= 1
        if self._ref_count <= 0 and self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None
            self._ref_count = 0

    async def _loop(self) -> None:
        monitor = get_vram_monitor()
        bus = get_event_bus()
        try:
            while True:
                snap = monitor.snapshot()
                await bus.publish(
                    "vram",
                    {
                        "type": "vram",
                        "available": snap.available,
                        "used_mb": snap.used_mb,
                        "total_mb": snap.total_mb,
                        "free_mb": snap.free_mb,
                        "pct": snap.pct,
                    },
                )
                await asyncio.sleep(_VRAM_SNAPSHOT_INTERVAL_S)
        except asyncio.CancelledError:
            pass


_vram_sampler: _VRAMSampler | None = None


def _ensure_vram_sampler() -> _VRAMSampler:
    global _vram_sampler
    if _vram_sampler is None:
        _vram_sampler = _VRAMSampler()
    # We need to schedule acquire synchronously since the caller may not
    # want to await; kick it off as a task so a concurrent accept isn't
    # blocked waiting on the sampler lock.
    asyncio.create_task(_vram_sampler.acquire())
    return _vram_sampler
