"""WebSocket progress streaming endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asrbench.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Track active WebSocket connections per resource."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, key: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(key, []).append(ws)

    def disconnect(self, key: str, ws: WebSocket) -> None:
        conns = self._connections.get(key, [])
        if ws in conns:
            conns.remove(ws)
        if not conns:
            self._connections.pop(key, None)

    async def broadcast(self, key: str, data: dict) -> None:
        for ws in self._connections.get(key, []):
            try:
                await ws.send_json(data)
            except Exception:
                pass


_manager = ConnectionManager()

_POLL_INTERVAL_S = 2.0


@router.websocket("/ws/runs/{run_id}")
async def ws_run_progress(ws: WebSocket, run_id: str) -> None:
    """Stream benchmark run progress — polls DB every 2s until terminal status."""
    await _manager.connect(f"run:{run_id}", ws)
    try:
        while True:
            conn = get_conn()
            cur = conn.cursor()

            status_row = cur.execute(
                "SELECT status FROM runs WHERE run_id = ?", [run_id]
            ).fetchone()
            if not status_row:
                await ws.send_json({"error": f"Run '{run_id}' not found."})
                break

            status = str(status_row[0])
            seg_count = cur.execute(
                "SELECT count(*) FROM segments WHERE run_id = ?", [run_id]
            ).fetchone()

            payload = {
                "run_id": run_id,
                "status": status,
                "segments_done": seg_count[0] if seg_count else 0,
            }

            # If completed, include aggregate
            if status == "completed":
                agg = cur.execute(
                    "SELECT wer_mean, rtfx_mean, wall_time_s FROM aggregates WHERE run_id = ?",
                    [run_id],
                ).fetchone()
                if agg:
                    payload["wer_mean"] = agg[0]
                    payload["rtfx_mean"] = agg[1]
                    payload["wall_time_s"] = agg[2]

            await ws.send_json(payload)

            if status in ("completed", "failed"):
                break

            await asyncio.sleep(_POLL_INTERVAL_S)
    except WebSocketDisconnect:
        pass
    finally:
        _manager.disconnect(f"run:{run_id}", ws)


@router.websocket("/ws/optimize/{study_id}")
async def ws_optimize_progress(ws: WebSocket, study_id: str) -> None:
    """Stream optimization study progress — polls DB every 2s."""
    await _manager.connect(f"opt:{study_id}", ws)
    try:
        while True:
            conn = get_conn()
            cur = conn.cursor()

            row = cur.execute(
                "SELECT status, total_trials, best_score FROM optimization_studies "
                "WHERE study_id = ?",
                [study_id],
            ).fetchone()
            if not row:
                await ws.send_json({"error": f"Study '{study_id}' not found."})
                break

            status = str(row[0])
            payload = {
                "study_id": study_id,
                "status": status,
                "total_trials": row[1] or 0,
                "best_score": row[2],
            }

            # Count trials per phase
            phase_rows = cur.execute(
                "SELECT phase, count(*) FROM optimization_trials WHERE study_id = ? GROUP BY phase",
                [study_id],
            ).fetchall()
            payload["phases"] = {str(p[0]): p[1] for p in phase_rows}

            await ws.send_json(payload)

            if status in ("completed", "failed"):
                break

            await asyncio.sleep(_POLL_INTERVAL_S)
    except WebSocketDisconnect:
        pass
    finally:
        _manager.disconnect(f"opt:{study_id}", ws)
