"""Process-wide asyncio event bus for WebSocket fan-out.

The bus replaces the 2 s DB polling loop that used to drive every WebSocket
endpoint. Producers (BenchmarkEngine, VRAMMonitor, DatasetManager, the
activity logger) publish typed events into well-known topics; subscribers
(the /ws/* handlers) receive them with sub-second latency.

Usage — producer::

    from asrbench.engine.events import get_event_bus
    bus = get_event_bus()
    await bus.publish(f"runs:{run_id}", {
        "type": "segment_done",
        "segments_done": i,
        "total_segments": n,
    })

Usage — subscriber::

    bus = get_event_bus()
    async with bus.subscribe("runs:abc") as queue:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=30.0)
            await ws.send_json(event)

Topic naming convention (string match — no wildcards):

- ``runs:<run_id>``     benchmark progress + completion
- ``optimize:<study_id>``  IAMS trial / phase updates
- ``datasets:<dataset_id>``  fetch progress + verified
- ``vram``              periodic GPU snapshots
- ``activity``          structured log lines

Each subscriber gets its own bounded queue; when the queue is full the
newest event wins — a slow WS client never blocks a producer. The bus is a
singleton per process (``get_event_bus``). It is in-memory only and does not
cross process boundaries; background tasks that run via
``BackgroundTasks`` live in the same event loop and reach it directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["EventBus", "get_event_bus", "reset_event_bus"]

# Bounded queue keeps memory in check if a subscriber stalls. Newest event
# wins when full; losing a heartbeat-like snapshot is preferable to blocking
# the benchmark loop.
_QUEUE_MAXSIZE = 256


class EventBus:
    """Topic-based pub/sub implemented on top of ``asyncio.Queue``."""

    def __init__(self, *, queue_maxsize: int = _QUEUE_MAXSIZE) -> None:
        self._queue_maxsize = queue_maxsize
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Send ``event`` to every subscriber of ``topic``.

        Best-effort: if a subscriber's queue is full we drop the oldest
        event in that queue and insert the new one. A drop is logged at
        DEBUG; producers should never fail because of a slow consumer.
        """
        async with self._lock:
            subs = list(self._subscribers.get(topic, ()))

        for queue in subs:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    _ = queue.get_nowait()
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.debug("EventBus: dropped event on %r (queue full)", topic)

    @contextlib.asynccontextmanager
    async def subscribe(self, topic: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        """Async context manager that yields a per-subscriber queue.

        The queue is registered on enter and removed on exit, so a WS client
        disconnect cleans up automatically without leaking queues.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_maxsize)
        async with self._lock:
            self._subscribers.setdefault(topic, set()).add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                subs = self._subscribers.get(topic)
                if subs is not None:
                    subs.discard(queue)
                    if not subs:
                        self._subscribers.pop(topic, None)

    def subscriber_count(self, topic: str) -> int:
        """Return how many subscribers are currently attached to ``topic``."""
        return len(self._subscribers.get(topic, ()))


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide EventBus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    """Drop the singleton — tests use this to isolate subscribers across cases."""
    global _bus
    _bus = None
