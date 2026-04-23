"""Structured activity logger — dual-writes to stderr JSON and the event bus.

The live-log stream on the Dashboard subscribes to the ``activity`` topic via
``/ws/activity``; operators tailing the process logs still see the same
records on stderr. A single call to :func:`log_activity` reaches both
surfaces so the UI and a human reader never diverge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from asrbench.engine.events import get_event_bus

logger = logging.getLogger(__name__)

__all__ = ["log_activity", "ActivityLogger"]

_ACTIVITY_TOPIC = "activity"


def _isoformat_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_stderr(record: dict[str, Any]) -> None:
    try:
        sys.stderr.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stderr.flush()
    except Exception as exc:
        logger.debug("ActivityLogger stderr write failed: %s", exc)


async def log_activity(
    level: str,
    message: str,
    *,
    source: str = "asrbench",
    **fields: Any,
) -> None:
    """Emit a structured activity record.

    ``level`` is a case-insensitive severity label (``info`` / ``warning`` /
    ``error`` / ``debug``). ``source`` identifies the emitting subsystem
    (``benchmark``, ``optimizer``, ``dataset``, ``api``). Additional keyword
    arguments are merged into the JSON record as-is, so callers can attach
    ``run_id``, ``study_id`` or any other identifying context.
    """
    record = {
        "ts": _isoformat_now(),
        "level": level.lower(),
        "source": source,
        "message": message,
        **fields,
    }
    _write_stderr(record)
    try:
        await get_event_bus().publish(_ACTIVITY_TOPIC, record)
    except Exception as exc:
        logger.debug("ActivityLogger publish failed: %s", exc)


class ActivityLogger:
    """Sync-friendly facade around :func:`log_activity` for non-async callers.

    ``BenchmarkEngine.run`` already runs inside an event loop, so it can call
    :func:`log_activity` directly. This class exists for helper code that is
    called from plain sync functions (CLI subcommands, thread-pool callbacks)
    where creating a one-shot coroutine would be awkward.
    """

    def __init__(self, source: str = "asrbench") -> None:
        self._source = source

    def info(self, message: str, **fields: Any) -> None:
        self._sync("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._sync("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._sync("error", message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self._sync("debug", message, **fields)

    def _sync(self, level: str, message: str, **fields: Any) -> None:
        record = {
            "ts": _isoformat_now(),
            "level": level,
            "source": self._source,
            "message": message,
            **fields,
        }
        _write_stderr(record)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not inside an event loop — stderr is already written; skip
            # the async broadcast. Pure-CLI callers do not need live fan-out.
            return
        loop.create_task(get_event_bus().publish(_ACTIVITY_TOPIC, record))
