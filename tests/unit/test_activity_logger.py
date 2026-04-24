"""Unit tests for the structured activity logger (Faz 9 coverage boost)."""

from __future__ import annotations

import asyncio
import json

import pytest

from asrbench.activity.logger import ActivityLogger, log_activity
from asrbench.engine.events import get_event_bus


@pytest.fixture()
def captured_stderr(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture the dict passed to ``_write_stderr``.

    Patches the private writer directly — the logger module binds the
    stderr handle at import time, so replacing ``sys.stderr`` after the
    fact would not help.
    """
    import asrbench.activity.logger as logger_mod

    records: list[dict] = []
    monkeypatch.setattr(logger_mod, "_write_stderr", records.append)
    return records


class TestLogActivity:
    async def test_writes_record_with_expected_fields(self, captured_stderr: list[dict]) -> None:
        await log_activity("info", "hello", source="benchmark", run_id="abc")
        assert len(captured_stderr) == 1
        record = captured_stderr[0]
        assert record["level"] == "info"
        assert record["message"] == "hello"
        assert record["source"] == "benchmark"
        assert record["run_id"] == "abc"
        assert "ts" in record

    async def test_level_is_normalised_to_lowercase(self, captured_stderr: list[dict]) -> None:
        await log_activity("WARNING", "careful")
        assert captured_stderr[0]["level"] == "warning"

    async def test_publishes_to_event_bus(self, captured_stderr: list[dict]) -> None:
        del captured_stderr
        bus = get_event_bus()
        async with bus.subscribe("activity") as queue:
            await log_activity("info", "bus check", source="test")
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["message"] == "bus check"
        assert event["source"] == "test"


class TestActivityLoggerSync:
    def test_sync_info_writes_record(self, captured_stderr: list[dict]) -> None:
        logger = ActivityLogger(source="cli")
        logger.info("a thing happened", run_id="r-1")
        assert len(captured_stderr) == 1
        record = captured_stderr[0]
        assert record["level"] == "info"
        assert record["source"] == "cli"
        assert record["message"] == "a thing happened"
        assert record["run_id"] == "r-1"

    def test_sync_warning_error_debug(self, captured_stderr: list[dict]) -> None:
        logger = ActivityLogger()
        logger.warning("watch out")
        logger.error("nope")
        logger.debug("dbg")
        levels = [r["level"] for r in captured_stderr]
        assert levels == ["warning", "error", "debug"]

    def test_outside_event_loop_does_not_raise(self, captured_stderr: list[dict]) -> None:
        """The sync path skips the async broadcast when no loop is running."""
        logger = ActivityLogger()
        logger.info("loop-less emit")
        # No loop → publish is skipped, but stderr write still happens.
        assert len(captured_stderr) == 1


class TestStderrJsonShape:
    """The real _write_stderr emits well-formed JSON lines."""

    def test_write_stderr_produces_valid_json_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io

        import asrbench.activity.logger as logger_mod

        buf = io.StringIO()
        monkeypatch.setattr(logger_mod.sys, "stderr", buf)

        logger_mod._write_stderr({"level": "info", "message": "hi"})
        line = buf.getvalue().strip()
        record = json.loads(line)
        assert record == {"level": "info", "message": "hi"}
