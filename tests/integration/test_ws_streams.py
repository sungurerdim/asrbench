"""Integration tests for WebSocket stream endpoints (EventBus-backed)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import insert_dataset, insert_model, insert_run


@pytest.fixture(autouse=True)
def _fresh_event_bus() -> None:
    """Isolate subscribers between tests so heartbeat counts stay deterministic."""
    from asrbench.engine.events import reset_event_bus

    reset_event_bus()


class TestWsRunSnapshot:
    def test_nonexistent_run_receives_error_and_closes(self, app_client: TestClient) -> None:
        fake = "00000000-0000-0000-0000-000000000000"
        with app_client.websocket_connect(f"/ws/runs/{fake}") as ws:
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert fake in msg["error"]

    def test_completed_run_sends_snapshot_and_closes(self, app_client: TestClient) -> None:
        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="completed")
        with app_client.websocket_connect(f"/ws/runs/{run_id}") as ws:
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "snapshot"
            assert msg["run_id"] == run_id
            assert msg["status"] == "completed"


class TestWsRunProgress:
    def test_running_run_receives_published_event(self, app_client: TestClient) -> None:
        """A published segment_done event reaches the WS subscriber."""
        import asyncio

        from asrbench.engine.events import get_event_bus

        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="running")

        with app_client.websocket_connect(f"/ws/runs/{run_id}") as ws:
            # First message is the snapshot sent on subscribe.
            snapshot = json.loads(ws.receive_text())
            assert snapshot["type"] == "snapshot"

            async def _publish() -> None:
                await get_event_bus().publish(
                    f"runs:{run_id}",
                    {
                        "type": "segment_done",
                        "run_id": run_id,
                        "segments_done": 1,
                        "total_segments": 10,
                    },
                )

            asyncio.run(_publish())

            msg = json.loads(ws.receive_text())
            assert msg["type"] == "segment_done"
            assert msg["run_id"] == run_id
            assert msg["segments_done"] == 1

    def test_terminal_status_event_closes_stream(self, app_client: TestClient) -> None:
        """A completion event lets the handler exit cleanly."""
        import asyncio

        from asrbench.engine.events import get_event_bus

        model_id = insert_model()
        dataset_id = insert_dataset()
        run_id = insert_run(model_id, dataset_id, status="running")

        with app_client.websocket_connect(f"/ws/runs/{run_id}") as ws:
            _ = ws.receive_text()  # snapshot

            async def _publish() -> None:
                await get_event_bus().publish(
                    f"runs:{run_id}",
                    {"type": "complete", "run_id": run_id, "status": "completed"},
                )

            asyncio.run(_publish())

            msg = json.loads(ws.receive_text())
            assert msg["status"] == "completed"


class TestWsDatasetProgress:
    def test_nonexistent_dataset_returns_error(self, app_client: TestClient) -> None:
        fake = "00000000-0000-0000-0000-000000000000"
        with app_client.websocket_connect(f"/ws/datasets/{fake}") as ws:
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"

    def test_event_reaches_subscriber(self, app_client: TestClient) -> None:
        import asyncio

        from asrbench.engine.events import get_event_bus

        dataset_id = insert_dataset()
        with app_client.websocket_connect(f"/ws/datasets/{dataset_id}") as ws:
            _ = ws.receive_text()  # initial snapshot

            async def _publish() -> None:
                await get_event_bus().publish(
                    f"datasets:{dataset_id}",
                    {
                        "type": "fetch_progress",
                        "dataset_id": dataset_id,
                        "bytes_downloaded": 1024,
                    },
                )

            asyncio.run(_publish())
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "fetch_progress"
            assert msg["bytes_downloaded"] == 1024


class TestWsActivity:
    def test_activity_event_reaches_subscriber(self, app_client: TestClient) -> None:
        import asyncio

        from asrbench.engine.events import get_event_bus

        with app_client.websocket_connect("/ws/activity") as ws:

            async def _publish() -> None:
                await get_event_bus().publish(
                    "activity",
                    {
                        "ts": "2026-04-23T00:00:00.000Z",
                        "level": "info",
                        "source": "test",
                        "message": "hello from test",
                    },
                )

            asyncio.run(_publish())
            msg = json.loads(ws.receive_text())
            assert msg["message"] == "hello from test"
            assert msg["source"] == "test"

    def test_logs_alias_delivers_same_events(self, app_client: TestClient) -> None:
        """The legacy /ws/logs alias subscribes to the same `activity` topic."""
        import asyncio

        from asrbench.engine.events import get_event_bus

        with app_client.websocket_connect("/ws/logs") as ws:

            async def _publish() -> None:
                await get_event_bus().publish(
                    "activity",
                    {"level": "warning", "message": "legacy alias works"},
                )

            asyncio.run(_publish())
            msg = json.loads(ws.receive_text())
            assert msg["message"] == "legacy alias works"


class TestWsVram:
    def test_vram_topic_event_reaches_subscriber(self, app_client: TestClient) -> None:
        """Independent of the sampler task: a published event still fans out."""
        import asyncio

        from asrbench.engine.events import get_event_bus

        with app_client.websocket_connect("/ws/vram") as ws:
            # Sampler task may already be running — we just need to see at
            # least one event of any kind reach us. Publish a synthetic one
            # to avoid racing on the 500 ms sampler tick.
            async def _publish() -> None:
                await get_event_bus().publish(
                    "vram",
                    {
                        "type": "vram",
                        "available": True,
                        "used_mb": 1024.0,
                        "total_mb": 8192.0,
                        "free_mb": 7168.0,
                        "pct": 12.5,
                    },
                )

            asyncio.run(_publish())

            # Pull messages until we see a type=vram that matches our synthetic
            # publish (used_mb == 1024). Heartbeat frames are skipped.
            for _ in range(5):
                msg = json.loads(ws.receive_text())
                if msg.get("type") == "vram" and msg.get("used_mb") == 1024.0:
                    assert msg["pct"] == 12.5
                    break
            else:
                pytest.fail("Expected vram event with used_mb=1024.0 was not received")
