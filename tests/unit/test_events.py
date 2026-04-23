"""Unit tests for the asyncio event bus."""

from __future__ import annotations

import asyncio

import pytest

from asrbench.engine.events import EventBus, get_event_bus, reset_event_bus


@pytest.fixture(autouse=True)
def _isolate_bus() -> None:
    reset_event_bus()


class TestSubscribe:
    async def test_subscriber_receives_published_event(self) -> None:
        bus = EventBus()
        async with bus.subscribe("runs:a") as q:
            await bus.publish("runs:a", {"type": "segment_done", "n": 1})
            event = await asyncio.wait_for(q.get(), timeout=0.5)
        assert event == {"type": "segment_done", "n": 1}

    async def test_subscriber_does_not_see_other_topics(self) -> None:
        bus = EventBus()
        async with bus.subscribe("runs:a") as q:
            await bus.publish("runs:b", {"type": "other"})
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), timeout=0.05)

    async def test_multiple_subscribers_get_same_event(self) -> None:
        bus = EventBus()
        async with bus.subscribe("runs:x") as qa, bus.subscribe("runs:x") as qb:
            await bus.publish("runs:x", {"type": "progress"})
            a = await asyncio.wait_for(qa.get(), timeout=0.5)
            b = await asyncio.wait_for(qb.get(), timeout=0.5)
        assert a == b == {"type": "progress"}

    async def test_subscribe_cleans_up_on_exit(self) -> None:
        bus = EventBus()
        async with bus.subscribe("runs:z"):
            assert bus.subscriber_count("runs:z") == 1
        assert bus.subscriber_count("runs:z") == 0


class TestBackpressure:
    async def test_full_queue_drops_oldest(self) -> None:
        bus = EventBus(queue_maxsize=2)
        async with bus.subscribe("t") as q:
            await bus.publish("t", {"n": 1})
            await bus.publish("t", {"n": 2})
            await bus.publish("t", {"n": 3})
            drained = []
            for _ in range(2):
                drained.append(await asyncio.wait_for(q.get(), timeout=0.1))
        # Queue held {n=1,n=2}; publishing n=3 evicted n=1. Remaining: n=2, n=3.
        assert [e["n"] for e in drained] == [2, 3]


class TestSingleton:
    async def test_get_event_bus_returns_same_instance(self) -> None:
        a = get_event_bus()
        b = get_event_bus()
        assert a is b

    async def test_reset_event_bus_creates_fresh_instance(self) -> None:
        a = get_event_bus()
        reset_event_bus()
        b = get_event_bus()
        assert a is not b
