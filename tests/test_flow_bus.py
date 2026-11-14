"""Verify the flow event bus delivers events from any thread and stays bounded."""
import asyncio
import unittest

from live_demo.flow import FlowBus


class FlowBusTests(unittest.IsolatedAsyncioTestCase):
    """Exercise thread-safe fan-out, replay of recent events and overflow."""

    async def test_events_from_worker_threads_reach_subscribers(self):
        """A publish from a worker thread arrives on the subscriber's queue."""
        bus = FlowBus()
        bus.bind(asyncio.get_running_loop())
        queue = bus.subscribe()
        await asyncio.to_thread(bus.publish, {'stage': 'd1', 'phase': 'start'})
        self.assertEqual((await asyncio.wait_for(queue.get(), 1))['stage'], 'd1')
        self.assertEqual(bus.recent[-1]['stage'], 'd1')

    async def test_slow_subscriber_keeps_newest_events(self):
        """A full queue drops its oldest event instead of blocking the pipeline."""
        bus = FlowBus(queue_size=2)
        bus.bind(asyncio.get_running_loop())
        queue = bus.subscribe()
        for index in range(3):
            bus.publish({'stage': str(index), 'phase': 'done'})
        await asyncio.sleep(0)
        self.assertEqual([queue.get_nowait()['stage'] for _ in range(2)], ['1', '2'])

    async def test_unbound_bus_ignores_events(self):
        """Publishing before any live session binds the bus is a no-op."""
        FlowBus().publish({'stage': 'camera', 'phase': 'done'})
