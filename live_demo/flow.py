"""Fan pipeline stage events out to flow-view subscribers without slowing the pipeline."""
import asyncio
from collections import deque


class FlowBus:
    """Thread-safe, bounded broadcast of stage events to any number of viewers."""

    def __init__(self, queue_size: int = 200, history: int = 60):
        """Create an unbound bus.

        Args:
            queue_size: Events buffered per subscriber before the oldest is dropped.
            history: Recent events replayed to a newly connected viewer.
        """
        self.queue_size = queue_size
        self.recent = deque(maxlen=history)
        self.queues: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the event loop that owns the subscriber queues.

        Args:
            loop: The server's running loop.
        """
        self.loop = loop

    def publish(self, event: dict) -> None:
        """Queue an event for every subscriber; safe to call from worker threads.

        Args:
            event: Stage event from `library_claim.telemetry`.
        """
        if self.loop is None or self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict) -> None:
        """Deliver an event on the loop thread, dropping each slow viewer's oldest event.

        Args:
            event: Stage event.
        """
        self.recent.append(event)
        for queue in self.queues:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        """Register a viewer.

        Returns:
            The viewer's event queue.
        """
        queue = asyncio.Queue(maxsize=self.queue_size)
        self.queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a viewer.

        Args:
            queue: Queue returned by `subscribe`.
        """
        self.queues.discard(queue)
