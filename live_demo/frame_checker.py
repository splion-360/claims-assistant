"""Schedule live D1 frame checks: one in flight, the newest frame waiting."""
import asyncio
import time
from typing import Awaitable, Callable

from library_claim.config import CONFIG
from library_claim.decisions.drex import DrexError
from library_claim.decisions.frame_check import FrameCheck, check_frame


class FrameChecker:
    """Run Drex frame checks off the event loop without queueing stale frames."""

    def __init__(self, check: Callable[[bytes], FrameCheck] = check_frame, *,
                 min_interval_s: float = CONFIG.capture.quality_interval_s):
        """Bind the decision function and request spacing.

        Args:
            check: Blocking function deciding one JPEG frame.
            min_interval_s: Minimum time between request starts.
        """
        self.check = check
        self.min_interval_s = min_interval_s
        self.last_error: DrexError | None = None
        self._waiting = None
        self._running = False
        self._last_start = float('-inf')

    async def submit(self, pending, on_result: Callable[[object, FrameCheck | None], Awaitable[None]]) -> None:
        """Check a frame now, or make it the single waiting frame if a check is running.

        The caller that starts a check keeps draining the waiting slot, so a
        submission made while busy returns immediately.

        Args:
            pending: Frame with `jpeg` bytes and a `frame_id`.
            on_result: Coroutine receiving the frame and its check, or None on failure.
        """
        self._waiting = (pending, on_result)
        if self._running:
            return
        self._running = True
        try:
            while self._waiting is not None:
                item, callback = self._waiting
                self._waiting = None
                # Space requests so a sweep costs at most one Drex call per interval.
                delay = self._last_start + self.min_interval_s - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                    # A newer frame that arrived during the wait supersedes this one.
                    if self._waiting is not None:
                        continue
                self._last_start = time.monotonic()
                try:
                    result = await asyncio.to_thread(self.check, item.jpeg)
                    self.last_error = None
                except DrexError as error:
                    self.last_error, result = error, None
                await callback(item, result)
        finally:
            self._running = False
