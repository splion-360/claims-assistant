"""Schedule live D1 frame checks: one in flight, the newest frame waiting, slower while no book is seen."""
import asyncio
import time
from typing import Awaitable, Callable

from library_claim.config import CONFIG
from library_claim.decisions.drex import DrexError
from library_claim.decisions.frame_check import FrameCheck, check_frame


class FrameChecker:
    """Run Drex frame checks off the event loop without queueing stale frames."""

    def __init__(self, check: Callable[[bytes], FrameCheck] = check_frame, *,
                 min_interval_s: float = CONFIG.capture.quality_interval_s,
                 quiet_streak: int = CONFIG.drex.quiet_streak,
                 quiet_interval_s: float = CONFIG.drex.quiet_interval_s):
        """Bind the decision function and request spacing.

        Args:
            check: Blocking function deciding one JPEG frame.
            min_interval_s: Minimum time between request starts while books are being seen.
            quiet_streak: Consecutive checks finding no book before requests slow down.
            quiet_interval_s: Spacing between requests once that streak is reached.
        """
        self.check = check
        self.min_interval_s = min_interval_s
        self.quiet_streak = quiet_streak
        self.quiet_interval_s = quiet_interval_s
        # Consecutive checks that saw no book; a Drex failure counts, so outages also back off.
        self.misses = 0
        self.last_error: DrexError | None = None
        self._waiting = None
        self._running = False
        self._last_start = float('-inf')

    @property
    def interval_s(self) -> float:
        """Report the spacing the next request will use.

        Returns:
            The quiet spacing once the miss streak is reached, else the base spacing.
        """
        return self.quiet_interval_s if self.misses >= self.quiet_streak else self.min_interval_s

    def _record(self, result: FrameCheck | None) -> None:
        """Update the miss streak from one check outcome.

        A visible book resumes the base cadence at once, so a shelf coming into
        view is never left waiting for a quiet-period interval to expire.

        Args:
            result: The frame's decision, or None when the call failed.
        """
        if result is not None and result.book_visible >= CONFIG.drex.book_visible_min:
            self.misses = 0
        else:
            self.misses += 1

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
                delay = self._last_start + self.interval_s - time.monotonic()
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
                self._record(result)
                await callback(item, result)
        finally:
            self._running = False
