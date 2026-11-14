"""Verify the live D1 scheduler keeps one request in flight and the newest frame waiting."""
import asyncio
import threading
import time
import unittest

from library_claim.decisions.drex import DrexError
from live_demo.frame_checker import FrameChecker
from test_frame_check import check


def pending(name: str):
    """Build a stand-in pending frame identified by name.

    Args:
        name: Frame identifier.

    Returns:
        An object with the frame_id and jpeg attributes the checker reads.
    """
    return type('Pending', (), {'frame_id': name, 'jpeg': name.encode()})()


class FrameCheckerTests(unittest.IsolatedAsyncioTestCase):
    """Exercise replacement, spacing and failure reporting."""

    async def test_newer_submission_replaces_waiting_one(self):
        """Check the in-flight frame and only the newest waiting frame."""
        release, checked, results = threading.Event(), [], []

        def slow_check(jpeg):
            """Block until released and record which frame was checked.

            Args:
                jpeg: Frame bytes.

            Returns:
                A passing frame check.
            """
            checked.append(jpeg)
            release.wait(2)
            return check(0.9)

        async def on_result(item, result):
            """Record results in order.

            Args:
                item: Submitted pending frame.
                result: Frame check or None.
            """
            results.append((item.frame_id, result is not None))

        checker = FrameChecker(slow_check, min_interval_s=0)
        first = asyncio.create_task(checker.submit(pending('A'), on_result))
        while not checked:
            await asyncio.sleep(0.001)
        await checker.submit(pending('B'), on_result)
        await checker.submit(pending('C'), on_result)
        release.set()
        await first
        self.assertEqual(checked, [b'A', b'C'])
        self.assertEqual(results, [('A', True), ('C', True)])

    async def test_interval_spacing(self):
        """Start consecutive checks at least min_interval_s apart."""
        starts = []

        def timed_check(jpeg):
            """Record when each check started.

            Args:
                jpeg: Frame bytes.

            Returns:
                A passing frame check.
            """
            starts.append(time.monotonic())
            return check(0.9)

        async def on_result(item, result):
            """Ignore results.

            Args:
                item: Submitted pending frame.
                result: Frame check or None.
            """

        checker = FrameChecker(timed_check, min_interval_s=0.2)
        await checker.submit(pending('A'), on_result)
        await checker.submit(pending('B'), on_result)
        self.assertGreaterEqual(starts[1] - starts[0], 0.19)

    async def test_drex_failure_reports_none(self):
        """Report a failed decision as None and keep the error for the caller."""
        results = []

        def failing_check(jpeg):
            """Fail like an exhausted Drex account.

            Args:
                jpeg: Frame bytes.

            Raises:
                DrexError: Always.
            """
            raise DrexError('Drex request failed (HTTP 402).', 402)

        async def on_result(item, result):
            """Record results.

            Args:
                item: Submitted pending frame.
                result: Frame check or None.
            """
            results.append(result)

        checker = FrameChecker(failing_check, min_interval_s=0)
        await checker.submit(pending('A'), on_result)
        self.assertEqual(results, [None])
        self.assertEqual(checker.last_error.status, 402)


class QuietBackoffTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the slower cadence that follows a run of frames holding no book."""

    @staticmethod
    async def run_checks(checker, outcomes):
        """Submit one frame per outcome and return the spacing chosen after each.

        Args:
            checker: Checker under test.
            outcomes: Callables returning a frame check or raising, one per frame.

        Returns:
            The checker's next-request spacing after each submission.
        """
        spacings = []

        async def on_result(item, result):
            """Ignore the result; spacing is read after the call returns.

            Args:
                item: Submitted pending frame.
                result: Frame check or None.
            """

        for index, outcome in enumerate(outcomes):
            checker.check = lambda jpeg, outcome=outcome: outcome()
            await checker.submit(pending(f'frame{index}'), on_result)
            spacings.append(checker.interval_s)
        return spacings

    async def test_cadence_slows_only_after_the_configured_streak(self):
        """Keep the base spacing until enough consecutive frames hold no book."""
        checker = FrameChecker(min_interval_s=0, quiet_streak=3, quiet_interval_s=0.05)
        spacings = await self.run_checks(checker, [lambda: check(0.1)] * 4)
        self.assertEqual(spacings, [0, 0, 0.05, 0.05])

    async def test_a_visible_book_restores_the_base_cadence_at_once(self):
        """Resume fast checking on the first frame that shows a book."""
        checker = FrameChecker(min_interval_s=0, quiet_streak=2, quiet_interval_s=0.05)
        outcomes = [lambda: check(0.1), lambda: check(0.1), lambda: check(0.9), lambda: check(0.1)]
        spacings = await self.run_checks(checker, outcomes)
        self.assertEqual(spacings, [0, 0.05, 0, 0])
        self.assertEqual(checker.misses, 1)

    async def test_failed_checks_back_off_like_frames_without_books(self):
        """Count a Drex failure as a miss so an outage is not retried every interval."""
        def failing():
            """Raise the error a Drex outage produces.

            Raises:
                DrexError: Always.
            """
            raise DrexError('unavailable')

        checker = FrameChecker(min_interval_s=0, quiet_streak=2, quiet_interval_s=0.05)
        spacings = await self.run_checks(checker, [failing, failing])
        self.assertEqual(spacings, [0, 0.05])
        self.assertIsInstance(checker.last_error, DrexError)
