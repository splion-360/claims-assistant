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
