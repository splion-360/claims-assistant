"""Verify live camera frames reach tracking independently of remote inference."""
import asyncio
import base64
import unittest
from unittest.mock import Mock, patch

import cv2

from library_claim.domain.models import Frame
from test_frame_check import check
from library_claim.storage.repository import Repository
from live_demo import server
import test_live_turns
from test_tracking import textured_view


class TrackingTransportTests(unittest.IsolatedAsyncioTestCase):
    """Check the live tracking integration with a real decoded camera frame."""

    async def test_tracking_reports_while_remote_capture_is_pending(self):
        """Produce a local reference before a blocked remote request completes."""
        _, encoded = cv2.imencode('.jpg', textured_view())
        ready = asyncio.Event()
        records = []
        count = 0

        def log(result, timestamp):
            """Capture a completed tracking diagnostic.

            Args:
                result: Tracking result for the current frame.
                timestamp: Frame arrival time.
            """
            records.append(result)
            ready.set()

        async def blocked(*args, **kwargs):
            """Keep the remote capture pending until transport cleanup.

            Args:
                *args: Capture arguments.
                **kwargs: Capture keyword arguments.
            """
            await asyncio.Event().wait()

        async def receive(session, sent):
            """Supply a frame and close once local tracking completes.

            Args:
                session: Active intake session.
                sent: Outgoing messages.

            Returns:
                The next browser message.
            """
            nonlocal count
            count += 1
            if count == 1:
                return {'type': 'video', 'data': base64.b64encode(encoded).decode()}
            await asyncio.wait_for(ready.wait(), 2)
            self.assertEqual(records[0].frame_id, session.last_frame_id)
            with Repository(server.CAPTURE_DATA_DIR / 'claims.sqlite3') as repository:
                saved = repository.get(Frame, session.last_frame_id)
                self.assertEqual(saved.sweep_id, session.session_id)
                self.assertTrue((server.CAPTURE_DATA_DIR / 'frames' / saved.evidence_key).exists())
            return {'type': 'close'}

        with patch.object(server.CameraTracking, 'log', side_effect=log), \
             patch.object(server, '_pin_evidence_photo', side_effect=blocked):
            await test_live_turns.LiveTurnTests.drive(self, receive, frame_check=lambda jpeg: check(0.9))
        self.assertEqual(records[0].status, 'reference_created')

    async def test_below_threshold_frames_do_not_call_remote_capture(self):
        """Run real tracking and quality while Drex sees no book."""
        _, encoded = cv2.imencode('.jpg', textured_view())
        ready = asyncio.Event()
        count = 0
        checked = Mock(side_effect=lambda jpeg: check(0.1))

        async def receive(session, sent):
            """Supply a negative frame and close after its local decision.

            Args:
                session: Active session.
                sent: Emitted browser messages.

            Returns:
                Next simulated camera message.
            """
            nonlocal count
            count += 1
            if count == 1:
                return {'type': 'video', 'data': base64.b64encode(encoded).decode()}
            await ready.wait()
            while not checked.called:
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.01)
            return {'type': 'close'}

        with patch.object(server.CameraTracking, 'log', side_effect=lambda *args: ready.set()), \
             patch.object(server, '_pin_evidence_photo') as remote:
            await test_live_turns.LiveTurnTests.drive(self, receive, frame_check=checked)
        remote.assert_not_called()
        checked.assert_called_once()

    async def test_new_arrival_cannot_replace_locally_approved_image(self):
        """Dispatch approved bytes even when another frame arrives during assessment."""
        from threading import Event
        from unittest.mock import AsyncMock

        _, first = cv2.imencode('.jpg', textured_view())
        _, second = cv2.imencode('.jpg', 255 - textured_view())
        started, release = Event(), Event()
        remote_started = asyncio.Event()
        first_id = None
        count = 0

        def assess(*args):
            """Pause the D1 check to exercise the newer-frame arrival race.

            Args:
                *args: Frame bytes supplied by the checker.

            Returns:
                A passing book assessment.
            """
            started.set()
            if not release.wait(2):
                raise TimeoutError('Test did not release the frame check')
            return check(0.9)

        async def remote(session, args, *, candidate):
            """Verify dispatch retains the assessed frame instead of the latest one.

            Args:
                session: Active camera session.
                args: Capture arguments.
                candidate: Locally approved snapshot.

            Returns:
                Simulated unsuccessful pin result.
            """
            self.assertEqual(candidate.data, first.tobytes())
            self.assertEqual(candidate.frame_id, first_id)
            self.assertEqual(session.last_frame, second.tobytes())
            remote_started.set()
            return {'pinned': False}

        async def receive(session, sent):
            """Advance transport while the first local assessment is paused.

            Args:
                session: Active session.
                sent: Browser events.

            Returns:
                Next simulated video or close message.
            """
            nonlocal count, first_id
            count += 1
            if count == 1:
                return {'type': 'video', 'data': base64.b64encode(first).decode()}
            if count == 2:
                while not started.is_set():
                    await asyncio.sleep(.001)
                first_id = session.last_frame_id
                return {'type': 'video', 'data': base64.b64encode(second).decode()}
            release.set()
            await remote_started.wait()
            return {'type': 'close'}

        with patch.object(server, '_pin_evidence_photo', new=AsyncMock(side_effect=remote)) as capture:
            await test_live_turns.LiveTurnTests.drive(self, receive, frame_check=assess)
        self.assertEqual(capture.call_count, 1)
