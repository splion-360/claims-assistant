"""Verify live camera frames reach tracking independently of remote inference."""
import asyncio
import base64
import unittest
from unittest.mock import Mock, patch

import cv2

from library_claim.domain.models import Frame
from library_claim.capture.relevance.classifier import decide
from library_claim.storage.repository import Repository
from live_demo import server
import test_live_turns
from test_tracking import textured_view


class TrackingTransportTests(unittest.IsolatedAsyncioTestCase):
    """Check the live tracking integration with a real decoded camera frame."""

    async def test_tracking_reports_while_remote_capture_is_pending(self):
        """Produce a local reference before a blocked remote request completes."""
        _, encoded = cv2.imencode('.jpg', textured_view())
        relevance = Mock(assess=Mock(return_value=decide(dict(books=.4, other_objects=.2, empty_background=.1), 12)))
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
                self.assertEqual(saved.relevance_metadata['label'], 'books')
                self.assertTrue((server.CAPTURE_DATA_DIR / 'frames' / saved.evidence_key).exists())
            return {'type': 'close'}

        with patch.object(server.CameraTracking, 'log', side_effect=log), \
             patch.object(server, '_pin_evidence_photo', side_effect=blocked):
            await test_live_turns.LiveTurnTests.drive(self, receive, relevance=relevance)
        self.assertEqual(records[0].status, 'reference_created')
