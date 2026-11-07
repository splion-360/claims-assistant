"""Verify camera-quality logging through the live transport."""
import asyncio
import base64
import unittest
from unittest.mock import patch

import cv2

from live_demo.camera_quality import CameraQuality, log_scores
from live_demo import server
import test_live_turns
from test_quality import sample_image


class CameraQualityTests(unittest.TestCase):
    """Check stream continuity and structured logger output."""

    def test_stream_logs_real_scores_and_repeated_frame_change(self):
        """Decode real JPEGs and report stable temporal scores on repeats."""
        success, encoded = cv2.imencode('.jpg', sample_image())
        self.assertTrue(success)
        stream = CameraQuality()
        first = stream.assess(encoded.tobytes())
        second = stream.assess(encoded.tobytes())
        self.assertIsNone(first.signals.temporal_change)
        self.assertEqual(second.signals.temporal_change, 0)
        with self.assertLogs('uvicorn.error.camera_quality', level='DEBUG') as logs:
            log_scores('session', 'frame', second)
        self.assertIn('camera_quality session=session frame=frame', logs.output[0])
        self.assertIn('"temporal_change":0.0', logs.output[0])
        self.assertEqual(logs.records[0].levelname, 'DEBUG')

    def test_invalid_jpeg_does_not_advance_stream(self):
        """Reject a malformed image without treating it as a previous frame."""
        stream = CameraQuality()
        with self.assertRaises(ValueError):
            stream.assess(b'\xff\xd8\xffinvalid')
        _, encoded = cv2.imencode('.jpg', sample_image())
        self.assertIsNone(stream.assess(encoded.tobytes()).signals.temporal_change)


class LiveQualityTests(unittest.IsolatedAsyncioTestCase):
    """Verify browser frames reach local scoring independently of vision calls."""

    async def test_camera_frame_logs_without_waiting_for_remote_analysis(self):
        """Log real local scores while a simulated remote capture is blocked."""
        _, encoded = cv2.imencode('.jpg', sample_image())
        logged = asyncio.Event()
        records = []
        count = 0

        def record(session_id, frame_id, result):
            """Capture the completed score for the test.

            Args:
                session_id: Browser session identifier.
                frame_id: Scored frame identifier.
                result: Local assessment.
            """
            records.append((session_id, frame_id, result))
            logged.set()

        async def blocked_capture(*args, **kwargs):
            """Simulate remote inference that has not returned.

            Args:
                *args: Capture arguments.
                **kwargs: Capture keyword arguments.
            """
            await asyncio.Event().wait()

        async def receive(session, sent):
            """Deliver a video frame and wait for its independent quality result.

            Args:
                session: Active intake session.
                sent: Outgoing transport messages.

            Returns:
                The next browser message.
            """
            nonlocal count
            count += 1
            if count == 1:
                return {'type': 'video', 'data': base64.b64encode(encoded).decode()}
            await asyncio.wait_for(logged.wait(), 2)
            self.assertEqual(records[0][1], session.last_frame_id)
            return {'type': 'close'}

        with patch.object(server, 'log_scores', side_effect=record), \
             patch.object(server, '_pin_evidence_photo', side_effect=blocked_capture):
            await test_live_turns.LiveTurnTests.drive(self, receive)
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0][2].signals.brisque)
