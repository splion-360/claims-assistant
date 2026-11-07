"""Verify exact-frame admission, bounded pending work, and remote retry limits."""
import unittest

from library_claim.capture.relevance.classifier import decide
from library_claim.config import CONFIG
from live_demo.camera_filter import CameraCandidate, CameraFilter


class CameraFilterTests(unittest.TestCase):
    """Exercise admission independently from network and pretrained weights."""

    def test_rejected_and_unavailable_frames_never_dispatch(self):
        """Keep below-threshold and failed local assessments off the remote path."""
        candidate = CameraCandidate(b'frame', 'id', 0, 1)
        for assessment in [None, decide({'books': .1}, 0),
                           decide({'books': .4}, 0).model_copy(update={'status': 'quality_blocked'}),
                           decide({'books': .4}, 0).model_copy(update={'status': 'unavailable'})]:
            gate = CameraFilter()
            gate.offer(candidate, assessment)
            self.assertIsNone(gate.take(0, 1, busy=False, at_limit=False))
            self.assertEqual(gate.submitted, 0)

    def test_background_does_not_veto_and_pending_is_latest_only(self):
        """Retain one approved frame while a remote request is busy."""
        gate = CameraFilter()
        first, second = CameraCandidate(b'a', 'a', 0, 1), CameraCandidate(b'b', 'b', 1, 1)
        passed = decide({'books': .4, 'empty_background': .9}, 0)
        gate.offer(first, passed)
        self.assertIsNone(gate.take(0, 1, busy=True, at_limit=False))
        gate.offer(second, passed)
        self.assertEqual(gate.take(1, 1, busy=False, at_limit=False), second)
        self.assertIsNone(gate.pending)

    def test_failed_request_retries_after_interval_and_success_deduplicates(self):
        """Retry a fresh failed candidate without resubmitting successful bytes."""
        gate = CameraFilter()
        frame = CameraCandidate(b'a', 'a', 0, 1)
        gate.offer(frame, decide({'books': .4}, 0))
        self.assertEqual(gate.take(0, 1, busy=False, at_limit=False), frame)
        gate.finish(frame, False)
        self.assertIsNone(gate.take(1, 1, busy=False, at_limit=False))
        self.assertEqual(gate.reason, 'rate_limited')
        self.assertEqual(gate.take(CONFIG.capture.remote_interval_s, 1, busy=False, at_limit=False), frame)
        gate.finish(frame, True)
        gate.offer(CameraCandidate(b'a', 'new-id', 9, 1), decide({'books': .4}, 0))
        self.assertIsNone(gate.pending)

    def test_stale_activation_expiry_and_photo_limit(self):
        """Never dispatch expired or prior-activation candidates or exceed capacity."""
        for now, revision in [(0, 2), (CONFIG.capture.frame_max_age_s + 1, 1)]:
            gate = CameraFilter()
            gate.offer(CameraCandidate(b'a', 'a', 0, 1), decide({'books': .4}, 0))
            self.assertIsNone(gate.take(now, revision, busy=False, at_limit=False))
            self.assertIsNone(gate.pending)
        gate.offer(CameraCandidate(b'a', 'a', 0, 1), decide({'books': .4}, 0))
        self.assertIsNone(gate.take(0, 1, busy=False, at_limit=True))
        self.assertEqual(gate.reason, 'photo_limit')


class FrozenCaptureTests(unittest.IsolatedAsyncioTestCase):
    """Verify the provider receives the approved image and respects camera changes."""

    async def test_provider_receives_candidate_not_latest_frame(self):
        """Keep candidate identity through the actual image request builder."""
        import base64
        import time
        from unittest.mock import AsyncMock, patch
        from live_demo import server

        session = server.IntakeSession('snapshot', camera_enabled=True, camera_mode_revision=1,
                                       last_frame=b'new', last_frame_id='new', last_frame_at=time.monotonic())
        candidate = CameraCandidate(b'approved', 'approved', time.monotonic(), 1)
        observation = server.FrameObservation(observation='Book', supports_claimant_description=False)
        with patch.object(server, 'structured', new=AsyncMock(return_value=observation)) as provider, \
             patch.object(server, 'save_book_analysis', return_value={'inventory': [], 'detected_count': 0}):
            result = await server._pin_evidence_photo(session, {}, candidate=candidate)
        self.assertTrue(result['pinned'])
        content = provider.call_args.args[1]
        self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',')[1]), b'approved')
        self.assertEqual(session.evidence_photos[0]['frame_id'], 'approved')

    async def test_camera_change_discards_inflight_result(self):
        """Avoid saving an old activation's result after the camera is disabled."""
        import time
        from unittest.mock import patch
        from live_demo import server

        session = server.IntakeSession('snapshot', camera_enabled=True, camera_mode_revision=1)
        candidate = CameraCandidate(b'approved', 'approved', time.monotonic(), 1)

        async def provider(*args, **kwargs):
            """Simulate a camera-off event during remote inference.

            Args:
                *args: Image request arguments.
                **kwargs: Model options.

            Returns:
                A stale provider observation.
            """
            server.set_camera_mode(session, False)
            return server.FrameObservation(observation='Book', supports_claimant_description=False)

        with patch.object(server, 'structured', side_effect=provider):
            result = await server._pin_evidence_photo(session, {}, candidate=candidate)
        self.assertFalse(result['pinned'])
        self.assertFalse(session.evidence_photos)
