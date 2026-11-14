"""Verify owner-scoped book review endpoints and evidence-focused guidance."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi.testclient import TestClient

from live_demo import server
from live_demo.book_guidance import capture_instruction
from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.storage.live_capture import LiveCaptureStore
from library_claim.identification.models import BookFrameAnalysis
from library_claim.identification.storage import save_analysis
from test_book_observations import detection
from test_tracking import textured_view


class BookReviewApiTests(unittest.TestCase):
    """Exercise real local HTTP ownership and SQLite review decisions."""

    def setUp(self):
        """Create a library session and two independently recorded sightings."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.patch = patch.object(server, 'CAPTURE_DATA_DIR', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = TestClient(server.app, base_url='http://127.0.0.1:4177')
        self.addCleanup(self.client.close)
        self.headers = {'Origin': 'http://127.0.0.1:4177'}
        response = self.client.post('/api/sessions', headers=self.headers).json()
        self.sid = response['session_id']
        self.session = server.sessions[self.sid]
        self.addCleanup(server.sessions.pop, self.sid, None)
        self.assertTrue(response['state']['library_sweep'])
        self.assertNotIn('safe place', response['state']['transcript'][0]['text'])
        store = LiveCaptureStore(self.root, self.sid)
        _, encoded = cv2.imencode('.jpg', textured_view())
        for frame, title in [('one', 'First reading'), ('two', 'Different reading')]:
            store.save(encoded.tobytes(), TrackingResult(frame_id=frame, reference_frame_id=frame,
                                                        status='reference_created', elapsed_ms=1),
                       datetime.now(timezone.utc), 480, 360, 'test')
            result = save_analysis(store.database, self.sid, frame, BookFrameAnalysis(books=[detection(title=title)]),
                                   encoded.tobytes(), 'test')
        self.session.book_inventory = result['inventory']

    def test_owner_can_review_and_fetch_crop_but_other_owner_cannot(self):
        """Require the owning session cookie for reads and review mutations."""
        a, b = self.session.book_inventory
        crop_url = f"/api/sessions/{self.sid}/books/evidence/{a['observations'][0]['id']}"
        response = self.client.get(crop_url)
        self.assertEqual(response.status_code, 200)
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(image.size, 0)
        other = TestClient(server.app, base_url='http://127.0.0.1:4177')
        self.addCleanup(other.close)
        self.assertEqual(other.get(crop_url).status_code, 404)
        self.assertEqual(other.post(f'/api/sessions/{self.sid}/books/review', headers=self.headers,
                                   json={'source_id':a['id'], 'target_id':b['id'], 'action':'same'}).status_code, 404)
        response = self.client.post(f'/api/sessions/{self.sid}/books/review', headers=self.headers,
                                    json={'source_id':a['id'], 'target_id':b['id'], 'action':'same'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['state']['book_inventory']), 1)

    def test_locked_capture_cycle_is_persisted(self):
        """Use the existing four states without introducing a review state."""
        target = self.session.book_inventory[0]['id']
        for state in ('Approaching', 'Capturing', 'Resuming', 'Sweeping'):
            server.transition_capture(self.session, state, target if state == 'Approaching' else None)
            self.assertEqual(self.session.sweep_phase, state)
        with self.assertRaises(ValueError):
            server.transition_capture(self.session, 'Resuming')

    def test_guidance_addresses_the_observed_problem(self):
        """Ask for stillness on blur and neighbouring books for uncertain identity."""
        self.assertIn('Hold the camera steady', capture_instruction({'title':'Example', 'observations':[{'issues':['blur']}]}))
        self.assertNotIn('Ready', capture_instruction({'status':'unresolved'}))


class BookGuidanceTransportTests(unittest.IsolatedAsyncioTestCase):
    """Exercise explicit drawer capture actions through the live socket handler."""

    async def exercise_guidance(self, deliver_frame):
        """Exercise capture success or timeout without manual state messages.

        Args:
            deliver_frame: Whether a suitable camera frame follows the request.
        """
        from test_live_turns import LiveTurnTests
        from unittest.mock import Mock
        from test_frame_check import check
        import asyncio
        import base64
        count = 0

        async def capture(session, args, candidate):
            """Supply a clearer title after a real local filter pass.

            Args:
                session: Active session.
                args: Tool arguments.
                candidate: Approved camera frame.

            Returns:
                Successful capture response.
            """
            self.assertEqual(session.sweep_phase, 'Capturing')
            book = session.book_inventory[0]
            book['title'] = 'The Good People'
            book['frame_ids'].append(candidate.frame_id)
            return {'pinned': True}


        async def receive(session, sent):
            """Supply review actions and verify their emitted state changes.

            Args:
                session: Live session under test.
                sent: Messages emitted by the transport.

            Returns:
                Next simulated browser message.
            """
            nonlocal count
            count += 1
            if count == 1:
                session.library_sweep = True
                session.camera_enabled = True
                store = LiveCaptureStore(server.CAPTURE_DATA_DIR, session.session_id)
                _, encoded = cv2.imencode('.jpg', textured_view())
                store.save(encoded.tobytes(), TrackingResult(frame_id='guide', reference_frame_id='guide',
                                                            status='reference_created', elapsed_ms=1),
                           datetime.now(timezone.utc), 480, 360, 'test')
                result = save_analysis(store.database, session.session_id, 'guide',
                                       BookFrameAnalysis(books=[detection()]), encoded.tobytes(), 'test')
                session.book_inventory = result['inventory']
                return {'type':'book_capture', 'book_id':session.book_inventory[0]['id']}
            if count == 2 and deliver_frame:
                self.assertEqual(session.sweep_phase, 'Approaching')
                _, encoded = cv2.imencode('.jpg', textured_view())
                return {'type':'video', 'data':base64.b64encode(encoded).decode()}
            # finish_capture reaches Sweeping before it speaks, so wait for the closing line itself.
            while session.sweep_phase != 'Sweeping' or not any(
                    'continue your sweep' in message.get('text', '') for message in sent):
                await asyncio.sleep(.001)
            self.assertFalse(any('tap' in message.get('text', '').lower() for message in sent))
            return {'type':'close'}

        with patch.object(server, '_pin_evidence_photo', side_effect=capture):
            await LiveTurnTests.drive(self, receive, frame_check=lambda jpeg: check(0.9))

    async def test_request_automatically_captures_and_resumes(self):
        """Resume after a suitable frame supplies the title without manual controls."""
        await self.exercise_guidance(True)

    async def test_no_qualifying_frame_times_out_and_resumes(self):
        """A blocked quality gate cannot trap the user in an intervention."""
        settings = server.CONFIG.model_copy(update={'identification': server.CONFIG.identification.model_copy(
            update={'guidance_timeout_s': .05})})
        with patch.object(server, 'CONFIG', settings):
            await self.exercise_guidance(False)
